import sys
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

sys.path.insert(0, "Depth-Anything-V2/metric_depth")
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
INPUT_SIZE = 308  # ponytail: 518 is the model default; lower = faster, raise if depth looks blocky
SCALE = 1.0  # calibration: put palm at a known distance (tape measure), set SCALE = real_cm / shown_cm
SMOOTH = 0.3  # 0..1, lower = steadier number, higher = faster response
TIPS = [4, 8, 12, 16, 20]  # thumb..pinky fingertips
PALM = [0, 1, 5, 9, 13, 17]  # wrist, thumb base, finger bases -> palm polygon

# metric model (indoor, Hypersim): outputs metres directly
depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20)
depth_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_metric_hypersim_vits.pth", map_location="cpu"))
depth_model = depth_model.to(DEVICE).eval()

hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="checkpoints/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
))


def label(img, text, org, scale=0.6, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Camera not available. macOS: System Settings > Privacy & Security > Camera > enable your terminal app, then restart it.")
t = 0
smoothed = {}  # handedness -> palm cm
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    depth_cm = depth_model.infer_image(frame, INPUT_SIZE) * 100 * SCALE  # metres -> cm

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t += 33
    res = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)

    for n, (lms, handed) in enumerate(zip(res.hand_landmarks, res.handedness)):
        side = handed[0].category_name
        pts = np.array([(min(max(int(p.x * w), 0), w - 1), min(max(int(p.y * h), 0), h - 1)) for p in lms])

        # palm distance = median depth over the palm polygon (robust to edge pixels)
        palm_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(palm_mask, cv2.convexHull(pts[PALM]), 255)
        palm_cm = float(np.median(depth_cm[palm_mask > 0]))
        palm_cm = smoothed[side] = SMOOTH * palm_cm + (1 - SMOOTH) * smoothed.get(side, palm_cm)

        # color the hand by depth: yellow = near, purple = far (stretched to the hand's own range)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
        hd = depth_cm[mask]
        near = np.clip((hd.max() - depth_cm) / (hd.max() - hd.min() + 1e-6), 0, 1)
        color = cv2.applyColorMap((near * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        frame[mask] = cv2.addWeighted(frame, 0.4, color, 0.6, 0)[mask]
        cv2.polylines(frame, [cv2.convexHull(pts[PALM])], True, (0, 255, 0), 2)

        for i in TIPS:
            cv2.circle(frame, tuple(int(v) for v in pts[i]), 6, (255, 255, 255), -1)
            cv2.putText(frame, f"{depth_cm[pts[i][1], pts[i][0]]:.0f}", (int(pts[i][0]) + 8, int(pts[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # big palm distance at palm center
        cx, cy = pts[PALM].mean(axis=0).astype(int)
        label(frame, f"{palm_cm:.0f} cm", (int(cx) - 40, int(cy)), 1.0, (0, 255, 0))

        # HUD bar top-left: fuller = closer (0..100 cm range)
        y = 30 + n * 40
        fill = int(np.clip(1 - palm_cm / 100, 0, 1) * 200)
        cv2.rectangle(frame, (10, y), (210, y + 20), (80, 80, 80), -1)
        cv2.rectangle(frame, (10, y), (10 + fill, y + 20), (0, 255, 0), -1)
        label(frame, f"{side} palm: {palm_cm:.1f} cm", (220, y + 16))

    cv2.imshow("hands + depth", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
