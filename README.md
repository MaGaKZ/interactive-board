# interactive-board

Hand tracking (MediaPipe) + metric depth (Depth Anything V2 Small, indoor) → palm distance from the camera in cm.

## Web
`web/` is a static site (runs in the browser, WebGPU). Serve it over HTTPS (or `localhost`):

    cd web && python3 -m http.server 8765

## Python (desktop)

    git clone --recursive git@github.com:MaGaKZ/interactive-board.git
    python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
    mkdir -p checkpoints
    curl -Lo checkpoints/depth_anything_v2_metric_hypersim_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
    curl -Lo checkpoints/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    .venv/bin/python main.py

`export_onnx.py` rebuilds `web/depth.onnx` from the checkpoint.
