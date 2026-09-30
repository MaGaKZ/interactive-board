# Export Depth Anything V2 metric (indoor) small to ONNX for the web app: .venv/bin/python export_onnx.py
import torch

from depth_anything_v2.dpt import DepthAnythingV2

m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20)
m.load_state_dict(torch.load("checkpoints/depth_anything_v2_metric_hypersim_vits.pth", map_location="cpu"))
# 294x392 = 21x28 patches of 14px, 4:3 like a webcam; must match DW/DH in docs/index.html
torch.onnx.export(m.eval(), torch.randn(1, 3, 294, 392), "docs/depth.onnx",
                  input_names=["image"], output_names=["depth"], opset_version=17, dynamo=False)
