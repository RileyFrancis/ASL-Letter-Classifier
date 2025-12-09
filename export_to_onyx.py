import torch
from torch import nn
import torch.nn.functional as F
from CNNClassifier import *

# --- REQUIRED: Safe globals so PyTorch can unpickle ---
torch.serialization.add_safe_globals([CNNClassifier])

# --- Now load model ---
print("Loading model...")
model = torch.load("models/asl_model_dataset.pth", map_location="cpu", weights_only=False)
model.eval()
print("Model loaded.")


# --- Export to ONNX ---
dummy = torch.randn(1, 3, 256, 256)
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy,
    "web_models/asl_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=16
)
print("Done. Saved as asl_model.onnx")
