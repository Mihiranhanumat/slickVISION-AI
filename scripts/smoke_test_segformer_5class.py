import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilspill.models.segformer_sar_5class import SegFormerSAR5Class

print("=== SegFormer 5-class smoke test ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = SegFormerSAR5Class(
    pretrained_name="nvidia/mit-b2",
    num_classes=5,
).to(device)

x = torch.rand(1, 1, 256, 256, device=device)

with torch.no_grad():
    y = model(x)

print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(y.shape))
print("Expected    : (1, 5, 256, 256)")

assert tuple(y.shape) == (1, 5, 256, 256)
print("SMOKE TEST: OK")
