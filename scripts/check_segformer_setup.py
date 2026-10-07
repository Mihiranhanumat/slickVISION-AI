from pathlib import Path
import sys, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

print("=== SegFormer setup check ===")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Manifest:", (ROOT/"data/splits/patch_manifest.csv").exists())

try:
    import transformers
    print("Transformers:", transformers.__version__)
except ImportError:
    print("Transformers: NOT INSTALLED")
    print("Run: pip install transformers")

try:
    from oilspill.models.segformer import SegFormerSAR
    print("SegFormer wrapper: OK")
except Exception as e:
    print("SegFormer wrapper error:", repr(e))
