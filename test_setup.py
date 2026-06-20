"""
Run after setup.py to verify everything works.
In VSCode terminal: python test_setup.py
"""
import sys

print("=" * 50)
print("  Verifying Setup")
print("=" * 50)

errors = []

# Check Python version
print(f"\n[1] Python: {sys.version.split()[0]}")

# Check PyTorch + CUDA
try:
    import torch
    print(f"[2] PyTorch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"[3] CUDA available: {cuda}")
    if cuda:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")
    else:
        print("    WARNING: No GPU detected. Will use CPU (slower but works).")
except ImportError:
    errors.append("torch not installed — run setup.py first")

# Check OpenCV
try:
    import cv2
    print(f"[4] OpenCV: {cv2.__version__}")
except ImportError:
    errors.append("opencv-python not installed")

# Check EasyOCR
try:
    import easyocr
    print(f"[5] EasyOCR: OK")
except ImportError:
    errors.append("easyocr not installed")

# Check Ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    print(f"[6] YOLOv8 (ultralytics): OK")
except ImportError:
    errors.append("ultralytics not installed")

# Check Streamlit
try:
    import streamlit
    print(f"[7] Streamlit: {streamlit.__version__}")
except ImportError:
    errors.append("streamlit not installed")

# Summary
print("\n" + "=" * 50)
if errors:
    print("  Issues found:")
    for e in errors:
        print(f"  ✗ {e}")
    print("\n  Fix: run python setup.py again")
else:
    print("  All checks passed! You are ready.")
    print("  Next: python main.py --image data/raw_images/test.jpg")
print("=" * 50)
