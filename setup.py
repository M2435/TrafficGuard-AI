"""
Run this file ONCE to set up your environment.
In VSCode terminal: python setup.py
"""
import subprocess
import sys
import os

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("=" * 50)
print("  Traffic Violation System - Setup")
print("=" * 50)

# Install packages directly (no venv needed)
packages = [
    "ultralytics",
    "opencv-python",
    "easyocr",
    "streamlit",
    "pandas",
    "matplotlib",
    "seaborn",
    "Pillow",
    "numpy",
]

for pkg in packages:
    run(f"{sys.executable} -m pip install {pkg} -q")

# PyTorch with CUDA
print("\n>>> Installing PyTorch with CUDA support...")
run(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q")

print("\n" + "=" * 50)
print("  All packages installed!")
print("  Now run: python test_setup.py")
print("=" * 50)
