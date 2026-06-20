# Traffic Violation Detection System
Flipkart × Bengaluru Traffic Police Hackathon — Theme 3

AI-powered system to detect and classify traffic violations from images and video frames using YOLOv8 and EasyOCR.

Key features
- Helmet detection (no helmet)
- Triple riding detection
- Illegal parking detection
- Stop-line violation detection
- License plate extraction (OCR)

---

## Requirements
- Python 3.8+ (3.10+ recommended)
- CUDA-capable GPU for accelerated inference (optional)
- See `requirements.txt` for Python packages

---

## Quick Setup

1. Create and activate a virtual environment (Windows):

```powershell
python -m venv traffic_env
traffic_env\Scripts\activate
```

2. Install dependencies (CPU):

```powershell
pip install -r requirements.txt
```

If you have CUDA and want GPU support, install the correct PyTorch wheel first (example for CUDA 12.1):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. Verify the environment by running a smoke test:

```powershell
python main.py --test
```

If the smoke test fails, see Troubleshooting below.

---

## Usage

Analyze a single image:

```powershell
python main.py --image data/raw_images/sample.jpg
```

Batch process a folder:

```powershell
python main.py --batch data/raw_images/
```

Process a video (if supported by your build):

```powershell
python main.py --video path/to/video.mp4
```

Launch the Streamlit dashboard:

```powershell
streamlit run src/dashboard.py
```

Outputs
- Annotated images and JSON results are written to `outputs/` by default. See `outputs/violations.json` for structured results.

---

## Models & Weights
- The repo includes a small YOLO weight (`yolov8n.pt`) for quick tests. For better accuracy, replace it with a larger/more accurate model placed in `models/` and update the path in `src/detect.py` or the CLI argument.

---

## Project Layout

```
traffic_violation_system/
├── data/                # input images and annotations
├── models/              # place custom weights here (optional)
├── outputs/             # generated annotated images and JSON
├── src/                 # core modules: detect, ocr, tracker, dashboard
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

---

## Troubleshooting
- If GPU inference fails, ensure the installed `torch` matches your CUDA version. See https://pytorch.org/get-started/locally/.
- If EasyOCR has issues, check locale and font dependencies; try installing pillow and opencv-python from wheels.
- For permission issues writing `outputs/`, ensure the process has write access to the folder.

---

## Development & Contribution
- To run linting/tests (if any): add your preferred tools (flake8/pytest) and run locally.
- To contribute: fork, make changes on a branch, and open a PR with a clear description.

---

## License
Specify your project license here (e.g., MIT). If unsure, add a `LICENSE` file.

---

If you'd like, I can also:
- initialize a git repo and create the first commit, and
- add a GitHub remote and push — tell me your GitHub repo URL.

