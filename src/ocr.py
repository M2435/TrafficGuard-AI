"""
Phase 5 - License Plate Detection & OCR
Uses YOLOv8 to locate plate, EasyOCR to read text
"""

import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO


class LicensePlateReader:
    def __init__(self, device: str = "cpu"):
        print("[OCR] Initializing EasyOCR (first run downloads language model)...")
        use_gpu = (device == "cuda")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)

        # Load a YOLOv8 model for plate detection
        # In production: use a fine-tuned plate detection model
        # Here we use the base model and filter by region heuristics
        self.detector = YOLO("yolov8n.pt")
        self.device = device
        print("[OCR] Ready.")

    def detect_plate_region(self, image: np.ndarray) -> list[tuple]:
        """
        Heuristic: License plates are typically in the lower 40% of a vehicle bbox.
        Returns list of (x1, y1, x2, y2) candidate regions.
        """
        h, w = image.shape[:2]
        # Bottom strip heuristic for Indian traffic cameras
        candidates = []

        # Run YOLO to find vehicles
        results = self.detector(image, conf=0.4, device=self.device, verbose=False)
        vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Plate is typically in the bottom 25% of the vehicle
                plate_y1 = y1 + int((y2 - y1) * 0.70)
                plate_y2 = y2
                plate_x1 = x1 + int((x2 - x1) * 0.10)
                plate_x2 = x2 - int((x2 - x1) * 0.10)
                candidates.append((plate_x1, plate_y1, plate_x2, plate_y2))

        return candidates

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Enhance plate image for better OCR accuracy."""
        # Upscale small plates
        h, w = plate_img.shape[:2]
        if w < 200:
            plate_img = cv2.resize(plate_img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to handle uneven lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        return thresh

    def clean_plate_text(self, text: str) -> str:
        """Clean and format Indian license plate text."""
        text = text.upper().replace(" ", "").replace("-", "")
        # Remove non-alphanumeric characters
        text = re.sub(r"[^A-Z0-9]", "", text)
        # Basic Indian plate format: XX00XX0000
        if len(text) >= 6:
            return text
        return "UNREADABLE"

    def read_plate(self, image: np.ndarray) -> list[dict]:
        """
        Full pipeline: detect plate regions → preprocess → OCR.
        Returns list of {text, confidence, bbox}.
        """
        plates = []
        regions = self.detect_plate_region(image)

        for (x1, y1, x2, y2) in regions:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed = self.preprocess_plate(crop)

            try:
                results = self.reader.readtext(processed)
                for (bbox, text, conf) in results:
                    cleaned = self.clean_plate_text(text)
                    if cleaned != "UNREADABLE" and conf > 0.4:
                        plates.append({
                            "text":       cleaned,
                            "raw_text":   text,
                            "confidence": conf,
                            "bbox":       (x1, y1, x2, y2),
                        })
            except Exception as e:
                print(f"[OCR Warning] {e}")

        return plates

    def draw_plates(self, image: np.ndarray, plates: list[dict]) -> np.ndarray:
        """Annotate image with detected plate numbers."""
        output = image.copy()
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Plate: {plate['text']} ({plate['confidence']:.0%})"
            cv2.putText(output, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output
