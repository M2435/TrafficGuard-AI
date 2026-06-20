"""
Phase 3 - Vehicle & Person Detection using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}

CLASS_COLORS = {
    "person":     (0, 255, 255),
    "bicycle":    (255, 128, 0),
    "car":        (0, 200, 255),
    "motorcycle": (255, 0, 128),
    "bus":        (128, 0, 255),
    "truck":      (0, 128, 255),
}


class TrafficDetector:
    def __init__(self, model_size: str = "yolov8n.pt", conf: float = 0.4, device: str = "cuda"):
        print(f"[Detector] Loading {model_size} on {device}...")
        self.model = YOLO(model_size)
        self.conf = conf
        self.device = device
        print("[Detector] Ready.")

    def detect(self, image: np.ndarray) -> list[dict]:
        results = self.model(image, conf=self.conf, device=self.device, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class_id":   class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox":       (x1, y1, x2, y2),
                })
        return detections

    def detect_from_path(self, image_path: str) -> tuple[np.ndarray, list[dict]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")
        return image, self.detect(image)

    def draw_detections(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(label, (200, 200, 200))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(output, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return output

    def count_by_class(self, detections: list[dict]) -> dict:
        counts = {}
        for det in detections:
            name = det["class_name"]
            counts[name] = counts.get(name, 0) + 1
        return counts
