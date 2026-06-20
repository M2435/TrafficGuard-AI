"""
Phase 2 - Image Preprocessing
Handles: low light, blur, rain, noise, normalization
"""

import cv2
import numpy as np
from pathlib import Path


def enhance_low_light(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to boost visibility in dark/night images."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Non-local means denoising — good for rainy/grainy frames."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10,
                                           templateWindowSize=7, searchWindowSize=21)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Sharpen blurry images using an unsharp mask."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def resize_normalize(image: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """Resize to model input size and normalize pixel values."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def detect_brightness(image: np.ndarray) -> str:
    """Classify image as bright, normal, or dark."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean < 60:
        return "dark"
    elif mean > 180:
        return "bright"
    return "normal"


def preprocess_image(image_path: str, save_path: str = None) -> np.ndarray:
    """
    Full preprocessing pipeline for a single traffic image.
    Returns a BGR numpy array ready for detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    brightness = detect_brightness(image)
    if brightness == "dark":
        image = enhance_low_light(image)

    image = reduce_noise(image)
    image = sharpen(image)
    image_resized = cv2.resize(image, (640, 640))

    if save_path:
        cv2.imwrite(save_path, image_resized)
        print(f"[Preprocessed] Saved to {save_path}")

    return image_resized


def preprocess_batch(input_dir: str, output_dir: str):
    """Preprocess all images in a folder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = (list(input_path.glob("*.jpg")) +
                   list(input_path.glob("*.jpeg")) +
                   list(input_path.glob("*.png")))

    print(f"[Batch] Found {len(image_files)} images...")
    for img_file in image_files:
        out_file = output_path / img_file.name
        try:
            preprocess_image(str(img_file), str(out_file))
        except Exception as e:
            print(f"[Error] {img_file.name}: {e}")

    print(f"[Done] Preprocessed {len(image_files)} images")
