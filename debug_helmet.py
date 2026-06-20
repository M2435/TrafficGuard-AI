"""
One-off debug script to trace exactly why helmet/triple-riding violations
aren't firing. Run: python debug_helmet.py data/raw_images/yourimage.jpg
"""
import sys
import cv2
from src.violations import ViolationDetector

if len(sys.argv) < 2:
    print("Usage: python debug_helmet.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
image = cv2.imread(image_path)
if image is None:
    print(f"Could not read image: {image_path}")
    sys.exit(1)

image = cv2.resize(image, (640, 640))

detector = ViolationDetector(device="cpu")
detections = detector._get_base_detections(image)

print(f"\n=== Raw detections ({len(detections)} total, conf threshold=0.25) ===")
for d in detections:
    print(f"  {d['class_name']:12s} conf={d['confidence']:.2f}  bbox={d['bbox']}")

motorcycles = [d for d in detections if d["class_name"] == "motorcycle"]
persons = [d for d in detections if d["class_name"] == "person"]
print(f"\nMotorcycles found: {len(motorcycles)}")
print(f"Persons found: {len(persons)}")

print("\n=== Rider matching (expanded zone, not strict overlap) ===")
for i, moto in enumerate(motorcycles):
    riders = detector._find_riders(moto["bbox"], persons)
    print(f"\nMotorcycle #{i} bbox={moto['bbox']} -> {len(riders)} rider(s) matched")
    for r in riders:
        print(f"    rider bbox={r['bbox']} conf={r['confidence']:.2f}")
    if not riders:
        print("    >>> STILL no riders matched even with expanded zone.")
        print("    >>> This means no 'person' detection exists near this motorcycle at all.")

print(f"\nhelmet_model loaded: {detector.helmet_model is not None}")

print("\n=== Running actual check_helmet() ===")
helmet_violations = detector.check_helmet(image, detections)
print(f"Helmet violations returned: {len(helmet_violations)}")
for v in helmet_violations:
    print(f"  {v.violation_type}: {v.description}")

print("\n=== Running actual check_triple_riding() ===")
triple_violations = detector.check_triple_riding(image, detections)
print(f"Triple riding violations returned: {len(triple_violations)}")
for v in triple_violations:
    print(f"  {v.violation_type}: {v.description}")
