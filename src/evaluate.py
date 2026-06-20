"""
Phase 8 - Performance Evaluation
Computes Accuracy, Precision, Recall, F1-score, and a simplified mAP
for the violation-detection pipeline against a labeled ground-truth set.

WHY THIS IS NEEDED:
The hackathon brief explicitly requires evaluation using Accuracy,
Precision, Recall, F1-score, and mAP. None of the other modules
measure correctness against ground truth -- they only produce
predictions. This script is what closes that gap.

HOW GROUND TRUTH WORKS HERE:
You provide a small folder of test images plus a matching JSON file
describing what SHOULD be detected in each one (ground truth labels).
This script runs the real pipeline on each image, compares predictions
to ground truth, and reports the standard metrics.

Ground truth JSON format (see data/ground_truth_example.json):
{
  "image1.jpg": {
    "violations": ["No Helmet", "Triple Riding"],
    "vehicle_boxes": [[x1,y1,x2,y2, "car"], [x1,y1,x2,y2, "motorcycle"]]
  },
  "image2.jpg": {
    "violations": [],
    "vehicle_boxes": [[x1,y1,x2,y2, "car"]]
  }
}

Two separate evaluations are produced:
1. VIOLATION CLASSIFICATION metrics (Accuracy/Precision/Recall/F1)
   -- "did the system correctly say YES/NO for each violation type?"
2. DETECTION mAP (simplified, IoU-based)
   -- "did the system find the right objects in the right place?"
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------- #
#  IoU helper (needed for mAP)
# ---------------------------------------------------------------------- #

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# ---------------------------------------------------------------------- #
#  Violation classification metrics (Accuracy / Precision / Recall / F1)
# ---------------------------------------------------------------------- #

class ViolationEvaluator:
    """
    Treats each violation TYPE as an independent binary classifier per
    image: "did this image actually contain a [No Helmet] violation?"
    vs "did the system predict one?" Aggregated across all images and
    all violation types to produce overall Precision/Recall/F1, plus a
    per-type breakdown.
    """

    ALL_VIOLATION_TYPES = [
        "No Helmet", "No Seatbelt", "Triple Riding", "Wrong Side",
        "Red Light", "Stop Line", "Illegal Parking",
    ]

    def __init__(self):
        # Confusion-matrix counts, per violation type
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.tn = defaultdict(int)
        self.total_images = 0

    def add_image_result(self, predicted_types: set, ground_truth_types: set):
        """Update confusion matrix counts for one image."""
        self.total_images += 1
        for vtype in self.ALL_VIOLATION_TYPES:
            predicted = vtype in predicted_types
            actual = vtype in ground_truth_types

            if predicted and actual:
                self.tp[vtype] += 1
            elif predicted and not actual:
                self.fp[vtype] += 1
            elif not predicted and actual:
                self.fn[vtype] += 1
            else:
                self.tn[vtype] += 1

    def per_type_metrics(self) -> dict:
        """Precision/Recall/F1/Accuracy for each violation type."""
        results = {}
        for vtype in self.ALL_VIOLATION_TYPES:
            tp, fp, fn, tn = self.tp[vtype], self.fp[vtype], self.fn[vtype], self.tn[vtype]
            total = tp + fp + fn + tn
            if total == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            accuracy = (tp + tn) / total if total > 0 else 0.0

            results[vtype] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "accuracy": round(accuracy, 3),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
        return results

    def overall_metrics(self) -> dict:
        """Micro-averaged metrics across ALL violation types combined."""
        total_tp = sum(self.tp.values())
        total_fp = sum(self.fp.values())
        total_fn = sum(self.fn.values())
        total_tn = sum(self.tn.values())
        total = total_tp + total_fp + total_fn + total_tn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        accuracy = (total_tp + total_tn) / total if total > 0 else 0.0

        return {
            "images_evaluated": self.total_images,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "accuracy": round(accuracy, 3),
        }


# ---------------------------------------------------------------------- #
#  Simplified detection mAP (IoU-based, single confidence threshold)
# ---------------------------------------------------------------------- #

def compute_detection_map(predictions: dict, ground_truth: dict,
                           iou_threshold: float = 0.5) -> dict:
    """
    Simplified mAP@0.5 across vehicle classes.

    predictions:    {image_name: [(x1,y1,x2,y2,class_name,confidence), ...]}
    ground_truth:   {image_name: [(x1,y1,x2,y2,class_name), ...]}

    NOTE: this is a simplified, single-IoU-threshold AP (not the full
    11-point or COCO-style 101-point interpolation), appropriate for a
    hackathon-scale evaluation set. It is computed per class then
    averaged -- i.e. genuinely "mean" AP, just with a simpler AP
    calculation per class than the official COCO protocol.
    """
    class_names = set()
    for boxes in ground_truth.values():
        for box in boxes:
            class_names.add(box[4])

    ap_per_class = {}

    for cls in class_names:
        # Gather all predictions of this class across all images, sorted by confidence desc
        all_preds = []
        for img_name, boxes in predictions.items():
            for (x1, y1, x2, y2, cname, conf) in boxes:
                if cname == cls:
                    all_preds.append((img_name, (x1, y1, x2, y2), conf))
        all_preds.sort(key=lambda p: p[2], reverse=True)

        gt_boxes_by_image = {
            img: [b[:4] for b in boxes if b[4] == cls]
            for img, boxes in ground_truth.items()
        }
        total_gt = sum(len(v) for v in gt_boxes_by_image.values())
        if total_gt == 0:
            continue

        matched = {img: [False] * len(boxes) for img, boxes in gt_boxes_by_image.items()}
        tp_list, fp_list = [], []

        for img_name, pred_box, conf in all_preds:
            gt_list = gt_boxes_by_image.get(img_name, [])
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt_list):
                if matched[img_name][idx]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_iou >= iou_threshold and best_idx >= 0:
                matched[img_name][best_idx] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        recalls = tp_cum / total_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        # Simple AP: average precision at each recall step (rectangle approximation)
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precisions, recalls):
            ap += p * (r - prev_recall)
            prev_recall = r

        ap_per_class[cls] = round(float(ap), 3)

    mean_ap = round(float(np.mean(list(ap_per_class.values()))), 3) if ap_per_class else 0.0

    return {
        "mAP@0.5": mean_ap,
        "per_class_AP": ap_per_class,
    }


# ---------------------------------------------------------------------- #
#  End-to-end runner
# ---------------------------------------------------------------------- #

def run_evaluation(test_images_dir: str, ground_truth_path: str,
                    device: str = "cpu") -> dict:
    """
    Runs the actual ViolationDetector pipeline on every image in
    test_images_dir, compares against ground_truth_path, and returns
    a full metrics report (also printed to console).
    """
    import cv2
    from src.violations import ViolationDetector

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}\n"
            f"See data/ground_truth_example.json for the expected format."
        )

    with open(gt_path) as f:
        ground_truth = json.load(f)

    detector = ViolationDetector(device=device)
    evaluator = ViolationEvaluator()

    det_predictions = {}
    det_ground_truth = {}

    images_dir = Path(test_images_dir)
    print(f"\n[Evaluate] Running pipeline on {len(ground_truth)} labeled test images...")

    for img_name, gt_entry in ground_truth.items():
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"  [Skip] {img_name} not found in {test_images_dir}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [Skip] Could not read {img_name}")
            continue

        detections, violations = detector.analyze_image(image)

        predicted_types = {v.violation_type for v in violations}
        gt_types = set(gt_entry.get("violations", []))
        evaluator.add_image_result(predicted_types, gt_types)

        det_predictions[img_name] = [
            (*d["bbox"], d["class_name"], d["confidence"]) for d in detections
        ]
        det_ground_truth[img_name] = [
            tuple(box) for box in gt_entry.get("vehicle_boxes", [])
        ]

        match = "OK" if predicted_types == gt_types else "DIFF"
        print(f"  [{match}] {img_name}: predicted={sorted(predicted_types)} "
              f"| ground_truth={sorted(gt_types)}")

    overall = evaluator.overall_metrics()
    per_type = evaluator.per_type_metrics()
    map_results = compute_detection_map(det_predictions, det_ground_truth)

    report = {
        "violation_classification": {
            "overall": overall,
            "per_violation_type": per_type,
        },
        "detection_map": map_results,
    }

    print("\n===== Violation Classification Metrics =====")
    print(f"  Images evaluated: {overall['images_evaluated']}")
    print(f"  Accuracy:  {overall['accuracy']:.1%}")
    print(f"  Precision: {overall['precision']:.1%}")
    print(f"  Recall:    {overall['recall']:.1%}")
    print(f"  F1-score:  {overall['f1_score']:.1%}")

    print("\n  Per-violation-type breakdown:")
    for vtype, m in per_type.items():
        print(f"    {vtype:18s} P={m['precision']:.2f}  R={m['recall']:.2f}  "
              f"F1={m['f1_score']:.2f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    print("\n===== Detection mAP@0.5 =====")
    print(f"  mAP@0.5: {map_results['mAP@0.5']:.1%}")
    for cls, ap in map_results["per_class_AP"].items():
        print(f"    {cls:12s} AP={ap:.2f}")

    # Save full report
    out_path = Path("outputs") / "evaluation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Evaluate] Full report saved -> {out_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate violation detection pipeline")
    parser.add_argument("--images", type=str, default="data/test_images",
                         help="Folder containing labeled test images")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth.json",
                         help="Path to ground truth JSON file")
    args = parser.parse_args()

    run_evaluation(args.images, args.ground_truth)
