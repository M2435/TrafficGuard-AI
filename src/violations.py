"""
Phase 4 - Traffic Violation Detection Logic
Violations: helmet, seatbelt, triple riding, wrong side,
            red light, stop line, illegal parking
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Violation:
    violation_type: str
    confidence: float
    bbox: tuple          # (x1, y1, x2, y2) of the violating object
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    plate_number: str = "Unknown"
    severity: str = "Medium"   # Low / Medium / High


VIOLATION_COLORS = {
    "No Helmet":        (0, 0, 255),
    "No Seatbelt":      (0, 100, 255),
    "Triple Riding":    (0, 0, 200),
    "Wrong Side":       (255, 0, 255),
    "Red Light":        (0, 0, 255),
    "Stop Line":        (0, 165, 255),
    "Illegal Parking":  (128, 0, 128),
}


class ViolationDetector:
    """
    Uses two YOLO models:
    1. Base YOLOv8 — for vehicles and persons
    2. Custom/fine-tuned model — for helmet, seatbelt detection
    Falls back to heuristic rules when custom model unavailable.
    """

    def __init__(self, base_model: str = "yolov8n.pt",
                 helmet_model: str = None, device: str = "cpu"):
        print("[ViolationDetector] Loading models...")
        self.base_model = YOLO(base_model)
        self.device = device

        # Load custom helmet/seatbelt model if provided
        self.helmet_model = None
        if helmet_model:
            try:
                self.helmet_model = YOLO(helmet_model)
                print(f"[ViolationDetector] Custom model loaded: {helmet_model}")
            except Exception as e:
                print(f"[Warning] Could not load custom model: {e}. Using heuristics.")

        print("[ViolationDetector] Ready.")

    def _get_base_detections(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        """
        conf=0.25 (lowered from 0.4): motorcycles in busy/angled traffic
        shots are frequently detected at lower confidence than cars/buses
        because their silhouette is broken up by the rider's body, wheels,
        and handlebars. A stricter 0.4 threshold was silently dropping
        valid motorcycle detections, which in turn skipped helmet and
        triple-riding checks (both require a detected motorcycle box).
        """
        results = self.base_model(image, conf=conf, device=self.device, verbose=False)
        dets = []
        target_classes = {0: "person", 1: "bicycle", 2: "car",
                          3: "motorcycle", 5: "bus", 7: "truck"}
        for result in results:
            for box in result.boxes:
                cid = int(box.cls[0])
                if cid not in target_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append({
                    "class_id": cid,
                    "class_name": target_classes[cid],
                    "confidence": float(box.conf[0]),
                    "bbox": (x1, y1, x2, y2),
                })
        return dets

    # ------------------------------------------------------------------ #
    #  Individual violation checks
    # ------------------------------------------------------------------ #

    def _find_riders(self, moto_bbox: tuple, persons: list[dict],
                      expand_ratio: float = 0.35) -> list[dict]:
        """
        Find persons riding a given motorcycle.

        Strict box-intersection (the old approach) frequently misses
        real riders, because a person's bounding box often sits mostly
        ABOVE the motorcycle's box (the bike box hugs the wheels/frame,
        while the rider's head/torso extends upward past it) -- so the
        two boxes barely touch or don't overlap at all even though the
        person is clearly riding that bike.

        Fix: expand the motorcycle's box upward and outward by
        `expand_ratio` before checking overlap, approximating the full
        "rider zone" above and around the bike rather than just the
        bike's own footprint.
        """
        mx1, my1, mx2, my2 = moto_bbox
        w, h = mx2 - mx1, my2 - my1

        # Expand the search zone: a lot upward (riders extend above the bike),
        # a bit sideways, a little downward.
        zx1 = mx1 - int(w * expand_ratio)
        zx2 = mx2 + int(w * expand_ratio)
        zy1 = my1 - int(h * 1.2)   # riders' heads can be well above the bike box
        zy2 = my2 + int(h * expand_ratio)

        riders = []
        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            ix1, iy1 = max(zx1, px1), max(zy1, py1)
            ix2, iy2 = min(zx2, px2), min(zy2, py2)
            if ix2 > ix1 and iy2 > iy1:
                riders.append(person)
        return riders

    def check_helmet(self, image: np.ndarray, detections: list[dict]) -> list[Violation]:
        """
        Check helmet compliance for motorcycle riders.
        Strategy: For each motorcycle, look for a person on it (using an
        expanded rider zone, not strict box overlap -- see _find_riders).
        If no helmet class detected in the head region → violation.
        """
        violations = []
        motorcycles = [d for d in detections if d["class_name"] == "motorcycle"]
        persons = [d for d in detections if d["class_name"] == "person"]

        for moto in motorcycles:
            mx1, my1, mx2, my2 = moto["bbox"]
            riders = self._find_riders(moto["bbox"], persons)

            if not riders:
                continue

            # If custom model available, use it to detect helmet
            if self.helmet_model:
                # Crop the motorcycle region
                crop = image[my1:my2, mx1:mx2]
                if crop.size == 0:
                    continue
                results = self.helmet_model(crop, verbose=False)
                has_helmet = False
                for r in results:
                    for box in r.boxes:
                        # Assumes custom model class 0 = helmet
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                            has_helmet = True
                if not has_helmet:
                    violations.append(Violation(
                        violation_type="No Helmet",
                        confidence=0.85,
                        bbox=moto["bbox"],
                        description=f"Motorcycle rider without helmet. {len(riders)} rider(s) detected.",
                        severity="High"
                    ))
            else:
                # Heuristic: flag all motorcycles with riders as potential violations
                # (In production, replace with trained model)
                violations.append(Violation(
                    violation_type="No Helmet",
                    confidence=0.65,
                    bbox=moto["bbox"],
                    description=f"Potential helmet violation — {len(riders)} rider(s) on motorcycle.",
                    severity="High"
                ))

        return violations

    def check_triple_riding(self, image: np.ndarray, detections: list[dict]) -> list[Violation]:
        """Flag motorcycles with 3+ persons (using the same expanded rider zone)."""
        violations = []
        motorcycles = [d for d in detections if d["class_name"] == "motorcycle"]
        persons = [d for d in detections if d["class_name"] == "person"]

        for moto in motorcycles:
            riders = self._find_riders(moto["bbox"], persons)
            rider_count = len(riders)
            if rider_count >= 3:
                violations.append(Violation(
                    violation_type="Triple Riding",
                    confidence=0.90,
                    bbox=moto["bbox"],
                    description=f"Triple riding detected — {rider_count} persons on motorcycle.",
                    severity="High"
                ))

        return violations

    def check_seatbelt(self, image: np.ndarray, detections: list[dict]) -> list[Violation]:
        """
        Seatbelt compliance for car/truck/bus drivers.

        HONEST LIMITATION: reliably detecting a seatbelt requires a
        close, mostly-frontal view of the driver through the windshield
        (to see the diagonal strap across their chest), plus a model
        trained specifically on that visual pattern. Wide traffic-camera
        footage rarely has this resolution/angle, and there is no
        seatbelt-specific class in any pretrained model used here.

        Without a fine-tuned seatbelt-detection model, this method does
        NOT claim to detect violations — it returns an empty list rather
        than guessing, to avoid reporting false "evidence" against a
        real vehicle/driver. If a custom-trained seatbelt model is
        provided via `self.helmet_model` (reused slot) with a "no_seatbelt"
        class, it will be used; otherwise this is a documented gap.
        """
        violations = []
        if self.helmet_model is None:
            return violations   # No model available -- do not guess.

        cars = [d for d in detections if d["class_name"] in ("car", "truck", "bus")]
        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            # Driver/windshield region heuristic: top 40% of vehicle box
            wy2 = y1 + int((y2 - y1) * 0.40)
            crop = image[y1:wy2, x1:x2]
            if crop.size == 0:
                continue

            results = self.helmet_model(crop, verbose=False)
            seatbelt_violation_found = False
            for r in results:
                for box in r.boxes:
                    # Assumes a fine-tuned model with class 1 = "no_seatbelt"
                    if int(box.cls[0]) == 1 and float(box.conf[0]) > 0.5:
                        seatbelt_violation_found = True

            if seatbelt_violation_found:
                violations.append(Violation(
                    violation_type="No Seatbelt",
                    confidence=0.70,
                    bbox=car["bbox"],
                    description=f"{car['class_name'].capitalize()} driver without seatbelt (custom model).",
                    severity="Medium"
                ))

        return violations

    def check_wrong_side_driving(self, tracked_objects: dict, tracker,
                                  expected_direction: str = None,
                                  min_track_frames: int = 8,
                                  angle_tolerance: float = 90.0) -> list[Violation]:
        """
        Detect vehicles travelling against the expected traffic direction.

        This needs VIDEO (multiple frames) to know which way a vehicle is
        actually moving -- a single image cannot show direction of travel.

        expected_direction: one of "up", "down", "left", "right" --
            the direction vehicles SHOULD be moving in this camera view.
            This must be set per-camera (just like stop_line_y) since it
            depends on the road's orientation in frame. If None, this
            check is skipped entirely (no guessing).
        min_track_frames: minimum frames of tracking history required
            before judging direction (avoids noise on brand-new tracks).
        angle_tolerance: how many degrees off the *opposite* of the
            expected direction still counts as "wrong way" (default 90 deg
            half-cone, i.e. moving generally backwards).
        """
        if expected_direction is None:
            return []

        direction_vectors = {
            "up":    np.array([0, -1]),
            "down":  np.array([0, 1]),
            "left":  np.array([-1, 0]),
            "right": np.array([1, 0]),
        }
        if expected_direction not in direction_vectors:
            return []

        expected_vec = direction_vectors[expected_direction]
        violations = []

        for oid, det in tracked_objects.items():
            if det["class_name"] not in ("car", "motorcycle", "truck", "bus"):
                continue

            history = tracker.position_history.get(oid, [])
            if len(history) < min_track_frames:
                continue

            start = np.array(history[-min_track_frames])
            end = np.array(history[-1])
            movement = end - start

            if np.linalg.norm(movement) < 5:
                continue  # essentially stationary, direction is meaningless/noisy

            movement_norm = movement / np.linalg.norm(movement)
            # cos(angle) between actual movement and expected direction
            cos_angle = np.dot(movement_norm, expected_vec)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            # If moving close to OPPOSITE of expected (angle near 180deg) -> wrong side
            if angle_deg > (180 - angle_tolerance / 2):
                violations.append(Violation(
                    violation_type="Wrong Side",
                    confidence=min(0.9, 0.5 + (angle_deg - 90) / 180),
                    bbox=det["bbox"],
                    description=(f"{det['class_name'].capitalize()} moving against "
                                  f"expected '{expected_direction}' traffic flow "
                                  f"(track #{oid}, {angle_deg:.0f}\u00b0 off expected)."),
                    severity="High",
                ))

        return violations

    def check_red_light(self, image: np.ndarray, detections: list[dict],
                         traffic_light_roi: tuple = None,
                         stop_line_y: int = None) -> list[Violation]:
        """
        Red-light violation: a vehicle crosses the stop line while the
        signal is red.

        HONEST LIMITATION: no pretrained model used here classifies
        traffic-light color directly. This implementation instead reads
        the actual light color from a region you mark once per camera
        (traffic_light_roi), using HSV color analysis on those pixels --
        a real, explainable signal rather than a guess. It requires BOTH
        traffic_light_roi and stop_line_y to be supplied; without them
        this check is skipped (returns no violations), matching the same
        "no silent guessing" policy used for stop-line and parking.

        traffic_light_roi: (x1, y1, x2, y2) pixel box tightly cropping
            just the active light bulb area of the signal in this camera
            view (find this once by inspecting a sample frame).
        """
        if traffic_light_roi is None or stop_line_y is None:
            return []

        x1, y1, x2, y2 = traffic_light_roi
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red wraps around hue 0/180 in OpenCV's HSV; cover both ends.
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        red_ratio = (cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)) / roi.size

        is_red = red_ratio > 0.08  # tuned threshold; adjust per camera/lighting
        if not is_red:
            return []

        violations = []
        vehicles = [d for d in detections
                    if d["class_name"] in ("car", "motorcycle", "truck", "bus")]

        for vehicle in vehicles:
            _, _, _, vy2 = vehicle["bbox"]
            if vy2 >= stop_line_y:
                violations.append(Violation(
                    violation_type="Red Light",
                    confidence=0.75,
                    bbox=vehicle["bbox"],
                    description=(f"{vehicle['class_name'].capitalize()} crossed stop line "
                                  f"while signal was red (red-pixel ratio {red_ratio:.0%})."),
                    severity="High",
                ))

        return violations

    def check_illegal_parking_single_frame(self, image: np.ndarray, detections: list[dict],
                                            no_parking_zones: list[tuple] = None) -> list[Violation]:
        """
        DEPRECATED for standalone use — kept only as a fallback when no
        tracking history exists (e.g. analyzing a single still image with
        no video). A single frame cannot prove a vehicle is "parked" since
        that requires observing it over time; this only flags vehicles
        sitting inside an explicitly provided no-parking zone, and does
        NOT use guessed frame-edge regions anymore.

        For real parking detection, use `check_illegal_parking_stationary()`
        on a video stream instead — it confirms a vehicle hasn't moved
        across many frames before flagging it.
        """
        if no_parking_zones is None:
            # No zones provided — we no longer guess regions, since that
            # caused false positives on normal traffic. Returns nothing.
            return []

        violations = []
        vehicles = [d for d in detections
                    if d["class_name"] in ("car", "truck", "bus", "motorcycle")]

        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2

            for zx1, zy1, zx2, zy2 in no_parking_zones:
                if zx1 <= vcx <= zx2 and zy1 <= vcy <= zy2:
                    violations.append(Violation(
                        violation_type="Illegal Parking",
                        confidence=0.70,
                        bbox=vehicle["bbox"],
                        description=f"{vehicle['class_name'].capitalize()} inside marked no-parking zone.",
                        severity="Medium"
                    ))
                    break

        return violations

    def check_illegal_parking_stationary(self, tracked_objects: dict, tracker,
                                          stationary_frame_threshold: int = 45,
                                          no_parking_zones: list = None) -> list[Violation]:
        """
        Video-based parking check: a vehicle is flagged only if it has
        stayed within a small movement radius for `stationary_frame_threshold`
        consecutive frames (e.g. 45 frames ~ 1.5s at 30fps -- raise this for
        real-world "parked" vs "stopped at light" distinction, e.g. 300+
        frames ~ 10s).

        tracked_objects: {object_id: detection_dict} for the CURRENT frame,
                          as returned by CentroidTracker.update()
        tracker:          the CentroidTracker instance (for movement history)
        no_parking_zones: optional list of (x1,y1,x2,y2) -- if given, only
                          flags stationary vehicles inside these zones.
                          If None, flags ANY vehicle stationary long enough.
        """
        violations = []

        for oid, det in tracked_objects.items():
            if det["class_name"] not in ("car", "truck", "bus", "motorcycle"):
                continue

            stationary_frames = tracker.get_stationary_duration(oid)
            if stationary_frames < stationary_frame_threshold:
                continue

            x1, y1, x2, y2 = det["bbox"]

            if no_parking_zones:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                inside_zone = any(
                    zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                    for zx1, zy1, zx2, zy2 in no_parking_zones
                )
                if not inside_zone:
                    continue

            violations.append(Violation(
                violation_type="Illegal Parking",
                confidence=min(0.95, 0.6 + stationary_frames / 200),
                bbox=det["bbox"],
                description=(f"{det['class_name'].capitalize()} stationary for "
                             f"{stationary_frames} frames (track #{oid})."),
                severity="High" if stationary_frames > 150 else "Medium",
            ))

        return violations

    def check_stop_line(self, image: np.ndarray, detections: list[dict],
                         stop_line_y: int = None) -> list[Violation]:
        """
        Check if vehicles crossed the stop line.

        stop_line_y: the actual pixel Y-coordinate of the painted stop
        line in THIS specific camera view. There is no reliable way to
        guess this from the image alone — a fixed "80% down the frame"
        default was previously used and caused constant false positives,
        since it really just measured "vehicle is near the camera," not
        "vehicle crossed a stop line."

        If stop_line_y is not provided, this check is skipped entirely
        (returns no violations) rather than guessing.
        """
        if stop_line_y is None:
            return []

        violations = []
        vehicles = [d for d in detections
                    if d["class_name"] in ("car", "motorcycle", "truck", "bus")]

        for vehicle in vehicles:
            _, _, _, vy2 = vehicle["bbox"]
            if vy2 >= stop_line_y:
                violations.append(Violation(
                    violation_type="Stop Line",
                    confidence=0.80,
                    bbox=vehicle["bbox"],
                    description=f"{vehicle['class_name'].capitalize()} crossed the stop line.",
                    severity="High"
                ))

        return violations

    # ------------------------------------------------------------------ #
    #  Main pipeline
    # ------------------------------------------------------------------ #

    def analyze_image(self, image: np.ndarray,
                       stop_line_y: int = None,
                       no_parking_zones: list = None,
                       traffic_light_roi: tuple = None) -> tuple[list[dict], list[Violation]]:
        """
        Run all violation checks on a SINGLE STILL IMAGE.
        Note: illegal parking and wrong-side driving cannot be reliably
        detected from one frame (they require movement over time) --
        for those, use analyze_video_frame() on a video stream.
        """
        detections = self._get_base_detections(image)
        violations = []
        violations += self.check_helmet(image, detections)
        violations += self.check_seatbelt(image, detections)
        violations += self.check_triple_riding(image, detections)
        violations += self.check_illegal_parking_single_frame(image, detections, no_parking_zones)
        violations += self.check_stop_line(image, detections, stop_line_y)
        violations += self.check_red_light(image, detections, traffic_light_roi, stop_line_y)
        return detections, violations

    def analyze_video_frame(self, image: np.ndarray, tracker,
                             stop_line_y: int = None,
                             no_parking_zones: list = None,
                             stationary_frame_threshold: int = 45,
                             traffic_light_roi: tuple = None,
                             expected_direction: str = None) -> tuple[list[dict], list[Violation], dict]:
        """
        Run all violation checks on ONE FRAME of a video, using the
        provided CentroidTracker to maintain identity + stationary
        duration + direction-of-travel across frames. Call this once
        per frame, reusing the SAME tracker instance across the whole
        video.

        Returns: (detections, violations, tracked_objects)
        """
        detections = self._get_base_detections(image)
        tracked_objects = tracker.update(detections)

        violations = []
        violations += self.check_helmet(image, detections)
        violations += self.check_seatbelt(image, detections)
        violations += self.check_triple_riding(image, detections)
        violations += self.check_illegal_parking_stationary(
            tracked_objects, tracker, stationary_frame_threshold, no_parking_zones
        )
        violations += self.check_stop_line(image, detections, stop_line_y)
        violations += self.check_red_light(image, detections, traffic_light_roi, stop_line_y)
        violations += self.check_wrong_side_driving(tracked_objects, tracker, expected_direction)

        return detections, violations, tracked_objects

    def draw_violations(self, image: np.ndarray,
                         detections: list[dict],
                         violations: list[Violation]) -> np.ndarray:
        """Draw detections and violations on image."""
        output = image.copy()

        # Draw all detections in green
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 1)

        # Draw violations with red overlays
        for v in violations:
            x1, y1, x2, y2 = v.bbox
            color = VIOLATION_COLORS.get(v.violation_type, (0, 0, 255))

            # Thick box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            # Semi-transparent fill
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

            # Label
            label = f"{v.violation_type} ({v.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(output, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Summary overlay (top-left corner)
        summary = f"Violations: {len(violations)}"
        cv2.rectangle(output, (5, 5), (210, 35), (0, 0, 0), -1)
        cv2.putText(output, summary, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255) if violations else (0, 255, 0), 2)

        return output
