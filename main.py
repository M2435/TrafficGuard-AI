"""
Main Pipeline Entry Point
Usage:
    python main.py --image path/to/image.jpg
    python main.py --batch data/raw_images/
    python main.py --test
    streamlit run src/dashboard.py   (for UI)
"""

import argparse
import cv2
import sys
from pathlib import Path


def run_on_image(image_path: str, show: bool = True):
    """Run full pipeline on a single image."""
    from src.preprocess import preprocess_image
    from src.violations import ViolationDetector
    from src.ocr import LicensePlateReader

    print(f"\n[Pipeline] Processing: {image_path}")

    # Step 1 - Preprocess
    preprocessed = preprocess_image(image_path)
    print("[Pipeline] Preprocessing done.")

    # Step 2 - Detect violations
    detector = ViolationDetector(device="cpu")
    detections, violations = detector.analyze_image(preprocessed)
    print(f"[Pipeline] Found {len(detections)} objects, {len(violations)} violations.")

    # Step 3 - Read license plates
    ocr = LicensePlateReader(device="cpu")
    plates = ocr.read_plate(preprocessed)
    print(f"[Pipeline] Plates detected: {[p['text'] for p in plates]}")

    # Step 4 - Annotate
    annotated = detector.draw_violations(preprocessed, detections, violations)
    annotated = ocr.draw_plates(annotated, plates)

    # Step 5 - Save output
    output_path = Path("outputs") / f"result_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)
    print(f"[Pipeline] Saved annotated image → {output_path}")

    # Step 6 - Print violation summary
    if violations:
        print("\n===== VIOLATIONS DETECTED =====")
        for v in violations:
            print(f"  [{v.severity}] {v.violation_type}: {v.description}")
            print(f"         Plate: {v.plate_number} | Confidence: {v.confidence:.0%}")
    else:
        print("\n[Pipeline] No violations detected.")

    if show:
        cv2.imshow("Traffic Violation Detection", annotated)
        print("[Pipeline] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return violations


def run_on_video(video_path: str, show: bool = True, skip_frames: int = 2,
                  stationary_threshold: int = 45, save_output: bool = True,
                  stop_line_y: int = None, traffic_light_roi: tuple = None,
                  expected_direction: str = None):
    """
    Run the full pipeline on a video file.

    skip_frames: process every Nth frame (2 = process every 2nd frame)
                 to keep CPU inference speed reasonable. OCR and full
                 detection still run on every PROCESSED frame.
    stationary_threshold: number of consecutive (processed) frames a
                 vehicle must stay still before being flagged as
                 illegally parked. Tune based on skip_frames + fps.
                 e.g. video at 30fps with skip_frames=2 -> ~15
                 processed frames/sec, so threshold=45 ~ 3 seconds.
    stop_line_y: pixel Y-coordinate of the stop line for THIS camera.
                 Required for stop-line AND red-light checks; if None,
                 both are skipped (no guessing).
    traffic_light_roi: (x1,y1,x2,y2) pixel box around the traffic light
                 bulb for THIS camera. Required for red-light check.
    expected_direction: "up"/"down"/"left"/"right" -- the direction
                 traffic SHOULD move in this camera view. Required for
                 wrong-side-driving check; if None, it's skipped.
    """
    from src.violations import ViolationDetector
    from src.ocr import LicensePlateReader
    from src.tracker import CentroidTracker

    print(f"\n[Pipeline] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] {width}x{height} @ {fps:.1f}fps, ~{total_frames} frames")

    if stop_line_y is None:
        print("[Pipeline] No --stop-line-y given -> Stop Line & Red Light checks are SKIPPED.")
    if traffic_light_roi is None:
        print("[Pipeline] No --traffic-light-roi given -> Red Light check is SKIPPED.")
    if expected_direction is None:
        print("[Pipeline] No --expected-direction given -> Wrong Side check is SKIPPED.")

    detector = ViolationDetector(device="cpu")
    ocr = LicensePlateReader(device="cpu")
    tracker = CentroidTracker(max_disappeared=int(fps * 2), max_distance=80)

    writer = None
    if save_output:
        out_path = Path("outputs") / f"result_{Path(video_path).stem}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps / max(skip_frames, 1), (width, height))

    frame_idx = 0
    processed_idx = 0
    all_violations = []
    seen_violation_keys = set()   # avoid logging the same parked vehicle every frame

    # OCR is expensive — only run it every N processed frames to keep things moving
    OCR_EVERY = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip_frames != 0:
            continue

        processed_idx += 1
        detections, violations, tracked = detector.analyze_video_frame(
            frame, tracker,
            stationary_frame_threshold=stationary_threshold,
            stop_line_y=stop_line_y,
            traffic_light_roi=traffic_light_roi,
            expected_direction=expected_direction,
        )

        plates = []
        if processed_idx % OCR_EVERY == 0:
            plates = ocr.read_plate(frame)

        annotated = detector.draw_violations(frame, detections, violations)
        if plates:
            annotated = ocr.draw_plates(annotated, plates)

        # Log only NEW violations (dedupe by type + track-ish position bucket)
        for v in violations:
            key = (v.violation_type, v.bbox[0] // 40, v.bbox[1] // 40)
            if key not in seen_violation_keys:
                seen_violation_keys.add(key)
                all_violations.append(v)
                print(f"  [Frame {frame_idx}] [{v.severity}] {v.violation_type}: {v.description}")

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("Traffic Violation Detection (video)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Pipeline] Stopped by user (q pressed).")
                break

        if processed_idx % 30 == 0:
            print(f"[Pipeline] ...processed frame {frame_idx}/{total_frames}")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n[Pipeline] Video processing complete.")
    print(f"[Pipeline] Total unique violations found: {len(all_violations)}")
    if save_output:
        print(f"[Pipeline] Annotated video saved -> outputs/result_{Path(video_path).stem}.mp4")

    return all_violations



    """Process all images in a folder."""
    folder_path = Path(folder)
    images = list(folder_path.glob("*.jpg")) + \
             list(folder_path.glob("*.jpeg")) + \
             list(folder_path.glob("*.png"))

    print(f"[Batch] Found {len(images)} images in {folder}")
    all_violations = []

    for img_path in images:
        try:
            violations = run_on_image(str(img_path), show=False)
            all_violations.extend(violations)
        except Exception as e:
            print(f"[Error] {img_path.name}: {e}")

    print(f"\n[Batch Complete] Total violations across {len(images)} images: {len(all_violations)}")


def run_test():
    """Quick environment test — no images needed."""
    print("\n===== Environment Test =====")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[FAIL] PyTorch not installed")

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("[FAIL] OpenCV not installed")

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("YOLOv8: OK")
    except Exception as e:
        print(f"[FAIL] YOLOv8: {e}")

    try:
        import easyocr
        print("EasyOCR: OK (model downloads on first use)")
    except ImportError:
        print("[FAIL] EasyOCR not installed")

    try:
        import streamlit
        print(f"Streamlit: {streamlit.__version__}")
    except ImportError:
        print("[FAIL] Streamlit not installed")

    print("\n===== All checks done =====")
    print("Run dashboard: streamlit run src/dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Violation Detection System")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--video",  type=str, help="Path to a video file (mp4, avi, etc.)")
    parser.add_argument("--batch",  type=str, help="Path to folder of images")
    parser.add_argument("--test",   action="store_true", help="Test environment setup")
    parser.add_argument("--no-show", action="store_true", help="Don't display output window")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame in video (default: 2)")
    parser.add_argument("--stationary-threshold", type=int, default=45,
                         help="Processed frames a vehicle must stay still to flag parking (default: 45)")
    parser.add_argument("--stop-line-y", type=int, default=None,
                         help="Pixel Y-coordinate of the stop line for this camera (required for Stop Line & Red Light checks)")
    parser.add_argument("--traffic-light-roi", type=int, nargs=4, default=None,
                         metavar=("X1", "Y1", "X2", "Y2"),
                         help="Pixel box (x1 y1 x2 y2) around the traffic light bulb (required for Red Light check)")
    parser.add_argument("--expected-direction", type=str, default=None,
                         choices=["up", "down", "left", "right"],
                         help="Direction traffic should move in this camera (required for Wrong Side check)")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.image:
        run_on_image(args.image, show=not args.no_show)
    elif args.video:
        roi = tuple(args.traffic_light_roi) if args.traffic_light_roi else None
        run_on_video(args.video, show=not args.no_show,
                     skip_frames=args.skip_frames,
                     stationary_threshold=args.stationary_threshold,
                     stop_line_y=args.stop_line_y,
                     traffic_light_roi=roi,
                     expected_direction=args.expected_direction)
    elif args.batch:
        run_batch(args.batch)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py --test")
        print("  python main.py --image data/raw_images/sample.jpg")
        print("  python main.py --video data/raw_images/sample.mp4")
        print("  streamlit run src/dashboard.py")
