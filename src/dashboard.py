"""
Phase 6 - Streamlit Dashboard
Run with: streamlit run src/dashboard.py
from inside the traffic_violation_system folder
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import sys
import tempfile
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Fix import path so src/ modules are found when running via streamlit
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.violations import ViolationDetector
from src.ocr import LicensePlateReader
from src.tracker import CentroidTracker

# Page config
st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stMetric { background: #f0f2f6; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Cache models so they load only once per session ────────────────────────────
@st.cache_resource
def load_models():
    detector = ViolationDetector(device="cpu")
    ocr = LicensePlateReader(device="cpu")
    return detector, ocr


# ── Records helpers ─────────────────────────────────────────────────────────────
RECORDS_FILE = ROOT / "outputs" / "violations.json"

def load_records() -> pd.DataFrame:
    if not RECORDS_FILE.exists():
        return pd.DataFrame()
    with open(RECORDS_FILE) as f:
        records = json.load(f)
    return pd.DataFrame(records) if records else pd.DataFrame()


def save_record(data: dict):
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE) as f:
            records = json.load(f)
    records.append(data)
    with open(RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=2)


# ── Image processing ─────────────────────────────────────────────────────────
def process_uploaded_image(file_bytes: bytes, detector, ocr,
                            stop_line_y=None, traffic_light_roi=None):
    """Decode → resize → detect → OCR → annotate. Returns originals + results."""
    arr = np.frombuffer(file_bytes, np.uint8)
    original = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if original is None:
        return None, None, [], [], []

    preprocessed = cv2.resize(original, (640, 640))
    detections, violations = detector.analyze_image(
        preprocessed, stop_line_y=stop_line_y, traffic_light_roi=traffic_light_roi
    )
    plates = ocr.read_plate(preprocessed)

    plate_text = plates[0]["text"] if plates else "Unknown"
    for v in violations:
        v.plate_number = plate_text

    annotated = detector.draw_violations(preprocessed, detections, violations)
    annotated = ocr.draw_plates(annotated, plates)

    return original, annotated, detections, violations, plates


# ── Video processing ─────────────────────────────────────────────────────────
def process_uploaded_video(video_path: str, detector, ocr, progress_callback=None,
                            skip_frames: int = 3, stationary_threshold: int = 30,
                            stop_line_y=None, traffic_light_roi=None,
                            expected_direction=None, max_seconds: int = None):
    """
    Run the full violation pipeline on a video file.
    Returns: output_video_path, all_violations (list of dicts), summary stats
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_seconds:
        total_frames = min(total_frames, int(max_seconds * fps))

    tracker = CentroidTracker(max_disappeared=int(fps * 2), max_distance=80)

    out_path = ROOT / "outputs" / "annotated" / f"video_{Path(video_path).stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps / skip_frames, 1), (width, height))

    frame_idx = 0
    processed_idx = 0
    all_violations = []
    seen_keys = set()
    OCR_EVERY = 12
    detections_total = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
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
        detections_total += len(detections)

        plates = []
        if processed_idx % OCR_EVERY == 0:
            plates = ocr.read_plate(frame)

        annotated = detector.draw_violations(frame, detections, violations)
        if plates:
            annotated = ocr.draw_plates(annotated, plates)
        writer.write(annotated)

        for v in violations:
            key = (v.violation_type, v.bbox[0] // 40, v.bbox[1] // 40)
            if key not in seen_keys:
                seen_keys.add(key)
                plate_text = plates[0]["text"] if plates else "Unknown"
                all_violations.append({
                    "type": v.violation_type,
                    "confidence": round(v.confidence, 3),
                    "description": v.description,
                    "plate": plate_text,
                    "severity": v.severity,
                    "timestamp": v.timestamp,
                    "frame": frame_idx,
                })

        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0), processed_idx, len(all_violations))

    cap.release()
    writer.release()

    summary = {
        "total_frames": frame_idx,
        "processed_frames": processed_idx,
        "total_detections": detections_total,
        "unique_violations": len(all_violations),
    }
    return str(out_path), all_violations, summary


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚦 Traffic AI System")
st.sidebar.markdown("**Flipkart × Bengaluru Traffic Police**")
st.sidebar.caption("AI-powered violation detection using YOLOv8 + EasyOCR")
st.sidebar.divider()
page = st.sidebar.radio("Navigate to", [
    "🔍 Analyze",
    "📊 Analytics Dashboard",
    "📋 Violation Records"
])

# ── Page 1: Analyze (Image OR Video) ──────────────────────────────────────────
if page == "🔍 Analyze":
    st.title("🔍 Traffic Violation Detection")

    # --- Mode switch ---
    mode = st.radio("Choose input type:", ["📷 Image", "🎥 Video"], horizontal=True)
    st.divider()

    # Advanced settings (shared, optional, collapsed by default)
    with st.expander("⚙️ Advanced settings (optional — for Stop Line / Red Light / Wrong Side checks)"):
        st.caption("These checks need camera-specific calibration. Leave blank to skip them safely.")
        use_stop_line = st.checkbox("Enable Stop Line check")
        stop_line_y = st.number_input("Stop line Y-coordinate (pixels)", min_value=0, max_value=2000, value=400, step=10) if use_stop_line else None

        use_red_light = st.checkbox("Enable Red Light check (requires Stop Line above)")
        roi = None
        if use_red_light:
            rc1, rc2, rc3, rc4 = st.columns(4)
            rx1 = rc1.number_input("Light X1", min_value=0, value=10)
            ry1 = rc2.number_input("Light Y1", min_value=0, value=10)
            rx2 = rc3.number_input("Light X2", min_value=0, value=40)
            ry2 = rc4.number_input("Light Y2", min_value=0, value=40)
            roi = (int(rx1), int(ry1), int(rx2), int(ry2))

        expected_direction = None
        if mode == "🎥 Video":
            use_wrong_side = st.checkbox("Enable Wrong Side Driving check (video only)")
            if use_wrong_side:
                expected_direction = st.selectbox("Expected traffic direction", ["up", "down", "left", "right"])

    # ════════════════════════════════════════════════════════════════════
    #  IMAGE MODE
    # ════════════════════════════════════════════════════════════════════
    if mode == "📷 Image":
        uploaded = st.file_uploader(
            "Upload a traffic image",
            type=["jpg", "jpeg", "png"],
            help="Best results with clear road/intersection images"
        )

        if uploaded is not None:
            with st.spinner("⚙️ Loading AI models (first time takes ~30 seconds)..."):
                detector, ocr = load_models()

            with st.spinner("🔍 Analyzing image..."):
                file_bytes = uploaded.read()
                original, annotated, detections, violations, plates = process_uploaded_image(
                    file_bytes, detector, ocr, stop_line_y=stop_line_y, traffic_light_roi=roi
                )

            if original is None:
                st.error("Could not read this image. Try a different file.")
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Original")
                st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("🚨 Analyzed")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🚗 Objects Detected", len(detections))
            m2.metric("⚠️ Violations Found", len(violations))
            m3.metric("🔢 Plates Read", len(plates))
            m4.metric("Status", "🔴 Violations!" if violations else "🟢 Clear")

            if violations:
                st.subheader("⚠️ Violation Details")
                for v in violations:
                    severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(v.severity, "⚪")
                    with st.expander(f"{severity_icon} {v.violation_type}  |  Confidence: {v.confidence:.0%}  |  Plate: {v.plate_number}"):
                        c1, c2, c3 = st.columns(3)
                        c1.write(f"**Type:** {v.violation_type}")
                        c2.write(f"**Severity:** {v.severity}")
                        c3.write(f"**Time:** {v.timestamp}")
                        st.write(f"**Description:** {v.description}")

                    save_record({
                        "type": v.violation_type,
                        "confidence": round(v.confidence, 3),
                        "description": v.description,
                        "plate": v.plate_number,
                        "severity": v.severity,
                        "timestamp": v.timestamp,
                        "source": uploaded.name,
                    })
            else:
                st.success("✅ No violations detected in this image.")

            if detections:
                st.subheader("🚘 Objects Detected")
                counts = {}
                for d in detections:
                    counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
                det_df = pd.DataFrame(list(counts.items()), columns=["Object", "Count"])
                st.dataframe(det_df, use_container_width=True, hide_index=True)

            if plates:
                st.subheader("🔢 License Plates")
                for p in plates:
                    st.info(f"Plate detected: **{p['text']}**  (confidence: {p['confidence']:.0%})")

            _, buf = cv2.imencode(".jpg", annotated)
            st.download_button(
                "⬇️ Download Annotated Image",
                buf.tobytes(),
                file_name=f"violation_{uploaded.name}",
                mime="image/jpeg"
            )
        else:
            st.info("👆 Upload a traffic image above to get started.")

    # ════════════════════════════════════════════════════════════════════
    #  VIDEO MODE
    # ════════════════════════════════════════════════════════════════════
    else:
        uploaded_video = st.file_uploader(
            "Upload a traffic video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Shorter clips (10-30s) process much faster on CPU"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            skip_frames = st.slider("Process every Nth frame", 1, 10, 3,
                                     help="Higher = faster but less smooth tracking")
        with col_b:
            max_seconds = st.slider("Limit processing to first N seconds", 5, 120, 20,
                                     help="Cap runtime for long videos")

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("▶️ Run Analysis", type="primary"):
                with st.spinner("⚙️ Loading AI models (first time takes ~30 seconds)..."):
                    detector, ocr = load_models()

                # Save uploaded video to a temp file (OpenCV needs a file path)
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(uploaded_video.read())
                    temp_video_path = tmp.name

                progress_bar = st.progress(0, text="Starting video analysis...")
                stats_placeholder = st.empty()

                def update_progress(frac, processed_count, violation_count):
                    progress_bar.progress(frac, text=f"Processing... frame progress {frac:.0%}")
                    stats_placeholder.caption(
                        f"Processed frames: {processed_count}  |  Violations found so far: {violation_count}"
                    )

                start_time = time.time()
                out_video_path, video_violations, summary = process_uploaded_video(
                    temp_video_path, detector, ocr,
                    progress_callback=update_progress,
                    skip_frames=skip_frames,
                    stationary_threshold=30,
                    stop_line_y=stop_line_y,
                    traffic_light_roi=roi,
                    expected_direction=expected_direction,
                    max_seconds=max_seconds,
                )
                elapsed = time.time() - start_time

                progress_bar.progress(1.0, text="Done!")
                Path(temp_video_path).unlink(missing_ok=True)

                if out_video_path is None:
                    st.error("Could not process this video. Try a different file.")
                    st.stop()

                st.success(f"✅ Analysis complete in {elapsed:.1f}s")

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frames Processed", summary["processed_frames"])
                m2.metric("Total Detections", summary["total_detections"])
                m3.metric("Unique Violations", summary["unique_violations"])
                m4.metric("Status", "🔴 Violations!" if video_violations else "🟢 Clear")

                st.subheader("🎬 Annotated Video")
                with open(out_video_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button("⬇️ Download Annotated Video", video_bytes,
                                    file_name=f"violation_{Path(out_video_path).name}",
                                    mime="video/mp4")

                if video_violations:
                    st.subheader("⚠️ Violations Found")
                    for v in video_violations:
                        severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(v["severity"], "⚪")
                        with st.expander(f"{severity_icon} {v['type']}  |  Frame {v['frame']}  |  Confidence: {v['confidence']:.0%}"):
                            st.write(f"**Description:** {v['description']}")
                            st.write(f"**Plate:** {v['plate']}")
                            st.write(f"**Severity:** {v['severity']}")
                        save_record({**v, "source": uploaded_video.name})
                else:
                    st.success("✅ No violations detected in this video.")
        else:
            st.info("👆 Upload a traffic video above to get started.")
            st.caption("Tip: CPU processing is slow — keep clips short (10-20s) for a fast demo.")


# ── Page 2: Analytics ─────────────────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.title("📊 Violation Analytics")
    df = load_records()

    if df.empty:
        st.info("No violation records yet. Analyze an image or video first!")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Violations", len(df))
    c2.metric("Unique Plates", df["plate"].nunique())
    c3.metric("High Severity", len(df[df["severity"] == "High"]))
    c4.metric("Most Common", df["type"].mode()[0])

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Violations by Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["type"].value_counts()
        ax.barh(counts.index, counts.values, color="#e94560")
        ax.set_xlabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Severity Breakdown")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sev = df["severity"].value_counts()
        ax2.pie(sev.values, labels=sev.index,
                colors=["#e94560", "#f5a623", "#27ae60"],
                autopct="%1.1f%%")
        st.pyplot(fig2)

    st.subheader("Violations Over Time")
    time_df = df.set_index("timestamp").resample("h").size().reset_index()
    time_df.columns = ["Time", "Count"]
    st.line_chart(time_df.set_index("Time"))


# ── Page 3: Records ───────────────────────────────────────────────────────────
elif page == "📋 Violation Records":
    st.title("📋 All Violation Records")
    df = load_records()

    if df.empty:
        st.info("No records yet.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.multiselect("Filter by type",
            df["type"].unique(), default=list(df["type"].unique()))
    with col2:
        sev_filter = st.multiselect("Filter by severity",
            ["High", "Medium", "Low"], default=["High", "Medium", "Low"])

    filtered = df[df["type"].isin(type_filter) & df["severity"].isin(sev_filter)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False)
    st.download_button("⬇️ Export as CSV", csv, "violations_export.csv", "text/csv")

    if st.button("🗑️ Clear All Records", type="secondary"):
        if RECORDS_FILE.exists():
            RECORDS_FILE.unlink()
        st.success("Records cleared!")
        st.rerun()
