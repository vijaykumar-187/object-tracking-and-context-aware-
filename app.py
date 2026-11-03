"""Local Streamlit dashboard for Phase 2 person detection and cropping."""

from __future__ import annotations

import io
import logging
import zipfile
import csv
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from src.detection.detect import load_detector
from src.detection.video_processor import process_video

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "phase2_results"
CAMERAS = [("Camera 1", "camera1"), ("Camera 2", "camera2"), ("Camera 3", "camera3")]
logging.basicConfig(filename=ROOT / "outputs" / "phase2_app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

st.set_page_config(page_title="Person Re-ID | Vision Analytics", page_icon="👁", layout="wide")
st.markdown("""
<style>
:root { --bg:#0b1017; --panel:#111a24; --panel2:#151f2b; --line:#263545; --text:#edf4fb; --muted:#91a1b3; --cyan:#50c8ed; --green:#47d7a1; }
.stApp { background: radial-gradient(ellipse at 80% -20%, #152b3d 0, transparent 42%), var(--bg); color:var(--text); font-family:'Segoe UI',Arial,sans-serif; }
[data-testid="stSidebar"] { background:#0e151e; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
h1,h2,h3 { font-family:'Segoe UI',Arial,sans-serif!important; letter-spacing:-.03em; }
.eyebrow { color:var(--cyan); font-size:.72rem; font-weight:700; letter-spacing:.16em; }
.hero { padding:1.35rem 0 .7rem; }
.hero p,.muted { color:var(--muted); }
.metric { background:linear-gradient(145deg,#141f2b,#101720); border:1px solid var(--line); border-radius:15px; padding:17px 19px; min-height:100px; }
.metric-label { color:var(--muted); font-size:.79rem; }
.metric-value { color:var(--text); font:700 1.8rem 'Segoe UI',Arial,sans-serif; margin-top:5px; }
.camera-card { background:linear-gradient(145deg,#141f2b,#101720); border:1px solid var(--line); border-radius:16px; padding:17px; min-height:110px; }
.camera-title { font:700 1rem 'Segoe UI',Arial,sans-serif; margin-bottom:4px; }
.status-pill { display:inline-block; color:var(--green); border:1px solid #286953; background:#10271f; padding:5px 10px; border-radius:99px; font-size:.72rem; font-weight:700; letter-spacing:.04em; }
.section-rule { border-top:1px solid var(--line); margin:1.1rem 0 1.5rem; }
div.stButton > button[kind="primary"] { background:#168bb6; border:1px solid #28b9e8; color:white; border-radius:10px; font-weight:700; padding:.65rem 1.1rem; }
div.stButton > button[kind="primary"]:hover { background:#11799e; border-color:#50c8ed; }
[data-testid="stFileUploader"] { background:transparent; }
[data-testid="stMetric"] { background:transparent; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load YOLO once per Streamlit process and reuse it for all cameras."""
    return load_detector()


def metric_card(label: str, value: int | str) -> None:
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def make_zip(run_dir: Path) -> Path:
    archive_path = run_dir / "complete_results.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in run_dir.rglob("*"):
            if file_path.is_file() and file_path != archive_path:
                archive.write(file_path, file_path.relative_to(run_dir))
    return archive_path


def show_results() -> None:
    results = st.session_state.get("results", [])
    run_dir_value = st.session_state.get("run_dir")
    if not results or not run_dir_value:
        return
    run_dir = Path(run_dir_value)
    st.markdown('<div class="section-rule"></div><div class="eyebrow">OUTPUT REVIEW</div>', unsafe_allow_html=True)
    st.header("Processing Results")
    st.caption("Detections are frame-level observations. No person identities or cross-camera matches are generated.")
    try:
        zip_path = make_zip(run_dir)
        st.download_button("↓  Download complete results (.zip)", zip_path.read_bytes(), file_name="person_detection_results.zip", mime="application/zip", key="download_all")
    except OSError:
        st.error("The results archive could not be created. Individual output files are still available below.")

    for result in results:
        camera = result["camera_id"].replace("camera", "Camera ")
        with st.expander(f"{camera}  ·  {result['source_name']}", expanded=True):
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown("**Processed video**")
                st.video(result["processed_video"])
                video_path = Path(result["processed_video"])
                st.download_button("Download processed video", video_path.read_bytes(), file_name=f"{result['camera_id']}_processed.mp4", mime="video/mp4", key=f"video_{result['camera_id']}")
            with right:
                m1, m2 = st.columns(2)
                m1.metric("Frames read", result["frames_read"])
                m2.metric("Frames analyzed", result["frames_processed"])
                m1.metric("Person detections", result["total_detections"])
                m2.metric("Crops saved", result["crops_saved"])
                st.caption(f"Processing time: {result['processing_seconds']:.2f} sec · Source: {result['source_name']}")
                source_path = Path(result["source_path"])
                with st.popover("Preview original video"):
                    st.video(str(source_path))
            crop_files = sorted(Path(result["crops_dir"]).glob("*.jpg"))
            confidence_by_crop = {}
            with Path(result["metadata_file"]).open(newline="", encoding="utf-8") as metadata_stream:
                confidence_by_crop = {row["crop_filename"]: float(row["confidence"]) for row in csv.DictReader(metadata_stream)}
            st.markdown("**Detected person crops**")
            if crop_files:
                cols = st.columns(5)
                for index, crop_path in enumerate(crop_files):
                    with cols[index % 5]:
                        with Image.open(crop_path) as crop_image:
                            st.image(crop_image.copy(), use_container_width=True)
                        confidence_value = confidence_by_crop.get(crop_path.name)
                        confidence_label = f" · {confidence_value:.2f} confidence" if confidence_value is not None else ""
                        st.caption(f"{camera} · frame {crop_path.stem.split('_frame')[-1].split('_')[0]}{confidence_label}")
            else:
                st.info("No person crops were produced for this video.")
            crop_bytes = io.BytesIO()
            with zipfile.ZipFile(crop_bytes, "w", zipfile.ZIP_DEFLATED) as crops_zip:
                for crop_path in crop_files:
                    crops_zip.write(crop_path, crop_path.name)
            st.download_button("Download camera crops (.zip)", crop_bytes.getvalue(), file_name=f"{result['camera_id']}_crops.zip", mime="application/zip", key=f"crops_{result['camera_id']}")


with st.sidebar:
    st.markdown("## ◉  PERSON RE-ID")
    st.caption("VISION ANALYTICS  /  PROTOTYPE")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("**Workspace**")
    st.markdown("▰  Detection dashboard")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("**CURRENT PHASE**  ·  PHASE 2")
    st.caption("FEATURE: PERSON DETECTION & CROPPING")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("**Processing status**")
    if st.session_state.get("is_processing"):
        st.warning("Processing videos")
    elif st.session_state.get("results"):
        st.success("Last run complete")
    else:
        st.markdown('<span class="status-pill">● SYSTEM STATUS: READY</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.caption("CPU inference · YOLO26n\n\nPhase 2 only. No identity matching.")

st.markdown('<div class="hero"><div class="eyebrow">PHASE 2  /  DETECTION PIPELINE</div><h1>Person Detection Dashboard</h1><p>Daytime Person Detection &amp; Cropping Prototype<br>Upload camera videos, detect people using YOLO, and generate cropped person images.</p></div>', unsafe_allow_html=True)

uploaded = []
for start in range(0, 3, 3):
    columns = st.columns(3, gap="medium")
    for column, (camera_name, camera_id) in zip(columns, CAMERAS[start:start + 3]):
        with column:
            st.markdown(f'<div class="camera-card"><div class="camera-title">{camera_name}</div><div class="muted">Daytime video input</div></div>', unsafe_allow_html=True)
            file = st.file_uploader(f"Choose {camera_name} video", type=["mp4", "avi", "mov", "mkv", "m4v"], key=f"upload_{camera_id}", label_visibility="collapsed")
            if file is not None:
                st.caption(f"Selected · {file.name} · {file.size / (1024 * 1024):.1f} MB")
                with st.expander("Preview upload"):
                    st.video(file.getvalue())
                uploaded.append((camera_name, camera_id, file))

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">PROCESSING CONFIGURATION</div>', unsafe_allow_html=True)
cfg1, cfg2, cfg3 = st.columns([1, 1, 2])
with cfg1:
    frame_interval = st.number_input("Analyze every N frames", min_value=1, max_value=300, value=15, step=1, help="A value of 15 analyzes one out of every 15 source frames.")
with cfg2:
    confidence = st.slider("Confidence threshold", min_value=0.10, max_value=0.95, value=0.40, step=0.05)
with cfg3:
    st.caption("Videos are processed one at a time on CPU. YOLO checks sampled frames; boxes stay visible between checks and refresh at the next check. Crops are saved from YOLO detection frames.")

if st.button("▶  START PROCESSING", type="primary", disabled=not uploaded, use_container_width=False):
    st.session_state.is_processing = True
    results = []
    run_dir = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        with st.spinner("Loading YOLO26n for CPU inference…"):
            model = get_model()
        for camera_name, camera_id, file in uploaded:
            camera_dir = run_dir / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            input_path = camera_dir / ("input" + Path(file.name).suffix.lower())
            input_path.write_bytes(file.getvalue())
            st.markdown(f"**{camera_name} — Processing**")
            status = st.empty()
            progress = st.progress(0.0)
            def update_progress(current: int, total: int, stage: str) -> None:
                total_label = str(total) if total > 0 else "unknown"
                status.caption(f"{stage} · Frame {current:,} of {total_label}")
                if total > 0:
                    progress.progress(min(current / total, 1.0))
            try:
                result = process_video(camera_id, input_path, model, camera_dir, int(frame_interval), float(confidence), update_progress)
                result_dict = result.__dict__.copy()
                result_dict["source_path"] = str(input_path)
                results.append(result_dict)
                status.success(f"{camera_name} complete · {result.frames_processed:,} sampled frames analyzed")
                if result.total_frames == 0:
                    st.warning("OpenCV did not report a total frame count; progress was shown as a frame counter.")
            except (ValueError, OSError) as exc:
                logging.exception("Input or processing error for %s", file.name)
                status.error(f"{camera_name}: {exc}")
            except Exception:
                logging.exception("Unexpected processing failure for %s", file.name)
                status.error(f"{camera_name} could not be processed. Technical details were saved to outputs/phase2_app.log.")
        st.session_state.results = results
        st.session_state.run_dir = str(run_dir)
        st.session_state.is_processing = False
        if results:
            st.success("Processing run finished. Review results below.")
        else:
            st.error("No videos completed successfully. Review the messages above and try again.")
    except Exception:
        logging.exception("Could not start processing run")
        st.session_state.is_processing = False
        st.error("The processing run could not be started. Check output folder permissions and free disk space.")

stats = st.session_state.get("results", [])
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">LIVE RUN SUMMARY</div>', unsafe_allow_html=True)
stats_cols = st.columns(4)
counts = [len(uploaded), sum(x["frames_processed"] for x in stats), sum(x["total_detections"] for x in stats), sum(x["crops_saved"] for x in stats)]
for col, label, value in zip(stats_cols, ["Videos uploaded", "Frames processed", "Person detections", "Crops generated"], counts):
    with col:
        metric_card(label, value)

show_results()
