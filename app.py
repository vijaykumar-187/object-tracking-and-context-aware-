"""Two-camera Review 3 dashboard for person tracking and appearance matching."""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.detection.detect import load_detector
from src.detection.video_processor import (
    process_video, render_selected_person_video,
)
from src.reid.appearance import (
    AGGREGATION_METHODS, evaluate_pairs, load_fingerprints,
    make_track_fingerprint_variants, rank_across_cameras, rank_candidates,
    save_crop_embeddings, save_fingerprints,
    validate_ground_truth_pairs,
)
from src.reid.hypotheses import (
    build_global_identities, load_hypotheses, save_hypothesis,
)
from src.reid.osnet_extractor import CHECKPOINT_NAME, OSNetExtractor
from src.reid.matcher import match_two_cameras, write_matches_csv
from src.reid.global_registry import (
    ensure_identity_source, find_identity_for_track, get_or_create_identity,
    load_registry, reject_track_association,
)
from src.reid.video_results import render_all_camera_tracking_videos
from src.tracking.selected_person import selected_tracking_state

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "review3_results"
logging.basicConfig(
    filename=ROOT / "outputs" / "review3_app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

st.set_page_config(page_title="Person Re-ID | Vision Analytics", page_icon="◉", layout="wide")
st.markdown("""
<style>
:root { --bg:#0a1017; --panel:#111a24; --panel2:#15212d; --line:#263747; --text:#edf4fb; --muted:#91a1b3; --cyan:#50c8ed; --green:#47d7a1; --amber:#ffc56d; }
.stApp { background:radial-gradient(ellipse at 82% -20%,#173247 0,transparent 40%),var(--bg); color:var(--text); font-family:'Segoe UI',Arial,sans-serif; }
[data-testid="stSidebar"] { background:#0e151e; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
h1,h2,h3 { font-family:'Segoe UI',Arial,sans-serif!important; letter-spacing:-.025em; }
.eyebrow { color:var(--cyan); font-size:.72rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase; }
.hero { padding:1.25rem 0 .8rem; }
.hero p,.muted { color:var(--muted); }
.metric { background:linear-gradient(145deg,#15212d,#101720); border:1px solid var(--line); border-radius:15px; padding:16px 18px; min-height:98px; }
.metric-label { color:var(--muted); font-size:.78rem; }
.metric-value { color:var(--text); font:700 1.75rem 'Segoe UI',Arial,sans-serif; margin-top:5px; }
.panel { background:linear-gradient(145deg,#141f2b,#101720); border:1px solid var(--line); border-radius:16px; padding:16px; }
.camera-title { font:700 1.05rem 'Segoe UI',Arial,sans-serif; }
.section-rule { border-top:1px solid var(--line); margin:1.15rem 0 1.5rem; }
.status-pill { display:inline-block; color:var(--green); border:1px solid #286953; background:#10271f; padding:5px 10px; border-radius:99px; font-size:.72rem; font-weight:700; }
div.stButton > button[kind="primary"] { background:#168bb6; border:1px solid #28b9e8; color:white; border-radius:10px; font-weight:700; padding:.65rem 1.1rem; }
div.stButton > button[kind="primary"]:hover { background:#11799e; border-color:#50c8ed; }
[data-testid="stMetric"] { background:transparent; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(model_path: str, model_mtime_ns: int, model_size_bytes: int):
    """Load YOLO once and reuse it for detection and its appearance features."""
    del model_mtime_ns, model_size_bytes
    return load_detector(Path(model_path))


@st.cache_resource(show_spinner=False)
def get_osnet_extractor(checkpoint_path: str, checkpoint_mtime_ns: int,
                        checkpoint_size_bytes: int):
    """Load the separate pretrained Re-ID model once per dashboard session."""
    del checkpoint_mtime_ns, checkpoint_size_bytes
    return OSNetExtractor(checkpoint_path)


def metric_card(label: str, value: int | str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def make_zip(run_dir: Path) -> bytes:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return archive_bytes.getvalue()


def read_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def read_processed_frame(video_path: str, frame_number: int, modified_time: float) -> np.ndarray | None:
    """Read an inspection frame; mtime invalidates cache when the video changes."""
    del modified_time
    capture = cv2.VideoCapture(video_path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def load_crop_rows(result: dict) -> list[dict]:
    path = Path(result["crops_manifest_file"])
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["track_id"] = int(row["track_id"])
        row["quality_score"] = float(row["quality_score"])
        row["path"] = str(Path(result["crops_dir"]) / row["crop_filename"])
    return rows


def all_tracks(results: list[dict], camera_id: str) -> list[dict]:
    result = next((item for item in results if item["camera_id"] == camera_id), None)
    if result is None:
        return []
    payload = read_json(result["track_summary_file"])
    tracks = payload.get("tracks", [])
    for track in tracks:
        track.setdefault("camera_id", camera_id)
    return tracks


def get_camera_name(result: dict) -> str:
    return str(result.get("camera_name") or result["camera_id"].replace("camera", "Camera "))


def load_track_history(result: dict, track_id: int) -> list[dict]:
    """Read one local track's frame history for the selected-person panel."""
    try:
        with Path(result["tracks_file"]).open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)
                    if int(row["track_id"]) == int(track_id)]
    except (OSError, ValueError, KeyError):
        return []


def show_track_crop(result: dict, track_id: int) -> str | None:
    matches = [row for row in load_crop_rows(result) if row["track_id"] == track_id]
    if not matches:
        return None
    best = max(matches, key=lambda row: float(row["quality_score"]))
    path = Path(best["path"])
    if path.is_file():
        st.image(str(path), caption=f"Best retained crop · frame {best['frame_number']}", width=180)
        with st.expander(f"All retained crops ({len(matches)})"):
            columns = st.columns(min(4, len(matches)))
            for index, row in enumerate(matches):
                crop_path = Path(row["path"])
                if crop_path.is_file():
                    with columns[index % len(columns)]:
                        st.image(
                            str(crop_path),
                            caption=(f"Frame {row['frame_number']} · score "
                                     f"{float(row['quality_score']):.2f}"),
                            use_container_width=True,
                        )
        rejected_file = result.get("rejected_crops_file")
        if rejected_file and Path(rejected_file).is_file():
            with Path(rejected_file).open(newline="", encoding="utf-8") as stream:
                rejected = [row for row in csv.DictReader(stream)
                            if int(row["track_id"]) == int(track_id)]
            with st.expander(f"Rejected crop samples and reasons ({len(rejected)} records)"):
                shown = 0
                for row in rejected[:12]:
                    preview = Path(rejected_file).parent / "rejected_crops" / row["preview_filename"]
                    if row["preview_filename"] and preview.is_file():
                        st.image(str(preview), caption=(
                            f"Frame {row['frame_number']} · {row['reason']} · "
                            f"{row['width']}×{row['height']} px · sharpness {row['sharpness']}"
                        ), width=130)
                        shown += 1
                if not shown:
                    st.caption("No preview images were retained for this track; reason records are in the run CSV.")
        return str(path)
    return None


def show_camera_panels(results: list[dict], run_dir: Path,
                       active_selection: str | None,
                       selected_track_id: int | None,
                       paused_selection: str | None = None) -> None:
    st.markdown('<div class="eyebrow">CAMERA WORKSPACE</div>', unsafe_allow_html=True)
    st.subheader("Two-camera video workspace")
    st.subheader("Processed camera videos")
    selected_camera_id = None
    if active_selection:
        selection_parts = active_selection.split(":")
        if len(selection_parts) >= 3:
            selected_camera_id = selection_parts[-2]
    columns = st.columns(min(3, len(results)), gap="large")
    for index, result in enumerate(results):
        camera_id = result["camera_id"]
        display_name = get_camera_name(result)
        column = columns[index % len(columns)]
        with column:
            st.markdown(f'<div class="panel"><div class="camera-title">{display_name}</div>', unsafe_allow_html=True)
            if not result:
                st.info("Upload and process a video for this camera to see its results.")
            else:
                video_path = Path(result["processed_video"])
                if (camera_id == selected_camera_id and active_selection and
                        paused_selection != active_selection and selected_track_id is not None):
                    selected_path = run_dir / "selected_views" / f"{camera_id}_track{selected_track_id:04d}.mp4"
                    try:
                        if not selected_path.is_file():
                            with st.spinner(f"Preparing playback for {get_camera_name(result)} · ID {selected_track_id}…"):
                                summary = read_json(result["track_summary_file"])
                                detector_settings = summary.get("detector_settings", {})
                                tracker_settings = summary.get("tracker_settings", {})
                                frame_interval = max(1, int(detector_settings.get("frame_interval", 1)))
                                lost_window = max(frame_interval, int(tracker_settings.get("max_age_frames", frame_interval)))
                                render_selected_person_video(
                                    result["source_path"], result["tracks_file"], selected_path,
                                    display_name, f"Track ID {selected_track_id}", selected_track_id,
                                    "SELECTED", max_prediction_gap_frames=min(max(3, 2 * frame_interval), lost_window),
                                )
                        video_path = selected_path
                    except (OSError, ValueError) as exc:
                        st.warning(f"Selected-person playback could not be prepared: {exc}")
                st.video(str(video_path))
                st.caption(
                    f"Uploaded file: {result['source_name']} · {result['frames_read']:,} frames · "
                    f"{result['track_count']} local tracks · {result['processing_seconds']:.1f}s"
                )
                if result["frames_read"]:
                    frame_number = st.slider(
                        f"Inspect processed {display_name} frame",
                        min_value=1, max_value=int(result["frames_read"]), value=1,
                        key=f"inspect_{run_dir.name}_{camera_id}",
                    )
                    inspected = read_processed_frame(
                        str(result["processed_video"]), frame_number - 1,
                        Path(result["processed_video"]).stat().st_mtime,
                    )
                    if inspected is not None:
                        st.image(inspected, caption=f"{display_name} · frame {frame_number}", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def show_selected_track(results: list[dict], run_dir: Path) -> tuple[int | None, str | None, str | None]:
    st.markdown('<div class="section-rule"></div><div class="eyebrow">SELECTED PERSON</div>', unsafe_allow_html=True)
    st.subheader("Inspect a camera-local track")
    camera_ids = [item["camera_id"] for item in results]
    if not camera_ids:
        st.info("Process at least one camera video first.")
        return None, None, None
    selected_camera_id = st.selectbox(
        "Camera to inspect", camera_ids,
        format_func=lambda value: value.replace("camera", "Camera "),
        key=f"track_inspection_camera_{run_dir.name}",
    )
    camera_result = next(item for item in results if item["camera_id"] == selected_camera_id)
    tracks = all_tracks(results, selected_camera_id)
    if not tracks:
        st.info(f"No person tracks were found in {selected_camera_id}.")
        return None, selected_camera_id, None
    track_ids = [int(track["track_id"]) for track in tracks]
    format_by_id = {int(track["track_id"]): track for track in tracks}
    selected_id = st.selectbox(
        f"Available {get_camera_name(camera_result)} track IDs",
        track_ids,
        format_func=lambda tid: (
            f"ID {tid} · first seen at {format_by_id[tid]['first_time_seconds']:.1f}s · "
            f"{format_by_id[tid]['observed_frames']} observed frames"
        ),
        key=f"select_track_{Path(camera_result['track_summary_file']).parent.name}_{selected_camera_id}",
    )
    track = format_by_id[selected_id]
    active_key = f"{Path(camera_result['track_summary_file']).parent.name}:{selected_camera_id}:{selected_id}"
    active_selection = st.session_state.get("active_selection")
    paused_selection = st.session_state.get("paused_selection")
    is_selected = active_selection == active_key
    tracking_state = selected_tracking_state(
        is_selected, paused_selection == active_key, track.get("status", "ended"),
        stopped=st.session_state.get("stopped_selection") == active_key,
    )
    cols = st.columns([1, 3])
    with cols[0]:
        show_track_crop(camera_result, selected_id)
    with cols[1]:
        a, b, c = st.columns(3)
        a.metric("Track ID", selected_id)
        b.metric("Observed frames", track["observed_frames"])
        status_label = {"active at end": "ACTIVE", "temporarily lost": "LOST",
                        "ended": "COMPLETED"}.get(track["status"], track["status"].upper())
        c.metric("Track status", status_label)
        a.caption(f"First seen: frame {track['first_frame'] + 1} · {track['first_time_seconds']:.2f}s")
        b.caption(f"Last detected: frame {track['last_frame'] + 1} · {track['last_time_seconds']:.2f}s")
        c.caption(f"Highest detection confidence: {track['max_confidence']:.2f}")
        st.caption(
            f"Track duration: {track.get('duration_seconds', 0):.2f}s · "
            f"Last observed box: {track.get('last_box', 'not available')}"
        )
        history = load_track_history(camera_result, selected_id)
        if history:
            with st.expander(f"Track history · last {min(20, len(history))} frame records"):
                st.dataframe(history[-20:], hide_index=True, use_container_width=True)

    st.markdown(f"**Selected-person status:** `{tracking_state}` · `{selected_camera_id}_t{selected_id:04d}`")
    if tracking_state == "Temporarily Lost":
        st.warning("This person is not currently detected. The video shows short predictions while available; the tracker keeps looking for this same camera-local ID.")
    elif tracking_state == "Lost":
        st.warning("This track has ended. The system will not switch to another person. Select another Track ID to follow someone else.")
    elif tracking_state == "Stopped":
        st.info("Following stopped. Select an ID and press Start Tracking to follow a person.")
    elif tracking_state == "Not Tracking":
        st.caption("Select a Track ID, then press Start Tracking to highlight that person in the processed video.")

    if is_selected:
        follow_cols = st.columns(2)
        if paused_selection == active_key:
            if follow_cols[0].button("▶  Resume selected-person highlight", key=f"resume_{active_key}"):
                st.session_state.paused_selection = None
                st.rerun()
        elif follow_cols[0].button("Ⅱ  Pause selected-person highlight", key=f"pause_{active_key}"):
            st.session_state.paused_selection = active_key
            st.rerun()
        if follow_cols[1].button("■  Stop Tracking", key=f"stop_{active_key}"):
            st.session_state.active_selection = None
            st.session_state.paused_selection = None
            st.session_state.stopped_selection = active_key
            st.rerun()
    else:
        if st.button(f"▶  Start Tracking · ID {selected_id}", type="primary", key=f"follow_{active_key}"):
            st.session_state.active_selection = active_key
            st.session_state.paused_selection = None
            st.session_state.stopped_selection = None
            st.rerun()
    if track["status"] != "active at end" and tracking_state not in {"Temporarily Lost", "Lost"}:
        st.caption("This track stopped receiving detections before the clip ended. The tracker does not claim it is still visible.")
    return selected_id, selected_camera_id, active_key if active_selection == active_key else None


def show_cross_camera_matching(results: list[dict], run_dir: Path,
                               fingerprint_sets: dict[str, dict[str, object]],
                               preferred_track_id: int | None = None) -> None:
    st.markdown('<div class="eyebrow">CROSS-CAMERA PERSON SEARCH</div>', unsafe_allow_html=True)
    st.subheader("Find a person in every other processed camera")
    if not results:
        st.info("Process a camera video to start person search.")
        return
    camera_by_id = {item["camera_id"]: item for item in results}
    camera_ids = list(camera_by_id)
    source_camera = st.selectbox(
        "Source camera", camera_ids,
        format_func=lambda value: get_camera_name(camera_by_id[value]),
        key=f"matching_source_camera_{run_dir.name}",
    )
    source_tracks = all_tracks(results, source_camera)
    if not source_tracks:
        st.info(f"No person tracks were found in {get_camera_name(camera_by_id[source_camera])}.")
        return
    source_ids = [int(row["track_id"]) for row in source_tracks]
    default_id = preferred_track_id if source_camera == st.session_state.get("track_inspection_camera_" + run_dir.name) and preferred_track_id in source_ids else source_ids[0]
    selected_id = st.selectbox(
        "Select the local person ID", source_ids,
        index=source_ids.index(default_id),
        key=f"matching_query_track_{run_dir.name}_{source_camera}",
    )
    query_key = f"{source_camera}:{selected_id}"
    source_rows = [row for row in load_crop_rows(camera_by_id[source_camera])
                   if row["track_id"] == selected_id]
    if source_rows:
        source_crop = max(source_rows, key=lambda row: row["quality_score"])
        if Path(source_crop["path"]).is_file():
            st.image(
                source_crop["path"],
                caption=f"Source · {get_camera_name(camera_by_id[source_camera])} ID {selected_id} · frame {source_crop['frame_number']} · {source_crop['timestamp_seconds']}s",
                width=220,
            )
    model_choices = list(fingerprint_sets)
    if not model_choices:
        st.info("No fingerprint model is available. You can still create the source-camera tracking video; other cameras will be marked unmatched.")
        model_choices = ["No appearance model"]
    preferred = "OSNet x0.25 (MSMT17) · mean"
    default_model = preferred if preferred in model_choices else model_choices[0]
    selected_model = st.selectbox(
        "Appearance model used for these results", model_choices,
        index=model_choices.index(default_model), key=f"reid_model_{run_dir.name}",
    )
    st.caption(f"Candidate rankings below are generated with **{selected_model}**. The similarity is not a probability or identity confirmation.")
    fingerprints = fingerprint_sets.get(selected_model, {})
    if fingerprints.get(query_key) is None:
        st.info("This track has no usable appearance fingerprint. Search results will show no-match for the other cameras, but you can still render its source-camera video.")
    threshold = st.slider(
        "Minimum similarity to show a candidate", min_value=-1.0, max_value=1.0,
        value=0.50, step=0.05, key=f"candidate_threshold_{run_dir.name}",
        help="This only filters displayed candidates; it does not confirm identity.",
    )
    rankings = rank_across_cameras(query_key, fingerprints, camera_ids, threshold)
    hypotheses_path = run_dir / "cross_camera_hypotheses.json"
    for destination_id in camera_ids:
        if destination_id == source_camera:
            continue
        destination = camera_by_id[destination_id]
        ranked = rankings.get(destination_id, [])
        st.markdown(f"#### {get_camera_name(destination)}")
        if not ranked:
            st.info("No candidates met the selected similarity threshold in this camera.")
            continue
        track_by_id = {int(row["track_id"]): row for row in all_tracks(results, destination_id)}
        crop_rows = load_crop_rows(destination)
        candidates = ranked[:3]
        cols = st.columns(min(3, len(candidates)))
        for rank, (column, candidate) in enumerate(zip(cols, candidates), start=1):
            candidate_id = int(candidate["key"].split(":", 1)[1])
            with column:
                st.markdown(f"**#{rank} · ID {candidate_id}**")
                matched_crops = [row for row in crop_rows if row["track_id"] == candidate_id]
                if matched_crops:
                    best = max(matched_crops, key=lambda row: row["quality_score"])
                    if Path(best["path"]).is_file():
                        st.image(best["path"], use_container_width=True)
                        st.caption(f"Frame {best['frame_number']} · {best['timestamp_seconds']}s")
                st.metric("Cosine similarity", f"{candidate['similarity']:.3f}")
                row = track_by_id.get(candidate_id, {})
                st.caption(f"{row.get('observed_frames', 0)} observed frames · {len(matched_crops)} crops")
        choice_id = st.selectbox(
            f"Track to review in {get_camera_name(destination)}",
            [int(item["key"].split(":", 1)[1]) for item in ranked],
            key=f"hypothesis_candidate_{run_dir.name}_{source_camera}_{selected_id}_{destination_id}",
        )
        picked = next(item for item in ranked if int(item["key"].split(":", 1)[1]) == choice_id)
        key = f"{source_camera}:{selected_id}|{destination_id}:{choice_id}"
        existing = load_hypotheses(hypotheses_path).get(key, {})
        selected_crops = [row for row in crop_rows if row["track_id"] == choice_id]
        if selected_crops:
            review_crop = max(selected_crops, key=lambda row: row["quality_score"])
            if Path(review_crop["path"]).is_file():
                st.image(
                    review_crop["path"],
                    caption=f"Review crop · {get_camera_name(destination)} ID {choice_id} · frame {review_crop['frame_number']} · {review_crop['timestamp_seconds']}s",
                    width=220,
                )
        st.metric("Selected pair similarity", f"{picked['similarity']:.3f}")
        options = ["UNVERIFIED", "SAME PERSON", "DIFFERENT PEOPLE", "UNCERTAIN"]
        prior = existing.get("status", "UNVERIFIED")
        decision = st.selectbox(
            f"Review {get_camera_name(camera_by_id[source_camera])} ID {selected_id} ↔ {get_camera_name(destination)} ID {choice_id}",
            options, index=options.index(prior) if prior in options else 0,
            key=f"hypothesis_status_{run_dir.name}_{source_camera}_{selected_id}_{destination_id}_{choice_id}",
        )
        if st.button("Save manual review", key=f"save_hypothesis_{run_dir.name}_{source_camera}_{selected_id}_{destination_id}_{choice_id}"):
            registry_path = run_dir / "global_identity_registry.json"
            source_name = get_camera_name(camera_by_id[source_camera])
            destination_name = get_camera_name(destination)
            try:
                if decision == "SAME PERSON":
                    global_id = find_identity_for_track(
                        registry_path, source_camera, selected_id,
                    ) or find_identity_for_track(registry_path, destination_id, choice_id)
                    global_id = global_id or get_or_create_identity(
                        registry_path, source_camera, selected_id, source_name,
                    )
                    ensure_identity_source(
                        registry_path, global_id, source_camera, selected_id, source_name,
                    )
                    associate_confirmed_track(
                        registry_path, global_id, destination_id, choice_id,
                        destination_name, picked["similarity"], selected_model,
                    )
                else:
                    global_id = find_identity_for_track(
                        registry_path, source_camera, selected_id,
                    )
                    candidate_identity = find_identity_for_track(
                        registry_path, destination_id, choice_id,
                    )
                    if global_id and candidate_identity == global_id:
                        reject_track_association(
                            registry_path, global_id, destination_id, choice_id,
                        )
                save_hypothesis(
                    hypotheses_path, selected_id, choice_id, picked["similarity"],
                    selected_model, decision, source_camera, destination_id,
                    source_name, destination_name,
                    global_person_id=global_id if decision == "SAME PERSON" else None,
                )
                if global_id:
                    video_dir = run_dir / "global_person_videos" / global_id
                    if video_dir.is_dir():
                        for old_video in video_dir.glob("*.mp4"):
                            old_video.unlink(missing_ok=True)
                        (video_dir / "video_manifest.json").unlink(missing_ok=True)
                        (video_dir / "global_person_assignments.csv").unlink(missing_ok=True)
                        (video_dir / "manifest_signature.json").unlink(missing_ok=True)
                    st.session_state.pop(
                        f"global_video_outputs_{run_dir.name}_{global_id}", None,
                    )
                st.success(f"Saved review: {decision}" + (f" · {global_id}" if decision == "SAME PERSON" else ""))
                st.rerun()
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Could not update the identity association: {exc}")
    st.caption("Similarity ranks visual candidates; only your manual decision records a confirmed or rejected pair.")
    st.markdown("#### Selected-person video results")
    st.caption("Confirmed matches receive the same persistent global ID. Other cameras still get a playable video with a clear no-match label and no bounding box.")
    if st.button(
        "▶  Track This Person Across All Cameras",
        type="primary", key=f"track_global_person_{run_dir.name}_{source_camera}_{selected_id}",
    ):
        try:
            video_status = st.empty()
            video_progress = st.progress(0.0)

            def update_video_progress(camera_label: str, camera_index: int,
                                      camera_total: int, current: int,
                                      total: int) -> None:
                total_label = f"{total:,}" if total else "unknown"
                video_status.caption(
                    f"Writing {camera_label} selected-person video · "
                    f"camera {camera_index} of {camera_total} · frame {current:,} of {total_label}"
                )
                fraction = min(1.0, current / total) if total > 0 else 0.0
                video_progress.progress(
                    min(1.0, ((camera_index - 1) + fraction) / max(1, camera_total))
                )

            with st.spinner("Writing and checking one selected-person video for every processed camera…"):
                global_id, output_rows = render_all_camera_tracking_videos(
                    results, run_dir, source_camera, selected_id, selected_model,
                    progress_callback=update_video_progress,
                )
            output_key = f"global_video_outputs_{run_dir.name}_{global_id}"
            st.session_state[output_key] = output_rows
            st.success(f"Created playable camera videos for {global_id}.")
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            st.error(f"Selected-person videos could not be created: {exc}")
    registry_path = run_dir / "global_identity_registry.json"
    selected_global_id = find_identity_for_track(registry_path, source_camera, selected_id)
    saved_video_rows = (
        st.session_state.get(f"global_video_outputs_{run_dir.name}_{selected_global_id}")
        if selected_global_id else None
    )
    if not saved_video_rows and selected_global_id:
        manifest_path = run_dir / "global_person_videos" / selected_global_id / "video_manifest.json"
        if manifest_path.is_file():
            try:
                manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
                if all(Path(row["video_path"]).is_file() for row in manifest_rows):
                    saved_video_rows = manifest_rows
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                saved_video_rows = None
    if saved_video_rows:
        st.markdown(f"### {selected_global_id} — Selected Person Tracking Results")
        for row in saved_video_rows:
            with st.expander(
                f"{row['camera_name']} · {row['match_status']}" +
                (f" · local ID {row['local_track_id']}" if row["local_track_id"] else " · no confirmed local track"),
                expanded=True,
            ):
                st.caption(
                    f"Original: {row['original_filename']} · "
                    f"{row['width']}×{row['height']} · {row['fps']:.2f} FPS · "
                    f"{row['duration_seconds']:.1f}s"
                )
                st.video(row["video_path"])
                video_path = Path(row["video_path"])
                st.download_button(
                    f"Download {video_path.name}", video_path.read_bytes(),
                    file_name=video_path.name, mime="video/mp4",
                    key=f"download_{selected_global_id}_{row['camera_id']}",
                )
        assignment_csv = run_dir / "global_person_videos" / selected_global_id / "global_person_assignments.csv"
        if assignment_csv.is_file():
            st.download_button(
                "Download this identity's camera and video report (.csv)",
                assignment_csv.read_bytes(), file_name=f"{selected_global_id}_camera_results.csv",
                mime="text/csv", key=f"download_{selected_global_id}_report",
            )


def show_evaluation(results: list[dict], fingerprint_sets: dict[str, dict[str, object]],
                    run_dir: Path) -> None:
    st.markdown('<div class="section-rule"></div><div class="eyebrow">MANUAL EVALUATION</div>', unsafe_allow_html=True)
    st.subheader("Check the matches against people you labelled")
    st.caption("Label same-person pairs using the source camera/ID and destination camera/ID. Mark a person who is absent from Camera 2 with candidate_track_id=NOT_IN_CAMERA2. For meaningful mAP and false-match counts, include every true Camera 2 track for each query.")
    template = b"query_camera_id,query_track_id,candidate_camera_id,candidate_track_id\n"
    st.download_button("Download ground-truth CSV template", template,
                       file_name="review3_ground_truth_template.csv", mime="text/csv")
    uploaded_labels = st.file_uploader("Upload your labelled pairs", type=["csv"], key="ground_truth_pairs")
    saved_hypotheses = load_hypotheses(run_dir / "cross_camera_hypotheses.json")
    confirmed_pairs = [{
        "query_camera_id": row.get("camera1_id", "camera1"),
        "query_track_id": row["camera1_track_id"],
        "candidate_camera_id": row.get("camera2_id", "camera2"),
        "candidate_track_id": row["camera2_track_id"],
    } for row in saved_hypotheses.values() if row.get("status") == "SAME PERSON"]
    if confirmed_pairs:
        st.download_button(
            "Download manually confirmed SAME PERSON pairs",
            "query_camera_id,query_track_id,candidate_camera_id,candidate_track_id\n" + "".join(
                f"{row['query_camera_id']},{row['query_track_id']},{row['candidate_camera_id']},{row['candidate_track_id']}\n"
                for row in confirmed_pairs
            ),
            file_name="manually_confirmed_pairs.csv", mime="text/csv",
        )
    if uploaded_labels is None and not confirmed_pairs:
        st.info("NOT EVALUATED · Add manually verified pairs in the matching panel or upload a labelled CSV.")
        return
    try:
        if uploaded_labels is not None:
            reader = csv.DictReader(io.StringIO(uploaded_labels.getvalue().decode("utf-8-sig")))
            required = {"query_camera_id", "query_track_id", "candidate_camera_id", "candidate_track_id"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"The CSV needs columns {', '.join(sorted(required))}. Uploaded columns: {reader.fieldnames or 'none'}")
            raw_pairs = list(reader)
        else:
            raw_pairs = confirmed_pairs
        tracks_by_camera = {item["camera_id"]: {int(track["track_id"]) for track in all_tracks(results, item["camera_id"])} for item in results}
        pairs = validate_ground_truth_pairs(raw_pairs, tracks_by_camera)
        threshold = st.slider(
            "Evaluation no-match threshold", -1.0, 1.0, 0.50, 0.05,
            key=f"evaluation_threshold_{run_dir.name}",
            help="Exploratory cutoff only. It is not calibrated until the labelled set is representative.",
        )
        model_metrics = {name: evaluate_pairs(pairs, vectors, threshold=threshold)
                         for name, vectors in fingerprint_sets.items()}
        if not any(values["evaluated"] for values in model_metrics.values()):
            st.warning("NOT EVALUATED · The labelled tracks do not have usable fingerprints in both cameras.")
            return
        st.dataframe([
            {"Appearance model": name,
             "Evaluated identities": values["evaluated"],
             "Missing fingerprints": values["missing_fingerprints"],
             "Unmatchable identities": values["missing_fingerprints"],
             "Top-1 incorrect": values["evaluated"] - values["top1_correct"],
             "Top-1 accuracy": f"{values['top1_accuracy']:.1%}" if values["top1_accuracy"] is not None else "—",
             "Top-3 accuracy": f"{values['top3_accuracy']:.1%}" if values["top3_accuracy"] is not None else "—",
             "Top-5 accuracy": f"{values['top5_accuracy']:.1%}" if values["top5_accuracy"] is not None else "—",
             "mAP (label completeness required)": f"{values['mean_average_precision']:.3f}" if values["mean_average_precision"] is not None else "—",
             "Match precision": f"{values['precision']:.1%}" if values["precision"] is not None else "—",
             "Match recall": f"{values['recall']:.1%}" if values["recall"] is not None else "—",
             "No-match correct / total": f"{values['no_match_correct']} / {values['no_match_queries']}",
             "False matches": values["false_matches"],
             "Missed matches": values["missed_matches"],
             "Mean reciprocal rank": f"{values['mrr']:.3f}" if values["mrr"] is not None else "—"}
            for name, values in model_metrics.items()
        ], hide_index=True, use_container_width=True)
        st.caption("Both models use the same manually verified pairs. Metrics apply only to these tracks and this run.")
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))


with st.sidebar:
    st.markdown("## ◉  PERSON RE-ID")
    st.caption("VISION ANALYTICS  /  REVIEW 3")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("**WORKFLOW**")
    st.caption("1 · Detect and track\n\n2 · Keep clear crops\n\n3 · Build appearance fingerprints\n\n4 · Search all other cameras")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("**PROCESSING**")
    if st.session_state.get("is_processing"):
        st.warning("Processing videos")
    elif st.session_state.get("results"):
        st.success("Last run complete")
    else:
        st.markdown('<span class="status-pill">● READY · CPU INFERENCE</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.caption("Track IDs are local to one video. Appearance scores are candidate rankings, not identity proof.")

st.markdown(
    '<div class="hero"><div class="eyebrow">REVIEW 3  /  MULTI-CAMERA PERSON TRACKING</div>'
    '<h1>Person Analytics Workspace</h1><p>Detect and track people in multiple recorded camera videos, '
    'retain clear crops, and compare track appearance.</p></div>', unsafe_allow_html=True,
)

if "camera_inputs" not in st.session_state:
    st.session_state.camera_inputs = [{"camera_id": "camera1", "name": "Camera 1"},
                                      {"camera_id": "camera2", "name": "Camera 2"}]
camera_inputs = st.session_state.camera_inputs
manager_cols = st.columns([1, 5])
with manager_cols[0]:
    if st.button("＋ Add camera", disabled=len(camera_inputs) >= 12, key="add_camera"):
        existing_numbers = [int(item["camera_id"].replace("camera", "")) for item in camera_inputs]
        next_number = max(existing_numbers, default=0) + 1
        camera_inputs.append({"camera_id": f"camera{next_number}", "name": f"Camera {next_number}"})
        st.rerun()
with manager_cols[1]:
    st.caption("Add up to 12 video sources. Each camera gets its own independent local track IDs.")
uploaded_videos: list[tuple[str, str, object]] = []
upload_columns = st.columns(min(3, len(camera_inputs)), gap="large")
for index, camera in enumerate(list(camera_inputs)):
    camera_id = camera["camera_id"]
    column = upload_columns[index % len(upload_columns)]
    with column:
        display_name = st.text_input("Camera name", camera.get("name", camera_id),
                                     key=f"camera_name_{camera_id}", max_chars=40)
        camera["name"] = display_name.strip() or camera_id
        st.markdown(f'<div class="panel"><div class="camera-title">{camera["name"]}</div>'
                    '<div class="muted">Recorded video input</div></div>', unsafe_allow_html=True)
        video = st.file_uploader(
            f"Choose {camera["name"]} video", type=["mp4", "avi", "mov", "mkv", "m4v"],
            key=f"upload_{camera_id}",
        )
        if video is not None:
            st.caption(f"Selected · {video.name} · {video.size / (1024 * 1024):.1f} MB")
            with st.expander(f"Preview {camera["name"]} upload"):
                st.video(video.getvalue())
            uploaded_videos.append((camera["name"], camera_id, video))
        if len(camera_inputs) > 1 and st.button(f"Remove {camera["name"]}", key=f"remove_{camera_id}"):
            st.session_state.camera_inputs = [item for item in camera_inputs if item["camera_id"] != camera_id]
            st.rerun()

st.markdown('<div class="section-rule"></div><div class="eyebrow">TRACKING SETTINGS</div>', unsafe_allow_html=True)
settings_a, settings_b, settings_c, settings_d, settings_e = st.columns([1, 1, 1, 1, 1])
with settings_a:
    frame_interval = st.number_input(
        "Run YOLO every N frames", min_value=1, max_value=30, value=10, step=1,
        help="The tracker predicts boxes between YOLO checks. Lower values are more accurate but slower on CPU.",
    )
with settings_b:
    confidence = st.slider("Detection confidence", 0.10, 0.95, 0.40, 0.05)
with settings_c:
    image_size = st.selectbox("YOLO image size", [416, 640, 960], index=1,
                              help="Smaller sizes run faster on CPU; large sizes may help small distant people.")
with settings_d:
    tracker_max_age = st.number_input(
        "Keep lost tracks (frames)", min_value=int(frame_interval), max_value=300,
        value=max(30, int(frame_interval)), step=5,
        help="Must be at least the sampling interval. After this many missed frames, the track is ended.",
    )
with settings_e:
    nms_iou_threshold = st.slider(
        "NMS overlap threshold", 0.10, 0.95, 0.70, 0.05,
        help="YOLO suppresses duplicate boxes above this overlap level.",
    )
with st.expander("Advanced tracker and crop settings"):
    crop_a, crop_b, crop_c, crop_d, crop_e = st.columns(5)
    min_crop_width = crop_a.number_input("Minimum crop width (pixels)", 20, 300, 40, 5)
    min_crop_height = crop_b.number_input("Minimum crop height (pixels)", 40, 500, 80, 5)
    blur_threshold = crop_c.slider("Minimum sharpness", 0.0, 150.0, 50.0, 5.0)
    tracker_max_center_distance = crop_d.slider(
        "Tracker motion association range", 0.5, 5.0, 2.5, 0.25,
        help="Allows a detection this many average person-heights from a predicted box. Wider is more tolerant but may swap IDs.",
    )
    tracker_iou_threshold = crop_e.slider(
        "Minimum box overlap", 0.0, 0.5, 0.05, 0.05,
        help="Minimum overlap gate for SORT association. The alternative motion-distance gate also helps recover moving boxes.",
    )
    st.caption("SORT-style tracking uses Kalman motion prediction, box overlap and Hungarian matching. IDs stay local to each camera. Up to eight clear crops per track are kept, spread across time.")

if st.button("▶  PROCESS CAMERA VIDEOS", type="primary", disabled=not uploaded_videos):
    st.session_state.is_processing = True
    results = []
    run_dir = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        detector_path = ROOT / "yolo26n.pt"
        if not detector_path.is_file():
            model = load_detector(detector_path)
        detector_stat = detector_path.stat()
        with st.spinner("Loading the pretrained YOLO26n model on CPU…"):
            model = get_model(str(detector_path), detector_stat.st_mtime_ns, detector_stat.st_size)
        for camera_name, camera_id, uploaded_file in uploaded_videos:
            camera_dir = run_dir / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            safe_upload_name = Path(uploaded_file.name).name
            input_path = camera_dir / safe_upload_name
            input_path.write_bytes(uploaded_file.getvalue())
            st.markdown(f"**{camera_name} — processing**")
            status = st.empty()
            progress = st.progress(0.0)

            def update_progress(current: int, total: int, stage: str) -> None:
                label = str(total) if total > 0 else "unknown"
                status.caption(f"{stage} · frame {current:,} of {label}")
                if total > 0:
                    progress.progress(min(current / total, 1.0))

            try:
                result = process_video(
                    camera_id, input_path, model, camera_dir,
                    int(frame_interval), float(confidence), update_progress,
                    image_size=int(image_size), tracker_max_age=int(tracker_max_age),
                    tracker_max_center_distance=float(tracker_max_center_distance),
                    tracker_iou_threshold=float(tracker_iou_threshold),
                    min_crop_width=int(min_crop_width), min_crop_height=int(min_crop_height),
                    blur_threshold=float(blur_threshold),
                    nms_iou_threshold=float(nms_iou_threshold),
                )
                result_dict = result.__dict__.copy()
                result_dict["source_name"] = uploaded_file.name
                result_dict["source_path"] = str(input_path)
                result_dict["camera_name"] = camera_name
                results.append(result_dict)
                status.success(
                    f"{camera_name} complete · {result.track_count} tracks · "
                    f"{result.crops_saved} clear crops · {result.processing_seconds:.1f}s"
                )
            except Exception as exc:
                logging.exception("Processing failed for %s", uploaded_file.name)
                status.error(f"{camera_name} could not be processed: {exc}")

        crop_rows = [row for result in results for row in load_crop_rows(result)]
        fingerprint_sets: dict[str, dict] = {}
        yolo_fingerprint_path = run_dir / "fingerprints_yolo.json"
        try:
            with st.spinner("Building YOLO baseline appearance fingerprints…"):
                yolo_variants, yolo_crop_embeddings = make_track_fingerprint_variants(
                    model, crop_rows, feature_kind="yolo_baseline",
                ) if crop_rows else ({method: {} for method in AGGREGATION_METHODS}, {})
            for method, vectors in yolo_variants.items():
                output_path = yolo_fingerprint_path if method == "mean" else run_dir / f"fingerprints_yolo_{method}.json"
                save_fingerprints(output_path, vectors, {
                    "model_id": "yolo26n_detector_features",
                    "model_label": "YOLO26n detector-feature baseline",
                    "feature_dimension": next(iter(vectors.values())).size if vectors else None,
                    "is_person_reid_model": False,
                    "aggregation": method,
                    "preprocessing": "existing YOLO26n .embed(), CPU, imgsz=256; BGR crops",
                })
                fingerprint_sets[f"YOLO baseline · {method.replace('_', ' ')}"] = vectors
            save_crop_embeddings(run_dir / "crop_embeddings_yolo.json", yolo_crop_embeddings, {
                "model_id": "yolo26n_detector_features", "feature_kind": "detector features",
            })
        except Exception as exc:
            logging.exception("YOLO baseline fingerprint extraction failed")
            st.warning(f"Detection and tracking are saved, but the YOLO baseline could not be generated: {exc}")
        osnet_path = ROOT / "models" / CHECKPOINT_NAME
        osnet_fingerprint_path = run_dir / "fingerprints_osnet.json"
        try:
            with st.spinner("Loading pretrained OSNet and building Re-ID fingerprints…"):
                if not osnet_path.is_file():
                    raise FileNotFoundError(
                        f"The pretrained OSNet checkpoint is missing: {osnet_path}. "
                        "Run tools/download_osnet_weights.py to install it."
                    )
                checkpoint_stat = osnet_path.stat()
                osnet = get_osnet_extractor(
                    str(osnet_path), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size,
                )
                osnet_variants, osnet_crop_embeddings = make_track_fingerprint_variants(
                    osnet, crop_rows, feature_kind="osnet",
                ) if crop_rows else ({method: {} for method in AGGREGATION_METHODS}, {})
            for method, vectors in osnet_variants.items():
                output_path = osnet_fingerprint_path if method == "mean" else run_dir / f"fingerprints_osnet_{method}.json"
                save_fingerprints(output_path, vectors, {
                    "model_id": f"{OSNetExtractor.model_label}",
                    "model_label": OSNetExtractor.model_label,
                    "feature_dimension": 512,
                    "checkpoint": CHECKPOINT_NAME,
                    "aggregation": method,
                    "is_person_reid_model": True,
                    "preprocessing": "Torchreid resize 256x128, RGB, ImageNet mean/std, CPU, L2 normalized",
                    "inference_seconds": round(osnet.inference_seconds, 4),
                    "embedding_count": osnet.embedding_count,
                    "average_inference_ms_per_crop": osnet.average_inference_ms,
                })
                label = f"OSNet x0.25 (MSMT17) · {method.replace('_', ' ')}"
                fingerprint_sets[label] = vectors
            save_crop_embeddings(run_dir / "crop_embeddings_osnet.json", osnet_crop_embeddings, {
                "model_id": OSNetExtractor.model_label, "checkpoint": CHECKPOINT_NAME,
                "feature_dimension": 512,
            })
        except Exception as exc:
            logging.info("Dedicated OSNet model unavailable: %s", exc)
            st.warning(
                f"The YOLO baseline is ready. Dedicated OSNet was not loaded: {exc} "
                "Run the setup steps in README.md (install dependencies, fetch Torchreid source, then download the checkpoint) to enable OSNet."
            )
        if len(results) >= 2:
            query_camera, gallery_camera = results[0]["camera_id"], results[1]["camera_id"]
            preferred_vectors = fingerprint_sets.get(
                "OSNet x0.25 (MSMT17) · mean",
                fingerprint_sets.get("YOLO baseline · mean", {}),
            )
            match_rows = match_two_cameras(
                preferred_vectors, crop_rows, query_camera, gallery_camera, top_k=3,
            )
            write_matches_csv(run_dir / "matches.csv", match_rows)
            (run_dir / "run_summary.json").write_text(json.dumps({
                "camera_count": len(results), "query_camera": query_camera,
                "gallery_camera": gallery_camera, "match_rows": len(match_rows),
                "feature_model": "OSNet x0.25 (MSMT17)" if
                "OSNet x0.25 (MSMT17) · mean" in fingerprint_sets
                else "YOLO detector-feature baseline",
                "aggregation": "mean",
                "processing_seconds": round(sum(item["processing_seconds"] for item in results), 3),
                "accuracy_evaluated": False,
                "accuracy_note": "No complete ground-truth labels were supplied for this run.",
            }, indent=2), encoding="utf-8")
        for result in results:
            result["fingerprints_yolo_file"] = str(yolo_fingerprint_path) if yolo_fingerprint_path.exists() else None
            result["fingerprints_osnet_file"] = str(osnet_fingerprint_path) if osnet_fingerprint_path.exists() else None
        (run_dir / "camera_manifest.json").write_text(
            json.dumps([{key: result.get(key) for key in ("camera_id", "camera_name", "source_name", "source_path")}
                        for result in results], indent=2), encoding="utf-8",
        )
        st.session_state.results = results
        st.session_state.run_dir = str(run_dir)
        st.session_state.fingerprint_models = list(fingerprint_sets)
        st.session_state.active_selection = None
        st.session_state.paused_selection = None
        st.session_state.is_processing = False
        if results:
            st.success("Run complete. Review tracks and search each camera against the others below.")
        else:
            st.error("No camera video completed. Check the messages above and try again.")
    except Exception:
        logging.exception("Could not start the run")
        st.session_state.is_processing = False
        st.error("The run could not start. Check folder permissions and free disk space.")

results = st.session_state.get("results", [])
run_dir_value = st.session_state.get("run_dir")
if results and run_dir_value:
    run_dir = Path(run_dir_value)
    fingerprint_sets = {}
    fingerprint_files = [(f"YOLO baseline · {method.replace('_', ' ')}",
                          "yolo26n_detector_features",
                          "fingerprints_yolo.json" if method == "mean" else f"fingerprints_yolo_{method}.json")
                         for method in AGGREGATION_METHODS]
    fingerprint_files += [(f"OSNet x0.25 (MSMT17) · {method.replace('_', ' ')}",
                           OSNetExtractor.model_label,
                           "fingerprints_osnet.json" if method == "mean" else f"fingerprints_osnet_{method}.json")
                          for method in AGGREGATION_METHODS]
    for label, expected_model, filename in fingerprint_files:
        path = run_dir / filename
        if path.is_file():
            try:
                fingerprint_sets[label] = load_fingerprints(
                    path, expected_model=expected_model,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                st.warning(f"Could not read the saved {label} fingerprints: {exc}")
    track_summaries = [read_json(item["track_summary_file"]).get("tracks", []) for item in results]
    all_track_rows = [track for group in track_summaries for track in group]
    active_tracks_count = sum(track["status"] == "active at end" for track in all_track_rows)
    lost_tracks_count = sum(track["status"] == "temporarily lost" for track in all_track_rows)
    completed_tracks_count = sum(track["status"] == "ended" for track in all_track_rows)
    osnet_count = len(fingerprint_sets.get(
        "OSNet x0.25 (MSMT17) · mean", {},
    ))
    candidate_count = max(0, osnet_count - len(results))
    selection_key = st.session_state.get("active_selection")
    selected_id = None
    if selection_key:
        try:
            selected_id = int(selection_key.rsplit(":", 1)[1])
        except ValueError:
            selected_id = None
    paused_selection = st.session_state.get("paused_selection")
    hypotheses = load_hypotheses(run_dir / "cross_camera_hypotheses.json")
    confirmed_count = sum(row.get("status") == "SAME PERSON" for row in hypotheses.values())
    global_identities_path = run_dir / "global_identity_registry.json"
    registry = load_registry(global_identities_path)
    derived_groups = build_global_identities(hypotheses)
    global_identities = []
    for record in registry.values():
        members = {(item["camera_id"], int(item["track_id"]))
                   for item in record.get("members", [])}
        conflict = any(
            group["review_conflict"] and
            {(item["camera_id"], int(item["track_id"])) for item in group["members"]} == members
            for group in derived_groups
        )
        global_identities.append({**record, "review_conflict": conflict})

    overview_tab, camera_tab, tracks_tab, matching_tab, evaluation_tab, downloads_tab = st.tabs([
        "Overview", "Camera workspace", "Track management", "Person Search", "Evaluation", "Downloads",
    ])
    with overview_tab:
        st.markdown('<div class="eyebrow">RUN SUMMARY · MEASURED VALUES</div>', unsafe_allow_html=True)
        metric_rows = [
            [("Videos processed", len(results)),
             ("Person detections", sum(item["total_detections"] for item in results)),
             ("Camera-local tracks", len(all_track_rows))],
            [("ACTIVE tracks", active_tracks_count),
             ("LOST tracks", lost_tracks_count),
             ("COMPLETED tracks", completed_tracks_count)],
            [("Clear crops retained", sum(item["crops_saved"] for item in results)),
             ("OSNet fingerprints · 512D", osnet_count),
             ("Other-camera candidate tracks", candidate_count)],
            [("Review-derived global identities", len(global_identities)),
             ("Confirmed same-person links", confirmed_count),
             ("Cameras processed", len(results))],
        ]
        for row in metric_rows:
            columns = st.columns(3)
            for column, (label, value) in zip(columns, row):
                with column:
                    metric_card(label, value)
        st.caption("OSNet candidate counts are saved fingerprints for the separately trained Re-ID model. Similarity scores remain unverified until a person reviews the pair.")
        if not confirmed_count:
            st.info("Identity evaluation: NOT EVALUATED · Manually confirm identity pairs first.")
        with st.expander("Model and inference details"):
            yolo_metadata = read_json(run_dir / "fingerprints_yolo.json")
            osnet_metadata = read_json(run_dir / "fingerprints_osnet.json")
            st.write({
                "Detection model": "YOLO26n · CPU",
                "YOLO baseline fingerprint dimension": yolo_metadata.get("feature_dimension"),
                "OSNet model": osnet_metadata.get("model_label", "Not available for this run"),
                "OSNet embedding dimension": osnet_metadata.get("feature_dimension"),
                "OSNet crops embedded": osnet_metadata.get("embedding_count", 0),
                "OSNet inference time (s)": osnet_metadata.get("inference_seconds"),
                "OSNet average ms per crop": osnet_metadata.get("average_inference_ms_per_crop"),
            })
        if hypotheses:
            with st.expander("Saved cross-camera review decisions"):
                st.dataframe([{
                    "Hypothesis": row.get("hypothesis_id", "—"),
                    "First camera": f"{row.get('camera1_name', row.get('camera1_id', 'Camera 1'))} ID {row['camera1_track_id']}",
                    "Second camera": f"{row.get('camera2_name', row.get('camera2_id', 'Camera 2'))} ID {row['camera2_track_id']}",
                    "Similarity": round(row["similarity"], 3),
                    "Model": row["model_label"],
                    "Review": row["status"],
                } for row in hypotheses.values()], hide_index=True, use_container_width=True)
        st.markdown("#### Confirmed global identities")
        if global_identities:
            st.dataframe([{
                "Global ID": group["global_person_id"],
                "Camera-local members": " · ".join(
                    f"{member['camera_name']} ID {member['track_id']}" for member in group["members"]
                ),
                "Review conflict": "Check decisions" if group["review_conflict"] else "None",
            } for group in global_identities], hide_index=True, use_container_width=True)
        else:
            st.caption("No global groups yet. Mark pairs as SAME PERSON in Appearance matching after reviewing the video crops.")
        track_table = [{
            "Camera": track["camera_id"], "Track ID": track["track_id"],
            "Status": track.get("lifecycle_status", track["status"]),
            "First appearance (s)": track["first_time_seconds"],
            "Last appearance (s)": track["last_time_seconds"],
            "Duration (s)": track.get("duration_seconds", 0),
            "Observed frames": track["observed_frames"],
            "Valid crops": track.get("crop_count", 0),
            "Max detection confidence": track["max_confidence"],
        } for track in all_track_rows]
        st.dataframe(track_table, hide_index=True, use_container_width=True)

    with camera_tab:
        show_camera_panels(results, run_dir, selection_key, selected_id, paused_selection)

    with tracks_tab:
        selected_id, inspected_camera_id, _ = show_selected_track(results, run_dir)
        st.markdown("#### All camera tracks")
        track_table = [{
            "Camera": track["camera_id"], "Track ID": track["track_id"],
            "Status": track.get("lifecycle_status", track["status"]),
            "First seen (s)": track["first_time_seconds"],
            "Last seen (s)": track["last_time_seconds"],
            "Duration (s)": track.get("duration_seconds", 0),
            "Observed frames": track["observed_frames"],
            "Crops": track.get("crop_count", 0),
            "Max confidence": track["max_confidence"],
        } for track in all_track_rows]
        st.dataframe(track_table, hide_index=True, use_container_width=True)

    with matching_tab:
        preferred_track_id = selected_id
        show_cross_camera_matching(results, run_dir, fingerprint_sets, preferred_track_id)

    with evaluation_tab:
        show_evaluation(results, fingerprint_sets, run_dir)

    with downloads_tab:
        st.markdown('<div class="eyebrow">DOWNLOAD COMPLETE RUN</div>', unsafe_allow_html=True)
        try:
            st.download_button(
                "↓  Download all run results (.zip)", make_zip(run_dir),
                file_name="review3_person_tracking_results.zip", mime="application/zip",
                key="save_results_zip",
            )
        except OSError as exc:
            st.error(f"Could not create the results ZIP: {exc}")
        if global_identities_path.is_file():
            st.download_button(
                "Download persistent global identity registry (.json)",
                global_identities_path.read_bytes(), file_name="global_identities.json",
                mime="application/json", key="download_global_identities",
            )
        for result in results:
            camera_name = result["camera_id"].replace("camera", "Camera ")
            st.markdown(f"**{camera_name} · {result['source_name']}**")
            st.caption(
                f"{result['frames_processed']:,} YOLO checks · {result['total_detections']:,} detections · "
                f"{result['processing_seconds']:.1f}s"
            )
            file_cols = st.columns(4)
            for column, path_key, label in zip(
                file_cols,
                ["processed_video", "tracks_file", "crops_manifest_file", "rejected_crops_file"],
                ["Processed video", "Frame-level track CSV", "Retained crop CSV", "Rejected crop CSV"],
            ):
                path = Path(result[path_key])
                with column:
                    if path.is_file():
                        column.download_button(
                            f"Download {label}", path.read_bytes(),
                            file_name=path.name, key=f"download_{result['camera_id']}_{path.name}",
                        )
else:
    st.markdown('<div class="section-rule"></div><div class="eyebrow">RUN SUMMARY</div>', unsafe_allow_html=True)
    st.info("Upload one or both recorded camera videos, then process them to see real detections, tracks and crops.")
