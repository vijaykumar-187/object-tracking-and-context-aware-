"""Shared video processing for detection, local tracking, crops and exports."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.detection.detect import (
    BLUR_THRESHOLD,
    MIN_HEIGHT,
    MIN_WIDTH,
    detect_people_in_frame,
    is_blurry,
)
from src.tracking.sort_tracker import SortTracker, TrackObservation

logger = logging.getLogger(__name__)
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ProgressCallback = Callable[[int, int, str], None]
OUTPUT_MAX_WIDTH = 1280
MAX_CROPS_PER_TRACK = 8
CROP_TIME_BUCKET_SECONDS = 0.75
TRACK_MAX_AGE = 30


@dataclass
class ProcessingResult:
    camera_id: str
    source_name: str
    processed_video: str
    crops_dir: str
    metadata_file: str
    frames_read: int
    frames_processed: int
    total_detections: int
    crops_saved: int
    fps: float
    total_frames: int
    processing_seconds: float
    tracks_file: str
    crops_manifest_file: str
    track_summary_file: str
    track_count: int
    rejected_crops_file: str
    rejected_crops_count: int


def validate_video_path(video_path: Path | str) -> Path:
    """Check path and extension; OpenCV validates the actual stream later."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise ValueError("The selected video file could not be found.")
    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
        raise ValueError(f"Unsupported video format. Choose one of: {supported}.")
    return path


def _safe_crop(frame: np.ndarray, box: object) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Clip coordinates and reject malformed, out-of-frame or empty crops."""
    height, width = frame.shape[:2]
    try:
        x1, y1, x2, y2 = (int(v) for v in box)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return ((x1, y1, x2, y2), crop) if crop.size else None


def _draw_track(frame: np.ndarray, track: TrackObservation, camera_id: str,
                frame_number: int) -> None:
    x1, y1, x2, y2 = track.box
    color = (42, 205, 155) if track.observed else (0, 170, 230)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if track.observed else 1)
    camera_label = camera_id.replace("camera", "Camera ").replace("cam", "Camera ")
    state = f"recovered · det {track.confidence:.2f}" if track.recovered else (
        f"det {track.confidence:.2f}" if track.observed
        else f"predicted · last det {track.confidence:.2f}"
    )
    label = f"{camera_label} | Frame {frame_number + 1} | ID {track.track_id} | {state}"
    cv2.putText(frame, label, (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _crop_quality(confidence: float, sharpness: float) -> float:
    confidence_part = max(0.0, min(1.0, confidence))
    sharpness_part = max(0.0, min(1.0, sharpness / (BLUR_THRESHOLD * 4.0)))
    return round(0.55 * confidence_part + 0.45 * sharpness_part, 5)


def _near_duplicate(crop: np.ndarray, existing_path: str | Path,
                    mean_difference_threshold: float = 1.25) -> bool:
    """Compare tiny grayscale thumbnails to detect near-identical samples cheaply."""
    previous = cv2.imread(str(existing_path))
    if previous is None or previous.size == 0 or crop is None or crop.size == 0:
        return False
    def signature(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (16, 32), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(signature(crop), signature(previous))
    return float(difference.mean()) <= mean_difference_threshold


def _retain_representative_crops(candidates: dict[int, dict]) -> None:
    """Keep the best crops while rewarding time separation across a track."""
    if len(candidates) <= MAX_CROPS_PER_TRACK:
        return
    items = list(candidates.items())
    timestamps = [float(record["timestamp_seconds"]) for _, record in items]
    time_span = max(CROP_TIME_BUCKET_SECONDS, max(timestamps) - min(timestamps))
    selected: list[tuple[int, dict]] = []
    first = max(items, key=lambda pair: pair[1]["quality_score"])
    selected.append(first)
    remaining = [item for item in items if item[0] != first[0]]
    while remaining and len(selected) < MAX_CROPS_PER_TRACK:
        def value(item: tuple[int, dict]) -> float:
            time = float(item[1]["timestamp_seconds"])
            nearest = min(abs(time - float(chosen[1]["timestamp_seconds"]))
                          for chosen in selected)
            diversity = min(1.0, nearest / time_span)
            return float(item[1]["quality_score"]) + 0.20 * diversity

        chosen = max(remaining, key=value)
        selected.append(chosen)
        remaining.remove(chosen)
    keep = {key for key, _ in selected}
    for bucket, record in items:
        if bucket not in keep:
            candidates.pop(bucket, None)
            Path(record["path"]).unlink(missing_ok=True)


def _store_crop_candidate(
    crop: np.ndarray,
    camera_id: str,
    frame_number: int,
    timestamp: float,
    track_id: int,
    confidence: float,
    sharpness: float,
    box: tuple[int, int, int, int],
    output_dir: Path,
    candidates: dict[int, dict[int, dict]],
) -> None:
    """Keep at most eight high-quality, time-spread crops for each track."""
    score = _crop_quality(confidence, sharpness)
    bucket = int(timestamp / CROP_TIME_BUCKET_SECONDS)
    selected = candidates.setdefault(track_id, {})
    previous = selected.get(bucket)
    if previous and previous["quality_score"] >= score:
        return

    filename = f"{camera_id}_track{track_id:04d}_frame{frame_number:06d}.jpg"
    path = output_dir / filename
    if not cv2.imwrite(str(path), crop):
        raise OSError(f"Could not save crop image {filename}.")
    record = {
        "camera_id": camera_id,
        "track_id": track_id,
        "frame_number": frame_number,
        "timestamp_seconds": round(timestamp, 3),
        "crop_filename": filename,
        "path": str(path),
        "confidence": round(confidence, 4),
        "sharpness": round(sharpness, 2),
        "quality_score": score,
        "x1": int(box[0]),
        "y1": int(box[1]),
        "x2": int(box[2]),
        "y2": int(box[3]),
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
    }
    selected[bucket] = record
    if previous:
        Path(previous["path"]).unlink(missing_ok=True)

    _retain_representative_crops(selected)


def process_video(
    camera_id: str,
    video_path: Path | str,
    model: object,
    output_dir: Path | str,
    frame_interval: int = 5,
    confidence_threshold: float = 0.40,
    progress_callback: Optional[ProgressCallback] = None,
    image_size: int = 640,
    tracker_max_age: int = TRACK_MAX_AGE,
    tracker_max_center_distance: float = 2.5,
    tracker_iou_threshold: float = 0.05,
    min_crop_width: int = MIN_WIDTH,
    min_crop_height: int = MIN_HEIGHT,
    blur_threshold: float = BLUR_THRESHOLD,
    nms_iou_threshold: float = 0.70,
) -> ProcessingResult:
    """Process a camera video with YOLO and a SORT-style per-video tracker.

    YOLO is evaluated at the requested interval. A Kalman filter predicts boxes
    between detector checks and Hungarian assignment associates new detections
    with existing tracks. IDs remain camera-local. Good crops are time-spread,
    quality-ranked, and capped at eight per track.
    """
    source = validate_video_path(video_path)
    if frame_interval < 1:
        raise ValueError("Frame sampling interval must be at least 1.")
    if image_size < 320 or image_size > 1280:
        raise ValueError("YOLO image size must be between 320 and 1280 pixels.")
    if tracker_max_age < frame_interval:
        raise ValueError("Lost-track retention must be at least the YOLO sampling interval.")
    if tracker_max_center_distance <= 0:
        raise ValueError("Tracker association distance must be greater than zero.")
    if not 0.0 <= tracker_iou_threshold <= 1.0:
        raise ValueError("Tracker IoU threshold must be between 0 and 1.")
    if not 0.0 < nms_iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be greater than 0 and at most 1.")
    if min_crop_width < 1 or min_crop_height < 1 or blur_threshold < 0:
        raise ValueError("Crop dimensions and blur threshold must be non-negative.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 50 * 1024 * 1024:
        raise OSError("There is not enough free disk space to process this video safely.")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open this video. It may be corrupted or use an unsupported codec.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video does not contain readable frames.")

    scale = min(1.0, OUTPUT_MAX_WIDTH / width)
    output_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    processed_video = output / "processed_video.mp4"
    writer = cv2.VideoWriter(str(processed_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError("OpenCV could not create the processed video. Check available disk space and codecs.")

    detections_file = output / "detections.csv"
    tracks_file = output / "tracks.csv"
    crops_manifest_file = output / "crops.csv"
    rejected_crops_file = output / "rejected_crops.csv"
    rejected_crops_dir = output / "rejected_crops"
    rejected_crops_dir.mkdir(parents=True, exist_ok=True)
    track_summary_file = output / "tracks.json"
    tracker = SortTracker(max_age=tracker_max_age, min_iou=tracker_iou_threshold,
                          max_center_distance=tracker_max_center_distance)
    crop_candidates: dict[int, dict[int, dict]] = {}
    track_stats: dict[int, dict] = {}
    frames_read = frames_processed = total_detections = 0
    rejected_crop_count = 0
    rejected_preview_count = 0
    detector_seconds = 0.0
    started = time.perf_counter()
    detection_fields = ["camera_id", "frame_number", "timestamp_seconds", "detection_count"]
    track_fields = ["camera_id", "frame_number", "timestamp_seconds", "track_id",
                    "x1", "y1", "x2", "y2", "confidence", "observed", "recovered",
                    "time_since_update"]
    crop_fields = ["camera_id", "track_id", "frame_number", "timestamp_seconds",
                   "crop_filename", "x1", "y1", "x2", "y2", "confidence",
                   "sharpness", "quality_score", "width", "height"]
    rejection_fields = ["camera_id", "track_id", "frame_number", "timestamp_seconds",
                        "reason", "confidence", "sharpness", "width", "height",
                        "preview_filename"]
    try:
        with (detections_file.open("w", newline="", encoding="utf-8") as detection_stream,
              tracks_file.open("w", newline="", encoding="utf-8") as tracks_stream,
              rejected_crops_file.open("w", newline="", encoding="utf-8") as rejected_stream):
            detection_writer = csv.DictWriter(detection_stream, fieldnames=detection_fields)
            track_writer = csv.DictWriter(tracks_stream, fieldnames=track_fields)
            rejection_writer = csv.DictWriter(rejected_stream, fieldnames=rejection_fields)
            detection_writer.writeheader()
            track_writer.writeheader()
            rejection_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number = frames_read
                frames_read += 1
                sampled = frame_number % frame_interval == 0
                detections = []
                stage = "Tracking between detector checks"
                if sampled:
                    inference_started = time.perf_counter()
                    detections = detect_people_in_frame(
                        model, frame, confidence_threshold, image_size=image_size,
                        nms_iou_threshold=nms_iou_threshold,
                    )
                    detector_seconds += time.perf_counter() - inference_started
                    frames_processed += 1
                    total_detections += len(detections)
                    stage = "Detecting people and checking crop quality"
                observations = tracker.update(detections, frame_number)
                timestamp = frame_number / fps
                detection_writer.writerow({
                    "camera_id": camera_id,
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 3),
                    "detection_count": len(detections),
                })
                for observation in observations:
                    x1, y1, x2, y2 = observation.box
                    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
                    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    track_writer.writerow({
                        "camera_id": camera_id, "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3), "track_id": observation.track_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": round(observation.confidence, 4),
                        "observed": int(observation.observed),
                        "recovered": int(observation.recovered),
                        "time_since_update": observation.time_since_update,
                    })
                    stats = track_stats.setdefault(observation.track_id, {
                        "camera_id": camera_id, "track_id": observation.track_id,
                        "first_frame": frame_number, "first_time_seconds": round(timestamp, 3),
                        "last_frame": frame_number, "last_time_seconds": round(timestamp, 3),
        "observed_frames": 0, "max_confidence": 0.0, "confidence_history": [],
                        "last_box": [x1, y1, x2, y2],
                    })
                    if observation.observed:
                        stats["last_frame"] = frame_number
                        stats["last_time_seconds"] = round(timestamp, 3)
                        stats["last_box"] = [x1, y1, x2, y2]
                        stats["observed_frames"] += 1
                        stats["max_confidence"] = max(stats["max_confidence"], observation.confidence)
                        stats["confidence_history"].append({
                            "frame_number": frame_number,
                            "timestamp_seconds": round(timestamp, 3),
                            "confidence": round(observation.confidence, 4),
                        })
                        if sampled and observation.detection_index is not None:
                            matched_detection = detections[observation.detection_index]
                            safe = _safe_crop(frame, matched_detection.get("box"))
                            crop = safe[1] if safe is not None else None
                            crop_box = safe[0] if safe is not None else (0, 0, 0, 0)
                            crop_width = crop.shape[1] if crop is not None else 0
                            crop_height = crop.shape[0] if crop is not None else 0
                            confidence = float(matched_detection.get("confidence", 0.0))
                            sharpness = 0.0
                            reason = ""
                            if crop is None:
                                reason = "invalid_box_or_empty_crop"
                            elif crop_width < min_crop_width or crop_height < min_crop_height:
                                reason = "below_minimum_size"
                            else:
                                blurry, sharpness = is_blurry(crop, blur_threshold)
                                if blurry:
                                    reason = "blurry"
                                else:
                                    bucket = int(timestamp / CROP_TIME_BUCKET_SECONDS)
                                    existing_crop = crop_candidates.get(observation.track_id, {}).get(bucket)
                                    if existing_crop:
                                        if _near_duplicate(crop, existing_crop["path"]):
                                            reason = "near_duplicate_in_time_window"
                                        elif existing_crop["quality_score"] >= _crop_quality(confidence, sharpness):
                                            reason = "lower_quality_in_time_window"
                            if reason:
                                rejected_crop_count += 1
                                preview_filename = ""
                                # Keep a small, bounded visual sample of rejected crops.
                                if crop is not None and rejected_preview_count < 300:
                                    preview_filename = (
                                        f"camera{camera_id[-1]}_track{observation.track_id:04d}_"
                                        f"frame{frame_number:06d}_{reason}.jpg"
                                    )
                                    preview_path = rejected_crops_dir / preview_filename
                                    scale = min(1.0, 160 / max(crop.shape[:2]))
                                    preview = cv2.resize(
                                        crop,
                                        (max(1, int(crop.shape[1] * scale)),
                                         max(1, int(crop.shape[0] * scale))),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                    cv2.imwrite(str(preview_path), preview)
                                    rejected_preview_count += 1
                                rejection_writer.writerow({
                                    "camera_id": camera_id, "track_id": observation.track_id,
                                    "frame_number": frame_number,
                                    "timestamp_seconds": round(timestamp, 3),
                                    "reason": reason, "confidence": round(confidence, 4),
                                    "sharpness": round(sharpness, 2), "width": crop_width,
                                    "height": crop_height, "preview_filename": preview_filename,
                                })
                            else:
                                _store_crop_candidate(
                                    crop, camera_id, frame_number, timestamp,
                                    observation.track_id, confidence, sharpness,
                                    crop_box, crops_dir, crop_candidates,
                                )
                    _draw_track(frame, observation, camera_id, frame_number)
                if progress_callback:
                    progress_callback(frames_read, total_frames, stage)
                output_frame = frame if output_size == (width, height) else cv2.resize(
                    frame, output_size, interpolation=cv2.INTER_AREA
                )
                writer.write(output_frame)
    except Exception:
        logger.exception("Failed while processing %s", source)
        raise
    finally:
        capture.release()
        writer.release()

    if frames_read and (not processed_video.is_file() or processed_video.stat().st_size == 0):
        raise OSError("OpenCV did not produce a valid processed video file.")

    crop_rows = [row for buckets in crop_candidates.values() for row in buckets.values()]
    crop_rows.sort(key=lambda row: (row["track_id"], row["frame_number"]))
    with crops_manifest_file.open("w", newline="", encoding="utf-8") as crop_stream:
        crop_writer = csv.DictWriter(crop_stream, fieldnames=crop_fields)
        crop_writer.writeheader()
        for row in crop_rows:
            crop_writer.writerow({key: row[key] for key in crop_fields})

    final_frame = max(0, frames_read - 1)
    for track_id, stats in track_stats.items():
        gap = final_frame - stats["last_frame"]
        stats["status"] = "active at end" if gap == 0 else (
            "temporarily lost" if gap <= tracker_max_age else "ended"
        )
        stats["lifecycle_status"] = {
            "active at end": "ACTIVE",
            "temporarily lost": "LOST",
            "ended": "COMPLETED",
        }[stats["status"]]
        stats["crop_count"] = len(crop_candidates.get(track_id, {}))
        stats["duration_seconds"] = round(
            max(0.0, stats["last_time_seconds"] - stats["first_time_seconds"]), 3,
        )
        stats["crop_filenames"] = sorted(
            row["crop_filename"] for row in crop_candidates.get(track_id, {}).values()
        )
        stats["max_confidence"] = round(stats["max_confidence"], 4)
    track_summary_file.write_text(json.dumps({
        "camera_id": camera_id,
        "tracker": "SORT-style Kalman filter + Hungarian assignment",
        "track_ids_are_camera_local": True,
        "max_lost_frames": tracker_max_age,
        "tracker_settings": {"max_age_frames": tracker_max_age,
                             "max_center_distance": tracker_max_center_distance,
                             "iou_threshold": tracker_iou_threshold},
        "detector_settings": {"image_size": image_size,
                              "confidence_threshold": confidence_threshold,
                              "frame_interval": frame_interval,
                              "nms_iou_threshold": nms_iou_threshold},
        "tracks": sorted(track_stats.values(), key=lambda item: item["track_id"]),
    }, indent=2), encoding="utf-8")

    result = ProcessingResult(
        camera_id=camera_id,
        source_name=source.name,
        processed_video=str(processed_video),
        crops_dir=str(crops_dir),
        metadata_file=str(tracks_file),
        frames_read=frames_read,
        frames_processed=frames_processed,
        total_detections=total_detections,
        crops_saved=len(crop_rows),
        fps=fps,
        total_frames=total_frames,
        processing_seconds=round(time.perf_counter() - started, 2),
        tracks_file=str(tracks_file),
        crops_manifest_file=str(crops_manifest_file),
        track_summary_file=str(track_summary_file),
        track_count=len(track_stats),
        rejected_crops_file=str(rejected_crops_file),
        rejected_crops_count=rejected_crop_count,
    )
    (output / "statistics.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    (output / "performance.json").write_text(json.dumps({
        "detector_inference_seconds": round(detector_seconds, 3),
        "detector_checks": frames_processed,
        "average_detector_ms": round(1000 * detector_seconds / frames_processed, 2)
        if frames_processed else None,
        "frames_decoded_and_written": frames_read,
        "total_seconds": result.processing_seconds,
        "rejected_crop_count": rejected_crop_count,
    }, indent=2), encoding="utf-8")
    return result


def render_selected_person_video(
    source_path: Path | str,
    tracks_file: Path | str,
    output_path: Path | str,
    camera_name: str,
    global_person_id: str,
    selected_track_id: int | None,
    match_status: str,
    max_prediction_gap_frames: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Render a real frame-by-frame video containing only the selected identity.

    A short Kalman prediction can be displayed in amber. Unmatched cameras and
    frames where the selected track is no longer reliable remain the original
    scene, with a status label and no fabricated bounding box.
    """
    source = validate_video_path(source_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The source video could not be opened for selected-person playback.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The source video has invalid frame dimensions.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError("OpenCV could not create selected-person playback. Check the MP4 codec and disk space.")
    frames_written = 0
    try:
        if selected_track_id is None:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                cv2.putText(frame, f"{global_person_id} · {match_status.lower()} in {camera_name}",
                            (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 190, 245), 2)
                writer.write(frame)
                frame_index += 1
                frames_written += 1
                if progress_callback and (frames_written % 100 == 0 or
                                          frames_written == expected_frames):
                    progress_callback(frames_written, expected_frames)
        else:
            with Path(tracks_file).open(newline="", encoding="utf-8") as stream:
                rows = iter(csv.DictReader(stream))
                row = next(rows, None)
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    selected_row = None
                    while row is not None and int(row["frame_number"]) < frame_index:
                        row = next(rows, None)
                    while row is not None and int(row["frame_number"]) == frame_index:
                        if int(row["track_id"]) == int(selected_track_id):
                            selected_row = row
                        row = next(rows, None)
                    drew_box = False
                    if selected_row is not None:
                        observed = selected_row.get("observed") == "1"
                        try:
                            prediction_age = int(selected_row.get("time_since_update", 0))
                        except (TypeError, ValueError):
                            prediction_age = max_prediction_gap_frames + 1
                        reliable_prediction = (
                            not observed and 0 < prediction_age <= max_prediction_gap_frames
                        )
                        if observed or reliable_prediction:
                            x1, y1, x2, y2 = (
                                int(selected_row[key]) for key in ("x1", "y1", "x2", "y2")
                            )
                            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
                            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
                            if x2 > x1 and y2 > y1:
                                color = (50, 215, 150) if observed else (0, 175, 245)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                state = "" if observed else " · short prediction"
                                cv2.putText(
                                    frame, f"{global_person_id}{state}",
                                    (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, color, 2,
                                )
                                drew_box = True
                    if not drew_box:
                        status = "not visible" if selected_row is not None else "not detected"
                        cv2.putText(
                            frame, f"{global_person_id} · {status} in {camera_name}",
                            (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 190, 245), 2,
                        )
                    writer.write(frame)
                    frame_index += 1
                    frames_written += 1
                    if progress_callback and (frames_written % 100 == 0 or
                                              frames_written == expected_frames):
                        progress_callback(frames_written, expected_frames)
    except Exception:
        capture.release()
        writer.release()
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        writer.release()
    if not destination.is_file() or destination.stat().st_size == 0 or frames_written == 0:
        raise OSError("Selected-person playback could not be written.")
    if expected_frames > 0 and frames_written != expected_frames:
        destination.unlink(missing_ok=True)
        raise OSError(
            f"The source ended after {frames_written} frames; expected {expected_frames}. "
            "The partial selected-person video was discarded."
        )
    validation = cv2.VideoCapture(str(destination))
    try:
        if not validation.isOpened():
            raise OSError("The selected-person output is not a playable video.")
        output_width = int(validation.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(validation.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_fps = float(validation.get(cv2.CAP_PROP_FPS))
        output_frames = int(validation.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = validation.read()
        if (not ok or output_width != width or output_height != height or
                output_frames != frames_written or not np.isfinite(output_fps) or
                abs(output_fps - fps) > max(0.1, fps * 0.01)):
            raise OSError(
                "The selected-person video failed playback validation "
                f"(expected {width}x{height} at {fps:.3f} FPS, {frames_written} frames; "
                f"got {output_width}x{output_height} at {output_fps:.3f} FPS, {output_frames} frames)."
            )
    except Exception:
        validation.release()
        destination.unlink(missing_ok=True)
        raise
    finally:
        validation.release()
    return destination


def render_selected_track_video(source_path: Path | str, tracks_file: Path | str,
                                output_path: Path | str, camera_id: str,
                                selected_track_id: int) -> Path:
    """Backward-compatible selected local-track video renderer."""
    return render_selected_person_video(
        source_path, tracks_file, output_path, camera_id,
        f"{camera_id.title()} ID {selected_track_id}", selected_track_id,
        "SOURCE", max_prediction_gap_frames=0,
    )
