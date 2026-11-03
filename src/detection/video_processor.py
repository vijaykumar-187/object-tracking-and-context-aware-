"""Reusable, Streamlit-independent video processing for Phase 2."""

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

from src.detection.detect import detect_people_in_frame

logger = logging.getLogger(__name__)
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ProgressCallback = Callable[[int, int, str], None]
OUTPUT_MAX_WIDTH = 1280


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


def validate_video_path(video_path: Path | str) -> Path:
    """Check the path and its extension; OpenCV validates the actual stream later."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise ValueError("The selected video file could not be found.")
    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
        raise ValueError(f"Unsupported video format. Choose one of: {supported}.")
    return path


def _safe_crop(frame: np.ndarray, box: object) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Clip coordinates and reject malformed, out-of-frame, or empty crops."""
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
    if crop.size == 0:
        return None
    return (x1, y1, x2, y2), crop


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _draw_track(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    camera_id: str,
    frame_number: int,
    track_id: int,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (42, 205, 155), 2)
    camera_label = camera_id.replace("camera", "Camera ").replace("cam", "Camera ")
    label = f"{camera_label} | Frame {frame_number + 1} | ID {track_id}"
    cv2.putText(frame, label, (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (42, 205, 155), 2)


def process_video(
    camera_id: str,
    video_path: Path | str,
    model: object,
    output_dir: Path | str,
    frame_interval: int = 30,
    confidence_threshold: float = 0.40,
    progress_callback: Optional[ProgressCallback] = None,
) -> ProcessingResult:
    """Detect sampled frames, save crops and metadata, and write an annotated MP4.

    The model is passed in so the caller can load it once and reuse it across videos.
    YOLO detects sampled frames. The last detected boxes remain visible between
    checks and are refreshed at the next sampled frame. This avoids expensive
    per-frame visual tracking on CPU while preventing the boxes from blinking.
    """
    source = validate_video_path(video_path)
    if frame_interval < 1:
        raise ValueError("Frame sampling interval must be at least 1.")
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
    output_size = (int(width * scale), int(height * scale))
    processed_video = output / "processed_video.mp4"
    writer = cv2.VideoWriter(
        str(processed_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError("OpenCV could not create the processed video. Check available disk space and codecs.")

    metadata_file = output / "detections.csv"
    frames_read = frames_processed = total_detections = crops_saved = 0
    active_tracks: list[dict] = []
    next_track_id = 1
    started = time.perf_counter()
    try:
        with metadata_file.open("w", newline="", encoding="utf-8") as csv_file:
            fields = ["camera_id", "frame_number", "timestamp_seconds", "crop_filename", "track_id", "x1", "y1", "x2", "y2", "confidence"]
            metadata = csv.DictWriter(csv_file, fieldnames=fields)
            metadata.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number = frames_read
                frames_read += 1
                annotated = frame.copy()
                stage = "Reading frames"
                if frame_number % frame_interval == 0:
                    detections = detect_people_in_frame(model, frame, confidence_threshold)
                    frames_processed += 1
                    total_detections += len(detections)
                    stage = "Detecting people and saving crops"
                    # A new YOLO observation refreshes the visual tracks. An empty
                    # observation clears them so boxes do not linger after exit.
                    valid_observations = []
                    for detection_index, detection in enumerate(detections, start=1):
                        safe = _safe_crop(frame, detection.get("box"))
                        if safe is None:
                            continue
                        (x1, y1, x2, y2), crop = safe
                        confidence = float(detection.get("confidence", 0.0))
                        valid_observations.append((detection_index, detection, (x1, y1, x2, y2), crop, confidence))

                    # Match fresh YOLO boxes to the previous frame-level tracks
                    # using overlap. IDs are local to this one camera video.
                    unmatched_previous = set(range(len(active_tracks)))
                    matched_ids = {}
                    candidates = []
                    for observation_index, observation in enumerate(valid_observations):
                        for previous_index, previous in enumerate(active_tracks):
                            overlap = _box_iou(observation[2], previous["box"])
                            if overlap >= 0.05:
                                candidates.append((overlap, observation_index, previous_index))
                    for _, observation_index, previous_index in sorted(candidates, reverse=True):
                        if observation_index not in matched_ids and previous_index in unmatched_previous:
                            matched_ids[observation_index] = active_tracks[previous_index]["track_id"]
                            unmatched_previous.remove(previous_index)

                    refreshed_tracks = []
                    for observation_index, (detection_index, detection, box, crop, confidence) in enumerate(valid_observations):
                        track_id = matched_ids.get(observation_index)
                        if track_id is None:
                            track_id = next_track_id
                            next_track_id += 1
                        crop_name = f"{camera_id}_frame{frame_number:06d}_person{detection_index:02d}.jpg"
                        if not cv2.imwrite(str(crops_dir / crop_name), crop):
                            raise OSError(f"Could not save crop image {crop_name}.")
                        crops_saved += 1
                        track = {"box": box, "track_id": track_id}
                        refreshed_tracks.append(track)
                        _draw_track(annotated, box, camera_id, frame_number, track_id)
                        metadata.writerow({
                            "camera_id": camera_id,
                            "frame_number": frame_number,
                            "timestamp_seconds": round(frame_number / fps, 3),
                            "crop_filename": crop_name,
                            "track_id": track_id,
                            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                            "confidence": round(confidence, 4),
                        })
                    active_tracks = refreshed_tracks
                else:
                    stage = "Drawing boxes between detection checks"
                    for track in active_tracks:
                        _draw_track(annotated, track["box"], camera_id, frame_number, track["track_id"])
                output_frame = annotated if output_size == (width, height) else cv2.resize(annotated, output_size, interpolation=cv2.INTER_AREA)
                writer.write(output_frame)
                if progress_callback:
                    progress_callback(frames_read, total_frames, stage)
    except Exception:
        logger.exception("Failed while processing %s", source)
        raise
    finally:
        capture.release()
        writer.release()

    if frames_read and (not processed_video.is_file() or processed_video.stat().st_size == 0):
        raise OSError("OpenCV did not produce a valid processed video file.")

    result = ProcessingResult(
        camera_id=camera_id,
        source_name=source.name,
        processed_video=str(processed_video),
        crops_dir=str(crops_dir),
        metadata_file=str(metadata_file),
        frames_read=frames_read,
        frames_processed=frames_processed,
        total_detections=total_detections,
        crops_saved=crops_saved,
        fps=fps,
        total_frames=total_frames,
        processing_seconds=round(time.perf_counter() - started, 2),
    )
    (output / "statistics.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
