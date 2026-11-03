"""
Context-Aware Person Re-Identification Across Multiple Cameras
Phase 2: Person Detection and Crop Extraction

This module processes video streams from multiple camera views, detects
pedestrians using a lightweight pretrained YOLO model, filters out low-quality
or blurry crops, and logs all bounding box coordinates and metadata to CSV.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detection")

# Directory and Path Resolutions
ROOT = Path(__file__).resolve().parents[2]

CAMERAS: Dict[str, Path] = {
    "cam1": ROOT / "data" / "videos" / "cam1.mp4",
    "cam2": ROOT / "data" / "videos" / "cam2.mp4",
    "cam3": ROOT / "data" / "videos" / "cam3.mp4",
}

CROP_DIR = ROOT / "data" / "crops"
OUTPUT_DIR = ROOT / "outputs" / "detections"
PREVIEW_DIR = OUTPUT_DIR / "previews"
METADATA_FILE = OUTPUT_DIR / "crop_metadata.csv"

# Pretrained model path (YOLOv8 / YOLO26 lightweight CPU model)
MODEL_PATH = ROOT / "yolo26n.pt"

# Detection and filtering parameters
FRAME_INTERVAL = 15          # Process 1 frame every 15 frames (e.g., 2 FPS from a 30 FPS video)
CONFIDENCE_THRESHOLD = 0.40  # Minimum detection confidence
PERSON_CLASS_ID = 0          # Class 0 in standard COCO dataset is 'person'

MIN_WIDTH = 40               # Minimum acceptable crop width (pixels)
MIN_HEIGHT = 80              # Minimum acceptable crop height (pixels)
BLUR_THRESHOLD = 50.0        # Laplacian variance threshold below which an image is considered blurry

CSV_FIELDNAMES = [
    "camera_id",
    "frame_number",
    "timestamp_seconds",
    "fps",
    "crop_filename",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "sharpness",
]


def compute_sharpness(image: np.ndarray) -> float:
    """
    Computes image sharpness using the variance of the Laplacian operator.
    Higher values indicate sharp edges; low values indicate motion or optical blur.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """
    Evaluates whether an extracted person crop meets the sharpness criteria.
    Returns a tuple of (is_blurry: bool, sharpness_score: float).
    """
    sharpness = compute_sharpness(image)
    return sharpness < threshold, sharpness


def load_detector(model_path: Optional[Path] = None) -> YOLO:
    """
    Loads the pretrained YOLO model from the given path.
    Falls back to MODEL_PATH if no path is provided.
    """
    path = model_path or MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Pretrained YOLO model weights not found at: {path}. "
            f"Please ensure yolo26n.pt exists in the project root."
        )
    logger.info("Loading pretrained YOLO model from: %s", path)
    model = YOLO(str(path))
    return model


def detect_people_in_frame(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> List[Dict]:
    """
    Runs YOLO inference on a single frame targeting only the 'person' class.
    Returns a list of valid detected bounding boxes with clipped coordinates.
    """
    height, width = frame.shape[:2]
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        classes=[PERSON_CLASS_ID],
        device="cpu",
        verbose=False,
    )

    detections = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].cpu().item())

        # Clip coordinates within the frame boundary
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        crop_w = x2 - x1
        crop_h = y2 - y1

        detections.append({
            "box": (x1, y1, x2, y2),
            "width": crop_w,
            "height": crop_h,
            "confidence": confidence,
        })

    return detections


def process_camera(
    camera_id: str,
    video_path: Path,
    model: YOLO,
    writer: Optional[csv.DictWriter] = None,
    save_previews: bool = True,
    max_frames: Optional[int] = None,
) -> int:
    """
    Processes a video for a single camera:
    - Samples frames at FRAME_INTERVAL.
    - Extracts and filters person crops.
    - Saves crop images to data/crops/<camera_id>/.
    - Logs metadata to the CSV writer.
    - Saves preview frames with bounding boxes (only when people are detected).

    Returns the total number of saved person crops.
    """
    if not video_path.exists():
        logger.warning("[SKIP] Video not found for %s: %s", camera_id, video_path)
        return 0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.error("[ERROR] Failed to open video: %s", video_path)
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Safe fallback if video header is missing FPS

    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "[START] Processing %s: %s (FPS: %.2f, Total Frames: %d)",
        camera_id, video_path.name, fps, total_video_frames
    )

    camera_crop_dir = CROP_DIR / camera_id
    camera_crop_dir.mkdir(parents=True, exist_ok=True)

    camera_preview_dir = PREVIEW_DIR / camera_id
    if save_previews:
        camera_preview_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_crops_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        # Process sampled frames only
        if frame_idx % FRAME_INTERVAL == 0:
            detections = detect_people_in_frame(model, frame, CONFIDENCE_THRESHOLD)
            annotated_frame = frame.copy() if save_previews else None
            frame_had_saved_crops = False

            person_idx = 0
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                crop_w = det["width"]
                crop_h = det["height"]
                confidence = det["confidence"]

                # Crop dimension filter
                if crop_w < MIN_WIDTH or crop_h < MIN_HEIGHT:
                    continue

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Blur filter
                blurry, sharpness = is_blurry(person_crop, BLUR_THRESHOLD)
                if blurry:
                    continue

                person_idx += 1
                saved_crops_count += 1
                frame_had_saved_crops = True

                filename = f"{camera_id}_frame{frame_idx:06d}_person{person_idx:02d}.jpg"
                crop_file_path = camera_crop_dir / filename
                cv2.imwrite(str(crop_file_path), person_crop)

                timestamp_sec = round(frame_idx / fps, 3)

                if writer is not None:
                    writer.writerow({
                        "camera_id": camera_id,
                        "frame_number": frame_idx,
                        "timestamp_seconds": timestamp_sec,
                        "fps": round(fps, 2),
                        "crop_filename": filename,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": round(confidence, 4),
                        "sharpness": round(sharpness, 2),
                    })

                if save_previews and annotated_frame is not None:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {confidence:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Save annotated preview only if at least one valid crop was retained
            if save_previews and frame_had_saved_crops and annotated_frame is not None:
                preview_path = camera_preview_dir / f"{camera_id}_frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(preview_path), annotated_frame)

        frame_idx += 1

    capture.release()
    logger.info(
        "[DONE] %s: Read %d frames | Extracted & saved %d valid person crops",
        camera_id, frame_idx, saved_crops_count
    )
    return saved_crops_count


def run_detection_pipeline(
    cameras: Optional[Dict[str, Path]] = None,
    save_previews: bool = True,
) -> None:
    """
    Executes the detection pipeline across all specified camera feeds.
    """
    camera_dict = cameras or CAMERAS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cam_id in camera_dict:
        (CROP_DIR / cam_id).mkdir(parents=True, exist_ok=True)

    model = load_detector(MODEL_PATH)

    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        total_saved = 0
        for cam_id, vid_path in camera_dict.items():
            saved = process_camera(
                camera_id=cam_id,
                video_path=vid_path,
                model=model,
                writer=writer,
                save_previews=save_previews,
            )
            total_saved += saved

    logger.info("Detection pipeline finished.")
    logger.info("Total crops saved across all cameras: %d", total_saved)
    logger.info("Metadata saved to: %s", METADATA_FILE)


if __name__ == "__main__":
    run_detection_pipeline()
