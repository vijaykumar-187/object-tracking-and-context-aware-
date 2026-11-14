"""Phase A before/after validation harness (standalone; does not modify src/).

Reprocesses the three baseline diagnostic videos with the SAME detector and
tracker settings used by the baseline run, using the Phase A tracker code, into
a NEW uniquely named output folder. The baseline folder is only read, never
written. Metrics are computed from each run's tracks.json / statistics.json /
performance.json and printed side by side.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from src.detection.detect import load_detector
from src.detection.video_processor import process_video

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "outputs" / "review3_results" / "run_20261114_113412_403361"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
NEW_RUN = ROOT / "outputs" / "review3_results" / f"phaseA_validation_{STAMP}"

CAMERAS = ["camera1", "camera2", "camera3"]

# Identical settings to the baseline run (run_summary + baseline tracks.json).
SETTINGS = dict(
    frame_interval=10,
    confidence_threshold=0.40,
    image_size=640,
    tracker_max_age=30,               # now counted in detection opportunities
    tracker_max_center_distance=2.5,
    tracker_iou_threshold=0.05,
    tracker_min_hits=2,               # Phase A confirmation gate
    min_crop_width=40,
    min_crop_height=80,
    blur_threshold=50.0,
    nms_iou_threshold=0.70,
)


def metrics_from_run(run_dir: Path, camera: str) -> dict:
    tracks = json.loads((run_dir / camera / "tracks.json").read_text(encoding="utf-8"))
    stats = json.loads((run_dir / camera / "statistics.json").read_text(encoding="utf-8"))
    perf_path = run_dir / camera / "performance.json"
    perf = json.loads(perf_path.read_text(encoding="utf-8")) if perf_path.is_file() else {}
    track_list = tracks["tracks"]
    lifespans = [int(t["observed_frames"]) for t in track_list]
    short = [n for n in lifespans if n <= 2]
    # Fragmentation proxy: mean observed-frames per track (higher = fewer, longer
    # tracks = less fragmentation) and the share of one/two-frame throwaway tracks.
    mean_life = statistics.mean(lifespans) if lifespans else 0.0
    median_life = statistics.median(lifespans) if lifespans else 0.0
    return {
        "track_count": len(track_list),
        "tracks_1_2_obs": len(short),
        "short_share_pct": round(100.0 * len(short) / len(track_list), 1) if track_list else 0.0,
        "median_lifespan_obs": median_life,
        "mean_lifespan_obs": round(mean_life, 2),
        "crops_saved": stats.get("crops_saved"),
        "total_detections": stats.get("total_detections"),
        "frames_read": stats.get("frames_read"),
        "detector_seconds": round(perf.get("detector_inference_seconds", 0.0), 2),
        "processing_seconds": round(stats.get("processing_seconds", 0.0), 2),
    }


def main() -> None:
    print(f"Baseline : {BASELINE}")
    print(f"New run  : {NEW_RUN}")
    NEW_RUN.mkdir(parents=True, exist_ok=False)

    detector_path = ROOT / "yolo26n.pt"
    load_detector(detector_path)
    from src.detection.detect import get_model  # cached loader used by the app
    dstat = detector_path.stat()
    model = get_model(str(detector_path), dstat.st_mtime_ns, dstat.st_size)

    for camera in CAMERAS:
        source = BASELINE / camera / f"{camera}.mp4"
        out_dir = NEW_RUN / camera
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy the source video in so the new run is self-contained (read-only
        # access to the baseline; we never write into it).
        dest_video = out_dir / f"{camera}.mp4"
        dest_video.write_bytes(source.read_bytes())
        t0 = time.perf_counter()
        result = process_video(camera, dest_video, model, out_dir, **SETTINGS)
        wall = time.perf_counter() - t0
        print(f"[{camera}] processed in {wall:.1f}s -> {result.track_count} tracks, "
              f"{result.crops_saved} crops")

    # Copy a small manifest so the new run records its own provenance.
    (NEW_RUN / "validation_settings.json").write_text(
        json.dumps({"baseline": str(BASELINE), "settings": SETTINGS,
                    "phase": "A - tracker stability"}, indent=2), encoding="utf-8")

    print("\n================ BEFORE / AFTER ================")
    header = f"{'metric':<24}{'BASE':>10}{'PHASE A':>10}{'delta':>10}"
    for camera in CAMERAS:
        base = metrics_from_run(BASELINE, camera)
        new = metrics_from_run(NEW_RUN, camera)
        print(f"\n----- {camera} -----")
        print(header)
        for key in base:
            b, n = base[key], new[key]
            delta = (n - b) if isinstance(b, (int, float)) and isinstance(n, (int, float)) else ""
            print(f"{key:<24}{str(b):>10}{str(n):>10}{str(delta):>10}")

    print("\nValidation complete. New run folder preserved for visual inspection:")
    print(NEW_RUN)


if __name__ == "__main__":
    main()
