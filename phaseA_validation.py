"""Phase A tracker-stability validation harness.

This script reprocesses the source videos of an existing baseline run with the
current Phase A tracker and writes every new artifact into a fresh, uniquely
named output directory. The baseline directory is only read, never written.

The baseline predates the switch from raw-frame retention to detection
opportunity retention. Its ``max_lost_frames`` value therefore cannot be passed
straight through as ``tracker_max_age``: the current tracker counts missed
*sampled* detection opportunities, while the baseline counted raw video frames.
Two comparisons are produced instead:

* ``equivalent_retention`` -- the legacy raw-frame window translated into the
  same number of sampled detection opportunities.
* ``new_default`` -- the current default retention window.

Heavy detector/video imports live inside :func:`main` so the pure metric and
settings helpers below can be imported and unit-tested without OpenCV or
Ultralytics.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "outputs" / "review3_results" / "run_20261114_113412_403361"
CAMERAS = ("camera1", "camera2", "camera3")

# Current Phase A defaults. ``max_age`` is counted in missed detection
# opportunities, not raw frames.
NEW_DEFAULT_MAX_AGE = 30
NEW_DEFAULT_MIN_HITS = 2

METRIC_FIELDS = (
    "track_count",
    "tracks_1_2_obs",
    "short_share_pct",
    "median_lifespan_obs",
    "mean_lifespan_obs",
    "max_lifespan_obs",
    "crops_saved",
    "total_detections",
    "frames_read",
    "frames_processed",
    "detector_seconds",
    "processing_seconds",
)


def read_json(path: Path | str) -> dict:
    """Read a JSON object, failing loudly when the file is absent or invalid."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required validation file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def equivalent_retention_opportunities(legacy_max_lost_frames: int,
                                       frame_interval: int) -> int:
    """Translate a legacy raw-frame retention window into sampled opportunities.

    The current tracker increments missed-detection counters only on frames where
    the detector actually ran. A legacy window of 30 raw frames at a sampling
    interval of 10 therefore corresponds to roughly three missed opportunities.
    Integer division is intentional: it never grants a *longer* retention window
    than the baseline had.
    """
    legacy = int(legacy_max_lost_frames)
    interval = int(frame_interval)
    if legacy < 1:
        raise ValueError("Legacy max_lost_frames must be at least one frame.")
    if interval < 1:
        raise ValueError("frame_interval must be at least one frame.")
    return max(1, legacy // interval)


def baseline_tracker_context(baseline_dir: Path | str, camera: str = "camera1") -> dict:
    """Extract the baseline retention semantics and detector settings."""
    baseline_dir = Path(baseline_dir)
    tracks_path = baseline_dir / camera / "tracks.json"
    tracks = read_json(tracks_path)
    tracker_settings = dict(tracks.get("tracker_settings") or {})
    detector_settings = dict(tracks.get("detector_settings") or {})

    frame_interval = int(detector_settings.get("frame_interval") or 1)
    legacy_frames = tracks.get("max_lost_frames")
    if legacy_frames is None:
        legacy_frames = tracker_settings.get("max_age_frames")
    if legacy_frames is None:
        raise ValueError(
            f"{tracks_path} does not record a legacy raw-frame retention window."
        )
    legacy_frames = int(legacy_frames)
    equivalent = equivalent_retention_opportunities(legacy_frames, frame_interval)

    return {
        "baseline_dir": str(baseline_dir),
        "camera": camera,
        "tracks_json": str(tracks_path),
        "legacy_max_lost_frames": legacy_frames,
        "frame_interval": frame_interval,
        "equivalent_retention_opportunities": equivalent,
        "equivalent_retention_max_age_frames": equivalent * frame_interval,
        "new_default_opportunities": NEW_DEFAULT_MAX_AGE,
        "new_default_max_age_frames": NEW_DEFAULT_MAX_AGE * frame_interval,
        "tracker_settings": tracker_settings,
        "detector_settings": detector_settings,
        "baseline_records_min_hits": "min_hits" in tracker_settings,
        "semantics": (
            "Baseline max_lost_frames counts raw video frames. The current "
            "tracker_max_age counts missed sampled detection opportunities, so "
            "reusing the legacy value directly would multiply the retention "
            "window by the frame interval."
        ),
    }


def _base_process_settings(context: dict) -> dict:
    """Build process_video settings from the baseline where it recorded them."""
    detector = context["detector_settings"]
    tracker = context["tracker_settings"]
    return {
        "frame_interval": int(detector.get("frame_interval", context["frame_interval"])),
        "confidence_threshold": float(detector.get("confidence_threshold", 0.40)),
        "image_size": int(detector.get("image_size", 640)),
        "tracker_max_center_distance": float(tracker.get("max_center_distance", 2.5)),
        "tracker_iou_threshold": float(tracker.get("iou_threshold", 0.05)),
        "tracker_min_hits": NEW_DEFAULT_MIN_HITS,
        "min_crop_width": 40,
        "min_crop_height": 80,
        "blur_threshold": 50.0,
        "nms_iou_threshold": float(detector.get("nms_iou_threshold", 0.70)),
    }


def build_run_configurations(baseline_context: dict) -> list[dict]:
    """Return the equivalent-retention and new-default comparison runs."""
    base = _base_process_settings(baseline_context)
    interval = int(baseline_context["frame_interval"])
    equivalent = int(baseline_context["equivalent_retention_opportunities"])
    confirmation_note = (
        f"tracker_min_hits={NEW_DEFAULT_MIN_HITS} is a new Phase A confirmation "
        "gate; the baseline tracks.json has no min_hits setting."
    )
    return [
        {
            "key": "equivalent_retention",
            "label": "Equivalent retention",
            "tracker_max_age": equivalent,
            "max_age_frames": equivalent * interval,
            "settings": {**base, "tracker_max_age": equivalent},
            "note": (
                f"Legacy {baseline_context['legacy_max_lost_frames']} raw frames "
                f"translated at frame_interval={interval} to {equivalent} missed "
                f"detection opportunities ({equivalent * interval} raw frames). "
                + confirmation_note
            ),
        },
        {
            "key": "new_default",
            "label": "New default retention",
            "tracker_max_age": NEW_DEFAULT_MAX_AGE,
            "max_age_frames": NEW_DEFAULT_MAX_AGE * interval,
            "settings": {**base, "tracker_max_age": NEW_DEFAULT_MAX_AGE},
            "note": (
                f"Current default of {NEW_DEFAULT_MAX_AGE} missed detection "
                f"opportunities = {NEW_DEFAULT_MAX_AGE * interval} raw frames at "
                f"frame_interval={interval}. " + confirmation_note
            ),
        },
    ]


def metrics_from_run(run_dir: Path | str, camera: str) -> dict:
    """Compute tracker-stability metrics from a run's per-camera artifacts."""
    run_dir = Path(run_dir)
    tracks = read_json(run_dir / camera / "tracks.json")
    stats = read_json(run_dir / camera / "statistics.json")
    perf_path = run_dir / camera / "performance.json"
    perf = read_json(perf_path) if perf_path.is_file() else {}

    track_list = tracks.get("tracks") or []
    lifespans = [int(track.get("observed_frames", 0)) for track in track_list]
    short = [count for count in lifespans if count <= 2]

    # Fragmentation proxies: mean/median observed detections per track (higher
    # means fewer, longer tracks) and the share of one/two-observation tracks.
    mean_life = statistics.mean(lifespans) if lifespans else 0.0
    median_life = statistics.median(lifespans) if lifespans else 0.0
    return {
        "track_count": len(track_list),
        "tracks_1_2_obs": len(short),
        "short_share_pct": round(100.0 * len(short) / len(track_list), 1) if track_list else 0.0,
        "median_lifespan_obs": median_life,
        "mean_lifespan_obs": round(mean_life, 2),
        "max_lifespan_obs": max(lifespans) if lifespans else 0,
        "crops_saved": stats.get("crops_saved"),
        "total_detections": stats.get("total_detections"),
        "frames_read": stats.get("frames_read"),
        "frames_processed": stats.get("frames_processed"),
        "detector_seconds": round(perf.get("detector_inference_seconds", 0.0), 2),
        "processing_seconds": round(stats.get("processing_seconds", 0.0), 2),
    }


def comparison_rows(baseline_metrics: dict, new_metrics: dict) -> list[dict]:
    """Build metric/baseline/new/delta rows for one camera comparison."""
    rows = []
    for key in METRIC_FIELDS:
        baseline_value = baseline_metrics.get(key)
        new_value = new_metrics.get(key)
        delta = None
        if isinstance(baseline_value, (int, float)) and isinstance(new_value, (int, float)):
            delta = round(new_value - baseline_value, 2)
        rows.append({
            "metric": key,
            "baseline": baseline_value,
            "new": new_value,
            "delta": delta,
        })
    return rows


def build_comparison(baseline_dir: Path | str, configs: list[dict],
                     run_root: Path | str) -> dict:
    """Compare every configuration/camera against the untouched baseline."""
    comparison: dict[str, dict] = {}
    for config in configs:
        key = config["key"]
        comparison[key] = {}
        for camera in CAMERAS:
            baseline_metrics = metrics_from_run(baseline_dir, camera)
            new_metrics = metrics_from_run(Path(run_root) / key, camera)
            comparison[key][camera] = {
                "baseline": baseline_metrics,
                "new": new_metrics,
                "rows": comparison_rows(baseline_metrics, new_metrics),
            }
    return comparison


def environment_snapshot() -> dict:
    """Record the interpreter and heavy dependencies without importing them."""
    packages = {}
    for name in ("numpy", "cv2", "torch", "ultralytics", "streamlit"):
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            packages[name] = False
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages_available": packages,
    }


def _format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_markdown_report(baseline_context: dict, configs: list[dict],
                           comparison: dict) -> str:
    """Render a human-readable comparison report."""
    lines = [
        "# Phase A tracker stability validation",
        "",
        f"- Baseline: `{baseline_context['baseline_dir']}`",
        f"- Baseline legacy raw-frame window: `{baseline_context['legacy_max_lost_frames']}` frames",
        f"- Detector frame interval: `{baseline_context['frame_interval']}` frames",
        f"- Equivalent retention: `{baseline_context['equivalent_retention_opportunities']}` "
        f"missed detection opportunities "
        f"(`{baseline_context['equivalent_retention_max_age_frames']}` raw frames)",
        f"- New default retention: `{baseline_context['new_default_opportunities']}` "
        f"opportunities (`{baseline_context['new_default_max_age_frames']}` raw frames)",
        "",
        "> The baseline stored a raw-frame `max_lost_frames` window and had no "
        "`min_hits` gate. The current tracker counts missed sampled detection "
        f"opportunities and applies `min_hits={NEW_DEFAULT_MIN_HITS}`, so deltas "
        "reflect both the retention translation and the new confirmation gate. "
        "No improvement is claimed unless the equivalent-retention numbers "
        "support it.",
        "",
    ]
    for config in configs:
        lines.append(f"## {config['label']} (`{config['key']}`)")
        lines.append("")
        lines.append(config["note"])
        lines.append("")
        for camera in CAMERAS:
            entry = comparison[config["key"]][camera]
            lines.append(f"### {camera}")
            lines.append("")
            lines.append("| metric | baseline | phase A | delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for row in entry["rows"]:
                lines.append(
                    f"| {row['metric']} | {_format_value(row['baseline'])} | "
                    f"{_format_value(row['new'])} | {_format_value(row['delta'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not BASELINE.is_dir():
        raise FileNotFoundError(f"Baseline run directory is missing: {BASELINE}")

    baseline_context = baseline_tracker_context(BASELINE)
    configs = build_run_configurations(baseline_context)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = ROOT / "outputs" / "review3_results" / f"phaseA_validation_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    print(f"Baseline : {BASELINE}")
    print(f"New run  : {run_root}")
    print(
        "Retention: baseline "
        f"{baseline_context['legacy_max_lost_frames']} raw frames -> "
        f"{baseline_context['equivalent_retention_opportunities']} equivalent "
        f"opportunities; new default {NEW_DEFAULT_MAX_AGE} opportunities."
    )

    settings_document = {
        "phase": "A - tracker stability",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline_context,
        "configurations": [
            {
                "key": config["key"],
                "label": config["label"],
                "tracker_max_age": config["tracker_max_age"],
                "max_age_frames": config["max_age_frames"],
                "settings": config["settings"],
                "note": config["note"],
            }
            for config in configs
        ],
        "environment": environment_snapshot(),
        "source_policy": (
            "Baseline source videos are read directly. The baseline directory is "
            "never written; all outputs go to this new run directory."
        ),
    }
    (run_root / "validation_settings.json").write_text(
        json.dumps(settings_document, indent=2), encoding="utf-8"
    )

    detector_path = ROOT / "yolo26n.pt"
    if not detector_path.is_file():
        raise FileNotFoundError(f"Detector weights are missing: {detector_path}")

    # Heavy imports stay local so the helpers above remain importable without
    # OpenCV/Ultralytics (and therefore unit-testable in a minimal environment).
    from src.detection.detect import load_detector
    from src.detection.video_processor import process_video

    model = load_detector(detector_path)  # exactly one detector load

    processing: dict[str, dict] = {}
    for config in configs:
        key = config["key"]
        processing[key] = {}
        for camera in CAMERAS:
            source = BASELINE / camera / f"{camera}.mp4"
            if not source.is_file():
                raise FileNotFoundError(f"Baseline source video is missing: {source}")
            out_dir = run_root / key / camera
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            result = process_video(camera, source, model, out_dir, **config["settings"])
            wall = time.perf_counter() - started
            processing[key][camera] = {
                "track_count": result.track_count,
                "crops_saved": result.crops_saved,
                "wall_seconds": round(wall, 2),
            }
            print(
                f"[{key}/{camera}] {result.track_count} tracks, "
                f"{result.crops_saved} crops, {wall:.1f}s"
            )

    comparison = build_comparison(BASELINE, configs, run_root)
    (run_root / "comparison.json").write_text(
        json.dumps({
            "baseline": str(BASELINE),
            "configurations": [
                {
                    "key": config["key"],
                    "label": config["label"],
                    "tracker_max_age": config["tracker_max_age"],
                    "max_age_frames": config["max_age_frames"],
                    "settings": config["settings"],
                }
                for config in configs
            ],
            "processing": processing,
            "comparison": comparison,
        }, indent=2),
        encoding="utf-8",
    )

    report = render_markdown_report(baseline_context, configs, comparison)
    (run_root / "comparison.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nValidation artifacts: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
