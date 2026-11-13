"""Repeatable CPU two-video Review 3 pipeline and command-line entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.detection.detect import load_detector
from src.detection.video_processor import ProcessingResult, process_video, validate_video_path
from src.reid.appearance import (
    AGGREGATION_METHODS, load_fingerprints, make_track_fingerprint_variants,
    save_crop_embeddings, save_fingerprints,
)
from src.reid.matcher import match_two_cameras, write_matches_csv
from src.reid.osnet_extractor import CHECKPOINT_NAME, OSNetExtractor


def _source_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns}


def _read_crop_rows(result: dict) -> list[dict]:
    rows: list[dict] = []
    with Path(result["crops_manifest_file"]).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            crop_path = Path(result["crops_dir"]) / row["crop_filename"]
            if not crop_path.is_file():
                continue
            for name in ("track_id", "frame_number", "width", "height"):
                row[name] = int(row[name])
            for name in ("quality_score", "confidence", "sharpness", "timestamp_seconds"):
                row[name] = float(row[name])
            row["path"] = str(crop_path)
            row["camera_id"] = result["camera_id"]
            rows.append(row)
    return rows


def _reusable_results(output: Path, videos: dict[str, Path], settings: dict) -> dict[str, dict] | None:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("settings") != settings:
            return None
        if manifest.get("sources") != {camera: _source_fingerprint(path)
                                        for camera, path in videos.items()}:
            return None
        results = {}
        for camera in videos:
            camera_dir = output / camera
            stats_path = camera_dir / "statistics.json"
            summary_path = camera_dir / "tracks.json"
            if not all(path.is_file() for path in (
                    stats_path, summary_path, camera_dir / "crops.csv", camera_dir / "tracks.csv")):
                return None
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            payload["source_path"] = str(videos[camera])
            results[camera] = payload
        return results
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _load_feature_cache(output: Path, crop_rows: list[dict], checkpoint: Path,
                        run_manifest: Path) -> tuple[dict, dict, dict] | None:
    cache_path = output / "feature_cache.json"
    crop_path = output / "crop_embeddings_osnet.json"
    fingerprint_paths = {
        method: output / f"fingerprints_osnet_{method}.json"
        for method in AGGREGATION_METHODS
    }
    if not cache_path.is_file() or not crop_path.is_file() or not all(
            path.is_file() for path in fingerprint_paths.values()):
        return None
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        current_crop_signature = sorted(
            (row["path"], Path(row["path"]).stat().st_size,
             Path(row["path"]).stat().st_mtime_ns) for row in crop_rows
        )
        current_run_signature = hashlib.sha256(run_manifest.read_bytes()).hexdigest()
        if (cache.get("model_id") != OSNetExtractor.model_label or
                cache.get("checkpoint_name") != CHECKPOINT_NAME or
                cache.get("run_manifest_sha256") != current_run_signature or
                cache.get("crop_signature") != [list(item) for item in current_crop_signature]):
            return None
        checkpoint_signature = None
        if checkpoint.is_file():
            stat = checkpoint.stat()
            checkpoint_signature = [stat.st_size, stat.st_mtime_ns]
            if cache.get("checkpoint_signature") != checkpoint_signature:
                return None
        variants = {
            method: load_fingerprints(path, expected_model=OSNetExtractor.model_label)
            for method, path in fingerprint_paths.items()
        }
        crop_payload = json.loads(crop_path.read_text(encoding="utf-8"))
        if crop_payload.get("model_id") != OSNetExtractor.model_label:
            return None
        counts = {"embedding_count": int(cache.get("embedding_count", 0)),
                  "inference_seconds": float(cache.get("inference_seconds", 0.0)),
                  "average_inference_ms": cache.get("average_inference_ms")}
        return variants, crop_payload.get("crop_embeddings", {}), counts
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def run_matching(
    camera1_video: Path | str,
    camera2_video: Path | str,
    output_dir: Path | str,
    model_path: Path | str = "yolo26n.pt",
    osnet_checkpoint: Path | str | None = None,
    frame_interval: int = 10,
    confidence: float = 0.40,
    image_size: int = 640,
    tracker_iou_threshold: float = 0.05,
) -> dict:
    """Process two cameras, create OSNet track fingerprints, and write top-three matches."""
    videos = {"camera1": validate_video_path(camera1_video),
              "camera2": validate_video_path(camera2_video)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {"frame_interval": int(frame_interval), "confidence": float(confidence),
                "image_size": int(image_size), "tracker_iou_threshold": float(tracker_iou_threshold)}
    started = time.perf_counter()
    results = _reusable_results(output, videos, settings)
    detector_load_seconds = 0.0
    processing_seconds = 0.0
    if results is None:
        model_started = time.perf_counter()
        detector = load_detector(model_path)
        detector_load_seconds = time.perf_counter() - model_started
        results = {}
        for camera, video in videos.items():
            run = process_video(
                camera, video, detector, output / camera,
                frame_interval=frame_interval, confidence_threshold=confidence,
                image_size=image_size, tracker_iou_threshold=tracker_iou_threshold,
            )
            payload = run.__dict__.copy()
            payload["source_path"] = str(video)
            results[camera] = payload
            processing_seconds += float(run.processing_seconds)
        (output / "run_manifest.json").write_text(json.dumps({
            "sources": {camera: _source_fingerprint(path) for camera, path in videos.items()},
            "settings": settings,
        }, indent=2), encoding="utf-8")

    checkpoint = Path(osnet_checkpoint) if osnet_checkpoint else (
        Path(model_path).resolve().parent / "models" / CHECKPOINT_NAME
    )
    if not checkpoint.is_file() and osnet_checkpoint is None:
        checkpoint = Path("models") / CHECKPOINT_NAME
    all_crop_rows = [row for result in results.values() for row in _read_crop_rows(result)]
    feature_cache = _load_feature_cache(
        output, all_crop_rows, checkpoint, output / "run_manifest.json",
    )
    osnet_load_seconds = 0.0
    if feature_cache is not None:
        variants, crop_embeddings, feature_stats = feature_cache
    else:
        osnet_started = time.perf_counter()
        osnet = OSNetExtractor(checkpoint)
        osnet_load_seconds = time.perf_counter() - osnet_started
        variants, crop_embeddings = make_track_fingerprint_variants(
            osnet, all_crop_rows, feature_kind="osnet",
        )
        feature_stats = {
            "embedding_count": osnet.embedding_count,
            "inference_seconds": osnet.inference_seconds,
            "average_inference_ms": osnet.average_inference_ms,
        }
        for method, vectors in variants.items():
            save_fingerprints(output / f"fingerprints_osnet_{method}.json", vectors, {
                "model_id": OSNetExtractor.model_label, "checkpoint": CHECKPOINT_NAME,
                "feature_dimension": 512, "aggregation": method,
                "inference_seconds": round(osnet.inference_seconds, 4),
                "embedding_count": osnet.embedding_count,
                "average_inference_ms_per_crop": osnet.average_inference_ms,
            })
        save_crop_embeddings(output / "crop_embeddings_osnet.json", crop_embeddings, {
            "model_id": OSNetExtractor.model_label, "checkpoint": CHECKPOINT_NAME,
            "feature_dimension": 512,
        })
        cache_manifest = {
            "model_id": OSNetExtractor.model_label,
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_signature": ([checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns]
                                     if checkpoint.is_file() else None),
            "run_manifest_sha256": hashlib.sha256(
                (output / "run_manifest.json").read_bytes()
            ).hexdigest(),
            "crop_signature": sorted(
                [row["path"], Path(row["path"]).stat().st_size,
                 Path(row["path"]).stat().st_mtime_ns] for row in all_crop_rows
            ),
            "embedding_count": osnet.embedding_count,
            "inference_seconds": osnet.inference_seconds,
            "average_inference_ms": osnet.average_inference_ms,
        }
        (output / "feature_cache.json").write_text(
            json.dumps(cache_manifest, indent=2), encoding="utf-8",
        )
    for camera in videos:
        camera_vectors = {key: value for key, value in variants["mean"].items()
                          if key.startswith(f"{camera}:")}
        np.savez_compressed(
            output / camera / "fingerprints.npz",
            track_keys=np.asarray(list(camera_vectors), dtype=str),
            vectors=np.stack(list(camera_vectors.values())).astype(np.float32)
            if camera_vectors else np.empty((0, 512), dtype=np.float32),
        )
    matches = match_two_cameras(
        variants["mean"], all_crop_rows, top_k=3,
    )
    write_matches_csv(output / "matches.csv", matches)
    elapsed = time.perf_counter() - started
    summary = {
        "cameras": {camera: {
            "source_name": result.get("source_name", Path(result["source_path"]).name),
            "frames_read": result["frames_read"],
            "detections": result["total_detections"],
            "tracks": result["track_count"], "crops": result["crops_saved"],
            "processing_seconds": result["processing_seconds"],
        } for camera, result in results.items()},
        "feature_model": OSNetExtractor.model_label,
        "aggregation_candidates": list(AGGREGATION_METHODS),
        "selected_match_aggregation": "mean",
        "crop_embeddings": feature_stats["embedding_count"],
        "osnet_inference_seconds": round(feature_stats["inference_seconds"], 3),
        "osnet_average_ms_per_crop": feature_stats["average_inference_ms"],
        "osnet_model_loading_seconds": round(osnet_load_seconds, 3),
        "reused_saved_embeddings": feature_cache is not None,
        "detector_model_loading_seconds": round(detector_load_seconds, 3),
        "video_processing_seconds_sum": round(processing_seconds, 3),
        "total_seconds": round(elapsed, 3),
        "matches_rows": len(matches), "cpu_only": True,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU two-camera person matching")
    parser.add_argument("--camera1", required=True, type=Path, help="Camera 1 recorded video")
    parser.add_argument("--camera2", required=True, type=Path, help="Camera 2 recorded video")
    parser.add_argument("--output", required=True, type=Path, help="Run output folder")
    parser.add_argument("--detector", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--frame-interval", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.05)
    args = parser.parse_args()
    summary = run_matching(
        args.camera1, args.camera2, args.output, args.detector,
        args.osnet_checkpoint, args.frame_interval, args.confidence,
        args.image_size, args.tracker_iou_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
