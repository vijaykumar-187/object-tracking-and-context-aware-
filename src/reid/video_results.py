"""Create camera-specific, playable videos for a confirmed global person."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable

import cv2

from src.detection.video_processor import render_selected_person_video
from src.reid.global_registry import (
    associate_confirmed_track, ensure_identity_source, get_or_create_identity,
    find_identity_for_track, identity_members,
)
from src.reid.hypotheses import confirmed_component, load_hypotheses


def _camera_name(result: dict) -> str:
    return str(result.get("camera_name") or
               result["camera_id"].replace("camera", "Camera "))


def _read_json(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def render_all_camera_tracking_videos(
    results: list[dict],
    run_dir: Path | str,
    source_camera_id: str,
    source_track_id: int,
    model_label: str,
    progress_callback: Callable[[str, int, int, int, int], None] | None = None,
) -> tuple[str, list[dict]]:
    """Reuse saved local trajectories and render one full MP4 per camera.

    Only a selected source track and human-confirmed SAME PERSON links receive
    boxes. Other cameras still produce original-timeline video with a no-match
    banner. No embeddings are recalculated here.
    """
    root = Path(run_dir)
    camera_by_id = {item["camera_id"]: item for item in results}
    if source_camera_id not in camera_by_id:
        raise ValueError("The selected source camera is not part of this processed run.")
    source = camera_by_id[source_camera_id]
    source_summary = _read_json(source["track_summary_file"])
    if int(source_track_id) not in {
        int(track["track_id"]) for track in source_summary.get("tracks", [])
    }:
        raise ValueError("The selected local track ID is not present in this camera.")

    hypotheses = load_hypotheses(root / "cross_camera_hypotheses.json")
    component = confirmed_component(hypotheses, source_camera_id, source_track_id)
    conflicts = []
    for item in hypotheses.values():
        if item.get("status") != "DIFFERENT PEOPLE":
            continue
        first = (str(item.get("camera1_id", "camera1")), int(item["camera1_track_id"]))
        second = (str(item.get("camera2_id", "camera2")), int(item["camera2_track_id"]))
        if first in component and second in component:
            conflicts.append((first, second))
    if conflicts:
        raise ValueError(
            "Confirmed identity links conflict with a DIFFERENT PEOPLE decision. "
            "Resolve the conflicting review before generating the videos."
        )

    registry_path = root / "global_identity_registry.json"
    global_id = find_identity_for_track(
        registry_path, source_camera_id, source_track_id,
    )
    if global_id is None:
        for node_camera, node_track in sorted(component):
            global_id = find_identity_for_track(registry_path, node_camera, node_track)
            if global_id:
                break
    global_id = global_id or get_or_create_identity(
        registry_path, source_camera_id, source_track_id, _camera_name(source),
    )
    ensure_identity_source(
        registry_path, global_id, source_camera_id, source_track_id,
        _camera_name(source),
    )
    for node_camera, node_track in sorted(component):
        if (node_camera, node_track) == (source_camera_id, int(source_track_id)):
            continue
        destination = camera_by_id.get(node_camera)
        if destination is None:
            continue
        evidence = next((item for item in hypotheses.values()
                         if item.get("status") == "SAME PERSON" and
                         ((str(item.get("camera1_id", "camera1")), int(item["camera1_track_id"])) == (node_camera, node_track) or
                          (str(item.get("camera2_id", "camera2")), int(item["camera2_track_id"])) == (node_camera, node_track))), None)
        associate_confirmed_track(
            registry_path, global_id, node_camera, node_track,
            _camera_name(destination),
            float(evidence.get("similarity", 0.0)) if evidence else 0.0,
            str(evidence.get("model_label", model_label)) if evidence else model_label,
        )

    members = identity_members(registry_path, global_id)
    output_dir = root / "global_person_videos" / global_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    total_cameras = len(results)
    for camera_index, result in enumerate(results, start=1):
        camera_id = result["camera_id"]
        label = _camera_name(result)
        member = members.get(camera_id)
        selected_local_id = int(member["track_id"]) if member else None
        match_status = "SOURCE TRACK" if camera_id == source_camera_id else (
            "HUMAN CONFIRMED" if member else "NO RELIABLE MATCH")
        summary = _read_json(result["track_summary_file"])
        detector_settings = summary.get("detector_settings", {})
        frame_interval = max(1, int(detector_settings.get("frame_interval", 1)))
        lost_window = max(1, int(summary.get("max_lost_frames", 1)))
        prediction_gap = min(max(3, 2 * frame_interval), lost_window) if member else 0
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or camera_id
        video_path = output_dir / f"{global_id}_{safe_label}_tracked.mp4"
        camera_progress = None
        if progress_callback:
            camera_progress = lambda current, total, name=label, index=camera_index: progress_callback(
                name, index, total_cameras, current, total,
            )
        render_selected_person_video(
            result["source_path"], result["tracks_file"], video_path,
            label, global_id, selected_local_id, match_status,
            max_prediction_gap_frames=prediction_gap,
            progress_callback=camera_progress,
        )
        capture = cv2.VideoCapture(str(video_path))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        output_rows.append({
            "global_person_id": global_id,
            "camera_id": camera_id,
            "camera_name": label,
            "original_filename": result["source_name"],
            "local_track_id": selected_local_id or "",
            "match_status": match_status,
            "match_similarity": member.get("similarity", "") if member else "",
            "appearance_model": member.get("model_label", "") if member else "",
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_seconds": round(frame_count / fps, 3) if fps > 0 else 0.0,
        })

    manifest_path = output_dir / "video_manifest.json"
    manifest_path.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    csv_path = output_dir / "global_person_assignments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["global_person_id", "camera_id", "camera_name", "original_filename",
                  "local_track_id", "match_status", "match_similarity", "appearance_model",
                  "video_filename", "width", "height", "fps", "frame_count", "duration_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in output_rows])
    return global_id, output_rows
