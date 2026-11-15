"""Create camera-specific and combined videos for a confirmed global person."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.detection.video_processor import render_selected_person_video
from src.reid.global_registry import (
    associate_confirmed_track, ensure_identity_source, get_or_create_identity,
    find_identity_for_track, identity_members, identity_members_all,
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


# ---------------------------------------------------------------------------
# Phase D: one combined multi-camera video for a confirmed global identity
#
# These helpers read the existing registry, track CSVs and source videos. They
# never create, merge or reject an identity. Tiles are aligned by elapsed time
# from the start of each source video; no camera synchronisation is claimed.
# ---------------------------------------------------------------------------

COMBINED_OUTPUT_FPS = 10.0
COMBINED_TILE_SIZE = (640, 360)
COMBINED_OUTPUT_SIZE = (1280, 720)
COMBINED_MAX_CAMERAS = 4
COMBINED_HEADER_HEIGHT = 56
COMBINED_FOOTER_HEIGHT = 24
COMBINED_SIGNATURE_VERSION = "combined_global_person_v1"
COMBINED_SYNC_NOTE = (
    "Frames are aligned by elapsed time from each source video. The cameras are "
    "not synchronised, so tiles are not asserted to show the same real-world moment."
)

COMBINED_BACKGROUND = (24, 26, 30)
COMBINED_PANEL = (34, 37, 42)
COMBINED_HEADER_FILL = (16, 18, 21)
COMBINED_BORDER = (70, 74, 80)
COMBINED_TEXT = (238, 240, 243)
COMBINED_MUTED = (168, 174, 182)
COMBINED_OBSERVED = (50, 215, 150)
COMBINED_PREDICTED = (0, 175, 245)
COMBINED_STATUS = (0, 190, 245)


def grid_shape(camera_count: int) -> tuple[int, int]:
    """Return (rows, columns) for the combined grid: one tile per camera."""
    count = int(camera_count)
    if count <= 0:
        raise ValueError("The combined grid needs at least one camera.")
    if count > COMBINED_MAX_CAMERAS:
        raise ValueError(
            f"The combined grid supports at most {COMBINED_MAX_CAMERAS} cameras."
        )
    if count == 1:
        return 1, 1
    if count == 2:
        return 1, 2
    return 2, 2


def prediction_window(track_summary: dict) -> int:
    """Reuse the existing amber-prediction window rule for the combined tiles."""
    detector = track_summary.get("detector_settings") or {}
    try:
        interval = max(1, int(detector.get("frame_interval", 1)))
    except (TypeError, ValueError):
        interval = 1
    try:
        lost_window = max(1, int(track_summary.get("max_lost_frames", 1)))
    except (TypeError, ValueError):
        lost_window = 1
    return min(max(3, 2 * interval), lost_window)


def load_track_index(tracks_file: Path | str,
                     track_ids: list[int] | None = None) -> dict:
    """Index saved track rows by track ID and frame for random-access lookup.

    Every member of a global identity is indexed, so several local tracks from
    one camera can be boxed on the same tile. Frames without a row stay absent.
    """
    wanted = None if track_ids is None else {int(value) for value in track_ids}
    rows: dict[int, dict[int, dict]] = {}
    spans: dict[int, tuple[int, int]] = {}
    with Path(tracks_file).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            try:
                track_id = int(raw["track_id"])
                frame_number = int(raw["frame_number"])
                rect = tuple(
                    int(round(float(raw[key]))) for key in ("x1", "y1", "x2", "y2")
                )
            except (KeyError, TypeError, ValueError):
                continue
            if wanted is not None and track_id not in wanted:
                continue
            observed = str(raw.get("observed", "1")).strip().lower() in {"1", "true"}
            try:
                age = max(0, int(raw.get("time_since_update") or 0))
            except (TypeError, ValueError):
                age = 0
            rows.setdefault(track_id, {})[frame_number] = {
                "rect": rect, "observed": observed, "age": age,
            }
            first, last = spans.get(track_id, (frame_number, frame_number))
            spans[track_id] = (min(first, frame_number), max(last, frame_number))
    return {"rows": rows, "spans": spans}


def plan_frame_boxes(members: list[dict], track_index: dict, frame_number: int,
                     prediction_frames: int) -> list[dict]:
    """Decide what may be drawn for every member at one source frame.

    A box is planned only for an observed row or a short Kalman prediction.
    Everything else keeps the frame clean and reports the real state instead of
    inventing continuous tracking.
    """
    boxes = []
    for member in sorted(members, key=lambda item: int(item["track_id"])):
        track_id = int(member["track_id"])
        row = track_index["rows"].get(track_id, {}).get(int(frame_number))
        span = track_index["spans"].get(track_id)
        if row is None:
            if span is None:
                state = "member track missing from this run"
            elif int(frame_number) < span[0]:
                state = "not yet in view"
            else:
                state = "left the scene"
            boxes.append({
                "track_id": track_id, "rect": None, "observed": False,
                "state": state, "age": None,
            })
            continue
        if row["observed"] or 0 < row["age"] <= int(prediction_frames):
            boxes.append({
                "track_id": track_id,
                "rect": row["rect"],
                "observed": row["observed"],
                "state": "" if row["observed"] else f"prediction ({row['age']}f)",
                "age": row["age"],
            })
        else:
            boxes.append({
                "track_id": track_id, "rect": None, "observed": False,
                "state": f"track lost ({row['age']}f)", "age": row["age"],
            })
    return boxes


def tile_caption(camera_name: str, global_person_id: str, track_ids: list[int],
                 timestamp_seconds: float,
                 duration_seconds: float) -> dict[str, str]:
    """Header and footer text: camera, local IDs, elapsed time and global ID."""
    ids = ", ".join(str(int(value)) for value in track_ids) if track_ids else "none"
    return {
        "header": f"{camera_name} | {global_person_id}",
        "timestamp": f"Local ID {ids} | t = {timestamp_seconds:0.3f} s (elapsed)",
        "footer": (
            f"elapsed {timestamp_seconds:0.3f} s of {duration_seconds:0.3f} s"
            " | per-video time, not synchronised"
        ),
    }


def end_card_lines(camera_name: str, global_person_id: str, track_ids: list[int],
                   duration_seconds: float) -> list[str]:
    """Text drawn when a source video has no further frames for the timeline."""
    ids = ", ".join(str(int(value)) for value in track_ids) if track_ids else "none"
    return [
        "END OF FOOTAGE",
        f"{camera_name} | source ended at {duration_seconds:0.3f} s",
        f"{global_person_id} | Local ID {ids}",
        "no further frames in this source",
    ]


def empty_slot_lines(label: str = "No camera in this run") -> list[str]:
    """Neutral text for an unused grid cell; it never implies a missing match."""
    return [label, "unused grid slot"]


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int],
               scale: float, colour: tuple[int, int, int],
               thickness: int = 1) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, thickness, cv2.LINE_AA)


def _draw_label(image: np.ndarray, text: str, x: int, y: int,
                colour: tuple[int, int, int]) -> None:
    """Small high-contrast box label that stays inside the tile."""
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1,
    )
    x = max(2, min(int(x), image.shape[1] - text_width - 4))
    y = max(text_height + 4, min(int(y), image.shape[0] - 4))
    cv2.rectangle(image, (x - 2, y - text_height - 3),
                  (x + text_width + 2, y + 3), (12, 14, 16), -1)
    _draw_text(image, text, (x, y), 0.42, colour)


def _draw_chrome(tile: np.ndarray, caption: dict[str, str]) -> None:
    tile_height, tile_width = tile.shape[:2]
    cv2.rectangle(tile, (0, 0), (tile_width - 1, COMBINED_HEADER_HEIGHT - 1),
                  COMBINED_HEADER_FILL, -1)
    _draw_text(tile, caption["header"], (12, 23), 0.52, COMBINED_TEXT)
    _draw_text(tile, caption["timestamp"], (12, 45), 0.42, COMBINED_MUTED)
    cv2.rectangle(tile, (0, tile_height - COMBINED_FOOTER_HEIGHT),
                  (tile_width - 1, tile_height - 1), COMBINED_HEADER_FILL, -1)
    _draw_text(tile, caption["footer"], (12, tile_height - 8), 0.34, COMBINED_MUTED)


def _centred_block(tile: np.ndarray,
                   entries: list[tuple[str, float, tuple[int, int, int]]]) -> None:
    """Centre a mixed-size block of text lines inside the tile content area."""
    if not entries:
        return
    metrics = []
    for text, scale, colour in entries:
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1,
        )
        metrics.append((text, scale, colour, text_width, text_height,
                        max(text_height + 6, int(round(24 * scale / 0.45)))))
    area_top = COMBINED_HEADER_HEIGHT
    area_height = (tile.shape[0] - COMBINED_HEADER_HEIGHT -
                   COMBINED_FOOTER_HEIGHT)
    cursor = area_top + (area_height - sum(item[5] for item in metrics)) // 2
    for text, scale, colour, text_width, text_height, line_height in metrics:
        cursor += line_height
        left = max(6, (tile.shape[1] - text_width) // 2)
        baseline = cursor - (line_height - text_height) // 2
        cv2.rectangle(tile, (left - 6, baseline - text_height - 4),
                      (left + text_width + 6, baseline + 4), (12, 14, 16), -1)
        _draw_text(tile, text, (left, baseline), scale, colour)


def compose_tile(frame: np.ndarray, boxes: list[dict], camera_name: str,
                 global_person_id: str, track_ids: list[int],
                 timestamp_seconds: float, duration_seconds: float,
                 tile_size: tuple[int, int] = COMBINED_TILE_SIZE) -> np.ndarray:
    """Letterbox one source frame into a tile and draw every planned member box."""
    tile_width, tile_height = tile_size
    area_width = tile_width
    area_height = tile_height - COMBINED_HEADER_HEIGHT - COMBINED_FOOTER_HEIGHT
    tile = np.full((tile_height, tile_width, 3), COMBINED_PANEL, dtype=np.uint8)

    source_height, source_width = frame.shape[:2]
    scale = min(area_width / source_width, area_height / source_height)
    draw_width = max(1, int(round(source_width * scale)))
    draw_height = max(1, int(round(source_height * scale)))
    offset_x = (area_width - draw_width) // 2
    offset_y = COMBINED_HEADER_HEIGHT + (area_height - draw_height) // 2
    resized = cv2.resize(frame, (draw_width, draw_height),
                         interpolation=cv2.INTER_AREA)
    tile[offset_y:offset_y + draw_height, offset_x:offset_x + draw_width] = resized

    drawn = 0
    for box in boxes:
        if box["rect"] is None:
            continue
        x1, y1, x2, y2 = box["rect"]
        x1, x2 = max(0, min(x1, source_width - 1)), max(0, min(x2, source_width))
        y1, y2 = max(0, min(y1, source_height - 1)), max(0, min(y2, source_height))
        if x2 <= x1 or y2 <= y1:
            continue
        left = offset_x + int(round(x1 * scale))
        right = offset_x + int(round(x2 * scale))
        top = offset_y + int(round(y1 * scale))
        bottom = offset_y + int(round(y2 * scale))
        colour = COMBINED_OBSERVED if box["observed"] else COMBINED_PREDICTED
        cv2.rectangle(tile, (left, top), (right, bottom), colour, 2)
        label = f"ID {box['track_id']}"
        if box["state"]:
            label = f"{label} | {box['state']}"
        _draw_label(tile, label, left, max(COMBINED_HEADER_HEIGHT + 14, top - 6),
                    colour)
        drawn += 1

    if not drawn:
        if boxes:
            lines = [f"ID {box['track_id']}: {box['state']}" for box in boxes]
        else:
            lines = ["no confirmed local track for this global ID"]
        _centred_block(tile, [(line, 0.45, COMBINED_STATUS) for line in lines])

    _draw_chrome(tile, tile_caption(
        camera_name, global_person_id, track_ids,
        timestamp_seconds, duration_seconds,
    ))
    cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1),
                  COMBINED_BORDER, 1)
    return tile


def compose_end_card(camera_name: str, global_person_id: str,
                     track_ids: list[int], duration_seconds: float,
                     tile_size: tuple[int, int] = COMBINED_TILE_SIZE) -> np.ndarray:
    """Explicit end-of-footage tile; no frame is repeated to fake continuity."""
    tile_width, tile_height = tile_size
    tile = np.full((tile_height, tile_width, 3), COMBINED_HEADER_FILL,
                   dtype=np.uint8)
    lines = end_card_lines(camera_name, global_person_id, track_ids,
                           duration_seconds)
    _centred_block(
        tile,
        [(lines[0], 0.85, COMBINED_STATUS)] +
        [(line, 0.45, COMBINED_MUTED) for line in lines[1:]],
    )
    _draw_chrome(tile, tile_caption(
        camera_name, global_person_id, track_ids,
        duration_seconds, duration_seconds,
    ))
    cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1),
                  COMBINED_BORDER, 1)
    return tile


def compose_empty_tile(label: str = "No camera in this run",
                       tile_size: tuple[int, int] = COMBINED_TILE_SIZE) -> np.ndarray:
    """Neutral placeholder for an unused grid cell."""
    tile_width, tile_height = tile_size
    tile = np.full((tile_height, tile_width, 3), COMBINED_BACKGROUND,
                   dtype=np.uint8)
    _centred_block(tile, [(line, 0.5, COMBINED_MUTED)
                          for line in empty_slot_lines(label)])
    cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1),
                  COMBINED_BORDER, 1)
    return tile


def compose_grid(tiles: list[np.ndarray],
                 tile_size: tuple[int, int] = COMBINED_TILE_SIZE,
                 output_size: tuple[int, int] = COMBINED_OUTPUT_SIZE,
                 empty_label: str = "No camera in this run") -> np.ndarray:
    """Place every camera tile in a 2x2 grid, filling unused cells neutrally."""
    rows, columns = grid_shape(len(tiles))
    output_width, output_height = output_size
    cell_width = output_width // columns
    cell_height = output_height // rows
    canvas = np.full((output_height, output_width, 3), COMBINED_BACKGROUND,
                     dtype=np.uint8)
    for position in range(rows * columns):
        row, column = divmod(position, columns)
        if position < len(tiles):
            cell = tiles[position]
            if cell.shape[1] != cell_width or cell.shape[0] != cell_height:
                cell = cv2.resize(cell, (cell_width, cell_height),
                                  interpolation=cv2.INTER_AREA)
        else:
            cell = compose_empty_tile(empty_label, (cell_width, cell_height))
        top = row * cell_height
        left = column * cell_width
        canvas[top:top + cell_height, left:left + cell_width] = cell
    return canvas


def _read_frame_at(camera: dict, target_frame: int) -> np.ndarray | None:
    """Advance one capture forward to the requested source frame."""
    capture = camera["capture"]
    while camera["next_frame"] <= target_frame:
        ok, frame = capture.read()
        if not ok:
            camera["ended"] = True
            return None
        camera["next_frame"] += 1
        camera["last_frame"] = frame
    return camera["last_frame"]


def _source_fingerprint(result: dict) -> list[int] | None:
    try:
        stat = Path(result["source_path"]).stat()
    except OSError:
        return None
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def combined_signature(results: list[dict], run_dir: Path | str,
                       global_person_id: str,
                       output_fps: float = COMBINED_OUTPUT_FPS,
                       tile_size: tuple[int, int] = COMBINED_TILE_SIZE) -> dict:
    """Describe the registry state and sources a combined video was built from."""
    registry_path = Path(run_dir) / "global_identity_registry.json"
    members = identity_members_all(registry_path, str(global_person_id))
    return {
        "renderer": COMBINED_SIGNATURE_VERSION,
        "global_person_id": str(global_person_id),
        "output_fps": round(float(output_fps), 3),
        "tile": [int(tile_size[0]), int(tile_size[1])],
        "members": {
            camera_id: [
                {
                    "track_id": int(member["track_id"]),
                    "status": str(member.get("status", "")),
                    "similarity": round(float(member.get("similarity", 0.0) or 0.0), 6),
                }
                for member in camera_members
            ]
            for camera_id, camera_members in sorted(members.items())
        },
        "sources": {
            result["camera_id"]: _source_fingerprint(result)
            for result in sorted(results, key=lambda item: item["camera_id"])
        },
    }


def combined_video_status(results: list[dict], run_dir: Path | str,
                          global_person_id: str,
                          output_fps: float = COMBINED_OUTPUT_FPS,
                          tile_size: tuple[int, int] = COMBINED_TILE_SIZE) -> dict | None:
    """Return the saved combined manifest only when it still matches this run."""
    output_dir = Path(run_dir) / "global_person_videos" / str(global_person_id)
    manifest_path = output_dir / "combined_video_manifest.json"
    signature_path = output_dir / "manifest_signature.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or stored != combined_signature(
        results, run_dir, global_person_id, output_fps, tile_size,
    ):
        return None
    try:
        video_path = Path(manifest["video_path"])
    except (KeyError, TypeError):
        return None
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return None
    return manifest


def _camera_source(result: dict,
                   members_by_camera: dict[str, list[dict]]) -> dict:
    """Open one source video and index only this identity's local tracks."""
    camera_id = result["camera_id"]
    label = _camera_name(result)
    source = Path(result["source_path"])
    if not source.is_file():
        raise ValueError(f"The source video for {label} is missing: {source.name}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"The source video for {label} could not be opened.")
    try:
        camera_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(camera_fps) or camera_fps <= 0:
            camera_fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError(
                f"The source video for {label} reports invalid frame dimensions."
            )
        if frame_count <= 0:
            raise ValueError(
                f"The source video for {label} does not report a readable frame count."
            )
        members = sorted(members_by_camera.get(camera_id, []),
                         key=lambda item: int(item["track_id"]))
        track_ids = [int(member["track_id"]) for member in members]
        return {
            "result": result,
            "camera_id": camera_id,
            "label": label,
            "capture": capture,
            "fps": camera_fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration": frame_count / camera_fps,
            "members": members,
            "track_ids": track_ids,
            "index": load_track_index(result["tracks_file"], track_ids),
            "window": prediction_window(_read_json(result["track_summary_file"])),
            "next_frame": 0,
            "last_frame": None,
            "ended": False,
        }
    except Exception:
        capture.release()
        raise


def render_combined_global_person_video(
    results: list[dict],
    run_dir: Path | str,
    global_person_id: str,
    output_fps: float = COMBINED_OUTPUT_FPS,
    tile_size: tuple[int, int] = COMBINED_TILE_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Render one combined multi-camera MP4 for every member of a global ID.

    The registry and the hypotheses are only read. Each tile is sampled by
    elapsed time from its own source video, so the timeline never claims that
    frames from different cameras show the same real-world moment.
    """
    root = Path(run_dir)
    global_id = str(global_person_id or "").strip()
    if not global_id:
        raise ValueError("Choose a global person ID before building the combined video.")
    fps = float(output_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Combined video FPS must be a positive number.")
    members_by_camera = identity_members_all(
        root / "global_identity_registry.json", global_id,
    )
    if not members_by_camera:
        raise ValueError(
            f"{global_id} is not a confirmed global identity in this run. "
            "Confirm a same-person pair before building the combined video."
        )
    if not results:
        raise ValueError("No processed camera is available for the combined video.")
    rows_count, columns = grid_shape(len(results))

    cameras = []
    try:
        for result in results:
            cameras.append(_camera_source(result, members_by_camera))
    except Exception:
        for camera in cameras:
            camera["capture"].release()
        raise

    output_dir = root / "global_person_videos" / global_id
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{global_id}_combined_multi_camera.mp4"
    partial = destination.with_name(destination.stem + ".partial.mp4")
    manifest_path = output_dir / "combined_video_manifest.json"
    signature_path = output_dir / "manifest_signature.json"
    assignments_path = output_dir / "combined_video_assignments.csv"
    partial.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    signature_path.unlink(missing_ok=True)

    total_duration = max(camera["duration"] for camera in cameras)
    total_frames = max(1, int(math.ceil(total_duration * fps - 1e-9)))

    def release_captures() -> None:
        for camera in cameras:
            camera["capture"].release()

    writer = cv2.VideoWriter(
        str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, COMBINED_OUTPUT_SIZE,
    )
    if not writer.isOpened():
        release_captures()
        raise OSError(
            "OpenCV could not create the combined video. "
            "Check the MP4 codec and disk space."
        )
    written = 0
    try:
        for frame_index in range(total_frames):
            timestamp = frame_index / fps
            tiles = []
            for camera in cameras:
                target = int(round(timestamp * camera["fps"]))
                frame = (None if target >= camera["frame_count"]
                         else _read_frame_at(camera, target))
                if frame is None:
                    tiles.append(compose_end_card(
                        camera["label"], global_id, camera["track_ids"],
                        camera["duration"], tile_size,
                    ))
                    continue
                boxes = plan_frame_boxes(
                    camera["members"], camera["index"], target, camera["window"],
                )
                tiles.append(compose_tile(
                    frame, boxes, camera["label"], global_id,
                    camera["track_ids"], target / camera["fps"],
                    camera["duration"], tile_size,
                ))
            writer.write(compose_grid(tiles, tile_size=tile_size))
            written += 1
            if progress_callback:
                progress_callback(written, total_frames)
    except Exception:
        writer.release()
        release_captures()
        partial.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
        release_captures()

    if not partial.is_file() or partial.stat().st_size == 0 or written == 0:
        partial.unlink(missing_ok=True)
        raise OSError("The combined video could not be written.")
    os.replace(partial, destination)

    validation = cv2.VideoCapture(str(destination))
    try:
        if not validation.isOpened():
            raise OSError("The combined video is not a playable MP4.")
        output_width = int(validation.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(validation.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_fps = float(validation.get(cv2.CAP_PROP_FPS))
        output_frames = int(validation.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = validation.read()
        if (not ok or output_width != COMBINED_OUTPUT_SIZE[0] or
                output_height != COMBINED_OUTPUT_SIZE[1] or
                output_frames != written or not np.isfinite(output_fps) or
                abs(output_fps - fps) > max(0.1, fps * 0.01)):
            raise OSError(
                "The combined video failed playback validation "
                f"(expected {COMBINED_OUTPUT_SIZE[0]}x{COMBINED_OUTPUT_SIZE[1]} "
                f"at {fps:.3f} FPS, {written} frames; got {output_width}x"
                f"{output_height} at {output_fps:.3f} FPS, {output_frames} frames)."
            )
    except Exception:
        validation.release()
        destination.unlink(missing_ok=True)
        raise
    finally:
        validation.release()

    camera_rows = []
    for position, camera in enumerate(cameras):
        row, column = divmod(position, columns)
        camera_rows.append({
            "global_person_id": global_id,
            "camera_id": camera["camera_id"],
            "camera_name": camera["label"],
            "source_name": camera["result"]["source_name"],
            "source_width": camera["width"],
            "source_height": camera["height"],
            "source_fps": round(camera["fps"], 3),
            "source_frame_count": camera["frame_count"],
            "source_duration_seconds": round(camera["duration"], 3),
            "local_track_ids": ", ".join(str(value) for value in camera["track_ids"]),
            "member_count": len(camera["track_ids"]),
            "tile_row": row,
            "tile_column": column,
        })

    manifest = {
        "global_person_id": global_id,
        "video_filename": destination.name,
        "video_path": str(destination),
        "width": COMBINED_OUTPUT_SIZE[0],
        "height": COMBINED_OUTPUT_SIZE[1],
        "fps": round(fps, 3),
        "frame_count": written,
        "duration_seconds": round(written / fps, 3),
        "grid": {"rows": rows_count, "columns": columns},
        "tile_width": int(tile_size[0]),
        "tile_height": int(tile_size[1]),
        "synchronisation": COMBINED_SYNC_NOTE,
        "cameras": camera_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    signature_path.write_text(json.dumps(
        combined_signature(results, root, global_id, fps, tile_size),
        indent=2, sort_keys=True,
    ), encoding="utf-8")

    fields = ["global_person_id", "camera_id", "camera_name", "source_name",
              "source_width", "source_height", "source_fps",
              "source_frame_count", "source_duration_seconds",
              "local_track_ids", "member_count", "tile_row", "tile_column"]
    with assignments_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(
            [{key: row.get(key, "") for key in fields} for row in camera_rows]
        )
    return manifest
