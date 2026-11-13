"""Two-camera OSNet fingerprint comparison and match-table export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.reid.appearance import rank_candidates


def best_crop_by_track(crop_rows: list[dict]) -> dict[str, str]:
    """Return each camera-local track's highest-quality retained crop path."""
    best: dict[str, dict] = {}
    for row in crop_rows:
        key = f"{row['camera_id']}_t{int(row['track_id']):04d}"
        current = best.get(key)
        if current is None or float(row.get("quality_score", 0.0)) > float(current.get("quality_score", 0.0)):
            best[key] = row
    return {key: str(row.get("path", row.get("crop_filename", "")))
            for key, row in best.items()}


def match_two_cameras(
    fingerprints: dict[str, np.ndarray],
    crop_rows: list[dict],
    camera1: str = "camera1",
    camera2: str = "camera2",
    top_k: int = 3,
) -> list[dict]:
    """Rank every camera1 track against camera2 tracks, one-way as Review 3 defines."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    crops = best_crop_by_track(crop_rows)
    queries = {key: value for key, value in fingerprints.items()
               if key.startswith(f"{camera1}:")}
    gallery = {key: value for key, value in fingerprints.items()
               if key.startswith(f"{camera2}:")}
    rows: list[dict] = []
    for query_key in sorted(queries, key=lambda key: int(key.rsplit(":", 1)[1])):
        ranked = rank_candidates(queries[query_key], gallery)[:top_k]
        query_track_key = f"{camera1}_t{int(query_key.rsplit(':', 1)[1]):04d}"
        if not ranked:
            rows.append({
                "query_track_key": query_track_key, "rank": 0,
                "candidate_track_key": "", "score": "",
                "query_best_crop": crops.get(query_track_key, ""),
                "candidate_best_crop": "",
            })
            continue
        for rank, item in enumerate(ranked, start=1):
            candidate_id = int(item["key"].rsplit(":", 1)[1])
            candidate_track_key = f"{camera2}_t{candidate_id:04d}"
            rows.append({
                "query_track_key": query_track_key,
                "rank": rank,
                "candidate_track_key": candidate_track_key,
                "score": round(float(item["similarity"]), 6),
                "query_best_crop": crops.get(query_track_key, ""),
                "candidate_best_crop": crops.get(candidate_track_key, ""),
            })
    return rows


def write_matches_csv(path: Path | str, rows: list[dict]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_track_key", "rank", "candidate_track_key", "score",
              "query_best_crop", "candidate_best_crop"]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination

