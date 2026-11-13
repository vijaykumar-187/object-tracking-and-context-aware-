"""Track fingerprints from the YOLO feature baseline or trained OSNet model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 1e-12:
        return None
    return vector / norm


AGGREGATION_METHODS = ("mean", "quality_weighted_mean", "medoid")


def aggregate_track_embeddings(vectors: list[np.ndarray],
                               quality_scores: list[float],
                               method: str) -> np.ndarray | None:
    """Pool normalized crop embeddings without losing unit-length consistency."""
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown track embedding aggregation: {method}")
    normalized = [_unit(vector) for vector in vectors]
    samples = [vector for vector in normalized if vector is not None]
    if not samples:
        return None
    matrix = np.stack(samples)
    if method == "medoid":
        similarities = matrix @ matrix.T
        return matrix[int(np.argmax(similarities.mean(axis=1)))].copy()
    if method == "mean":
        return _unit(matrix.mean(axis=0))
    scores = np.asarray(quality_scores[:len(samples)], dtype=np.float32)
    if len(scores) != len(samples) or not np.isfinite(scores).all():
        scores = np.ones(len(samples), dtype=np.float32)
    weights = np.maximum(scores, 0.05)
    weights /= weights.sum()
    return _unit(np.sum(matrix * weights[:, None], axis=0))


def make_track_fingerprint_variants(
    model: object,
    crop_rows: list[dict],
    batch_size: int = 32,
    feature_kind: str = "yolo_baseline",
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict]]]:
    """Encode each retained crop once and build three comparable track pools.

    Returns per-method fingerprints and normalized crop embeddings with their
    source frame/quality metadata for reproducibility and inspection.
    """
    if batch_size < 1:
        raise ValueError("Embedding batch size must be at least one.")
    grouped: dict[str, list[dict]] = {}
    for row in crop_rows:
        key = f"{row['camera_id']}:{int(row['track_id'])}"
        grouped.setdefault(key, []).append(row)
    fingerprints = {method: {} for method in AGGREGATION_METHODS}
    crop_embeddings: dict[str, list[dict]] = {}
    for key, all_rows in grouped.items():
        rows = sorted(all_rows, key=lambda row: (
            -float(row.get("quality_score", 0.0)), int(row.get("frame_number", 0)),
        ))[:8]
        vectors: list[np.ndarray] = []
        metadata: list[dict] = []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = [cv2.imread(str(row["path"])) for row in batch_rows]
            valid = [(row, image) for row, image in zip(batch_rows, images)
                     if image is not None and image.size]
            if not valid:
                continue
            valid_rows, valid_images = zip(*valid)
            if feature_kind == "osnet":
                embedded = model.embed_crops(list(valid_images))
            elif feature_kind == "yolo_baseline":
                embedded = model.embed(source=list(valid_images), device="cpu",
                                        imgsz=256, verbose=False)
            else:
                raise ValueError(f"Unknown appearance feature kind: {feature_kind}")
            for row, value in zip(valid_rows, embedded):
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                vector = _unit(np.asarray(value))
                if vector is None:
                    continue
                vectors.append(vector)
                metadata.append({
                    "frame_number": int(row.get("frame_number", 0)),
                    "timestamp_seconds": float(row.get("timestamp_seconds", 0.0)),
                    "quality_score": float(row.get("quality_score", 0.0)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "sharpness": float(row.get("sharpness", 0.0)),
                    "width": int(row.get("width", 0)),
                    "height": int(row.get("height", 0)),
                    "embedding": vector.astype(float).tolist(),
                })
        if vectors:
            crop_embeddings[key] = metadata
            scores = [item["quality_score"] for item in metadata]
            for method in AGGREGATION_METHODS:
                vector = aggregate_track_embeddings(vectors, scores, method)
                if vector is not None:
                    fingerprints[method][key] = vector
    return fingerprints, crop_embeddings


def make_track_fingerprints(model: object, crop_rows: list[dict],
                            batch_size: int = 32,
                            feature_kind: str = "yolo_baseline") -> dict[str, np.ndarray]:
    """Backward-compatible normalized mean fingerprint builder."""
    variants, _ = make_track_fingerprint_variants(
        model, crop_rows, batch_size=batch_size, feature_kind=feature_kind,
    )
    return variants["mean"]


def save_crop_embeddings(path: Path | str, crop_embeddings: dict[str, list[dict]],
                         model_metadata: dict | None = None) -> None:
    """Save the exact representative crop vectors used to make track pools."""
    payload = {"crop_embeddings": crop_embeddings}
    if model_metadata:
        payload.update(model_metadata)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def save_fingerprints(path: Path | str,
                      fingerprints: dict[str, np.ndarray],
                      model_metadata: dict | None = None) -> None:
    payload = {
        "model_label": "YOLO26n pretrained detector-feature similarity baseline",
        "is_person_reid_model": False,
        "feature_dimension": next(iter(fingerprints.values())).size if fingerprints else None,
        "preprocessing": "YOLO embed(source=crops, device=cpu, imgsz=256)",
        "fingerprints": {key: value.astype(float).tolist()
                         for key, value in fingerprints.items()},
    }
    if model_metadata:
        payload.update(model_metadata)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_fingerprints(path: Path | str,
                      expected_model: str | None = None) -> dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_model and data.get("model_id") != expected_model:
        raise ValueError("Fingerprint model metadata does not match the requested extractor.")
    result: dict[str, np.ndarray] = {}
    for key, value in data.get("fingerprints", {}).items():
        vector = _unit(np.asarray(value, dtype=np.float32))
        if vector is not None:
            result[key] = vector
    return result


def rank_candidates(query: np.ndarray | None,
                    candidates: dict[str, np.ndarray]) -> list[dict]:
    if query is None:
        return []
    query = _unit(query)
    if query is None:
        return []
    ranked = []
    for key, candidate in candidates.items():
        candidate = _unit(candidate)
        if candidate is None or candidate.shape != query.shape:
            continue
        ranked.append({"key": key, "similarity": float(np.dot(query, candidate))})
    return sorted(ranked, key=lambda item: item["similarity"], reverse=True)


def rank_across_cameras(query_key: str, fingerprints: dict[str, np.ndarray],
                        camera_ids: Iterable[str] | None = None,
                        threshold: float | None = None) -> dict[str, list[dict]]:
    """Rank one camera-local track independently against every other camera."""
    query_camera = query_key.split(":", 1)[0]
    allowed = set(camera_ids) if camera_ids is not None else None
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, vector in fingerprints.items():
        camera_id = key.split(":", 1)[0]
        if camera_id == query_camera or (allowed is not None and camera_id not in allowed):
            continue
        grouped.setdefault(camera_id, {})[key] = vector
    target_ids = (allowed - {query_camera}) if allowed is not None else set(grouped)
    result = {}
    for camera_id in sorted(target_ids):
        candidates = grouped.get(camera_id, {})
        ranked = rank_candidates(fingerprints.get(query_key), candidates)
        if threshold is not None:
            ranked = [item for item in ranked if item["similarity"] >= threshold]
        result[camera_id] = ranked
    return result


def evaluate_pairs(pairs: Iterable[dict],
                   fingerprints: dict[str, np.ndarray],
                   threshold: float = 0.50) -> dict:
    """Evaluate each labelled query/destination-camera pair independently.

    Accepts the dynamic schema query_camera_id/query_track_id/
    candidate_camera_id/candidate_track_id, plus the historical two-camera
    camera1_track_id/camera2_track_id schema.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("The no-match threshold must be between -1 and 1.")
    grouped_labels: dict[tuple[str, int, str], set[str]] = {}
    no_match_labels: set[tuple[str, int, str]] = set()
    for pair in pairs:
        if "query_camera_id" in pair:
            source_camera = str(pair["query_camera_id"])
            source_track = int(pair["query_track_id"])
            target_camera = str(pair["candidate_camera_id"])
            raw_target = str(pair["candidate_track_id"]).strip()
            if raw_target.upper() in {"NOT_IN_CAMERA2", "NO_MATCH"}:
                target_track = None
            else:
                target_track = int(raw_target)
        else:
            source_camera, target_camera = "camera1", "camera2"
            source_track = int(pair["camera1_track_id"])
            target_track = int(pair["camera2_track_id"])
        if source_camera == target_camera:
            continue
        label_key = (source_camera, source_track, target_camera)
        if target_track is None:
            no_match_labels.add(label_key)
        else:
            grouped_labels.setdefault(label_key, set()).add(f"{target_camera}:{target_track}")

    if no_match_labels & set(grouped_labels):
        raise ValueError("A query cannot be labelled both as a match and NOT_IN_CAMERA2.")
    all_queries = set(grouped_labels) | no_match_labels
    evaluated = top1 = top3 = top5 = missing = 0
    reciprocal_rank_sum = average_precision_sum = 0.0
    positive_queries = correct_positive = false_matches = missed_matches = 0
    no_match_count = no_match_correct = 0
    for (source_camera, source_track, target_camera) in all_queries:
        query_key = f"{source_camera}:{source_track}"
        candidates = {key: value for key, value in fingerprints.items()
                      if key.startswith(f"{target_camera}:")}
        ranked = rank_candidates(fingerprints.get(query_key), candidates)
        has_match = (source_camera, source_track, target_camera) in grouped_labels
        if ranked and ranked[0]["similarity"] >= threshold:
            predicted_key = ranked[0]["key"]
            if has_match and predicted_key in grouped_labels[(source_camera, source_track, target_camera)]:
                correct_positive += 1
            else:
                false_matches += 1
                if has_match:
                    missed_matches += 1
        elif has_match:
            missed_matches += 1

        if not has_match:
            no_match_count += 1
            no_match_correct += int(not ranked or ranked[0]["similarity"] < threshold)
            continue

        positive_queries += 1
        expected_keys = grouped_labels[(source_camera, source_track, target_camera)]
        relevant = expected_keys & set(candidates)
        if not ranked or not relevant:
            missing += 1
            continue
        evaluated += 1
        rank = next((index for index, item in enumerate(ranked, start=1)
                     if item["key"] in expected_keys), None)
        top1 += int(rank == 1)
        top3 += int(rank is not None and rank <= 3)
        top5 += int(rank is not None and rank <= 5)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
        found = 0
        precision_sum = 0.0
        for index, item in enumerate(ranked, start=1):
            if item["key"] in relevant:
                found += 1
                precision_sum += found / index
        average_precision_sum += precision_sum / len(relevant)
    return {
        "evaluated": evaluated,
        "missing_fingerprints": missing,
        "top1_correct": top1,
        "top3_correct": top3,
        "top5_correct": top5,
        "top1_accuracy": top1 / evaluated if evaluated else None,
        "top3_accuracy": top3 / evaluated if evaluated else None,
        "top5_accuracy": top5 / evaluated if evaluated else None,
        "mean_average_precision": average_precision_sum / evaluated if evaluated else None,
        "mrr": reciprocal_rank_sum / evaluated if evaluated else None,
        "positive_queries": positive_queries,
        "no_match_queries": no_match_count,
        "no_match_correct": no_match_correct,
        "no_match_accuracy": no_match_correct / no_match_count if no_match_count else None,
        "false_matches": false_matches,
        "missed_matches": missed_matches,
        "precision": correct_positive / (correct_positive + false_matches)
        if correct_positive + false_matches else None,
        "recall": correct_positive / positive_queries if positive_queries else None,
        "threshold": threshold,
    }


def validate_ground_truth_pairs(rows: list[dict],
                                camera1_track_ids: set[int] | dict[str, set[int]],
                                camera2_track_ids: set[int] | None = None) -> list[dict]:
    """Validate camera-local identity pairs in dynamic or legacy CSV format."""
    dynamic = isinstance(camera1_track_ids, dict)
    required = ({"query_camera_id", "query_track_id", "candidate_camera_id", "candidate_track_id"}
                if dynamic else {"camera1_track_id", "camera2_track_id"})
    validated: list[dict] = []
    seen: set[tuple[str, int, str, int | str]] = set()
    for line_number, row in enumerate(rows, start=2):
        if not required.issubset(row) or any(row.get(key) in (None, "") for key in required):
            raise ValueError(f"CSV row {line_number} is missing a camera track ID.")
        try:
            if dynamic:
                source_camera, target_camera = str(row["query_camera_id"]).strip(), str(row["candidate_camera_id"]).strip()
                source_id = int(row["query_track_id"])
                raw_target = str(row["candidate_track_id"]).strip()
                target_id = ("NOT_IN_CAMERA2" if raw_target.upper() in
                             {"NOT_IN_CAMERA2", "NO_MATCH"} else int(raw_target))
            else:
                source_camera, target_camera = "camera1", "camera2"
                source_id, target_id = int(row["camera1_track_id"]), int(row["camera2_track_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV row {line_number} must contain integer track IDs.") from exc
        no_match = target_id == "NOT_IN_CAMERA2"
        if (source_id <= 0 or (not no_match and int(target_id) <= 0) or
                not source_camera or not target_camera or source_camera == target_camera):
            raise ValueError(f"CSV row {line_number} track IDs must be positive integers.")
        pair = (source_camera, source_id, target_camera, target_id)
        if pair in seen:
            raise ValueError(f"CSV contains duplicate pair {source_camera} ID {source_id} → {target_camera} ID {target_id}.")
        if dynamic:
            source_set = camera1_track_ids.get(source_camera, set())
            target_set = camera1_track_ids.get(target_camera, set())
        else:
            source_set, target_set = camera1_track_ids, camera2_track_ids or set()
        if source_id not in source_set or (not no_match and target_id not in target_set):
            raise ValueError(
                f"CSV row {line_number} references a track ID that does not exist in the current run."
            )
        seen.add(pair)
        validated.append({"query_camera_id": source_camera, "query_track_id": source_id,
                          "candidate_camera_id": target_camera, "candidate_track_id": target_id})
    if not validated:
        raise ValueError("The ground-truth CSV contains no identity pairs.")
    return validated

