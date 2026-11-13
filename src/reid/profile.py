"""Versioned, view-aware appearance profiles with conservative updates.

Orientation is deliberately uncertain unless supplied by a validated estimator.
Motion direction is never used as a proxy for body facing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import cv2
import numpy as np

PROFILE_VERSION = 1
VIEW_CATEGORIES = ("front", "back", "left_side", "right_side", "partial", "uncertain")


def _unit(value: np.ndarray) -> np.ndarray | None:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 and np.isfinite(value).all() else None


def describe_crop(image: np.ndarray) -> dict:
    """Extract inspectable coarse clothing color/texture descriptors (BGR input)."""
    if image is None or image.size == 0:
        return {"available": False}
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def region(y0: float, y1: float) -> dict:
        y, end = int(height * y0), max(int(height * y0) + 1, int(height * y1))
        crop = hsv[y:min(height, end), :]
        hist = cv2.calcHist([crop], [0, 1], None, [12, 4], [0, 180, 0, 256])
        hist = hist.reshape(-1).astype(float)
        hist /= max(float(hist.sum()), 1.0)
        return {"hsv_histogram": hist.tolist()}
    edges = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "available": True,
        "upper_color": region(0.08, 0.55),
        "lower_color": region(0.50, 0.98),
        "texture": {"laplacian_variance": float(edges),
                    "gray_mean": float(gray.mean() / 255.0)},
        "shape": {"box_aspect_ratio": float(width / max(height, 1))},
        "accessories": {"status": "not_automatically_detected"},
    }


@dataclass
class AppearanceObservation:
    embedding: list[float] | None
    descriptors: dict
    quality: float
    detection_confidence: float
    frame_number: int
    view_category: str = "uncertain"
    view_confidence: float = 0.0
    trusted: bool = False


@dataclass
class TrackAppearanceProfile:
    camera_id: str
    track_id: int
    trusted: list[AppearanceObservation] = field(default_factory=list)
    pending: list[AppearanceObservation] = field(default_factory=list)

    def add(self, item: AppearanceObservation, *, continuity: bool,
            appearance_consistent: bool, min_quality: float = 0.45) -> bool:
        """Promote only after consecutive reliable associations; otherwise pend."""
        safe = (continuity and appearance_consistent and item.quality >= min_quality
                and item.detection_confidence >= 0.25)
        item.trusted = safe
        target = self.trusted if safe else self.pending
        target.append(item)
        # Bound memory and retain diverse examples rather than unbounded frame data.
        if len(target) > 24:
            del target[:-24]
        return safe

    def references(self, view: str | None = None) -> list[AppearanceObservation]:
        rows = self.trusted
        if view and view != "uncertain":
            specific = [row for row in rows if row.view_category == view]
            if specific:
                return specific
        return rows


def embedding_similarity(vector: np.ndarray | None,
                         references: list[AppearanceObservation],
                         view: str | None = None) -> float | None:
    if vector is None:
        return None
    query = _unit(vector)
    if query is None:
        return None
    candidates = []
    for row in references:
        if row.embedding is None:
            continue
        ref = _unit(np.asarray(row.embedding))
        if ref is not None and ref.shape == query.shape:
            candidates.append(float(np.dot(query, ref)))
    return max(candidates) if candidates else None


def save_profiles(path: Path | str, profiles: dict[str, TrackAppearanceProfile]) -> None:
    payload = {"version": PROFILE_VERSION, "profiles": {
        key: {"camera_id": profile.camera_id, "track_id": profile.track_id,
              "trusted": [asdict(row) for row in profile.trusted],
              "pending": [asdict(row) for row in profile.pending]}
        for key, profile in profiles.items()}}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_profiles(path: Path | str) -> dict[str, TrackAppearanceProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != PROFILE_VERSION:
        raise ValueError("Unsupported appearance profile version.")
    result = {}
    for key, row in payload.get("profiles", {}).items():
        result[key] = TrackAppearanceProfile(
            row["camera_id"], int(row["track_id"]),
            [AppearanceObservation(**item) for item in row.get("trusted", [])],
            [AppearanceObservation(**item) for item in row.get("pending", [])])
    return result
