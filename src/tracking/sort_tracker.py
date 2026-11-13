"""A small SORT-style tracker using Kalman prediction and Hungarian matching.

Track IDs are local to one video. The tracker does not use clothing or appearance
features, and it cannot recover a person after the configured lost-track window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return minimum-cost row/column pairs using the O(n^3) Hungarian method."""
    if cost.size == 0:
        return []
    rows, cols = cost.shape
    transposed = rows > cols
    matrix = cost.T if transposed else cost
    n, m = matrix.shape
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j] - 1), j - 1) for j in range(1, m + 1) if p[j] != 0]
    return [(col, row) for row, col in pairs] if transposed else pairs


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    age: int
    hits: int
    time_since_update: int
    last_frame: int
    confidence: float

    @staticmethod
    def _measurement(box: Box) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2,
                         max(2, x2 - x1), max(2, y2 - y1)], dtype=np.float64)

    @classmethod
    def create(cls, track_id: int, box: Box, confidence: float, frame_index: int) -> "Track":
        state = np.zeros(8, dtype=np.float64)
        state[:4] = cls._measurement(box)
        covariance = np.diag([16, 16, 16, 16, 100, 100, 100, 100]).astype(np.float64)
        return cls(track_id, state, covariance, 1, 1, 0, frame_index, confidence)

    def predict(self) -> None:
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = transition[1, 5] = 1.0
        transition[2, 6] = transition[3, 7] = 1.0
        process_noise = np.diag([2, 2, 2, 2, 8, 8, 8, 8]).astype(np.float64)
        self.state = transition @ self.state
        self.state[2:4] = np.maximum(self.state[2:4], 2.0)
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.age += 1
        self.time_since_update += 1

    def correct(self, box: Box, confidence: float, frame_index: int) -> None:
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[:4, :4] = np.eye(4, dtype=np.float64)
        noise = np.diag([9, 9, 16, 16]).astype(np.float64)
        measurement = self._measurement(box)
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + noise
        gain = np.linalg.solve(innovation_covariance.T,
                               (self.covariance @ observation.T).T).T
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(8) - gain @ observation) @ self.covariance
        self.state[2:4] = np.maximum(self.state[2:4], 2.0)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_index
        self.confidence = confidence

    def box(self) -> Box:
        cx, cy, width, height = self.state[:4]
        return (int(round(cx - width / 2)), int(round(cy - height / 2)),
                int(round(cx + width / 2)), int(round(cy + height / 2)))


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    box: Box
    confidence: float
    observed: bool
    recovered: bool
    detection_index: int | None
    age: int
    time_since_update: int


class SortTracker:
    """Maintain local track IDs through movement and short missed-detection gaps."""

    def __init__(self, max_age: int = 30, min_iou: float = 0.01,
                 max_center_distance: float = 2.5) -> None:
        self.max_age = max_age
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.tracks: list[Track] = []
        self.next_id = 1

    @staticmethod
    def _center_distance(track_box: Box, detection_box: Box) -> float:
        tx, ty = (track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2
        dx, dy = (detection_box[0] + detection_box[2]) / 2, (detection_box[1] + detection_box[3]) / 2
        scale = max(10.0, ((track_box[3] - track_box[1]) +
                           (detection_box[3] - detection_box[1])) / 2)
        return float(np.hypot(tx - dx, ty - dy) / scale)

    def update(self, detections: list[dict], frame_index: int) -> list[TrackObservation]:
        for track in self.tracks:
            track.predict()
        boxes = [tuple(int(v) for v in det["box"]) for det in detections]
        scores = [float(det.get("confidence", 0.0)) for det in detections]
        cost = np.full((len(self.tracks), len(boxes)), 1e6, dtype=np.float64)
        track_boxes = [track.box() for track in self.tracks]
        for row, track_box in enumerate(track_boxes):
            for col, detection_box in enumerate(boxes):
                overlap = _iou(track_box, detection_box)
                distance = self._center_distance(track_box, detection_box)
                if overlap >= self.min_iou or distance <= self.max_center_distance:
                    cost[row, col] = 0.75 * (1.0 - overlap) + 0.25 * min(distance / 3.0, 1.0)

        matched_detections: set[int] = set()
        observed_by_track: dict[int, int] = {}
        recovered_tracks: set[int] = set()
        for row, col in _linear_assignment(cost):
            if cost[row, col] >= 0.95:
                continue
            track = self.tracks[row]
            if track.time_since_update > 1:
                recovered_tracks.add(track.track_id)
            track.correct(boxes[col], scores[col], frame_index)
            matched_detections.add(col)
            observed_by_track[track.track_id] = col

        for index, (box, score) in enumerate(zip(boxes, scores)):
            if index not in matched_detections:
                track = Track.create(self.next_id, box, score, frame_index)
                observed_by_track[track.track_id] = index
                self.tracks.append(track)
                self.next_id += 1

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        observations = []
        for track in self.tracks:
            box = track.box()
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            observations.append(TrackObservation(
                track_id=track.track_id,
                box=box,
                confidence=track.confidence,
                observed=track.track_id in observed_by_track,
                recovered=track.track_id in recovered_tracks,
                detection_index=observed_by_track.get(track.track_id),
                age=track.age,
                time_since_update=track.time_since_update,
            ))
        return observations

