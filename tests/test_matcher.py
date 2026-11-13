"""Tests for the Review 3 two-camera top-three match export."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.reid.matcher import match_two_cameras, write_matches_csv


class TestTwoCameraMatcher(unittest.TestCase):
    def test_top_three_ranked_by_similarity_with_camera_local_crop_names(self):
        vectors = {
            "camera1:7": np.array([1.0, 0.0]),
            "camera2:3": np.array([0.9, 0.1]),
            "camera2:4": np.array([0.0, 1.0]),
            "camera2:5": np.array([0.8, 0.2]),
            "camera2:6": np.array([0.6, 0.4]),
        }
        crops = [
            {"camera_id": "camera1", "track_id": 7, "quality_score": 0.8, "path": "q.jpg"},
            {"camera_id": "camera2", "track_id": 3, "quality_score": 0.9, "path": "a.jpg"},
        ]
        rows = match_two_cameras(vectors, crops)
        self.assertEqual([row["candidate_track_key"] for row in rows],
                         ["camera2_t0003", "camera2_t0005", "camera2_t0006"])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["query_best_crop"], "q.jpg")
        self.assertEqual(rows[0]["candidate_best_crop"], "a.jpg")

    def test_empty_gallery_and_csv_export(self):
        rows = match_two_cameras({"camera1:7": np.array([1.0, 0.0])}, [])
        self.assertEqual(rows[0]["rank"], 0)
        self.assertEqual(rows[0]["candidate_track_key"], "")
        with tempfile.TemporaryDirectory() as folder:
            path = write_matches_csv(Path(folder) / "matches.csv", rows)
            with path.open(newline="", encoding="utf-8") as stream:
                data = list(csv.DictReader(stream))
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["rank"], "0")
            self.assertEqual(data[0]["candidate_track_key"], "")


if __name__ == "__main__":
    unittest.main()
