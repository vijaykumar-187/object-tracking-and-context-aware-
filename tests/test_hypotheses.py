"""Tests for manually reviewed cross-camera identity hypotheses."""

import tempfile
import unittest
from pathlib import Path

from src.reid.hypotheses import (
    build_global_identities, load_hypotheses, save_hypothesis,
)


class TestCrossCameraHypotheses(unittest.TestCase):
    def test_human_decision_is_saved_and_reloaded_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run" / "hypotheses.json"
            record = save_hypothesis(path, 7, 8, 0.91, "OSNet", "SAME PERSON")
            loaded = load_hypotheses(path)
        self.assertEqual(record["status"], "SAME PERSON")
        self.assertEqual(loaded["camera1:7|camera2:8"]["model_label"], "OSNet")
        self.assertEqual(loaded["camera1:7|camera2:8"]["similarity"], 0.91)

    def test_invalid_status_and_score_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hypotheses.json"
            with self.assertRaisesRegex(ValueError, "Status must be"):
                save_hypothesis(path, 7, 8, 0.5, "OSNet", "CONFIRMED")
            with self.assertRaisesRegex(ValueError, "between -1 and 1"):
                save_hypothesis(path, 7, 8, 1.5, "OSNet", "UNCERTAIN")

    def test_dynamic_camera_pair_is_stored_with_its_own_ids_and_names(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hypotheses.json"
            save_hypothesis(path, 3, 9, 0.82, "OSNet", "SAME PERSON",
                            "camera3", "camera5", "Lobby", "Exit")
            loaded = load_hypotheses(path)
        row = loaded["camera3:3|camera5:9"]
        self.assertEqual(row["camera1_name"], "Lobby")
        self.assertEqual(row["camera2_name"], "Exit")
        self.assertEqual(row["camera1_id"], "camera3")

    def test_confirmed_links_form_global_group_and_conflict_is_flagged(self):
        hypotheses = {
            "camera1:2|camera2:8": {
                "camera1_track_id": 2, "camera2_track_id": 8,
                "camera1_id": "camera1", "camera2_id": "camera2",
                "status": "SAME PERSON",
            },
            "camera2:8|camera3:4": {
                "camera1_track_id": 8, "camera2_track_id": 4,
                "camera1_id": "camera2", "camera2_id": "camera3",
                "status": "SAME PERSON",
            },
            "camera1:2|camera3:4": {
                "camera1_track_id": 2, "camera2_track_id": 4,
                "camera1_id": "camera1", "camera2_id": "camera3",
                "status": "DIFFERENT PEOPLE",
            },
        }
        groups = build_global_identities(hypotheses)
        self.assertEqual(groups[0]["global_id"], "G001")
        self.assertEqual(len(groups[0]["members"]), 3)
        self.assertTrue(groups[0]["review_conflict"])


if __name__ == "__main__":
    unittest.main()
