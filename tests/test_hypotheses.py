"""Tests for manually reviewed cross-camera identity hypotheses."""

import tempfile
import unittest
from pathlib import Path

from src.reid.hypotheses import (
    build_global_identities, confirmed_component, load_hypotheses,
    save_hypothesis,
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

    def test_same_camera_tracks_remain_distinct_in_a_confirmed_component(self):
        hypotheses = {
            "camera1:1|camera2:8": {
                "camera1_track_id": 1, "camera2_track_id": 8,
                "camera1_id": "camera1", "camera2_id": "camera2",
                "status": "SAME PERSON",
            },
            "camera2:8|camera1:5": {
                "camera1_track_id": 8, "camera2_track_id": 5,
                "camera1_id": "camera2", "camera2_id": "camera1",
                "status": "SAME PERSON",
            },
        }
        component = confirmed_component(hypotheses, "camera1", 1)
        self.assertEqual(component,
                         {("camera1", 1), ("camera1", 5), ("camera2", 8)})

    def test_build_global_identities_keeps_same_camera_tracks_distinct(self):
        hypotheses = {
            "camera1:1|camera2:8": {
                "camera1_track_id": 1, "camera2_track_id": 8,
                "camera1_id": "camera1", "camera2_id": "camera2",
                "status": "SAME PERSON",
            },
            "camera1:1|camera1:5": {
                "camera1_track_id": 1, "camera2_track_id": 5,
                "camera1_id": "camera1", "camera2_id": "camera1",
                "status": "SAME PERSON",
            },
        }
        groups = build_global_identities(hypotheses)
        self.assertEqual(len(groups), 1)
        members = {(member["camera_id"], member["track_id"])
                   for member in groups[0]["members"]}
        self.assertEqual(members,
                         {("camera1", 1), ("camera1", 5), ("camera2", 8)})
        self.assertFalse(groups[0]["review_conflict"])

    def test_same_camera_different_people_conflict_is_flagged(self):
        hypotheses = {
            "camera1:1|camera1:5": {
                "camera1_track_id": 1, "camera2_track_id": 5,
                "camera1_id": "camera1", "camera2_id": "camera1",
                "status": "SAME PERSON",
            },
            "camera1:1|camera2:8": {
                "camera1_track_id": 1, "camera2_track_id": 8,
                "camera1_id": "camera1", "camera2_id": "camera2",
                "status": "SAME PERSON",
            },
            "camera1:5|camera2:8": {
                "camera1_track_id": 5, "camera2_track_id": 8,
                "camera1_id": "camera1", "camera2_id": "camera2",
                "status": "DIFFERENT PEOPLE",
            },
        }
        groups = build_global_identities(hypotheses)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["review_conflict"])
        self.assertEqual(len(groups[0]["conflicts"]), 1)

    def test_same_camera_hypothesis_round_trip_uses_a_distinct_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hypotheses.json"
            save_hypothesis(path, 1, 5, 0.71, "OSNet", "SAME PERSON",
                            "camera1", "camera1", "Entrance", "Entrance",
                            global_person_id="GLOBAL-001")
            save_hypothesis(path, 1, 8, 0.93, "OSNet", "SAME PERSON",
                            "camera1", "camera2", "Entrance", "Lobby")
            loaded = load_hypotheses(path)
        self.assertEqual(set(loaded), {"camera1:1|camera1:5", "camera1:1|camera2:8"})
        same_camera = loaded["camera1:1|camera1:5"]
        self.assertEqual(same_camera["camera1_id"], "camera1")
        self.assertEqual(same_camera["camera2_id"], "camera1")
        self.assertEqual(same_camera["global_person_id"], "GLOBAL-001")
        self.assertEqual(same_camera["status"], "SAME PERSON")


if __name__ == "__main__":
    unittest.main()
