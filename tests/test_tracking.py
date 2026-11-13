"""Behavior checks for the local SORT-style tracker."""

import unittest

from src.tracking.sort_tracker import SortTracker


class TestSortTracker(unittest.TestCase):
    def test_id_follows_moving_detection_and_reports_observation_state(self):
        tracker = SortTracker(max_age=3)
        first = tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 0)
        second = tracker.update([{"box": (18, 12, 68, 152), "confidence": 0.88}], 1)
        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)
        self.assertTrue(second[0].observed)
        self.assertFalse(second[0].recovered)
        self.assertEqual(second[0].detection_index, 0)

    def test_track_survives_short_missed_detection(self):
        tracker = SortTracker(max_age=3)
        tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 0)
        lost = tracker.update([], 1)
        self.assertEqual(lost[0].track_id, 1)
        self.assertFalse(lost[0].observed)
        self.assertEqual(lost[0].time_since_update, 1)
        found = tracker.update([{"box": (20, 10, 70, 150), "confidence": 0.85}], 2)
        self.assertEqual(found[0].track_id, 1)
        self.assertTrue(found[0].observed)
        self.assertTrue(found[0].recovered)

    def test_expired_track_does_not_reuse_its_id(self):
        tracker = SortTracker(max_age=2)
        tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 0)
        tracker.update([], 1)
        tracker.update([], 2)
        self.assertEqual(tracker.update([], 3), [])
        new = tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 4)
        self.assertEqual(new[0].track_id, 2)

    def test_multiple_people_receive_distinct_ids(self):
        tracker = SortTracker()
        observations = tracker.update([
            {"box": (10, 10, 60, 150), "confidence": 0.9},
            {"box": (300, 10, 350, 150), "confidence": 0.8},
        ], 0)
        self.assertEqual({item.track_id for item in observations}, {1, 2})


if __name__ == "__main__":
    unittest.main()
