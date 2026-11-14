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
        # A fresh detection after expiry starts a new (tentative) track; it is
        # hidden until confirmed, then it must use a brand-new ID, never the
        # expired track's ID.
        self.assertEqual(tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 4), [])
        new = tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 5)
        self.assertEqual(new[0].track_id, 2)

    def test_multiple_people_receive_distinct_ids(self):
        tracker = SortTracker()
        observations = tracker.update([
            {"box": (10, 10, 60, 150), "confidence": 0.9},
            {"box": (300, 10, 350, 150), "confidence": 0.8},
        ], 0)
        self.assertEqual({item.track_id for item in observations}, {1, 2})

    def test_tentative_track_is_hidden_until_min_hits_real_matches(self):
        tracker = SortTracker(max_age=10, min_hits=2, frame_interval=1)
        # Advance past the short video-start grace with empty sampled frames.
        tracker.update([], 0)
        tracker.update([], 1)
        # A brand-new track has one detection association: still tentative.
        self.assertEqual(
            tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 2), [])
        # A Kalman prediction-only frame must not count toward confirmation.
        self.assertEqual(tracker.update([], 3), [])
        # The second real detection association confirms it with a stable ID.
        confirmed = tracker.update([{"box": (14, 11, 64, 151), "confidence": 0.9}], 4)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].track_id, 1)
        self.assertTrue(confirmed[0].observed)

    def test_track_survives_intended_missed_detection_opportunities(self):
        # frame_interval=5, so max_age counts sampled misses, not raw frames.
        tracker = SortTracker(max_age=2, min_hits=1, frame_interval=5)
        first = tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 0)
        self.assertEqual(first[0].track_id, 1)
        # A non-sampled frame does not consume the lost-track budget.
        self.assertEqual(len(tracker.update([], 3)), 1)
        self.assertEqual(len(tracker.update([], 5)), 1)   # missed opportunity 1
        self.assertEqual(len(tracker.update([], 10)), 1)  # missed opportunity 2
        self.assertEqual(tracker.update([], 15), [])      # missed 3 -> expired
        new = tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 20)
        self.assertEqual(new[0].track_id, 2)              # expired ID is not reused

    def test_recovered_flag_respects_sampling_interval(self):
        tracker = SortTracker(max_age=5, min_hits=1, frame_interval=10)
        tracker.update([{"box": (10, 10, 60, 150), "confidence": 0.9}], 0)
        # A normal consecutive sampled match is NOT flagged as a recovery even
        # though ten raw frames (time_since_update) elapsed between detections.
        consecutive = tracker.update([{"box": (14, 11, 64, 151), "confidence": 0.9}], 10)
        self.assertTrue(consecutive[0].observed)
        self.assertFalse(consecutive[0].recovered)
        # Miss one detection opportunity, then match again -> genuine recovery.
        tracker.update([], 20)
        recovered = tracker.update([{"box": (18, 12, 68, 152), "confidence": 0.9}], 30)
        self.assertTrue(recovered[0].observed)
        self.assertTrue(recovered[0].recovered)

    def test_invalid_tracker_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            SortTracker(max_age=0)
        with self.assertRaises(ValueError):
            SortTracker(frame_interval=0)
        with self.assertRaises(ValueError):
            SortTracker(min_hits=0)


if __name__ == "__main__":
    unittest.main()
