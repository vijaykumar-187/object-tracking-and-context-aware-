"""Tests for user-selected camera-local follow controls."""

import unittest

from src.tracking.selected_person import selected_track_key, selected_tracking_state


class TestSelectedPersonState(unittest.TestCase):
    def test_track_key_is_camera_and_run_specific(self):
        self.assertEqual(selected_track_key("run_a", "camera1", 3), "run_a:camera1_t0003")
        self.assertNotEqual(selected_track_key("run_a", "camera1", 3),
                            selected_track_key("run_a", "camera2", 3))

    def test_start_pause_stop_and_lost_states(self):
        self.assertEqual(selected_tracking_state(False, False, "active at end"), "Not Tracking")
        self.assertEqual(selected_tracking_state(True, False, "active at end"), "Tracking")
        self.assertEqual(selected_tracking_state(True, False, "temporarily lost"), "Temporarily Lost")
        self.assertEqual(selected_tracking_state(True, False, "ended"), "Lost")
        self.assertEqual(selected_tracking_state(True, True, "active at end"), "Paused")
        self.assertEqual(selected_tracking_state(False, False, "active at end", stopped=True), "Stopped")


if __name__ == "__main__":
    unittest.main()
