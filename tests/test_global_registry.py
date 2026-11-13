"""Tests for persistent, review-confirmed global identities."""

import tempfile
import unittest
from pathlib import Path

from src.reid.global_registry import (
    associate_confirmed_track, ensure_identity_source, find_identity_for_track,
    get_or_create_identity, identity_members, reject_track_association,
)


class TestGlobalIdentityRegistry(unittest.TestCase):
    def test_one_persistent_id_maps_confirmed_tracks_across_three_cameras(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "global_identity_registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            self.assertEqual(global_id, "GLOBAL-001")
            self.assertEqual(get_or_create_identity(path, "camera1", 1, "Entrance"), global_id)
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.91, "OSNet")
            associate_confirmed_track(path, global_id, "camera3", 4, "Exit", 0.88, "OSNet")
            members = identity_members(path, global_id)
            self.assertEqual(find_identity_for_track(path, "camera3", 4), global_id)
        self.assertEqual(set(members), {"camera1", "camera2", "camera3"})
        self.assertEqual([members[camera]["track_id"] for camera in members], [1, 8, 4])

    def test_rejection_removes_association_and_conflicts_are_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.9, "OSNet")
            reject_track_association(path, global_id, "camera2", 8)
            self.assertIsNone(find_identity_for_track(path, "camera2", 8))
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.9, "OSNet")
            with self.assertRaisesRegex(ValueError, "different local track"):
                associate_confirmed_track(path, global_id, "camera2", 9, "Lobby", 0.9, "OSNet")

    def test_repeated_confirmed_association_does_not_duplicate_member(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.9, "OSNet")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.92, "OSNet")
            self.assertEqual(len(identity_members(path, global_id)), 2)
            self.assertEqual(identity_members(path, global_id)["camera2"]["similarity"], 0.92)

    def test_rejecting_original_anchor_reanchors_to_another_confirmed_member(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera2", 8, "Lobby")
            ensure_identity_source(path, global_id, "camera1", 1, "Entrance")
            reject_track_association(path, global_id, "camera2", 8)
            members = identity_members(path, global_id)
        self.assertEqual(set(members), {"camera1"})
        self.assertEqual(members["camera1"]["status"], "SOURCE")


if __name__ == "__main__":
    unittest.main()
