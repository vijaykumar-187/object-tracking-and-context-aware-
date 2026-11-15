"""Tests for persistent, review-confirmed global identities."""

import json
import tempfile
import unittest
from pathlib import Path

from src.reid.global_registry import (
    associate_confirmed_track, ensure_identity_source, find_identity_for_track,
    get_or_create_identity, identity_members, identity_members_all, load_registry,
    reject_track_association,
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

    def test_multiple_same_camera_tracks_share_one_global_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.91, "OSNet")
            associate_confirmed_track(path, global_id, "camera1", 5, "Entrance", 0.74, "OSNet")
            grouped = identity_members_all(path, global_id)
            record = load_registry(path)[global_id]
            sibling_identity = find_identity_for_track(path, "camera1", 5)
        self.assertEqual([member["track_id"] for member in grouped["camera1"]], [1, 5])
        self.assertEqual([member["track_id"] for member in grouped["camera2"]], [8])
        self.assertEqual(sibling_identity, global_id)
        self.assertEqual(
            record["source"],
            {"camera_id": "camera1", "camera_name": "Entrance", "track_id": 1},
        )
        sources = [member for member in record["members"]
                   if member["status"] == "SOURCE"]
        self.assertEqual(len(sources), 1, "A global ID must keep exactly one SOURCE member.")

    def test_identity_members_keeps_one_track_per_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera1", 5, "Entrance", 0.95, "OSNet")
            members = identity_members(path, global_id)
        self.assertEqual(set(members), {"camera1"})
        self.assertEqual(members["camera1"]["track_id"], 1,
                         "The compatibility accessor should prefer the SOURCE anchor.")
        self.assertEqual(members["camera1"]["status"], "SOURCE")

    def test_ensure_identity_source_adds_same_camera_track_without_moving_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            ensure_identity_source(path, global_id, "camera1", 5, "Entrance")
            ensure_identity_source(path, global_id, "camera1", 1, "Entrance")
            record = load_registry(path)[global_id]
            grouped = identity_members_all(path, global_id)
        self.assertEqual([member["track_id"] for member in grouped["camera1"]], [1, 5])
        self.assertEqual(record["source"]["track_id"], 1)
        added = next(member for member in record["members"] if member["track_id"] == 5)
        self.assertEqual(added["status"], "CONFIRMED")
        self.assertEqual(len([member for member in record["members"]
                              if member["status"] == "SOURCE"]), 1)

    def test_repeated_confirmed_association_does_not_duplicate_member(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.9, "OSNet")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.92, "OSNet")
            grouped = identity_members_all(path, global_id)
            members = identity_members(path, global_id)
        self.assertEqual(len(grouped["camera2"]), 1)
        self.assertEqual(grouped["camera2"][0]["similarity"], 0.92)
        self.assertEqual(len(members), 2)

    def test_rejecting_same_camera_sibling_preserves_identity_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera1", 5, "Entrance", 0.7, "OSNet")
            associate_confirmed_track(path, global_id, "camera2", 8, "Lobby", 0.9, "OSNet")
            reject_track_association(path, global_id, "camera1", 5)
            record = load_registry(path)[global_id]
            rejected = find_identity_for_track(path, "camera1", 5)
            anchor = find_identity_for_track(path, "camera1", 1)
            other_camera = find_identity_for_track(path, "camera2", 8)
        self.assertIsNone(rejected)
        self.assertEqual(anchor, global_id)
        self.assertEqual(other_camera, global_id)
        self.assertEqual(record["source"]["track_id"], 1)
        self.assertEqual([member["status"] for member in record["members"]],
                         ["SOURCE", "CONFIRMED"])

    def test_rejecting_original_anchor_reanchors_to_one_remaining_member(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            global_id = get_or_create_identity(path, "camera2", 8, "Lobby")
            ensure_identity_source(path, global_id, "camera1", 1, "Entrance")
            associate_confirmed_track(path, global_id, "camera1", 5, "Entrance", 0.6, "OSNet")
            reject_track_association(path, global_id, "camera2", 8)
            record = load_registry(path)[global_id]
            members = identity_members(path, global_id)
        self.assertEqual(set(members), {"camera1"})
        self.assertEqual(record["source"], {
            "camera_id": "camera1", "camera_name": "Entrance", "track_id": 1,
        })
        sources = [member for member in record["members"]
                   if member["status"] == "SOURCE"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["track_id"], 1)

    def test_associating_a_track_merges_identities_and_keeps_one_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            first = get_or_create_identity(path, "camera1", 1, "Entrance")
            associate_confirmed_track(path, first, "camera2", 8, "Lobby", 0.9, "OSNet")
            second = get_or_create_identity(path, "camera1", 5, "Entrance")
            associate_confirmed_track(path, second, "camera2", 9, "Lobby", 0.8, "OSNet")
            associate_confirmed_track(path, first, "camera1", 5, "Entrance", 0.7, "OSNet")
            registry = load_registry(path)
            grouped = identity_members_all(path, first)
        self.assertNotIn(second, registry)
        self.assertEqual(set(registry), {first})
        self.assertEqual([member["track_id"] for member in grouped["camera1"]], [1, 5])
        self.assertEqual([member["track_id"] for member in grouped["camera2"]], [8, 9])
        sources = [member for member in registry[first]["members"]
                   if member["status"] == "SOURCE"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(registry[first]["source"]["track_id"], 1)

    def test_local_track_cannot_be_claimed_by_two_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            first = get_or_create_identity(path, "camera1", 1, "Entrance")
            second = get_or_create_identity(path, "camera1", 5, "Entrance")
            with self.assertRaisesRegex(ValueError, "already belongs"):
                ensure_identity_source(path, first, "camera1", 5, "Entrance")
            remaining = find_identity_for_track(path, "camera1", 5)
        self.assertNotEqual(first, second)
        self.assertEqual(remaining, second,
                         "The rejected attempt must not move the track.")

    def test_legacy_registry_record_still_loads_and_accepts_same_camera_members(self):
        legacy = {
            "GLOBAL-001": {
                "global_person_id": "GLOBAL-001",
                "source": {"camera_id": "camera1", "camera_name": "Entrance",
                           "track_id": 1},
                "members": [
                    {"camera_id": "camera1", "camera_name": "Entrance",
                     "track_id": 1, "status": "SOURCE"},
                    {"camera_id": "camera2", "camera_name": "Lobby",
                     "track_id": 8, "status": "CONFIRMED",
                     "similarity": 0.9, "model_label": "OSNet"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            members = identity_members(path, "GLOBAL-001")
            associate_confirmed_track(path, "GLOBAL-001", "camera1", 5,
                                      "Entrance", 0.7, "OSNet")
            grouped = identity_members_all(path, "GLOBAL-001")
            record = load_registry(path)["GLOBAL-001"]
        self.assertEqual(members["camera2"]["track_id"], 8)
        self.assertEqual([member["track_id"] for member in grouped["camera1"]], [1, 5])
        self.assertEqual(record["source"]["track_id"], 1)
        self.assertEqual(grouped["camera2"][0]["similarity"], 0.9)

    def test_legacy_duplicate_source_flags_are_read_and_normalised(self):
        legacy = {
            "GLOBAL-001": {
                "global_person_id": "GLOBAL-001",
                "source": {"camera_id": "camera1", "camera_name": "Entrance",
                           "track_id": 1},
                "members": [
                    {"camera_id": "camera1", "camera_name": "Entrance",
                     "track_id": 5, "status": "SOURCE"},
                    {"camera_id": "camera1", "camera_name": "Entrance",
                     "track_id": 1, "status": "SOURCE"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            members = identity_members(path, "GLOBAL-001")
        self.assertEqual(members["camera1"]["track_id"], 1,
                         "The recorded source anchor wins over a stale flag.")

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            associate_confirmed_track(path, "GLOBAL-001", "camera1", 9,
                                      "Entrance", 0.5, "OSNet")
            record = load_registry(path)["GLOBAL-001"]
        self.assertEqual(record["source"]["track_id"], 1)
        self.assertEqual([member["status"] for member in record["members"]],
                         ["SOURCE", "CONFIRMED", "CONFIRMED"])


if __name__ == "__main__":
    unittest.main()
