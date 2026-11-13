"""Synthetic multi-camera integration coverage for global-person MP4 output."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.reid.hypotheses import save_hypothesis
from src.reid.global_registry import (
    get_or_create_identity, load_registry,
)
from src.reid.video_results import render_all_camera_tracking_videos


class TestGlobalPersonVideoResults(unittest.TestCase):
    def test_three_confirmed_views_and_one_no_match_create_same_global_id_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = []
            selected_ids = {"camera1": 1, "camera2": 8, "camera3": 4,
                            "camera4": None}
            for index, (camera_id, selected_id) in enumerate(selected_ids.items(), start=1):
                camera_dir = root / camera_id
                camera_dir.mkdir()
                source_path = camera_dir / f"original_{camera_id}.avi"
                writer = cv2.VideoWriter(
                    str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60),
                )
                self.assertTrue(writer.isOpened())
                for _ in range(6):
                    writer.write(np.full((60, 80, 3), 80, dtype=np.uint8))
                writer.release()

                actual_local_id = selected_id if selected_id is not None else 99
                tracks_path = camera_dir / "tracks.csv"
                with tracks_path.open("w", newline="", encoding="utf-8") as stream:
                    fields = ["frame_number", "track_id", "x1", "y1", "x2", "y2",
                              "observed", "time_since_update"]
                    output = csv.DictWriter(stream, fieldnames=fields)
                    output.writeheader()
                    for frame_number in range(6):
                        output.writerow({
                            "frame_number": frame_number,
                            "track_id": actual_local_id,
                            "x1": 8 + frame_number, "y1": 8,
                            "x2": 28 + frame_number, "y2": 48,
                            "observed": 1, "time_since_update": 0,
                        })
                        output.writerow({
                            "frame_number": frame_number,
                            "track_id": actual_local_id + 1,
                            "x1": 40, "y1": 8, "x2": 62, "y2": 48,
                            "observed": 1, "time_since_update": 0,
                        })
                summary_path = camera_dir / "tracks.json"
                summary_path.write_text(json.dumps({
                    "max_lost_frames": 4,
                    "detector_settings": {"frame_interval": 2},
                    "tracks": [{"track_id": actual_local_id}],
                }), encoding="utf-8")
                results.append({
                    "camera_id": camera_id,
                    "camera_name": f"Camera {index}",
                    "source_name": source_path.name,
                    "source_path": str(source_path),
                    "tracks_file": str(tracks_path),
                    "track_summary_file": str(summary_path),
                })

            hypotheses = root / "cross_camera_hypotheses.json"
            save_hypothesis(hypotheses, 1, 8, 0.91, "OSNet", "SAME PERSON",
                            "camera1", "camera2", "Camera 1", "Camera 2")
            save_hypothesis(hypotheses, 8, 4, 0.88, "OSNet", "SAME PERSON",
                            "camera2", "camera3", "Camera 2", "Camera 3")
            # The same person may already have a global ID from a prior search
            # that started in another camera. A new source search must reuse it.
            registry_path = root / "global_identity_registry.json"
            preexisting_id = get_or_create_identity(
                registry_path, "camera2", 8, "Camera 2",
            )

            global_id, rows = render_all_camera_tracking_videos(
                results, root, "camera1", 1, "OSNet",
            )
            self.assertEqual(global_id, preexisting_id)
            self.assertEqual(global_id, "GLOBAL-001")
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["global_person_id"] for row in rows}, {global_id})
            by_camera = {row["camera_id"]: row for row in rows}
            self.assertEqual([by_camera[key]["local_track_id"] for key in
                              ("camera1", "camera2", "camera3")], [1, 8, 4])
            self.assertEqual(by_camera["camera4"]["match_status"], "NO RELIABLE MATCH")
            self.assertEqual(by_camera["camera4"]["local_track_id"], "")
            self.assertTrue((root / "global_person_videos" / global_id /
                             "global_person_assignments.csv").is_file())
            self.assertTrue((root / "global_person_videos" / global_id /
                             "video_manifest.json").is_file())

            for row in rows:
                video_path = Path(row["video_path"])
                self.assertTrue(video_path.is_file())
                self.assertIn("GLOBAL-001_", video_path.name)
                capture = cv2.VideoCapture(str(video_path))
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FPS)), 10)
                ok, frame = capture.read()
                capture.release()
                self.assertTrue(ok)
                if row["camera_id"] == "camera4":
                    # The only person in this camera is unrelated and is not boxed.
                    self.assertLess(abs(int(frame[8, 45, 1]) - 80), 30)
                else:
                    self.assertGreater(int(frame[8, 12, 1]), 140)
                    self.assertLess(abs(int(frame[8, 45, 1]) - 80), 30)

            repeated_id, repeated_rows = render_all_camera_tracking_videos(
                results, root, "camera1", 1, "OSNet",
            )
            self.assertEqual(repeated_id, global_id)
            self.assertEqual([row["local_track_id"] for row in repeated_rows],
                             [row["local_track_id"] for row in rows])
            registry = load_registry(root / "global_identity_registry.json")
            self.assertEqual(set(registry[global_id]["members"][i]["camera_id"]
                                 for i in range(3)), {"camera1", "camera2", "camera3"})


if __name__ == "__main__":
    unittest.main()
