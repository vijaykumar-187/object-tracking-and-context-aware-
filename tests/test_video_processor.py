"""Tests for input validation and tracked-video processing."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.detection.video_processor import (
    process_video,
    render_selected_person_video,
    render_selected_track_video,
    validate_video_path,
)
from src.tracking.sort_tracker import SortTracker


def textured_frame() -> np.ndarray:
    """A sharp, patterned frame for testing the crop quality gate."""
    yy, xx = np.indices((480, 640))
    pattern = (((xx // 4 + yy // 4) % 2) * 220 + 20).astype(np.uint8)
    return np.dstack([pattern, pattern, pattern])


class TestVideoValidation(unittest.TestCase):
    def test_unsupported_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.txt"
            path.write_text("not a video", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported video format"):
                validate_video_path(path)

    def test_missing_video_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "could not be found"):
            validate_video_path(Path("missing-video.mp4"))


class TestVideoProcessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video_path = self.root / "sample.avi"
        writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG video writer is unavailable")
        for _ in range(3):
            writer.write(textured_frame())
        writer.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("src.detection.video_processor.detect_people_in_frame")
    def test_tracks_persist_and_quality_crops_are_limited(self, detect):
        detect.return_value = [
            {"box": (10, 10, 150, 250), "confidence": 0.91},
            {"box": (100, 100, 90, 120), "confidence": 0.80},  # invalid crop
            {"box": (450, 200, 700, 400), "confidence": 0.75},  # clipped to frame
        ]
        output = self.root / "new" / "camera1"
        result = process_video("camera1", self.video_path, object(), output, frame_interval=2)

        self.assertTrue(output.is_dir())
        self.assertTrue(Path(result.processed_video).is_file())
        self.assertTrue(Path(result.tracks_file).is_file())
        self.assertTrue(Path(result.crops_manifest_file).is_file())
        self.assertTrue(Path(result.rejected_crops_file).is_file())
        self.assertTrue(Path(result.track_summary_file).is_file())
        self.assertTrue((output / "statistics.json").is_file())
        self.assertEqual(result.frames_read, 3)
        self.assertEqual(result.frames_processed, 2)
        self.assertEqual(result.total_detections, 6)
        self.assertEqual(result.crops_saved, 2)  # one best crop per track/time bucket

        with Path(result.tracks_file).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        track_one = [row for row in rows if row["track_id"] == "1"]
        self.assertEqual([row["frame_number"] for row in track_one], ["0", "1", "2"])
        self.assertEqual([row["observed"] for row in track_one], ["1", "0", "1"])
        self.assertEqual([row["time_since_update"] for row in track_one], ["0", "1", "0"])

        with Path(result.crops_manifest_file).open(newline="", encoding="utf-8") as stream:
            crop_rows = list(csv.DictReader(stream))
        self.assertEqual(len(crop_rows), 2)
        self.assertTrue(all(float(row["sharpness"]) >= 50 for row in crop_rows))
        self.assertTrue(all(row["x1"] and row["x2"] for row in crop_rows))
        self.assertTrue(all((Path(result.crops_dir) / row["crop_filename"]).is_file() for row in crop_rows))
        with Path(result.rejected_crops_file).open(newline="", encoding="utf-8") as stream:
            rejected_rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rejected_rows), 2)
        self.assertIn("invalid_box_or_empty_crop", {row["reason"] for row in rejected_rows})

        summary = json.loads(Path(result.track_summary_file).read_text(encoding="utf-8"))
        self.assertTrue(summary["track_ids_are_camera_local"])
        self.assertEqual(len(summary["tracks"]), 3)
        self.assertIn("duration_seconds", summary["tracks"][0])
        self.assertIn("lifecycle_status", summary["tracks"][0])
        self.assertIn("confidence_history", summary["tracks"][0])
        self.assertEqual(summary["tracker_settings"]["iou_threshold"], 0.05)
        self.assertTrue((output / "performance.json").is_file())
        self.assertEqual(result.rejected_crops_count, len(rejected_rows))

        selected_video = render_selected_track_video(
            self.video_path, result.tracks_file, output / "selected.mp4", "camera1", 1
        )
        self.assertTrue(selected_video.is_file())

    def test_bad_sampling_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            process_video("camera1", self.video_path, object(), self.root / "out", frame_interval=0)

    def test_lost_track_window_must_be_a_positive_opportunity_count(self):
        with self.assertRaisesRegex(ValueError, "at least one missed detection"):
            process_video("camera1", self.video_path, object(), self.root / "out",
                          frame_interval=5, tracker_max_age=0)

    def test_min_hits_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least one detection association"):
            process_video("camera1", self.video_path, object(), self.root / "out",
                          tracker_min_hits=0)

    @patch("src.detection.video_processor.detect_people_in_frame")
    def test_max_age_may_be_smaller_than_frame_interval(self, detect):
        # max_age is now counted in detection opportunities, so a value below the
        # sampling interval is valid and must not raise (the old frame-based
        # guard rejected it).
        detect.return_value = []
        result = process_video("camera1", self.video_path, object(), self.root / "out",
                               frame_interval=5, tracker_max_age=1)
        summary = json.loads(Path(result.track_summary_file).read_text(encoding="utf-8"))
        self.assertEqual(summary["tracker_settings"]["max_age_detections"], 1)
        self.assertEqual(summary["tracker_settings"]["frame_interval"], 5)
        self.assertEqual(summary["tracker_settings"]["max_age_frames"], 5)

    @patch("src.detection.video_processor.detect_people_in_frame")
    def test_sampling_config_is_passed_to_the_tracker(self, detect):
        detect.return_value = [{"box": (10, 10, 150, 250), "confidence": 0.9}]
        captured: dict = {}

        class SpyTracker(SortTracker):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch("src.detection.video_processor.SortTracker", SpyTracker):
            result = process_video("camera1", self.video_path, object(), self.root / "spy",
                                   frame_interval=2, tracker_min_hits=3, tracker_max_age=7)
        self.assertEqual(captured.get("frame_interval"), 2)
        self.assertEqual(captured.get("min_hits"), 3)
        self.assertEqual(captured.get("max_age"), 7)
        summary = json.loads(Path(result.track_summary_file).read_text(encoding="utf-8"))
        self.assertEqual(summary["tracker_settings"]["min_hits"], 3)
        self.assertEqual(summary["tracker_settings"]["frame_interval"], 2)

    def test_tracker_iou_threshold_is_validated(self):
        with self.assertRaisesRegex(ValueError, "Tracker IoU threshold"):
            process_video("camera1", self.video_path, object(), self.root / "out",
                          tracker_iou_threshold=1.1)

    def test_global_person_videos_are_continuous_single_track_playable_videos(self):
        source_frames = []
        for frame_index in range(4):
            frame = np.full((60, 80, 3), 80, dtype=np.uint8)
            source_frames.append(frame)

        output_videos = []
        for camera_id, selected_id in (("camera1", 1), ("camera2", 8), ("camera3", 4)):
            source_path = self.root / f"{camera_id}.avi"
            writer = cv2.VideoWriter(
                str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60),
            )
            self.assertTrue(writer.isOpened())
            for frame in source_frames:
                writer.write(frame)
            writer.release()
            tracks_path = self.root / f"{camera_id}_tracks.csv"
            with tracks_path.open("w", newline="", encoding="utf-8") as stream:
                fields = ["frame_number", "track_id", "x1", "y1", "x2", "y2",
                          "observed", "time_since_update"]
                rows = csv.DictWriter(stream, fieldnames=fields)
                rows.writeheader()
                for frame_index in range(4):
                    rows.writerow({
                        "frame_number": frame_index, "track_id": selected_id,
                        "x1": 10 + frame_index, "y1": 10, "x2": 30 + frame_index,
                        "y2": 45, "observed": 0 if frame_index in (1, 2) else 1,
                        "time_since_update": 1 if frame_index == 1 else (
                            3 if frame_index == 2 else 0
                        ),
                    })
                    rows.writerow({
                        "frame_number": frame_index, "track_id": selected_id + 1,
                        "x1": 40, "y1": 10, "x2": 60, "y2": 45,
                        "observed": 1, "time_since_update": 0,
                    })
            output_path = self.root / f"GLOBAL-001_{camera_id}_tracked.mp4"
            video = render_selected_person_video(
                source_path, tracks_path, output_path, camera_id,
                "GLOBAL-001", selected_id, "HUMAN CONFIRMED",
                max_prediction_gap_frames=2,
            )
            output_videos.append(video)

        for video in output_videos:
            self.assertTrue(video.is_file())
            capture = cv2.VideoCapture(str(video))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            self.assertEqual((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))), (80, 60))
            ok, observed_frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            # The selected person's green box exists; the unrelated person's
            # expected box location remains the original grey scene.
            self.assertGreater(int(observed_frame[10, 15, 1]), 140)
            self.assertLess(abs(int(observed_frame[10, 45, 1]) - 80), 30)

        # A short predicted box is visibly different, but a stale Kalman box
        # beyond the reliability window is suppressed.
        capture = cv2.VideoCapture(str(output_videos[0]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ok, predicted = capture.read()
        self.assertTrue(ok)
        self.assertGreater(int(predicted[10, 12, 1]), 130)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, stale = capture.read()
        capture.release()
        self.assertTrue(ok)
        self.assertLess(abs(int(stale[10, 12, 1]) - 80), 30)

        no_match = render_selected_person_video(
            self.video_path, self.root / "unused.csv",
            self.root / "GLOBAL-001_camera4_tracked.mp4", "Camera 4",
            "GLOBAL-001", None, "NO RELIABLE MATCH",
        )
        self.assertTrue(no_match.is_file())


if __name__ == "__main__":
    unittest.main()
