"""Tests for upload validation and the Streamlit-independent video processor."""

import tempfile
import csv
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.detection.video_processor import process_video, validate_video_path


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
            str(self.video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG video writer is unavailable")
        for _ in range(3):
            writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        writer.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("src.detection.video_processor.detect_people_in_frame")
    def test_output_folders_video_statistics_safe_crops_and_continuous_boxes(self, detect):
        detect.return_value = [
            {"box": (5, 4, 25, 35), "confidence": 0.91},
            {"box": (30, 20, 20, 30), "confidence": 0.80},  # invalid, must be skipped
            {"box": (-10, -10, 70, 60), "confidence": 0.75},  # clipped to frame
        ]
        output = self.root / "new" / "camera1"
        result = process_video("camera1", self.video_path, object(), output, frame_interval=2)

        self.assertTrue(output.is_dir())
        self.assertTrue(Path(result.processed_video).is_file())
        self.assertTrue(Path(result.metadata_file).is_file())
        self.assertTrue((output / "statistics.json").is_file())
        crops = list(Path(result.crops_dir).glob("*.jpg"))
        self.assertEqual(result.frames_read, 3)
        self.assertEqual(result.frames_processed, 2)
        self.assertEqual(result.total_detections, 6)
        self.assertEqual(result.crops_saved, 4)
        self.assertEqual(len(crops), 4)
        self.assertTrue(all("camera1_frame" in crop.name and "_person" in crop.name for crop in crops))
        with Path(result.metadata_file).open(newline="", encoding="utf-8") as metadata_file:
            rows = list(csv.DictReader(metadata_file))
        self.assertEqual([row["track_id"] for row in rows], ["1", "2", "1", "2"])

        # The output video must keep the box visible on the unsampled middle frame.
        processed = cv2.VideoCapture(result.processed_video)
        visible_box_frames = 0
        while True:
            ok, frame = processed.read()
            if not ok:
                break
            green_box_pixels = (frame[:, :, 1] > frame[:, :, 2] * 1.15) & (frame[:, :, 1] > frame[:, :, 0] * 1.15)
            visible_box_frames += int(np.count_nonzero(green_box_pixels) > 0)
        processed.release()
        self.assertEqual(visible_box_frames, 3)

    def test_bad_sampling_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            process_video("camera1", self.video_path, object(), self.root / "out", frame_interval=0)


if __name__ == "__main__":
    unittest.main()
