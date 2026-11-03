"""
Unit and Integration Tests for Phase 2: Person Detection and Crop Extraction

These tests verify the sharpness filter, model loading, frame processing,
and metadata generation on CPU without requiring the actual dataset.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detect import (
    BLUR_THRESHOLD,
    CSV_FIELDNAMES,
    compute_sharpness,
    detect_people_in_frame,
    is_blurry,
    load_detector,
    process_camera,
)


class TestSharpnessAndBlurFilter(unittest.TestCase):
    """Verifies that the blur filtering mechanism operates accurately."""

    def test_sharp_image_passes_filter(self):
        # Create a high-contrast checkerboard image (very sharp edges)
        sharp_img = np.zeros((120, 60, 3), dtype=np.uint8)
        sharp_img[::10, :, :] = 255
        sharp_img[:, ::10, :] = 255

        blurry, sharpness = is_blurry(sharp_img, threshold=BLUR_THRESHOLD)
        self.assertFalse(blurry, "A sharp checkerboard pattern should not be classified as blurry.")
        self.assertGreater(sharpness, BLUR_THRESHOLD)

    def test_blurry_image_rejected_by_filter(self):
        # Create an image blurred heavily with Gaussian blur
        sharp_img = np.random.randint(0, 256, (120, 60, 3), dtype=np.uint8)
        blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)

        blurry, sharpness = is_blurry(blurry_img, threshold=BLUR_THRESHOLD)
        self.assertTrue(blurry, "A heavily blurred image must be flagged as blurry.")
        self.assertLess(sharpness, BLUR_THRESHOLD)

    def test_empty_image_handles_safely(self):
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        sharpness = compute_sharpness(empty_img)
        self.assertEqual(sharpness, 0.0)


class TestDetectionPipelineIntegration(unittest.TestCase):
    """Verifies the detection pipeline execution using a synthetic video stream."""

    @classmethod
    def setUpClass(cls):
        # Load model once on CPU for test suite
        cls.model = load_detector()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_frame_detection(self):
        # Black frame containing no humans
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detect_people_in_frame(self.model, black_frame)
        self.assertEqual(len(detections), 0, "A black frame should produce 0 person detections.")

    def test_video_processing_and_metadata_generation(self):
        # Generate a small 30-frame synthetic test video
        synthetic_video_path = self.temp_path / "test_cam.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(synthetic_video_path), fourcc, 30.0, (640, 480))

        for frame_num in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            # Add simple changing graphics
            cv2.putText(
                frame,
                f"Frame {frame_num}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
        writer.release()

        self.assertTrue(synthetic_video_path.exists())

        # Test process_camera execution on this synthetic video
        test_csv_path = self.temp_path / "test_metadata.csv"
        with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()

            saved_crops = process_camera(
                camera_id="test_cam",
                video_path=synthetic_video_path,
                model=self.model,
                writer=csv_writer,
                save_previews=False,
            )

        # In a synthetic frame with only text, 0 people are detected, so 0 crops are saved
        self.assertEqual(saved_crops, 0)
        self.assertTrue(test_csv_path.exists())

        # Verify that CSV headers were created properly
        with open(test_csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip().split(",")
            self.assertEqual(header_line, CSV_FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
