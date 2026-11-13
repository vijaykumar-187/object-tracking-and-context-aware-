"""Small integration checks for the repeatable Review 3 runner."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.reid.run_matching import run_matching


class FakeOSNet:
    model_label = "Fake OSNet for test only"
    instances = 0

    def __init__(self, checkpoint):
        type(self).instances += 1
        self.inference_seconds = 0.0
        self.embedding_count = 0
        self.average_inference_ms = 0.1

    def embed_crops(self, crops):
        values = []
        for image in crops:
            vector = np.zeros(512, dtype=np.float32)
            vector[int(float(image.mean())) % 512] = 1.0
            values.append(vector)
        self.embedding_count += len(values)
        return np.stack(values)


class TestRunMatching(unittest.TestCase):
    def test_runner_writes_artifacts_and_reuses_video_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            videos = {}
            for camera in ("camera1", "camera2"):
                path = root / f"{camera}.avi"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60))
                self.assertTrue(writer.isOpened())
                writer.write(np.full((60, 80, 3), 40 if camera == "camera1" else 70, np.uint8))
                writer.release()
                videos[camera] = path
            checkpoint = root / "osnet.pth"
            checkpoint.write_bytes(b"x" * 120_000)
            output = root / "run"
            calls = []
            FakeOSNet.instances = 0

            def fake_process(camera, video, model, camera_dir, **kwargs):
                calls.append(camera)
                camera_dir.mkdir(parents=True, exist_ok=True)
                crops_dir = camera_dir / "crops"
                crops_dir.mkdir()
                crop_name = f"{camera}_track0001_frame000000.jpg"
                cv2.imwrite(str(crops_dir / crop_name), np.full((40, 20, 3),
                                                              40 if camera == "camera1" else 70,
                                                              np.uint8))
                with (camera_dir / "crops.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "camera_id", "track_id", "frame_number", "timestamp_seconds",
                        "crop_filename", "quality_score", "confidence", "sharpness", "width", "height",
                    ])
                    writer.writeheader()
                    writer.writerow({"camera_id": camera, "track_id": 1, "frame_number": 0,
                                     "timestamp_seconds": 0, "crop_filename": crop_name,
                                     "quality_score": 0.9, "confidence": 0.9, "sharpness": 80,
                                     "width": 20, "height": 40})
                (camera_dir / "tracks.csv").write_text("frame_number,track_id\n0,1\n", encoding="utf-8")
                (camera_dir / "tracks.json").write_text("{\"tracks\": []}", encoding="utf-8")
                payload = {"camera_id": camera, "source_name": video.name,
                           "crops_dir": str(crops_dir), "crops_manifest_file": str(camera_dir / "crops.csv"),
                           "total_frames": 1, "frames_read": 1, "frames_processed": 1,
                           "total_detections": 1, "crops_saved": 1, "processing_seconds": 0.2,
                           "track_count": 1, "processed_video": "", "metadata_file": "",
                           "fps": 10, "tracks_file": str(camera_dir / "tracks.csv"),
                           "rejected_crops_file": "", "rejected_crops_count": 0,
                           "track_summary_file": "", "statistics.json": ""}
                (camera_dir / "statistics.json").write_text(json.dumps(payload), encoding="utf-8")
                return SimpleNamespace(**payload)

            with patch("src.reid.run_matching.load_detector", return_value=object()), \
                 patch("src.reid.run_matching.process_video", side_effect=fake_process), \
                 patch("src.reid.run_matching.OSNetExtractor", FakeOSNet):
                summary = run_matching(videos["camera1"], videos["camera2"], output,
                                       model_path=root / "detector.pt", osnet_checkpoint=checkpoint)
                self.assertEqual(calls, ["camera1", "camera2"])
                self.assertEqual(summary["matches_rows"], 1)
                self.assertTrue((output / "matches.csv").is_file())
                self.assertTrue((output / "camera1" / "fingerprints.npz").is_file())
                second = run_matching(videos["camera1"], videos["camera2"], output,
                                      model_path=root / "detector.pt", osnet_checkpoint=checkpoint)
                self.assertEqual(calls, ["camera1", "camera2"])
                self.assertEqual(FakeOSNet.instances, 1)
                self.assertTrue(second["reused_saved_embeddings"])
                self.assertEqual(second["cameras"]["camera1"]["tracks"], 1)


if __name__ == "__main__":
    unittest.main()
