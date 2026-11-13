"""CPU proof-of-concept tests using a fake Torchreid API, no random model."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.reid.osnet_extractor import CHECKPOINT_NAME, OSNetExtractor


class TestOSNetExtractor(unittest.TestCase):
    def test_missing_checkpoint_fails_clearly_without_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint is missing"):
                OSNetExtractor(Path(folder) / CHECKPOINT_NAME)

    def test_cpu_embeddings_are_finite_512d_and_input_is_converted_to_rgb(self):
        seen = {}

        class FakeFeatureExtractor:
            def __init__(self, **kwargs):
                seen["kwargs"] = kwargs

            def __call__(self, images):
                seen["first_rgb"] = images[0][0, 0].tolist()
                return np.tile(np.arange(1, 513, dtype=np.float32), (len(images), 1))

        fake_root = types.ModuleType("torchreid")
        fake_utils = types.ModuleType("torchreid.utils")
        fake_utils.FeatureExtractor = FakeFeatureExtractor
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / CHECKPOINT_NAME
            checkpoint.write_bytes(b"x" * 100_001)
            with patch.dict(sys.modules, {"torchreid": fake_root, "torchreid.utils": fake_utils}):
                extractor = OSNetExtractor(checkpoint)
                crop = np.zeros((4, 4, 3), dtype=np.uint8)
                crop[0, 0] = [11, 22, 33]  # OpenCV BGR
                result = extractor.embed_crops([crop, crop])
        self.assertEqual(result.shape, (2, 512))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0], atol=1e-5)
        self.assertEqual(seen["first_rgb"], [33, 22, 11])
        self.assertEqual(seen["kwargs"]["device"], "cpu")
        self.assertEqual(seen["kwargs"]["image_size"], (256, 128))
        self.assertIsNotNone(extractor.average_inference_ms)


if __name__ == "__main__":
    unittest.main()
