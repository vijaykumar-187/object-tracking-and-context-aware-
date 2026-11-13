"""Check OSNet loading, CPU feature shape/values, cosine score and timing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reid.osnet_extractor import CHECKPOINT_NAME, OSNetExtractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop_a", type=Path, help="First person crop image")
    parser.add_argument("crop_b", type=Path, help="Second person crop image")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models" / CHECKPOINT_NAME)
    args = parser.parse_args()
    images = [cv2.imread(str(path)) for path in (args.crop_a, args.crop_b)]
    if any(image is None for image in images):
        raise SystemExit("Could not read both crop image paths.")
    extractor = OSNetExtractor(args.checkpoint)
    started = time.perf_counter()
    features = extractor.embed_crops(images)
    elapsed = time.perf_counter() - started
    if features.shape != (2, 512) or not np.isfinite(features).all():
        raise SystemExit(f"OSNet check failed: returned shape {features.shape} or invalid values.")
    cosine = float(np.dot(features[0], features[1]))
    print(f"Model: {extractor.model_label}")
    print(f"Device: CPU · preprocessing: RGB 256x128, ImageNet mean/std")
    print(f"Embedding shape: {features.shape}; all values finite: yes")
    print(f"Cosine similarity between the supplied crops: {cosine:.6f}")
    print(f"Wall time for 2 crops: {elapsed:.4f}s; average: {elapsed * 500:.2f} ms/crop")


if __name__ == "__main__":
    main()
