"""CPU inference adapter for the official pretrained Torchreid OSNet x0.25 model."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

MODEL_NAME = "osnet_x0_25"
CHECKPOINT_NAME = (
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
    "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
INPUT_SIZE = (256, 128)  # height, width; Torchreid ImageNet normalization
FEATURE_DIM = 512  # Torchreid's published OSNet feature projection dimension


class OSNetExtractor:
    """Wrap Torchreid preprocessing/model and expose OpenCV BGR crop input."""

    model_label = f"OSNet x0.25 · MSMT17 Re-ID · {CHECKPOINT_NAME}"

    def __init__(self, checkpoint_path: Path | str) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000:
            raise FileNotFoundError(
                "The pretrained OSNet checkpoint is missing or incomplete. "
                f"Expected: {checkpoint}. Run tools/download_osnet_weights.py after "
                "installing requirements-reid.txt. The app will not substitute random weights."
            )
        try:
            source_checkout = Path(__file__).resolve().parents[2] / "third_party" / "deep-person-reid"
            if source_checkout.is_dir():
                sys.path.insert(0, str(source_checkout))
            from torchreid.utils import FeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "Torchreid source or its runtime dependencies are missing. Run "
                "tools/setup_reid_source.py, install requirements-reid.txt, and restart the app."
            ) from exc
        self._extractor = FeatureExtractor(
            model_name=MODEL_NAME,
            model_path=str(checkpoint),
            image_size=INPUT_SIZE,
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            pixel_norm=True,
            device="cpu",
            verbose=False,
        )
        self.checkpoint_path = checkpoint
        self.inference_seconds = 0.0
        self.embedding_count = 0

    def embed_crops(self, bgr_crops: list[np.ndarray]) -> np.ndarray:
        """Return finite (N, 512) L2-normalized features from BGR crops."""
        if not bgr_crops:
            return np.empty((0, FEATURE_DIM), dtype=np.float32)
        rgb_crops = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in bgr_crops]
        started = perf_counter()
        raw = self._extractor(rgb_crops)
        elapsed = perf_counter() - started
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.shape != (len(bgr_crops), FEATURE_DIM):
            raise ValueError(
                f"OSNet returned {vectors.shape}; expected ({len(bgr_crops)}, {FEATURE_DIM})."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("OSNet returned NaN or infinite embedding values.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("OSNet returned a zero-length embedding.")
        vectors /= norms
        self.inference_seconds += elapsed
        self.embedding_count += len(bgr_crops)
        return vectors

    @property
    def average_inference_ms(self) -> float | None:
        if not self.embedding_count:
            return None
        return 1000.0 * self.inference_seconds / self.embedding_count
