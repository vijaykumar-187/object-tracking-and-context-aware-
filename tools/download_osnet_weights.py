"""Download the official small OSNet Re-ID checkpoint into models/."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reid.osnet_extractor import CHECKPOINT_NAME  # noqa: E402

URL = f"https://huggingface.co/kaiyangzhou/osnet/resolve/main/{CHECKPOINT_NAME}?download=true"
DESTINATION = ROOT / "models" / CHECKPOINT_NAME


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(DESTINATION.suffix + ".download")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "person-reid-student-project/1.0"})
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
                if total:
                    print(f"Downloaded {size / total:.0%}", end="\r", flush=True)
        if size < 100_000:
            raise OSError(f"Download was unexpectedly small ({size} bytes).")
        os.replace(temporary, DESTINATION)
        print(f"\nSaved pretrained OSNet checkpoint: {DESTINATION}")
        print(f"Size: {size:,} bytes · SHA-256: {digest.hexdigest()}")
        print("Checkpoint source: kaiyangzhou/osnet on Hugging Face; MSMT17 Re-ID-trained OSNet x0.25.")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
