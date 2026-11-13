"""Fetch upstream Torchreid source without compiling its optional Cython extension."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "third_party" / "deep-person-reid"
REPOSITORY = "https://github.com/KaiyangZhou/deep-person-reid.git"


def main() -> None:
    if not shutil.which("git"):
        raise SystemExit("Git is required. Install Git for Windows, reopen PowerShell, then rerun this script.")
    if DESTINATION.exists():
        if (DESTINATION / "torchreid" / "utils" / "feature_extractor.py").is_file():
            print(f"Torchreid source is already available at: {DESTINATION}")
            return
        raise SystemExit(f"The destination exists but is not a valid Torchreid source checkout: {DESTINATION}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPOSITORY, str(DESTINATION)],
        check=True,
    )
    print(f"Fetched the official Torchreid source to: {DESTINATION}")
    print("No Cython extension was built; the CPU feature-extractor API uses the Python model code.")


if __name__ == "__main__":
    main()
