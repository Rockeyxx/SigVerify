"""Download the two shipped weights of the MODEL_VERSION release into models/weights and check
their SHA-256 against config/settings.py. Used by the installer build (issue #42).

    python bundle/fetch_weights.py        (from the repository root; needs the gh CLI)

A file that does not match never reaches models/weights: all files are checked in a temporary
folder first, and the script exits non-zero.
"""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check(path, expected: str) -> None:
    with open(path, "rb") as f:  # skylos: ignore[SKY-D215,SKY-D325] our own temp folder + a fixed weight name from settings
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual} does not match the expected {expected} "
                         "(config/settings.py)")


def install(downloaded, dest, expected: dict) -> None:
    """Check every file in `downloaded`, then copy them all into `dest`."""
    for name, sha in expected.items():
        check(Path(downloaded) / name, sha)
    for name in expected:
        shutil.copyfile(Path(downloaded) / name, Path(dest) / name)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.settings import (MODEL_VERSION, MODEL_WEIGHTS_DIR, SPATIAL_WEIGHTS_SHA256,
                                 TEMPORAL_WEIGHTS_SHA256)

    expected = {"spatial_cnn.pth": SPATIAL_WEIGHTS_SHA256, "temporal_lstm.pth": TEMPORAL_WEIGHTS_SHA256}
    Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        patterns = [arg for name in expected for arg in ("-p", name)]
        subprocess.run(["gh", "release", "download", MODEL_VERSION, "--dir", tmp, *patterns], check=True)
        install(tmp, MODEL_WEIGHTS_DIR, expected)
    print(f"Weights of {MODEL_VERSION} checked and placed in {MODEL_WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
