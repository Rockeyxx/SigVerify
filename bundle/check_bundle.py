"""Checks on the built package (issue #42), run after PyInstaller:

    python bundle/check_bundle.py dist/SigVerify

1. layout: the two weights, the icon, the common-password list and VERSION are where
   config/settings.py, src/gui/app.py and src/app_version.py look.
2. start: the shipped app runs for 30 s with --debug, offscreen, on an empty data folder,
   without exiting and without a Traceback in debug.log. It does not catch a weight that
   fails to load (warm-up hides that error); the layout check and the manual test cover it.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED = ["models/weights/spatial_cnn.pth", "models/weights/temporal_lstm.pth",
            "src/gui/assets/signature_icon.png", "src/database/common_passwords.txt", "VERSION"]


def layout(dist) -> None:
    internal = Path(dist) / "_internal"
    missing = [rel for rel in REQUIRED if not (internal / rel).is_file()]
    if missing:
        raise SystemExit(f"missing from the package: {', '.join(missing)}")


def start(exe, seconds: int = 30) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:   # Windows may hold a file briefly
        env = {**os.environ, "SIGVER_HOME": home, "QT_QPA_PLATFORM": "offscreen"}
        proc = subprocess.Popen([exe, "--debug"], env=env)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            code = None
            proc.kill()
            proc.wait()
        log_path = Path(home, "debug.log")
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    print(log or "(debug.log is empty)")
    if code is not None:
        raise SystemExit(f"the app exited with code {code} within {seconds} s")
    if "Traceback" in log:
        raise SystemExit("the app logged a Traceback")


def main() -> None:
    dist = Path(sys.argv[1])
    layout(dist)
    start(str(dist / ("SigVerify.exe" if sys.platform == "win32" else "SigVerify")))
    print("Package checks passed")


if __name__ == "__main__":
    main()
