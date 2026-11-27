"""Post-build checks for the PyInstaller bundle: `python bundle/check_bundle.py dist/SigVerify`."""

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
