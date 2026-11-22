# PyInstaller recipe for the packaged app (issue #42). From the repository root, after
# bundle/fetch_weights.py:
#     pyinstaller bundle/sigverify.spec
# One-folder build in dist/SigVerify/ (one-file mode would unpack torch on every start).
# Only files the app reads by path are listed; PyInstaller finds the code by following imports
# from launch.py, so tools/, scripts/, notebooks and tests stay out on their own.
import os
import sys

ROOT = os.path.dirname(SPECPATH)
sys.path.insert(0, ROOT)
from src.app_version import app_version  # noqa: E402

# The package has no git, so the version is worked out now and shipped as a file.
VERSION_FILE = os.path.join(workpath, "VERSION")
os.makedirs(workpath, exist_ok=True)
with open(VERSION_FILE, "w") as f:
    f.write(app_version() + "\n")
ICON = os.path.join(ROOT, "src", "gui", "assets", "signature_icon.png")

# Same relative paths as in the repository, so PROJECT_ROOT and BASE_DIR resolve unchanged.
datas = [
    (os.path.join(ROOT, "models", "weights", "spatial_cnn.pth"), os.path.join("models", "weights")),
    (os.path.join(ROOT, "models", "weights", "temporal_lstm.pth"), os.path.join("models", "weights")),
    (ICON, os.path.join("src", "gui", "assets")),
    (os.path.join(ROOT, "src", "database", "common_passwords.txt"), os.path.join("src", "database")),
    (VERSION_FILE, "."),
]

a = Analysis([os.path.join(SPECPATH, "launch.py")], pathex=[ROOT], datas=datas)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SigVerify", console=False, icon=ICON)
coll = COLLECT(exe, a.binaries, a.datas, name="SigVerify")
