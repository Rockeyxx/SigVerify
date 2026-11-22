"""Build the whole-system test kit (issue #43) a team member runs on the installed app.

    python tools/make_test_kit.py          # dist/SigVerify-test-kit.zip, a made-up signer
    python tools/make_test_kit.py --p01    # in-house participant P01's real signings (send privately)

The kit holds one signer (paper scans and the signing window's event logs), the scores the
shipped model gives those files scored directly, run_test.bat / run_test.sh and a README. The
member installs the app, unzips the kit, runs the script, watches the app test itself, and
sends back SigVerify-test-results.txt (src/self_test.py writes it).
"""
import argparse
import csv
import json
import os
import shutil
import stat
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (ALPHA_DEFAULT, MODEL_VERSION, NUM_ENROLLMENT_REFERENCES,  # noqa: E402
                             PROJECT_ROOT, SPATIAL_WEIGHTS_PATH, TEMPORAL_WEIGHTS_PATH)
from scripts.evaluate import embed_samples  # noqa: E402
from src.data.campaign import load_inhouse_tablet_file, parse_inhouse  # noqa: E402
from src.data.renderer import render_trajectory_to_image  # noqa: E402
from src.gui.image_cleanup import crop_to_ink  # noqa: E402
from src.verification import load_model, nearest_reference_distances, similarity_from_distance  # noqa: E402

BLUE_INK_BGR = (170, 60, 20)            # the app reads blue pen on white paper only (ENG_REPORT A6)
KIT_NAME = "SigVerify-test-kit"
PRESS, MOVE, RELEASE = 92, 87, 93       # Qt tablet event types, as the signing window logs them

RUN_BAT = r"""@echo off
rem SigVerify whole-system test (issue #43). Double-click this file.
cd /d "%~dp0"
if not exist "expected.json" (
  echo Unzip the whole kit into a folder first, then run run_test.bat from that folder.
  pause
  exit /b 1
)
set "APP=%LOCALAPPDATA%\Programs\SigVerify\SigVerify.exe"
if not exist "%APP%" set "APP=%ProgramFiles%\SigVerify\SigVerify.exe"
if not exist "%APP%" (
  echo SigVerify is not installed. Install it first, then run this again.
  pause
  exit /b 1
)
echo The SigVerify window opens and tests itself for about two minutes.
echo Watch it, but do not click or type in it until this window shows the results.
start "" /wait "%APP%" --self-test "%~dp0."
echo.
type "SigVerify-test-results.txt"
echo.
echo Send the file SigVerify-test-results.txt (in this folder) to the team.
pause
"""

RUN_SH = """#!/bin/sh
# SigVerify whole-system test (issue #43): ./run_test.sh [path/to/SigVerify-x86_64.AppImage]
cd "$(dirname "$0")" || exit 1
APP="$1"
[ -n "$APP" ] || APP=$(ls SigVerify-*.AppImage "$HOME"/Downloads/SigVerify-*.AppImage 2>/dev/null | head -n 1)
if [ -z "$APP" ]; then
  echo "Put the SigVerify AppImage in this folder (or pass its path), then run this again."
  exit 1
fi
chmod +x "$APP"
echo "The SigVerify window opens and tests itself for about two minutes. Do not click or type in it."
"$APP" --self-test "$PWD"
cat SigVerify-test-results.txt
echo
echo "Send the file SigVerify-test-results.txt (in this folder) to the team."
"""

README = """SigVerify test kit (issue #43)

1. Install SigVerify (Windows: the setup .exe; Linux: the AppImage).
2. Unzip this kit into a folder.
3. Windows: double-click run_test.bat.  Linux: put the AppImage in the folder, run ./run_test.sh
4. The SigVerify window opens and tests itself for about two minutes. The title bar says what
   it is doing. Watch it, but do not click or type in it.
5. A black window then shows the results. Send the file SigVerify-test-results.txt from the
   kit folder to the team.

No tablet is needed: a virtual tablet plays recorded signings into the app's signing window.
The test makes its own vault in a temporary folder and deletes it at the end; your own vaults
are not touched. The results file holds no user name, machine name or file path.
"""


# ------------------------------------------------------------------ signers
def _write_event_log(path, xy, pressure):
    """One stroke as the signing window logs it: press, moves at 4 ms (250 Hz), release."""
    with open(path, "w", newline="") as f:  # skylos: ignore[SKY-D215,SKY-D324] the tool's own scratch folder for the drawn signer
        w = csv.writer(f)
        w.writerow(["t_ms", "kind", "x", "y", "pressure", "xtilt", "ytilt", "inside"])
        for i, ((x, y), p) in enumerate(zip(xy, pressure)):
            w.writerow([1000 + 4 * i, PRESS if i == 0 else MOVE, f"{x:.3f}", f"{y:.3f}", f"{p:.5f}", 0, 0, 1])
        w.writerow([1000 + 4 * len(xy), RELEASE, f"{xy[-1][0]:.3f}", f"{xy[-1][1]:.3f}", 0, 0, 0, 1])


def _write_signing(folder, name, xy, pressure):
    """The event log of one signing and a scan of it in blue ink on white paper."""
    pen = os.path.join(folder, f"{name}.csv")
    _write_event_log(pen, xy, pressure)
    ink = render_trajectory_to_image(load_inhouse_tablet_file(pen), target_size=(880, 330))[0][..., None]
    png = os.path.join(folder, f"{name}.png")
    cv2.imwrite(png, (255 - ink * (255 - np.array(BLUE_INK_BGR, dtype=np.float32))).astype(np.uint8))
    return png, pen


def synthetic_signings(folder, shape_seed, count):
    """`count` signings of one made-up signature: the same curve, each with its own small wobble."""
    shape = np.random.default_rng(shape_seed)
    freq, phase, amp = shape.uniform(1, 6, (2, 4)), shape.uniform(0, 6.3, (2, 4)), shape.uniform(.3, 1, (2, 4))
    t = np.linspace(0, 1, 700)
    out = []
    for i in range(count):
        a = amp * np.random.default_rng(1000 * shape_seed + i).normal(1, 0.04, amp.shape)
        x = 400 + 800 * t + 100 * (a[0, :, None] * np.sin(2 * np.pi * freq[0, :, None] * t + phase[0, :, None])).sum(0)
        y = 500 - 120 * (a[1, :, None] * np.sin(2 * np.pi * freq[1, :, None] * t + phase[1, :, None])).sum(0)
        out.append(_write_signing(folder, f"s{shape_seed}_{i}", np.c_[x, y], 0.5 + 0.3 * np.sin(2 * np.pi * 3 * t)))
    return out


def synthetic_signer(folder):
    """{"genuine": 12 (scan, pen log) in signing order, "forged": 2 of another signature}."""
    os.makedirs(folder, exist_ok=True)  # skylos: ignore[SKY-D215,SKY-D324] the tool's own scratch folder for the drawn signer
    return {"genuine": synthetic_signings(folder, 1, 12), "forged": synthetic_signings(folder, 2, 2)}


def p01_signer():
    """In-house P01 (cleared for this use): every signing with both paper and tablet, in order."""
    found = next(iter(parse_inhouse(participants=("P01",)).values()))

    def paired(entries):
        return [(e["paper"], e["tablet"]) for e in entries if e.get("paper") and e.get("tablet")]
    return {"genuine": paired(found["genuine"]), "forged": paired(found["forged"])}


# ------------------------------------------------------------------ the model scored directly
def load_models():
    return load_model("spatial", SPATIAL_WEIGHTS_PATH), load_model("temporal", TEMPORAL_WEIGHTS_PATH)


def _embed(models, signings):
    crops = [crop_to_ink(png) for png, _ in signings]
    try:
        scans = embed_samples(models[0], crops, "spatial", "cpu")
    finally:
        for c in crops:
            os.remove(c)
    return scans, embed_samples(models[1], [load_inhouse_tablet_file(pen) for _, pen in signings], "temporal", "cpu")


def direct_scores(models, references, probes):
    """(paper, pen, fused) similarity of each probe: the path every published number uses."""
    def sim(p, r):
        return similarity_from_distance(nearest_reference_distances(p, r))
    (ref_s, ref_t), (s, t) = _embed(models, references), _embed(models, probes)
    paper, pen = sim(s, ref_s), sim(t, ref_t)
    return paper, pen, ALPHA_DEFAULT * paper + (1 - ALPHA_DEFAULT) * pen


# ------------------------------------------------------------------ the kit
def build(kit, signer, models=None):
    """Write the kit folder: signer/ files, expected.json, run scripts, README. Returns kit."""
    folder = os.path.join(kit, "signer")
    os.makedirs(folder, exist_ok=True)  # skylos: ignore[SKY-D215,SKY-D324] the kit folder this tool creates (dist/ or a test's tmp_path)
    k = NUM_ENROLLMENT_REFERENCES
    references, genuine, forged = signer["genuine"][:k], signer["genuine"][k:], signer["forged"]
    named = ([(f"reference_{i + 1}", s) for i, s in enumerate(references)]
             + [(f"genuine_{i + 1}", s) for i, s in enumerate(genuine)]
             + [(f"forgery_{i + 1}", s) for i, s in enumerate(forged)])
    for name, (png, pen) in named:
        shutil.copyfile(png, os.path.join(folder, f"{name}.png"))
        shutil.copyfile(pen, os.path.join(folder, f"{name}.csv"))
    paper, pen, fused = direct_scores(models or load_models(), references, genuine + forged)
    probes = [{"name": name, "png_file": f"signer/{name}.png", "pen_file": f"signer/{name}.csv",
               "genuine": name.startswith("genuine"), "paper": float(a), "pen": float(b), "fused": float(c)}
              for (name, _), a, b, c in zip(named[k:], paper, pen, fused)]
    spec = {"model_version": MODEL_VERSION, "made": time.strftime("%Y-%m-%d"),
            "references": [[f"signer/{n}.png", f"signer/{n}.csv"] for n, _ in named[:k]], "probes": probes}
    with open(os.path.join(kit, "expected.json"), "w", encoding="utf-8") as f:  # skylos: ignore[SKY-D215,SKY-D324] the kit folder this tool creates (dist/ or a test's tmp_path)
        json.dump(spec, f, indent=1)
    with open(os.path.join(kit, "run_test.bat"), "w", newline="\r\n") as f:  # skylos: ignore[SKY-D215,SKY-D324] the kit folder this tool creates (dist/ or a test's tmp_path)
        f.write(RUN_BAT)
    sh = os.path.join(kit, "run_test.sh")
    with open(sh, "w", newline="\n") as f:  # skylos: ignore[SKY-D215,SKY-D324] the kit folder this tool creates (dist/ or a test's tmp_path)
        f.write(RUN_SH)
    os.chmod(sh, os.stat(sh).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with open(os.path.join(kit, "README.txt"), "w", newline="\r\n") as f:  # skylos: ignore[SKY-D215,SKY-D324] the kit folder this tool creates (dist/ or a test's tmp_path)
        f.write(README)
    return kit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p01", action="store_true",
                        help="use in-house P01's real signings (send the zip privately, never publish it)")
    args = parser.parse_args(argv)
    out = os.path.join(PROJECT_ROOT, "dist")
    kit = os.path.join(out, KIT_NAME)
    shutil.rmtree(kit, ignore_errors=True)
    with tempfile.TemporaryDirectory() as scratch:
        build(kit, p01_signer() if args.p01 else synthetic_signer(scratch))
    archive = shutil.make_archive(kit, "zip", out, KIT_NAME)
    print(f"Kit: {archive}" + ("  (real signatures: send privately)" if args.p01 else ""))


if __name__ == "__main__":
    main()
