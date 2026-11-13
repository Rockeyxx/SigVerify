"""The XPPen in-house campaign as data: what tools/capture_session.py saved, read back.

Files per participant (tools/capture_session.py `signature_path`):
  P05/P05_tablet_genuine_01.csv              every Qt tablet event the signing box saw
  P05/P05_paper_genuine_01.png               the raw paper box (its _ink.png crop is ignored:
                                             callers apply the app's own image preparation)
  P05/P05_<tablet|paper>_forgery_byP02_a1.*  forger P02's attempt 1 at P05
The forger copy has no pen pair and is ignored; so is tilt (dropped by the owner, ENG_REPORT B11).
"""
import csv
import hashlib
import json
import os
import re

import numpy as np

from config.settings import INHOUSE_DIR, INHOUSE_SPLIT_PATH, INHOUSE_TEST_FRACTION, SEED
from src.data.parsers import _require_dir
from src.data.pen_stream import strokes_to_trajectory

_PARTICIPANT = re.compile(r"^P\d{2,3}$")
_FILE = re.compile(r"^(P\d{2,3})_(tablet|paper)_"
                   r"(?:genuine_(\d{2})|forgery_by(P\d{2,3})_a(\d+))\.(csv|png)$")
_EXT = {"tablet": "csv", "paper": "png"}

# The QEvent types the canvas logs in the `kind` column (Qt 5).
_PRESS, _MOVE, _RELEASE = 92, 87, 93


def _tablet_os(root, participants):
    """The OS of each tablet file, from the manifest row written when it was saved."""
    path = os.path.join(root, "manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:  # skylos: ignore[SKY-D215,SKY-D325] manifest.csv inside the operator-supplied campaign folder
        return {row["file"]: row["os"] or None for row in csv.DictReader(f)
                if row["modality"] == "tablet"
                and (participants is None or row["participant"] in participants)}


def parse_inhouse(root=INHOUSE_DIR, participants=None):
    """Pair each pen signing with its paper box, by the number in the file name.

    Returns {"inhouse_P05": {"genuine": [{"index", "tablet", "paper", "os"}], "forged":
    [{"forger", "attempt", "tablet", "paper", "os"}]}}, sorted by index or (forger, attempt).
    A side that was not saved is None: never shifted, so a gap cannot misalign the pairs.
    "os" is the pen file's own manifest entry (None without a pen file): Windows pressure is on
    another scale (ENG_REPORT G10), so never pool it unscaled. participants limits the read.
    """
    _require_dir(root, "In-house campaign")
    os_of = _tablet_os(root, participants)
    writers = {}
    for code in sorted(os.listdir(root)):
        folder = os.path.join(root, code)
        if not (_PARTICIPANT.match(code) and os.path.isdir(folder)):
            continue
        if participants is not None and code not in participants:
            continue
        genuine, forged = {}, {}
        for name in sorted(os.listdir(folder)):
            match = _FILE.match(name)
            if not match:
                continue                         # ink crops, forger copy, anything else
            owner, modality, index, forger, attempt, ext = match.groups()
            if owner != code:
                raise ValueError(f"{name} is in {code}'s folder but belongs to {owner}.")
            if ext != _EXT[modality]:
                raise ValueError(f"{name}: a {modality} file must be .{_EXT[modality]}.")
            if index:
                entry = genuine.setdefault(int(index), {"index": int(index)})
            else:
                key = (forger, int(attempt))
                entry = forged.setdefault(key, {"forger": forger, "attempt": int(attempt)})
            entry[modality] = os.path.join(folder, name)
        if not (genuine or forged):
            continue
        for entry in [*genuine.values(), *forged.values()]:
            entry.setdefault("tablet", None)
            entry.setdefault("paper", None)
            entry["os"] = (os_of.get(os.path.relpath(entry["tablet"], root))
                           if entry["tablet"] else None)
        writers[f"inhouse_{code}"] = {"genuine": [genuine[i] for i in sorted(genuine)],
                                      "forged": [forged[k] for k in sorted(forged)]}
    return writers


def _strokes(rows):
    """The signing box's own rule (src/capture/tablet.py DrawingCanvas.tabletEvent): a press
    starts a new stroke, a lift or leaving the box ends one, and only samples with pressure
    inside the box are ink."""
    strokes, current = [], []

    def end():
        nonlocal current
        if current:
            strokes.append(current)
            current = []

    for row in rows:
        kind, inside, p = int(row["kind"]), int(row["inside"]), float(row["pressure"])
        if kind == _PRESS:
            end()
        if kind in (_PRESS, _MOVE):
            if p > 0.0 and inside:
                current.append((float(row["t_ms"]), float(row["x"]), float(row["y"]), p))
            elif not inside:
                end()
        elif kind == _RELEASE:
            end()
    end()
    return strokes


def load_inhouse_tablet_file(path):
    """One saved tablet signature as the [L, 4] trajectory the app made of it at signing."""
    with open(path, newline="") as f:  # skylos: ignore[SKY-D215,SKY-D325] tablet file enumerated by parse_inhouse from the campaign folder
        strokes = _strokes(csv.DictReader(f))
    if not strokes:
        raise ValueError(f"{path}: no pen-down samples inside the box.")
    # Qt pressure is already 0..1, as in DrawingCanvas.get_final_trajectory_tensor.
    return strokes_to_trajectory([np.asarray(s, dtype=np.float64) for s in strokes],
                                 pressure_max=1.0)


def _goes_to_test(writer, seed, test_fraction):
    digest = hashlib.sha256(f"{seed}:{writer}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < test_fraction


def inhouse_split(writers, seed=SEED, test_fraction=INHOUSE_TEST_FRACTION):
    """Train and test writers. Every forger who is also a writer is test: their own hand in
    training would teach the model their forgeries. Each other writer's side depends only on
    their own code, so nobody moves when volunteers are added (splits must not shift)."""
    forgers = sorted({f"inhouse_{f['forger']}" for w in writers.values() for f in w["forged"]}
                     & set(writers))
    test = sorted(w for w in writers
                  if w in forgers or _goes_to_test(w, seed, test_fraction))
    return {"train": sorted(set(writers) - set(test)), "test": test, "forgers": forgers}


def write_inhouse_split(writers, path=INHOUSE_SPLIT_PATH, seed=SEED,
                        test_fraction=INHOUSE_TEST_FRACTION):
    """Save the split for training to hold out: `train.py --holdout-writers <test...>`."""
    record = {"seed": seed, "test_fraction": test_fraction,
              "rule": "every forger is test; others by sha256(seed:writer) < test_fraction",
              **inhouse_split(writers, seed, test_fraction)}
    with open(path, "w") as f:  # skylos: ignore[SKY-D215,SKY-D324] INHOUSE_SPLIT_PATH from config/settings.py, or the operator's own path
        json.dump(record, f, indent=1)
    return record
