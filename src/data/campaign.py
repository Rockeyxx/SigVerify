"""Parser for XPPen in-house capture sessions saved by tools/capture_session.py."""
import csv
import hashlib
import json
import os
import re

import numpy as np

from config.settings import (INHOUSE_DIR, INHOUSE_EXCLUDED, INHOUSE_NO_DRIVER,
                             INHOUSE_NO_DRIVER_PRESSURE_SCALE, INHOUSE_SPLIT_PATH,
                             INHOUSE_TEST_FRACTION, SEED)
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
        # Windows sessions save backslash paths; match them to the forward-slash relpath below.
        return {row["file"].replace("\\", "/"): row["os"] or None for row in csv.DictReader(f)
                if row["modality"] == "tablet"
                and (participants is None or row["participant"] in participants)}


def parse_inhouse(root=INHOUSE_DIR, participants=None, exclude=INHOUSE_EXCLUDED,
                  no_driver=INHOUSE_NO_DRIVER, no_driver_scale=INHOUSE_NO_DRIVER_PRESSURE_SCALE):
    """Pair tablet CSVs and paper PNGs per participant by index or (forger, attempt)."""
    _require_dir(root, "In-house campaign")
    os_of = _tablet_os(root, participants)
    writers = {}
    for code in sorted(os.listdir(root)):
        folder = os.path.join(root, code)
        if not (_PARTICIPANT.match(code) and os.path.isdir(folder)):
            continue
        if (participants is not None and code not in participants) or code in exclude:
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
        for entry in genuine.values():
            entry["pressure_scale"] = 1.0
        for entry in forged.values():
            entry["pressure_scale"] = no_driver_scale if code in no_driver else 1.0
        for entry in [*genuine.values(), *forged.values()]:
            entry.setdefault("tablet", None)
            entry.setdefault("paper", None)
            entry["os"] = (os_of.get(os.path.relpath(entry["tablet"], root))
                           if entry["tablet"] else None)
        writers[f"inhouse_{code}"] = {"genuine": [genuine[i] for i in sorted(genuine)],
                                      "forged": [forged[k] for k in sorted(forged)]}
    return writers


def _strokes(rows):
    """Split recorded Qt events into pen-down strokes inside the canvas box."""
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


def load_inhouse_tablet_file(path, pressure_scale=1.0):
    """Load a saved tablet CSV into a normalized [L, 4] trajectory."""
    with open(path, newline="") as f:  # skylos: ignore[SKY-D215,SKY-D325] tablet file enumerated by parse_inhouse from the campaign folder
        strokes = _strokes(csv.DictReader(f))
    if not strokes:
        raise ValueError(f"{path}: no pen-down samples inside the box.")
    return strokes_to_trajectory([np.asarray(s, dtype=np.float64) for s in strokes],
                                 pressure_max=1.0 / pressure_scale)


def _goes_to_test(writer, seed, test_fraction):
    digest = hashlib.sha256(f"{seed}:{writer}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < test_fraction


def inhouse_split(writers, seed=SEED, test_fraction=INHOUSE_TEST_FRACTION):
    """Deterministic hash-based train/test split per participant code."""
    test = sorted(w for w in writers if _goes_to_test(w, seed, test_fraction))
    return {"train": sorted(set(writers) - set(test)), "test": test}


def write_inhouse_split(writers, path=INHOUSE_SPLIT_PATH, seed=SEED,
                        test_fraction=INHOUSE_TEST_FRACTION):
    """Write the in-house train/test split JSON."""
    record = {"seed": seed, "test_fraction": test_fraction,
              "rule": "test when sha256(seed:writer) < test_fraction",
              **inhouse_split(writers, seed, test_fraction)}
    with open(path, "w") as f:  # skylos: ignore[SKY-D215,SKY-D324] INHOUSE_SPLIT_PATH from config/settings.py, or the operator's own path
        json.dump(record, f, indent=1)
    return record
