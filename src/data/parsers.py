"""
Real-Dataset Parsers — Signature Verification System

Each parser returns the same shape, so downstream code never branches on which
corpus it is looking at:

    {writer_id: {"genuine": [...], "forged": [...]}}

Writer IDs are namespaced (`cedar_7`, `bhsig_Hindi_042`, `utsig_3`, `svc_11`)
so several corpora can be merged into one writer-independent split without two
different people colliding on the same integer.

Spatial parsers yield file paths (images are loaded lazily by the Dataset).
Temporal parsers yield [L, 3] float32 arrays — trajectories are small enough
to hold in memory, and eager loading keeps the collate path simple.
"""

import csv
import os
import re

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import (
    BHSIG260_DIR,
    CEDAR_DIR,
    MOBISIG_DIR,
    PRESSURE_IMPUTATION_VALUE,
    SVC2004_DIR,
    UTSIG_DIR,
)

# SVC2004 tablet pressure is reported on a fixed 0–1023 scale. Normalizing by
# this constant (rather than each file's own max) keeps pressure comparable
# across signatures — a light signature must stay light.
SVC_PRESSURE_MAX = 1023.0

_CEDAR_GENUINE = re.compile(r"^original_(\d+)_(\d+)\.png$", re.IGNORECASE)
_CEDAR_FORGED = re.compile(r"^forgeries_(\d+)_(\d+)\.png$", re.IGNORECASE)
_BHSIG_FILE = re.compile(r"^[BH]-S-(\d+)-([GF])-(\d+)\.tif$", re.IGNORECASE)
_SVC_FILE = re.compile(r"^U(\d+)S(\d+)\.TXT$", re.IGNORECASE)
# SIGN_GEN_USER7_USER7_3.csv  -> genuine, writer 7
# SIGN_FOR_USER7_USER2_3.csv  -> forgery of writer 7, drawn by user 2
_MOBISIG_FILE = re.compile(r"^SIGN_(GEN|FOR)_USER(\d+)_USER(\d+)_(\d+)\.csv$",
                           re.IGNORECASE)


def _require_dir(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found: {path}")


def _new_writer():
    return {"genuine": [], "forged": []}


def normalize_trajectory(xy, pressure):
    """
    Normalize a raw trajectory into the model's expected [L, 3] input.

    **The capture layer must call this too.** The tablet emits raw device units
    exactly like the SVC2004 files do; if inference skips this step the model
    sees a completely different input distribution than it trained on and the
    embeddings are meaningless.

    Args:
        xy:       [L, 2] raw X/Y in device units.
        pressure: [L] pressure already scaled into [0, 1], or None to impute.

    Returns:
        float32 [L, 3] — X and Y z-scored per signature, P in [0, 1].

    X/Y are standardized per-signature so the model keys on shape and dynamics
    rather than where on the tablet the person happened to sign, or how large.
    """
    xy = np.asarray(xy, dtype=np.float32)

    mean = xy.mean(axis=0)
    std = xy.std(axis=0)
    # A dot or perfectly straight stroke has zero variance on an axis; guard the divide.
    std = np.where(std < 1e-8, 1.0, std)
    xy = (xy - mean) / std

    if pressure is None:
        pressure = np.full(len(xy), PRESSURE_IMPUTATION_VALUE, dtype=np.float32)
    else:
        pressure = np.clip(np.asarray(pressure, dtype=np.float32), 0.0, 1.0)

    return np.column_stack([xy, pressure]).astype(np.float32)


def merge_corpora(target, incoming):
    """
    Merge one parsed corpus into another, appending samples per writer.

    A plain `dict.update` would replace a writer's whole record when two corpora
    use the same ID, silently discarding one of them. That is not hypothetical:
    SVC2004 Task1 and Task2 are separate recordings of the same 40 signers, so
    both parse to `svc_1`..`svc_40`, and updating would drop 1600 signatures
    without a word.

    Writer IDs stay namespaced per corpus (`cedar_7`, `mobisig_3`), so an ID
    collision means the same person, and appending is the correct reading.

    Returns `target`, mutated in place, so calls can be chained.
    """
    for writer_id, record in incoming.items():
        existing = target.setdefault(writer_id, _new_writer())
        existing["genuine"].extend(record["genuine"])
        existing["forged"].extend(record["forged"])
    return target


def parse_dataset(path):
    """
    Dispatch a dataset directory to its parser by directory name.

    `./data/cedar` → parse_cedar, `./data/svc2004` → parse_svc2004, etc. An
    unrecognized name raises rather than returning an empty dict — silently
    training on zero writers is the worse failure.
    """
    name = os.path.basename(os.path.normpath(path)).lower()
    if name not in _PARSERS:
        raise ValueError(
            f"Unrecognized dataset directory '{name}'. "
            f"Expected one of: {', '.join(sorted(_PARSERS))}. "
            f"Rename the directory to match, or add a parser."
        )
    return _PARSERS[name](path)


def parse_cedar(root=CEDAR_DIR):
    """
    Parse CEDAR: flat `full_org/original_{w}_{s}.png` + `full_forg/forgeries_{w}_{s}.png`.

    55 writers × 24 genuine + 24 skilled forgeries. Non-matching entries
    (notably the `Thumbs.db` left by the Windows packaging) are ignored —
    a loose glob would otherwise invent a phantom writer.
    """
    _require_dir(root, "CEDAR")
    writers = {}

    for subdir, pattern, label in (
        ("full_org", _CEDAR_GENUINE, "genuine"),
        ("full_forg", _CEDAR_FORGED, "forged"),
    ):
        directory = os.path.join(root, subdir)
        if not os.path.isdir(directory):
            continue

        for filename in sorted(os.listdir(directory)):
            match = pattern.match(filename)
            if not match:
                continue
            writer_id = f"cedar_{int(match.group(1))}"
            writers.setdefault(writer_id, _new_writer())
            writers[writer_id][label].append(os.path.join(directory, filename))

    return writers


def parse_bhsig260(root=BHSIG260_DIR, scripts=("Bengali", "Hindi")):
    """
    Parse BHSig260: `{script}/{writer}/X-S-{w}-{G|F}-{n}.tif`.

    100 Bengali + 160 Hindi writers, 24 genuine + 30 forged each. Writer IDs
    are script-scoped because Bengali writer 001 and Hindi writer 001 are
    different people — merging them would break the writer-independent split.
    """
    _require_dir(root, "BHSig260")
    writers = {}

    for script in scripts:
        script_dir = os.path.join(root, script)
        if not os.path.isdir(script_dir):
            continue

        for writer_dir in sorted(os.listdir(script_dir)):
            full_dir = os.path.join(script_dir, writer_dir)
            if not os.path.isdir(full_dir):
                continue  # skips the list.genuine / list.forgery / *_pairs.txt index files

            writer_id = f"bhsig_{script}_{writer_dir}"
            for filename in sorted(os.listdir(full_dir)):
                match = _BHSIG_FILE.match(filename)
                if not match:
                    continue
                label = "genuine" if match.group(2).upper() == "G" else "forged"
                writers.setdefault(writer_id, _new_writer())
                writers[writer_id][label].append(os.path.join(full_dir, filename))

    return writers


def parse_utsig(root=UTSIG_DIR, forgery_types=("Skilled",)):
    """
    Parse UTSig: `Genuine/{w}/{n}.tif` and `Forgery/{type}/{w}/{n}.tif`.

    115 writers × 27 genuine + 6 skilled forgeries. Defaults to Skilled only —
    AGENTS.md reports all metrics under the skilled-forgery scenario, and mixing
    in Simple / Opposite-Hand forgeries would quietly make the numbers look better.
    """
    _require_dir(root, "UTSig")
    writers = {}

    genuine_root = os.path.join(root, "Genuine")
    if os.path.isdir(genuine_root):
        for writer_dir in sorted(os.listdir(genuine_root)):
            full_dir = os.path.join(genuine_root, writer_dir)
            if not os.path.isdir(full_dir):
                continue
            writer_id = f"utsig_{writer_dir}"
            writers.setdefault(writer_id, _new_writer())
            writers[writer_id]["genuine"].extend(
                os.path.join(full_dir, f)
                for f in sorted(os.listdir(full_dir))
                if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
            )

    for forgery_type in forgery_types:
        forgery_root = os.path.join(root, "Forgery", forgery_type)
        if not os.path.isdir(forgery_root):
            continue
        for writer_dir in sorted(os.listdir(forgery_root)):
            full_dir = os.path.join(forgery_root, writer_dir)
            if not os.path.isdir(full_dir):
                continue
            writer_id = f"utsig_{writer_dir}"
            writers.setdefault(writer_id, _new_writer())
            writers[writer_id]["forged"].extend(
                os.path.join(full_dir, f)
                for f in sorted(os.listdir(full_dir))
                if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
            )

    return writers


def load_svc2004_file(path):
    """
    Load one SVC2004 signature into a normalized [L, 3] (X, Y, P) array.

    Auto-detects the task variant by column count:
      - Task1 (4 cols: X Y T button)                     → no pressure, P imputed to 1.0
      - Task2 (7 cols: X Y T button azimuth altitude P)  → real pressure, scaled to [0, 1]

    Either way the extra channels (timestamp, button, azimuth, altitude) are
    dropped: the architecture takes [1, L, 3] and nothing wider.

    The first line is a point count, which is skipped — the actual rows are
    trusted over the declared number.
    """
    with open(path, "r") as fh:
        lines = [line.split() for line in fh.read().splitlines() if line.strip()]

    rows = np.array([[float(v) for v in row] for row in lines[1:]], dtype=np.float32)
    if rows.size == 0:
        raise ValueError(f"No trajectory points in {path}")

    xy = rows[:, :2]
    pressure = rows[:, 6] / SVC_PRESSURE_MAX if rows.shape[1] >= 7 else None
    return normalize_trajectory(xy, pressure)


def svc2004_has_pressure(root=SVC2004_DIR):
    """
    Report whether this SVC2004 copy carries a real pressure channel.

    Task1 downloads do not. Training should warn rather than silently produce a
    permanently pressure-blind temporal model, since pressure-channel dropout
    becomes a no-op when every P is already 1.0.
    """
    _require_dir(root, "SVC2004")
    for filename in sorted(os.listdir(root)):
        if not _SVC_FILE.match(filename):
            continue
        with open(os.path.join(root, filename), "r") as fh:
            fh.readline()  # point-count header
            first = fh.readline().split()
        return len(first) >= 7
    return False


def parse_svc2004(root=SVC2004_DIR):
    """
    Parse SVC2004: flat `U{user}S{sample}.TXT`.

    40 users × 40 signatures. Samples S1–S20 are genuine; S21–S40 are the
    skilled forgeries — the label lives in the filename, not in any index file.
    """
    _require_dir(root, "SVC2004")
    writers = {}

    for filename in sorted(os.listdir(root)):
        match = _SVC_FILE.match(filename)
        if not match:
            continue

        user, sample = int(match.group(1)), int(match.group(2))
        writer_id = f"svc_{user}"
        label = "genuine" if sample <= 20 else "forged"

        writers.setdefault(writer_id, _new_writer())
        writers[writer_id][label].append(load_svc2004_file(os.path.join(root, filename)))

    return writers


def load_mobisig_file(path):
    """
    Load one MOBISIG CSV into the model's [L, 3] input.

    The file carries 13 columns, including velocity, accelerometer and gyroscope
    channels. Only x, y and pressure are read: the capture contract is three
    channels, and the model derives its own dynamics from them.
    """
    xs, ys, ps = [], [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            ps.append(float(row["pressure"]))

    # MOBISIG pressure already arrives in [0, 1]; normalize_trajectory clips it.
    return normalize_trajectory(np.column_stack([xs, ys]), np.asarray(ps))


def parse_mobisig(root=MOBISIG_DIR):
    """
    Parse MOBISIG: `USER{n}/SIGN_{GEN,FOR}_USER{owner}_USER{signer}_{k}.csv`.

    83 users x 45 genuine + 20 skilled forgeries. A forgery filename names two
    users: the one being imitated and the one who drew it. The signature belongs
    to the **owner** being imitated, since that is whose identity the system
    would be deciding on. Filing it under the forger instead would put an
    impostor attempt in that person's genuine reference set.

    These are finger-drawn signatures on a capacitive touchscreen, so pressure
    derives from contact area rather than a pen sensor. Suitable for pre-training
    the temporal branch, not for training the shipped pen-tablet model.
    """
    _require_dir(root, "MOBISIG")
    writers = {}

    for user_dir in sorted(os.listdir(root)):
        user_path = os.path.join(root, user_dir)
        if not os.path.isdir(user_path):
            continue

        for filename in sorted(os.listdir(user_path)):
            match = _MOBISIG_FILE.match(filename)
            if not match:
                continue

            kind, owner = match.group(1), int(match.group(2))
            writer_id = f"mobisig_{owner}"
            label = "genuine" if kind == "GEN" else "forged"

            writers.setdefault(writer_id, _new_writer())
            writers[writer_id][label].append(
                load_mobisig_file(os.path.join(user_path, filename)))

    return writers


# Registered after definition so parse_dataset can dispatch by directory name.
_PARSERS = {
    "cedar": parse_cedar,
    "bhsig260": parse_bhsig260,
    "utsig": parse_utsig,
    # Both SVC2004 tasks use the identical filename convention and the same
    # parser — load_svc2004_file auto-detects 4-column vs 7-column per file.
    "svc2004": parse_svc2004,
    "svc2004_task1": parse_svc2004,
    "svc2004_task2": parse_svc2004,
    "mobisig": parse_mobisig,
}

SPATIAL_DATASETS = ("cedar", "bhsig260", "utsig")
TEMPORAL_DATASETS = ("svc2004", "mobisig")
