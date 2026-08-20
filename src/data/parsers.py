"""
Dataset Parsers — Signature Verification System

Parser layer that outputs a uniform data dictionary format across datasets:

    {writer_id: {"genuine": [...], "forged": [...]}}

Writer IDs use namespaced prefixes (`cedar_7`, `bhsig_Hindi_042`, `utsig_3`, `svc_11`)
to ensure global uniqueness when partitioning writers into independent splits.

Offline parsers return image file paths (loaded lazily by dataset classes).
Online parsers return memory-resident `[L, 3]` float32 trajectory arrays.

For details on parser design, pressure scaling, and dataset schemas, see:
  - SUMMARY.md (Part 1 — Dataset Parsers)
  - AGENTS.md (Capture Contract & Temporal Branch Constraints)
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
    DEEPSIGNDB_DIR,
    MOBISIG_DIR,
    PRESSURE_IMPUTATION_VALUE,
    SIGCOMP2011_DIR,
    SVC2004_DIR,
    UTSIG_DIR,
)

# Standard pressure normalization scale factor for SVC2004 (0–1023 hardware dynamic range).
SVC_PRESSURE_MAX = 1023.0
# Angular tilt boundaries from digitizer specifications (expressed in tenths of a degree).
SVC_AZIMUTH_MAX = 3600.0
SVC_ALTITUDE_MAX = 900.0
# Maximum gap threshold in milliseconds applied to inter-sample duration deltas.
MAX_GAP_MS = 100.0

# Pressure scaling constant for SigComp2011 datasets (WACOM Intuos3 force resolution).
SIGCOMP_PRESSURE_MAX = 1023.0

# Decimation step factor aligning 200 Hz SigComp2011 data to 100 Hz target sampling rate.
SIGCOMP_DECIMATION = 2
# Milliseconds per row *after* decimation — 200 Hz halved. This is what makes
# a dropped pen-up run measurable as elapsed time rather than just a hole.
SIGCOMP_SAMPLE_MS = 10.0

# ─── DeepSignDB ───────────────────────────────────────────────────────
# Mapping definition matching source database distributions across Development and Evaluation partitions.
DEEPSIGN_SOURCE_RANGES = {
    "Development": ((1, 230, "MCYT"), (231, 498, "BiosecurID"),
                    (1009, 1038, "eBioSign_DS1"), (1039, 1084, "eBioSign_DS2")),
    "Evaluation": ((1, 100, "MCYT"), (101, 232, "BiosecurID"),
                   (233, 372, "BiosecureDS2"), (373, 407, "eBioSign_DS1"),
                   (408, 442, "eBioSign_DS2")),
}

# Column index mapping for pen force across constituent source formats.
DEEPSIGN_PRESSURE_COLUMN = {"MCYT": 5, "BiosecurID": 6, "BiosecureDS2": 6,
                            "eBioSign_DS1": 3, "eBioSign_DS2": 3}

# Digitizer pen sources included in default training splits.
DEEPSIGN_PEN_SOURCES = ("MCYT", "BiosecurID", "BiosecureDS2")
DEEPSIGN_PRESSURE_MAX = 1023.0

# Target temporal period in milliseconds (10 ms period = 100 Hz frequency).
DEEPSIGN_TARGET_PERIOD_MS = 10.0

_CEDAR_GENUINE = re.compile(r"^original_(\d+)_(\d+)\.png$", re.IGNORECASE)
_CEDAR_FORGED = re.compile(r"^forgeries_(\d+)_(\d+)\.png$", re.IGNORECASE)
_BHSIG_FILE = re.compile(r"^[BH]-S-(\d+)-([GF])-(\d+)\.tif$", re.IGNORECASE)
_SVC_FILE = re.compile(r"^U(\d+)S(\d+)\.TXT$", re.IGNORECASE)
# SIGN_GEN_USER7_USER7_3.csv  -> genuine, writer 7
# SIGN_FOR_USER7_USER2_3.csv  -> forgery of writer 7, drawn by user 2
_MOBISIG_FILE = re.compile(r"^SIGN_(GEN|FOR)_USER(\d+)_USER(\d+)_(\d+)\.csv$",
                           re.IGNORECASE)
# u0231_g_u1013s0001_sg0001.txt -> genuine, DeepSignDB user 231
# u0231_s_u1013s0001_sg0002.txt -> skilled forgery of that same user
# The tail is the original database's own filename and is not interpreted; the
# ID and the g/s flag are the only fields DeepSignDB guarantees across sources.
_DEEPSIGN_FILE = re.compile(r"^u(\d+)_([gs])_.+\.txt$", re.IGNORECASE)


def _require_dir(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found: {path}")


def _new_writer():
    return {"genuine": [], "forged": []}


def normalize_trajectory(xy, pressure, extra=None):
    """
    Normalize a raw trajectory into the model's expected [L, 3] input.

    **The capture layer must call this too.** The tablet emits raw device units
    exactly like the SVC2004 files do; if inference skips this step the model
    sees a completely different input distribution than it trained on and the
    embeddings are meaningless.

    Args:
        xy:       [L, 2] raw X/Y in device units.
        pressure: [L] pressure already scaled into [0, 1], or None to impute.
        extra:    optional [L, K] of additional channels, ALREADY scaled into
                  roughly [0, 1] by the caller using documented device maxima.
                  Scaling here per-file would make one signature's range depend
                  on its own extremes, the same mistake a per-file pressure max
                  would be.

    Returns:
        float32 [L, 3 + K] — X and Y z-scored per signature, P in [0, 1].

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

    columns = [xy, pressure]
    if extra is not None:
        columns.append(np.asarray(extra, dtype=np.float32))
    return np.column_stack(columns).astype(np.float32)


def merge_datasets(target, incoming):
    """
    Merge one parsed writer_samples into another, appending samples per writer.

    A plain `dict.update` would replace a writer's whole record when two datasets
    use the same ID, silently discarding one of them. That is not hypothetical:
    SVC2004 Task1 and Task2 are separate recordings of the same 40 signers, so
    both parse to `svc_1`..`svc_40`, and updating would drop 1600 signatures
    without a word.

    Writer IDs stay namespaced per writer_samples (`cedar_7`, `mobisig_3`), so an ID
    collision means the same person, and appending is the correct reading.

    Returns `target`, mutated in place, so calls can be chained.
    """
    for writer_id, record in incoming.items():
        existing = target.setdefault(writer_id, _new_writer())
        existing["genuine"].extend(record["genuine"])
        existing["forged"].extend(record["forged"])
    return target


def parse_dataset(path, **kwargs):
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
    # kwargs reach only the parser that declares them; passing
    # extra_channels at an image dataset is a caller error, not a silent no-op.
    return _PARSERS[name](path, **kwargs)


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


def load_svc2004_file(path, extra_channels=None):
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

    extra = None
    if extra_channels and rows.shape[1] < 7:
        # Task1 has four columns and can supply neither channel. Returning [L, 3]
        # here would merge ragged widths into one run instead of failing.
        raise ValueError(
            f"{path} has {rows.shape[1]} columns, so it cannot supply "
            f"'{extra_channels}' — only 7-column Task2 files can."
        )
    if extra_channels:
        if extra_channels == "tilt":
            # WACOM reports both in tenths of a degree against documented maxima
            # (azimuth 0-3600, altitude 0-900), so these are fixed divisors, not
            # per-file statistics.
            extra = np.column_stack([rows[:, 4] / SVC_AZIMUTH_MAX,
                                     rows[:, 5] / SVC_ALTITUDE_MAX])
        elif extra_channels == "dt":
            # Real inter-sample time. The model otherwise assumes a uniform
            # 100 Hz grid, but measured dt has a 10 ms median and a 311 ms tail,
            # so pauses and pen lifts are currently invisible to it.
            gaps = np.zeros(len(rows), dtype=np.float32)
            gaps[1:] = np.diff(rows[:, 2])
            extra = np.clip(gaps, 0.0, MAX_GAP_MS)[:, None] / MAX_GAP_MS
        else:
            raise ValueError(f"Unknown extra_channels '{extra_channels}' "
                             "— expected 'tilt', 'dt' or None")

    return normalize_trajectory(xy, pressure, extra)


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


def parse_svc2004(root=SVC2004_DIR, extra_channels=None):
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
        writers[writer_id][label].append(load_svc2004_file(os.path.join(root, filename), extra_channels))

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


def load_sigcomp2011_file(path, extra_channels=None):
    """
    Load one SigComp2011 `.HWR` file into the model's [L, 3] input.

    The format is three whitespace-separated columns per line — X, Y, Z — with
    no header, where Z is WACOM pen force on the same 0–1023 scale SVC2004 uses.

    Two writer_samples-level alignments are applied here, both of which exist because
    these trajectories are merged with SVC2004 and read by a single encoder.
    Neither widens the [L, 3] contract.

    1. **Decimate 200 Hz to 100 Hz** — see `SIGCOMP_DECIMATION`.
    2. **Drop pen-up samples (Z = 0).** These writers signed on paper laid over
       the tablet, which keeps sensing while the pen travels between the twelve
       preprinted boxes, so about 30% of raw points are in-air moves in runs
       averaging 19 samples. SVC2004 Task2 contains **no** zero-pressure points
       at all (measured: 0 of 100,494). Mixing the two unfiltered would make the
       pressure channel and its derivative mean different things per writer_samples,
       which is precisely what collapsed the earlier SVC2004 Task1 + Task2 merge
       (see `docs/SOLUTIONS.md`).

    The cost is stated plainly: in-air trajectory is discriminative in the
    online-verification literature, and this discards it. It is discarded
    because the dataset the deployment device matches does not have it.

    `extra_channels="dt"` recovers the one thing the drop throws away that the
    model can still use: **how long** the pen was in the air. This archive has no
    clock column, but it samples on a fixed grid, so the elapsed time between two
    surviving pen-down points is just how far apart they sat before filtering.
    That is a measured duration, on the same millisecond scale as SVC2004's real
    timestamps — not a constant standing in for a channel this corpus lacks.
    The distinction matters: a constant would let the model identify the corpus
    instead of the writer, which is the failure recorded against padding
    SigComp with constant tilt.

    `extra_channels="tilt"` is refused rather than ignored. Silently returning
    three channels would let a five-channel run score this corpus and report a
    plausible number instead of failing.
    """
    rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if rows.size == 0:
        raise ValueError(f"No trajectory points in {path}")

    rows = rows[::SIGCOMP_DECIMATION]
    pen_down = np.flatnonzero(rows[:, 2] > 0)
    if len(pen_down) < 2:
        raise ValueError(
            f"{path} has fewer than 2 pen-down samples after alignment — "
            f"the file carries no usable trajectory."
        )

    extra = None
    if extra_channels == "dt":
        # Row distance before filtering *is* elapsed time on a fixed grid: 10 ms
        # inside a stroke, one lift's full duration across a dropped run.
        gaps = np.zeros(len(pen_down), dtype=np.float32)
        gaps[1:] = np.diff(pen_down) * SIGCOMP_SAMPLE_MS
        extra = np.clip(gaps, 0.0, MAX_GAP_MS)[:, None] / MAX_GAP_MS
    elif extra_channels is not None:
        raise ValueError(
            f"SigComp2011 cannot supply '{extra_channels}' — the archive ships "
            f"three columns (X, Y, force). Only 'dt' is reconstructable."
        )

    rows = rows[pen_down]
    return normalize_trajectory(rows[:, :2], rows[:, 2] / SIGCOMP_PRESSURE_MAX,
                                extra)


def _sigcomp_parts(dirpath, root):
    """
    Lowercased path components of `dirpath` *below* `root`.

    Only the components inside the dataset are examined. Matching against the
    absolute path would let a directory anywhere above it decide the label —
    a checkout under `~/forgeries/` would classify the whole writer_samples as forged.
    """
    relative = os.path.relpath(dirpath, root)
    if relative == os.curdir:
        return []
    return [p.lower() for p in relative.split(os.sep)]


def _sigcomp_kind(parts):
    """
    Classify a SigComp2011 directory by the archive layout it belongs to.

    The two shipped archives name their signatures with the *same* tokens in
    *opposite* order — a training genuine is `writer_index` while a test genuine
    is `index_writer` — so `001_23.HWR` and `01_011.HWR` cannot be told apart
    from the filename alone. The containing folder is what disambiguates them,
    and it is a structural fact of the archives rather than a heuristic.

    Returns "train_genuine", "train_forged", "test", or None.
    """
    if any("genuine" in p for p in parts):
        return "train_genuine"
    if any("forg" in p for p in parts):
        return "train_forged"
    # Test signatures sit one level deeper, in a per-writer directory under
    # `Questioned(1259)/` or `Reference(648)/`.
    if any(p.startswith("questioned") or p.startswith("ref") for p in parts):
        return "test"
    return None


def _sigcomp_language(parts):
    """Recover 'Dutch' or 'Chinese' from the path, or None outside both trees."""
    for language in ("Dutch", "Chinese"):
        if language.lower() in parts:
            return language
    return None


def _sigcomp_train_writer(filename, kind):
    """
    Writer ID for a training-set file, or None if the name does not fit.

    Genuine files are `III_NN.HWR`. Forgeries name the forger first, in one of
    two conventions the archive mixes: Dutch splits them (`170_001_02.HWR`)
    while Chinese runs them together (`0120001_1.HWR`, forger 0120, writer 001).
    Taking the last three digits of the second-to-last token covers both.
    """
    tokens = os.path.splitext(filename)[0].split("_")
    if len(tokens) < 2 or not all(t.isdigit() for t in tokens):
        return None

    if kind == "train_genuine":
        return tokens[0] if len(tokens) == 2 and len(tokens[0]) == 3 else None

    writer = tokens[-2][-3:]
    return writer if len(tokens[-2]) in (3, 7) else None


def _sigcomp_test_entry(filename, writer_dir):
    """
    (writer, label) for a test-set file, or None if the name does not fit.

    Genuine is `NN_III.HWR`; a forgery inserts the forger, either as a third
    token (Dutch `01_182_013.HWR`) or glued to the writer (Chinese
    `01_0130011.HWR`). So a two-token name whose second token is the bare
    three-digit writer is the only genuine shape.

    The writer directory is the authority on identity — it is checked against
    the filename rather than trusted blindly, since a mismatch would mean the
    archive was rearranged and every label below it is suspect.
    """
    tokens = os.path.splitext(filename)[0].split("_")
    if len(tokens) < 2 or not all(t.isdigit() for t in tokens):
        return None

    genuine = len(tokens) == 2 and len(tokens[1]) == 3
    writer = tokens[1] if genuine else tokens[-1][-3:]
    if writer != writer_dir:
        raise ValueError(
            f"SigComp2011 file '{filename}' names writer {writer} but sits in "
            f"directory '{writer_dir}'. The archive layout is not as expected."
        )
    return writer, "genuine" if genuine else "forged"


def parse_sigcomp2011(root=SIGCOMP2011_DIR, extra_channels=None):
    """
    Parse the SigComp2011 **online** writer_samples into the standard writer dict.

    84 pen-tablet writers — 64 Dutch, 20 Chinese — recorded on a WACOM Intuos3
    A3 at 200 Hz with real 0–1023 pen force. That capture stack is the same
    family as SVC2004's, which is the reason this dataset is worth having: the
    measured result on this project is that pen accuracy only improves with more
    *pen-domain* writers, and touchscreen writers do not substitute at any
    quantity. It more than triples the pen-domain writer count.

    Both shipped archives extract into `root` and this walks whatever is there,
    so a partial download (training set only) parses fine and simply yields
    fewer writers. Offline PNGs, if extracted alongside, are ignored.

    Writer IDs carry the language (`sigcomp_Dutch_013`) because Dutch writer 013
    and Chinese writer 013 are different people. Within a language the training
    and test author IDs are disjoint, so no further namespacing is needed.
    """
    _require_dir(root, "SigComp2011")
    writers = {}

    for dirpath, _, filenames in os.walk(root):
        parts = _sigcomp_parts(dirpath, root)
        kind = _sigcomp_kind(parts)
        language = _sigcomp_language(parts)
        if kind is None or language is None:
            continue

        writer_dir = os.path.basename(dirpath)
        for filename in sorted(filenames):
            if not filename.lower().endswith(".hwr"):
                continue

            if kind == "test":
                entry = _sigcomp_test_entry(filename, writer_dir)
                if entry is None:
                    continue
                writer, label = entry
            else:
                writer = _sigcomp_train_writer(filename, kind)
                if writer is None:
                    continue
                label = "genuine" if kind == "train_genuine" else "forged"

            writer_id = f"sigcomp_{language}_{writer}"
            writers.setdefault(writer_id, _new_writer())
            writers[writer_id][label].append(
                load_sigcomp2011_file(os.path.join(dirpath, filename),
                                      extra_channels))

    return writers


def deepsigndb_source(user_id, split):
    """
    Which source database a DeepSignDB user ID belongs to, or None if outside
    every documented range.

    The ID alone is not enough — `u0001` is an MCYT signer in both halves, but a
    different MCYT signer in each — so the split has to be named explicitly.
    """
    for low, high, source in DEEPSIGN_SOURCE_RANGES[split]:
        if low <= user_id <= high:
            return source
    return None


def load_deepsigndb_file(path, source, extra_channels=None):
    """
    Load one DeepSignDB signature into the model's [L, 3] input.

    Row 1 is a point count, which is skipped — the actual rows are trusted over
    the declared number, as in SVC2004. Columns 1–3 are X, Y and a millisecond
    timestamp for every source; pen force sits at `DEEPSIGN_PRESSURE_COLUMN[source]`.

    Two alignments are applied, both because these trajectories are merged with
    SVC2004 and read by one encoder. Neither widens the [L, 3] contract.

    1. **Resample to 100 Hz.** The factor comes from this file's own median
       inter-sample gap rather than a per-dataset constant, because BiosecurID
       mixes 100 Hz and 200 Hz captures roughly half and half (see
       `DEEPSIGN_TARGET_PERIOD_MS`). Files already at or below 100 Hz are left
       alone — this only ever drops samples, never invents them.
    2. **Drop pen-up samples (force = 0).** Measured 20–30% of raw points across
       every stylus source, matching SigComp2011's 30% and against SVC2004
       Task2's zero. The same reasoning applies: mixing them unfiltered would
       make the pressure channel and its derivative mean different things per
       dataset. The cost is the same too — in-air trajectory is discriminative
       in the literature, and this discards it.
    """
    rows = np.loadtxt(path, dtype=np.float32, skiprows=1, ndmin=2)
    if rows.size == 0:
        raise ValueError(f"No trajectory points in {path}")

    column = DEEPSIGN_PRESSURE_COLUMN[source]
    if rows.shape[1] <= column:
        raise ValueError(
            f"{path} has {rows.shape[1]} columns but {source} keeps pressure in "
            f"column {column + 1}. The file does not match its documented source."
        )

    if len(rows) >= 3:
        period = float(np.median(np.diff(rows[:, 2])))
        # A zero or negative median means the clock is unusable for rate
        # detection; keeping every sample is the conservative reading.
        if period > 0:
            rows = rows[::max(1, round(DEEPSIGN_TARGET_PERIOD_MS / period))]

    rows = rows[rows[:, column] > 0]
    if len(rows) < 2:
        raise ValueError(
            f"{path} has fewer than 2 pen-down samples after alignment — "
            f"the file carries no usable trajectory."
        )

    extra = None
    if extra_channels == "dt":
        # Unlike SigComp2011 this corpus carries a real millisecond clock, so the
        # gap is read directly rather than reconstructed from dropped rows. Both
        # end up on the same scale, which is what lets the three corpora train
        # one encoder.
        gaps = np.zeros(len(rows), dtype=np.float32)
        gaps[1:] = np.diff(rows[:, 2])
        extra = np.clip(gaps, 0.0, MAX_GAP_MS)[:, None] / MAX_GAP_MS
    elif extra_channels is not None:
        raise ValueError(
            f"DeepSignDB cannot supply '{extra_channels}' — only 'dt' is "
            f"available from its columns."
        )

    return normalize_trajectory(rows[:, :2], rows[:, column] / DEEPSIGN_PRESSURE_MAX,
                                extra)


def parse_deepsigndb(root=DEEPSIGNDB_DIR, split="Development", modality="stylus",
                     sources=DEEPSIGN_PEN_SOURCES, extra_channels=None):
    """
    Parse DeepSignDB: `DeepSignDB/{split}/{modality}/u{id}_{g|s}_*.txt`.

    Defaults to the **Development** half, **stylus**, **Wacom pen sources** —
    498 writers, the largest pen-domain corpus this project has. The defaults are
    what training uses; the arguments exist so the held-out half stays reachable
    without editing code.

    `split="Evaluation"` is the authors' held-out benchmark and the split every
    published DeepSignDB result is reported on. Training on it would forfeit the
    only directly comparable external number this project can quote, so it is
    never a default and never merged with Development.

    `modality="finger"` is touchscreen capture by the same people as the stylus
    half. It carries no pressure at all (measured: the force column is 0 in every
    finger file), which puts it in MOBISIG's domain rather than the pen tablet's.
    Writer IDs deliberately omit the modality, so if both are ever parsed and
    merged, one person's stylus and finger signatures land on one writer — which
    is what they are.

    Raises on an unknown source name rather than silently yielding fewer writers:
    a typo in `sources` would otherwise read as "this corpus is smaller than
    documented".
    """
    unknown = set(sources) - set(DEEPSIGN_PRESSURE_COLUMN)
    if unknown:
        raise ValueError(
            f"Unknown DeepSignDB source(s) {sorted(unknown)}. "
            f"Expected any of: {', '.join(sorted(DEEPSIGN_PRESSURE_COLUMN))}."
        )
    if split not in DEEPSIGN_SOURCE_RANGES:
        raise ValueError(f"Unknown DeepSignDB split '{split}' — "
                         f"expected 'Development' or 'Evaluation'.")

    directory = os.path.join(root, "DeepSignDB", split, modality)
    _require_dir(directory, f"DeepSignDB {split}/{modality}")

    split_tag = "dev" if split == "Development" else "eval"
    wanted = set(sources)
    writers = {}

    for filename in sorted(os.listdir(directory)):
        match = _DEEPSIGN_FILE.match(filename)
        if not match:
            continue

        source = deepsigndb_source(int(match.group(1)), split)
        if source not in wanted:
            continue

        # `deepsign_MCYT-dev_0001` — the source keeps the per-dataset breakdown
        # meaningful (benchmark.py groups on the second token) and the split tag
        # keeps Development and Evaluation from ever colliding.
        writer_id = f"deepsign_{source}-{split_tag}_{int(match.group(1)):04d}"
        label = "genuine" if match.group(2).lower() == "g" else "forged"

        writers.setdefault(writer_id, _new_writer())
        writers[writer_id][label].append(
            load_deepsigndb_file(os.path.join(directory, filename), source,
                                 extra_channels))

    return writers


def parse_sigcomp2011_offline(root=None):
    """
    Parse ICDAR SigComp2011 offline scanned image signature dataset.

    Returns:
        {writer_id: {"genuine": [img_path_strings], "forged": [img_path_strings]}}
    """
    if root is None:
        root = os.path.join("data", "sigcomp2011_offline")
    _require_dir(root, "SigComp2011 Offline")

    writers = {}
    base_dir = os.path.join(root, "trainingSet", "OfflineSignatures")
    if not os.path.exists(base_dir):
        base_dir = root

    for script in ("Chinese", "Dutch"):
        script_dir = os.path.join(base_dir, script, "TrainingSet")
        if not os.path.exists(script_dir):
            continue

        gen_dir = os.path.join(script_dir, "Offline Genuine")
        forg_dir = os.path.join(script_dir, "Offline Forgeries")

        if os.path.exists(gen_dir):
            for fname in sorted(os.listdir(gen_dir)):
                if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg")):
                    continue
                prefix = fname.split('_')[0]
                try:
                    w_num = int(prefix)
                    w_id = f"sigcomp11_off_{script}_{w_num:03d}"
                    writers.setdefault(w_id, _new_writer())["genuine"].append(os.path.join(gen_dir, fname))
                except ValueError:
                    continue

        if os.path.exists(forg_dir):
            for fname in sorted(os.listdir(forg_dir)):
                if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg")):
                    continue
                prefix = fname.split('_')[0]
                try:
                    w_num = int(prefix[-3:])
                    w_id = f"sigcomp11_off_{script}_{w_num:03d}"
                    writers.setdefault(w_id, _new_writer())["forged"].append(os.path.join(forg_dir, fname))
                except ValueError:
                    continue

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
    "sigcomp2011": parse_sigcomp2011,
    "sigcomp2011_offline": parse_sigcomp2011_offline,
    "deepsigndb": parse_deepsigndb,
}

# Which corpora can supply which extra temporal channel, keyed exactly as
# `_PARSERS` is — a substring test would let `sigcomp2011_offline` claim its
# online sibling's channels.
#
# Every corpus merged into one run has to return the same width, so a channel is
# usable only when *every* requested corpus appears here with it. `dt` is the
# only channel two corpora share: SVC2004 measures it against a real clock,
# SigComp2011 reconstructs it from the length of the pen-up runs it drops.
# SVC2004 Task1 is absent because its four columns carry neither.
EXTRA_CHANNEL_SUPPORT = {
    "svc2004": frozenset({"tilt", "dt"}),
    "svc2004_task2": frozenset({"tilt", "dt"}),
    "sigcomp2011": frozenset({"dt"}),
    "deepsigndb": frozenset({"dt"}),
}


def is_known_dataset(path):
    """
    Whether `path`'s basename dispatches to a parser.

    Callers validating channels need this so an unrecognized directory reports
    *that*, rather than a confusing complaint about a channel it was never going
    to supply either way.
    """
    return os.path.basename(os.path.normpath(path)).lower() in _PARSERS


def supports_extra_channel(path, channel):
    """Whether the corpus rooted at `path` can supply `channel`."""
    name = os.path.basename(os.path.normpath(path)).lower()
    return channel in EXTRA_CHANNEL_SUPPORT.get(name, frozenset())


SPATIAL_DATASETS = ("cedar", "bhsig260", "utsig", "sigcomp2011_offline")
TEMPORAL_DATASETS = ("svc2004", "mobisig", "sigcomp2011", "deepsigndb")

