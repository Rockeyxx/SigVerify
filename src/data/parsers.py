"""Dataset parsers for signature verification.

This module provides dataset parsers that convert signature files into a
uniform dictionary format:

    {writer_id: {"genuine": [...], "forged": [...]}}

Writer IDs use prefixes (such as `cedar_7`, `bhsig_Hindi_042`, `utsig_3`, `svc_11`)
so that writer IDs remain unique across different datasets.

Offline parsers return file paths to images (loaded when needed by dataset classes).
Online parsers return numpy float32 trajectory arrays with shape [L, C].
"""

import csv
import os
import re

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import (
    BHSIG260_DIR,
    BIOSECURID_SONOF_DIR,
    CEDAR_DIR,
    DEEPSIGNDB_DIR,
    MOBISIG_DIR,
    PRESSURE_IMPUTATION_VALUE,
    SIGCOMP2011_DIR,
    SVC2004_DIR,
    TEMPORAL_SAMPLE_PERIOD_MS,
    UTSIG_DIR,
)

# Standard maximum pressure scale for SVC2004 digitizers.
SVC_PRESSURE_MAX = 1023.0
# Maximum tilt angles in tenths of a degree from digitizer hardware specifications.
SVC_AZIMUTH_MAX = 3600.0
SVC_ALTITUDE_MAX = 900.0
# Maximum gap threshold in milliseconds applied to inter-sample duration deltas.
def max_gap_ms_from_env(value, default):
    """Parse SIGVER_MAX_GAP_MS environment override for the dt time gap limit."""
    if value is None:
        return default
    try:
        gap = float(value)
    except ValueError:
        gap = -1.0
    if not gap > 0:
        raise ValueError(f"SIGVER_MAX_GAP_MS must be a positive number of milliseconds; got {value!r}")
    return gap


MAX_GAP_MS = max_gap_ms_from_env(os.environ.get("SIGVER_MAX_GAP_MS"), 100.0)

# Maximum pen force value for SigComp2011 digitizers.
SIGCOMP_PRESSURE_MAX = 1023.0

# Decimation factor to align 200 Hz SigComp2011 recordings to the target 100 Hz rate.
SIGCOMP_DECIMATION = 2
# Sampling period in milliseconds after decimation.
SIGCOMP_SAMPLE_MS = TEMPORAL_SAMPLE_PERIOD_MS

# DeepSignDB writer ID ranges and corresponding source sub-datasets.
DEEPSIGN_SOURCE_RANGES = {
    "Development": ((1, 230, "MCYT"), (231, 498, "BiosecurID"),
                    (1009, 1038, "eBioSign_DS1"), (1039, 1084, "eBioSign_DS2")),
    "Evaluation": ((1, 100, "MCYT"), (101, 232, "BiosecurID"),
                   (233, 372, "BiosecureDS2"), (373, 407, "eBioSign_DS1"),
                   (408, 442, "eBioSign_DS2")),
}

# Column index for pen pressure in each sub-dataset.
DEEPSIGN_PRESSURE_COLUMN = {"MCYT": 5, "BiosecurID": 6, "BiosecureDS2": 6,
                            "eBioSign_DS1": 3, "eBioSign_DS2": 3}

# Sub-datasets collected using pen tablets.
DEEPSIGN_PEN_SOURCES = ("MCYT", "BiosecurID", "BiosecureDS2")
DEEPSIGN_PRESSURE_MAX = 1023.0

# Target sampling period in milliseconds.
DEEPSIGN_TARGET_PERIOD_MS = TEMPORAL_SAMPLE_PERIOD_MS

_CEDAR_GENUINE = re.compile(r"^original_(\d+)_(\d+)\.png$", re.IGNORECASE)
_CEDAR_FORGED = re.compile(r"^forgeries_(\d+)_(\d+)\.png$", re.IGNORECASE)
_BHSIG_FILE = re.compile(r"^[BH]-S-(\d+)-([GF])-(\d+)\.tif$", re.IGNORECASE)
_SVC_FILE = re.compile(r"^U(\d+)S(\d+)\.TXT$", re.IGNORECASE)
_MOBISIG_FILE = re.compile(r"^SIGN_(GEN|FOR)_USER(\d+)_USER(\d+)_(\d+)\.csv$",
                           re.IGNORECASE)
_DEEPSIGN_FILE = re.compile(r"^u(\d+)_([gs])_.+\.txt$", re.IGNORECASE)
# BiosecurID names carry their session ("u1202s0003_sg0001"); MCYT names carry none.
_DEEPSIGN_SESSION = re.compile(r"_u\d+s(\d{4})_sg\d+\.txt$", re.IGNORECASE)


def _require_dir(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found: {path}")


def _new_writer():
    return {"genuine": [], "forged": []}


def normalize_trajectory(xy, pressure, extra=None):
    """Normalize raw trajectory coordinates and pressure into a standard format.

    Args:
        xy: [L, 2] raw coordinates in device units.
        pressure: [L] pressure values scaled to [0.0, 1.0], or None.
        extra: optional [L, K] extra channels scaled to [0.0, 1.0].

    Returns:
        float32 array with shape [L, 3 + K] where X and Y are z-scored per signature.
    """
    xy = np.asarray(xy, dtype=np.float32)

    mean = xy.mean(axis=0)
    std = xy.std(axis=0)
    # Avoid division by zero if variance is zero.
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
    """Merge incoming writer samples into target dictionary, appending samples per writer.

    "genuine_session", when either side has it, stays aligned with "genuine";
    samples whose corpus records no session get None.
    """
    for writer_id, record in incoming.items():
        existing = target.setdefault(writer_id, _new_writer())
        if "genuine_session" in existing or "genuine_session" in record:
            sessions = existing.get("genuine_session", [None] * len(existing["genuine"]))
            sessions.extend(record.get("genuine_session", [None] * len(record["genuine"])))
            existing["genuine_session"] = sessions
        existing["genuine"].extend(record["genuine"])
        existing["forged"].extend(record["forged"])
    return target


def parse_dataset(path, float32_parse=False, **kwargs):
    """Dispatch dataset folder to its parser based on folder name.

    float32_parse reaches only the parsers that read a device clock (SVC2004 and
    DeepSignDB); every other corpus is read the same either way.
    """
    name = os.path.basename(os.path.normpath(path)).lower()
    if name not in _PARSERS:
        raise ValueError(
            f"Unrecognized dataset directory '{name}'. "
            f"Expected one of: {', '.join(sorted(_PARSERS))}. "
            f"Rename the directory to match, or add a parser."
        )
    parser = _PARSERS[name]
    if float32_parse and parser in (parse_svc2004, parse_deepsigndb):
        kwargs["float32_parse"] = True
    return parser(path, **kwargs)


def parse_cedar(root=CEDAR_DIR):
    """Parse the offline CEDAR signature dataset."""
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
    """Parse the offline BHSig260 signature dataset."""
    _require_dir(root, "BHSig260")
    writers = {}

    for script in scripts:
        script_dir = os.path.join(root, script)
        if not os.path.isdir(script_dir):
            continue

        for writer_dir in sorted(os.listdir(script_dir)):
            full_dir = os.path.join(script_dir, writer_dir)
            if not os.path.isdir(full_dir):
                continue

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
    """Parse the offline UTSig signature dataset."""
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


def load_svc2004_file(path, extra_channels=None, float32_parse=False):
    """Load an SVC2004 signature file into a normalized trajectory array.

    Supports 4-column Task1 (X, Y, T, Button) and 7-column Task2 (with pressure and angles).
    float32_parse reads the rows as float32, as training does (TEMPORAL_TRAIN_FLOAT32_PARSE).
    """
    with open(path, "r") as fh:  # skylos: ignore[SKY-D215,SKY-D325] corpus file enumerated from --data-dir
        lines = [line.split() for line in fh.read().splitlines() if line.strip()]

    # float64: the clock reaches 7.5e7 ms, past the 2**24 where float32 stops holding every integer.
    rows = np.array([[float(v) for v in row] for row in lines[1:]],
                    dtype=np.float32 if float32_parse else np.float64)
    if rows.size == 0:
        raise ValueError(f"No trajectory points in {path}")

    xy = rows[:, :2]
    pressure = rows[:, 6] / SVC_PRESSURE_MAX if rows.shape[1] >= 7 else None

    extra = None
    if extra_channels and rows.shape[1] < 7:
        raise ValueError(
            f"{path} has {rows.shape[1]} columns, so it cannot supply "
            f"'{extra_channels}' — only 7-column Task2 files can."
        )
    if extra_channels:
        if extra_channels == "tilt":
            extra = np.column_stack([rows[:, 4] / SVC_AZIMUTH_MAX,
                                     rows[:, 5] / SVC_ALTITUDE_MAX])
        elif extra_channels == "dt":
            gaps = np.zeros(len(rows), dtype=np.float32)
            gaps[1:] = np.diff(rows[:, 2])
            extra = np.clip(gaps, 0.0, MAX_GAP_MS)[:, None] / MAX_GAP_MS
        else:
            raise ValueError(f"Unknown extra_channels '{extra_channels}' "
                             "— expected 'tilt', 'dt' or None")

    return normalize_trajectory(xy, pressure, extra)


def svc2004_has_pressure(root=SVC2004_DIR):
    """Check if the SVC2004 dataset files include a pressure column."""
    _require_dir(root, "SVC2004")
    for filename in sorted(os.listdir(root)):
        if not _SVC_FILE.match(filename):
            continue
        with open(os.path.join(root, filename), "r") as fh:  # skylos: ignore[SKY-D215,SKY-D325] filename matched _SVC_FILE
            fh.readline()
            first = fh.readline().split()
        return len(first) >= 7
    return False


def parse_svc2004(root=SVC2004_DIR, extra_channels=None, float32_parse=False):
    """Parse the SVC2004 online signature dataset."""
    _require_dir(root, "SVC2004")
    writers = {}

    for filename in sorted(os.listdir(root)):
        match = _SVC_FILE.match(filename)
        if not match:
            continue

        user, sample = int(match.group(1)), int(match.group(2))
        writer_id = f"svc_{user}"
        label = "genuine" if sample <= 20 else "forged"

        record = writers.setdefault(writer_id, {**_new_writer(), "genuine_session": []})
        record[label].append(load_svc2004_file(os.path.join(root, filename), extra_channels,
                                               float32_parse))
        if label == "genuine":
            # Ten genuines per session, recorded about two weeks apart (D8).
            record["genuine_session"].append(1 if sample <= 10 else 2)

    return writers


def load_mobisig_file(path):
    """Load a MOBISIG CSV signature file into a normalized trajectory array."""
    xs, ys, ps = [], [], []
    with open(path, newline="") as handle:  # skylos: ignore[SKY-D215,SKY-D325] corpus file enumerated from --data-dir
        for row in csv.DictReader(handle):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            ps.append(float(row["pressure"]))

    return normalize_trajectory(np.column_stack([xs, ys]), np.asarray(ps))


def parse_mobisig(root=MOBISIG_DIR):
    """Parse the MOBISIG touchscreen signature dataset."""
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
    """Load a SigComp2011 online signature file into a normalized trajectory array."""
    rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if rows.size == 0:
        raise ValueError(f"No trajectory points in {path}")
    return _sigcomp_rows_to_sample(rows, extra_channels, path, "SigComp2011")


def _sigcomp_rows_to_sample(rows, extra_channels, path, corpus, decimation=SIGCOMP_DECIMATION):
    """Process raw rows from SigComp collections into a normalized trajectory sample."""
    rows = rows[::decimation]
    pen_down = np.flatnonzero(rows[:, 2] > 0)
    if len(pen_down) < 2:
        raise ValueError(
            f"{path} has fewer than 2 pen-down samples after alignment — "
            f"the file carries no usable trajectory."
        )

    extra = None
    if extra_channels == "dt":
        # Compute elapsed time between pen-down samples based on sampling period.
        gaps = np.zeros(len(pen_down), dtype=np.float32)
        gaps[1:] = np.diff(pen_down) * SIGCOMP_SAMPLE_MS
        extra = np.clip(gaps, 0.0, MAX_GAP_MS)[:, None] / MAX_GAP_MS
    elif extra_channels is not None:
        raise ValueError(
            f"{corpus} cannot supply '{extra_channels}' — the archive ships "
            f"three columns (X, Y, force). Only 'dt' is reconstructable."
        )

    rows = rows[pen_down]
    return normalize_trajectory(rows[:, :2], rows[:, 2] / SIGCOMP_PRESSURE_MAX,
                                extra)


def load_nfi2009_file(path, extra_channels=None):
    """Load a SigComp2009 online signature file into a normalized trajectory array."""
    with open(path, encoding="latin-1") as handle:
        lines = handle.read().splitlines()
    points = [line.split() for line in lines[1:]]
    rows = np.array([[float(c) for c in cols[:3]] for cols in points if len(cols) == 5],
                    dtype=np.float32)
    if rows.size == 0:
        raise ValueError(f"No trajectory points in {path}")
    return _sigcomp_rows_to_sample(rows, extra_channels, path, "SigComp2009")


# BiosecurID-SONOF: Wacom Intuos3 at 100 Hz (already the target rate), x, y and
# pressure only, pen-up recorded as pressure 0, so the SigComp core applies undecimated.
SONOF_ONLINE_ID_OFFSET = 1000          # pen file u1001 is scan folder u0001
SONOF_GENUINE_SLOTS = (1, 2, 6, 7)     # per session; slots 3 to 5 are skilled forgeries
_SONOF_ONLINE_FILE = re.compile(r"^u(\d{4})(s\d{4}_sg(\d{4}))\.mat$", re.IGNORECASE)


def load_sonof_online_file(path, extra_channels=None):
    """Load a BiosecurID-SONOF pen recording (.mat with x, y, p) into a normalized trajectory array."""
    from scipy.io import loadmat
    mat = loadmat(path)
    rows = np.column_stack([np.ravel(mat[key]) for key in ("x", "y", "p")]).astype(np.float32)
    return _sigcomp_rows_to_sample(rows, extra_channels, path, "BiosecurID-SONOF", decimation=1)


def parse_biosecurid_sonof_pairs(root=BIOSECURID_SONOF_DIR):
    """Pair each SONOF pen recording with the scan of the same signing.

    Returns {writer_id: {"genuine": [(png, mat)], "forged": [(png, mat)]}}. A
    missing scan, or one whose label disagrees with its slot, raises: a silent
    gap or a shifted join would corrupt every fused number built on it.
    """
    online = os.path.join(root, "OnlineReal")
    _require_dir(online, "BiosecurID-SONOF OnlineReal")
    writers = {}
    for filename in sorted(os.listdir(online)):
        match = _SONOF_ONLINE_FILE.match(filename)
        if not match:
            continue
        pen_user, stem, slot = int(match.group(1)), match.group(2), int(match.group(3))
        scan_user = pen_user - SONOF_ONLINE_ID_OFFSET
        genuine = slot in SONOF_GENUINE_SLOTS
        scan_dir = os.path.join(root, "OfflineReal", f"u{scan_user:04d}")
        scan = os.path.join(scan_dir, f"u{scan_user:04d}{stem}{'g' if genuine else 'f'}.png")
        if not os.path.exists(scan):
            if os.path.exists(scan[:-5] + ("f" if genuine else "g") + ".png"):
                raise ValueError(f"Scan for {filename} carries the wrong label for slot {slot}.")
            raise FileNotFoundError(f"No scan for {filename}: expected u{scan_user:04d}{stem}")
        record = writers.setdefault(f"sonof_u{pen_user:04d}", _new_writer())
        record["genuine" if genuine else "forged"].append((scan, os.path.join(online, filename)))
    return writers


_SONOF_SCAN_FILE = re.compile(r"^u(\d{4})s\d{4}_sg(\d{4})([gf])\.png$", re.IGNORECASE)


def parse_biosecurid_sonof_scans(root=BIOSECURID_SONOF_DIR):
    """Parse the SONOF scans (OfflineReal) as an offline corpus.

    Writer IDs follow the pen files (`sonof_u1001` is scan folder u0001), so a
    writer list carries over to the fused pairs. A label that disagrees with its
    slot raises, as in parse_biosecurid_sonof_pairs.
    """
    offline = os.path.join(root, "OfflineReal")
    _require_dir(offline, "BiosecurID-SONOF OfflineReal")
    writers = {}
    for writer_dir in sorted(os.listdir(offline)):
        full_dir = os.path.join(offline, writer_dir)
        if not os.path.isdir(full_dir):
            continue
        for filename in sorted(os.listdir(full_dir)):
            match = _SONOF_SCAN_FILE.match(filename)
            if not match:
                continue
            genuine = int(match.group(2)) in SONOF_GENUINE_SLOTS
            if genuine != (match.group(3).lower() == "g"):
                raise ValueError(f"Scan {filename} carries the wrong label for its slot.")
            record = writers.setdefault(
                f"sonof_u{int(match.group(1)) + SONOF_ONLINE_ID_OFFSET:04d}", _new_writer())
            record["genuine" if genuine else "forged"].append(os.path.join(full_dir, filename))
    return writers


def _sigcomp_parts(dirpath, root):
    """Return lowercased relative directory path components below root."""
    relative = os.path.relpath(dirpath, root)
    if relative == os.curdir:
        return []
    return [p.lower() for p in relative.split(os.sep)]


def _sigcomp_kind(parts):
    """Determine the subset kind of a SigComp directory."""
    if any("genuine" in p for p in parts):
        return "train_genuine"
    if any("forg" in p for p in parts):
        return "train_forged"
    if any(p.startswith("questioned") or p.startswith("ref") for p in parts):
        return "test"
    return None


def _sigcomp_language(parts):
    """Return 'Dutch' or 'Chinese' based on path components, or None."""
    for language in ("Dutch", "Chinese"):
        if language.lower() in parts:
            return language
    return None


def _sigcomp_train_writer(filename, kind):
    """Extract writer ID for a training file in SigComp2011."""
    tokens = os.path.splitext(filename)[0].split("_")
    if len(tokens) < 2 or not all(t.isdigit() for t in tokens):
        return None

    if kind == "train_genuine":
        return tokens[0] if len(tokens) == 2 and len(tokens[0]) == 3 else None

    writer = tokens[-2][-3:]
    return writer if len(tokens[-2]) in (3, 7) else None


def _sigcomp_test_entry(filename, writer_dir):
    """Extract (writer, label) from a test file in SigComp2011."""
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
    """Parse the SigComp2011 online signature dataset."""
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
    """Return the source database name for a DeepSignDB user ID, or None."""
    for low, high, source in DEEPSIGN_SOURCE_RANGES[split]:
        if low <= user_id <= high:
            return source
    return None


def load_deepsigndb_file(path, source, extra_channels=None, float32_parse=False):
    """Load a DeepSignDB trajectory file into a normalized array.

    float32_parse reads the rows as float32, as training does (TEMPORAL_TRAIN_FLOAT32_PARSE).
    """
    # float64: timestamps pass 2**24 ms, where float32 stops holding every integer.
    rows = np.loadtxt(path, dtype=np.float32 if float32_parse else np.float64, skiprows=1, ndmin=2)
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
        # Calculate time delta between consecutive samples.
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
                     sources=DEEPSIGN_PEN_SOURCES, extra_channels=None, float32_parse=False):
    """Parse the DeepSignDB online signature dataset."""
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

        writer_id = f"deepsign_{source}-{split_tag}_{int(match.group(1)):04d}"
        label = "genuine" if match.group(2).lower() == "g" else "forged"

        record = writers.setdefault(writer_id, {**_new_writer(), "genuine_session": []})
        record[label].append(
            load_deepsigndb_file(os.path.join(directory, filename), source,
                                 extra_channels, float32_parse))
        if label == "genuine":
            session = _DEEPSIGN_SESSION.search(filename)
            record["genuine_session"].append(int(session.group(1)) if session else None)

    return writers


def parse_sigcomp2011_offline(root=None):
    """Parse the ICDAR SigComp2011 offline scanned image signature dataset."""
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


# Registered parsers by dataset directory name.
_PARSERS = {
    "cedar": parse_cedar,
    "bhsig260": parse_bhsig260,
    "utsig": parse_utsig,
    "svc2004": parse_svc2004,
    "svc2004_task1": parse_svc2004,
    "svc2004_task2": parse_svc2004,
    "mobisig": parse_mobisig,
    "sigcomp2011": parse_sigcomp2011,
    "sigcomp2011_offline": parse_sigcomp2011_offline,
    "deepsigndb": parse_deepsigndb,
    # Scans only; the pen files are read through parse_biosecurid_sonof_pairs.
    "biosecurid_sonof": parse_biosecurid_sonof_scans,
}

# Supported additional channels by dataset.
EXTRA_CHANNEL_SUPPORT = {
    "svc2004": frozenset({"tilt", "dt"}),
    "svc2004_task2": frozenset({"tilt", "dt"}),
    "sigcomp2011": frozenset({"dt"}),
    "deepsigndb": frozenset({"dt"}),
}


def is_known_dataset(path):
    """Check if the directory name matches a known dataset parser."""
    return os.path.basename(os.path.normpath(path)).lower() in _PARSERS


def supports_extra_channel(path, channel):
    """Check if the dataset at the given path supports the extra channel."""
    name = os.path.basename(os.path.normpath(path)).lower()
    return channel in EXTRA_CHANNEL_SUPPORT.get(name, frozenset())


TEMPORAL_DATASETS = ("svc2004", "svc2004_task1", "svc2004_task2",
                     "mobisig", "sigcomp2011", "deepsigndb")


def is_temporal_dataset(path):
    """Check if the dataset yields online trajectories rather than offline images."""
    return os.path.basename(os.path.normpath(path)).lower() in TEMPORAL_DATASETS
