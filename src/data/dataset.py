"""
Dataset Pipeline Operations — Signature Verification System

Manages data ingestion, spatial preprocessing, augmentation strategies, quality verification gating,
and writer-independent split partition generation across offline image and online dynamic signature datasets.

Directory Layout Conventions (`./data`):
--------------------------------------------
data/
├── cedar/                  # Offline CEDAR dataset (55 writers)
│   ├── full_org/           # original_{writer}_{sample}.png    (Genuine)
│   └── full_forg/          # forgeries_{writer}_{sample}.png   (Skilled Forgeries)
├── bhsig260/               # Offline BHSig260 dataset (260 writers)
│   ├── Bengali/            # 100 writers: {writer}/B-S-{w}-{G|F}-{n}.tif
│   └── Hindi/              # 160 writers: {writer}/H-S-{w}-{G|F}-{n}.tif
├── utsig/                  # Offline UTSig dataset (115 writers)
│   ├── Genuine/            # {writer}/{n}.tif
│   └── Forgery/            # {Skilled,Simple,Opposite Hand}/{writer}/{n}.tif
└── svc2004/                # Online SVC2004 dataset (40 writers)
    └── U{user}S{sample}.TXT     # Task1 files are 4-column (X, Y, T, Button) — no pressure. Task2 files are 7-column (X, Y, T, Button, Azimuth, Altitude, Pressure).

Supported Datasets:
  Spatial:  CEDAR, BHSig260, UTSig
  Temporal: SVC2004, SigComp2011, MOBISIG, DeepSignDB
"""

import itertools
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import (
    SPATIAL_INPUT_SIZE,
    SPATIAL_SQUARE_PAD,
    MIN_BBOX_PIXELS,
    MIN_TRAJECTORY_POINTS,
    MAX_NORMALIZED_XY_MEAN,
    NORMALIZED_XY_STD_RANGE,
    MIN_LAPLACIAN_VARIANCE,
    PAIRS_GENUINE_PER_WRITER,
    PAIRS_IMPOSTOR_PER_WRITER,
    RANDOM_IMPOSTOR_RATIO,
    TEMPORAL_CHANNEL_WIDTHS,
    TEMPORAL_INPUT_FEATURES,
    SPATIAL_AUG_MAX_ROTATION_DEG, SPATIAL_AUG_MAX_SHEAR, SPATIAL_AUG_MAX_SCALE,
    SPATIAL_AUG_MAX_TRANSLATE, SPATIAL_AUG_STROKE_PROB,
    TEMPORAL_AUG_MAX_ROTATION_DEG, TEMPORAL_AUG_JITTER, TEMPORAL_AUG_MAX_SPEED_CHANGE,
    SPATIAL_SPLIT_RATIOS,
)
from src.data.renderer import render_trajectory_to_image

# Channel names for gate errors, keyed off the shipped capture contract so the
# message names the columns the gate is actually enforcing rather than a literal
# that drifts the next time the contract widens.
_CHANNEL_NAMES = {
    3: "X, Y, P",
    4: "X, Y, P, dt",
    5: "X, Y, P, azimuth, altitude",
}.get(TEMPORAL_INPUT_FEATURES, f"{TEMPORAL_INPUT_FEATURES} channels")


def preprocess_image(image_path, target_size=SPATIAL_INPUT_SIZE):
    """
    Preprocess signature image sample for Spatial Siamese CNN processing.

    Steps:
      1. Ingest image in 8-bit single-channel grayscale format.
      2. Apply background padding or direct bilinear scaling based on configuration.
      3. Interpolate onto standard canvas dimensions (INTER_AREA).
      4. Invert pixel values to map background to 0.0 and stroke ink to positive values [0.0, 1.0].
      5. Add channel tensor dimension [1, H, W] for PyTorch layer ingestion.

    Returns:
        float32 numpy array of shape (1, H, W).
    """
    if isinstance(image_path, np.ndarray):
        # A raw scan handed over as an array rather than a path. Same contract as
        # a file: paper bright, ink dark, any size. It still gets resized and
        # inverted below, so an array and a path reach the model identically.
        img = image_path[0] if image_path.ndim == 3 else image_path
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    if SPATIAL_SQUARE_PAD:
        # Pre-pad image onto square canvas while maintaining original aspect ratio
        h, w = img.shape
        max_dim = max(h, w)
        padded = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = img
    else:
        # Scale directly to target input geometry (155x220) matching Dey et al. Siamese baseline
        padded = img

    # Resize to spatial tensor dimensions and invert polarity (background 0, ink high).
    # Inverting polarity is critical: ink covers ~1% of the canvas surface, so an
    # un-inverted image feeds convolutions a large constant background offset carrying
    # zero identity signal. Inverting polarity dropped CEDAR error from 4.86% to 0.00%.
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    normalized = (255.0 - resized.astype(np.float32)) / 255.0
    return normalized[np.newaxis, :, :]


# ─── Spatial sample kinds ─────────────────────────────────────────────
# `as_spatial_input` below decides what an incoming array *is*. The two ranges
# it separates are deliberately far apart and are both read from config:
#
#   * a trajectory is [L, C] with C drawn from TEMPORAL_CHANNEL_WIDTHS — 3 (xyp),
#     4 (dt, shipped) or 5 (tilt). Widening the captured contract past that table
#     is banned outright under AGENTS.md Drift Prevention, so 5 is a hard ceiling
#     rather than today's value. Sourcing it here keeps `TEMPORAL_CHANNELS` the
#     single knob: a literal 4 in a gate is what once made the shipped 4-channel
#     model unreachable through `verify()`.
#   * a scan is at least MIN_BBOX_PIXELS across. The ink bounding box cannot
#     exceed the canvas, so an image narrower or shorter than that floor already
#     fails `quality_gate` and is not a signature this project accepts.
_TRAJECTORY_MIN_COLUMNS = min(TEMPORAL_CHANNEL_WIDTHS.values())
_TRAJECTORY_MAX_COLUMNS = max(TEMPORAL_CHANNEL_WIDTHS.values())


def as_spatial_input(sample, allow_trajectory=False):
    """
    Bring one spatial sample to the model's [1, H, W] contract, or refuse it.

    This is the single resolver for "which kind of thing is this" — `verify()`
    reaches it through `src.verification._as_spatial_input` and the training
    loader through `SpatialSignatureDataset`. Two copies of this rule would drift
    against each other, and a training/serving disagreement about what an array
    means is invisible in every number either side reports.

    Four kinds are legitimate:

    * a **path** — read, resized and inverted by `preprocess_image`;
    * a **raw scan array** (paper bright, ink dark, any size) — same treatment,
      so an array and the file it came from reach the model identically;
    * a **canonical [1, H, W] float array in [0, 1]**, which the renderer already
      emits — passed straight through. Sending it through `preprocess_image`
      would invert it a second time and hand the model a photographic negative;
    * a **pen trajectory [L, C]**, rasterized by `render_trajectory_to_image`,
      which lets one dataset hold scans and tablet captures at once. Training
      only (`allow_trajectory=True`): `verify()` renders explicitly before it
      reaches this point, and silently accepting a raw trajectory as an image
      probe there would widen the serving contract AGENTS.md pins.

    **The hard case is that a trajectory and a scan are both 2-D float arrays.**
    Nothing in the type says which. The fork keys on the **column count**,
    because it is the one property whose two admissible ranges cannot overlap:
    a trajectory has at most `_TRAJECTORY_MAX_COLUMNS` (5) of them and a scan at
    least `MIN_BBOX_PIXELS` (64), and the gap between is refused rather than
    guessed at. A row floor closes the transposed `[C, L]` case, whose column
    count would otherwise read as a scan.

    Value range and dtype were **rejected** as the discriminator and must not be
    reintroduced: a float array in [0, 1] is the legitimate canonical kind, a
    trajectory's pressure and `dt` columns are also in [0, 1], and a trajectory
    still in raw device units is indistinguishable from a scan by magnitude.
    Only the shape carries the answer.

    Anything else raises. Accepting it quietly is the defect this function
    exists to close: `SpatialSiamese` pools adaptively, so a wrong-sized or
    un-inverted array yields a perfectly plausible 512-d embedding with no
    exception — enrolment then writes that template to the database, and
    training would fit it. The inversion alone is worth 4.86% to 0.00% EER on
    CEDAR, so a silent miss here is not cosmetic (SUMMARY.md Part 9 §12).
    """
    if not isinstance(sample, np.ndarray):
        return preprocess_image(sample)

    # Non-finiteness is checked here, before the fork, because neither branch
    # below can catch it. The canonical branch's range test is blind to NaN —
    # every comparison against NaN is False, so `low < -1e-6 or high > 1.0`
    # passed a poisoned array through untouched and `embed()` returned a 512-d
    # embedding of NaN, which enrolment stores as a template whose distance to
    # every probe is NaN. The scan branch destroys the evidence instead of
    # propagating it: `preprocess_image` casts through
    # `np.clip(...).astype(np.uint8)`, which turns NaN into an arbitrary byte and
    # infinity into 255, so the poisoned pixel disappears into a perfectly
    # ordinary-looking embedding. `trajectory_gate` has always refused a
    # non-finite trajectory explicitly; this is the spatial counterpart.
    # Integer arrays cannot represent either value, so only floats are scanned.
    if np.issubdtype(sample.dtype, np.floating) and not np.all(np.isfinite(sample)):
        raise ValueError(
            f"Spatial array of shape {sample.shape} contains NaN or infinity. A "
            f"signature pixel is a measurement, and there is no value to guess "
            f"for one that is not — fix it at the capture or render step.")

    width, height = SPATIAL_INPUT_SIZE
    canonical = (sample.ndim == 3 and sample.shape == (1, height, width)
                 and np.issubdtype(sample.dtype, np.floating))
    if canonical:
        low, high = float(sample.min()), float(sample.max())
        if low < -1e-6 or high > 1.0 + 1e-6:
            raise ValueError(
                f"Spatial array has the canonical shape {sample.shape} but values "
                f"in [{low:.3f}, {high:.3f}] — the contract is [0.0, 1.0] with "
                f"background 0. A raw scan belongs in a 2-D array so it gets "
                f"resized and inverted; this one would reach the model unscaled.")
        return np.ascontiguousarray(sample, dtype=np.float32)

    if sample.ndim == 3 and sample.shape[0] == 1:
        # A raw scan carrying an explicit channel axis: preprocess_image resizes
        # and inverts it exactly as it would the same scan on disk.
        return preprocess_image(sample)

    if sample.ndim == 2:
        rows, columns = sample.shape

        if columns <= _TRAJECTORY_MAX_COLUMNS:
            if columns < _TRAJECTORY_MIN_COLUMNS or not allow_trajectory:
                raise ValueError(
                    f"Cannot interpret a spatial sample of shape {sample.shape}. "
                    f"It is too narrow to be a scan, and "
                    + (f"a trajectory needs at least {_TRAJECTORY_MIN_COLUMNS} "
                       f"columns (X, Y, P)."
                       if columns < _TRAJECTORY_MIN_COLUMNS else
                       "trajectories are not accepted here — render it with "
                       "render_trajectory_to_image() first."))
            if rows < MIN_TRAJECTORY_POINTS:
                # The renderer returns a near-blank canvas for a stub rather than
                # raising, and a blank "signature" is a valid-looking training
                # sample carrying no identity at all.
                raise ValueError(
                    f"Trajectory has {rows} points, need {MIN_TRAJECTORY_POINTS} "
                    f"to render a signature.")
            return render_trajectory_to_image(sample)

        if min(rows, columns) >= MIN_BBOX_PIXELS:
            return preprocess_image(sample)

        raise ValueError(
            f"Cannot interpret a spatial sample of shape {sample.shape}. Its "
            f"column count is too wide for a capture contract (at most "
            f"{_TRAJECTORY_MAX_COLUMNS} channels) and one of its axes is under "
            f"the {MIN_BBOX_PIXELS}px floor a scan must clear. A transposed "
            f"[C, L] trajectory lands here; so does a thumbnail no quality gate "
            f"would pass. Neither is guessed at.")

    raise ValueError(
        f"Cannot interpret a spatial sample of shape {sample.shape}. Expected a "
        f"path, a 2-D scan array, a rendered [1, {height}, {width}] float array "
        f"in [0, 1], or an [L, C] trajectory.")


def quality_gate(image_path):
    """
    Assess spatial signature image quality via resolution and sharpness metrics.

    Calculates:
      - Stroke bounding-box dimensions.
      - Laplacian variance focus sharpness score.

    Returns: (passed: bool, details: dict)
    """
    if isinstance(image_path, np.ndarray):
        arr = image_path[0] if image_path.ndim == 3 else image_path
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                # Inverted float array (background 0, ink high):
                # Convert to standard scan polarity (paper 255, ink 0) for Otsu THRESH_BINARY_INV
                img = np.clip(255.0 - arr * 255.0, 0, 255).astype(np.uint8)
            else:
                img = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            img = arr
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, {"error": "Cannot load image"}

    # Binarize scan using Otsu thresholding to identify stroke region
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, {"error": "No signature content detected"}


    # Calculate bounding-box span across all signature contours
    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Calculate image focus/sharpness using Laplacian variance
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = laplacian.var()

    details = {"bbox_width": bbox_w, "bbox_height": bbox_h, "sharpness": round(sharpness, 2)}
    passed = bool(
        min(bbox_w, bbox_h) >= MIN_BBOX_PIXELS
        and sharpness >= MIN_LAPLACIAN_VARIANCE
    )
    return passed, details


def trajectory_gate(trajectory):
    """
    Check a trajectory is shaped and normalized the way the model was trained.

    The temporal counterpart to `quality_gate`, and it exists because the
    temporal failure mode is the worse of the two. A blurry image degrades a
    score; a trajectory still in raw device units produces an embedding with no
    relationship to the enrolled templates, and the system returns a *confident*
    wrong verdict. Measured on one SigComp writer: the same signature scored
    0.9370 normalized and 0.4888 in raw tablet units — accepted, then rejected.

    `normalize_trajectory` z-scores X and Y per signature, so the check is
    cheap: real input has mean ~0 and std ~1 on both axes, and raw tablet units
    have means in the thousands. This detects the *class* of error rather than a
    specific device's scale, so it needs no per-device configuration.

    Pressure is checked for range only. It is not z-scored — it is scaled to
    [0, 1] by a fixed device maximum — and a flat P = 1.0 is the legitimate
    pressure-blind case that `verify` reports as degraded, not an error.

    Returns: (passed: bool, details: dict) — same contract as `quality_gate`.
    """
    trajectory = np.asarray(trajectory, dtype=np.float32)

    if trajectory.ndim != 2 or trajectory.shape[1] != TEMPORAL_INPUT_FEATURES:
        return False, {"error": f"Expected [L, {TEMPORAL_INPUT_FEATURES}] "
                                f"({_CHANNEL_NAMES}), got {trajectory.shape}"}
    if len(trajectory) < MIN_TRAJECTORY_POINTS:
        return False, {"error": f"Only {len(trajectory)} points, need "
                                f"{MIN_TRAJECTORY_POINTS}"}
    if not np.all(np.isfinite(trajectory)):
        return False, {"error": "Trajectory contains NaN or infinity"}

    xy = trajectory[:, :2]
    means = np.abs(xy.mean(axis=0))
    stds = xy.std(axis=0)
    low, high = NORMALIZED_XY_STD_RANGE

    details = {
        "points": len(trajectory),
        "x_mean": round(float(xy[:, 0].mean()), 3),
        "y_mean": round(float(xy[:, 1].mean()), 3),
        "x_std": round(float(stds[0]), 3),
        "y_std": round(float(stds[1]), 3),
        "pressure_min": round(float(trajectory[:, 2].min()), 3),
        "pressure_max": round(float(trajectory[:, 2].max()), 3),
    }

    if np.any(means > MAX_NORMALIZED_XY_MEAN) or np.any(stds < low) or np.any(stds > high):
        return False, {**details, "error":
                       "X/Y are not z-scored — call parsers.normalize_trajectory() "
                       "in the capture layer. Expected mean ~0 and std ~1 per axis."}

    if trajectory[:, 2].min() < 0.0 or trajectory[:, 2].max() > 1.0:
        return False, {**details, "error":
                       "Pressure outside [0, 1] — scale by the device's fixed "
                       "maximum, never the file's own max."}

    if TEMPORAL_INPUT_FEATURES >= 4:
        # `dt` is the gap to the previous sample, clipped at 100 ms and scaled onto
        # [0, 1] by the parsers. A raw timestamp column reaching here would be in
        # device milliseconds and is the same class of error as unnormalized X/Y:
        # it scores confidently against templates it has no relationship to.
        details["dt_min"] = round(float(trajectory[:, 3].min()), 3)
        details["dt_max"] = round(float(trajectory[:, 3].max()), 3)
        if trajectory[:, 3].min() < 0.0 or trajectory[:, 3].max() > 1.0:
            return False, {**details, "error":
                           "dt outside [0, 1] — it is the inter-sample gap clipped at "
                           "100 ms and scaled, not a raw timestamp."}

    return True, details


def writer_independent_split(writer_ids, train_ratio=SPATIAL_SPLIT_RATIOS[0],
                              val_ratio=SPATIAL_SPLIT_RATIOS[1],
                              test_ratio=SPATIAL_SPLIT_RATIOS[2], seed=42):
    """
    Split writer IDs into non-overlapping train/val/test sets.

    Ensures zero writer overlap across partitions (ISO/IEC 23053:2022 standard).
    No individual writer appears in more than one partition, ensuring true
    writer-independent open-set verification evaluation.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    ids = list(writer_ids)
    rng.shuffle(ids)
    n = len(ids)
    t1 = int(n * train_ratio)
    t2 = t1 + int(n * val_ratio)
    return ids[:t1], ids[t1:t2], ids[t2:]


def pin_holdout(writer_ids, pinned, train_ratio=SPATIAL_SPLIT_RATIOS[0],
                val_ratio=SPATIAL_SPLIT_RATIOS[1],
                test_ratio=SPATIAL_SPLIT_RATIOS[2], seed=42):
    """
    Split writers, forcing `pinned` into the test partition.

    `writer_independent_split` shuffles every ID it is handed, so the partition a
    writer lands in depends on how many *other* writers are present. That makes
    it impossible to add a corpus and still compare against an earlier number:
    adding DeepSignDB's 498 writers relocates the 31 SVC2004/SigComp writers the
    current result was scored on, and some of them land in training. The run
    still completes and still reports a plausible EER, so the leak is silent.

    Pinning holds those writers out no matter what else joins the corpus. The
    remainder is split by the same seeded shuffle, so the unpinned partition is
    reproducible and unaffected by the pinned set's size.

    An unknown ID raises rather than being ignored: a typo would otherwise hold
    out nothing and quietly score the model on the wrong writers.

    `test_ratio` does not size the test partition here — the pinned writers do —
    but it is still checked against the other two. Callers pass the same
    `*ratios` triple to this and to `writer_independent_split`, so accepting a
    triple that does not sum to 1 would mean the two paths disagree about the
    split while both look like they were configured identically.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    ids = list(writer_ids)
    unknown = sorted(set(pinned) - set(ids))
    if unknown:
        raise ValueError(
            f"Cannot pin writers absent from the corpus: {', '.join(unknown)}. "
            f"Check the IDs against the datasets passed to --data-dir."
        )

    pinned = set(pinned)
    rest = [w for w in ids if w not in pinned]

    # The pinned writers already occupy the test partition, so the remaining
    # ratios are renormalized over train and val only. Splitting `rest` by the
    # original test_ratio as well would hold out far more than asked.
    denominator = train_ratio + val_ratio
    if denominator <= 0:
        raise ValueError("train_ratio + val_ratio must be positive to pin a holdout")

    rng = random.Random(seed)
    rng.shuffle(rest)
    cut = int(len(rest) * (train_ratio / denominator))
    return rest[:cut], rest[cut:], sorted(pinned)


def generate_skilled_pairs(samples_by_writer, num_genuine_per_writer=PAIRS_GENUINE_PER_WRITER,
                           num_forged_per_writer=PAIRS_IMPOSTOR_PER_WRITER, seed=42,
                           random_impostor_ratio=RANDOM_IMPOSTOR_RATIO):
    """
    Generate pairs under the **skilled-forgery** scenario (AGENTS.md eval protocol).

    Negatives are by default a genuine signature paired
    against a forgery *of that same writer* — a deliberate imitation, not a
    random stranger's signature. Random-impostor negatives are trivially
    separable and inflate the reported EER into a number that does not describe
    the attack the system actually faces. **All reported metrics use that pairing
    and nothing here changes it**: leave `random_impostor_ratio` at 0 for every
    evaluation loader.

    `random_impostor_ratio` exists for *training* only. Supervising exclusively
    on same-writer forgeries never tells the encoder that two different people
    should be far apart, and measurement showed it does not infer that on its
    own: on CEDAR the shipped model placed a stranger's genuine signature at mean
    distance 0.048 against 0.038 for the same writer's, while that writer's own
    skilled forgery sat at 2.043 — a genuine/forged axis rather than an identity
    one, and 36.64% EER against random impostors. The ratio takes its share out
    of the negative budget rather than adding to it, so the 1:1 positive-negative
    balance per writer is unchanged and the only variable is what the negatives
    are made of.

    Args:
        samples_by_writer: dict mapping writer_id to
            {"genuine": [paths/arrays], "forged": [paths/arrays]}.
        num_genuine_per_writer: target genuine pairs per writer (label = 1).
        num_forged_per_writer: target negative pairs per writer (label = 0).
        random_impostor_ratio: fraction of the negative budget drawn from other
            writers' genuine signatures instead of this writer's forgeries.

    Returns:
        (pairs: list of (sample1, sample2), labels: list of int)
    """
    if not 0.0 <= random_impostor_ratio <= 1.0:
        raise ValueError(
            f"random_impostor_ratio must be in [0, 1], got {random_impostor_ratio}")

    rng = random.Random(seed)
    pairs, labels = [], []

    for writer in samples_by_writer:
        genuine = samples_by_writer[writer].get("genuine", [])
        forged = samples_by_writer[writer].get("forged", [])

        # A writer needs 2 genuines to form a positive and >=1 forgery for a
        # skilled negative. Without both, this writer cannot be scored.
        if len(genuine) < 2 or not forged:
            continue

        # A writer contributes equal numbers of positives and negatives, so the
        # scarcer arm caps both. Capping each arm only by its own supply silently
        # reweights datasets that ship few forgeries: UTSig has 27 genuine but 6
        # skilled forgeries per writer, giving C(27,2)=351 genuine combinations
        # against 27x6=162 impostor ones. At a 276 budget that produced a 1.70:1
        # genuine-heavy batch for UTSig while every other dataset stayed 1:1, and
        # UTSig's held-out EER rose 38.83% -> 43.43% when the budget was raised.
        # At the old budget of 40 everything was balanced by accident, which is
        # why this only surfaced once the budget went up.
        num_pairs = min(
            num_genuine_per_writer,
            num_forged_per_writer,
            len(genuine) * (len(genuine) - 1) // 2,   # available genuine combinations
            len(genuine) * len(forged),               # available impostor combinations
        )

        # Both arms enumerate their pair space, shuffle it, and take the first
        # `num_pairs` — they do **not** draw `num_pairs` times with replacement.
        #
        # Drawing with replacement from a space of the same size returns only
        # about 1 - 1/e of it, so a writer whose budget said 276 was really
        # training on ~174 distinct pairs and ~102 duplicates. `num_pairs` is
        # already capped at the available combination count above, so the slice
        # can never run short of the requested budget.
        genuine_space = list(itertools.combinations(genuine, 2))
        rng.shuffle(genuine_space)
        for s1, s2 in genuine_space[:num_pairs]:
            pairs.append((s1, s2))
            labels.append(1)

        # The negative budget splits between the two attack types. Rounding down
        # keeps at least one skilled pair unless the ratio is exactly 1.0, so a
        # small ratio never silently wipes out the arm that already works.
        num_random = int(num_pairs * random_impostor_ratio)
        num_skilled = num_pairs - num_random

        # Skilled-forgery pairs: authentic vs. an imitation of this same writer
        impostor_space = list(itertools.product(genuine, forged))
        rng.shuffle(impostor_space)
        for s1, s2 in impostor_space[:num_skilled]:
            pairs.append((s1, s2))
            labels.append(0)

        # Random-impostor pairs: authentic vs. a *different* writer's authentic
        # signature. Drawn with replacement, unlike the two same-writer arms:
        # the cross-writer space is the whole corpus (301 training writers x ~24
        # genuines is over 7000 candidates per slot), so the coupon-collector
        # duplication that cost the same-writer arms ~37% of their diversity is
        # negligible here, and enumerating that space per writer is not.
        others = [w for w in samples_by_writer if w != writer
                  and samples_by_writer[w].get("genuine")]
        if not others:
            continue
        for _ in range(num_random):
            other = rng.choice(others)
            pairs.append((rng.choice(genuine),
                          rng.choice(samples_by_writer[other]["genuine"])))
            labels.append(0)

    return pairs, labels


def collate_temporal_pairs(batch):
    """
    Collate variable-length trajectory pairs into a padded batch.

    Signatures have different point counts, so a batch must be padded to its
    longest member. The true lengths are returned alongside so the LSTM can
    `pack_padded_sequence` them — otherwise the recurrent state keeps consuming
    pad rows and short samples get a different embedding inside a batch than
    they do alone.

    Args:
        batch: list of (seq1 [L1, 3], seq2 [L2, 3], label) from TemporalSignatureDataset.

    Returns:
        (seq1 [B, maxL1, 3], lengths1 [B], seq2 [B, maxL2, 3], lengths2 [B], labels [B])
    """
    seqs1, seqs2, labels = zip(*batch)

    lengths1 = torch.tensor([len(s) for s in seqs1], dtype=torch.long)
    lengths2 = torch.tensor([len(s) for s in seqs2], dtype=torch.long)

    padded1 = torch.nn.utils.rnn.pad_sequence(list(seqs1), batch_first=True)
    padded2 = torch.nn.utils.rnn.pad_sequence(list(seqs2), batch_first=True)

    return padded1, lengths1, padded2, lengths2, torch.stack(list(labels))


def augment_image(image, rng, max_rotation_deg=SPATIAL_AUG_MAX_ROTATION_DEG,
                  max_shear=SPATIAL_AUG_MAX_SHEAR, max_scale=SPATIAL_AUG_MAX_SCALE,
                  max_translate=SPATIAL_AUG_MAX_TRANSLATE,
                  stroke_prob=SPATIAL_AUG_STROKE_PROB):
    """
    Perturb one preprocessed [1, H, W] image for training-time augmentation.

    The spatial branch trains on 301 writers and saturates inside a single
    epoch, so the limit is how much *variety* the training set holds, not how
    many pairs are drawn from it. This is the only lever that adds variety
    without new writers, and it is what the Siamese references (Dey et al.,
    Hafemann et al.) rely on at comparable dataset sizes.

    Every warp is small and label-preserving: a signature rotated 6 degrees or
    written with a slightly wider pen is still that person's signature. Large
    warps would manufacture pairs whose label is no longer true.

    Ranges are deliberately conservative because forgery detection keys on fine
    stroke geometry — the thing an aggressive warp destroys first.

    Args:
        image: [1, H, W] float32, ink high, background 0 (post-`preprocess_image`).
        rng:   a `numpy.random.Generator`.

    Returns: a new [1, H, W] float32 array in the same range. Never in-place.
    """
    channels, height, width = image.shape
    canvas = image[0]

    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    shear = rng.uniform(-max_shear, max_shear)

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 1] += shear                      # horizontal skew, as in slanted handwriting
    matrix[0, 2] += rng.uniform(-max_translate, max_translate) * width
    matrix[1, 2] += rng.uniform(-max_translate, max_translate) * height

    # borderValue=0 keeps the background black, matching the inverted polarity
    # preprocess_image produces. A white border would inject a bright constant
    # the convolutions would read as ink.
    warped = cv2.warpAffine(canvas, matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # Stroke width stands in for pen choice and writing pressure. Applied to
    # only some samples so the unmodified width stays the dominant mode.
    if rng.random() < stroke_prob:
        kernel = np.ones((2, 2), dtype=np.uint8)
        warped = (cv2.dilate(warped, kernel) if rng.random() < 0.5
                  else cv2.erode(warped, kernel))

    return warped.reshape(channels, height, width).astype(np.float32)


class SpatialSignatureDataset(Dataset):
    """
    PyTorch Dataset loading spatial signature pairs for Siamese training.

    A pair element is a file path, a raw scan array, an already-rendered
    canonical image, or an `[L, C]` pen trajectory — freely mixed, so one dataset
    can draw on the image corpora and the pen corpora at once. `as_spatial_input`
    resolves each side and is the only place that decides which kind a sample is.

    Each item returns:
        - img1_tensor: [1, H, W] float32, H/W from SPATIAL_INPUT_SIZE
        - img2_tensor: [1, H, W] float32
        - label: float tensor (1.0 for genuine, 0.0 for forgery)
    """

    def __init__(self, pairs, labels, transform=None, augment=False, seed=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform
        self.augment = augment
        # Seeded so an augmented run reproduces from --seed. An unseeded
        # default_rng() draws OS entropy, which silently made augmented training
        # irreproducible even though every other RNG in the pipeline was seeded.
        # DataLoader workers re-seed this per worker — see `worker_init_fn` in
        # scripts/train.py — because fork-spawned workers otherwise inherit one
        # generator state and replay identical augmentations in lockstep.
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sample1, sample2 = self.pairs[index]
        # Each side is resolved independently, so one pair may hold a scanned
        # reference against a rasterized tablet capture. `as_spatial_input`
        # decides which kind each is; the fork is documented there because the
        # two look identical from here.
        img1 = as_spatial_input(sample1, allow_trajectory=True)
        img2 = as_spatial_input(sample2, allow_trajectory=True)

        if self.augment:
            # Each side is warped independently. Warping a pair identically would
            # move both embeddings the same way and leave the distance between
            # them unchanged, which teaches the metric nothing.
            img1 = augment_image(img1, self._rng)
            img2 = augment_image(img2, self._rng)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (
            torch.tensor(img1, dtype=torch.float32),
            torch.tensor(img2, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def augment_trajectory(sequence, rng, max_rotation_deg=TEMPORAL_AUG_MAX_ROTATION_DEG,
                       jitter=TEMPORAL_AUG_JITTER,
                       max_speed_change=TEMPORAL_AUG_MAX_SPEED_CHANGE):
    """
    Perturb one [L, C] trajectory for training-time augmentation, preserving C.

    Three perturbations, each chosen because it mimics variation a genuine
    signer actually produces:

      * **Speed change** — resample the sequence slightly faster or slower.
        People do not sign at a fixed tempo, and speed is the feature family the
        derived (dX, dY, dP) channels key on, so varying it teaches tolerance
        where the model is most sensitive.
      * **Small rotation** — nobody aligns the pen identically to the tablet
        every time. Rotation applies to X/Y only, never pressure.
      * **Coordinate jitter** — sensor noise and hand tremor.

    Deliberately absent: mirroring, which produces a trajectory the signer could
    never draw, and scaling, which is already normalized away because X/Y are
    z-scored per signature by `normalize_trajectory`.

    Sequences too short to resample are returned unchanged; perturbing three
    points corrupts rather than varies them.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) < 8:
        return sequence

    xy = sequence[:, :2].copy()

    theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xy = xy @ np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    xy += rng.normal(0.0, jitter, size=xy.shape).astype(np.float32)

    # Resample onto a stretched or compressed timeline. Interpolating each
    # channel against the original index keeps the stroke order intact.
    scale = 1.0 + rng.uniform(-max_speed_change, max_speed_change)
    new_length = max(8, int(round(len(sequence) * scale)))
    source = np.arange(len(sequence), dtype=np.float32)
    target = np.linspace(0, len(sequence) - 1, new_length, dtype=np.float32)

    columns = [np.interp(target, source, xy[:, 0]),
               np.interp(target, source, xy[:, 1])]

    # Pressure and every trailing capture channel (`dt`, and whatever a widened
    # contract adds later) ride the same resampled timeline. They are physical
    # quantities in [0, 1] rather than coordinates, so they are never rotated or
    # jittered — only carried through the new timing — and interpolation can
    # drift them a hair outside the range at the endpoints.
    #
    # Rebuilding a fixed three-column array here used to drop the 4th channel
    # silently: the narrowed batch then fed a wider LSTM, which raises for
    # batches of two or more but runs to completion on a single-sample batch and
    # returns a plausible embedding computed from the wrong tensor.
    for channel in range(2, sequence.shape[1]):
        columns.append(np.clip(np.interp(target, source, sequence[:, channel]), 0.0, 1.0))

    return np.column_stack(columns).astype(np.float32)


class TemporalSignatureDataset(Dataset):
    """
    PyTorch Dataset loading temporal (dynamic) signature pairs for Siamese training.

    Each item returns:
        - seq1_tensor: trajectory sequence tensor [L1, 3] containing (X, Y, Pressure)
        - seq2_tensor: trajectory sequence tensor [L2, 3] containing (X, Y, Pressure)
        - label: float tensor (1.0 for genuine, 0.0 for forgery)
    """

    def __init__(self, pairs, labels, augment=False, augment_scale=1.0, seed=42):
        self.pairs = pairs
        self.labels = labels
        # Training only. Perturbing validation or test data would measure the
        # model against inputs the deployed system never sees.
        self.augment = augment
        self.augment_scale = augment_scale
        # Seeded from the run, not a hard-coded 0. `_seed_worker` in scripts/train.py
        # reads `_seed` to give each forked DataLoader worker its own augmentation
        # stream; without the attribute it silently no-ops, so every worker replayed
        # the identical draws in lockstep — exactly the bug already found and fixed
        # on the spatial side. A fixed 0 also meant --seed never varied the temporal
        # augmentation, so "independent" re-runs of that arm were not independent.
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def _augment(self, sequence):
        """Perturb one side of a pair, with every range scaled by `augment_scale`."""
        return augment_trajectory(sequence, self._rng,
                                  max_rotation_deg=TEMPORAL_AUG_MAX_ROTATION_DEG * self.augment_scale,
                                  jitter=TEMPORAL_AUG_JITTER * self.augment_scale,
                                  max_speed_change=TEMPORAL_AUG_MAX_SPEED_CHANGE * self.augment_scale)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        seq1, seq2 = self.pairs[index]
        if self.augment:
            # Each side is perturbed independently: two recordings of the same
            # signature differ by exactly this kind of variation, so the pair
            # stays honest.
            #
            # `augment_scale` shrinks all three ranges together. At the default
            # 1.0 strength, augmentation measured 2.54 EER points *worse* than
            # none, and the spatial branch's rule — "keep the ranges small;
            # forgery detection keys on fine stroke geometry" — applies with
            # more force to a trajectory, which is nothing but fine geometry.
            # Strength is therefore a tunable, not a fixed cost of enabling.
            seq1 = self._augment(seq1)
            seq2 = self._augment(seq2)
        return (
            torch.tensor(seq1, dtype=torch.float32),
            torch.tensor(seq2, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )

