"""Dataset loading, preprocessing, quality gates, splits, pair sampling, and augmentation."""

import itertools
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import (
    TILT_CHANNEL_MODES,
    SPATIAL_INPUT_SIZE,
    SPATIAL_SQUARE_PAD,
    MIN_BBOX_PIXELS,
    MIN_TRAJECTORY_POINTS,
    MAX_NORMALIZED_XY_MEAN,
    NORMALIZED_XY_STD_RANGE,
    MIN_LAPLACIAN_VARIANCE,
    PAIRS_GENUINE_PER_WRITER,
    PAIRS_IMPOSTOR_PER_WRITER,
    TEMPORAL_CHANNEL_WIDTHS,
    TEMPORAL_INPUT_FEATURES,
    SPATIAL_AUG_MAX_ROTATION_DEG, SPATIAL_AUG_MAX_SHEAR, SPATIAL_AUG_MAX_SCALE,
    SPATIAL_AUG_MAX_TRANSLATE, SPATIAL_AUG_STROKE_PROB,
    SPATIAL_AUG_BLUR_SIGMA, SPATIAL_AUG_INK_GAIN_MIN, SPATIAL_AUG_NOISE_STD,
    TEMPORAL_AUG_MAX_ROTATION_DEG, TEMPORAL_AUG_JITTER, TEMPORAL_AUG_MAX_SPEED_CHANGE,
    TEMPORAL_CHANNELS, TEMPORAL_SAMPLE_RATE_HZ,
    TEMPORAL_DEVICE_AUG_RATE_HZ, TEMPORAL_DEVICE_AUG_PRESSURE_GAMMA,
    TEMPORAL_DEVICE_AUG_PRESSURE_GAIN, TEMPORAL_DEVICE_AUG_PRESSURE_LEVELS,
    SPATIAL_SPLIT_RATIOS,
    TEMPORAL_SAMPLE_PERIOD_MS, TRAJECTORY_DT_MEDIAN_TOLERANCE,
)
from src.data.renderer import render_trajectory_to_image
from src.data.parsers import MAX_GAP_MS

_CHANNEL_NAMES = {
    3: "X, Y, P",
    4: "X, Y, P, dt",
    5: "X, Y, P, azimuth, altitude",
}.get(TEMPORAL_INPUT_FEATURES, f"{TEMPORAL_INPUT_FEATURES} channels")


def preprocess_image(image, target_size=SPATIAL_INPUT_SIZE):
    """Load an 8-bit grayscale scan, resize to target_size, and invert to [1, H, W] float32 in [0, 1]."""
    if isinstance(image, np.ndarray):
        img = image[0] if image.ndim == 3 else image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image}")

    if SPATIAL_SQUARE_PAD:
        h, w = img.shape
        max_dim = max(h, w)
        framed = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        framed[y_off:y_off + h, x_off:x_off + w] = img
    else:
        framed = img

    # Invert polarity (background 0, ink high) so convolutions focus on ink rather than paper.
    resized = cv2.resize(framed, target_size, interpolation=cv2.INTER_AREA)
    normalized = (255.0 - resized.astype(np.float32)) / 255.0
    return normalized[np.newaxis, :, :]


_TRAJECTORY_MIN_COLUMNS = min(TEMPORAL_CHANNEL_WIDTHS.values())
_TRAJECTORY_MAX_COLUMNS = max(TEMPORAL_CHANNEL_WIDTHS.values())


def as_spatial_input(sample, allow_trajectory=False):
    """Normalize a path, scan array, canonical [1, H, W] float array, or [L, C] trajectory into [1, H, W]."""
    if not isinstance(sample, np.ndarray):
        return preprocess_image(sample)

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


def quality_gate(image):
    """Check ink bounding box (>= MIN_BBOX_PIXELS) and Laplacian sharpness (>= MIN_LAPLACIAN_VARIANCE)."""
    if isinstance(image, np.ndarray):
        arr = image[0] if image.ndim == 3 else image
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                # Convert inverted float [0, 1] back to dark-ink uint8 for Otsu thresholding.
                img = np.clip(255.0 - arr * 255.0, 0, 255).astype(np.uint8)
            else:
                img = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            img = arr
    else:
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, {"error": "Cannot load image"}

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, {"error": "No signature content detected"}

    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = laplacian.var()

    details = {"bbox_width": bbox_w, "bbox_height": bbox_h, "sharpness": round(sharpness, 2)}
    passed = bool(
        min(bbox_w, bbox_h) >= MIN_BBOX_PIXELS
        and sharpness >= MIN_LAPLACIAN_VARIANCE
    )
    return passed, details


def trajectory_gate(trajectory):
    """Validate trajectory shape [L, TEMPORAL_INPUT_FEATURES], finiteness, normalization, and dt grid."""
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
        # Validate time delta channel values.
        details["dt_min"] = round(float(trajectory[:, 3].min()), 3)
        details["dt_max"] = round(float(trajectory[:, 3].max()), 3)
        if trajectory[:, 3].min() < 0.0 or trajectory[:, 3].max() > 1.0:
            return False, {**details, "error":
                            "dt outside [0, 1] — it is the inter-sample gap clipped at "
                            "100 ms and scaled, not a raw timestamp."}
        # One grid step is TEMPORAL_SAMPLE_PERIOD_MS of the MAX_GAP_MS clip (0.1 for 10 of 100 ms).
        # Pen lifts are a minority of points, so the median sits on the step.
        step = TEMPORAL_SAMPLE_PERIOD_MS / MAX_GAP_MS
        median_dt = float(np.median(trajectory[:, 3]))
        details["dt_median"] = round(median_dt, 4)
        if abs(median_dt - step) > TRAJECTORY_DT_MEDIAN_TOLERANCE * step:
            return False, {**details, "error":
                            f"dt median {median_dt:.4f} is a {median_dt * MAX_GAP_MS:.1f} ms "
                            f"grid, expected {step:.4f} ({TEMPORAL_SAMPLE_PERIOD_MS:.0f} ms). "
                            "Resample by timestamp onto the 10 ms grid (AGENTS.md, "
                            "Device timing and pressure)."}

    return True, details


def writer_independent_split(writer_ids, train_ratio=SPATIAL_SPLIT_RATIOS[0],
                              val_ratio=SPATIAL_SPLIT_RATIOS[1],
                              test_ratio=SPATIAL_SPLIT_RATIOS[2], seed=42):
    """Split writer IDs into non-overlapping train, validation, and test sets.

    Guarantees no writer appears in more than one partition.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    # Sort first so partition is deterministic and independent of input dict order.
    ids = sorted(writer_ids)  # same order-independence guarantee as writer_independent_split
    rng.shuffle(ids)
    n = len(ids)
    t1 = int(n * train_ratio)
    t2 = t1 + int(n * val_ratio)
    return ids[:t1], ids[t1:t2], ids[t2:]


def split_record_path(weights_path):
    """Return the companion `.split.json` path for a given model weights file."""
    root, _ = os.path.splitext(weights_path)
    return root + ".split.json"


def save_split_record(weights_path, train_ids, val_ids, test_ids, data_dirs, split_seed,
                      holdout_writers=None, input_size=None, max_gap_ms=None,
                      train_only_dirs=None, temporal_channels=None):
    """Save the writer split record beside the trained model weights.

    train_only_dirs (corpora whose writers joined training after the split) is
    written only when set, so every other record keeps its exact shape.
    """
    record = {
        "data_dirs": [str(d) for d in data_dirs],
        "split_seed": split_seed,
        "holdout_writers": sorted(holdout_writers) if holdout_writers else None,
        "train": list(train_ids), "val": list(val_ids), "test": list(test_ids),
        # Save image input size so evaluation uses matching dimensions.
        "input_size": list(input_size) if input_size else None,
        # Save max gap threshold for dt delta calculations.
        "max_gap_ms": float(max_gap_ms) if max_gap_ms else None,
    }
    if train_only_dirs:
        record["train_only_dirs"] = [str(d) for d in train_only_dirs]
    if temporal_channels in TILT_CHANNEL_MODES:
        # Only tilt modes record it, so every existing record keeps its shape.
        record["temporal_channels"] = temporal_channels
    with open(split_record_path(weights_path), "w") as fh:  # skylos: ignore[SKY-D215,SKY-D324] beside the operator-supplied --weights-dir checkpoint
        json.dump(record, fh, indent=1)
    return record


def load_split_record(weights_path):
    """Load the stored writer partition record, or None if not found."""
    path = split_record_path(weights_path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:  # skylos: ignore[SKY-D215,SKY-D325] beside the operator-supplied checkpoint path
        return json.load(fh)


def check_split_against_record(weights_path, test_ids, require_same_test=True,
                               input_size=SPATIAL_INPUT_SIZE, max_gap_ms=None,
                               temporal_channels=None):
    """Check that evaluation split matches the record saved during training.

    Ensures no training writer is evaluated as a test subject, and that
    preprocessing parameters match training configuration.
    """
    record = load_split_record(weights_path)
    if record is None:
        return None
    recorded_size = record.get("input_size")
    if recorded_size and tuple(recorded_size) != tuple(input_size):
        raise ValueError(
            f"Refusing to score {weights_path} at input size "
            f"{input_size[0]}x{input_size[1]}: it was trained at "
            f"{recorded_size[0]}x{recorded_size[1]}. Set "
            f"SIGVER_SPATIAL_INPUT_SIZE={recorded_size[0]}x{recorded_size[1]} to score it.")
    recorded_gap = record.get("max_gap_ms")
    if recorded_gap and max_gap_ms is not None and float(recorded_gap) != float(max_gap_ms):
        raise ValueError(
            f"Refusing to score {weights_path} with MAX_GAP_MS={max_gap_ms:g}: it was "
            f"trained with {recorded_gap:g}. Set SIGVER_MAX_GAP_MS={recorded_gap:g} to score it.")
    recorded_mode = record.get("temporal_channels")
    if recorded_mode and temporal_channels is not None and recorded_mode != temporal_channels:
        raise ValueError(
            f"Refusing to score {weights_path} with --temporal-channels {temporal_channels}: "
            f"it was trained as {recorded_mode}. A different fill loads cleanly and "
            f"gives a plausible wrong number.")
    scoring = set(test_ids)
    leaked = sorted(scoring & set(record["train"]))
    if leaked:
        raise ValueError(
            f"Refusing to score {len(leaked)} writers the checkpoint trained on "
            f"(e.g. {leaked[:5]}). {weights_path} was trained with split seed "
            f"{record['split_seed']} on {record['data_dirs']}"
            + (f" with {len(record['holdout_writers'])} pinned holdout writers" if record["holdout_writers"] else "")
            + "; the partition recomputed here does not match it. Pass the same "
            "--data-dir, --split-seed and --holdout-writers as training, or use "
            "--writers with writers held out by every model compared.")
    if require_same_test and scoring != set(record["test"]):
        raise ValueError(
            f"The recomputed held-out set ({len(scoring)} writers) is not the one "
            f"{weights_path} was trained against ({len(record['test'])} writers, "
            f"{len(scoring - set(record['test']))} differ). No writer leaked, but the "
            "number would describe a different partition than the documented one. "
            f"Training used split seed {record['split_seed']} on {record['data_dirs']}.")
    return record


def pin_holdout(writer_ids, pinned, train_ratio=SPATIAL_SPLIT_RATIOS[0],
                val_ratio=SPATIAL_SPLIT_RATIOS[1],
                test_ratio=SPATIAL_SPLIT_RATIOS[2], seed=42):
    """Split writers into sets, placing all `pinned` writers into the test set."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    ids = sorted(writer_ids)  # same order-independence guarantee as writer_independent_split
    unknown = sorted(set(pinned) - set(ids))
    if unknown:
        raise ValueError(
            f"Cannot pin writers absent from the corpus: {', '.join(unknown)}. "
            f"Check the IDs against the datasets passed to --data-dir."
        )

    pinned = set(pinned)
    rest = [w for w in ids if w not in pinned]

    # Renormalize remaining train and validation ratios over unpinned writers.
    denominator = train_ratio + val_ratio
    if denominator <= 0:
        raise ValueError("train_ratio + val_ratio must be positive to pin a holdout")

    rng = random.Random(seed)
    rng.shuffle(rest)
    cut = int(len(rest) * (train_ratio / denominator))
    return rest[:cut], rest[cut:], sorted(pinned)


def generate_skilled_pairs(samples_by_writer, num_genuine_per_writer=PAIRS_GENUINE_PER_WRITER,
                           num_forged_per_writer=PAIRS_IMPOSTOR_PER_WRITER, seed=42,
                           random_impostor_ratio=0.0, pair_caps=None):
    """Build balanced positive (genuine-genuine) and negative (forged/impostor) pairs per writer."""
    if not 0.0 <= random_impostor_ratio <= 1.0:
        raise ValueError(
            f"random_impostor_ratio must be in [0, 1], got {random_impostor_ratio}")

    rng = random.Random(seed)
    pairs, labels = [], []

    for writer in samples_by_writer:
        genuine = samples_by_writer[writer].get("genuine", [])
        forged = samples_by_writer[writer].get("forged", [])

        if len(genuine) < 2 or not forged:
            continue

        num_pairs = min(
            num_genuine_per_writer,
            num_forged_per_writer,
            len(genuine) * (len(genuine) - 1) // 2,
            len(genuine) * len(forged),
        )
        if pair_caps and writer in pair_caps:
            num_pairs = min(num_pairs, pair_caps[writer])

        genuine_space = list(itertools.combinations(range(len(genuine)), 2))
        rng.shuffle(genuine_space)
        for i, j in genuine_space[:num_pairs]:
            pairs.append((genuine[i], genuine[j]))
            labels.append(1)

        num_random = int(num_pairs * random_impostor_ratio)
        num_skilled = num_pairs - num_random

        impostor_space = list(itertools.product(genuine, forged))
        rng.shuffle(impostor_space)
        for s1, s2 in impostor_space[:num_skilled]:
            pairs.append((s1, s2))
            labels.append(0)

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
    """Pad variable-length trajectory pairs into `(padded1, len1, padded2, len2, labels)`."""
    seqs1, seqs2, labels = zip(*batch)

    lengths1 = torch.tensor([len(s) for s in seqs1], dtype=torch.long)
    lengths2 = torch.tensor([len(s) for s in seqs2], dtype=torch.long)

    padded1 = torch.nn.utils.rnn.pad_sequence(list(seqs1), batch_first=True)
    padded2 = torch.nn.utils.rnn.pad_sequence(list(seqs2), batch_first=True)

    return padded1, lengths1, padded2, lengths2, torch.stack(list(labels))


def augment_image(image, rng, max_rotation_deg=SPATIAL_AUG_MAX_ROTATION_DEG,
                  max_shear=SPATIAL_AUG_MAX_SHEAR, max_scale=SPATIAL_AUG_MAX_SCALE,
                  max_translate=SPATIAL_AUG_MAX_TRANSLATE,
                  stroke_prob=SPATIAL_AUG_STROKE_PROB,
                  blur_sigma_max=SPATIAL_AUG_BLUR_SIGMA,
                  ink_gain_min=SPATIAL_AUG_INK_GAIN_MIN,
                  noise_std_max=SPATIAL_AUG_NOISE_STD):
    """Apply affine warps and optional stroke/scanner perturbations to a [1, H, W] image."""
    channels, height, width = image.shape
    canvas = image[0]

    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    shear = rng.uniform(-max_shear, max_shear)

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 1] += shear
    matrix[0, 2] += rng.uniform(-max_translate, max_translate) * width
    matrix[1, 2] += rng.uniform(-max_translate, max_translate) * height

    # borderValue=0 keeps black background for inverted polarity.
    warped = cv2.warpAffine(canvas, matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    if rng.random() < stroke_prob:
        kernel = np.ones((2, 2), dtype=np.uint8)
        warped = (cv2.dilate(warped, kernel) if rng.random() < 0.5
                  else cv2.erode(warped, kernel))

    # Only draw from RNG when an effect is enabled so default runs keep the exact random stream.
    if blur_sigma_max > 0:
        sigma = rng.uniform(0.0, blur_sigma_max)
        if sigma > 0:
            warped = cv2.GaussianBlur(warped, (0, 0), sigma)
    if ink_gain_min < 1.0:
        warped = warped * rng.uniform(ink_gain_min, 1.0)
    if noise_std_max > 0:
        noise = rng.normal(0.0, rng.uniform(0.0, noise_std_max), warped.shape)
        warped = np.clip(warped + noise, 0.0, 1.0)

    return warped.reshape(channels, height, width).astype(np.float32)


class SpatialSignatureDataset(Dataset):
    """PyTorch Dataset yielding spatial image pairs for Siamese training."""

    def __init__(self, pairs, labels, transform=None, augment=False, seed=None,
                 augment_params=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.augment_params = dict(augment_params or {})
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sample1, sample2 = self.pairs[index]
        img1 = as_spatial_input(sample1, allow_trajectory=True)
        img2 = as_spatial_input(sample2, allow_trajectory=True)

        if self.augment:
            # Warp each side independently so genuine pairs do not stay identical.
            img1 = augment_image(img1, self._rng, **self.augment_params)
            img2 = augment_image(img2, self._rng, **self.augment_params)

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
    """Apply small rotation, coordinate jitter, and speed resampling to a trajectory."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) < 8:
        return sequence

    xy = sequence[:, :2].copy()

    theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xy = xy @ np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    xy += rng.normal(0.0, jitter, size=xy.shape).astype(np.float32)

    scale = 1.0 + rng.uniform(-max_speed_change, max_speed_change)
    new_length = max(8, int(round(len(sequence) * scale)))
    source = np.arange(len(sequence), dtype=np.float32)
    target = np.linspace(0, len(sequence) - 1, new_length, dtype=np.float32)

    columns = [np.interp(target, source, xy[:, 0]),
               np.interp(target, source, xy[:, 1])]

    for channel in range(2, sequence.shape[1]):
        columns.append(np.clip(np.interp(target, source, sequence[:, channel]), 0.0, 1.0))

    return np.column_stack(columns).astype(np.float32)


def augment_device(sequence, rng, rate_range_hz=TEMPORAL_DEVICE_AUG_RATE_HZ,
                   gamma_range=TEMPORAL_DEVICE_AUG_PRESSURE_GAMMA,
                   gain_range=TEMPORAL_DEVICE_AUG_PRESSURE_GAIN,
                   pressure_levels=TEMPORAL_DEVICE_AUG_PRESSURE_LEVELS,
                   dt_channel=3 if TEMPORAL_CHANNELS == "dt" else None):
    """Simulate capture under another tablet's decimation rate and pressure curve."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) < 8:
        return sequence

    rate = rng.uniform(*rate_range_hz)
    factor = rate / max(1, round(rate / TEMPORAL_SAMPLE_RATE_HZ)) / TEMPORAL_SAMPLE_RATE_HZ
    gamma = float(np.exp(rng.uniform(np.log(gamma_range[0]), np.log(gamma_range[1]))))
    gain = rng.uniform(*gain_range)

    new_length = max(8, int(round(len(sequence) * factor)))
    source = np.arange(len(sequence), dtype=np.float32)
    target = np.linspace(0, len(sequence) - 1, new_length, dtype=np.float32)
    out = np.column_stack([np.interp(target, source, sequence[:, c])
                           for c in range(sequence.shape[1])]).astype(np.float32)

    steps = pressure_levels - 1
    curved = gain * np.clip(out[:, 2], 0.0, 1.0) ** gamma
    out[:, 2] = np.round(np.clip(curved, 0.0, 1.0) * steps) / steps
    if dt_channel is not None and out.shape[1] > dt_channel:
        out[:, dt_channel] = np.clip(out[:, dt_channel] / factor, 0.0, 1.0)
    return out


class TemporalSignatureDataset(Dataset):
    """PyTorch Dataset yielding trajectory pairs for Siamese training."""

    def __init__(self, pairs, labels, augment=False, augment_scale=1.0, seed=42,
                 device_augment=False, tilt_hide_prob=0.0):
        self.pairs = pairs
        self.labels = labels
        self.augment = augment
        self.augment_scale = augment_scale
        self.device_augment = device_augment
        self.tilt_hide_prob = tilt_hide_prob
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def _maybe_hide_tilt(self, sequence):
        """Clear tilt columns and presence flag with tilt_hide_prob."""
        if not self.tilt_hide_prob:
            return sequence
        sequence = np.asarray(sequence)
        if sequence.shape[1] != 7:
            raise ValueError(f"tilt hiding needs the 7-wide dt_tilt_flag layout, "
                             f"got width {sequence.shape[1]}")
        if sequence[0, 6] != 1.0 or self._rng.random() >= self.tilt_hide_prob:
            return sequence
        hidden = sequence.copy()
        hidden[:, 4:7] = 0.0
        return hidden

    def _augment(self, sequence):
        return augment_trajectory(sequence, self._rng,
                                  max_rotation_deg=TEMPORAL_AUG_MAX_ROTATION_DEG * self.augment_scale,
                                  jitter=TEMPORAL_AUG_JITTER * self.augment_scale,
                                  max_speed_change=TEMPORAL_AUG_MAX_SPEED_CHANGE * self.augment_scale)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        seq1, seq2 = self.pairs[index]
        if self.augment:
            seq1 = self._augment(seq1)
            seq2 = self._augment(seq2)
        if self.device_augment:
            seq1 = augment_device(seq1, self._rng)
            seq2 = augment_device(seq2, self._rng)
        seq1 = self._maybe_hide_tilt(seq1)
        seq2 = self._maybe_hide_tilt(seq2)
        return (
            torch.tensor(seq1, dtype=torch.float32),
            torch.tensor(seq2, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )
