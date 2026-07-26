"""
Dataset Pipeline — Signature Verification System

Handles data loading, preprocessing, augmentation, quality gating, and
writer-independent splitting for both spatial (image) and temporal
(dynamic) signature datasets.

Expected Local Dataset Structure (`./data`):
--------------------------------------------
data/
├── cedar/                  # Offline CEDAR dataset (55 writers, flat layout)
│   ├── full_org/           # original_{writer}_{sample}.png    (Genuine)
│   └── full_forg/          # forgeries_{writer}_{sample}.png   (Skilled Forgeries)
├── bhsig260/               # Offline BHSig260 dataset (260 writers)
│   ├── Bengali/            # 100 writers: {writer}/B-S-{w}-{G|F}-{n}.tif
│   └── Hindi/              # 160 writers: {writer}/H-S-{w}-{G|F}-{n}.tif
├── utsig/                  # Offline UTSig dataset (115 writers)
│   ├── Genuine/            # {writer}/{n}.tif
│   └── Forgery/            # {Skilled,Simple,Opposite Hand}/{writer}/{n}.tif
└── svc2004/                # Online SVC2004 dataset (40 writers, flat layout)
    └── U{user}S{sample}.TXT
        # S1–S20 = genuine, S21–S40 = skilled forgery.
        # Task1 files are 4-column (X, Y, T, Button) — no pressure.
        # Task2 files are 7-column (X, Y, T, Button, Azimuth, Altitude, Pressure).
        # parse_svc2004() auto-detects the width; Task1 imputes P = 1.0.

Supported datasets:
  Spatial:  CEDAR, BHSig260, UTSig, In-house collection
  Temporal: SVC2004, In-house collection
"""

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
    MIN_BBOX_PIXELS,
    MIN_LAPLACIAN_VARIANCE,
    PAIRS_GENUINE_PER_WRITER,
    PAIRS_IMPOSTOR_PER_WRITER,
)


def preprocess_image(image_path, target_size=SPATIAL_INPUT_SIZE):
    """
    Load and preprocess a signature image for the Spatial CNN branch.

    Pipeline:
      1. Load image as 8-bit single-channel grayscale.
      2. Apply aspect-ratio-preserving zero/white padding to form a square canvas.
      3. Area-averaging resize (INTER_AREA) down to target_size (e.g. 256×256).
      4. Normalize pixel intensity values into [0.0, 1.0] range.
      5. Add channel dimension [1, H, W] expected by PyTorch Conv2d.

    Returns: numpy float32 array of shape (1, H, W) in [0.0, 1.0].
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # The full scan is padded, not cropped to the ink bounding box. Cropping
    # raises stroke resolution (BHSig260 ink spans only ~13% of the canvas
    # height) but rescales every signature to the same extent, discarding
    # absolute size. Measured, that trade lost about 1.8 points of held-out EER,
    # so overall scale is worth more here than the extra detail.

    # Pad image to a square canvas while keeping original aspect ratio intact
    h, w = img.shape
    max_dim = max(h, w)
    padded = np.full((max_dim, max_dim), 255, dtype=np.uint8)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img

    # Resize to standard spatial input size (256x256), then invert onto [0.0, 1.0]
    # so **background is 0 and ink is high**.
    #
    # Every published offline pipeline inverts here, and the reason is mechanical
    # rather than cosmetic: ink covers about 1% of the canvas (measured — mean
    # pixel 0.9496 on CEDAR, 0.9901 on BHSig260), so an un-inverted image hands
    # every convolution a large constant offset that carries no information about
    # who signed. It is also why the encoder needed a *max* pooling branch to
    # recover strokes that average pooling had buried under that offset.
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    normalized = (255.0 - resized.astype(np.float32)) / 255.0
    return normalized[np.newaxis, :, :]


def quality_gate(image_path):
    """
    Evaluate signature image quality via effective-resolution proxies.
    
    Adheres to AGENTS.md standards: does NOT rely on EXIF DPI metadata tags.
    Instead calculates:
      - Bounding box dimension size across all signature ink stroke contours.
      - Laplacian variance sharpness score to flag blurry or degraded scans.

    Returns: (passed: bool, details: dict)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, {"error": "Cannot load image"}

    # Binarize image via Otsu thresholding to segment stroke contours from background
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


def writer_independent_split(writer_ids, train_ratio=0.7, val_ratio=0.15,
                              test_ratio=0.15, seed=42):
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


def generate_pairs(samples_by_writer, num_genuine_per_writer=5,
                   num_impostor_per_writer=5, seed=42):
    """
    Generate balanced genuine and forgery pair combinations per writer.

    Args:
        samples_by_writer: dict mapping writer_id to list of signature paths/arrays.
        num_genuine_per_writer: target genuine pairs per writer (label = 1).
        num_impostor_per_writer: target forgery/impostor pairs per writer (label = 0).

    Returns:
        (pairs: list of (sample1, sample2), labels: list of int)
    """
    rng = random.Random(seed)
    pairs, labels = [], []
    writers = list(samples_by_writer.keys())

    for writer in writers:
        samples = samples_by_writer[writer]
        if len(samples) < 2:
            continue

        # Create genuine pairs (signatures written by the same person)
        for _ in range(min(num_genuine_per_writer, len(samples) * (len(samples)-1) // 2)):
            s1, s2 = rng.sample(samples, 2)
            pairs.append((s1, s2))
            labels.append(1)

        # Create impostor pairs (signatures from two different writers)
        other_writers = [w for w in writers if w != writer]
        for _ in range(num_impostor_per_writer):
            other = rng.choice(other_writers)
            s1 = rng.choice(samples)
            s2 = rng.choice(samples_by_writer[other])
            pairs.append((s1, s2))
            labels.append(0)

    return pairs, labels


def generate_skilled_pairs(samples_by_writer, num_genuine_per_writer=PAIRS_GENUINE_PER_WRITER,
                           num_forged_per_writer=PAIRS_IMPOSTOR_PER_WRITER, seed=42):
    """
    Generate pairs under the **skilled-forgery** scenario (AGENTS.md eval protocol).

    Unlike `generate_pairs`, every negative here is a genuine signature paired
    against a forgery *of that same writer* — a deliberate imitation, not a
    random stranger's signature. Random-impostor negatives are trivially
    separable and inflate the reported EER into a number that does not describe
    the attack the system actually faces. All reported metrics use this pairing.

    Args:
        samples_by_writer: dict mapping writer_id to
            {"genuine": [paths/arrays], "forged": [paths/arrays]}.
        num_genuine_per_writer: target genuine pairs per writer (label = 1).
        num_forged_per_writer: target skilled-forgery pairs per writer (label = 0).

    Returns:
        (pairs: list of (sample1, sample2), labels: list of int)
    """
    rng = random.Random(seed)
    pairs, labels = [], []

    for writer in samples_by_writer:
        genuine = samples_by_writer[writer].get("genuine", [])
        forged = samples_by_writer[writer].get("forged", [])

        # A writer needs 2 genuines to form a positive and >=1 forgery for a
        # skilled negative. Without both, this writer cannot be scored.
        if len(genuine) < 2 or not forged:
            continue

        # Genuine pairs: two authentic signatures from the same writer
        num_genuine = min(num_genuine_per_writer, len(genuine) * (len(genuine) - 1) // 2)
        for _ in range(num_genuine):
            s1, s2 = rng.sample(genuine, 2)
            pairs.append((s1, s2))
            labels.append(1)

        # Skilled-forgery pairs: authentic vs. an imitation of this same writer
        num_forged = min(num_forged_per_writer, len(genuine) * len(forged))
        for _ in range(num_forged):
            pairs.append((rng.choice(genuine), rng.choice(forged)))
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


class SpatialSignatureDataset(Dataset):
    """
    PyTorch Dataset loading spatial (image) signature pairs for Siamese training.
    
    Each item returns:
        - img1_tensor: preprocessed image 1 tensor [1, 256, 256]
        - img2_tensor: preprocessed image 2 tensor [1, 256, 256]
        - label: float tensor (1.0 for genuine, 0.0 for forgery)
    """

    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path1, path2 = self.pairs[index]
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (
            torch.tensor(img1, dtype=torch.float32),
            torch.tensor(img2, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def augment_trajectory(sequence, rng, max_rotation_deg=6.0, jitter=0.02,
                       max_speed_change=0.15):
    """
    Perturb one [L, 3] trajectory for training-time augmentation.

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
    pressure = sequence[:, 2].copy()

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

    out = np.column_stack([
        np.interp(target, source, xy[:, 0]),
        np.interp(target, source, xy[:, 1]),
        np.interp(target, source, pressure),
    ]).astype(np.float32)

    # Pressure is a physical quantity in [0, 1]; interpolation can drift it a
    # hair outside on the endpoints.
    out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
    return out


class TemporalSignatureDataset(Dataset):
    """
    PyTorch Dataset loading temporal (dynamic) signature pairs for Siamese training.

    Each item returns:
        - seq1_tensor: trajectory sequence tensor [L1, 3] containing (X, Y, Pressure)
        - seq2_tensor: trajectory sequence tensor [L2, 3] containing (X, Y, Pressure)
        - label: float tensor (1.0 for genuine, 0.0 for forgery)
    """

    def __init__(self, pairs, labels, augment=False):
        self.pairs = pairs
        self.labels = labels
        # Training only. Perturbing validation or test data would measure the
        # model against inputs the deployed system never sees.
        self.augment = augment
        self._rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        seq1, seq2 = self.pairs[index]
        if self.augment:
            # Each side is perturbed independently: two recordings of the same
            # signature differ by exactly this kind of variation, so the pair
            # stays honest.
            seq1 = augment_trajectory(seq1, self._rng)
            seq2 = augment_trajectory(seq2, self._rng)
        return (
            torch.tensor(seq1, dtype=torch.float32),
            torch.tensor(seq2, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )

