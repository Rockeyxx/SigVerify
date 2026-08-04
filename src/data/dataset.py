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
    MIN_LAPLACIAN_VARIANCE,
    PAIRS_GENUINE_PER_WRITER,
    PAIRS_IMPOSTOR_PER_WRITER,
    RANDOM_IMPOSTOR_RATIO,
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

    if SPATIAL_SQUARE_PAD:
        # Pad image to a square canvas while keeping original aspect ratio intact
        h, w = img.shape
        max_dim = max(h, w)
        padded = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = img
    else:
        # Resize straight onto the target, as the Siamese reference does at
        # 155x220. Aspect is distorted but consistently so, and every signature
        # fills the frame instead of spending most of it on pad — BHSig260's
        # ribbons are roughly 1:3.5, so square padding leaves them ~70% blank.
        padded = img

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
                           num_forged_per_writer=PAIRS_IMPOSTOR_PER_WRITER, seed=42,
                           random_impostor_ratio=RANDOM_IMPOSTOR_RATIO):
    """
    Generate pairs under the **skilled-forgery** scenario (AGENTS.md eval protocol).

    Unlike `generate_pairs`, negatives are by default a genuine signature paired
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


def augment_image(image, rng, max_rotation_deg=6.0, max_shear=0.08,
                  max_scale=0.08, max_translate=0.04, stroke_prob=0.3):
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
    PyTorch Dataset loading spatial (image) signature pairs for Siamese training.
    
    Each item returns:
        - img1_tensor: preprocessed image 1 tensor [1, 256, 256]
        - img2_tensor: preprocessed image 2 tensor [1, 256, 256]
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
        path1, path2 = self.pairs[index]
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)

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

