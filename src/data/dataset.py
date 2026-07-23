"""
Dataset Pipeline — Signature Verification System

Handles data loading, preprocessing, augmentation, quality gating, and
writer-independent splitting for both spatial (image) and temporal
(dynamic) signature datasets.

Expected Local Dataset Structure (`./data`):
--------------------------------------------
data/
├── cedar/                  # Offline CEDAR dataset (55 writers)
│   ├── writer_01/
│   │   ├── original_1_1.png ... original_1_24.png  (Genuine)
│   │   └── forged_1_1.png   ... forged_1_24.png    (Skilled Forgeries)
│   └── ...
├── bhsig260/               # Offline BHSig260 dataset (260 writers)
│   ├── Bengali/
│   └── Hindi/
└── svc2004/                # Online SVC2004 dataset (40 writers)
    └── Task1/              # Coordinates (X, Y, T, Button, Azimuth, Altitude, Pressure)
        ├── User1.txt ... User40.txt
        └── ...

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
)


def preprocess_image(image_path, target_size=SPATIAL_INPUT_SIZE):
    """
    Load and preprocess a signature image for the Spatial CNN branch.

    Pipeline:
      1. Load image as 8-bit single-channel grayscale.
      2. Apply aspect-ratio-preserving zero/white padding to form a square canvas.
      3. Bilinear resize down to target_size (e.g. 256×256).
      4. Normalize pixel intensity values into [0.0, 1.0] range.
      5. Add channel dimension [1, H, W] expected by PyTorch Conv2d.

    Returns: numpy float32 array of shape (1, H, W) in [0.0, 1.0].
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Pad image to a square canvas while keeping original aspect ratio intact
    h, w = img.shape
    max_dim = max(h, w)
    padded = np.full((max_dim, max_dim), 255, dtype=np.uint8)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img

    # Resize to standard spatial input size (256x256) and scale intensity to [0.0, 1.0]
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
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


class TemporalSignatureDataset(Dataset):
    """
    PyTorch Dataset loading temporal (dynamic) signature pairs for Siamese training.

    Each item returns:
        - seq1_tensor: trajectory sequence tensor [L1, 3] containing (X, Y, Pressure)
        - seq2_tensor: trajectory sequence tensor [L2, 3] containing (X, Y, Pressure)
        - label: float tensor (1.0 for genuine, 0.0 for forgery)
    """

    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        seq1, seq2 = self.pairs[index]
        return (
            torch.tensor(seq1, dtype=torch.float32),
            torch.tensor(seq2, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )

