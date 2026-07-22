"""
Dataset Pipeline — Signature Verification System

Handles data loading, preprocessing, augmentation, quality gating, and
writer-independent splitting for both spatial (image) and temporal
(dynamic) signature datasets.

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
    Load and preprocess a signature image.
    Pipeline: grayscale → aspect-ratio-preserving pad → resize → normalize.
    Returns: numpy array of shape (1, H, W) as float32 in [0, 1].
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = img.shape
    max_dim = max(h, w)
    padded = np.full((max_dim, max_dim), 255, dtype=np.uint8)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img

    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, :, :]


def quality_gate(image_path):
    """
    Check quality via effective-resolution proxies.
    Does NOT gate on file DPI tag. Uses bounding-box size + Laplacian sharpness.
    Returns: (passed: bool, details: dict)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, {"error": "Cannot load image"}

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, {"error": "No signature content detected"}

    # Bounding-box size check across all signature contours
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


def writer_independent_split(writer_ids, train_ratio=0.7, val_ratio=0.15,
                              test_ratio=0.15, seed=42):
    """
    Split writer IDs into non-overlapping train/val/test sets.
    No writer appears in more than one partition (ISO/IEC 23053).
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
    Generate genuine and impostor pairs from samples grouped by writer.

    Args:
        samples_by_writer: dict[writer_id → list of sample paths/arrays]
        num_genuine_per_writer: number of genuine pairs per writer
        num_impostor_per_writer: number of impostor pairs per writer

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

        # Genuine pairs (same writer)
        for _ in range(min(num_genuine_per_writer, len(samples) * (len(samples)-1) // 2)):
            s1, s2 = rng.sample(samples, 2)
            pairs.append((s1, s2))
            labels.append(1)

        # Impostor pairs (different writer)
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
    Dataset for spatial (image) signature pairs.
    Each sample: (img1_tensor, img2_tensor, label).
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
    Dataset for temporal (dynamic) signature pairs.
    Each sample: (seq1_tensor, seq2_tensor, label).
    Sequences are variable-length (L, 3); collation handles padding.
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
