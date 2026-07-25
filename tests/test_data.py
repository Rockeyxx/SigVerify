"""Tests for the data pipeline — preprocessing, quality gates, splitting, datasets."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import (
    SpatialSignatureDataset,
    TemporalSignatureDataset,
    collate_temporal_pairs,
    generate_skilled_pairs,
    preprocess_image,
    quality_gate,
    writer_independent_split,
)


class TestPreprocessImage:
    """Test suite for signature image preprocessing functions."""

    def _make_test_image(self, h=300, w=400):
        """Create a temporary synthetic grayscale signature image on white background."""
        img = np.ones((h, w), dtype=np.uint8) * 255
        # Draw mock signature strokes using OpenCV
        cv2.line(img, (50, 150), (350, 150), 0, 3)
        cv2.line(img, (100, 100), (200, 200), 0, 2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    def test_output_shape(self):
        """Verify preprocessed image tensor yields standard shape (1, 256, 256)."""
        path = self._make_test_image()
        result = preprocess_image(path)
        assert result.shape == (1, 256, 256)
        os.unlink(path)

    def test_output_normalized(self):
        """Verify pixel intensity values are strictly scaled to float range [0.0, 1.0]."""
        path = self._make_test_image()
        result = preprocess_image(path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        os.unlink(path)

    def test_nonexistent_file_raises(self):
        """Verify FileNotFoundError is raised when image path is invalid."""
        with pytest.raises(FileNotFoundError):
            preprocess_image("/nonexistent/path.png")


class TestQualityGate:
    """Test suite for effective-resolution and sharpness quality gating."""

    def _make_sharp_image(self):
        """Create a sharp image containing distinct high-frequency strokes."""
        img = np.ones((400, 400), dtype=np.uint8) * 255
        for i in range(50, 350, 5):
            cv2.line(img, (i, 100), (i + 30, 300), 0, 2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    def _make_tiny_image(self):
        """Create a small low-resolution image below threshold bounds."""
        img = np.ones((30, 30), dtype=np.uint8) * 255
        cv2.circle(img, (15, 15), 5, 0, 1)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    def test_sharp_image_passes(self):
        """Verify high-resolution sharp image passes quality gate checks."""
        path = self._make_sharp_image()
        passed, details = quality_gate(path)
        assert passed is True
        assert "sharpness" in details
        os.unlink(path)

    def test_tiny_image_fails(self):
        """Verify undersized image fails bounding-box dimension proxy check."""
        path = self._make_tiny_image()
        passed, details = quality_gate(path)
        assert passed is False
        os.unlink(path)


class TestWriterIndependentSplit:
    """Test suite for ISO/IEC 23053 writer-independent data partition splitting."""

    def test_no_overlap(self):
        """Ensure writer IDs in train, validation, and test sets are completely disjoint."""
        ids = list(range(100))
        train, val, test = writer_independent_split(ids)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_all_ids_covered(self):
        """Ensure all writer IDs are accounted for across the partitions."""
        ids = list(range(100))
        train, val, test = writer_independent_split(ids)
        assert set(train) | set(val) | set(test) == set(ids)

    def test_approximate_ratios(self):
        """Verify writer split counts match requested ratios (70% / 15% / 15%)."""
        ids = list(range(100))
        train, val, test = writer_independent_split(ids, 0.7, 0.15, 0.15)
        assert 60 <= len(train) <= 80
        assert 10 <= len(val) <= 20
        assert 10 <= len(test) <= 20


class TestGenerateSkilledPairs:
    """Test suite for skilled-forgery-aware pairing (AGENTS.md evaluation protocol)."""

    def _writers(self, n=4, n_genuine=6, n_forged=6):
        """Build a synthetic {writer: {genuine: [...], forged: [...]}} corpus."""
        return {
            f"w{w}": {
                "genuine": [f"w{w}_g{i}" for i in range(n_genuine)],
                "forged": [f"w{w}_f{i}" for i in range(n_forged)],
            }
            for w in range(n)
        }

    def test_impostor_pairs_are_same_writer_forgeries(self):
        """
        The whole point of A8: every label-0 pair must be
        (genuine, that same writer's forgery) — never a random cross-writer sample.
        """
        pairs, labels = generate_skilled_pairs(self._writers(),
                                               num_genuine_per_writer=5,
                                               num_forged_per_writer=5)
        forgery_pairs = [p for p, y in zip(pairs, labels) if y == 0]
        assert len(forgery_pairs) > 0

        for s1, s2 in forgery_pairs:
            writer1, kind1 = s1.split("_")
            writer2, kind2 = s2.split("_")
            assert writer1 == writer2, f"cross-writer impostor leaked in: {s1} vs {s2}"
            assert kind1.startswith("g") and kind2.startswith("f")

    def test_genuine_pairs_are_same_writer_genuines(self):
        """Every label-1 pair must be two genuine samples from one writer."""
        pairs, labels = generate_skilled_pairs(self._writers())
        for (s1, s2), y in zip(pairs, labels):
            if y == 1:
                assert s1.split("_")[0] == s2.split("_")[0]
                assert s1.split("_")[1].startswith("g")
                assert s2.split("_")[1].startswith("g")

    def test_labels_are_balanced(self):
        """Equal budgets must yield a balanced genuine/forgery label distribution."""
        _, labels = generate_skilled_pairs(self._writers(),
                                           num_genuine_per_writer=5,
                                           num_forged_per_writer=5)
        assert labels.count(1) == labels.count(0)

    def test_writer_without_forgeries_is_skipped(self):
        """A writer with no forgeries cannot form skilled pairs and must be dropped."""
        writers = self._writers(n=2)
        writers["w0"]["forged"] = []
        pairs, labels = generate_skilled_pairs(writers)
        assert all("w0" not in s1 for s1, _ in pairs)

    def test_deterministic_under_seed(self):
        """Same seed must reproduce the same pair list (ISO/IEC 23053 reproducibility)."""
        a = generate_skilled_pairs(self._writers(), seed=7)
        b = generate_skilled_pairs(self._writers(), seed=7)
        assert a == b


class TestCollateTemporalPairs:
    """Test suite for variable-length temporal batch collation."""

    def test_pads_to_longest_and_reports_true_lengths(self):
        """Sequences pad to the batch max while true lengths survive for packing."""
        batch = [
            (torch.randn(300, 3), torch.randn(120, 3), torch.tensor(1.0)),
            (torch.randn(150, 3), torch.randn(200, 3), torch.tensor(0.0)),
        ]
        seq1, len1, seq2, len2, labels = collate_temporal_pairs(batch)

        assert seq1.shape == (2, 300, 3)
        assert seq2.shape == (2, 200, 3)
        assert torch.equal(len1, torch.tensor([300, 150]))
        assert torch.equal(len2, torch.tensor([120, 200]))
        assert torch.equal(labels, torch.tensor([1.0, 0.0]))

    def test_padding_region_is_zero(self):
        """Padded tail must be zeros so packing has a well-defined no-op region."""
        batch = [
            (torch.ones(10, 3), torch.ones(10, 3), torch.tensor(1.0)),
            (torch.ones(4, 3), torch.ones(4, 3), torch.tensor(0.0)),
        ]
        seq1, _, _, _, _ = collate_temporal_pairs(batch)
        assert torch.all(seq1[1, 4:] == 0)


class TestSpatialSignatureDataset:
    """Test suite for PyTorch SpatialSignatureDataset pair loader."""

    def test_len_and_getitem(self):
        """Verify PyTorch Dataset length and item indexing tensor shapes."""
        imgs = []
        for _ in range(4):
            img = np.ones((100, 100), dtype=np.uint8) * 255
            cv2.line(img, (10, 50), (90, 50), 0, 2)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, img)
            imgs.append(tmp.name)

        pairs = [(imgs[0], imgs[1]), (imgs[2], imgs[3])]
        labels = [1, 0]
        ds = SpatialSignatureDataset(pairs, labels)
        assert len(ds) == 2

        img1, img2, label = ds[0]
        assert img1.shape == (1, 256, 256)
        assert img2.shape == (1, 256, 256)
        assert label.item() == 1.0

        for p in imgs:
            os.unlink(p)

