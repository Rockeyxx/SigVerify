"""Tests for the data pipeline — preprocessing, quality gates, splitting, datasets."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch

from config.settings import SPATIAL_INPUT_SIZE
from src.data.dataset import (
    SpatialSignatureDataset,
    TemporalSignatureDataset,
    collate_temporal_pairs,
    generate_skilled_pairs,
    augment_trajectory,
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
        """
        Verify the preprocessed tensor matches the configured input geometry.

        Read from config rather than hardcoded: SPATIAL_INPUT_SIZE is a measured
        choice (155x220 direct resize beat 256x256 square-pad on every writer_samples),
        so a test that pins 256 would fail the next time it is measured again.
        """
        path = self._make_test_image()
        result = preprocess_image(path)
        width, height = SPATIAL_INPUT_SIZE
        assert result.shape == (1, height, width)
        os.unlink(path)

    def test_output_normalized(self):
        """Verify pixel intensity values are strictly scaled to float range [0.0, 1.0]."""
        path = self._make_test_image()
        result = preprocess_image(path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        os.unlink(path)

    def test_background_is_zero_and_ink_is_high(self, tmp_path):
        """
        Published pipelines invert so background reads 0 and ink reads high, and
        the reason is mechanical: ink covers about 1% of the canvas (measured —
        cedar mean pixel 0.9496, bhsig260 0.9901), so an un-inverted image feeds
        every convolution a large constant offset carrying no identity signal.

        Asserted per region rather than on min/max, which are 0.0 and 1.0 either
        way round and so cannot tell the two polarities apart.
        """
        path = str(tmp_path / "stroke.png")
        img = np.full((200, 200), 255, dtype=np.uint8)
        img[90:110, 40:160] = 0                      # a horizontal stroke
        cv2.imwrite(path, img)

        result = preprocess_image(path)[0]
        height, width = result.shape

        background = result[:height // 8, :width // 8]      # top-left corner
        ink = result[height // 2, width // 2]               # centre of the stroke

        assert background.max() == pytest.approx(0.0, abs=1e-6)
        assert ink > 0.9

    def test_blank_page_is_entirely_zero(self, tmp_path):
        """Background and the square pad must both read as absence of ink."""
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, np.full((100, 200), 255, dtype=np.uint8))

        assert preprocess_image(path).sum() == pytest.approx(0.0, abs=1e-4)

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
        """Build a synthetic {writer: {genuine: [...], forged: [...]}} mapping."""
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
        width, height = SPATIAL_INPUT_SIZE
        assert img1.shape == (1, height, width)
        assert img2.shape == (1, height, width)
        assert label.item() == 1.0

        for p in imgs:
            os.unlink(p)



class TestAugmentTrajectory:
    """Training-time augmentation for online signatures."""

    @staticmethod
    def _traj(n=60):
        t = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
        return np.column_stack([
            np.sin(t), np.cos(t), np.full(n, 0.5, dtype=np.float32)
        ]).astype(np.float32)

    def test_shape_and_channel_count_survive(self):
        out = augment_trajectory(self._traj(), np.random.RandomState(0))
        assert out.ndim == 2 and out.shape[1] == 3
        assert out.dtype == np.float32

    def test_pressure_stays_in_range(self):
        """Rotation and jitter apply to X/Y only; P must remain a valid pressure."""
        traj = self._traj()
        traj[:, 2] = np.linspace(0.0, 1.0, len(traj))
        out = augment_trajectory(traj, np.random.RandomState(1))
        assert out[:, 2].min() >= 0.0 and out[:, 2].max() <= 1.0

    def test_output_differs_from_input(self):
        traj = self._traj()
        out = augment_trajectory(traj, np.random.RandomState(2))
        assert not np.allclose(out[:, :2], traj[:len(out), :2])

    def test_stroke_shape_is_broadly_preserved(self):
        """
        Augmentation must perturb, not destroy. A signature rotated a few degrees
        with slight speed variation is still that person's signature; one that no
        longer correlates with the original is a different object and teaches the
        model something false.
        """
        traj = self._traj(200)
        out = augment_trajectory(traj, np.random.RandomState(3))
        # Speed change alters the length, so resample onto the original timeline
        # before comparing; otherwise this measures the time warp, not the shape.
        resampled = np.interp(
            np.linspace(0, 1, len(traj)),
            np.linspace(0, 1, len(out)),
            out[:, 0],
        )
        corr = np.corrcoef(resampled, traj[:, 0])[0, 1]
        assert corr > 0.9, f"X trajectory correlation {corr:.3f} is too low"

    def test_is_deterministic_given_the_same_rng_seed(self):
        a = augment_trajectory(self._traj(), np.random.RandomState(7))
        b = augment_trajectory(self._traj(), np.random.RandomState(7))
        assert np.allclose(a, b)

    def test_short_sequences_are_returned_untouched(self):
        """Too few points to resample; augmenting would corrupt rather than vary."""
        tiny = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 0.5]], dtype=np.float32)
        assert np.allclose(augment_trajectory(tiny, np.random.RandomState(0)), tiny)


class TestTemporalDatasetAugmentation:
    """Augmentation must be opt-in and confined to training."""

    @staticmethod
    def _pairs():
        seq = np.column_stack([
            np.linspace(0, 1, 50), np.linspace(1, 0, 50), np.full(50, 0.5)
        ]).astype(np.float32)
        return [(seq, seq.copy())], [1.0]

    def test_off_by_default_is_reproducible(self):
        pairs, labels = self._pairs()
        ds = TemporalSignatureDataset(pairs, labels)
        first, _, _ = ds[0]
        second, _, _ = ds[0]
        assert torch.allclose(first, second)

    def test_enabled_augmentation_varies_between_reads(self):
        pairs, labels = self._pairs()
        ds = TemporalSignatureDataset(pairs, labels, augment=True)
        reads = [ds[0][0] for _ in range(6)]
        # A differing length is itself variation (the speed change resamples),
        # so only compare values when the shapes happen to match.
        varied = any(r.shape != reads[0].shape or not torch.allclose(reads[0], r)
                     for r in reads[1:])
        assert varied, "augment=True produced identical tensors every read"

    def test_label_is_untouched_by_augmentation(self):
        pairs, labels = self._pairs()
        ds = TemporalSignatureDataset(pairs, labels, augment=True)
        assert float(ds[0][2]) == 1.0


class TestSkilledPairBalance:
    """
    A writer must contribute equal numbers of positives and negatives.

    UTSig ships 27 genuine but only 6 skilled forgeries per writer, so it
    supports C(27,2)=351 genuine combinations against 27*6=162 impostor ones.
    Capping each arm only by its own supply gave UTSig a 1.70:1 genuine-heavy
    batch at a 276 budget while every other dataset stayed 1:1 — and UTSig's EER
    rose 38.83% -> 43.43% when the budget was raised.
    """

    def _dataset(self, num_genuine, num_forged):
        return {"utsig_1": {"genuine": [f"g{i}" for i in range(num_genuine)],
                            "forged": [f"f{i}" for i in range(num_forged)]}}

    def test_scarce_forgeries_cap_the_genuine_arm_too(self):
        pairs, labels = generate_skilled_pairs(
            self._dataset(27, 6), num_genuine_per_writer=276,
            num_forged_per_writer=276, seed=42)
        assert labels.count(1) == labels.count(0)

    def test_balanced_datasets_are_unaffected(self):
        """CEDAR's 24/24 supports 276 of each; this must not reduce it."""
        pairs, labels = generate_skilled_pairs(
            self._dataset(24, 24), num_genuine_per_writer=276,
            num_forged_per_writer=276, seed=42)
        assert labels.count(1) == labels.count(0) == 276

    def test_small_budget_still_honoured(self):
        pairs, labels = generate_skilled_pairs(
            self._dataset(27, 6), num_genuine_per_writer=40,
            num_forged_per_writer=40, seed=42)
        assert labels.count(1) == labels.count(0) == 40


class TestSpatialInputGeometry:
    """
    Square-pad versus direct resize. Which one wins is a measurement, not a
    preference — see docs/SOLUTIONS.md for the numbers.
    """

    def _wide_image(self, tmp_path):
        """A 1:3.5 ribbon, the shape BHSig260 ships."""
        path = str(tmp_path / "ribbon.png")
        img = np.full((100, 350), 255, dtype=np.uint8)
        img[45:55, 20:330] = 0
        cv2.imwrite(path, img)
        return path

    def test_square_pad_preserves_aspect(self, tmp_path, monkeypatch):
        """Padded, a 1:3.5 ribbon keeps its shape and the frame is mostly empty."""
        import src.data.dataset as ds
        monkeypatch.setattr(ds, "SPATIAL_SQUARE_PAD", True)
        out = ds.preprocess_image(self._wide_image(tmp_path), target_size=(256, 256))[0]
        rows_with_ink = (out.max(axis=1) > 0.5).sum()
        assert rows_with_ink < out.shape[0] // 3      # ink confined to a band

    def test_direct_resize_gives_the_stroke_more_pixels_than_padding(
            self, tmp_path, monkeypatch):
        """
        The trade this measurement is about. Padding a 1:3.5 ribbon to square
        spends ~70% of the canvas on blank and shrinks the stroke to roughly a
        third of the vertical resolution a direct resize would give it.

        Compared against each other rather than a fixed threshold, since the
        ratio is the meaningful quantity and it follows from the aspect ratio.
        """
        import src.data.dataset as ds
        path = self._wide_image(tmp_path)

        monkeypatch.setattr(ds, "SPATIAL_SQUARE_PAD", True)
        padded = (ds.preprocess_image(path, target_size=(256, 256))[0].max(axis=1) > 0.5).sum()

        monkeypatch.setattr(ds, "SPATIAL_SQUARE_PAD", False)
        direct = (ds.preprocess_image(path, target_size=(256, 256))[0].max(axis=1) > 0.5).sum()

        assert direct > 2 * padded

    def test_non_square_target_shape_is_honoured(self, tmp_path, monkeypatch):
        """155x220 is the reference geometry; cv2 takes (width, height)."""
        import src.data.dataset as ds
        monkeypatch.setattr(ds, "SPATIAL_SQUARE_PAD", False)
        out = ds.preprocess_image(self._wide_image(tmp_path), target_size=(220, 155))
        assert out.shape == (1, 155, 220)
