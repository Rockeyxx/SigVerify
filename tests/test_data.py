"""Tests for the data pipeline — preprocessing, quality gates, splitting, datasets."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch

from config.settings import (MIN_BBOX_PIXELS, MIN_TRAJECTORY_POINTS,
                             SPATIAL_INPUT_SIZE, TEMPORAL_INPUT_FEATURES)
from src.data.dataset import (
    pin_holdout,
    SpatialSignatureDataset,
    TemporalSignatureDataset,
    collate_temporal_pairs,
    generate_skilled_pairs,
    augment_trajectory,
    augment_image,
    preprocess_image,
    quality_gate,
    trajectory_gate,
    writer_independent_split,
)


class TestTrajectoryGate:
    """
    The temporal counterpart to the image quality gate.

    These exist because the failure this catches is silent and confident: an
    unnormalized trajectory scored 0.4888 (Forged) where the same signature
    normalized scored 0.9370 (Genuine), with nothing flagged.
    """

    def _normalized(self, n=100, seed=0):
        """
        A trajectory shaped the way `normalize_trajectory` leaves one, at
        whatever width `TEMPORAL_CHANNELS` currently selects.
        """
        rng = np.random.default_rng(seed)
        xy = rng.normal(0.0, 1.0, size=(n, 2))
        xy = (xy - xy.mean(axis=0)) / xy.std(axis=0)
        columns = [xy, rng.uniform(0.1, 1.0, size=(n, 1))]
        for _ in range(TEMPORAL_INPUT_FEATURES - 3):
            columns.append(np.full((n, 1), 0.1))
        return np.hstack(columns).astype(np.float32)

    def test_normalized_trajectory_passes(self):
        passed, details = trajectory_gate(self._normalized())
        assert passed, details

    def test_raw_device_units_are_rejected(self):
        """The exact failure measured against the deployed path."""
        raw = self._normalized()
        raw[:, 0] = raw[:, 0] * 1200 + 8000
        raw[:, 1] = raw[:, 1] * 1200 + 6000
        raw[:, 2] = raw[:, 2] * 1023

        passed, details = trajectory_gate(raw)
        assert not passed
        assert "z-scored" in details["error"]

    def test_offset_but_scaled_trajectory_is_rejected(self):
        """A capture that centred nothing still fails — the mean gives it away."""
        shifted = self._normalized()
        shifted[:, :2] += 4.0
        passed, _ = trajectory_gate(shifted)
        assert not passed

    def test_pressure_outside_unit_range_is_rejected(self):
        """Scaling by the file's own max instead of the device max lands here."""
        traj = self._normalized()
        traj[:, 2] = traj[:, 2] * 1023
        passed, details = trajectory_gate(traj)
        assert not passed
        assert "Pressure" in details["error"]

    def test_flat_unit_pressure_passes(self):
        """
        P = 1.0 everywhere is the legitimate pressure-blind device, which
        `verify` reports as degraded. It must not be mistaken for bad input.
        """
        traj = self._normalized()
        traj[:, 2] = 1.0
        assert trajectory_gate(traj)[0]

    def test_wrong_shape_is_rejected(self):
        assert not trajectory_gate(np.zeros((50, 2), dtype=np.float32))[0]
        assert not trajectory_gate(np.zeros(50, dtype=np.float32))[0]

    def test_too_short_is_rejected(self):
        assert not trajectory_gate(self._normalized(n=4))[0]

    def test_the_length_boundary_is_exact(self):
        """One point under the minimum must fail and the minimum itself must pass."""
        assert not trajectory_gate(self._normalized(n=MIN_TRAJECTORY_POINTS - 1))[0]
        assert trajectory_gate(self._normalized(n=MIN_TRAJECTORY_POINTS))[0]

    def test_non_finite_is_rejected(self):
        traj = self._normalized()
        traj[10, 0] = np.nan
        passed, details = trajectory_gate(traj)
        assert not passed
        assert "NaN" in details["error"]


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
        # Assert it failed on *size*, not sharpness. `quality_gate` ANDs two
        # conditions, so a sharpness check that always returned False would
        # satisfy a bare `passed is False` and never be noticed.
        assert min(details["bbox_width"], details["bbox_height"]) < MIN_BBOX_PIXELS

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

    def test_ratios_are_exact_not_merely_approximate(self):
        """
        The split is deterministic at a fixed seed, so a band of 60-80% would
        pass an off-by-one in the `int(n * ratio)` truncation. Pin the counts.
        """
        train, val, test = writer_independent_split([f"w{i}" for i in range(100)], seed=42)
        assert (len(train), len(val), len(test)) == (70, 15, 15)

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
                                               num_forged_per_writer=5,
                                               random_impostor_ratio=0.0)
        forgery_pairs = [p for p, y in zip(pairs, labels) if y == 0]
        assert len(forgery_pairs) > 0

        for s1, s2 in forgery_pairs:
            writer1, kind1 = s1.split("_")
            writer2, kind2 = s2.split("_")
            assert writer1 == writer2, f"cross-writer impostor leaked in: {s1} vs {s2}"
            assert kind1.startswith("g") and kind2.startswith("f")

    def test_genuine_pairs_are_same_writer_genuines(self):
        """Every label-1 pair must be two genuine samples from one writer."""
        pairs, labels = generate_skilled_pairs(self._writers(), random_impostor_ratio=0.0)
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
        pairs, labels = generate_skilled_pairs(writers, random_impostor_ratio=0.0)
        assert all("w0" not in s1 for s1, _ in pairs)

    def test_random_impostor_ratio_introduces_cross_writer_negatives(self):
        """
        At a non-zero ratio some negatives must pair two *different* writers'
        genuine signatures.

        This is the one supervision signal the skilled-only pairing never gives:
        measured on the shipped spatial model, a stranger's genuine CEDAR
        signature sat at mean distance 0.048 against 0.038 for the same writer's,
        so the encoder had learned a genuine/forged axis rather than an identity
        one.
        """
        pairs, labels = generate_skilled_pairs(self._writers(),
                                               num_genuine_per_writer=4,
                                               num_forged_per_writer=4,
                                               random_impostor_ratio=0.5)
        negatives = [p for p, y in zip(pairs, labels) if y == 0]
        cross = [(a, b) for a, b in negatives if a.split("_")[0] != b.split("_")[0]]

        assert cross, "no cross-writer negatives generated at ratio 0.5"
        for s1, s2 in cross:
            assert s1.split("_")[1].startswith("g")
            assert s2.split("_")[1].startswith("g"), (
                f"random impostor must be another writer's genuine, got {s2}")

    def test_random_impostors_come_out_of_the_negative_budget(self):
        """
        The ratio re-composes the negatives; it must not add to them.

        Class balance is the confound this guards: if random impostors were
        appended rather than substituted, an arm testing the ratio would also be
        testing a 1:2 positive-negative imbalance, and the two effects could not
        be separated.
        """
        kwargs = dict(num_genuine_per_writer=4, num_forged_per_writer=4)
        _, baseline = generate_skilled_pairs(self._writers(), **kwargs, random_impostor_ratio=0.0)
        _, mixed = generate_skilled_pairs(self._writers(),
                                          random_impostor_ratio=0.5, **kwargs)

        assert mixed.count(0) == baseline.count(0)
        assert mixed.count(1) == baseline.count(1)
        assert mixed.count(0) == mixed.count(1)

    def test_random_impostor_never_pairs_a_writer_with_themselves(self):
        """A 'random impostor' drawn from the same writer would be a positive mislabelled 0."""
        pairs, labels = generate_skilled_pairs(self._writers(n=3),
                                               num_genuine_per_writer=6,
                                               num_forged_per_writer=6,
                                               random_impostor_ratio=1.0)
        for (s1, s2), y in zip(pairs, labels):
            if y == 0:
                assert s1.split("_")[0] != s2.split("_")[0]

    def test_ratio_zero_reproduces_the_skilled_only_pairing(self):
        """
        Ratio 0 must give the pre-2026-08-04 pairing: every negative a same-writer
        forgery, no cross-writer pair anywhere.

        This used to compare against the *default*, which was 0.0. The default is
        now 0.5 — the shipped value — so the assertion has to name the behaviour
        it wants rather than inherit it.
        """
        pairs, labels = generate_skilled_pairs(self._writers(), seed=7,
                                               random_impostor_ratio=0.0)
        for (s1, s2), y in zip(pairs, labels):
            if y == 0:
                assert s1.split("_")[0] == s2.split("_")[0]
                assert s2.split("_")[1].startswith("f")

    def test_ratio_outside_zero_to_one_is_rejected(self):
        """A ratio above 1 would silently produce a negative skilled budget."""
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="random_impostor_ratio"):
                generate_skilled_pairs(self._writers(), random_impostor_ratio=bad)

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
            num_forged_per_writer=276, seed=42,
                random_impostor_ratio=0.0)
        assert labels.count(1) == labels.count(0)

    def test_balanced_datasets_are_unaffected(self):
        """CEDAR's 24/24 supports 276 of each; this must not reduce it."""
        pairs, labels = generate_skilled_pairs(
            self._dataset(24, 24), num_genuine_per_writer=276,
            num_forged_per_writer=276, seed=42,
                random_impostor_ratio=0.0)
        assert labels.count(1) == labels.count(0) == 276

    def test_small_budget_still_honoured(self):
        pairs, labels = generate_skilled_pairs(
            self._dataset(27, 6), num_genuine_per_writer=40,
            num_forged_per_writer=40, seed=42,
                random_impostor_ratio=0.0)
        assert labels.count(1) == labels.count(0) == 40


class TestSpatialInputGeometry:
    """
    Square-pad versus direct resize. Which one wins is a measurement, not a
    preference: direct resize won on every corpus, so SPATIAL_SQUARE_PAD ships
    False. Padding spends canvas on the wide BHSig260 ribbons and measured worse.
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


class TestSpatialAugmentation:
    """
    Augmentation exists to add variety the 301 training writers do not hold.
    It is only safe if every warp is label-preserving and confined to training.
    """

    @staticmethod
    def _image():
        # Ink high on a black field, matching what preprocess_image emits.
        img = np.zeros((1, 155, 220), dtype=np.float32)
        img[0, 60:95, 30:190] = 1.0
        return img

    def test_output_keeps_shape_dtype_and_range(self):
        out = augment_image(self._image(), np.random.default_rng(0))
        assert out.shape == (1, 155, 220)
        assert out.dtype == np.float32
        # Warping must not push values outside the normalized range the model
        # was trained to expect.
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_it_actually_changes_the_image(self):
        source = self._image()
        out = augment_image(source, np.random.default_rng(0))
        assert not np.array_equal(out, source)

    def test_the_source_image_is_never_modified_in_place(self):
        source = self._image()
        before = source.copy()
        augment_image(source, np.random.default_rng(0))
        assert np.array_equal(source, before)

    def test_ink_survives_the_warp(self):
        # A warp that rotated or scaled the signature off-canvas would produce a
        # blank image whose "genuine" label is a lie.
        for seed in range(25):
            out = augment_image(self._image(), np.random.default_rng(seed))
            assert out.sum() > 0.0, f"seed {seed} erased the signature"

    def test_background_stays_black(self):
        # borderValue must match the inverted polarity. A white fill would read
        # as a large constant of ink to every convolution.
        out = augment_image(self._image(), np.random.default_rng(3))
        assert out[0, 0, 0] == 0.0 and out[0, -1, -1] == 0.0

    def test_different_seeds_give_different_warps(self):
        a = augment_image(self._image(), np.random.default_rng(1))
        b = augment_image(self._image(), np.random.default_rng(2))
        assert not np.array_equal(a, b)

    def test_dataset_leaves_images_untouched_when_augment_is_off(self, tmp_path):
        pairs, labels = _one_pair(tmp_path)
        plain = SpatialSignatureDataset(pairs, labels, augment=False)
        first, second = plain[0][0], plain[0][1]
        again = SpatialSignatureDataset(pairs, labels, augment=False)[0]
        # Deterministic without augmentation, so repeat reads must agree.
        assert torch.equal(first, again[0]) and torch.equal(second, again[1])

    def test_dataset_augments_both_sides_independently(self, tmp_path):
        # Warping a pair identically would leave the distance between the two
        # embeddings unchanged, which teaches the metric nothing.
        pairs, labels = _one_pair(tmp_path, identical=True)
        item = SpatialSignatureDataset(pairs, labels, augment=True)[0]
        assert not torch.equal(item[0], item[1])


def _one_pair(tmp_path, identical=False):
    """Write one genuine pair to disk and return it in (pairs, labels) form."""
    paths = []
    for i in range(2):
        img = np.full((155, 220), 255, dtype=np.uint8)
        offset = 0 if identical else i * 4
        cv2.rectangle(img, (30 + offset, 60), (190, 95), 0, -1)
        path = str(tmp_path / f"sig_{i}.png")
        cv2.imwrite(path, img)
        paths.append(path)
    return [(paths[0], paths[1])], [1.0]


class TestPinnedHoldout:
    """
    `pin_holdout` exists so a corpus can be *added* without moving the writers a
    previous result was measured on.

    `writer_independent_split` shuffles every ID it is given, so adding 498
    DeepSignDB writers silently relocates the 31 SVC2004/SigComp writers the
    current numbers were scored against — some of them into training. That is
    both a broken comparison and writer leakage, and it is invisible: the run
    completes and reports a plausible EER.
    """

    def test_pinned_writers_all_land_in_test(self):
        ids = [f"w{i}" for i in range(100)]
        pinned = ["w3", "w41", "w77"]

        train, val, test = pin_holdout(ids, pinned, 0.5, 0.25, 0.25, seed=42)

        assert set(pinned) <= set(test)
        assert not set(pinned) & set(train)
        assert not set(pinned) & set(val)

    def test_partitions_stay_disjoint_and_complete(self):
        ids = [f"w{i}" for i in range(100)]

        train, val, test = pin_holdout(ids, ["w5", "w6"], 0.5, 0.25, 0.25, seed=42)

        assert set(train) | set(val) | set(test) == set(ids)
        assert len(train) + len(val) + len(test) == len(ids)
        for a, b in ((train, val), (train, test), (val, test)):
            assert not set(a) & set(b)

    def test_unpinned_writers_split_reproducibly(self):
        ids = [f"w{i}" for i in range(100)]

        first = pin_holdout(ids, ["w5"], 0.5, 0.25, 0.25, seed=7)
        again = pin_holdout(ids, ["w5"], 0.5, 0.25, 0.25, seed=7)

        assert first == again

    def test_adding_writers_does_not_move_the_pinned_ones(self):
        """The whole point: a bigger corpus must not disturb the held-out set."""
        small = [f"w{i}" for i in range(50)]
        big = small + [f"deep{i}" for i in range(400)]
        pinned = ["w3", "w41"]

        _, _, test_small = pin_holdout(small, pinned, 0.5, 0.25, 0.25, seed=42)
        train_big, val_big, test_big = pin_holdout(big, pinned, 0.5, 0.25, 0.25, seed=42)

        assert set(pinned) <= set(test_small) & set(test_big)
        assert not set(pinned) & (set(train_big) | set(val_big))

    def test_unknown_pinned_writer_is_an_error(self):
        """
        A typo in a writer ID would otherwise hold out nothing at all and report
        a number scored on the wrong writers.
        """
        with pytest.raises(ValueError, match="ghost"):
            pin_holdout([f"w{i}" for i in range(10)], ["w1", "ghost"],
                        0.5, 0.25, 0.25, seed=42)


class TestChannelContractFollowsConfig:
    """
    `TEMPORAL_CHANNELS` is the single knob for the capture width.

    `xyp` -> 3 (what shipped before), `dt` -> 4 (what ships now, and what
    `models/weights/temporal_lstm.pth` was trained as), `tilt` -> 5 (available
    once the device is confirmed to report azimuth and altitude). Flipping that
    one setting must move the gate with it, so a checkpoint trained under any
    setting stays reproducible without a code change.

    These assert the *relationship* to config rather than a fixed number, so
    they keep their meaning across every one of those settings.
    """

    def _at_width(self, width, length=60):
        rng = np.random.default_rng(3)
        columns = [rng.standard_normal((length, 2)),
                   rng.uniform(0.0, 1.0, size=(length, 1))]
        for _ in range(width - 3):
            columns.append(np.full((length, 1), 0.1))
        return np.hstack(columns).astype(np.float32)[:, :width]

    def test_the_configured_width_passes(self):
        passed, details = trajectory_gate(self._at_width(TEMPORAL_INPUT_FEATURES))
        assert passed, details.get("error")

    def test_a_narrower_probe_is_rejected(self):
        """A 3-wide probe cannot score a 4-wide model; the gate must catch it."""
        narrow = self._at_width(TEMPORAL_INPUT_FEATURES)[:, :TEMPORAL_INPUT_FEATURES - 1]
        passed, details = trajectory_gate(narrow)
        assert not passed
        assert str(TEMPORAL_INPUT_FEATURES) in details["error"]

    def test_a_wider_probe_is_rejected(self):
        wide = np.hstack([self._at_width(TEMPORAL_INPUT_FEATURES),
                          np.full((60, 1), 0.1, dtype=np.float32)])
        assert not trajectory_gate(wide)[0]

    def test_the_error_names_the_expected_width(self):
        _, details = trajectory_gate(np.zeros((60, 2), dtype=np.float32))
        assert f"[L, {TEMPORAL_INPUT_FEATURES}]" in details["error"]

    @pytest.mark.skipif(TEMPORAL_INPUT_FEATURES < 4,
                        reason="no dt channel under the current contract")
    def test_dt_outside_unit_range_is_rejected(self):
        """dt is a gap clipped at 100 ms and scaled, never a raw timestamp."""
        traj = self._at_width(TEMPORAL_INPUT_FEATURES)
        traj[5, 3] = 4.2
        passed, details = trajectory_gate(traj)
        assert not passed
        assert "dt" in details["error"].lower()

    def test_unnormalized_xy_is_still_rejected_at_this_width(self):
        traj = self._at_width(TEMPORAL_INPUT_FEATURES)
        traj[:, :2] += 4000.0
        passed, details = trajectory_gate(traj)
        assert not passed
        assert "z-scored" in details["error"]


class TestAugmentTrajectoryPreservesWidth:
    """
    `augment_trajectory` rebuilt a hard-coded 3-column array, silently dropping
    `dt`. At batch size 1 the resulting 6-wide packed sequence fed an 8-wide
    LSTM without raising, returning a plausible embedding computed from the
    wrong tensor.
    """

    def test_four_channel_input_keeps_four_channels(self):
        rng = np.random.default_rng(11)
        traj = np.hstack([
            rng.standard_normal((40, 2)),
            rng.uniform(0.0, 1.0, size=(40, 1)),
            np.full((40, 1), 0.1),
        ]).astype(np.float32)
        out = augment_trajectory(traj, np.random.default_rng(0))
        assert out.shape[1] == traj.shape[1]

    def test_three_channel_input_still_returns_three(self):
        rng = np.random.default_rng(12)
        traj = np.hstack([rng.standard_normal((40, 2)),
                          rng.uniform(0.0, 1.0, size=(40, 1))]).astype(np.float32)
        assert augment_trajectory(traj, np.random.default_rng(0)).shape[1] == 3

    def test_extra_channel_stays_in_range_after_resampling(self):
        rng = np.random.default_rng(13)
        traj = np.hstack([
            rng.standard_normal((40, 2)),
            rng.uniform(0.0, 1.0, size=(40, 1)),
            rng.uniform(0.0, 1.0, size=(40, 1)),
        ]).astype(np.float32)
        out = augment_trajectory(traj, np.random.default_rng(0))
        assert out[:, 3].min() >= 0.0 and out[:, 3].max() <= 1.0
