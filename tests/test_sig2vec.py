"""
Tests for Sig2Vec and the average-precision ranking loss — phase 2 of SynSig2Vec.

Clean-room from the paper (arXiv:1911.05358, Tables 1-2). The reference
implementation is GPL-3.0 and this repository is MIT, so no code was taken.

Two properties carry the whole method:

1. **Padding must be invisible.** Signatures are ragged, batches are padded, and
   masked average pooling is the only thing standing between a sample and an
   embedding that depends on which batch it landed in. The temporal LSTM has the
   same invariant pinned in `tests/test_models.py`; this is its Sig2Vec twin.
2. **The AP loss must actually reward the right ranking.** It is trained to rank
   the low-distortion pool above the median-distortion one. If the loss does not
   fall when that ordering improves, the network learns nothing from the
   synthesis and the whole pipeline is decorative.
"""

import numpy as np
import pytest
import torch

from tools.sig2vec import (FEATURES, Sig2Vec, average_precision_loss,
                           time_functions, velocity_features)


def _stroke(length, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    xy = np.stack([np.sin(t), np.cos(1.7 * t)], axis=1)
    return (xy + 0.01 * rng.standard_normal((length, 2))).astype(np.float32)


class TestVelocityFeatures:
    """The paper feeds three channels — vx, vy, |v| — and no pressure at all."""

    def test_emits_three_channels(self):
        assert velocity_features(_stroke(50)).shape == (50, 3)

    def test_preserves_length(self):
        for n in (12, 51, 200):
            assert len(velocity_features(_stroke(n))) == n

    def test_channels_are_standardized(self):
        f = velocity_features(_stroke(300))
        assert np.allclose(f.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(f.std(axis=0), 1.0, atol=1e-3)

    def test_is_translation_invariant(self):
        """Velocity cannot see where on the tablet the signature was written."""
        sig = _stroke(80)
        assert np.allclose(velocity_features(sig),
                           velocity_features(sig + 5.0), atol=1e-4)

    def test_is_finite_on_a_stationary_pen(self):
        assert np.all(np.isfinite(velocity_features(np.zeros((40, 2), dtype=np.float32))))


class TestTimeFunctions:
    """
    The 12-function set the authors' later work (DsDTW) uses.

    The three-channel set above is what the AAAI paper specifies; this is the
    wider input, and the one that matters for a project whose temporal branch is
    already known to be channel-starved.
    """

    def test_emits_twelve_channels(self):
        assert time_functions(_stroke(50)).shape == (50, 12)

    def test_preserves_length(self):
        for n in (12, 51, 200):
            assert len(time_functions(_stroke(n))) == n

    def test_channels_are_standardized(self):
        """All twelve, pressure included — so it needs a trajectory that has one."""
        sig = _stroke(300)
        pressure = np.linspace(0.2, 0.8, 300, dtype=np.float32)[:, None]
        f = time_functions(np.hstack([sig, pressure]))
        assert np.allclose(f.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(f.std(axis=0), 1.0, atol=1e-3)

    def test_is_translation_invariant(self):
        """Every channel is built from derivatives, so absolute position cannot leak."""
        sig = _stroke(80)
        assert np.allclose(time_functions(sig), time_functions(sig + 5.0), atol=1e-4)

    def test_reads_pressure_when_present(self):
        """Channel 6 is pressure — it must track column 2, not be invented."""
        sig = _stroke(120)
        pressure = np.linspace(0.1, 0.9, 120, dtype=np.float32)[:, None]
        with_pressure = time_functions(np.hstack([sig, pressure]))
        # Standardised, so compare the shape of the ramp rather than its scale.
        assert np.corrcoef(with_pressure[:, 6], pressure[:, 0])[0, 1] > 0.99

    def test_pressure_channel_is_flat_when_absent(self):
        """A synthetic trajectory has no pressure; it must not become noise."""
        assert np.allclose(time_functions(_stroke(90))[:, 6], 0.0)

    def test_is_finite_on_a_stationary_pen(self):
        """`rho` divides by angular velocity, which is zero for a pen that never moved."""
        flat = time_functions(np.zeros((40, 2), dtype=np.float32))
        assert np.all(np.isfinite(flat))

    def test_is_finite_on_a_straight_line(self):
        """A straight stroke has zero curvature — the log curvature radius must survive it."""
        line = np.stack([np.linspace(0, 10, 60), np.zeros(60)], axis=1).astype(np.float32)
        assert np.all(np.isfinite(time_functions(line)))

    def test_short_input_returns_zeros(self):
        assert time_functions(_stroke(2)).shape == (2, 12)

    def test_registry_channel_counts_match_output(self):
        """A wrong count here silently builds a CNN with the wrong input width."""
        for name, (fn, channels) in FEATURES.items():
            assert fn(_stroke(40)).shape[1] == channels, name


class TestSig2Vec:

    def test_embedding_width_matches_the_paper(self):
        model = Sig2Vec().eval()
        out = model(torch.randn(2, 3, 128))
        assert out.shape == (2, 512)

    def test_accepts_variable_lengths(self):
        model = Sig2Vec().eval()
        for n in (32, 97, 256):
            assert model(torch.randn(1, 3, n)).shape == (1, 512)

    def test_padded_batch_matches_a_single_sample(self):
        """
        The invariant. Masked pooling must ignore padded frames, or a sample's
        embedding depends on the longest sequence it was batched with.
        """
        torch.manual_seed(0)
        model = Sig2Vec().eval()
        short = torch.randn(1, 3, 40)
        padded = torch.zeros(2, 3, 96)
        padded[0, :, :40] = short[0]
        padded[1] = torch.randn(3, 96)
        lengths = torch.tensor([40, 96])
        with torch.no_grad():
            batched = model(padded, lengths)
            alone = model(short, torch.tensor([40]))
        assert torch.allclose(batched[0], alone[0], atol=1e-5)

    def test_uses_no_batch_normalization(self):
        """
        The paper is explicit: each batch is one signature and its synthetics,
        which are not i.i.d., so BatchNorm would normalize over correlated
        samples. SELU provides the self-normalization instead.
        """
        assert not any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                       for m in Sig2Vec().modules())


class TestAveragePrecisionLoss:
    """G1 (low distortion) must rank above G2 (median distortion)."""

    def _similarities(self, g1_scores, g2_scores):
        return (torch.tensor(g1_scores, dtype=torch.float32),
                torch.tensor(g2_scores, dtype=torch.float32))

    def test_perfect_ranking_gives_near_zero_loss(self):
        g1, g2 = self._similarities([0.9, 0.85, 0.8], [0.2, 0.1, 0.05])
        assert float(average_precision_loss(g1, g2)) < 0.05

    def test_inverted_ranking_is_penalized(self):
        good = average_precision_loss(*self._similarities([0.9, 0.8], [0.1, 0.05]))
        bad = average_precision_loss(*self._similarities([0.1, 0.05], [0.9, 0.8]))
        assert float(bad) > float(good)

    def test_loss_decreases_as_the_ranking_improves(self):
        """The gradient signal the whole synthesis exists to provide."""
        worse = average_precision_loss(*self._similarities([0.5, 0.4], [0.6, 0.3]))
        better = average_precision_loss(*self._similarities([0.8, 0.7], [0.2, 0.1]))
        assert float(better) < float(worse)

    def test_is_bounded_in_zero_one(self):
        for g1, g2 in ([[0.9], [0.1]], [[0.1], [0.9]], [[0.5, 0.5], [0.5, 0.5]]):
            v = float(average_precision_loss(*self._similarities(g1, g2)))
            assert 0.0 <= v <= 1.0

    def test_is_differentiable(self):
        """A non-differentiable AP would silently train nothing."""
        g1 = torch.tensor([0.4, 0.3], requires_grad=True)
        g2 = torch.tensor([0.6, 0.2], requires_grad=True)
        average_precision_loss(g1, g2).backward()
        assert g1.grad is not None and torch.any(g1.grad != 0)

    def test_rejects_an_empty_pool(self):
        with pytest.raises(ValueError, match="empty"):
            average_precision_loss(torch.tensor([]), torch.tensor([0.5]))
