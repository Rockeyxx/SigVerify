"""
Tests for the Sigma-Lognormal (SigmaLambda) signature model.

This is phase 1 of a clean-room SynSig2Vec (Lai et al., AAAI 2020). The method
synthesises training data from genuine signatures alone, which is why it is
worth having here: it needs no skilled forgeries and no licensed corpus.

**The reference implementation is GPL-3.0 and this repository is MIT, so no code
was taken from it.** Everything below is written from the paper's equations.

The whole method rests on one property: signatures synthesised at a *low*
distortion level must sit closer to their template than ones at a *median*
level, because the AP loss learns to rank exactly that ordering. If the
decomposition is poor the synthetics are noise, the ordering is random, and
every downstream number is meaningless — silently. So reconstruction fidelity
and distortion ordering are the two gates, and they are tested rather than
assumed.
"""

import numpy as np
import pytest

from tools.sigma_lognormal import (LognormalStroke, decompose, distort,
                                   reconstruct, stroke_velocity, synthesize)


def _spiral(length=90, turns=1.5):
    """
    A looping path with a *modulated* speed profile.

    Constant-speed curves are the wrong fixture here: the Kinematic Theory
    segments handwriting at velocity peaks, and a curve traced at uniform speed
    has exactly one. Real signatures accelerate and decelerate several times per
    stroke, so the phase is warped to produce that oscillation — otherwise the
    decomposition is being asked to fit something no hand ever produced.
    """
    u = np.linspace(0, 1, length)
    phase = u + 0.06 * np.sin(2 * np.pi * 6 * u)      # speed-up / slow-down
    t = phase * turns * 2 * np.pi
    r = 0.25 + 0.75 * u + 0.10 * np.sin(2 * np.pi * 3 * u)
    xy = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    return ((xy - xy.mean(0)) / xy.std(0)).astype(np.float32)


class TestStrokeVelocity:
    """Equation 1: a single lognormal impulse response."""

    def test_is_zero_before_onset(self):
        t = np.linspace(0.0, 1.0, 200)
        v = stroke_velocity(t, LognormalStroke(D=1.0, t0=0.5, mu=-1.0, sigma=0.3))
        assert np.all(v[t <= 0.5] == 0.0)

    def test_is_positive_and_finite_after_onset(self):
        t = np.linspace(0.0, 1.0, 200)
        v = stroke_velocity(t, LognormalStroke(D=1.0, t0=0.0, mu=-1.5, sigma=0.3))
        assert np.all(np.isfinite(v)) and v.max() > 0

    def test_area_under_the_curve_equals_D(self):
        """D is the stroke amplitude: the integral of its velocity profile."""
        t = np.linspace(0.0, 4.0, 40000)
        v = stroke_velocity(t, LognormalStroke(D=2.5, t0=0.0, mu=-1.0, sigma=0.4))
        assert np.trapezoid(v, t) == pytest.approx(2.5, rel=0.02)

    def test_larger_sigma_spreads_the_profile(self):
        t = np.linspace(0.0, 4.0, 4000)
        narrow = stroke_velocity(t, LognormalStroke(1.0, 0.0, -1.0, 0.2))
        wide = stroke_velocity(t, LognormalStroke(1.0, 0.0, -1.0, 0.6))
        assert wide.max() < narrow.max()


class TestDecomposeReconstruct:
    """The gate: a signature must survive a round trip through its parameters."""

    def test_reconstruction_preserves_length(self):
        sig = _spiral()
        assert len(reconstruct(decompose(sig))) == len(sig)

    def test_reconstruction_is_faithful_to_a_real_shape(self):
        """
        The paper's Figure 2 shows the recovered trajectory overlaying the
        original almost exactly. Above ~0.25 normalized RMSE the synthetics stop
        resembling the template and the method measures noise.
        """
        sig = _spiral()
        rebuilt = reconstruct(decompose(sig))
        rmse = np.sqrt(np.mean((rebuilt[:, :2] - sig[:, :2]) ** 2))
        assert rmse < 0.25, f"reconstruction RMSE {rmse:.3f} too high to synthesise from"

    def test_finds_more_than_one_stroke_on_a_complex_path(self):
        """A 2.5-turn spiral is many velocity peaks, not one."""
        assert len(decompose(_spiral()).strokes) >= 3

    def test_handles_a_degenerate_input_without_crashing(self):
        flat = np.zeros((40, 3), dtype=np.float32)
        out = reconstruct(decompose(flat))
        assert len(out) == 40 and np.all(np.isfinite(out))

    def test_is_deterministic(self):
        sig = _spiral()
        a, b = reconstruct(decompose(sig)), reconstruct(decompose(sig))
        assert np.allclose(a, b)


class TestDistortionOrdering:
    """
    The property the AP loss depends on. G1 (low distortion) must land closer to
    the template than G2 (median distortion). If this ordering does not hold,
    the ranking objective is being trained on noise.
    """

    def _deviation(self, sig, level, seed):
        rng = np.random.default_rng(seed)
        params = decompose(sig)
        rebuilt = reconstruct(params)
        out = reconstruct(distort(params, level=level, rng=rng))
        n = min(len(out), len(rebuilt))
        return float(np.sqrt(np.mean((out[:n, :2] - rebuilt[:n, :2]) ** 2)))

    def test_low_distortion_stays_closer_than_median_distortion(self):
        sig = _spiral()
        g1 = np.median([self._deviation(sig, 1, s) for s in range(12)])
        g2 = np.median([self._deviation(sig, 2, s) for s in range(12)])
        assert g1 < g2, f"G1 ({g1:.4f}) must rank closer than G2 ({g2:.4f})"

    def test_distortion_actually_changes_the_signature(self):
        """A synthesis that returns the template teaches the ranker nothing."""
        assert self._deviation(_spiral(), 1, seed=3) > 1e-4

    def test_rejects_an_unknown_distortion_level(self):
        with pytest.raises(ValueError, match="level"):
            distort(decompose(_spiral()), level=7, rng=np.random.default_rng(0))


class TestSynthesize:
    """The public entry point: template in, ranked synthetic pools out."""

    def test_returns_the_requested_pool_sizes(self):
        g1, g2 = synthesize(_spiral(), n_low=5, n_median=10, seed=0)
        assert len(g1) == 5 and len(g2) == 10

    def test_every_sample_is_finite_and_two_dimensional(self):
        g1, g2 = synthesize(_spiral(), n_low=3, n_median=3, seed=1)
        for s in list(g1) + list(g2):
            assert s.ndim == 2 and s.shape[1] >= 2 and np.all(np.isfinite(s))

    def test_samples_differ_from_one_another(self):
        """Identical synthetics would collapse the ranking to a constant."""
        g1, _ = synthesize(_spiral(), n_low=4, n_median=1, seed=2)
        n = min(len(g1[0]), len(g1[1]))
        assert not np.allclose(g1[0][:n], g1[1][:n])

    def test_is_reproducible_for_a_fixed_seed(self):
        a, _ = synthesize(_spiral(), n_low=2, n_median=2, seed=5)
        b, _ = synthesize(_spiral(), n_low=2, n_median=2, seed=5)
        assert np.allclose(a[0], b[0])
