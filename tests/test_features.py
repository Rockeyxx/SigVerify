"""Tests for derived temporal time functions: length preservation, finiteness, and leading `(X, Y, P)` order."""

import numpy as np
import pytest

from src.data.features import NUM_DERIVED_CHANNELS, derive_time_functions


def _signature(length=120, seed=0):
    """Looping path with varying speed representing a normalized trajectory."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    xy = np.stack([np.sin(t) + 0.3 * np.sin(3 * t), np.cos(t)], axis=1)
    xy = (xy - xy.mean(axis=0)) / np.where(xy.std(axis=0) < 1e-8, 1.0, xy.std(axis=0))
    pressure = np.clip(0.5 + 0.2 * np.sin(t) + 0.01 * rng.standard_normal(length), 0, 1)
    return np.column_stack([xy, pressure]).astype(np.float32)


class TestDeriveTimeFunctions:

    def test_preserves_sequence_length(self):
        for length in (10, 37, 120, 501):
            assert len(derive_time_functions(_signature(length))) == length

    def test_first_three_channels_are_the_originals(self):
        """`trajectory_gate` reads z-scored X/Y at indices 0 and 1."""
        sig = _signature()
        out = derive_time_functions(sig)
        assert np.allclose(out[:, :3], sig, atol=1e-5)

    def test_widens_the_channel_set(self):
        out = derive_time_functions(_signature())
        # Pin the exact count, not just "more than the input". The module
        # exports NUM_DERIVED_CHANNELS so callers can size a model without
        # running it; emitting 7 or 30 would break that and still satisfy > 6.
        assert out.shape[1] == NUM_DERIVED_CHANNELS, "must add more than the existing (X,Y,P)+deltas"

    def test_output_is_finite_for_a_normal_signature(self):
        assert np.all(np.isfinite(derive_time_functions(_signature())))

    def test_a_stationary_pen_does_not_produce_nan(self):
        """
        Zero velocity divides by zero in curvature and logs it. A held pen, a
        dot, and a duplicated sample all occur in real captures.
        """
        held = np.zeros((40, 3), dtype=np.float32)
        held[:, 2] = 0.5
        assert np.all(np.isfinite(derive_time_functions(held)))

    def test_a_repeated_sample_mid_stroke_does_not_produce_nan(self):
        sig = _signature()
        sig[50] = sig[49]
        assert np.all(np.isfinite(derive_time_functions(sig)))

    def test_short_sequences_survive(self):
        """MIN_TRAJECTORY_POINTS is 10, and a filter needs room for its padlen."""
        for length in (1, 2, 3, 10):
            out = derive_time_functions(np.zeros((length, 3), dtype=np.float32))
            assert len(out) == length
            assert np.all(np.isfinite(out))

    def test_is_deterministic(self):
        sig = _signature()
        assert np.array_equal(derive_time_functions(sig), derive_time_functions(sig))

    def test_translation_leaves_the_derived_channels_alone(self):
        """Constant `(X, Y)` translations leave derived kinematic channels unchanged within float32 tolerance."""
        sig = _signature()
        shifted = sig.copy()
        shifted[:, :2] += 3.0
        assert np.allclose(derive_time_functions(sig)[:, 3:],
                           derive_time_functions(shifted)[:, 3:],
                           rtol=1e-3, atol=1e-4)

    def test_dtype_is_float32(self):
        """The LSTM path builds tensors straight from these arrays."""
        assert derive_time_functions(_signature()).dtype == np.float32

    def test_rejects_the_wrong_channel_count(self):
        with pytest.raises(ValueError, match="3 channels"):
            derive_time_functions(np.zeros((20, 2), dtype=np.float32))
