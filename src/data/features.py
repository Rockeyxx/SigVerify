"""16 derived kinematic time functions for online signatures."""

import numpy as np
from scipy.signal import butter, filtfilt

from config.settings import TEMPORAL_SAMPLE_RATE_HZ

_FILTER_ORDER = 3
_CUTOFF_HZ = 15.0
_SAMPLE_RATE_HZ = TEMPORAL_SAMPLE_RATE_HZ
_EPSILON = 1e-6
SPEED_WINDOW = 5
NUM_DERIVED_CHANNELS = 16


def _smooth(values):
    """Zero-phase low-pass filter along one axis; returns short arrays unchanged."""
    padlen = 3 * _FILTER_ORDER
    if len(values) <= padlen + 1:
        return values.astype(np.float64)

    coefficients = butter(_FILTER_ORDER, _CUTOFF_HZ / (0.5 * _SAMPLE_RATE_HZ), btype="low")
    return filtfilt(*coefficients, values.astype(np.float64))


def _derivative(values):
    out = np.zeros_like(values)
    out[1:] = values[1:] - values[:-1]
    return out


def _speed_ratio(speed, window=SPEED_WINDOW):
    """Min-to-max speed ratio in a sliding window."""
    half = window // 2
    padded = np.pad(speed, half, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)
    return frames.min(axis=1) / np.maximum(frames.max(axis=1), _EPSILON)


def derive_time_functions(trajectory):
    """Expand a normalized [L, 3] (X, Y, P) array into a [L, 16] kinematic matrix."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(
            f"expected 3 channels (X, Y, P), got shape {trajectory.shape}. "
            "Derived features are computed from the capture contract only."
        )

    x = _smooth(trajectory[:, 0])
    y = _smooth(trajectory[:, 1])

    dx, dy = _derivative(x), _derivative(y)
    speed = np.sqrt(dx * dx + dy * dy)

    # Compute tangent unit vector components.
    safe_speed = np.maximum(speed, _EPSILON)
    sin_theta, cos_theta = dy / safe_speed, dx / safe_speed

    # Compute angular velocity between consecutive steps.
    cross = np.zeros_like(dx)
    dot = np.zeros_like(dx)
    cross[1:] = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]
    dot[1:] = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
    d_theta = np.arctan2(cross, dot)

    ddx, ddy = _derivative(dx), _derivative(dy)
    tangential = _derivative(speed)
    normal = speed * d_theta

    # Logarithm of curvature radius (speed divided by turn rate).
    log_curvature_radius = np.log(safe_speed / np.maximum(np.abs(d_theta), _EPSILON))

    return np.column_stack([
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        dx, dy, speed,
        sin_theta, cos_theta, d_theta, log_curvature_radius,
        ddx, ddy,
        tangential, normal, np.sqrt(tangential ** 2 + normal ** 2),
        _speed_ratio(speed),
    ]).astype(np.float32)
