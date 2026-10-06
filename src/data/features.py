"""
Derived Kinematic Time Functions for Online Signatures.

Derives 16 time functions from raw (X, Y, Pressure) coordinates:
  - Velocity components and speed
  - Tangent direction (sine and cosine)
  - Angular velocity and log curvature radius
  - Second derivatives and acceleration components
  - Local speed consistency ratio

Coordinates are smoothed with a low-pass filter before computing derivatives
to reduce sensor noise.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from config.settings import TEMPORAL_SAMPLE_RATE_HZ

# Low-pass filter settings to smooth hand movement trajectories.
_FILTER_ORDER = 3
_CUTOFF_HZ = 15.0
_SAMPLE_RATE_HZ = TEMPORAL_SAMPLE_RATE_HZ

# Small constant to prevent division by zero when pen is stationary.
_EPSILON = 1e-6

SPEED_WINDOW = 5

# Total number of output channels produced by this feature extractor.
NUM_DERIVED_CHANNELS = 16


def _smooth(values):
    """
    Apply zero-phase low-pass filtering along one coordinate axis.
    Short trajectories below filter padding length are returned unchanged.
    """
    padlen = 3 * _FILTER_ORDER
    if len(values) <= padlen + 1:
        return values.astype(np.float64)

    coefficients = butter(_FILTER_ORDER, _CUTOFF_HZ / (0.5 * _SAMPLE_RATE_HZ), btype="low")
    return filtfilt(*coefficients, values.astype(np.float64))


def _derivative(values):
    """Calculate first difference across consecutive points."""
    out = np.zeros_like(values)
    out[1:] = values[1:] - values[:-1]
    return out


def _speed_ratio(speed, window=SPEED_WINDOW):
    """
    Calculate min-to-max speed ratio within a sliding window.
    Measures stroke smoothness and speed consistency over time.
    """
    half = window // 2
    padded = np.pad(speed, half, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)
    return frames.min(axis=1) / np.maximum(frames.max(axis=1), _EPSILON)


def derive_time_functions(trajectory):
    """
    Expand a normalized [L, 3] trajectory into a [L, 16] kinematic feature matrix.

    Args:
        trajectory: Array with shape [L, 3] containing (X, Y, Pressure).

    Returns:
        Array with shape [L, 16] float32:
          - Channels 0-2:   Original X, Y, Pressure
          - Channels 3-4:   Velocity components (dX, dY)
          - Channel 5:      Total speed magnitude
          - Channels 6-7:   Tangent angle direction (sin, cos)
          - Channel 8:      Angular velocity
          - Channel 9:      Log curvature radius
          - Channels 10-11: Second derivatives (ddX, ddY)
          - Channel 12:     Tangential acceleration
          - Channel 13:     Normal acceleration
          - Channel 14:     Total acceleration magnitude
          - Channel 15:     Sliding window speed ratio
    """
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
