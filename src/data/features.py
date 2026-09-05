"""
Derived time functions for the temporal branch.

The capture contract is fixed at `(X, Y, Pressure)` and widening it is banned
under Drift Prevention. Deriving further features from those three channels is
explicitly allowed, and it is the single largest gap between this branch and
the systems it is measured against: TA-RNN (Tolosana et al., DeepSignDB) feeds
23 time functions, DsDTW (ICDAR 2021 winner) feeds 12, and this branch feeds 6
— `(X, Y, P)` plus first-order deltas.

The families added here are the ones both references share: path-tangent
angle with its sine and cosine, velocity magnitude, log curvature radius, total
acceleration split into tangential and normal components, angular velocity,
second derivatives, and a sliding-window speed ratio.

**Smoothing comes first, and it is not optional.** A finite difference
amplifies high-frequency noise, and curvature is a second-order quantity — a
second difference of unsmoothed tablet coordinates is close to pure noise.
DsDTW low-passes x and y before computing anything; so does this.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from config.settings import TEMPORAL_SAMPLE_RATE_HZ

# Butterworth low-pass matching DsDTW's preprocessing. Signature sampling rates
# in this corpus sit around 100 Hz, and 15 Hz is well above the bandwidth of
# human handwriting (a fast stroke reversal is a few Hz) while cutting the
# sensor chatter that differentiation would otherwise square.
_FILTER_ORDER = 3
_CUTOFF_HZ = 15.0
_SAMPLE_RATE_HZ = TEMPORAL_SAMPLE_RATE_HZ

# Guard for the ratios below. Velocity reaches exactly zero whenever the pen is
# held still, which real captures are full of — dots, pauses, and duplicated
# samples all produce it — so every division is floored before dividing rather
# than cleaned up afterwards, which would already have produced inf/nan.
_EPSILON = 1e-6

SPEED_WINDOW = 5

# Channel count this module emits, so callers can size a model without running
# it. `TemporalSiamese` doubles it internally when it concatenates deltas.
NUM_DERIVED_CHANNELS = 16


def _smooth(values):
    """
    Zero-phase low-pass along one axis, falling back when the sequence is short.

    `filtfilt` needs more samples than its padding length, and signatures
    shorter than that exist — `MIN_TRAJECTORY_POINTS` is 10. A short sequence is
    returned unfiltered rather than raising, because dropping it would silently
    shrink the corpus.
    """
    padlen = 3 * _FILTER_ORDER
    if len(values) <= padlen + 1:
        return values.astype(np.float64)

    coefficients = butter(_FILTER_ORDER, _CUTOFF_HZ / (0.5 * _SAMPLE_RATE_HZ), btype="low")
    return filtfilt(*coefficients, values.astype(np.float64))


def _derivative(values):
    """
    Per-frame first difference, length preserved.

    The first frame has no predecessor and takes zero, matching the convention
    `TemporalSiamese._add_dynamics` already uses for its deltas.
    """
    out = np.zeros_like(values)
    out[1:] = values[1:] - values[:-1]
    return out


def _speed_ratio(speed, window=SPEED_WINDOW):
    """
    Min-over-max speed within a sliding window — TA-RNN's time function 17.

    Reads as local speed consistency: a practised signature accelerates
    smoothly through a stroke, while a traced forgery starts and stops. The
    window is centred and edge-padded so the output keeps its length.
    """
    half = window // 2
    padded = np.pad(speed, half, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)
    return frames.min(axis=1) / np.maximum(frames.max(axis=1), _EPSILON)


def derive_time_functions(trajectory):
    """
    Expand a normalized `[L, 3]` trajectory into `[L, 16]` time functions.

    Input must already be through `parsers.normalize_trajectory` — X and Y
    z-scored per signature, P in [0, 1]. Output preserves length exactly, so it
    drops into the existing padding and packing path unchanged, and keeps
    `(X, Y, P)` at indices 0-2 so `dataset.trajectory_gate` still reads the
    channels it bounds-checks.

    Channels:
        0-2   X, Y, P                      (unchanged)
        3-4   dX, dY                       velocity components
        5     v                            path velocity magnitude
        6-7   sin(theta), cos(theta)       path-tangent direction
        8     d(theta)                     angular velocity
        9     log curvature radius
        10-11 ddX, ddY                     second derivatives
        12    dv                           tangential acceleration
        13    v * d(theta)                 normal acceleration
        14    total acceleration magnitude
        15    min/max speed ratio over a 5-sample window

    Every channel is finite by construction: the ratios are floored at
    `_EPSILON` before dividing rather than cleaned up afterwards, because a
    stationary pen makes the true value undefined, not merely large.
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

    # sin/cos of the tangent angle taken as normalized components rather than
    # through atan2: same quantity, no branch cut, and no unwrapping needed.
    safe_speed = np.maximum(speed, _EPSILON)
    sin_theta, cos_theta = dy / safe_speed, dx / safe_speed

    # Angular velocity as the signed turn between consecutive tangents. The
    # cross/dot form stays continuous across the +/-pi wrap that differencing
    # raw atan2 angles would tear.
    cross = np.zeros_like(dx)
    dot = np.zeros_like(dx)
    cross[1:] = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]
    dot[1:] = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
    d_theta = np.arctan2(cross, dot)

    ddx, ddy = _derivative(dx), _derivative(dy)
    tangential = _derivative(speed)
    normal = speed * d_theta

    # Curvature radius is speed / |turn rate|; logged because it spans orders
    # of magnitude between a straight run and a tight loop.
    log_curvature_radius = np.log(safe_speed / np.maximum(np.abs(d_theta), _EPSILON))

    return np.column_stack([
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        dx, dy, speed,
        sin_theta, cos_theta, d_theta, log_curvature_radius,
        ddx, ddy,
        tangential, normal, np.sqrt(tangential ** 2 + normal ** 2),
        _speed_ratio(speed),
    ]).astype(np.float32)
