"""Convert raw tablet stroke samples into normalized [L, C] trajectories."""
from __future__ import annotations

import numpy as np

from config.settings import TEMPORAL_INPUT_FEATURES, TEMPORAL_SAMPLE_PERIOD_MS
from src.data.parsers import MAX_GAP_MS, normalize_trajectory

_WRAP_MS = 2.0 ** 32   # Qt timestamps are unsigned 32-bit milliseconds


def repair_clock(t_ms, period_ms):
    """Build a monotonic float64 clock, unwrapping 32-bit wrap and spreading burst timestamps."""
    t = np.asarray(t_ms, dtype=np.float64)
    if len(t) < 2:
        return t.copy()
    d = np.diff(t)
    d = np.where(d < -_WRAP_MS / 2, d + _WRAP_MS, d)
    d = np.where(d < 0, period_ms, d)
    t = t[0] + np.r_[0.0, np.cumsum(d)]
    u, start = np.unique(t, return_index=True)
    size = np.diff(np.r_[start, len(t)])
    run = np.repeat(np.arange(len(u)), size)
    nxt = np.minimum(np.r_[u[1:], u[-1] + period_ms * size[-1]], u + period_ms * size)
    frac = (np.arange(len(t)) - start[run]) / size[run]
    return u[run] + frac * (nxt[run] - u[run])


def device_period(strokes):
    """Return median positive inter-sample interval in ms."""
    gaps = [np.diff(np.asarray(s, dtype=np.float64)[:, 0]) for s in strokes if len(s) > 1]
    gaps = np.concatenate(gaps) if gaps else np.array([])
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else TEMPORAL_SAMPLE_PERIOD_MS


def _hold_or_interp(grid, t, v, hold_after):
    """Interpolate linearly between samples, holding the last value across gaps > hold_after."""
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    j = np.minimum(i + 1, len(t) - 1)
    span = t[j] - t[i]
    frac = np.where(span > 0, (grid - t[i]) / np.where(span > 0, span, 1.0), 0.0)
    frac = np.where(span > hold_after, 0.0, np.clip(frac, 0.0, 1.0))
    return v[i] + frac * (v[j] - v[i])


def resample_strokes(strokes, pressure_max, flip_y=True, period_ms=TEMPORAL_SAMPLE_PERIOD_MS):
    """Resample pen-down strokes onto a uniform period_ms grid and scale pressure to [0, 1]."""
    kept = [np.asarray(s, dtype=np.float64) for s in strokes if len(s)]
    if not kept:
        raise ValueError("no pen-down samples")
    dev = device_period(kept)
    t_all = repair_clock(np.concatenate([s[:, 0] for s in kept]), dev)
    bounds = np.cumsum([0] + [len(s) for s in kept])
    t0 = t_all[0]
    out = []
    for k, s in enumerate(kept):
        t = t_all[bounds[k]:bounds[k + 1]]
        first = t0 + np.ceil((t[0] - t0) / period_ms - 1e-9) * period_ms
        grid = np.arange(first, t[-1] + 1e-9, period_ms)
        if len(grid) == 0:
            grid = t[:1]
        cols = [_hold_or_interp(grid, t, s[:, c], 2 * dev) for c in (1, 2, 3)]
        out.append(np.column_stack([grid] + cols))
    g = np.vstack(out)
    y = -g[:, 2] if flip_y else g[:, 2]
    p = np.clip(g[:, 3] / pressure_max, 0.0, 1.0)
    return g[:, 0], g[:, 1], y, p


def strokes_to_trajectory(strokes, pressure_max, flip_y=True):
    """Resample strokes, compute clipped dt, and normalize into [L, TEMPORAL_INPUT_FEATURES]."""
    if TEMPORAL_INPUT_FEATURES > 4:
        raise NotImplementedError("tilt channels are recorded but not wired into capture (spec D3)")
    t, x, y, p = resample_strokes(strokes, pressure_max, flip_y)
    extra = None
    if TEMPORAL_INPUT_FEATURES == 4:
        dt = np.zeros(len(t))
        dt[1:] = np.diff(t)
        extra = (np.clip(dt, 0.0, MAX_GAP_MS) / MAX_GAP_MS)[:, None].astype(np.float32)
    return normalize_trajectory(np.column_stack([x, y]), p, extra)
