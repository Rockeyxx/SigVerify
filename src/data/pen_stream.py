"""Raw pen samples to the model's trajectory, shared by live capture and in-house parsers.

The XPPen Deco 640 (ENG_REPORT G4), like Qt on Windows, gives no device clock, reports only
when a value changes, and grows y downward. The pen corpora the model trained on grow y upward
on an exact 10 ms grid. AGENTS.md, "Device timing and pressure", sets the rules; spec decision
D8 adapts the pen-down rule to devices that report only on change.
"""
from __future__ import annotations

import numpy as np

from config.settings import TEMPORAL_INPUT_FEATURES, TEMPORAL_SAMPLE_PERIOD_MS
from src.data.parsers import MAX_GAP_MS, normalize_trajectory

_WRAP_MS = 2.0 ** 32   # Qt timestamps are unsigned 32-bit milliseconds


def repair_clock(t_ms, period_ms):
    """A monotonic float64 clock from host-stamped samples.

    A 32-bit millisecond wrap is unwrapped, any other backward step becomes one device period,
    and samples that share one timestamp (Windows receipt-time bursts) are spread evenly up to
    the next timestamp, at most one period apart. Forward gaps are kept: on devices that report
    only on change they are real time, a still pen or a pen lift.
    """
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
    """Median interval between consecutive samples inside strokes, in ms (one grid step if none)."""
    gaps = [np.diff(np.asarray(s, dtype=np.float64)[:, 0]) for s in strokes if len(s) > 1]
    gaps = np.concatenate(gaps) if gaps else np.array([])
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else TEMPORAL_SAMPLE_PERIOD_MS


def _hold_or_interp(grid, t, v, hold_after):
    """Value at each grid time: linear between neighbouring samples, but the last sample is held
    through a gap longer than hold_after, because the pen was still and the device sent nothing."""
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    j = np.minimum(i + 1, len(t) - 1)
    span = t[j] - t[i]
    frac = np.where(span > 0, (grid - t[i]) / np.where(span > 0, span, 1.0), 0.0)
    frac = np.where(span > hold_after, 0.0, np.clip(frac, 0.0, 1.0))
    return v[i] + frac * (v[j] - v[i])


def resample_strokes(strokes, pressure_max, flip_y=True, period_ms=TEMPORAL_SAMPLE_PERIOD_MS):
    """Pen-down strokes to grid samples, before normalisation.

    Returns (t, x, y, p) for the pen-down points of one exact period_ms grid anchored at the
    first sample: y negated when flip_y (screen y grows down, the pen corpora grow up), p divided
    by pressure_max and clipped to [0, 1]. A stroke shorter than one grid step keeps one point at
    its first sample, so dots survive.
    """
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
    """Model input [L, TEMPORAL_INPUT_FEATURES] from pen-down strokes.

    resample_strokes, then dt = time since the previous kept point (a lift's duration on the first
    point of a stroke), clipped at MAX_GAP_MS and scaled to [0, 1], then normalize_trajectory.
    """
    if TEMPORAL_INPUT_FEATURES > 4:
        raise NotImplementedError("tilt channels are recorded but not wired into capture (spec D3)")
    t, x, y, p = resample_strokes(strokes, pressure_max, flip_y)
    extra = None
    if TEMPORAL_INPUT_FEATURES == 4:
        dt = np.zeros(len(t))
        dt[1:] = np.diff(t)
        extra = (np.clip(dt, 0.0, MAX_GAP_MS) / MAX_GAP_MS)[:, None].astype(np.float32)
    return normalize_trajectory(np.column_stack([x, y]), p, extra)
