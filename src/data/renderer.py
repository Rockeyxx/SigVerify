"""
Trajectory-to-Raster Stroke Renderer.

Converts an online digitizer trajectory [L, 4] (X, Y, P, dt) into the spatial
branch's canonical input [1, 155, 220], background 0.0 and ink high, so a single
signing event on the tablet can feed both branches.

Rendering attributes:
- Coordinates are scaled onto the canvas under the same framing rule the scan
  pipeline uses, so a rendered signature and a scanned one reach the CNN with the
  same geometry (see `_canvas_scale`).
- Pressure modulates stroke width.
- A pen lift breaks stroke continuity, detected by vanishing pressure OR a jump
  in the inter-sample clock.

Every constant comes from `config.settings`. A literal `(220, 155)` here would
duplicate `SPATIAL_INPUT_SIZE` and silently disagree with it the moment either
moved — the same failure a hardcoded channel count already caused on the
temporal side.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.settings import (
    RENDER_BASE_THICKNESS,
    RENDER_MARGIN_RATIO,
    RENDER_MIN_THICKNESS,
    RENDER_PEN_UP_DT_RATIO,
    RENDER_PEN_UP_PRESSURE,
    SPATIAL_INPUT_SIZE,
    SPATIAL_SQUARE_PAD,
)

# Below this the coordinate span is a single point or a straight line along one
# axis, and dividing by it would throw the stroke off-canvas.
_DEGENERATE_SPAN = 1e-4


def _canvas_scale(span_x, span_y, avail_w, avail_h):
    """
    Per-axis scale factors, matching how `preprocess_image` frames a real scan.

    `SPATIAL_SQUARE_PAD` is the project's measured framing knob. It ships False,
    meaning a scan is stretched straight onto SPATIAL_INPUT_SIZE — aspect ratio
    is deliberately sacrificed to maximise canvas occupancy (config/settings.py,
    "Verified via empirical evaluation"). Rendering with a single uniform scale
    would preserve aspect instead, so every rendered signature would reach the
    CNN with geometry no training image ever had. Following the same knob keeps
    the two input paths consistent.

    A degenerate span (all points on one axis) falls back to the other axis's
    scale rather than dividing by ~0, which would push the stroke off-canvas.
    """
    wide = span_x > _DEGENERATE_SPAN
    tall = span_y > _DEGENERATE_SPAN

    if SPATIAL_SQUARE_PAD:
        # Preserve aspect: one scale, bounded by whichever axis runs out first.
        candidates = [avail_w / span_x for _ in (1,) if wide]
        candidates += [avail_h / span_y for _ in (1,) if tall]
        scale = min(candidates) if candidates else 1.0
        return scale, scale

    scale_x = avail_w / span_x if wide else (avail_h / span_y if tall else 1.0)
    scale_y = avail_h / span_y if tall else (avail_w / span_x if wide else 1.0)
    return scale_x, scale_y


def _pen_up_dt_threshold(dt):
    """
    The gap length that counts as a pen lift for THIS signature.

    Scaled to the trajectory's own median sampling period rather than an absolute
    number of seconds. Sampling rates differ per corpus -- SVC2004 ~100 Hz,
    SigComp2011 200 Hz, a 10 Hz capture in the test fixtures -- so a fixed cutoff
    is wrong in both directions: 0.05 s marks every single sample as a lift at
    10 Hz (drawing nothing at all) and catches no lift at 200 Hz.

    Returns None when there is no usable clock, which falls back to pressure.
    """
    if dt is None or len(dt) == 0:
        return None
    period = float(np.median(dt))
    if not np.isfinite(period) or period <= 0.0:
        return None
    return period * RENDER_PEN_UP_DT_RATIO


def _pen_is_up(pressure_a, pressure_b, dt_next, dt_threshold):
    """
    Whether the segment between two consecutive samples is a pen lift.

    Two independent signals, because the corpora disagree on which one they
    carry: SigComp2011 records pen-up travel as force = 0, while SVC2004 and
    DeepSignDB drop the samples entirely and leave only a gap in the clock.
    Testing pressure alone would draw a line straight across every lift in the
    latter.
    """
    if pressure_a <= RENDER_PEN_UP_PRESSURE or pressure_b <= RENDER_PEN_UP_PRESSURE:
        return True
    return (dt_threshold is not None and dt_next is not None
            and dt_next > dt_threshold)


def render_trajectory_to_image(trajectory, target_size=SPATIAL_INPUT_SIZE,
                               margin_ratio=RENDER_MARGIN_RATIO,
                               base_thickness=RENDER_BASE_THICKNESS,
                               min_thickness=RENDER_MIN_THICKNESS):
    """
    Render a trajectory to a [1, H, W] float32 image.

    Args:
        trajectory: [L, >=3] array of (X, Y, Pressure, [dt]). The 4th column is
            read when present; a 3-wide trajectory falls back to pressure-only
            pen-lift detection.
        target_size: (width, height); defaults to SPATIAL_INPUT_SIZE.
        margin_ratio: blank fraction of each edge.
        base_thickness: stroke width in pixels at full pressure.
        min_thickness: thinnest drawable stroke.

    Returns:
        np.ndarray [1, H, W] float32 in [0.0, 1.0], background 0.0, ink high —
        the polarity `preprocess_image` produces, so both paths agree.
    """
    width, height = target_size
    canvas = np.zeros((height, width), dtype=np.uint8)

    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or len(trajectory) < 2:
        return canvas[np.newaxis, :, :].astype(np.float32)

    x, y, p = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    dt = trajectory[:, 3] if trajectory.shape[1] > 3 else None

    min_x, min_y = float(x.min()), float(y.min())
    span_x = float(x.max()) - min_x
    span_y = float(y.max()) - min_y

    avail_w = width * (1.0 - 2.0 * margin_ratio)
    avail_h = height * (1.0 - 2.0 * margin_ratio)
    scale_x, scale_y = _canvas_scale(span_x, span_y, avail_w, avail_h)

    # Centre whatever the scaling produced, so a degenerate axis sits mid-canvas
    # rather than pinned to the top or left margin.
    offset_x = (width - span_x * scale_x) / 2.0 - min_x * scale_x
    offset_y = (height - span_y * scale_y) / 2.0 - min_y * scale_y

    px = np.clip(np.round(x * scale_x + offset_x), 0, width - 1).astype(np.int32)
    py = np.clip(np.round(y * scale_y + offset_y), 0, height - 1).astype(np.int32)
    dt_threshold = _pen_up_dt_threshold(dt)

    for i in range(len(trajectory) - 1):
        if _pen_is_up(p[i], p[i + 1],
                      float(dt[i + 1]) if dt is not None else None, dt_threshold):
            continue

        thickness = max(int(round(base_thickness * float(p[i] + p[i + 1]) / 2.0)),
                        int(min_thickness))
        cv2.line(canvas, (int(px[i]), int(py[i])), (int(px[i + 1]), int(py[i + 1])),
                 color=255, thickness=thickness, lineType=cv2.LINE_AA)

    if canvas.max() > 0:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :]
