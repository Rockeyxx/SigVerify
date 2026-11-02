"""
Trajectory-to-Raster Stroke Renderer.

Converts an online pen trajectory [L, 4] (X, Y, P, dt) into a synthetic
grayscale image [1, 155, 220] with white ink on black background.
This allows a single online signature to be verified by the spatial branch.
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

# Threshold to detect a single point or line to avoid division by zero.
_DEGENERATE_SPAN = 1e-4


def _canvas_scale(span_x, span_y, avail_w, avail_h):
    """
    Calculate scaling factors along X and Y axes to match image preprocessing.
    If SPATIAL_SQUARE_PAD is True, aspect ratio is preserved.
    Otherwise, the signature is scaled directly to fill the canvas width and height.
    """
    wide = span_x > _DEGENERATE_SPAN
    tall = span_y > _DEGENERATE_SPAN

    if SPATIAL_SQUARE_PAD:
        candidates = [avail_w / span_x for _ in (1,) if wide]
        candidates += [avail_h / span_y for _ in (1,) if tall]
        scale = min(candidates) if candidates else 1.0
        return scale, scale

    scale_x = avail_w / span_x if wide else (avail_h / span_y if tall else 1.0)
    scale_y = avail_h / span_y if tall else (avail_w / span_x if wide else 1.0)
    return scale_x, scale_y


def _pen_up_dt_threshold(dt):
    """
    Determine the time gap threshold that marks a pen lift for this trajectory.
    Calculated relative to the signature's median sampling interval.
    """
    if dt is None or len(dt) == 0:
        return None
    period = float(np.median(dt))
    if not np.isfinite(period) or period <= 0.0:
        return None
    return period * RENDER_PEN_UP_DT_RATIO


def _pen_is_up(pressure_a, pressure_b, dt_next, dt_threshold):
    """
    Check if the stroke segment between two points represents a pen lift.
    Detected by either near-zero pressure or a large time gap between points.
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
    Render a trajectory sequence into a [1, H, W] float32 image array.

    Args:
        trajectory: Array with shape [L, >=3] containing (X, Y, Pressure, [dt]).
        target_size: Output image dimensions as (width, height).
        margin_ratio: Margin fraction left empty around the edges.
        base_thickness: Base line thickness in pixels at full pressure.
        min_thickness: Minimum line thickness for visible strokes.

    Returns:
        np.ndarray with shape [1, H, W], normalized to [0.0, 1.0] with inverted polarity.
    """
    width, height = target_size
    canvas = np.zeros((height, width), dtype=np.uint8)

    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or len(trajectory) < 2:
        return canvas[np.newaxis, :, :].astype(np.float32)

    # Pen trajectories grow y upward (SVC2004, SigComp2011, and XPPen captures once capture
    # flips them); image rows grow downward, so y is negated before mapping (ENG_REPORT A4).
    x, y, p = trajectory[:, 0], -trajectory[:, 1], trajectory[:, 2]
    dt = trajectory[:, 3] if trajectory.shape[1] > 3 else None

    min_x, min_y = float(x.min()), float(y.min())
    span_x = float(x.max()) - min_x
    span_y = float(y.max()) - min_y

    avail_w = width * (1.0 - 2.0 * margin_ratio)
    avail_h = height * (1.0 - 2.0 * margin_ratio)
    scale_x, scale_y = _canvas_scale(span_x, span_y, avail_w, avail_h)

    # Center the signature within the available canvas area.
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
