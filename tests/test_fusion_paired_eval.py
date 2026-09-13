"""The paired fusion tool must load trajectories at the configured channel width.

It hardcoded the 3-channel `xyp` loader, so it crashed against the shipped
4-channel `dt` model (`mat1 and mat2 shapes cannot be multiplied (…x6 and 8x1024)`)
and had silently reported the older 3-channel model's fusion number since the
`dt` release. Everything downstream derives from `TEMPORAL_INPUT_FEATURES`.
"""
import numpy as np

from config.settings import TEMPORAL_INPUT_FEATURES
from tools.fusion_paired_eval import load_paired_trajectory


def test_paired_loader_yields_the_configured_channel_width(tmp_path):
    # X Y Z, 200 Hz grid, with a pen-up run (Z = 0) so `dt` has something to measure.
    z = [600] * 30 + [0] * 12 + [700] * 30
    lines = [f"{100 + i} {200 + (i % 7)} {p}" for i, p in enumerate(z)]
    path = tmp_path / "001_01.hwr"
    path.write_text("\n".join(lines) + "\n")

    trajectory = load_paired_trajectory(str(path))

    assert isinstance(trajectory, np.ndarray)
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == TEMPORAL_INPUT_FEATURES
