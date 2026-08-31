"""
The latency and memory budget, asserted rather than documented.

AGENTS.md and docs/rules.md state < 3.0 s end-to-end and <= 4 GB RSS as hard
constraints. Nothing measured either until `tools/profile_latency.py` existed,
so a change that doubled inference cost would have shipped silently.

These are deliberately loose: they are a regression tripwire on the order of
magnitude, not a benchmark. The measured warm path is ~0.06 s against a 3.0 s
budget, so a failure here means something structural broke — a model rebuilt
per call, gradient checkpointing switched on at inference, a full corpus
materialized on the probe path — not that the box was briefly busy.

Weight files are gitignored and distributed via GitHub Releases, so every test
here skips cleanly on a fresh clone rather than failing.
"""

import os

import numpy as np
import pytest

from config.settings import (
    INFERENCE_NUM_THREADS,
    INFERENCE_RAM_CEILING_GB,
    LATENCY_SLA_SECONDS,
    MODEL_WEIGHTS_DIR,
)

SPATIAL_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth")
TEMPORAL_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, "temporal_lstm.pth")

needs_spatial = pytest.mark.skipif(not os.path.exists(SPATIAL_WEIGHTS),
                                   reason="spatial weights not present (gitignored)")
needs_temporal = pytest.mark.skipif(not os.path.exists(TEMPORAL_WEIGHTS),
                                    reason="temporal weights not present (gitignored)")


class TestLatencyBudget:

    @needs_spatial
    def test_warm_spatial_verification_is_inside_the_sla(self, tmp_path):
        import time

        from tools.profile_latency import measure_warm

        median, worst, _ = measure_warm("spatial", repeats=5, tmpdir=str(tmp_path))
        assert worst < LATENCY_SLA_SECONDS, (
            f"worst-case spatial verification {worst:.3f}s breached the "
            f"{LATENCY_SLA_SECONDS}s budget")

    @needs_spatial
    @needs_temporal
    def test_warm_fused_verification_is_inside_the_sla(self, tmp_path):
        from tools.profile_latency import measure_warm

        median, worst, _ = measure_warm("fused", repeats=5, tmpdir=str(tmp_path))
        assert worst < LATENCY_SLA_SECONDS

    @needs_spatial
    def test_peak_memory_is_inside_the_ceiling(self, tmp_path):
        import resource

        from tools.profile_latency import measure_warm

        measure_warm("spatial", repeats=3, tmpdir=str(tmp_path))
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        assert peak_gb < INFERENCE_RAM_CEILING_GB


class TestThreadCap:
    """
    Torch and OpenCV each size their pool to every logical CPU independently.
    Measured: an identical spatial forward took 41 ms idle and 172-322 ms under
    contention on the same box. The cap bounds the contended case.
    """

    @needs_spatial
    def test_load_model_caps_torch_threads(self):
        import torch

        from src.verification import load_model

        load_model("spatial", SPATIAL_WEIGHTS, "cpu")
        assert torch.get_num_threads() == INFERENCE_NUM_THREADS

    @needs_spatial
    def test_load_model_caps_opencv_threads(self):
        import cv2

        from src.verification import load_model

        load_model("spatial", SPATIAL_WEIGHTS, "cpu")
        assert cv2.getNumThreads() == INFERENCE_NUM_THREADS
