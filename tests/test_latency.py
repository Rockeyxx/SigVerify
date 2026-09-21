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

On a shared CI runner the budget is checked against the median of the repeats,
not the worst. Two docs-only commits (2026-09-20, runs 35519510031 and
35545056959) failed on a throttled runner where the whole test took 38 s
instead of 3 s and one of five repeats stalled to 3.4 s; the same path measures
0.07 s median / 0.09 s worst on the deployment-class laptop. A stalled sample on
a rented VM is a property of the VM, and a structural break moves the median
just as surely. Off CI the worst case is still held to the budget.
"""

import os

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


ON_CI = os.environ.get("CI", "").lower() in ("1", "true", "yes")


def assert_inside_sla(branch, median, worst):
    """The budget applies to the worst repeat on a real machine, to the median on CI."""
    measured = median if ON_CI else worst
    assert measured < LATENCY_SLA_SECONDS, (
        f"{'median' if ON_CI else 'worst-case'} {branch} verification {measured:.3f}s "
        f"breached the {LATENCY_SLA_SECONDS}s budget "
        f"(median {median:.3f}s, worst {worst:.3f}s over 5 warm repeats)")


class TestLatencyBudget:

    @needs_spatial
    def test_warm_spatial_verification_is_inside_the_sla(self, tmp_path):
        from tools.profile_latency import measure_warm

        median, worst, _ = measure_warm("spatial", repeats=5, tmpdir=str(tmp_path))
        assert_inside_sla("spatial", median, worst)

    @needs_spatial
    @needs_temporal
    def test_warm_fused_verification_is_inside_the_sla(self, tmp_path):
        from tools.profile_latency import measure_warm

        median, worst, _ = measure_warm("fused", repeats=5, tmpdir=str(tmp_path))
        assert_inside_sla("fused", median, worst)

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
