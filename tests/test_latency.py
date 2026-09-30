"""
The latency and memory budget, asserted rather than documented.

AGENTS.md and docs/rules.md state < 3.0 s end-to-end and <= 4 GB RSS as hard
constraints. Nothing measured either until `tools/profile_latency.py` existed,
so a change that doubled inference cost would have shipped silently.

Off CI (the deployment-class laptop) the worst warm repeat is held to the 3.0 s
budget directly. On a shared CI runner wall-clock time measures the rented host
more than the code: commit 38e834f took 3 s on an eastus runner and 47 s on a
westus3 one (runs 36316908474 and 36316801302, 2026-09-27), same image, same
code, every repeat 4.2 s. So on CI the test times a fixed convolution with no
project code in it, in the same process, and asserts verification time as a
multiple of that. A slow host slows both; a slower verification path moves the
ratio. Measured 2026-09-27 the healthy ratio is 7 to 9 on 1 to 4 cores, while
absolute time doubled across that range, which is what makes it host-independent.

What this does and does not catch, measured rather than assumed. The ceiling of
30 fails a verification path about three times slower relative to the host than
today's: a corpus loaded on the probe path, a per-call embedding of every
reference. It does **not** catch a model rebuilt on every call: the 26 MB weight
loads from page cache in about 100 ms, which moves the ratio to about 15 and the
laptop to 0.23 s, inside both budgets. That regression, which an earlier version
of this docstring claimed to catch, is caught deterministically by
`test_models_load_once_across_verifications` below.

Weight files are gitignored and distributed via GitHub Releases, so every test
here skips cleanly on a fresh clone rather than failing.
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


# Verification time as a multiple of the reference convolution. Healthy is 7 to 9
# (measured 2026-09-27 on 1 to 4 cores); the ceiling allows roughly 3x before failing.
CI_RATIO_CEILING = 30.0


def reference_seconds(repeats=5):
    """Median time of a fixed convolution with no project code in it: this host's speed."""
    import statistics
    import time

    import torch

    torch.manual_seed(0)
    x, w = torch.randn(1, 64, 155, 220), torch.randn(64, 64, 3, 3)
    with torch.no_grad():
        torch.nn.functional.conv2d(x, w, padding=1)            # warm the kernels
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            torch.nn.functional.conv2d(x, w, padding=1)
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def assert_inside_sla(branch, median, worst, reference=None):
    """The 3.0 s budget off CI; on CI, verification as a multiple of this host's speed."""
    if ON_CI:
        ratio = median / reference
        assert ratio < CI_RATIO_CEILING, (
            f"{branch} verification is {ratio:.1f}x the reference convolution on this host "
            f"(ceiling {CI_RATIO_CEILING:g}, healthy 7 to 9): median {median:.3f}s against "
            f"a reference of {reference * 1000:.1f} ms. The path got slower, not the host.")
    else:
        assert worst < LATENCY_SLA_SECONDS, (
            f"worst-case {branch} verification {worst:.3f}s breached the "
            f"{LATENCY_SLA_SECONDS}s budget (median {median:.3f}s over 5 warm repeats)")


class TestLatencyBudget:

    @needs_spatial
    def test_warm_spatial_verification_is_inside_the_sla(self, tmp_path):
        from tools.profile_latency import measure_warm

        reference = reference_seconds()
        median, worst, _ = measure_warm("spatial", repeats=5, tmpdir=str(tmp_path))
        assert_inside_sla("spatial", median, worst, reference)

    @needs_spatial
    @needs_temporal
    def test_warm_fused_verification_is_inside_the_sla(self, tmp_path):
        from tools.profile_latency import measure_warm

        reference = reference_seconds()
        median, worst, _ = measure_warm("fused", repeats=5, tmpdir=str(tmp_path))
        assert_inside_sla("fused", median, worst, reference)

    @needs_spatial
    @needs_temporal
    def test_models_load_once_across_verifications(self, monkeypatch):
        """
        A model rebuilt on every call is invisible to a timing test on this
        hardware (about 100 ms against a 3 s budget), so count the loads instead.
        """
        import numpy as np

        import src.verification as verification

        loads = []
        real = verification.load_model
        monkeypatch.setattr(verification, "load_model",
                            lambda branch, *a, **k: (loads.append(branch), real(branch, *a, **k))[1])
        service = verification.SignatureService()
        from src.data.parsers import normalize_trajectory

        rng = np.random.default_rng(0)
        trajectories = []
        for _ in range(6):                                          # through the real capture normaliser
            xy = np.cumsum(rng.normal(0, 5.0, (80, 2)), axis=0)
            trajectories.append(normalize_trajectory(xy, np.full(80, 0.5), extra=np.full((80, 1), 0.1)))  # 10 ms of the 100 ms clip
        templates = service.enroll_from_tablet(trajectories[:5])
        for _ in range(3):
            service.verify(mode="temporal-multi", probe_trajectory=trajectories[5],
                           reference_temporal=templates["temporal_embeddings"],
                           reference_spatial=templates["spatial_embeddings"],
                           reference_spatial_origin=templates["spatial_origin"])
        assert loads.count("spatial") == 1 and loads.count("temporal") == 1, (
            f"models loaded {loads} across one enrolment and three verifications; "
            f"each branch must load once and stay resident")

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
