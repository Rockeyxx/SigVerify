"""
tools/reproduce_release.py: score (or retrain) a published release and compare
with the numbers its notes published.
"""

import os

import pytest

from config.settings import SPATIAL_TRAIN_DIRS, TEMPORAL_CHANNEL_WIDTHS, TEMPORAL_TRAIN_DIRS
from tools.reproduce_release import (RELEASES, benchmark_command, benchmark_env, parse_benchmark, release_dir,
                                     train_command, verdict)

# Trimmed from a real `benchmark.py --enroll 5` run on the v0.1.0 weight.
SAMPLE = """\
Enrollment protocol: 5 references x 10 draws, nearest-reference distance, 65 of 65 writers scoreable
  EER over 10 draws: mean 12.31%  std 0.96%  min 10.37%  max 14.35%
==================================================
SEPARATION REPORT — spatial
==================================================
  EER:                      12.40%
  Cohen's d:                1.778
==================================================
PER-DATASET BREAKDOWN — spatial
==================================================
Enrollment protocol: 5 references x 10 draws, nearest-reference distance, 16 of 16 writers scoreable
  EER over 10 draws: mean 14.69%  std 1.76%  min 12.13%  max 17.47%
  bhsig_Bengali     16 writers   EER  14.26%   Cohen's d  1.600     784 pairs
Enrollment protocol: 5 references x 10 draws, nearest-reference distance, 8 of 8 writers scoreable
  EER over 10 draws: mean 0.00%  std 0.00%  min 0.00%  max 0.00%
  cedar              8 writers   EER   0.00%   Cohen's d  7.177     344 pairs
"""


def test_parse_reads_the_overall_and_per_corpus_eers():
    r = parse_benchmark(SAMPLE)
    assert (r["mean"], r["sd"], r["pooled"]) == (12.31, 0.96, 12.40)
    assert r["corpora"]["bhsig_Bengali"] == {"mean": 14.69, "sd": 1.76, "pooled": 14.26}
    assert r["corpora"]["cedar"]["mean"] == 0.00


def test_every_release_weight_is_well_formed():
    assert list(RELEASES)[0] == "v0.1.0" and "v0.10.0" in RELEASES
    for tag, weights in RELEASES.items():
        assert weights, tag
        for w in weights:
            assert w.branch in ("spatial", "temporal"), (tag, w)
            assert w.asset.endswith(".pth"), (tag, w)
            assert w.published and set(w.published) <= {"mean", "sd", "pooled"} | {
                k for k in w.published if "_" in k or k in ("cedar", "utsig", "svc")}, (tag, w)
            if w.branch == "temporal":
                assert w.channels in TEMPORAL_CHANNEL_WIDTHS, (tag, w)


def test_verdict_allows_only_rounding():
    """Published figures carry two decimals, so only an exact two-decimal match is a match."""
    assert verdict(12.31, 12.31) == "match"
    assert verdict(12.31, 12.32) == "differs by +0.01"
    assert verdict(8.59, 8.56) == "differs by -0.03"


def test_benchmark_command_takes_corpora_and_channels_from_settings():
    spatial = next(w for w in RELEASES["v0.10.0"] if w.branch == "spatial")
    cmd = benchmark_command(spatial, "/cache/spatial_cnn.pth")
    assert cmd[cmd.index("--data-dir") + 1:cmd.index("--data-dir") + 1 + len(SPATIAL_TRAIN_DIRS)] \
        == list(SPATIAL_TRAIN_DIRS)
    assert "--temporal-channels" not in cmd

    old = next(w for w in RELEASES["v0.5.0"] if w.branch == "temporal")
    cmd = benchmark_command(old, "/cache/temporal_lstm.pth")
    assert cmd[cmd.index("--temporal-channels") + 1] == "xyp"
    assert all(d in cmd for d in TEMPORAL_TRAIN_DIRS)


def test_retrain_refuses_a_release_that_recorded_no_recipe():
    with pytest.raises(ValueError, match="no training command"):
        train_command("v0.1.0", RELEASES["v0.1.0"][0], "/out")


def test_retrain_runs_the_recorded_recipe_at_its_commit():
    spatial = next(w for w in RELEASES["v0.10.0"] if w.branch == "spatial")
    commit, argv = train_command("v0.10.0", spatial, "/out")
    assert commit == "8bb6f5e"
    assert argv[argv.index("--aug-stroke-prob") + 1] == "0"
    assert argv[argv.index("--weights-dir") + 1] == "/out"


def test_scoring_runs_on_the_device_the_release_was_measured_on():
    """
    The same weights score differently in the second decimal on CPU and CUDA:
    v0.8.0's temporal 7.81 ± 0.63 is a CUDA figure and v0.9.0's 7.77 ± 0.67 is the
    same file on CPU. Each weight names its device, and CPU runs hide the GPU.
    """
    for weights in RELEASES.values():
        for w in weights:
            assert w.device in ("cpu", "cuda"), w
    cpu = next(w for w in RELEASES["v0.9.0"] if w.branch == "temporal")
    cuda = next(w for w in RELEASES["v0.8.0"] if w.branch == "temporal")
    assert benchmark_env(cpu)["CUDA_VISIBLE_DEVICES"] == ""
    assert benchmark_env(cuda).get("CUDA_VISIBLE_DEVICES") != ""


def test_only_a_known_release_tag_reaches_the_filesystem():
    """The cache path is built from the tag, so an unknown or crafted tag is refused before any mkdir."""
    assert release_dir("v0.10.0").endswith(os.path.join("releases", "v0.10.0"))
    for bad in ("../../etc", "v9.9.9", ""):
        with pytest.raises(ValueError, match="unknown release"):
            release_dir(bad)
