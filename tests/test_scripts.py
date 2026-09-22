"""
End-to-end tests for the training and evaluation entry points.

These run real (tiny) training loops on generated fixtures — a few writers,
64×64 images, 2 epochs — purely to prove the wiring executes and produces
artifacts. They are not a substitute for a real training run.
"""

import os

import cv2
import numpy as np
import pytest

from scripts.train import main as train_main
from scripts.evaluate import (
    alpha_grid_search,
    compute_eer,
    compute_far_frr,
    main as evaluate_main,
    plot_det,
    plot_roc,
)


def _make_cedar_fixture(root, num_writers=10, num_genuine=3, num_forged=2):
    """Generate a CEDAR-shaped dataset of small synthetic signature images."""
    org = os.path.join(root, "full_org")
    forg = os.path.join(root, "full_forg")
    os.makedirs(org, exist_ok=True)
    os.makedirs(forg, exist_ok=True)

    rng = np.random.default_rng(0)
    for writer in range(1, num_writers + 1):
        for sample in range(1, num_genuine + 1):
            img = np.full((64, 64), 255, dtype=np.uint8)
            cv2.line(img, (5, 10 + writer % 20), (58, 50 - writer % 20), 0, 2)
            cv2.imwrite(os.path.join(org, f"original_{writer}_{sample}.png"), img)
        for sample in range(1, num_forged + 1):
            img = np.full((64, 64), 255, dtype=np.uint8)
            cv2.line(img, (5, 50), (58, 10), 0, 2)
            img = np.clip(img.astype(int) - rng.integers(0, 20, img.shape), 0, 255)
            cv2.imwrite(os.path.join(forg, f"forgeries_{writer}_{sample}.png"),
                        img.astype(np.uint8))
    return root


def _make_svc_fixture(root, num_writers=10, num_each=3, with_pressure=False):
    """
    Generate an SVC2004-shaped writer_samples of variable-length trajectories.

    `with_pressure=False` emits Task1 (4-column); True emits Task2 (7-column).
    """
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(1)

    for user in range(1, num_writers + 1):
        for index in range(num_each):
            for sample, offset in ((index + 1, 0), (index + 21, 500)):
                length = int(rng.integers(30, 60))
                path = os.path.join(root, f"U{user}S{sample}.TXT")
                with open(path, "w") as fh:
                    fh.write(f"{length}\n")
                    for t in range(length):
                        x = 1000 + offset + t * 7 + int(rng.integers(0, 30))
                        y = 2000 + offset + t * 3 + int(rng.integers(0, 30))
                        if with_pressure:
                            fh.write(f"{x} {y} {t * 10} {1 if t else 0} "
                                     f"1500 700 {int(rng.integers(0, 1024))}\n")
                        else:
                            fh.write(f"{x} {y} {t * 10} {1 if t else 0}\n")
    return root


class TestTrainMain:
    """The wiring A9 closes: discovery → split → pairs → loaders → epoch loop."""

    def test_spatial_run_writes_weights_and_history(self, tmp_path):
        """A 2-epoch spatial run must produce finite losses and a saved .pth."""
        data = _make_cedar_fixture(str(tmp_path / "cedar"))
        weights = str(tmp_path / "weights")

        history = train_main([
            "--branch", "spatial", "--data-dir", data,
            "--epochs", "2", "--batch-size", "8",
            "--weights-dir", weights, "--num-workers", "0",
        ])

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(np.isfinite(v) for v in history["train_loss"])
        assert os.path.exists(os.path.join(weights, "spatial_cnn.pth"))

        # No writer may appear in more than one partition (ISO/IEC 23053). The
        # split is assembled before the epoch loop, so this needs no extra run.
        train_ids, val_ids, test_ids = (set(history["split"][k])
                                        for k in ("train", "val", "test"))
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

        # `--temporal-channels` defaults to the shipped `dt`, but it is
        # temporal-only: `build_writer_samples` forwards it solely to trajectory
        # corpora. Asking CEDAR for a timing channel used to abort a valid
        # spatial run, and this run reaching this point is that guard holding.
        assert history["split"]

    def test_temporal_run_handles_variable_lengths(self, tmp_path):
        """The temporal path must survive ragged sequences via the collate/pack route."""
        data = _make_svc_fixture(str(tmp_path / "svc2004_task2"), with_pressure=True)
        weights = str(tmp_path / "weights")

        history = train_main([
            "--branch", "temporal", "--data-dir", data,
            "--epochs", "2", "--batch-size", "4",
            "--weights-dir", weights, "--num-workers", "0",
        ])

        assert all(np.isfinite(v) for v in history["train_loss"])
        assert os.path.exists(os.path.join(weights, "temporal_lstm.pth"))

    def test_pressure_ablation_runs_write_separate_weights(self, tmp_path):
        """
        AGENTS.md pressure ablation: the Task1 (pressure-blind) run must never
        overwrite the Task2 (shipped) weights, or the two results collapse into one.
        """
        weights = str(tmp_path / "weights")
        task1 = _make_svc_fixture(str(tmp_path / "svc2004_task1"), with_pressure=False)
        task2 = _make_svc_fixture(str(tmp_path / "svc2004_task2"), with_pressure=True)

        # Pinned to xyp on both arms. Task1's four columns cannot supply `dt`, and
        # the ablation is only meaningful if both halves see the same channels —
        # so the shipped default is overridden here rather than compared across.
        common = ["--branch", "temporal", "--epochs", "1", "--batch-size", "4",
                  "--temporal-channels", "xyp",
                  "--weights-dir", weights, "--num-workers", "0", "--seed", "42"]
        history_a = train_main(["--data-dir", task1] + common)
        history_b = train_main(["--data-dir", task2] + common)

        assert os.path.exists(os.path.join(weights, "temporal_lstm_task1.pth"))
        assert os.path.exists(os.path.join(weights, "temporal_lstm.pth"))

        # Same seed ⇒ identical writer partition, so the two EERs are comparable.
        assert history_a["split"] == history_b["split"]

    def test_unknown_dataset_directory_raises(self, tmp_path):
        """An unrecognized dataset name must fail loudly, not train on nothing."""
        mystery = tmp_path / "mystery_dataset"
        mystery.mkdir()
        with pytest.raises(ValueError, match="Unrecognized dataset"):
            train_main(["--branch", "spatial", "--data-dir", str(mystery),
                        "--epochs", "1", "--weights-dir", str(tmp_path / "w")])


class TestPairBudget:
    """
    `--pairs-per-writer` overrides the pair budget for the branch being trained.
    It existed for the spatial screen only, which left the temporal budget
    (`TEMPORAL_PAIRS_*_PER_WRITER = 180`) untestable and therefore unmeasured.
    Both branches must honour it, and the default must stay the config value.
    """

    def _writers(self, n_writers=4, n_samples=14, temporal=False):
        rng = np.random.default_rng(0)
        out = {}
        for w in range(n_writers):
            if temporal:
                gen = [rng.random((30, 4)).astype(np.float32) for _ in range(n_samples)]
                forg = [rng.random((30, 4)).astype(np.float32) for _ in range(n_samples)]
            else:
                gen = [rng.random((1, 155, 220)).astype(np.float32) for _ in range(n_samples)]
                forg = [rng.random((1, 155, 220)).astype(np.float32) for _ in range(n_samples)]
            out[f"w{w}"] = {"genuine": gen, "forged": forg}
        return out

    @pytest.mark.parametrize("branch,temporal", [("spatial", False), ("temporal", True)])
    def test_budget_is_honoured_on_both_branches(self, branch, temporal):
        from scripts.train import build_loader

        writers = self._writers(temporal=temporal)
        ids = sorted(writers)
        _, small = build_loader(writers, ids, branch, 4, 0, False, 42, pairs_per_writer=3)
        _, large = build_loader(writers, ids, branch, 4, 0, False, 42, pairs_per_writer=12)
        # Positives and negatives are drawn per writer, so the total scales with
        # the budget. 14 samples supply C(14,2)=91 genuine pairs, well above 12.
        assert small == 3 * 2 * len(ids), f"{branch}: expected {3 * 2 * len(ids)} pairs, got {small}"
        assert large == 12 * 2 * len(ids)

    def test_default_uses_the_config_budget_for_the_temporal_branch(self):
        from scripts.train import build_loader
        from config.settings import TEMPORAL_PAIRS_GENUINE_PER_WRITER

        writers = self._writers(n_writers=2, n_samples=6, temporal=True)
        _, n = build_loader(writers, sorted(writers), "temporal", 4, 0, False, 42)
        # 6 samples cap each writer at C(6,2)=15 pairs, below the 180 default,
        # so this asserts the default is read rather than any number being passed.
        assert TEMPORAL_PAIRS_GENUINE_PER_WRITER == 180
        assert n == 15 * 2 * 2


class TestEvaluationMetrics:
    """Metric correctness on constructed score distributions."""

    def test_perfectly_separable_scores_give_zero_eer(self):
        eer, _ = compute_eer(np.array([0.9, 0.95, 1.0]), np.array([0.0, 0.05, 0.1]))
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_identical_distributions_give_high_eer(self):
        scores = np.linspace(0, 1, 50)
        eer, _ = compute_eer(scores, scores.copy())
        assert eer > 0.4

    def test_far_frr_at_threshold(self):
        """At τ=0.5: 1 of 4 impostors accepted, 1 of 4 genuines rejected."""
        far, frr = compute_far_frr(np.array([0.4, 0.6, 0.7, 0.8]),
                                   np.array([0.1, 0.2, 0.3, 0.9]), 0.5)
        assert far == pytest.approx(0.25)
        assert frr == pytest.approx(0.25)

    def test_alpha_grid_search_prefers_informative_branch(self):
        """When temporal separates and spatial is noise, α must collapse toward 0."""
        rng = np.random.default_rng(0)
        n = 200
        spatial_genuine = rng.normal(0.5, 0.3, n)
        spatial_impostor = rng.normal(0.5, 0.3, n)
        temporal_genuine = np.full(n, 0.95)
        temporal_impostor = np.full(n, 0.05)

        best_alpha, best_eer, results = alpha_grid_search(
            spatial_genuine, spatial_impostor, temporal_genuine, temporal_impostor
        )
        assert best_alpha < 0.5
        assert best_eer < 0.05
        assert len(results) == 21  # α from 0.00 to 1.00 in 0.05 steps

    def test_plots_are_written(self, tmp_path):
        genuine = np.array([0.8, 0.9, 0.95])
        impostor = np.array([0.1, 0.2, 0.3])
        roc = str(tmp_path / "roc.png")
        det = str(tmp_path / "det.png")
        plot_roc(genuine, impostor, "ROC", roc)
        plot_det(genuine, impostor, "DET", det)
        assert os.path.getsize(roc) > 0
        assert os.path.getsize(det) > 0


class TestEvaluateMain:
    """The wiring A10 closes: weights → inference → S=1/(1+d) → metrics → plots."""

    def test_spatial_evaluation_produces_metrics_and_plots(self, tmp_path):
        """Train tiny, then evaluate those weights end-to-end."""
        data = _make_cedar_fixture(str(tmp_path / "cedar"))
        weights_dir = str(tmp_path / "weights")
        output = str(tmp_path / "results")

        train_main(["--branch", "spatial", "--data-dir", data,
                    "--epochs", "1", "--batch-size", "8",
                    "--weights-dir", weights_dir, "--num-workers", "0"])

        report = evaluate_main([
            "--branch", "spatial",
            "--weights", os.path.join(weights_dir, "spatial_cnn.pth"),
            "--data-dir", data, "--output-dir", output, "--num-workers", "0",
        ])

        assert 0.0 <= report["eer"] <= 1.0
        assert 0.0 <= report["far"] <= 1.0
        assert 0.0 <= report["frr"] <= 1.0
        for plot in ("roc_spatial.png", "det_spatial.png", "confusion_spatial.png"):
            assert os.path.exists(os.path.join(output, plot)), f"missing {plot}"

        # AGENTS.md: tau comes from validation, metrics are reported on test.
        # Same weights, same argv, so this needs no second train+evaluate cycle.
        assert report["threshold_tuned_on"] == "validation"
        assert report["reported_on"] == "test"

    def test_evaluating_on_a_split_the_checkpoint_did_not_train_under_raises(self, tmp_path):
        """
        The 2026-09-06 leak: a checkpoint scored on writers it trained on because
        the scorer recomputed a different partition than training used. Training
        now records its split beside the weights and evaluation refuses a
        partition that disagrees with it.
        """
        from src.data.dataset import load_split_record
        data = _make_cedar_fixture(str(tmp_path / "cedar"))
        weights_dir = str(tmp_path / "weights")
        weights = os.path.join(weights_dir, "spatial_cnn.pth")

        history = train_main(["--branch", "spatial", "--data-dir", data,
                              "--epochs", "1", "--batch-size", "8", "--split-seed", "42",
                              "--weights-dir", weights_dir, "--num-workers", "0"])
        record = load_split_record(weights)
        assert record is not None and set(record["test"]) == set(history["split"]["test"])

        with pytest.raises(ValueError, match="trained on|held-out"):
            evaluate_main(["--branch", "spatial", "--weights", weights, "--data-dir", data,
                           "--split-seed", "7", "--output-dir", str(tmp_path / "r"), "--num-workers", "0"])


class TestFusedChannelDispatch:
    """
    `extra_channels` is a temporal-only concept, and a fused run lists image and
    trajectory corpora in one `--data-dir`.

    `build_writer_samples` used to offer the channel to every directory, so with
    the shipped default (`--temporal-channels dt`) the first spatial corpus
    aborted the run: `--branch fused` was unrunnable out of the box.
    """

    def test_spatial_corpora_are_not_asked_for_a_temporal_channel(self, tmp_path):
        from scripts.train import build_writer_samples

        cedar = tmp_path / "cedar"
        _make_cedar_fixture(str(cedar), num_writers=3)
        svc = tmp_path / "svc2004_task2"
        _make_svc_fixture(str(svc), num_writers=3, with_pressure=True)

        merged = build_writer_samples([str(cedar), str(svc)], extra_channels="dt")
        assert len(merged) == 6

    def test_a_temporal_corpus_that_cannot_supply_the_channel_still_raises(self, tmp_path):
        """The width-consistency guard must survive the fix that unblocked fusion."""
        from scripts.train import build_writer_samples

        task1 = tmp_path / "svc2004_task1"
        _make_svc_fixture(str(task1), num_writers=2)
        with pytest.raises(ValueError, match="cannot supply"):
            build_writer_samples([str(task1)], extra_channels="tilt")
