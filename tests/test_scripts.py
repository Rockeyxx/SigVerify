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
    build_eval_loader,
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

    def test_device_augment_reaches_only_the_training_loader(self):
        from scripts.train import build_loader

        writers = self._writers(n_writers=2, n_samples=6, temporal=True)
        train, _ = build_loader(writers, sorted(writers), "temporal", 4, 0, True, 42,
                                device_augment=True)
        val, _ = build_loader(writers, sorted(writers), "temporal", 4, 0, False, 42,
                              device_augment=True)
        assert train.dataset.device_augment and not val.dataset.device_augment

    def test_the_new_augmentation_flags_parse(self):
        from scripts.train import parse_args

        args = parse_args(["--branch", "temporal", "--data-dir", "data/x",
                           "--device-augment", "--aug-blur", "1.0",
                           "--aug-ink-gain-min", "0.7", "--aug-noise", "0.02"])
        assert args.device_augment
        assert (args.aug_blur, args.aug_ink_gain_min, args.aug_noise) == (1.0, 0.7, 0.02)


def _make_utsig_fixture(root, num_writers=8, num_genuine=6):
    """UTSig-shaped tree: 2 Skilled, 3 Simple and 1 Opposite Hand forgery per writer."""
    rng = np.random.default_rng(2)
    counts = {"Skilled": 2, "Simple": 3, "Opposite Hand": 1}
    for writer in range(1, num_writers + 1):
        folders = [("Genuine", num_genuine)] + [
            (os.path.join("Forgery", kind), n) for kind, n in counts.items()]
        for folder, n in folders:
            out = os.path.join(root, folder, str(writer))
            os.makedirs(out, exist_ok=True)
            for sample in range(1, n + 1):
                img = np.full((64, 64), 255, dtype=np.uint8)
                cv2.line(img, (5, 10 + writer), (58, 50 - writer), 0, 2)
                img = np.clip(img.astype(int) - rng.integers(0, 20, img.shape), 0, 255)
                cv2.imwrite(os.path.join(out, f"{sample}.png"), img.astype(np.uint8))
    return root


class TestUTSigTrainingForgeries:
    """
    UTSig's Simple and Opposite Hand forgeries are a training-only lever.

    Validation and test must keep skilled forgeries only, or every reported
    UTSig number would silently change protocol along with the training data.
    """

    def test_default_is_skilled_only(self):
        from config.settings import UTSIG_TRAIN_FORGERY_TYPES

        assert tuple(UTSIG_TRAIN_FORGERY_TYPES) == ("Skilled",)

    def test_extra_types_reach_training_writers_only(self, tmp_path):
        from scripts.train import add_training_forgeries, build_writer_samples

        data = _make_utsig_fixture(str(tmp_path / "utsig"))
        samples = build_writer_samples([data])
        train_ids = ["utsig_1", "utsig_2"]
        before = {w: list(r["forged"]) for w, r in samples.items()}

        train_samples, caps = add_training_forgeries(
            samples, train_ids, [data], ("Skilled", "Simple", "Opposite Hand"),
            hold_pairs=False)

        assert caps is None
        for w in train_ids:
            assert len(train_samples[w]["forged"]) == 2 + 3 + 1
        # Every other writer, and the shared mapping itself, is untouched.
        for w, forged in before.items():
            assert samples[w]["forged"] == forged
            assert all(os.sep + "Skilled" + os.sep in p for p in forged)

    def test_hold_pairs_caps_at_the_skilled_only_supply(self, tmp_path):
        from scripts.train import add_training_forgeries, build_writer_samples

        data = _make_utsig_fixture(str(tmp_path / "utsig"))
        samples = build_writer_samples([data])

        _, caps = add_training_forgeries(
            samples, ["utsig_1"], [data], ("Skilled", "Simple", "Opposite Hand"),
            hold_pairs=True)

        # Skilled-only supply: min(C(6,2)=15, 6 x 2 = 12) = 12.
        assert caps == {"utsig_1": 12}

    def test_default_types_change_nothing(self, tmp_path):
        from scripts.train import add_training_forgeries, build_writer_samples

        data = _make_utsig_fixture(str(tmp_path / "utsig"))
        samples = build_writer_samples([data])

        train_samples, caps = add_training_forgeries(
            samples, ["utsig_1"], [data], ("Skilled",), hold_pairs=True)

        assert train_samples is samples
        assert caps is None


class TestEarlyStoppingSurvivesResume:
    """Verifies that early stopping patience state is preserved when resuming training."""

    def test_checkpoint_carries_the_counter_and_baseline(self, tmp_path):
        import torch
        import torch.optim as optim

        from scripts.train import EarlyStopping, save_checkpoint

        stopper = EarlyStopping(patience=5)
        for loss in (0.30, 0.31, 0.32):        # one improvement then two stalls
            stopper(loss)
        assert (stopper.counter, stopper.best_loss) == (2, 0.30)

        model = torch.nn.Linear(2, 2)
        path = str(tmp_path / "ckpt.pth")
        save_checkpoint(model, optim.Adam(model.parameters()), 3, 0.32, path,
                        best_val_loss=0.30, early_stopping=stopper)
        state = torch.load(path, map_location="cpu", weights_only=False)["early_stopping"]
        assert state == {"counter": 2, "best_loss": 0.30}

    def test_restored_stopper_continues_instead_of_restarting(self):
        from scripts.train import EarlyStopping

        stopper = EarlyStopping(patience=5)
        stopper.load_state({"counter": 4, "best_loss": 0.30})
        assert stopper.counter == 4 and stopper.best_loss == 0.30
        # The fifth stall must stop the run, not the ninth.
        assert stopper(0.31) is True

    def test_the_checkpoint_includes_this_epochs_verdict(self, tmp_path, monkeypatch):
        import torch
        import torch.optim as optim

        import scripts.train as train

        losses = iter([0.30, 0.31])            # one baseline, then one stall
        monkeypatch.setattr(train, "train_one_epoch", lambda *a, **k: 0.0)
        monkeypatch.setattr(train, "validate_one_epoch", lambda *a, **k: next(losses))
        model = torch.nn.Linear(2, 2)
        path = str(tmp_path / "ckpt.pth")
        train.fit(model, None, None, None, optim.Adam(model.parameters()), "cpu", 2,
                  str(tmp_path / "w.pth"), path)
        state = torch.load(path, map_location="cpu", weights_only=False)["early_stopping"]
        # Saved after the stall was counted; a stale record resumes one epoch behind.
        assert state == {"counter": 1, "best_loss": 0.30}

    @pytest.mark.slow  # real training loop; skipped by default, run in CI
    def test_main_hands_the_saved_counter_to_the_resumed_run(self, tmp_path):
        import torch

        data = _make_utsig_fixture(str(tmp_path / "utsig"), num_writers=10)
        common = ["--branch", "spatial", "--data-dir", data, "--batch-size", "8",
                  "--weights-dir", str(tmp_path / "w"), "--num-workers", "0"]
        train_main(common + ["--epochs", "1"])
        ckpt = str(tmp_path / "w" / "checkpoint_spatial_cnn.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state["early_stopping"] = {"counter": 4, "best_loss": -1.0}   # unbeatable, one stall left
        torch.save(state, ckpt)

        history = train_main(common + ["--epochs", "5", "--resume", ckpt])
        # Patience 5 with four stalls already counted: the first resumed epoch stops it.
        assert len(history["val_loss"]) == 1

    def test_a_record_without_the_field_starts_fresh(self):
        from scripts.train import EarlyStopping

        stopper = EarlyStopping(patience=5)
        stopper.load_state(None)               # checkpoint predating the field
        assert stopper.counter == 0 and stopper.best_loss is None


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

    def test_eval_loader_negatives_are_same_writer_forgeries(self):
        """Verify that evaluation loader negatives are skilled forgeries from the same writer."""
        writers = {f"w{w}": {"genuine": [f"w{w}_g{i}" for i in range(6)],
                             "forged": [f"w{w}_f{i}" for i in range(6)]}
                   for w in range(4)}
        loader = build_eval_loader(writers, sorted(writers), "spatial",
                                   batch_size=4, num_workers=0, seed=42)
        dataset = loader.dataset
        cross = [(s1, s2) for (s1, s2), y in zip(dataset.pairs, dataset.labels)
                 if y == 0 and s1.split("_")[0] != s2.split("_")[0]]
        assert cross == [], f"{len(cross)} cross-writer negatives, e.g. {cross[0]}"


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
        """Verify that evaluation fails if the partition differs from the training split record."""
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


class _Stop(Exception):
    """Raised by a stand-in loader to end a run once it has been asked for its data."""


def _stop_at_loading(seen):
    def spy(data_dirs, extra_channels=None, float32_parse=False):
        seen.append(float32_parse)
        raise _Stop
    return spy


class TestFloat32Parse:
    """
    Training reads SVC2004 and DeepSignDB as float32, the read the shipped dt
    recipe was trained on, and every scorer reads them exactly. On exact
    timestamps the recipe is 2.8 EER points worse.
    """

    @staticmethod
    def _svc_with_a_large_clock(root):
        os.makedirs(root)
        for user in (1, 2):
            for sample in (1, 21):
                with open(os.path.join(root, f"U{user}S{sample}.TXT"), "w") as fh:
                    fh.write("30\n" + "".join(f"{100 + t} {200 + t} {70_000_000 + 10 * t} 1 1500 700 500\n"
                                              for t in range(30)))
        return root

    def test_only_the_corpora_with_a_device_clock_are_asked_for_it(self, tmp_path):
        from scripts.train import build_writer_samples
        from src.data.parsers import MAX_GAP_MS

        cedar = _make_cedar_fixture(str(tmp_path / "cedar"), num_writers=3)
        svc = self._svc_with_a_large_clock(str(tmp_path / "svc2004_task2"))
        exact = build_writer_samples([svc], extra_channels="dt")
        old = build_writer_samples([cedar, svc], extra_channels="dt", float32_parse=True)
        assert np.allclose(exact["svc_1"]["genuine"][0][1:, 3], 10.0 / MAX_GAP_MS)
        assert not np.allclose(old["svc_1"]["genuine"][0][1:, 3], 10.0 / MAX_GAP_MS)

    def test_training_reads_the_corpora_as_the_shipped_recipe_did(self, tmp_path, monkeypatch):
        import scripts.train as train
        from config.settings import TEMPORAL_TRAIN_FLOAT32_PARSE

        seen = []
        monkeypatch.setattr(train, "build_writer_samples", _stop_at_loading(seen))
        with pytest.raises(_Stop):
            train.main(["--branch", "temporal", "--data-dir", str(tmp_path), "--weights-dir", str(tmp_path)])
        assert TEMPORAL_TRAIN_FLOAT32_PARSE is True
        assert seen == [True]

    def test_the_benchmark_reads_exactly_unless_asked(self, tmp_path, monkeypatch):
        """`--float32-parse` exists so the releases scored before the fix still reproduce."""
        import scripts.benchmark as benchmark
        import scripts.train as train

        seen = []
        monkeypatch.setattr(train, "build_writer_samples", _stop_at_loading(seen))
        for extra in ([], ["--float32-parse"]):
            with pytest.raises(_Stop):
                benchmark.main(["--branch", "temporal", "--weights", str(tmp_path / "w.pth"),
                                "--data-dir", str(tmp_path), *extra])
        assert seen == [False, True]
