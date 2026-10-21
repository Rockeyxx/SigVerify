"""
Tests for the integration facade consumed by the GUI and database teams.

This is the contract boundary: if these pass, teammates can build against
`src.verification` without touching model internals or distance math.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from config.settings import TEMPORAL_INPUT_FEATURES
from scripts.evaluate import fit_writer_classifier, writer_dependent_scores
from src.models.siamese import FusionEngine, SpatialSiamese, TemporalSiamese
from src.verification import (
    bytes_to_embedding,
    embed,
    embedding_to_bytes,
    load_model,
    verify,
)


def _sharp_signature_image():
    """Write a temp image with enough ink and sharpness to pass the quality gate."""
    img = np.full((400, 400), 255, dtype=np.uint8)
    for i in range(50, 350, 5):
        cv2.line(img, (i, 100), (i + 30, 300), 0, 2)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


def _blank_image():
    """A near-empty image that must fail the quality gate."""
    img = np.full((40, 40), 255, dtype=np.uint8)
    cv2.circle(img, (20, 20), 3, 0, 1)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


def _valid_trajectory(length=50, seed=1):
    """
    A gate-passing trajectory of whatever width the shipped contract declares.

    Built from `TEMPORAL_INPUT_FEATURES` rather than a literal so that flipping
    `TEMPORAL_CHANNELS` in `config/settings.py` — the single knob selecting
    3 (`xyp`), 4 (`dt`) or 5 (`tilt`) channels — moves the whole suite with it.
    A checkpoint trained under any of those settings stays reproducible without
    anyone editing a test.

    X/Y are z-scored per signature; every trailing channel (P, dt, ...) is a
    physical quantity the parsers have already scaled onto [0, 1].
    """
    rng = np.random.default_rng(seed)
    columns = [rng.standard_normal((length, 2)),
               rng.uniform(0.0, 1.0, size=(length, 1))]
    for _ in range(TEMPORAL_INPUT_FEATURES - 3):
        columns.append(np.full((length, 1), 0.1))
    return np.hstack(columns).astype(np.float32)


class TestLoadModel:
    """Contract ①: teammates load weights without knowing the class names."""

    def test_spatial_round_trip(self, tmp_path):
        """load_model must reproduce the exact embedding the saved model produced."""
        original = SpatialSiamese(embedding_dim=512).eval()
        path = tmp_path / "spatial_cnn.pth"
        torch.save(original.state_dict(), path)

        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            expected = original.encode(x)
            actual = load_model("spatial", str(path)).encode(x)

        assert torch.allclose(expected, actual)

    def test_returns_model_in_eval_mode(self, tmp_path):
        """Inference must never run with dropout active."""
        path = tmp_path / "spatial_cnn.pth"
        torch.save(SpatialSiamese().state_dict(), path)
        assert load_model("spatial", str(path)).training is False

    def test_temporal_round_trip(self, tmp_path):
        original = TemporalSiamese(embedding_dim=128).eval()
        path = tmp_path / "temporal_lstm.pth"
        torch.save(original.state_dict(), path)

        # Not a literal 3. `TemporalSiamese` now takes its default width from
        # TEMPORAL_INPUT_FEATURES, so a fixed 3 here fed a 6-wide LSTM into a
        # model built 8 wide the moment the shipped contract moved to `dt`.
        x = torch.randn(1, 40, TEMPORAL_INPUT_FEATURES)
        with torch.no_grad():
            expected = original.encode(x)
            actual = load_model("temporal", str(path)).encode(x)

        assert torch.allclose(expected, actual)

    def test_unknown_branch_raises(self, tmp_path):
        path = tmp_path / "w.pth"
        torch.save(SpatialSiamese().state_dict(), path)
        with pytest.raises(ValueError, match="branch"):
            load_model("olfactory", str(path))

    def test_missing_weights_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("spatial", "/nonexistent/weights.pth")


class TestEmbeddingBytes:
    """Contract ②: the byte convention the DB team wraps in AES-256-GCM."""

    def test_spatial_embedding_is_2048_bytes(self):
        """512 float32 values → 512 × 4 = 2048 bytes."""
        assert len(embedding_to_bytes(np.zeros(512, dtype=np.float32))) == 2048

    def test_temporal_embedding_is_512_bytes(self):
        """128 float32 values → 128 × 4 = 512 bytes."""
        assert len(embedding_to_bytes(np.zeros(128, dtype=np.float32))) == 512

    def test_round_trip_is_exact(self):
        """Serialization must be lossless — a drifted embedding is a wrong identity."""
        original = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        assert np.array_equal(bytes_to_embedding(embedding_to_bytes(original)), original)

    def test_accepts_torch_tensor(self):
        """Callers hold tensors straight off the model; don't force manual conversion."""
        assert len(embedding_to_bytes(torch.zeros(512))) == 2048

    def test_decoded_embedding_is_writable(self):
        """np.frombuffer aliases read-only memory — callers must get a real array."""
        decoded = bytes_to_embedding(embedding_to_bytes(np.zeros(128, dtype=np.float32)))
        decoded[0] = 1.0  # must not raise

    def test_corrupt_payload_raises(self):
        """A truncated blob is not silently reinterpreted as a shorter embedding."""
        with pytest.raises(ValueError):
            bytes_to_embedding(b"\x00\x00\x00")


class TestEmbed:
    """Preprocessing + encoding in one call, so callers never build tensors."""

    def test_spatial_embed_from_path(self):
        path = _sharp_signature_image()
        embedding = embed(SpatialSiamese(embedding_dim=512).eval(), path, "spatial")
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        os.unlink(path)

    def test_temporal_embed_from_array(self):
        # Not a literal 3 — same reason as test_temporal_round_trip above: the
        # constructor's default width now follows TEMPORAL_INPUT_FEATURES.
        trajectory = np.random.default_rng(0).standard_normal(
            (60, TEMPORAL_INPUT_FEATURES)).astype(np.float32)
        embedding = embed(TemporalSiamese(embedding_dim=128).eval(), trajectory, "temporal")
        assert embedding.shape == (128,)


class TestVerify:
    """Contract ③/④: one entrypoint returning the GUI's dict, no distance math outside."""

    def _spatial_setup(self):
        model = SpatialSiamese(embedding_dim=512).eval()
        path = _sharp_signature_image()
        references = [embed(model, path, "spatial")]
        return model, references, path

    def _temporal_setup(self):
        # Width follows `TEMPORAL_CHANNELS`, so this fixture keeps working across
        # a 3 -> 4 -> 5 channel change without being rewritten. Pressure and the
        # trailing channels are in [0, 1] — they are scaled by a fixed device
        # maximum, not z-scored like X/Y.
        model = TemporalSiamese(input_features=TEMPORAL_INPUT_FEATURES,
                                embedding_dim=128).eval()
        trajectory = _valid_trajectory()
        references = [embed(model, trajectory, "temporal")]
        return model, references, trajectory

    def test_unnormalized_trajectory_raises_instead_of_scoring(self):
        """
        The defect this gate exists for: raw device units used to produce a
        confident *wrong* verdict, not an error. Measured on real data — one
        writer's own signature scored 0.9370 normalized and 0.4888 in tablet
        units, so the system rejected a genuine user and said nothing was amiss.
        """
        model, references, trajectory = self._temporal_setup()
        raw = trajectory.copy()
        raw[:, 0] = raw[:, 0] * 1200 + 8000
        raw[:, 1] = raw[:, 1] * 1200 + 6000
        raw[:, 2] = raw[:, 2] * 1023

        with pytest.raises(ValueError, match="z-scored"):
            verify(temporal=(model, references, raw))

    def test_returns_the_gui_contract(self):
        model, references, path = self._spatial_setup()
        result = verify(spatial=(model, references, path))
        # `reasons` is additive: a list, empty on a clean multimodal verdict, and
        # carrying the gate's own message for every piece of evidence that was
        # dropped. `degraded: True` alone could not say *why*.
        assert set(result) == {"decision", "confidence", "routing", "degraded", "reasons"}
        assert isinstance(result["reasons"], list)
        os.unlink(path)

    def test_spatial_only_is_degraded(self):
        """A single modality is reduced evidence and must be flagged as such."""
        model, references, path = self._spatial_setup()
        result = verify(spatial=(model, references, path))
        assert result["routing"] == "spatial_only"
        assert result["degraded"] is True
        os.unlink(path)

    def test_identical_probe_scores_near_maximum(self):
        """Probing with the enrolled signature itself gives d≈0, so S=1/(1+d)≈1."""
        model, references, path = self._spatial_setup()
        result = verify(spatial=(model, references, path),
                        engine=FusionEngine(threshold=0.5))
        assert result["decision"] == "Genuine"
        assert result["confidence"] > 95.0
        os.unlink(path)

    def test_multimodal_routing(self):
        smodel, srefs, spath = self._spatial_setup()
        tmodel, trefs, traj = self._temporal_setup()
        result = verify(spatial=(smodel, srefs, spath), temporal=(tmodel, trefs, traj))
        assert result["routing"] == "multimodal"
        os.unlink(spath)

    def test_paired_digitizer_input_auto_renders_spatial_for_multimodal_routing(self):
        """When user provides temporal trajectory and spatial=(smodel, srefs, None),
        verify() auto-renders the trajectory to an image and routes as multimodal."""
        smodel, srefs, _ = self._spatial_setup()
        tmodel, trefs, traj = self._temporal_setup()
        result = verify(
            spatial=(smodel, srefs, None),
            temporal=(tmodel, trefs, traj),
            auto_render_spatial=True,
        )
        assert result["routing"] == "multimodal"
        assert result["decision"] in ("Genuine", "Forged")

    def test_failed_quality_gate_drops_the_spatial_branch(self):
        """
        A probe that fails the quality gate must be excluded from fusion, not
        merely flagged.

        This assertion was inverted deliberately: it previously required the
        gate-failing image to keep contributing to a `multimodal` verdict, which
        let a blurry or undersized capture carry the same weight as a clean one
        and flip the decision. The README's Capture Contract says the gate
        rejects such an image, so the route must collapse to the modality that
        still has evidence.
        """
        smodel, srefs, good_path = self._spatial_setup()
        tmodel, trefs, traj = self._temporal_setup()
        bad_path = _blank_image()

        clean = verify(spatial=(smodel, srefs, good_path), temporal=(tmodel, trefs, traj))
        assert clean["degraded"] is False
        assert clean["routing"] == "multimodal"

        gated = verify(spatial=(smodel, srefs, bad_path), temporal=(tmodel, trefs, traj))
        assert gated["routing"] == "temporal_only"
        assert gated["degraded"] is True

        os.unlink(good_path)
        os.unlink(bad_path)

    def test_failed_quality_gate_with_no_other_modality_raises(self):
        """
        Spatial-only verification of a gate-failing image has no evidence left,
        so it must raise rather than return a confident verdict built on nothing.
        """
        smodel, srefs, _ = self._spatial_setup()
        bad_path = _blank_image()

        with pytest.raises(ValueError, match="No usable evidence"):
            verify(spatial=(smodel, srefs, bad_path))

        os.unlink(bad_path)

    def test_nearest_reference_matching(self):
        """
        With several enrolled references the best match wins, so an unrelated
        extra reference cannot drag a genuine probe below threshold.
        """
        model, references, path = self._spatial_setup()
        noisy = np.random.default_rng(9).standard_normal(512).astype(np.float32) * 50
        result = verify(spatial=(model, references + [noisy], path),
                        engine=FusionEngine(threshold=0.5))
        assert result["decision"] == "Genuine"
        os.unlink(path)

    def test_no_modality_raises(self):
        with pytest.raises(ValueError):
            verify()

    def test_accepts_decoded_reference_bytes(self):
        """End-to-end DB path: stored bytes → embedding → verify, no reshaping by callers."""
        model, references, path = self._spatial_setup()
        stored = [embedding_to_bytes(r) for r in references]
        result = verify(spatial=(model, [bytes_to_embedding(b) for b in stored], path),
                        engine=FusionEngine(threshold=0.5))
        assert result["decision"] == "Genuine"
        os.unlink(path)

    def test_empty_reference_list_names_the_real_problem(self):
        """
        An identity with no enrolled templates used to surface as numpy's
        "need at least one array to concatenate", raised from inside
        `np.vstack` — a message about numpy's argument, not about the caller's
        empty enrolment. `SignatureService.verify()` truthy-checks its reference
        lists, so only the free function is exposed, and a GUI holding an empty
        row is exactly the caller that reaches it.
        """
        model, _, path = self._spatial_setup()

        with pytest.raises(ValueError, match="empty reference list"):
            verify(spatial=(model, [], path))

        os.unlink(path)


class TestSeveralSignings:
    """
    `VERIFICATION_SIGNINGS` > 1: the app asks for several signings of each kind and
    `verify()` scores their mean. Measured 2026-10-21 on BiosecureDS2 (140 writers,
    K = 5, references from one session): two signings took the pen EER from 16.06% to
    13.70% a session later and from 8.42% to 6.04% at the same sitting. At a fixed
    threshold averaging can only lower FAR, which is why the thresholds are shared.
    """

    def _temporal(self):
        model = TemporalSiamese(input_features=TEMPORAL_INPUT_FEATURES,
                                embedding_dim=128).eval()
        references = [embed(model, _valid_trajectory(seed=1), "temporal")]
        return model, references, _valid_trajectory(seed=2), _valid_trajectory(seed=3)

    def test_the_score_is_the_mean_of_the_signings(self):
        model, references, first, second = self._temporal()
        one = verify(temporal=(model, references, first))["confidence"]
        two = verify(temporal=(model, references, second))["confidence"]
        assert one != pytest.approx(two), "the two signings must score differently"

        both = verify(temporal=(model, references, [first, second]))
        assert both["confidence"] == pytest.approx((one + two) / 2)

    def test_one_signing_in_a_list_matches_a_single_probe(self):
        model, references, first, _ = self._temporal()
        assert (verify(temporal=(model, references, [first]))
                == verify(temporal=(model, references, first)))

    def test_a_failed_signing_drops_its_whole_branch(self):
        """
        Averaging only the signings that pass would let a forger hand in one
        careful forgery and one scribble and be scored on the forgery alone.
        """
        smodel = SpatialSiamese(embedding_dim=512).eval()
        good, bad = _sharp_signature_image(), _blank_image()
        srefs = [embed(smodel, good, "spatial")]
        tmodel, trefs, first, second = self._temporal()

        result = verify(spatial=(smodel, srefs, [good, bad]),
                        temporal=(tmodel, trefs, [first, second]))

        assert result["routing"] == "temporal_only"
        assert result["degraded"] is True
        assert any("signing 2" in r for r in result["reasons"])
        os.unlink(good)
        os.unlink(bad)

    def test_an_empty_list_of_signings_raises(self):
        model, references, _, _ = self._temporal()
        with pytest.raises(ValueError, match="at least one signing"):
            verify(temporal=(model, references, []))

    def test_rendered_spatial_probes_follow_several_trajectories(self):
        smodel = SpatialSiamese(embedding_dim=512).eval()
        path = _sharp_signature_image()
        srefs = [embed(smodel, path, "spatial")]
        tmodel, trefs, first, second = self._temporal()

        result = verify(spatial=(smodel, srefs, None),
                        temporal=(tmodel, trefs, [first, second]),
                        auto_render_spatial=True)

        assert result["routing"] == "multimodal"
        os.unlink(path)

    def test_the_default_is_one_signing(self):
        from config.settings import VERIFICATION_SIGNINGS
        assert VERIFICATION_SIGNINGS == 1


class TestStepUp:
    """
    Step-up verification (`VERIFICATION_STEP_UP`): one signing of each kind first. A fused
    score of at least STEP_UP_ACCEPT is accepted, one below STEP_UP_REJECT is rejected, and
    anything between asks for one more signing of each kind, whose mean with the first must
    reach STEP_UP_TWO. Measured on 132 BiosecurID-SONOF writers three sessions after
    enrolment: 11.4% of genuine signers refused at 5% of forgeries passed, against 15.7% for
    one signing; of those asked again, 36.5% of genuine signers pass and under 1% of forgers.
    """

    @staticmethod
    def _reference_at(probe_embedding, target):
        # S = 1 / (1 + d)  ->  d = 1/S - 1, placed on a single axis.
        offset = np.zeros_like(probe_embedding)
        offset[0] = 1.0 / target - 1.0
        return [probe_embedding + offset]

    def _setup(self, target):
        """Both branches score `target`, so the fused score is `target` as well."""
        smodel = SpatialSiamese(embedding_dim=512).eval()
        tmodel = TemporalSiamese(input_features=TEMPORAL_INPUT_FEATURES, embedding_dim=128).eval()
        path, trajectory = _sharp_signature_image(), _valid_trajectory(seed=2)
        spatial = (smodel, self._reference_at(embed(smodel, path, "spatial"), target), path)
        temporal = (tmodel, self._reference_at(embed(tmodel, trajectory, "temporal"), target), trajectory)
        return spatial, temporal

    def _verify(self, target, signings=1, step_up=True):
        (smodel, srefs, path), (tmodel, trefs, trajectory) = self._setup(target)
        image = path if signings == 1 else [path] * signings
        tablet = trajectory if signings == 1 else [trajectory] * signings
        try:
            return verify(spatial=(smodel, srefs, image), temporal=(tmodel, trefs, tablet),
                          step_up=step_up)
        finally:
            os.unlink(path)

    def test_a_first_signing_too_close_to_call_asks_for_another(self):
        from config.settings import STEP_UP_ACCEPT, STEP_UP_REJECT
        from src.verification import SIGN_AGAIN

        result = self._verify((STEP_UP_REJECT + STEP_UP_ACCEPT) / 2)

        assert result["decision"] == SIGN_AGAIN
        assert result["routing"] == "multimodal"
        assert any("one more signing" in r for r in result["reasons"])

    def test_a_clear_first_signing_is_decided_at_once(self):
        from config.settings import STEP_UP_ACCEPT, STEP_UP_REJECT

        assert self._verify(min(STEP_UP_ACCEPT + 0.02, 0.99))["decision"] == "Genuine"
        assert self._verify(STEP_UP_REJECT / 2)["decision"] == "Forged"

    def test_the_second_round_decides_the_mean_against_its_own_threshold(self):
        # The score sits above STEP_UP_TWO but below both other bars: the one band where the
        # first signing alone asks again and a plain two-signing verify() rejects. A literal
        # would leave the band whenever a threshold is re-set, and the test would pass anyway.
        from config.settings import DECISION_THRESHOLD, STEP_UP_ACCEPT, STEP_UP_REJECT, STEP_UP_TWO
        from src.verification import SIGN_AGAIN
        assert STEP_UP_REJECT < STEP_UP_TWO < min(STEP_UP_ACCEPT, DECISION_THRESHOLD)
        target = (STEP_UP_TWO + min(STEP_UP_ACCEPT, DECISION_THRESHOLD)) / 2

        assert self._verify(target)["decision"] == SIGN_AGAIN
        assert self._verify(target, signings=2)["decision"] == "Genuine"
        assert self._verify(target, signings=2, step_up=False)["decision"] == "Forged"
        assert self._verify((STEP_UP_REJECT + STEP_UP_TWO) / 2, signings=2)["decision"] == "Forged"

    def test_there_is_never_a_third_signing(self):
        from config.settings import STEP_UP_ACCEPT, STEP_UP_REJECT
        with pytest.raises(ValueError, match="third"):
            self._verify((STEP_UP_REJECT + STEP_UP_ACCEPT) / 2, signings=3)

    def test_paper_and_tablet_need_the_same_number_of_signings(self):
        (smodel, srefs, path), (tmodel, trefs, trajectory) = self._setup(0.6)
        try:
            with pytest.raises(ValueError, match="same number"):
                verify(spatial=(smodel, srefs, path), temporal=(tmodel, trefs, [trajectory, trajectory]),
                       step_up=True)
        finally:
            os.unlink(path)

    def test_a_single_branch_keeps_its_own_threshold(self):
        """The three thresholds were calibrated on fused scores only."""
        from config.settings import STEP_UP_ACCEPT, STEP_UP_REJECT
        (_, _, path), temporal = self._setup((STEP_UP_REJECT + STEP_UP_ACCEPT) / 2)
        os.unlink(path)

        result = verify(temporal=temporal, step_up=True)

        assert result["routing"] == "temporal_only"
        assert result["decision"] in ("Genuine", "Forged")

    def test_it_is_off_until_the_gui_handles_sign_again(self, monkeypatch):
        import src.verification as verification
        from config.settings import STEP_UP_ACCEPT, STEP_UP_REJECT, VERIFICATION_SIGNINGS, VERIFICATION_STEP_UP
        assert VERIFICATION_STEP_UP is False
        if VERIFICATION_STEP_UP:
            assert VERIFICATION_SIGNINGS == 1, "step-up starts from one signing of each kind"
        target = (STEP_UP_REJECT + STEP_UP_ACCEPT) / 2

        assert self._verify(target, step_up=None)["decision"] in ("Genuine", "Forged")
        monkeypatch.setattr(verification, "VERIFICATION_STEP_UP", True)
        assert self._verify(target, step_up=None)["decision"] == verification.SIGN_AGAIN


def _embedded_writer(genuine, forged):
    return {"genuine": np.asarray(genuine, dtype=np.float32),
            "forged": np.asarray(forged, dtype=np.float32)}


class TestWriterDependentClassifier:
    """
    A per-user decision boundary instead of one global distance threshold.

    This is the stage published systems have and this project did not. On the
    same features, SigNet-F reports 1.78% EER with a global threshold against
    0.4% with per-user thresholds, and every "99%+ on CEDAR" result in the
    literature is writer-dependent.

    The classifier lives entirely downstream of the Siamese encoder — it reads
    embeddings and never touches the model.
    """

    def test_separates_a_writer_from_random_others(self):
        rng = np.random.default_rng(0)
        references = rng.normal(0.0, 0.1, size=(6, 8)).astype(np.float32)
        negatives = rng.normal(5.0, 0.1, size=(30, 8)).astype(np.float32)

        classifier = fit_writer_classifier(references, negatives)

        assert classifier.decision_function(references).min() > 0
        assert classifier.decision_function(negatives).max() < 0

    def test_writers_own_forgeries_do_not_affect_the_fitted_boundary(self):
        """
        Skilled forgeries of the enrolled user are the attack being measured, so
        fitting on them leaks the evaluation — and the deployed system never
        holds a labelled forgery of a real customer, so a result that needed
        them could not ship.

        Tested by moving the forgeries somewhere completely different and
        asserting the *genuine* probe scores do not budge. Those scores depend
        only on the fitted boundary, so if a forgery had entered the fit as a
        negative the boundary would shift and they would change.
        """
        rng = np.random.default_rng(1)
        w1_genuine = rng.normal(0.0, 0.05, size=(9, 4)).astype(np.float32)
        w2 = _embedded_writer(rng.normal(9.0, 0.05, (9, 4)),
                              rng.normal(9.4, 0.05, (4, 4)))

        near = {"w1": _embedded_writer(w1_genuine, rng.normal(1.0, 0.05, (4, 4))),
                "w2": w2}
        far = {"w1": _embedded_writer(w1_genuine, rng.normal(-30.0, 0.05, (4, 4))),
               "w2": w2}

        near_scores, near_labels = writer_dependent_scores(near, num_references=5, seed=42)
        far_scores, far_labels = writer_dependent_scores(far, num_references=5, seed=42)

        assert np.allclose(near_scores[near_labels == 1], far_scores[far_labels == 1])

    def test_scores_are_finite_and_ordered(self):
        rng = np.random.default_rng(2)
        embedded = {
            f"w{i}": _embedded_writer(rng.normal(i * 10, 0.1, (9, 4)),
                                      rng.normal(i * 10 + 3, 0.1, (4, 4)))
            for i in range(4)
        }

        scores, labels = writer_dependent_scores(embedded, num_references=5, seed=42)

        assert np.all(np.isfinite(scores))
        assert set(labels.tolist()) == {0, 1}
        assert scores[labels == 1].mean() > scores[labels == 0].mean()

    def test_single_writer_dataset_is_skipped(self):
        """With no other writers there are no random-forgery negatives to fit on."""
        embedded = {"w1": _embedded_writer(np.zeros((9, 4)), np.ones((4, 4)))}

        scores, labels = writer_dependent_scores(embedded, num_references=5, seed=42)

        assert len(scores) == 0 and len(labels) == 0

    def test_writer_without_enough_genuines_is_skipped(self):
        """K references plus a probe, or the writer cannot contribute both arms."""
        rng = np.random.default_rng(3)
        embedded = {
            "short": _embedded_writer(rng.normal(0, 0.1, (5, 4)), rng.normal(1, 0.1, (4, 4))),
            "ok": _embedded_writer(rng.normal(9, 0.1, (9, 4)), rng.normal(10, 0.1, (4, 4))),
        }

        scores, labels = writer_dependent_scores(embedded, num_references=5, seed=42)

        # Only "ok" contributes: 4 genuine probes (9 - 5) and 4 forgeries.
        assert len(scores) == 8

    def test_is_reproducible_for_a_seed(self):
        rng = np.random.default_rng(4)
        embedded = {
            f"w{i}": _embedded_writer(rng.normal(i * 10, 0.1, (9, 4)),
                                      rng.normal(i * 10 + 3, 0.1, (4, 4)))
            for i in range(3)
        }

        a, _ = writer_dependent_scores(embedded, num_references=5, seed=7)
        b, _ = writer_dependent_scores(embedded, num_references=5, seed=7)

        assert np.allclose(a, b)


class TestThresholdFollowsRouting:
    """
    A branch dropped by its own gate must not change the surviving branch's verdict.

    `verify()` used to pick the decision threshold from *which arguments were
    passed*, before either gate ran. So a probe that ended up scored
    `spatial_only` was compared against the fused threshold (0.8196) whenever a
    temporal probe had merely been offered and rejected, instead of against
    SPATIAL_DECISION_THRESHOLD (0.6299). The fused bar is the highest of the
    three, so the error only ever pushed toward `Forged`: a genuine signer whose
    tablet timed out was rejected where the same image alone was accepted.
    """

    def _spatial_at_score(self, target_score):
        """A spatial setup whose nearest-reference score is exactly `target_score`."""
        model = SpatialSiamese(embedding_dim=512).eval()
        path = _sharp_signature_image()
        probe = embed(model, path, "spatial")
        # S = 1 / (1 + d)  ->  d = 1/S - 1, placed on a single axis.
        offset = np.zeros_like(probe)
        offset[0] = (1.0 / target_score) - 1.0
        return model, [probe + offset], path

    def _rejected_trajectory(self):
        """Raw tablet units — passes nothing, so the temporal branch is dropped."""
        raw = _valid_trajectory()
        raw[:, 0] = raw[:, 0] * 1200 + 8000     # back into raw tablet units
        raw[:, 1] = raw[:, 1] * 1200 + 6000
        return raw

    def test_dropped_temporal_branch_does_not_move_the_spatial_verdict(self):
        # The score sits halfway between SPATIAL_DECISION_THRESHOLD and
        # DECISION_THRESHOLD, the only band where the two disagree. A literal
        # (0.70 once) leaves the band whenever either threshold is re-set, and the
        # test then passes with the bug back in.
        from config.settings import DECISION_THRESHOLD, SPATIAL_DECISION_THRESHOLD
        assert SPATIAL_DECISION_THRESHOLD != DECISION_THRESHOLD
        model, references, path = self._spatial_at_score(
            (SPATIAL_DECISION_THRESHOLD + DECISION_THRESHOLD) / 2)
        temporal_model = TemporalSiamese(
            input_features=TEMPORAL_INPUT_FEATURES, embedding_dim=128).eval()
        dead_probe = self._rejected_trajectory()

        alone = verify(spatial=(model, references, path))
        offered = verify(spatial=(model, references, path),
                         temporal=(temporal_model, [np.zeros(128, np.float32)], dead_probe))

        assert alone["routing"] == offered["routing"] == "spatial_only"
        assert offered["decision"] == alone["decision"]
        assert offered["confidence"] == pytest.approx(alone["confidence"])

    def test_gate_failure_reason_reaches_the_caller(self):
        """`degraded: True` alone cannot tell the GUI *why* evidence was dropped."""
        model, references, path = self._spatial_at_score(0.70)
        temporal_model = TemporalSiamese(
            input_features=TEMPORAL_INPUT_FEATURES, embedding_dim=128).eval()

        result = verify(spatial=(model, references, path),
                        temporal=(temporal_model, [np.zeros(128, np.float32)],
                                  self._rejected_trajectory()))

        assert result["degraded"] is True
        assert result["reasons"], "a degraded verdict must say what was dropped"
        assert any("temporal" in r for r in result["reasons"])


class TestShippedChannelContract:
    """
    A probe of the configured width verifies end to end.

    `TEMPORAL_CHANNELS` is the single knob: `xyp` -> 3, `dt` -> 4 (what ships
    today, and what `models/weights/temporal_lstm.pth` was trained as), `tilt`
    -> 5. This test asserts the shipped setting round-trips, whichever it is.
    """

    def _setup(self):
        model = TemporalSiamese(input_features=TEMPORAL_INPUT_FEATURES,
                                embedding_dim=128).eval()
        trajectory = _valid_trajectory(length=60, seed=7)
        return model, [embed(model, trajectory, "temporal")], trajectory

    def test_probe_of_the_configured_width_verifies_end_to_end(self):
        model, references, trajectory = self._setup()
        result = verify(temporal=(model, references, trajectory))
        assert result["decision"] in ("Genuine", "Forged")
        assert result["routing"] == "temporal_only"


class TestSpatialArrayInputsAreCanonicalized:
    """
    An array reaching the spatial branch must be treated like the path it stands
    in for.

    `SpatialSiamese` pools adaptively, so a wrong-sized, un-inverted uint8 scan
    produces a perfectly plausible 512-d embedding and raises nothing — and
    enrolment writes that template to the database. Polarity inversion alone is
    recorded as worth 4.86% -> 0.00% EER on CEDAR, so the failure is silent and
    expensive. Shape and dtype are therefore gated here rather than trusted.
    """

    def _model(self):
        from config.settings import SPATIAL_WEIGHTS_PATH
        return load_model("spatial", SPATIAL_WEIGHTS_PATH)

    def test_raw_scan_array_matches_the_same_scan_on_disk(self):
        """
        The array and the path are the same signature, so they must produce the
        same embedding. Before this gate the array skipped both the resize and
        the inversion, and the two disagreed completely.
        """
        path = _sharp_signature_image()
        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        model = self._model()
        from_path = embed(model, path, "spatial")
        from_array = embed(model, array, "spatial")

        np.testing.assert_allclose(from_path, from_array, rtol=1e-5, atol=1e-6)

    def test_rendered_trajectory_is_not_inverted_a_second_time(self):
        """
        The renderer already emits the canonical contract. Sending it through
        `preprocess_image` would invert it again and hand the model a negative.
        """
        from config.settings import SPATIAL_INPUT_SIZE
        from src.verification import _as_spatial_input

        width, height = SPATIAL_INPUT_SIZE
        rendered = np.zeros((1, height, width), dtype=np.float32)
        rendered[0, 70:80, 40:180] = 1.0

        out = _as_spatial_input(rendered)
        np.testing.assert_array_equal(out, rendered)

    def test_canonical_shape_with_scan_range_is_refused(self):
        """
        The shape says 'already canonical' while the values say 'raw scan'. That
        combination is exactly what would sail through unscaled, so it must
        raise rather than be guessed at.
        """
        from config.settings import SPATIAL_INPUT_SIZE
        from src.verification import _as_spatial_input

        width, height = SPATIAL_INPUT_SIZE
        raw = np.full((1, height, width), 255.0, dtype=np.float32)

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            _as_spatial_input(raw)

    def test_uninterpretable_shape_raises(self):
        from src.verification import _as_spatial_input

        with pytest.raises(ValueError, match="Cannot interpret"):
            _as_spatial_input(np.zeros((3, 64, 64), dtype=np.float32))

    def test_canonical_array_carrying_nan_is_refused(self):
        """
        The range check cannot see a NaN: every comparison against NaN is False,
        so `low < -1e-6 or high > 1.0 + 1e-6` was False for a poisoned array and
        it passed straight through. `embed()` then returned a 512-d embedding of
        NaN, which enrolment stores and which every later distance against it
        reports as NaN — a template that can never match anything, silently.
        `trajectory_gate` has always checked this explicitly; the spatial path
        did not.
        """
        from config.settings import SPATIAL_INPUT_SIZE
        from src.verification import _as_spatial_input

        width, height = SPATIAL_INPUT_SIZE
        poisoned = np.zeros((1, height, width), dtype=np.float32)
        poisoned[0, height // 2, width // 2] = np.nan

        with pytest.raises(ValueError, match="NaN or infinity"):
            _as_spatial_input(poisoned)

        with pytest.raises(ValueError, match="NaN or infinity"):
            embed(SpatialSiamese(embedding_dim=512).eval(), poisoned, "spatial")

    def test_scan_array_carrying_a_non_finite_pixel_is_refused(self):
        """
        The scan branch is reachable with a float array too, and it loses the
        poison rather than propagating it: `preprocess_image` casts through
        `np.clip(...).astype(np.uint8)`, which turns NaN into an arbitrary byte
        and infinity into 255. Either way the embedding looks perfectly ordinary
        and nothing says the pixel was never a measurement.
        """
        from config.settings import MIN_BBOX_PIXELS
        from src.verification import _as_spatial_input

        scan = np.full((MIN_BBOX_PIXELS * 2, MIN_BBOX_PIXELS * 2), 255.0, dtype=np.float32)
        scan[0, 0] = np.inf

        with pytest.raises(ValueError, match="NaN or infinity"):
            _as_spatial_input(scan)
