"""
Tests for the cold-start (cross-corpus) evaluation harness.

Two properties are gated here, and neither is arithmetic.

The first is that the harness scores through `verify()`. That is the entire
reason this tool exists rather than another `benchmark.py` flag: `trajectory_gate`
and the routing-dependent threshold selection are reachable *only* from
`verify()`, so a harness that embedded and compared scores itself would report a
number for a code path the deployment does not run. Three serving defects
were invisible to the suite for exactly that reason.

The second is that tau is never fitted on the writers it is then scored on. A
constant selected on the scores it is quoted against is optimistically biased
and, in this repo's own words, "a lower bound on nothing" -- the mistake that
produced a retracted "+8.87 point" fusion result.

Neither property is visible in the numbers the tool prints, which is why both are
asserted structurally instead.
"""
import os

import numpy as np
import pytest

from tools import cold_start_eval

NFI_SCREEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "sigcomp2009_clean", "clean_writers.txt")


def _writer(num_genuine=8, num_forged=4, dim=6, seed=0, num_models=1):
    """
    One writer's worth of embeddings, with placeholder raw samples.

    The embedding slot holds one matrix *per model*, so a single checkpoint is
    the one-element case and an ensemble is the same shape with more entries.
    """
    rng = np.random.default_rng(seed)
    return {
        "genuine": ([f"g{i}" for i in range(num_genuine)],
                    [rng.normal(size=(num_genuine, dim)).astype(np.float32)
                     for _ in range(num_models)]),
        "forged": ([f"f{i}" for i in range(num_forged)],
                   [rng.normal(size=(num_forged, dim)).astype(np.float32)
                    for _ in range(num_models)]),
    }


def _corpus(num_writers=6, **kwargs):
    return {f"w{i}": _writer(seed=i, **kwargs) for i in range(num_writers)}


class TestScoringGoesThroughVerify:
    """The deployed call must be the one that produces the reported number."""

    def test_every_probe_is_scored_by_verify(self, monkeypatch):
        calls = []

        def spy(**kwargs):
            calls.append(kwargs)
            return {"decision": "Genuine", "confidence": 50.0,
                    "routing": "spatial_only", "degraded": True}

        monkeypatch.setattr(cold_start_eval, "verify", spy)
        embedded = _corpus(num_writers=2)

        scores, truths, accepted = cold_start_eval.shipped_operating_point(
            model=None, embedded=embedded, branch="spatial",
            num_references=5, seed=42)

        # 2 writers x (8 genuine - 5 enrolled = 3 probes, + 4 forgeries) = 14.
        # A harness that scored inline would leave `calls` empty while still
        # returning 14 plausible scores, which is the failure being gated.
        assert len(calls) == 14
        assert len(scores) == len(truths) == len(accepted) == 14
        assert truths.sum() == 6

    def test_references_reach_verify_as_decoded_templates(self, monkeypatch):
        """
        Enrolment stores bytes, so the harness must hand `verify()` what the
        database would hand it -- arrays read back out of that blob, not the
        live embeddings it happens to still be holding.
        """
        calls = []
        monkeypatch.setattr(cold_start_eval, "verify",
                            lambda **kw: calls.append(kw) or
                            {"decision": "Forged", "confidence": 10.0,
                             "routing": "spatial_only", "degraded": True})

        cold_start_eval.shipped_operating_point(
            model=None, embedded=_corpus(num_writers=1), branch="spatial",
            num_references=5, seed=42)

        _, references, _ = calls[0]["spatial"]
        assert len(references) == 5
        assert all(isinstance(r, np.ndarray) for r in references)
        assert all(r.dtype == np.float32 for r in references)

    def test_branch_selects_the_verify_keyword(self, monkeypatch):
        """A temporal run must not be scored through the spatial argument."""
        calls = []
        monkeypatch.setattr(cold_start_eval, "verify",
                            lambda **kw: calls.append(kw) or
                            {"decision": "Genuine", "confidence": 90.0,
                             "routing": "temporal_only", "degraded": True})

        cold_start_eval.shipped_operating_point(
            model=None, embedded=_corpus(num_writers=1), branch="temporal",
            num_references=5, seed=42)

        assert all(set(kw) == {"temporal"} for kw in calls)


class TestCalibrationIsWriterDisjoint:
    """Tau must never be fitted on a writer it is then scored against."""

    def test_fit_and_test_writers_never_overlap(self, monkeypatch):
        seen = []
        real_pool = cold_start_eval.pool

        def spy(per, writers):
            seen.append(list(writers))
            return real_pool(per, writers)

        monkeypatch.setattr(cold_start_eval, "pool", spy)
        cold_start_eval.writer_disjoint(_corpus(num_writers=6), num_references=5,
                                        repeats=3, base_seed=42, far_target=0.05,
                                        shipped_tau=0.63)

        # `pool` is called once for the fit half and once for the test half of
        # every split, in that order.
        assert len(seen) == 6
        for fit, test in zip(seen[::2], seen[1::2]):
            assert fit and test
            assert not set(fit) & set(test)

    def test_too_few_writers_raises_rather_than_degenerating(self):
        """
        Three writers cannot be split two-and-two. Returning a one-writer fit
        set would still produce a tau and a plausible HTER, so this has to be an
        error rather than a quiet approximation.
        """
        with pytest.raises(ValueError, match="writer-disjoint split"):
            cold_start_eval.writer_disjoint(_corpus(num_writers=3), num_references=5,
                                            repeats=1, base_seed=42, far_target=0.05,
                                            shipped_tau=0.63)


class TestFarConstrainedThreshold:
    """The FAR target is a property of the impostor distribution alone."""

    def test_threshold_cannot_read_genuine_scores(self):
        """
        `threshold_for_far` takes only impostor scores. If it ever grew a
        genuine-side dependency the FAR guarantee would stop being a guarantee,
        so the signature itself is the gate.
        """
        impostor = np.linspace(0.0, 1.0, 101)
        assert cold_start_eval.threshold_for_far(impostor, 0.05) == pytest.approx(0.95, abs=0.02)

    def test_tighter_far_target_demands_a_higher_threshold(self):
        impostor = np.linspace(0.0, 1.0, 101)
        strict = cold_start_eval.threshold_for_far(impostor, 0.01)
        loose = cold_start_eval.threshold_for_far(impostor, 0.20)
        assert strict > loose

    def test_threshold_actually_holds_the_target_on_its_own_scores(self):
        rng = np.random.default_rng(0)
        impostor = rng.normal(loc=0.4, scale=0.1, size=5000)
        tau = cold_start_eval.threshold_for_far(impostor, 0.05)
        assert float(np.mean(impostor >= tau)) == pytest.approx(0.05, abs=0.01)


class TestEnsembleScoring:
    """An ensemble averages member scores over one shared reference draw."""

    def test_repeating_one_model_changes_nothing(self):
        """
        Averaging a model with copies of itself must reproduce it exactly. If
        the ensemble path also re-drew references per member it would average
        away enrolment luck too, and this arm would beat the single-model arm
        for a reason that has nothing to do with ensembling.
        """
        single = {"w0": _writer(seed=0, num_models=1)}
        tripled = {"w0": _writer(seed=0, num_models=1)}
        tripled["w0"]["genuine"] = (tripled["w0"]["genuine"][0],
                                    tripled["w0"]["genuine"][1] * 3)
        tripled["w0"]["forged"] = (tripled["w0"]["forged"][0],
                                   tripled["w0"]["forged"][1] * 3)

        one = cold_start_eval.scores_per_writer(single, 5, seed=42)
        three = cold_start_eval.scores_per_writer(tripled, 5, seed=42)

        np.testing.assert_allclose(one["w0"][0], three["w0"][0])
        np.testing.assert_allclose(one["w0"][1], three["w0"][1])

    def test_members_disagreeing_moves_the_score(self):
        """A genuinely different member must change the averaged score."""
        single = {"w0": _writer(seed=0, num_models=1)}
        pair = {"w0": _writer(seed=0, num_models=2)}

        one = cold_start_eval.scores_per_writer(single, 5, seed=42)
        two = cold_start_eval.scores_per_writer(pair, 5, seed=42)

        assert not np.allclose(one["w0"][0], two["w0"][0])


class TestCorpusSelection:

    @pytest.mark.skipif(not os.path.exists(NFI_SCREEN),
                        reason="SigComp2009 screen list not present (gitignored data)")
    def test_nfi2009_corpus_reaches_its_parser(self, monkeypatch):
        """
        `--corpus nfi2009` read SIGCOMP2009_DIR without importing it, so the option
        died with a NameError before a single file was parsed.
        """
        from config.settings import SIGCOMP2009_DIR

        seen = {}
        monkeypatch.setattr(cold_start_eval, "parse_nfi2009",
                            lambda root, allowed: seen.update(root=root) or {})
        cold_start_eval.load_corpus("spatial", "nfi2009")
        assert seen["root"] == SIGCOMP2009_DIR
