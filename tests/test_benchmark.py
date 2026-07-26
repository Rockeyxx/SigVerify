"""Tests for the EER regression harness (scripts/benchmark.py).

The harness exists so model changes are judged on separation quality (EER and
Cohen's d over held-out writers) rather than on FAR/FRR at a fixed threshold —
reading FAR/FRR at a fixed tau is what masked the P-001 accuracy failure.
"""

import numpy as np
import pytest

from scripts.benchmark import cohens_d, corpus_of, select_test_writers, separation_report


class TestCohensD:
    """Effect size between the genuine and impostor distance distributions."""

    def test_identical_distributions_give_zero(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0.5, 0.1, 500)
        y = rng.normal(0.5, 0.1, 500)
        assert cohens_d(y, x) == pytest.approx(0.0, abs=0.2)

    def test_sign_is_positive_when_impostors_are_farther(self):
        """Impostor distances should exceed genuine ones, so d > 0 for a working model."""
        genuine = np.full(100, 0.2)
        impostor = np.full(100, 0.8)
        # Zero variance in both arms -> guard against divide-by-zero, not NaN.
        assert cohens_d(impostor, genuine) > 0

    def test_known_separation(self):
        """Two unit-variance arms one apart must give exactly d = 1.0."""
        rng = np.random.RandomState(42)
        genuine = rng.normal(0.0, 1.0, 200_000)
        impostor = rng.normal(1.0, 1.0, 200_000)
        assert cohens_d(impostor, genuine) == pytest.approx(1.0, abs=0.02)

    def test_pooled_std_uses_both_arms(self):
        """A wide impostor arm must lower the effect size versus a narrow one."""
        rng = np.random.RandomState(1)
        genuine = rng.normal(0.0, 1.0, 50_000)
        narrow = rng.normal(1.0, 1.0, 50_000)
        wide = rng.normal(1.0, 3.0, 50_000)
        assert cohens_d(narrow, genuine) > cohens_d(wide, genuine)


class TestSeparationReport:
    """The report bundle the harness prints and returns."""

    @staticmethod
    def _scores(genuine_d, impostor_d):
        """Build (scores, labels) from raw distances via S = 1/(1+d)."""
        d = np.concatenate([genuine_d, impostor_d])
        labels = np.concatenate([np.ones(len(genuine_d)), np.zeros(len(impostor_d))])
        return 1.0 / (1.0 + d), labels

    def test_reports_every_required_field(self):
        rng = np.random.RandomState(0)
        scores, labels = self._scores(rng.uniform(0, 0.4, 200), rng.uniform(0.6, 1.5, 200))
        r = separation_report(scores, labels, margin=1.0)
        for key in ("eer", "threshold", "cohens_d", "genuine_mean_distance",
                    "impostor_mean_distance", "impostors_inside_margin",
                    "num_genuine", "num_impostor"):
            assert key in r, f"missing {key}"

    def test_perfect_separation_gives_zero_eer(self):
        scores, labels = self._scores(np.full(100, 0.01), np.full(100, 5.0))
        assert separation_report(scores, labels, margin=1.0)["eer"] == pytest.approx(0.0, abs=1e-6)

    def test_indistinguishable_arms_give_chance_eer(self):
        rng = np.random.RandomState(3)
        d = rng.uniform(0, 1, 2000)
        scores, labels = self._scores(d[:1000], d[1000:])
        assert separation_report(scores, labels, margin=1.0)["eer"] == pytest.approx(0.5, abs=0.06)

    def test_counts_impostors_inside_the_margin(self):
        """Impostors with d < margin are the ones a 0.5 threshold wrongly accepts."""
        scores, labels = self._scores(np.full(10, 0.1), np.array([0.2] * 3 + [2.0] * 7))
        r = separation_report(scores, labels, margin=1.0)
        assert r["impostors_inside_margin"] == pytest.approx(0.3)

    def test_distance_means_invert_the_similarity_mapping(self):
        """The report must undo S = 1/(1+d) so distances are comparable to the margin."""
        scores, labels = self._scores(np.full(50, 0.25), np.full(50, 1.5))
        r = separation_report(scores, labels, margin=1.0)
        assert r["genuine_mean_distance"] == pytest.approx(0.25, abs=1e-6)
        assert r["impostor_mean_distance"] == pytest.approx(1.5, abs=1e-6)

    def test_empty_arm_does_not_raise(self):
        """A degenerate split must report NaN, not explode mid-benchmark."""
        scores, labels = self._scores(np.full(10, 0.2), np.array([]))
        r = separation_report(scores, labels, margin=1.0)
        assert r["num_impostor"] == 0
        assert np.isnan(r["cohens_d"])


class TestCorpusOf:
    """Writer IDs carry a script or language only when it is worth splitting on."""

    def test_language_scoped_corpora_keep_their_middle_token(self):
        assert corpus_of("bhsig_Bengali_001") == "bhsig_Bengali"
        assert corpus_of("sigcomp_Dutch_013") == "sigcomp_Dutch"
        assert corpus_of("sigcomp_Chinese_011") == "sigcomp_Chinese"

    def test_numeric_writer_ids_collapse_to_the_corpus_name(self):
        assert corpus_of("svc_11") == "svc"
        assert corpus_of("cedar_7") == "cedar"
        assert corpus_of("mobisig_3") == "mobisig"


class TestSelectTestWriters:
    """
    Comparing two models whose corpus lists differ means scoring them on the
    writers both held out — and refusing anything else.
    """

    SPLIT = (["svc_1", "svc_2"], ["svc_3"], ["svc_4", "svc_5"])

    def test_keeps_only_the_requested_held_out_writers(self):
        assert select_test_writers(["svc_4"], *self.SPLIT) == ["svc_4"]

    def test_validation_writers_are_allowed(self):
        """They are outside training, which is the property that matters."""
        assert select_test_writers(["svc_3", "svc_5"], *self.SPLIT) == ["svc_3", "svc_5"]

    def test_training_writer_is_refused(self):
        """Silently scoring a memorized writer is the failure this prevents."""
        with pytest.raises(ValueError, match="trained on"):
            select_test_writers(["svc_4", "svc_1"], *self.SPLIT)

    def test_unknown_writer_is_refused(self):
        with pytest.raises(ValueError, match="Unknown writer IDs"):
            select_test_writers(["svc_99"], *self.SPLIT)

    def test_duplicates_are_collapsed(self):
        assert select_test_writers(["svc_4", "svc_4"], *self.SPLIT) == ["svc_4"]
