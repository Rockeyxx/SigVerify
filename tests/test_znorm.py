"""
Tests for writer-level score normalization (Z-norm).

Scoring is `S = 1/(1+d)` against one global threshold for every writer in every
corpus. That asks every signer to be equally self-consistent. They are not, and
the per-dataset spread the branch shows (4.24% on sigcomp_Dutch against 16.32%
on svc2004) is exactly what a single global threshold handles badly.

The enrolled reference set already measures a writer's own scale for free: the
distances of the K genuine references *to each other*. Z-norm divides by that,
so a naturally variable signer stops being judged on the same absolute scale as
a naturally precise one.

This learns no decision boundary — only a scale. That distinction matters,
because the per-user SVM tried earlier made EER 2-3x worse; it failed because
random-forgery negatives cannot teach a skilled-forgery boundary. Z-norm never
sees a negative, so that failure does not carry over.
"""
import pytest

# Closed experiment: skipped by default, run in CI (see pytest.ini).
pytestmark = pytest.mark.archived


import numpy as np

from scripts.evaluate import enrolled_trial_scores, znorm_trial_scores


def _writer(genuine, forged):
    return {"genuine": np.array(genuine, dtype=np.float32),
            "forged": np.array(forged, dtype=np.float32)}


def _cluster(centre, spread, count, seed):
    rng = np.random.default_rng(seed)
    return (np.array(centre) + spread * rng.standard_normal((count, 2))).tolist()


class TestZnormTrialScores:

    def _embedded(self):
        genuine = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                   [0.1, 0.1], [0.2, 0.0], [0.0, 0.2]]
        forged = [[5.0, 0.0], [0.0, 5.0]]
        return {"w1": _writer(genuine, forged)}

    def test_draws_the_same_probes_as_the_unnormalized_protocol(self):
        """
        Only the metric changes. If the label vector differs, the draw changed
        too and the comparison against the baseline is not like-for-like.
        """
        embedded = self._embedded()
        _, baseline = enrolled_trial_scores(embedded, num_references=5, seed=42)
        _, normalized = znorm_trial_scores(embedded, num_references=5, seed=42)
        assert normalized.tolist() == baseline.tolist()

    def test_preserves_ranking_within_one_writer(self):
        """
        Dividing by a single positive per-writer constant is monotone, so a
        writer's own genuine/forgery ordering must be untouched. Z-norm can only
        help by making *different* writers comparable.
        """
        embedded = self._embedded()
        base, labels = enrolled_trial_scores(embedded, num_references=5, seed=42)
        norm, _ = znorm_trial_scores(embedded, num_references=5, seed=42)
        assert np.array_equal(np.argsort(base), np.argsort(norm))
        assert norm[labels == 1].min() > norm[labels == 0].max()

    def test_makes_two_writers_of_different_scale_comparable(self):
        """
        The point of the whole exercise. Two writers with identical geometry,
        one signing ten times larger: unnormalized, the loose writer's genuine
        probes look worse than the tight writer's forgeries, so no single
        threshold separates both. Z-norm removes the scale.
        """
        # The tight writer's *forgeries* (d ~ 0.6) sit closer to their references
        # than the loose writer's own *genuines* (d ~ 2) sit to theirs. No single
        # absolute threshold can separate both writers; that is the whole point.
        tight = _writer(_cluster([0, 0], 0.05, 6, 1), _cluster([0.6, 0], 0.05, 3, 2))
        loose = _writer(_cluster([0, 0], 1.5, 6, 3), _cluster([20, 0], 1.5, 3, 4))
        embedded = {"tight": tight, "loose": loose}

        base, labels = enrolled_trial_scores(embedded, num_references=5, seed=7)
        norm, _ = znorm_trial_scores(embedded, num_references=5, seed=7)

        base_gap = base[labels == 1].min() - base[labels == 0].max()
        norm_gap = norm[labels == 1].min() - norm[labels == 0].max()
        assert base_gap < 0, "fixture must be one the unnormalized metric cannot separate"
        assert norm_gap > base_gap

    def test_identical_references_do_not_produce_nan(self):
        """Zero spread is a division by zero — a writer who signs identically twice."""
        embedded = {"w1": _writer([[0.0, 0.0]] * 6, [[5.0, 0.0], [0.0, 5.0]])}
        scores, _ = znorm_trial_scores(embedded, num_references=5, seed=42)
        assert np.all(np.isfinite(scores))

    def test_a_writer_short_of_references_is_skipped_not_halved(self):
        embedded = {"w1": _writer([[0.0, 0.0], [0.1, 0.0]], [[5.0, 0.0]])}
        scores, labels = znorm_trial_scores(embedded, num_references=5, seed=42)
        assert scores.size == 0 and labels.size == 0

    def test_is_reproducible_for_a_fixed_seed(self):
        embedded = self._embedded()
        a, _ = znorm_trial_scores(embedded, num_references=5, seed=3)
        b, _ = znorm_trial_scores(embedded, num_references=5, seed=3)
        assert np.array_equal(a, b)
