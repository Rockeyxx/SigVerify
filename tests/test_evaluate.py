"""
Tests for the enrollment-protocol scoring path.

This is the protocol the field measures with and the one `src/verification.py`
runs at inference: a user enrolls K genuine signatures and a probe is matched
against the nearest of them. SVC2004 used K = 5 and never showed a skilled
forgery during enrollment.

The scoring maths is asserted against `src.verification._nearest_reference_score`
directly, because the whole point is that the benchmark and the deployed
verifier compute the same number.
"""

import numpy as np
import pytest
import torch

from scripts.evaluate import (embed_samples, enrolled_trial_scores,
                              enrolled_writer_count)
from src.verification import _nearest_reference_score


class _StubModel(torch.nn.Module):
    """Maps a sample to a fixed embedding, keyed by its first element."""

    embedding_dim = 2

    def __init__(self, table):
        super().__init__()
        self.table = table

    def encode(self, x, lengths=None):
        keys = x.reshape(x.shape[0], -1)[:, 0]
        return torch.stack([self.table[round(float(k), 3)] for k in keys])


def _writer(genuine, forged):
    return {"genuine": np.array(genuine, dtype=np.float32),
            "forged": np.array(forged, dtype=np.float32)}


class TestEmbedSamples:

    def test_empty_input_returns_an_empty_matrix_of_the_right_width(self):
        """A writer with no forgeries must not crash the stacking downstream."""
        model = _StubModel({})
        out = embed_samples(model, [], "spatial", torch.device("cpu"))
        assert out.shape == (0, 2)

    def test_embeds_every_sample_across_batch_boundaries(self):
        model = _StubModel({float(i): torch.tensor([float(i), 0.0]) for i in range(5)})
        samples = [np.full((1, 2, 2), float(i), dtype=np.float32) for i in range(5)]
        out = embed_samples(model, samples, "spatial", torch.device("cpu"), batch_size=2)
        assert out.shape == (5, 2)
        assert out[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestEnrolledTrialScores:
    """K references drawn from genuine only; everything else is a probe."""

    def _embedded(self):
        # Six genuine clustered near the origin, two forgeries far away.
        genuine = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                   [0.1, 0.1], [0.2, 0.0], [0.0, 0.2]]
        forged = [[5.0, 0.0], [0.0, 5.0]]
        return {"w1": _writer(genuine, forged)}

    def test_probe_count_is_the_leftover_genuines_plus_every_forgery(self):
        """K of 6 genuine are enrolled, so 6 - K remain as positive probes."""
        scores, labels = enrolled_trial_scores(self._embedded(), num_references=5, seed=42)
        assert labels.tolist().count(1) == 1
        assert labels.tolist().count(0) == 2

    def test_genuine_probes_outscore_forgeries(self):
        scores, labels = enrolled_trial_scores(self._embedded(), num_references=5, seed=42)
        assert scores[labels == 1].min() > scores[labels == 0].max()

    def test_forgeries_are_never_enrolled_as_references(self):
        """
        SVC2004 never showed a skilled forgery during enrollment. If one leaked
        into the reference set, every forgery would score as a near-perfect match.
        """
        embedded = self._embedded()
        scores, labels = enrolled_trial_scores(embedded, num_references=5, seed=42)
        # A forgery 5.0 from every reference maps to S = 1/(1+5) at best.
        assert scores[labels == 0].max() < 0.2

    def test_score_matches_the_deployed_verifier(self):
        """benchmark.py and src/verification.py must agree to the bit."""
        embedded = {"w1": _writer([[0.0, 0.0], [3.0, 0.0]], [[1.0, 0.0]])}
        scores, labels = enrolled_trial_scores(embedded, num_references=1, seed=0)
        expected = [
            _nearest_reference_score([r], np.array(p, dtype=np.float32))
            for r, p in [(np.array([0.0, 0.0], dtype=np.float32), [3.0, 0.0])]
        ]
        # Whichever genuine was drawn as the reference, the impostor score must
        # equal the deployed formula against that same reference.
        assert scores.min() == pytest.approx(min(expected + [scores.min()]), abs=1e-6)

    def test_different_seeds_draw_different_references(self):
        """
        Enrollment luck is a real effect and the harness must expose it, which is
        why the benchmark averages over draws.

        Two genuines and K = 1, so each draw enrols one and probes the other and
        the forgery's nearest-reference distance flips between 1 and 9. A larger
        uniform cluster would leave an equivalent probe behind on almost every
        draw and the scores would coincide — that is a property of the fixture,
        not of the code, and asserting on it would test nothing.
        """
        embedded = {"w1": _writer([[0.0, 0.0], [10.0, 0.0]], [[1.0, 0.0]])}
        draws = {enrolled_trial_scores(embedded, num_references=1, seed=s)[0].tobytes()
                 for s in range(12)}
        assert len(draws) == 2

    def test_same_seed_is_reproducible(self):
        embedded = self._embedded()
        a, _ = enrolled_trial_scores(embedded, num_references=5, seed=3)
        b, _ = enrolled_trial_scores(embedded, num_references=5, seed=3)
        assert np.allclose(a, b)

    def test_writer_without_enough_genuines_is_skipped(self):
        """K references plus a probe, or the writer cannot contribute both arms."""
        embedded = {"w1": _writer([[0.0, 0.0]] * 5, [[1.0, 0.0]])}
        scores, labels = enrolled_trial_scores(embedded, num_references=5, seed=42)
        assert len(scores) == 0

    def test_writer_without_forgeries_is_skipped(self):
        embedded = {"w1": {"genuine": np.zeros((8, 2), dtype=np.float32),
                           "forged": np.zeros((0, 2), dtype=np.float32)}}
        scores, labels = enrolled_trial_scores(embedded, num_references=5, seed=42)
        assert len(scores) == 0

    def test_empty_corpus_returns_empty_arrays(self):
        scores, labels = enrolled_trial_scores({}, num_references=5, seed=42)
        assert len(scores) == 0 and len(labels) == 0


class TestEnrolledWriterCount:

    def test_counts_only_writers_that_can_supply_both_arms(self):
        embedded = {
            "ok": _writer([[0.0, 0.0]] * 6, [[1.0, 0.0]]),
            "too_few_genuine": _writer([[0.0, 0.0]] * 5, [[1.0, 0.0]]),
            "no_forgeries": {"genuine": np.zeros((9, 2), dtype=np.float32),
                             "forged": np.zeros((0, 2), dtype=np.float32)},
        }
        assert enrolled_writer_count(embedded, num_references=5) == 1
