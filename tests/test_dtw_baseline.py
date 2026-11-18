"""
Tests for the DTW diagnostic baseline.

DTW is not shipped and not fused (AGENTS.md) — it exists to answer one
question: is the temporal branch's SVC2004 gap a *feature* problem or an
*alignment* problem. That makes two properties load-bearing.

1. **DTW must actually warp.** A signature traced at a different speed is the
   same signature. If the implementation does not collapse that difference to
   zero it is measuring Euclidean distance with extra steps, and every
   conclusion drawn from it about alignment is void.
2. **It must be scored under the same enrollment draw as the LSTM.** The whole
   point is a like-for-like comparison, so the reference selection has to come
   from `enrolled_trial_scores` itself rather than a reimplementation that can
   drift away from it.
"""

import numpy as np
import pytest

import torch

from scripts.evaluate import enrolled_trial_scores
from tools.research.dtw_baseline import (dtw_distance, dtw_distance_batch,
                                precomputed_distances, trajectory_writers)


def _stroke(length, phase=0.0):
    """A smooth 2-channel path — stands in for a normalized (X, Y) trajectory."""
    t = np.linspace(0.0, 2.0 * np.pi, length)
    return np.stack([np.sin(t + phase), np.cos(t + phase)], axis=1).astype(np.float32)


class TestDtwDistance:

    def test_a_sequence_against_itself_is_zero(self):
        a = _stroke(40)
        assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_a_time_warped_copy_is_zero(self):
        """
        The defining property. Holding each frame for two samples is the same
        path drawn at half speed — DTW must warp that away entirely.
        """
        a = _stroke(40)
        slowed = np.repeat(a, 2, axis=0)
        assert dtw_distance(a, slowed) == pytest.approx(0.0, abs=1e-6)

    def test_a_time_warped_copy_is_far_under_frame_wise_comparison(self):
        """
        Guards against an implementation that silently resamples to a common
        length and takes a straight distance: that would also score ~0 here, so
        this test pins that the *stretched* pair is genuinely misaligned
        frame-to-frame and DTW's zero is the warping, not a coincidence.
        """
        a = _stroke(40)
        truncated = np.repeat(a, 2, axis=0)[:40]
        assert float(np.linalg.norm(a - truncated, axis=1).mean()) > 0.5

    def test_is_symmetric(self):
        a, b = _stroke(30), _stroke(45, phase=0.7)
        assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a), rel=1e-9)

    def test_different_paths_are_positive(self):
        a, b = _stroke(30), _stroke(30, phase=np.pi)
        assert dtw_distance(a, b) > 0.1

    def test_length_does_not_inflate_the_distance(self):
        """
        Normalization check. An unnormalized DP sum grows with path length, so
        a long genuine pair would score worse than a short impostor pair and
        the EER would partly measure sequence length. Two equally-mismatched
        pairs of different lengths must land in the same range.
        """
        short = dtw_distance(_stroke(30), _stroke(30, phase=np.pi))
        long = dtw_distance(_stroke(300), _stroke(300, phase=np.pi))
        assert long == pytest.approx(short, rel=0.15)

    def test_single_frame_sequences_are_handled(self):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        assert dtw_distance(a, b) == pytest.approx(5.0, rel=1e-6)

    def test_channel_count_must_match(self):
        with pytest.raises(ValueError, match="channel"):
            dtw_distance(_stroke(10), np.zeros((10, 3), dtype=np.float32))


class TestPluggableDistanceKeepsTheProtocol:
    """
    `enrolled_trial_scores` grows an optional `distance=`. The default path must
    stay byte-identical, and the plugged-in path must see the same draw.
    """

    def _embedded(self):
        genuine = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                   [0.1, 0.1], [0.2, 0.0], [0.0, 0.2]]
        forged = [[5.0, 0.0], [0.0, 5.0]]
        return {"w1": {"genuine": np.array(genuine, dtype=np.float32),
                       "forged": np.array(forged, dtype=np.float32)}}

    def test_explicit_l2_reproduces_the_default_exactly(self):
        """If these diverge, the plug-in changed the protocol rather than the metric."""
        embedded = self._embedded()
        default, default_labels = enrolled_trial_scores(embedded, num_references=5, seed=42)
        plugged, plugged_labels = enrolled_trial_scores(
            embedded, num_references=5, seed=42,
            distance=lambda p, r: float(np.linalg.norm(np.asarray(p) - np.asarray(r))))
        assert plugged.tolist() == pytest.approx(default.tolist())
        assert plugged_labels.tolist() == default_labels.tolist()

    def test_the_supplied_distance_is_the_one_used(self):
        embedded = self._embedded()
        scores, _ = enrolled_trial_scores(embedded, num_references=5, seed=42,
                                          distance=lambda p, r: 3.0)
        # Every probe sits at d = 3 from every reference, so S = 1/(1+3).
        assert scores.tolist() == pytest.approx([0.25] * len(scores))

    def test_variable_length_samples_survive_the_draw(self):
        """
        Trajectories are ragged, so the reference arrays are object lists rather
        than a stacked [N, D] matrix. The draw must index them without stacking.
        """
        embedded = {"w1": {
            "genuine": [_stroke(n) for n in (20, 25, 30, 35, 40, 45)],
            "forged": [_stroke(22, phase=np.pi), _stroke(28, phase=np.pi)],
        }}
        scores, labels = enrolled_trial_scores(embedded, num_references=5, seed=42,
                                               distance=dtw_distance)
        assert labels.tolist().count(1) == 1
        assert labels.tolist().count(0) == 2
        assert np.all(np.isfinite(scores))


class TestBatchedDtw:
    """
    The batched path exists for speed only. Speed is worthless if the number
    moves, so every test here is an agreement test against `dtw_distance`,
    which stays the reference implementation.
    """

    def _pairs(self, count=12, seed=0):
        rng = np.random.default_rng(seed)
        pairs = []
        for _ in range(count):
            n, m = int(rng.integers(5, 40)), int(rng.integers(5, 40))
            pairs.append((rng.standard_normal((n, 3)).astype(np.float32),
                          rng.standard_normal((m, 3)).astype(np.float32)))
        return pairs

    def test_matches_the_reference_on_ragged_pairs(self):
        """Different (n, m) per pair — padding must not leak into the result."""
        pairs = self._pairs()
        expected = [dtw_distance(a, b) for a, b in pairs]
        assert dtw_distance_batch(pairs).tolist() == pytest.approx(expected, rel=1e-6)

    def test_batch_size_does_not_change_the_answer(self):
        """
        Pairs are padded to the longest in their batch, so a pair's result must
        not depend on which other pairs it was batched with.
        """
        pairs = self._pairs(count=9, seed=7)
        one = dtw_distance_batch(pairs, batch_size=1)
        many = dtw_distance_batch(pairs, batch_size=9)
        assert one.tolist() == pytest.approx(many.tolist(), rel=1e-6)

    def test_preserves_warp_invariance(self):
        a = _stroke(40)
        pairs = [(a, np.repeat(a, 2, axis=0)), (a, a)]
        assert dtw_distance_batch(pairs).tolist() == pytest.approx([0.0, 0.0], abs=1e-5)

    def test_rejects_mismatched_channel_counts(self):
        with pytest.raises(ValueError, match="channel"):
            dtw_distance_batch([(_stroke(10), np.zeros((10, 3), dtype=np.float32))])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_cuda_agrees_with_cpu(self):
        """The whole point of the GPU path: same numbers, less waiting."""
        pairs = self._pairs(count=16, seed=3)
        cpu = dtw_distance_batch(pairs, device="cpu")
        gpu = dtw_distance_batch(pairs, device="cuda")
        assert gpu.tolist() == pytest.approx(cpu.tolist(), rel=1e-5)


class TestPrecomputedDistances:

    def _writers(self):
        rng = np.random.default_rng(11)
        return {"w1": {
            "genuine": [_stroke(int(rng.integers(15, 30))) for _ in range(6)],
            "forged": [_stroke(int(rng.integers(15, 30)), phase=np.pi) for _ in range(3)],
        }}

    def test_batched_scoring_equals_pair_at_a_time_scoring(self):
        """
        The cache is a speed optimization over calling `dtw_distance` per pair.
        If the two disagree, the optimization changed the measurement.
        """
        writers = self._writers()
        naive, naive_labels = enrolled_trial_scores(writers, num_references=3, seed=5,
                                                    distance=dtw_distance)
        cached, cached_labels = enrolled_trial_scores(
            writers, num_references=3, seed=5,
            distance=precomputed_distances(writers))
        assert cached.tolist() == pytest.approx(naive.tolist(), rel=1e-5)
        assert cached_labels.tolist() == naive_labels.tolist()

    def test_covers_every_pair_a_draw_can_request(self):
        """A KeyError here means a draw asked for a comparison that was not precomputed."""
        writers = self._writers()
        distance = precomputed_distances(writers)
        for k in (1, 3, 5):
            for seed in range(4):
                scores, _ = enrolled_trial_scores(writers, num_references=k,
                                                  seed=seed, distance=distance)
                assert np.all(np.isfinite(scores))


class TestTrajectoryWriters:
    """
    The DTW analogue of `embed_writers`: same output shape, raw trajectories
    where the model path puts embeddings, so both feed the identical protocol.
    """

    def _samples(self):
        return {
            "svc_1": {"genuine": [_stroke(20), _stroke(21)], "forged": [_stroke(22)]},
            "svc_2": {"genuine": [_stroke(23)]},
            "svc_3": {"genuine": [_stroke(24)], "forged": [_stroke(25)]},
        }

    def test_selects_only_the_requested_writers(self):
        out = trajectory_writers(self._samples(), ["svc_1", "svc_3"])
        assert sorted(out) == ["svc_1", "svc_3"]

    def test_shape_matches_the_embed_writers_contract(self):
        out = trajectory_writers(self._samples(), ["svc_1"])
        assert set(out["svc_1"]) == {"genuine", "forged"}
        assert len(out["svc_1"]["genuine"]) == 2
        assert len(out["svc_1"]["forged"]) == 1

    def test_a_writer_with_no_forgeries_gets_an_empty_list_not_a_crash(self):
        """`embed_writers` uses .get(..., []); the skip happens downstream, not here."""
        out = trajectory_writers(self._samples(), ["svc_2"])
        assert out["svc_2"]["forged"] == []

    def test_samples_are_passed_through_unmodified(self):
        """
        `embed_samples` applies no temporal preprocessing — the parsers already
        normalized. DTW must see exactly those arrays, or it is not measuring
        the same input as the LSTM.
        """
        samples = self._samples()
        out = trajectory_writers(samples, ["svc_1"])
        assert np.array_equal(out["svc_1"]["genuine"][0], samples["svc_1"]["genuine"][0])
