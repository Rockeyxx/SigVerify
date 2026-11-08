"""
Tests for the cross-session positive pairs and the triplet term (research arms).

Both are off by default. The shipped recipe must draw the same pairs and compute
the same loss as before, so every test of a new arm also pins the default path.
"""

import os

import numpy as np
import pytest
import torch

from scripts.train import main as train_main
from src.data.dataset import collate_temporal_pairs, generate_skilled_pairs
from src.data.parsers import merge_datasets, parse_deepsigndb, parse_svc2004
from src.models.siamese import ContrastiveLoss


def _write_svc(root, user, sample, length):
    with open(os.path.join(root, f"U{user}S{sample}.TXT"), "w") as fh:
        fh.write(f"{length}\n")
        for t in range(length):
            fh.write(f"{100 + t * 7} {200 + (t * t) % 13} {t * 10} 1 1500 700 500\n")


class TestSessionsAreParsed:

    def test_svc2004_genuines_1_to_10_are_session_1_and_11_to_20_session_2(self, tmp_path):
        """
        Files are listed in name order (S1, S10, S11, S2, ...), so the session list
        must follow the genuine list, not the sample index. Each file has its own
        length, which ties every loaded array back to its sample number.
        """
        root = str(tmp_path)
        for sample in range(1, 21):
            _write_svc(root, 1, sample, length=sample + 5)
        _write_svc(root, 1, 21, length=40)

        record = parse_svc2004(root)["svc_1"]
        assert len(record["genuine_session"]) == len(record["genuine"]) == 20
        for array, session in zip(record["genuine"], record["genuine_session"]):
            sample = len(array) - 5
            assert session == (1 if sample <= 10 else 2), f"S{sample} got session {session}"

    def test_deepsigndb_biosecurid_session_comes_from_the_file_name(self, tmp_path):
        """BiosecurID names carry `s000N`; MCYT names carry none, so they get None."""
        stylus = tmp_path / "DeepSignDB" / "Development" / "stylus"
        stylus.mkdir(parents=True)
        mcyt = "3\n10 20 0 3600 900 512\n11 21 10 3600 900 600\n12 22 20 3600 900 700\n"
        bio = "3\n10 20 0 1 1500 700 512\n11 21 10 1 1500 700 600\n12 22 20 1 1500 700 700\n"
        (stylus / "u0001_g_0100v00.txt").write_text(mcyt)
        (stylus / "u0001_s_0100f00.txt").write_text(mcyt)
        for session in (1, 3):
            (stylus / f"u0231_g_u1001s000{session}_sg0001.txt").write_text(bio)
        (stylus / "u0231_s_u1001s0001_sg0003.txt").write_text(bio)

        writers = parse_deepsigndb(str(tmp_path))
        assert writers["deepsign_MCYT-dev_0001"]["genuine_session"] == [None]
        assert writers["deepsign_BiosecurID-dev_0231"]["genuine_session"] == [1, 3]

    def test_merge_keeps_sessions_aligned_with_genuines(self):
        """A corpus without sessions merged with one that has them pads with None."""
        target = {"w": {"genuine": ["a"], "forged": []}}
        merge_datasets(target, {"w": {"genuine": ["b", "c"], "forged": [],
                                      "genuine_session": [1, 2]}})
        assert target["w"]["genuine"] == ["a", "b", "c"]
        assert target["w"]["genuine_session"] == [None, 1, 2]

    def test_merge_without_sessions_adds_no_session_key(self):
        target = {}
        merge_datasets(target, {"w": {"genuine": ["a"], "forged": ["f"]}})
        assert "genuine_session" not in target["w"]


def _writers(sessions=True):
    out = {}
    for w in range(3):
        out[f"w{w}"] = {"genuine": [f"w{w}_g{i}" for i in range(8)],
                        "forged": [f"w{w}_f{i}" for i in range(8)]}
        if sessions:
            out[f"w{w}"]["genuine_session"] = [1, 1, 1, 1, 2, 2, 2, 2]
    return out


def _session(sample):
    return 1 if int(sample.split("_g")[1]) < 4 else 2


class TestCrossSessionPositives:

    def test_every_positive_crosses_sessions_and_the_count_is_unchanged(self):
        """8 genuines in 2 sessions: 28 pairs, 16 across. A budget of 20 needs repeats."""
        base_pairs, base_labels = generate_skilled_pairs(_writers(), 20, 20, seed=3)
        pairs, labels = generate_skilled_pairs(_writers(), 20, 20, seed=3,
                                               cross_session=True)
        positives = [p for p, y in zip(pairs, labels) if y == 1]
        assert labels == base_labels
        assert len(positives) == 60
        assert all(_session(a) != _session(b) for a, b in positives)

    def test_negatives_are_the_same_draw_as_the_default(self):
        """The arm changes which positives, never which negatives."""
        base_pairs, base_labels = generate_skilled_pairs(_writers(), 20, 20, seed=3,
                                                         random_impostor_ratio=0.5)
        pairs, labels = generate_skilled_pairs(_writers(), 20, 20, seed=3,
                                               random_impostor_ratio=0.5,
                                               cross_session=True)
        assert ([p for p, y in zip(pairs, labels) if y == 0]
                == [p for p, y in zip(base_pairs, base_labels) if y == 0])

    def test_default_ignores_sessions(self):
        """With the flag off, a session field changes nothing."""
        assert (generate_skilled_pairs(_writers(sessions=True), 20, 20, seed=3)
                == generate_skilled_pairs(_writers(sessions=False), 20, 20, seed=3))

    def test_writers_with_one_session_or_none_keep_random_positives(self):
        writers = _writers(sessions=False)
        writers["w1"]["genuine_session"] = [1] * 8
        writers["w2"]["genuine_session"] = [None] * 8
        assert (generate_skilled_pairs(writers, 20, 20, seed=3, cross_session=True)
                == generate_skilled_pairs(writers, 20, 20, seed=3))

    def test_return_writers_names_the_writer_of_every_pair(self):
        pairs, labels, owners = generate_skilled_pairs(_writers(), 5, 5, seed=1,
                                                       random_impostor_ratio=0.4,
                                                       return_writers=True)
        assert len(owners) == len(pairs)
        for (a, _), owner in zip(pairs, owners):
            assert a.split("_")[0] == owner


class TestTripletTerm:

    def test_off_by_default_and_ignores_a_negative(self):
        e1, e2 = torch.randn(4, 3), torch.randn(4, 3)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        plain = ContrastiveLoss(margin=1.0)(e1, e2, labels)
        assert torch.equal(ContrastiveLoss(margin=1.0)(e1, e2, labels, torch.randn(2, 3)), plain)

    def test_adds_weighted_hinge_over_the_positive_rows(self):
        """loss = contrastive + w * mean(relu(d(a, p) - d(a, n) + m)) over label-1 rows."""
        e1 = torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
        e2 = torch.tensor([[3.0, 0.0], [9.0, 9.0], [0.0, 1.0]])
        labels = torch.tensor([1.0, 0.0, 1.0])
        negative = torch.tensor([[1.0, 0.0], [5.0, 5.0]])   # for rows 0 and 2
        base = ContrastiveLoss(margin=1.0, hard_negative_ratio=1.0)(e1, e2, labels)
        loss = ContrastiveLoss(margin=1.0, hard_negative_ratio=1.0, triplet_weight=0.5,
                               triplet_margin=2.0)(e1, e2, labels, negative)
        d_ap = torch.tensor([3.0, float(np.hypot(5, 4))])
        d_an = torch.tensor([1.0, 0.0])
        expected = base + 0.5 * torch.relu(d_ap - d_an + 2.0).mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_triplet_margin_defaults_to_the_contrastive_margin(self):
        assert ContrastiveLoss(margin=1.7, triplet_weight=0.1).triplet_margin == 1.7

    def test_collate_carries_a_padded_negative(self):
        batch = [(torch.ones(5, 4), torch.ones(3, 4), torch.tensor(1.0), torch.ones(7, 4)),
                 (torch.ones(2, 4), torch.ones(6, 4), torch.tensor(0.0), torch.ones(4, 4))]
        out = collate_temporal_pairs(batch)
        assert len(out) == 7
        assert out[5].shape == (2, 7, 4)
        assert torch.equal(out[6], torch.tensor([7, 4]))


def _svc_two_sessions(root, writers=8):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    for user in range(1, writers + 1):
        for sample in list(range(1, 13)) + list(range(21, 25)):
            length = int(rng.integers(20, 40))
            with open(os.path.join(root, f"U{user}S{sample}.TXT"), "w") as fh:
                fh.write(f"{length}\n")
                for t in range(length):
                    fh.write(f"{1000 + t * 7 + int(rng.integers(0, 30))} "
                             f"{2000 + t * 3 + user * 40} {t * 10} 1 1500 700 "
                             f"{int(rng.integers(1, 1024))}\n")
    return root


class TestTrainWiring:

    @pytest.mark.parametrize("extra", [["--cross-session-positives"],
                                       ["--triplet-weight", "0.5"],
                                       ["--cross-session-positives", "--triplet-weight", "0.5"]])
    def test_research_arms_train(self, tmp_path, extra):
        data = _svc_two_sessions(str(tmp_path / "svc2004_task2"))
        history = train_main(["--branch", "temporal", "--data-dir", data,
                              "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
                              "--weights-dir", str(tmp_path / "w")] + extra)
        assert all(np.isfinite(v) for v in history["train_loss"])

    def test_triplet_term_changes_the_training_loss(self, tmp_path):
        """Same seed and pairs: a nonzero weight must reach the optimised loss."""
        data = _svc_two_sessions(str(tmp_path / "svc2004_task2"))
        common = ["--branch", "temporal", "--data-dir", data, "--epochs", "1",
                  "--batch-size", "8", "--num-workers", "0"]
        plain = train_main(common + ["--weights-dir", str(tmp_path / "a")])
        triplet = train_main(common + ["--weights-dir", str(tmp_path / "b"),
                                       "--triplet-weight", "1.0"])
        assert triplet["train_loss"][0] != plain["train_loss"][0]
