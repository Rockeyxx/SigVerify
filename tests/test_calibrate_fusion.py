import numpy as np

from scripts.calibrate_fusion import choose, draw_scores, evaluate, parse_nfi_pairs


def _draws(scan_noise, pen_noise, num_writers=12, draws=3):
    """Draws of (scan, pen) similarities: genuines near 0.8, forgeries near 0.5."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(draws):
        rows = {}
        for w in range(num_writers):
            g = 0.8 + rng.normal(0, 1, (2, 10)) * [[scan_noise], [pen_noise]]
            f = 0.5 + rng.normal(0, 1, (2, 10)) * [[scan_noise], [pen_noise]]
            rows[f"w{w}"] = {"genuine": (g[0], g[1]), "forged": (f[0], f[1])}
        out.append(rows)
    return out


class TestChoose:

    def test_alpha_follows_the_better_branch(self):
        # Pen separates cleanly, scan is noise: the weight goes to the pen (alpha near 0),
        # and the other way round. Several alphas can tie at zero EER; the first wins.
        alpha, _ = choose(_draws(scan_noise=1.0, pen_noise=0.01))
        assert alpha <= 0.1
        alpha, _ = choose(_draws(scan_noise=0.01, pen_noise=1.0))
        assert alpha >= 0.9

    def test_tau_is_the_mean_eer_point_at_the_chosen_alpha(self):
        draws = _draws(scan_noise=0.1, pen_noise=0.1)
        alpha, tau = choose(draws)
        assert tau == evaluate(draws, alpha, 0.0)["eer_tau"][0]
        assert 0.5 < tau < 0.8


class TestDrawScores:

    def _embedded(self, genuines):
        rng = np.random.default_rng(1)
        bank = rng.normal(size=(genuines, 4)).astype(np.float32)
        forged = rng.normal(size=(3, 4)).astype(np.float32)
        return {"genuine": (bank, bank.copy()), "forged": (forged, forged.copy())}

    def test_both_branches_share_one_reference_draw(self):
        # Identical scan and pen embeddings score identically only if the references match.
        rows = draw_scores({"a": self._embedded(8)}, draw=0, num_references=5)
        scan, pen = rows["a"]["genuine"]
        assert len(scan) == 3 and np.array_equal(scan, pen)

    def test_a_writer_without_a_spare_genuine_is_skipped(self):
        rows = draw_scores({"a": self._embedded(5), "b": self._embedded(6)}, draw=0,
                           num_references=5)
        assert list(rows) == ["b"]


class TestParseNfiPairs:

    def test_pairs_scan_and_pen_by_name_and_keeps_listed_writers_only(self, tmp_path):
        for folder, names in (("genuines", ["NFI-00201002", "NFI-00801008"]),
                              ("forgeries", ["NFI-01501002", "NFI-01601099"])):
            (tmp_path / folder).mkdir()
            for name in names:
                (tmp_path / folder / f"{name}.png").write_bytes(b"")
                (tmp_path / folder / f"{name}.hwr").write_text("")
        (tmp_path / "genuines" / "NFI-00202002.png").write_bytes(b"")   # no pen file

        pairs = parse_nfi_pairs(str(tmp_path), writers=("nfi_002", "nfi_099"))

        assert sorted(pairs) == ["nfi_002", "nfi_099"]
        assert [p[0].endswith("NFI-00201002.png") for p in pairs["nfi_002"]["genuine"]] == [True]
        assert [p[1].endswith("NFI-01501002.hwr") for p in pairs["nfi_002"]["forged"]] == [True]
        assert pairs["nfi_099"]["genuine"] == [] and len(pairs["nfi_099"]["forged"]) == 1
