"""
Session labels on genuine signatures ("genuine_session"), parsed per corpus and kept
aligned when corpora merge. The cross-session and triplet training arms that first
used them were removed on 18 November 2026; they are in commit 3c80324.
"""

import os

from src.data.parsers import merge_datasets, parse_deepsigndb, parse_svc2004


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
