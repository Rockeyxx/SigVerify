"""
Tests for SONOF scans as a training-only spatial corpus (research arm).

The arm adds one half of the SONOF writers to training and must leave the
shipped CEDAR + BHSig260 + UTSig partition exactly as it was, so every test of
the new path also pins the split it is not allowed to move.
"""

import json
import os

import cv2
import numpy as np
import pytest

from config.settings import BIOSECURID_SONOF_DIR
from scripts.train import main as train_main
from src.data.parsers import (parse_biosecurid_sonof_pairs, parse_biosecurid_sonof_scans,
                              parse_dataset)


def _png(path, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(img, (5, 10 + seed % 20), (58, 50 - seed % 20), 0, 2)
    cv2.imwrite(path, img)


def _sonof_scans(root, writers=(1, 2), sessions=(1, 2)):
    for u in writers:
        for s in sessions:
            for slot in range(1, 8):
                label = "g" if slot in (1, 2, 6, 7) else "f"
                _png(os.path.join(root, "OfflineReal", f"u{u:04d}",
                                  f"u{u:04d}s{s:04d}_sg{slot:04d}{label}.png"), u * 7 + slot)
    return root


def _cedar(root, writers=10):
    for w in range(1, writers + 1):
        for i in range(1, 4):
            _png(os.path.join(root, "full_org", f"original_{w}_{i}.png"), w)
        for i in range(1, 3):
            _png(os.path.join(root, "full_forg", f"forgeries_{w}_{i}.png"), w + 30)
    return root


class TestScanParser:

    def test_writers_use_the_pen_ids_and_labels_follow_the_file(self, tmp_path):
        writers = parse_biosecurid_sonof_scans(_sonof_scans(str(tmp_path)))
        assert sorted(writers) == ["sonof_u1001", "sonof_u1002"]
        rec = writers["sonof_u1002"]
        assert len(rec["genuine"]) == 8 and len(rec["forged"]) == 6
        assert all(p.endswith("g.png") and "/u0002/" in p for p in rec["genuine"])
        assert all(p.endswith("f.png") for p in rec["forged"])

    def test_a_label_that_disagrees_with_its_slot_stops_the_parse(self, tmp_path):
        root = _sonof_scans(str(tmp_path), writers=(1,), sessions=(1,))
        bad = os.path.join(root, "OfflineReal", "u0001", "u0001s0001_sg0003f.png")
        os.rename(bad, bad[:-5] + "g.png")
        with pytest.raises(ValueError, match="slot"):
            parse_biosecurid_sonof_scans(root)

    def test_dispatch_by_folder_name(self, tmp_path):
        root = _sonof_scans(str(tmp_path / "biosecurid_sonof"))
        assert sorted(parse_dataset(root)) == ["sonof_u1001", "sonof_u1002"]


@pytest.mark.skipif(not os.path.isdir(os.path.join(BIOSECURID_SONOF_DIR, "OfflineReal")),
                    reason="BiosecurID-SONOF not licensed/downloaded")
def test_real_sonof_scans_match_the_fused_join():
    """132 writers x (16 genuine + 12 forged) = 3,696 scans, the same ones the pairs use."""
    scans = parse_biosecurid_sonof_scans(BIOSECURID_SONOF_DIR)
    assert len(scans) == 132
    assert sum(len(r["genuine"]) + len(r["forged"]) for r in scans.values()) == 3696
    assert all(len(r["genuine"]) == 16 and len(r["forged"]) == 12 for r in scans.values())
    pairs = parse_biosecurid_sonof_pairs(BIOSECURID_SONOF_DIR)
    assert sorted(scans) == sorted(pairs)
    for w, rec in pairs.items():
        for label in ("genuine", "forged"):
            assert sorted(scans[w][label]) == sorted(png for png, _ in rec[label])


class TestTrainOnlyCorpus:

    def _run(self, tmp_path, name, extra):
        cedar = os.path.join(str(tmp_path), "cedar")
        if not os.path.isdir(cedar):
            _cedar(cedar)
        return train_main(["--branch", "spatial", "--data-dir", cedar, "--epochs", "1",
                           "--batch-size", "8", "--num-workers", "0",
                           "--weights-dir", str(tmp_path / name)] + extra)

    @pytest.mark.slow  # real training loop; skipped by default, run in CI
    def test_listed_writers_join_training_and_the_base_split_does_not_move(self, tmp_path):
        sonof = _sonof_scans(str(tmp_path / "biosecurid_sonof"))
        base = self._run(tmp_path, "base", [])["split"]
        arm = self._run(tmp_path, "arm", ["--train-only-data-dir", sonof,
                                          "--train-only-writers", "sonof_u1001"])["split"]
        assert arm["val"] == base["val"] and arm["test"] == base["test"]
        assert arm["train"] == base["train"] + ["sonof_u1001"]
        record = json.load(open(tmp_path / "arm" / "spatial_cnn.split.json"))
        assert "sonof_u1001" in record["train"]
        assert record["train_only_dirs"] == [sonof]
        assert not any("sonof_u1002" in record[k] for k in ("train", "val", "test"))

    def test_an_unknown_writer_is_refused(self, tmp_path):
        sonof = _sonof_scans(str(tmp_path / "biosecurid_sonof"))
        with pytest.raises(ValueError, match="sonof_u9999"):
            self._run(tmp_path, "w", ["--train-only-data-dir", sonof,
                                      "--train-only-writers", "sonof_u9999"])

    def test_a_dir_without_a_writer_list_is_refused(self, tmp_path):
        sonof = _sonof_scans(str(tmp_path / "biosecurid_sonof"))
        with pytest.raises(SystemExit):
            self._run(tmp_path, "w", ["--train-only-data-dir", sonof])
