"""Dataset leakage and split validation tests.

Verifies writer independence across train, validation, and test partitions,
preventing data leakage and overlap across datasets.
"""

import collections
import hashlib
import os

import pytest

from config.settings import (
    PROJECT_ROOT, SIGCOMP2009_DIR, SIGCOMP2011_OFFLINE_DIR, SPATIAL_SPLIT_RATIOS,
    SPATIAL_TRAIN_DIRS, SPATIAL_WEIGHTS_PATH, TEMPORAL_SPLIT_RATIOS, TEMPORAL_TRAIN_DIRS,
    TEMPORAL_WEIGHTS_PATH,
)
from config.settings import spatial_input_size_from_env
from src.data.parsers import max_gap_ms_from_env
from src.data.dataset import (check_split_against_record, load_split_record, pin_holdout,
                              save_split_record, split_record_path, writer_independent_split)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _have(*dirs):
    return all(os.path.isdir(d) for d in dirs)


def _corpus_of(writer_id):
    return writer_id.rsplit("_", 1)[0]


def _image_digests(root):
    """{sha256: relative path} for every image under root, first path wins."""
    digests = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                digests.setdefault(hashlib.sha256(fh.read()).hexdigest(), path)
    return digests


def cross_corpus_duplicates(training_roots, evaluation_roots):
    """
    Pairs of (training path, evaluation path) whose bytes are identical.

    An evaluation corpus that shares a file with a training corpus measures
    memorisation for that file, not generalisation. Byte identity is the cheap,
    zero-false-positive half of the check; near-duplicates need the perceptual
    hash screen in tools/research/cold_start_eval.py.
    """
    train = {}
    for root in training_roots:
        train.update(_image_digests(root))
    pairs = []
    for root in evaluation_roots:
        for digest, path in _image_digests(root).items():
            if digest in train:
                pairs.append((train[digest], path))
    return pairs


def sample_conflicts(writer_samples):
    """
    Samples filed under two writers, or as both genuine and forged.

    `merge_datasets` once replaced a writer's record on an ID collision and
    silently dropped 1,600 signatures; the opposite failure, one file counted
    twice, would let a probe be scored against itself as a reference.
    """
    owners = collections.defaultdict(set)
    for writer, record in writer_samples.items():
        for label in ("genuine", "forged"):
            for sample in record[label]:
                owners[sample].add((writer, label))
    return {sample: sorted(where) for sample, where in owners.items() if len(where) > 1}


# ─── Mechanism, synthetic data ──────────────────────────────────────────

class TestLeakDetectors:
    def _tree(self, tmp_path, name, files):
        root = tmp_path / name
        root.mkdir()
        for fname, content in files.items():
            (root / fname).write_bytes(content)
        return str(root)

    def test_byte_identical_image_across_corpora_is_reported(self, tmp_path):
        train = self._tree(tmp_path, "train", {"a.png": b"\x89PNG-A", "b.png": b"\x89PNG-B"})
        evaluation = self._tree(tmp_path, "eval", {"x.png": b"\x89PNG-A", "y.png": b"\x89PNG-Y"})

        pairs = cross_corpus_duplicates([train], [evaluation])

        assert len(pairs) == 1
        assert pairs[0][0].endswith("a.png") and pairs[0][1].endswith("x.png")

    def test_distinct_corpora_report_nothing(self, tmp_path):
        train = self._tree(tmp_path, "train", {"a.png": b"\x89PNG-A"})
        evaluation = self._tree(tmp_path, "eval", {"x.png": b"\x89PNG-X", "notes.txt": b"\x89PNG-A"})

        # The .txt has the same bytes as a.png and must be ignored: only images count.
        assert cross_corpus_duplicates([train], [evaluation]) == []

    def test_sample_filed_under_two_writers_is_reported(self):
        samples = {
            "w1": {"genuine": ["s1", "s2"], "forged": ["f1"]},
            "w2": {"genuine": ["s3"], "forged": ["s2"]},
        }
        conflicts = sample_conflicts(samples)
        assert list(conflicts) == ["s2"]
        assert conflicts["s2"] == [("w1", "genuine"), ("w2", "forged")]

    def test_clean_samples_report_nothing(self):
        samples = {"w1": {"genuine": ["s1"], "forged": ["f1"]}, "w2": {"genuine": ["s2"], "forged": ["f2"]}}
        assert sample_conflicts(samples) == {}


class TestSplitLeakage:
    def test_pinned_writers_never_reach_training_whatever_the_input_order(self):
        """The leak's failure shape: the same pin list, three orderings of the corpus."""
        ids = [f"w{i}" for i in range(80)]
        pinned = ["w7", "w70"]
        for order in (ids, list(reversed(ids)), sorted(ids, key=lambda s: s[::-1])):
            train, val, _ = pin_holdout(order, pinned, *SPATIAL_SPLIT_RATIOS, seed=42)
            assert not set(pinned) & (set(train) | set(val))


class TestSplitRecord:
    """Tests that model checkpoints verify writer split records during evaluation."""

    def _record(self, tmp_path, train, val, test):
        weights = str(tmp_path / "model.pth")
        save_split_record(weights, train, val, test, data_dirs=["data/x"], split_seed=42)
        return weights

    def test_record_is_written_next_to_the_weights_with_the_split(self, tmp_path):
        weights = self._record(tmp_path, ["a", "b"], ["c"], ["d", "e"])
        path = split_record_path(weights)
        assert path.endswith("model.split.json") and os.path.exists(path)
        record = load_split_record(weights)
        assert record["train"] == ["a", "b"] and record["test"] == ["d", "e"]
        assert record["split_seed"] == 42 and record["data_dirs"] == ["data/x"]

    def test_matching_partition_passes(self, tmp_path):
        weights = self._record(tmp_path, ["a", "b"], ["c"], ["d", "e"])
        assert check_split_against_record(weights, test_ids=["e", "d"]) is not None

    def test_scoring_a_writer_the_checkpoint_trained_on_raises(self, tmp_path):
        weights = self._record(tmp_path, ["a", "b"], ["c"], ["d", "e"])
        with pytest.raises(ValueError, match="trained on"):
            check_split_against_record(weights, test_ids=["a", "e"])

    def test_a_different_held_out_set_raises_even_without_a_leak(self, tmp_path):
        weights = self._record(tmp_path, ["a", "b"], ["c"], ["d", "e"])
        # "c" was validation, not training: no leak, but not the documented test set either.
        with pytest.raises(ValueError, match="held-out"):
            check_split_against_record(weights, test_ids=["c", "d", "e"])
        # An explicit --writers narrowing is allowed to score a subset.
        assert check_split_against_record(weights, test_ids=["d"],
                                          require_same_test=False) is not None

    def test_checkpoint_without_a_record_is_not_checked(self, tmp_path):
        assert check_split_against_record(str(tmp_path / "old.pth"), test_ids=["b"]) is None

    def test_record_stores_the_input_size_and_a_mismatch_refuses_to_score(self, tmp_path):
        """
        A checkpoint trained at another image size is not recoverable from its
        weights (adaptive pooling makes the trunk size-agnostic), so the record
        must carry it. Scoring it at the config size would preprocess every scan
        at the wrong geometry and print a plausible EER for a model it never was.
        """
        weights = str(tmp_path / "model.pth")
        save_split_record(weights, ["a"], ["b"], ["c"], data_dirs=["data/x"], split_seed=42,
                          input_size=(110, 78))
        assert load_split_record(weights)["input_size"] == [110, 78]
        with pytest.raises(ValueError, match="input size"):
            check_split_against_record(weights, test_ids=["c"], input_size=(220, 155))
        assert check_split_against_record(weights, test_ids=["c"],
                                          input_size=(110, 78)) is not None

    def test_record_without_an_input_size_is_scored_at_the_config_size(self, tmp_path):
        weights = self._record(tmp_path, ["a"], ["b"], ["c"])   # legacy record without input_size
        assert "input_size" not in load_split_record(weights) or load_split_record(weights)["input_size"] is None
        assert check_split_against_record(weights, test_ids=["c"], input_size=(220, 155)) is not None


class TestRecipeRecord:
    """The record carries every training-time constant that is not recoverable
    from the weights and that scoring must reproduce; a mismatch refuses."""

    def test_max_gap_ms_is_recorded_and_a_mismatch_refuses_to_score(self, tmp_path):
        weights = str(tmp_path / "t.pth")
        save_split_record(weights, ["a"], ["b"], ["c"], data_dirs=["data/x"], split_seed=42, max_gap_ms=200.0)
        assert load_split_record(weights)["max_gap_ms"] == 200.0
        with pytest.raises(ValueError, match="MAX_GAP_MS"):
            check_split_against_record(weights, test_ids=["c"], max_gap_ms=100.0)
        assert check_split_against_record(weights, test_ids=["c"], max_gap_ms=200.0) is not None

    def test_old_record_without_the_field_is_accepted(self, tmp_path):
        weights = str(tmp_path / "t.pth")
        save_split_record(weights, ["a"], ["b"], ["c"], data_dirs=["data/x"], split_seed=42)
        assert check_split_against_record(weights, test_ids=["c"], max_gap_ms=100.0) is not None

    def test_gap_env_parser(self):
        assert max_gap_ms_from_env(None, 100.0) == 100.0
        assert max_gap_ms_from_env("200", 100.0) == 200.0
        assert max_gap_ms_from_env("50.5", 100.0) == 50.5
        for bad in ("0", "-5", "fast", ""):
            with pytest.raises(ValueError, match="SIGVER_MAX_GAP_MS"):
                max_gap_ms_from_env(bad, 100.0)


class TestInputSizeOverride:
    """`SIGVER_SPATIAL_INPUT_SIZE=WxH` is the sweep's only way to train and score
    at another geometry; a malformed value must fail loudly, not fall back."""

    def test_unset_keeps_the_config_default(self):
        assert spatial_input_size_from_env(None, (220, 155)) == (220, 155)

    def test_width_by_height_is_parsed_as_the_cv2_tuple(self):
        assert spatial_input_size_from_env("110x78", (220, 155)) == (110, 78)

    @pytest.mark.parametrize("bad", ["78", "110x", "wide", "0x78", "110x78x1"])
    def test_malformed_values_raise(self, bad):
        with pytest.raises(ValueError, match="SIGVER_SPATIAL_INPUT_SIZE"):
            spatial_input_size_from_env(bad, (220, 155))


# ─── The corpora on disk (skipped when not downloaded) ───────────────────

_IMAGE_CORPORA = SPATIAL_TRAIN_DIRS
_PEN_CORPORA = TEMPORAL_TRAIN_DIRS


@pytest.mark.parametrize("weights, train_dirs", [(SPATIAL_WEIGHTS_PATH, SPATIAL_TRAIN_DIRS),
                                                 (TEMPORAL_WEIGHTS_PATH, TEMPORAL_TRAIN_DIRS)])
def test_settings_training_corpora_match_the_shipped_split_record(weights, train_dirs):
    """Verify that settings training corpora match the split record of shipped weights."""
    record = load_split_record(weights)
    if record is None:
        pytest.skip(f"no split record next to {weights}")
    recorded = {os.path.normpath(os.path.join(PROJECT_ROOT, d)) for d in record["data_dirs"]}
    assert recorded == {os.path.normpath(d) for d in train_dirs}


@pytest.mark.skipif(not _have(*_IMAGE_CORPORA), reason="CEDAR, BHSig260 and UTSig not all downloaded")
def test_real_spatial_split_matches_the_published_held_out_set():
    """
    The README quotes 65 held-out spatial writers: 8 CEDAR, 23 Hindi,
    16 Bengali, 18 UTSig, from seed 42. Every published spatial number stands on
    that set. If the corpus list, the seed, the ratios or the split function
    change, this fails and the documents have to be re-derived rather than
    quietly drifting from the code.
    """
    from scripts.train import build_writer_samples
    samples = build_writer_samples(list(_IMAGE_CORPORA), None)

    held_out = writer_independent_split(sorted(samples), *SPATIAL_SPLIT_RATIOS, seed=42)[2]
    from_dict_order = writer_independent_split(list(samples.keys()), *SPATIAL_SPLIT_RATIOS, seed=42)[2]

    assert set(held_out) == set(from_dict_order), "the split depends on input order again"
    assert collections.Counter(map(_corpus_of, held_out)) == {
        "cedar": 8, "bhsig_Hindi": 23, "bhsig_Bengali": 16, "utsig": 18}


@pytest.mark.skipif(not _have(*_PEN_CORPORA), reason="SVC2004 Task2 and SigComp2011 not all downloaded")
def test_real_temporal_split_matches_the_published_held_out_set():
    """31 held-out pen writers: 10 SVC2004, 15 Dutch, 6 Chinese (README results table)."""
    from scripts.train import build_writer_samples
    samples = build_writer_samples(list(_PEN_CORPORA), "dt")

    held_out = writer_independent_split(sorted(samples), *TEMPORAL_SPLIT_RATIOS, seed=42)[2]
    from_dict_order = writer_independent_split(list(samples.keys()), *TEMPORAL_SPLIT_RATIOS, seed=42)[2]

    assert set(held_out) == set(from_dict_order), "the split depends on input order again"
    assert collections.Counter(map(_corpus_of, held_out)) == {
        "svc": 10, "sigcomp_Dutch": 15, "sigcomp_Chinese": 6}


@pytest.mark.skipif(not _have(*_IMAGE_CORPORA), reason="CEDAR, BHSig260 and UTSig not all downloaded")
def test_real_image_samples_belong_to_one_writer_and_one_label():
    from scripts.train import build_writer_samples
    conflicts = sample_conflicts(build_writer_samples(list(_IMAGE_CORPORA), None))
    assert conflicts == {}, f"{len(conflicts)} files counted twice, e.g. {next(iter(conflicts.items()))}"


@pytest.mark.skipif(not _have(*_IMAGE_CORPORA), reason="CEDAR, BHSig260 and UTSig not all downloaded")
def test_real_training_corpora_share_no_image_with_each_other():
    """
    A file present in two training corpora would be one signature under two
    writer IDs, both in the training mix or one of them held out.
    """
    seen, dupes = {}, []
    for root in _IMAGE_CORPORA:
        for digest, path in _image_digests(root).items():
            if digest in seen and os.path.dirname(seen[digest]) != os.path.dirname(path):
                dupes.append((seen[digest], path))
            seen.setdefault(digest, path)
    assert dupes == [], f"{len(dupes)} byte-identical images across corpora, e.g. {dupes[0]}"


@pytest.mark.skipif(
    not _have(*_IMAGE_CORPORA) or not (os.path.isdir(SIGCOMP2011_OFFLINE_DIR) or os.path.isdir(SIGCOMP2009_DIR)),
    reason="training corpora or an evaluation-only corpus not downloaded")
def test_real_evaluation_corpora_share_no_image_with_the_training_corpora():
    """
    SigComp2011 offline and SigComp2009/NFI are scored as corpora the shipped
    spatial weight never saw. That is only true if none of their files is also a
    training file; sign_data once contained 362 such copies and reported 0.39%.
    """
    evaluation = [d for d in (SIGCOMP2011_OFFLINE_DIR, SIGCOMP2009_DIR) if os.path.isdir(d)]
    pairs = cross_corpus_duplicates(list(_IMAGE_CORPORA), evaluation)
    assert pairs == [], f"{len(pairs)} evaluation images are byte-identical to training images, e.g. {pairs[0]}"
