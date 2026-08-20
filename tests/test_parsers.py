"""
Tests for the real-dataset parsers.

Two layers:
  1. Synthetic-fixture tests — always run, pin the naming/label logic.
  2. Real-dataset tests — skipped unless ./data is populated, and assert the
     exact writer/sample counts each dataset is documented to have.
"""

import os

import numpy as np
import pytest

from config.settings import (
    BHSIG260_DIR,
    CEDAR_DIR,
    DEEPSIGNDB_DIR,
    MOBISIG_DIR,
    SIGCOMP2011_DIR,
    SVC2004_TASK1_DIR,
    SVC2004_TASK2_DIR,
    UTSIG_DIR,
)
from src.data.parsers import (
    SIGCOMP_DECIMATION,
    deepsigndb_source,
    load_sigcomp2011_file,
    load_svc2004_file,
    merge_datasets,
    parse_bhsig260,
    parse_cedar,
    parse_deepsigndb,
    parse_mobisig,
    parse_sigcomp2011,
    parse_svc2004,
    parse_utsig,
    svc2004_has_pressure,
)


def _touch(path):
    """Create an empty file, making parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "a").close()


class TestParseCedar:
    """CEDAR: flat full_org/full_forg with original_/forgeries_ prefixes."""

    def _fixture(self, root):
        for writer in (1, 2):
            for sample in (1, 2, 3):
                _touch(os.path.join(root, "full_org", f"original_{writer}_{sample}.png"))
                _touch(os.path.join(root, "full_forg", f"forgeries_{writer}_{sample}.png"))
        return root

    def test_groups_by_writer_and_label(self, tmp_path):
        """Genuine and forged samples must land under the right writer and key."""
        writers = parse_cedar(self._fixture(str(tmp_path)))
        assert set(writers) == {"cedar_1", "cedar_2"}
        assert len(writers["cedar_1"]["genuine"]) == 3
        assert len(writers["cedar_1"]["forged"]) == 3
        assert all("full_org" in p for p in writers["cedar_1"]["genuine"])
        assert all("full_forg" in p for p in writers["cedar_1"]["forged"])

    def test_ignores_non_matching_files(self, tmp_path):
        """Thumbs.db and other strays must not become a phantom writer."""
        root = self._fixture(str(tmp_path))
        _touch(os.path.join(root, "full_org", "Thumbs.db"))
        _touch(os.path.join(root, "full_org", "notes.txt"))
        assert set(parse_cedar(root)) == {"cedar_1", "cedar_2"}

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_cedar("/nonexistent/cedar")


class TestParseBHSig260:
    """BHSig260: {script}/{writer}/X-S-{w}-{G|F}-{n}.tif"""

    def _fixture(self, root):
        for script, code in (("Bengali", "B"), ("Hindi", "H")):
            for writer in ("001", "002"):
                w = int(writer)
                for n in (1, 2):
                    base = os.path.join(root, script, writer)
                    _touch(os.path.join(base, f"{code}-S-{w}-G-{n:02d}.tif"))
                    _touch(os.path.join(base, f"{code}-S-{w}-F-{n:02d}.tif"))
        return root

    def test_writer_ids_are_script_scoped(self, tmp_path):
        """Bengali writer 001 and Hindi writer 001 are different people."""
        writers = parse_bhsig260(self._fixture(str(tmp_path)))
        assert "bhsig_Bengali_001" in writers
        assert "bhsig_Hindi_001" in writers
        assert len(writers) == 4

    def test_g_and_f_split(self, tmp_path):
        writers = parse_bhsig260(self._fixture(str(tmp_path)))
        entry = writers["bhsig_Bengali_001"]
        assert len(entry["genuine"]) == 2
        assert len(entry["forged"]) == 2
        assert all("-G-" in p for p in entry["genuine"])
        assert all("-F-" in p for p in entry["forged"])

    def test_single_script_selection(self, tmp_path):
        """Restricting to one script must exclude the other entirely."""
        writers = parse_bhsig260(self._fixture(str(tmp_path)), scripts=("Bengali",))
        assert all(w.startswith("bhsig_Bengali_") for w in writers)


class TestParseUTSig:
    """UTSig: Genuine/{w}/{n}.tif and Forgery/{type}/{w}/{n}.tif"""

    def _fixture(self, root):
        for writer in ("1", "2"):
            for n in (1, 2, 3):
                _touch(os.path.join(root, "Genuine", writer, f"{n}.tif"))
            for n in (1, 2):
                _touch(os.path.join(root, "Forgery", "Skilled", writer, f"{n}.tif"))
                _touch(os.path.join(root, "Forgery", "Simple", writer, f"{n}.tif"))
        return root

    def test_defaults_to_skilled_forgeries_only(self, tmp_path):
        """
        AGENTS.md reports metrics under the skilled-forgery scenario, so Simple
        and Opposite-Hand forgeries must be excluded unless explicitly asked for.
        """
        writers = parse_utsig(self._fixture(str(tmp_path)))
        assert len(writers["utsig_1"]["forged"]) == 2
        assert all("Skilled" in p for p in writers["utsig_1"]["forged"])

    def test_explicit_forgery_types(self, tmp_path):
        writers = parse_utsig(self._fixture(str(tmp_path)),
                              forgery_types=("Skilled", "Simple"))
        assert len(writers["utsig_1"]["forged"]) == 4


class TestLoadSVC2004File:
    """SVC2004 trajectory loading — auto-detects Task1 (4-col) vs Task2 (7-col)."""

    def _write(self, path, rows):
        with open(path, "w") as fh:
            fh.write(f"{len(rows)}\n")
            for row in rows:
                fh.write(" ".join(str(v) for v in row) + "\n")
        return str(path)

    def test_task1_four_column_imputes_pressure(self, tmp_path):
        """Task1 has no pressure channel — P must be imputed to 1.0 throughout."""
        path = self._write(tmp_path / "U1S1.TXT",
                           [[100, 200, 1000, 0], [110, 210, 1010, 1], [120, 190, 1020, 1]])
        seq = load_svc2004_file(path)
        assert seq.shape == (3, 3)
        assert seq.dtype == np.float32
        assert np.allclose(seq[:, 2], 1.0)

    def test_task2_seven_column_uses_real_pressure(self, tmp_path):
        """Task2 pressure (column 7) must be read and scaled into [0, 1]."""
        path = self._write(tmp_path / "U1S1.TXT",
                           [[100, 200, 1000, 0, 1500, 700, 0],
                            [110, 210, 1010, 1, 1500, 700, 511],
                            [120, 190, 1020, 1, 1500, 700, 1023]])
        seq = load_svc2004_file(path)
        assert seq.shape == (3, 3)
        assert not np.allclose(seq[:, 2], 1.0)
        assert seq[:, 2].min() >= 0.0 and seq[:, 2].max() <= 1.0
        assert np.isclose(seq[2, 2], 1.0)

    def test_coordinates_are_z_scored(self, tmp_path):
        """X/Y are raw tablet units in the thousands; they must be standardized."""
        path = self._write(tmp_path / "U1S1.TXT",
                           [[1000, 5000, 0, 0], [2000, 6000, 10, 1],
                            [3000, 7000, 20, 1], [4000, 8000, 30, 1]])
        seq = load_svc2004_file(path)
        assert np.allclose(seq[:, :2].mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(seq[:, :2].std(axis=0), 1.0, atol=1e-5)

    def test_constant_coordinate_does_not_divide_by_zero(self, tmp_path):
        """A degenerate dot signature has zero variance — must not emit NaN."""
        path = self._write(tmp_path / "U1S1.TXT",
                           [[500, 500, 0, 1], [500, 500, 10, 1]])
        seq = load_svc2004_file(path)
        assert np.all(np.isfinite(seq))

    def test_header_count_mismatch_is_tolerated(self, tmp_path):
        """Trust the actual rows, not the declared header count."""
        path = str(tmp_path / "U1S1.TXT")
        with open(path, "w") as fh:
            fh.write("99\n100 200 1000 0\n110 210 1010 1\n")
        assert load_svc2004_file(path).shape == (2, 3)


class TestParseSVC2004:
    """SVC2004: flat U{u}S{s}.TXT, S1-20 genuine, S21-40 skilled forgery."""

    def _fixture(self, root):
        os.makedirs(root, exist_ok=True)
        for user in (1, 2):
            for sample in (1, 2, 21, 22):
                with open(os.path.join(root, f"U{user}S{sample}.TXT"), "w") as fh:
                    fh.write("2\n100 200 1000 0\n110 210 1010 1\n")
        return root

    def test_sample_index_determines_label(self, tmp_path):
        """S1-20 are genuine, S21-40 are the skilled forgeries."""
        writers = parse_svc2004(self._fixture(str(tmp_path)))
        assert set(writers) == {"svc_1", "svc_2"}
        assert len(writers["svc_1"]["genuine"]) == 2
        assert len(writers["svc_1"]["forged"]) == 2

    def test_returns_arrays_not_paths(self, tmp_path):
        """Temporal samples are loaded eagerly — sequences are small."""
        writers = parse_svc2004(self._fixture(str(tmp_path)))
        sample = writers["svc_1"]["genuine"][0]
        assert isinstance(sample, np.ndarray)
        assert sample.shape[1] == 3

    def test_pressure_detection(self, tmp_path):
        """Task1 datasets must be reported as pressure-less so training can warn."""
        assert svc2004_has_pressure(self._fixture(str(tmp_path))) is False


# ─── Real-dataset validation (skipped when ./data is not populated) ──────

_have = os.path.isdir


@pytest.mark.skipif(not _have(CEDAR_DIR), reason="CEDAR not downloaded")
def test_real_cedar_counts():
    """CEDAR ships 55 writers with 24 genuine + 24 forged each."""
    writers = parse_cedar(CEDAR_DIR)
    assert len(writers) == 55
    assert all(len(v["genuine"]) == 24 and len(v["forged"]) == 24
               for v in writers.values())


@pytest.mark.skipif(not _have(BHSIG260_DIR), reason="BHSig260 not downloaded")
def test_real_bhsig260_counts():
    """BHSig260 = 100 Bengali + 160 Hindi writers, 24 genuine + 30 forged each."""
    writers = parse_bhsig260(BHSIG260_DIR)
    assert len(writers) == 260
    assert sum(w.startswith("bhsig_Bengali_") for w in writers) == 100
    assert sum(w.startswith("bhsig_Hindi_") for w in writers) == 160
    assert all(len(v["genuine"]) == 24 and len(v["forged"]) == 30
               for v in writers.values())


@pytest.mark.skipif(not _have(UTSIG_DIR), reason="UTSig not downloaded")
def test_real_utsig_counts():
    """UTSig = 115 writers, 27 genuine + 6 skilled forgeries each."""
    writers = parse_utsig(UTSIG_DIR)
    assert len(writers) == 115
    assert all(len(v["genuine"]) == 27 and len(v["forged"]) == 6
               for v in writers.values())


@pytest.mark.parametrize("root", [SVC2004_TASK1_DIR, SVC2004_TASK2_DIR])
def test_real_svc2004_counts(root):
    """Both SVC2004 tasks = 40 users, 20 genuine + 20 skilled forgeries each."""
    if not os.path.isdir(root):
        pytest.skip(f"{root} not downloaded")

    writers = parse_svc2004(root)
    assert len(writers) == 40
    assert all(len(v["genuine"]) == 20 and len(v["forged"]) == 20
               for v in writers.values())
    sample = writers["svc_1"]["genuine"][0]
    assert sample.ndim == 2 and sample.shape[1] == 3
    assert np.all(np.isfinite(sample))


@pytest.mark.skipif(not _have(SVC2004_TASK1_DIR), reason="SVC2004 Task1 not downloaded")
def test_real_task1_has_no_pressure():
    """Task1 is the pressure-blind ablation baseline — every P must be the imputed 1.0."""
    assert svc2004_has_pressure(SVC2004_TASK1_DIR) is False
    sample = parse_svc2004(SVC2004_TASK1_DIR)["svc_1"]["genuine"][0]
    assert np.allclose(sample[:, 2], 1.0)


@pytest.mark.skipif(not _have(SVC2004_TASK2_DIR), reason="SVC2004 Task2 not downloaded")
def test_real_task2_has_real_pressure():
    """Task2 must yield genuinely varying pressure inside [0, 1], not a constant."""
    assert svc2004_has_pressure(SVC2004_TASK2_DIR) is True

    sample = parse_svc2004(SVC2004_TASK2_DIR)["svc_1"]["genuine"][0]
    pressure = sample[:, 2]
    assert pressure.min() >= 0.0 and pressure.max() <= 1.0
    assert pressure.std() > 0.0, "pressure is constant — the channel carries no signal"
    assert not np.allclose(pressure, 1.0)


class TestParseMobisig:
    """MOBISIG: USER<n>/SIGN_{GEN,FOR}_USER<owner>_USER<signer>_<k>.csv."""

    HEADER = ("x,y,timestamp,pressure,fingerarea,velocityx,velocityy,"
              "accelx,accely,accelz,gyrox,gyroy,gyroz")

    def _write(self, path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(self.HEADER + "\n")
            for x, y, p in rows:
                fh.write(f"{x},{y},1000,{p},0.1,0,0,0,0,0,0,0,0\n")
        return path

    def _fixture(self, root):
        rows = [(10.0, 20.0, 0.5), (11.0, 22.0, 0.6), (13.0, 25.0, 0.4)]
        for owner in (1, 2):
            for k in (1, 2):
                self._write(os.path.join(
                    root, f"USER{owner}", f"SIGN_GEN_USER{owner}_USER{owner}_{k}.csv"), rows)
            # A forgery of `owner`, produced by a different user.
            self._write(os.path.join(
                root, f"USER{owner}", f"SIGN_FOR_USER{owner}_USER9_1.csv"), rows)
        return root

    def test_groups_by_owner_not_by_forger(self, tmp_path):
        """A forgery belongs to the writer being imitated, never to whoever drew it."""
        writers = parse_mobisig(self._fixture(str(tmp_path)))
        assert set(writers) == {"mobisig_1", "mobisig_2"}
        assert len(writers["mobisig_1"]["genuine"]) == 2
        assert len(writers["mobisig_1"]["forged"]) == 1
        assert "mobisig_9" not in writers

    def test_returns_normalized_l3_trajectories(self, tmp_path):
        sample = parse_mobisig(self._fixture(str(tmp_path)))["mobisig_1"]["genuine"][0]
        assert sample.ndim == 2 and sample.shape[1] == 3
        assert np.allclose(sample[:, :2].mean(axis=0), 0.0, atol=1e-5)
        assert sample[:, 2].min() >= 0.0 and sample[:, 2].max() <= 1.0

    def test_ignores_non_signature_files(self, tmp_path):
        root = self._fixture(str(tmp_path))
        _touch(os.path.join(root, "users.csv"))
        _touch(os.path.join(root, "statistics.txt"))
        assert set(parse_mobisig(root)) == {"mobisig_1", "mobisig_2"}

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_mobisig("/nonexistent/mobisig")


@pytest.mark.skipif(not _have(MOBISIG_DIR), reason="MOBISIG not downloaded")
def test_real_mobisig_counts():
    """MOBISIG ships 83 users, each with 45 genuine and 20 skilled forgeries."""
    writers = parse_mobisig(MOBISIG_DIR)
    assert len(writers) == 83
    assert all(len(v["genuine"]) == 45 and len(v["forged"]) == 20
               for v in writers.values())

    sample = writers["mobisig_1"]["genuine"][0]
    assert sample.ndim == 2 and sample.shape[1] == 3
    assert np.all(np.isfinite(sample))
    # Capacitive screens report a real, varying contact pressure.
    assert sample[:, 2].std() > 0.0


class TestParseSigComp2011:
    """
    SigComp2011 online: two archives whose filename conventions are mirror
    images of each other, disambiguated only by the folder they sit in.
    """

    # Six raw 200 Hz rows. Decimation keeps indices 0, 2, 4; of those, index 0
    # is a pen-up sample (Z = 0) and is dropped, leaving two pen-down points.
    ROWS = [(27353, 24570, 0), (27349, 24574, 999),
            (27345, 24584, 512), (27341, 24590, 999),
            (27337, 24600, 1023), (27333, 24610, 999)]

    def _write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            for x, y, z in self.ROWS:
                fh.write(f"{x} {y} {z}\n")

    def _fixture(self, root):
        train = os.path.join(root, "trainingSet", "OnlineSignatures")
        # Training layout: writer first, and two different forgery conventions.
        self._write(os.path.join(train, "Dutch", "TrainingSet",
                                 "Online Genuine", "001_01.HWR"))
        self._write(os.path.join(train, "Dutch", "TrainingSet",
                                 "Online Forgeries", "170_001_02.HWR"))
        self._write(os.path.join(train, "Chinese", "TrainingSet",
                                 "Online Genuine", "001_1.HWR"))
        self._write(os.path.join(train, "Chinese", "TrainingSet",
                                 "Online Forgeries", "0120001_1.HWR"))

        # Test layout: index first, writer resolved from its own directory.
        test = os.path.join(root, "Testdata_SigComp2011", "SigComp11-Onlinetestset")
        self._write(os.path.join(test, "Dutch", "Reference(648)", "013", "13_013.HWR"))
        self._write(os.path.join(test, "Dutch", "Questioned(1259)", "013", "01_013.HWR"))
        self._write(os.path.join(test, "Dutch", "Questioned(1259)", "013",
                                 "01_182_013.HWR"))
        self._write(os.path.join(test, "Chinese", "Questioned(586)", "011",
                                 "01_011.HWR"))
        self._write(os.path.join(test, "Chinese", "Questioned(586)", "011",
                                 "01_0130011.HWR"))
        return root

    def test_writer_ids_are_language_scoped(self, tmp_path):
        """Dutch 013 and Chinese 013 are different people."""
        writers = parse_sigcomp2011(self._fixture(str(tmp_path)))
        assert set(writers) == {
            "sigcomp_Dutch_001", "sigcomp_Chinese_001",
            "sigcomp_Dutch_013", "sigcomp_Chinese_011",
        }

    def test_training_layout_reads_writer_before_index(self, tmp_path):
        writers = parse_sigcomp2011(self._fixture(str(tmp_path)))
        assert len(writers["sigcomp_Dutch_001"]["genuine"]) == 1
        assert len(writers["sigcomp_Dutch_001"]["forged"]) == 1

    def test_glued_chinese_forger_id_resolves_to_the_imitated_writer(self, tmp_path):
        """`0120001_1` is forger 0120 imitating writer 001, not writer 0120001."""
        writers = parse_sigcomp2011(self._fixture(str(tmp_path)))
        assert len(writers["sigcomp_Chinese_001"]["forged"]) == 1
        assert "sigcomp_Chinese_0120001" not in writers

    def test_test_layout_reads_index_before_writer(self, tmp_path):
        """
        `01_013.HWR` is genuine while the training set's `001_01.HWR` is also
        genuine with the tokens reversed — only the folder tells them apart.
        """
        writers = parse_sigcomp2011(self._fixture(str(tmp_path)))
        dutch = writers["sigcomp_Dutch_013"]
        assert len(dutch["genuine"]) == 2      # one Reference, one Questioned
        assert len(dutch["forged"]) == 1       # 01_182_013

    def test_questioned_forgeries_are_labelled_from_the_filename(self, tmp_path):
        chinese = parse_sigcomp2011(self._fixture(str(tmp_path)))["sigcomp_Chinese_011"]
        assert len(chinese["genuine"]) == 1
        assert len(chinese["forged"]) == 1

    def test_rearranged_test_directory_raises(self, tmp_path):
        """A file under the wrong writer directory means every label is suspect."""
        root = self._fixture(str(tmp_path))
        self._write(os.path.join(root, "Testdata_SigComp2011",
                                 "SigComp11-Onlinetestset", "Dutch",
                                 "Reference(648)", "013", "13_099.HWR"))
        with pytest.raises(ValueError, match="not as expected"):
            parse_sigcomp2011(root)

    def test_returns_normalized_l3_trajectories(self, tmp_path):
        sample = parse_sigcomp2011(self._fixture(str(tmp_path)))[
            "sigcomp_Dutch_001"]["genuine"][0]
        assert sample.ndim == 2 and sample.shape[1] == 3
        assert np.allclose(sample[:, :2].mean(axis=0), 0.0, atol=1e-5)
        # Z = 0..1023 device units scaled onto [0, 1], not renormalized per file.
        assert np.allclose(sample[:, 2], [512 / 1023.0, 1.0], atol=1e-6)

    def test_decimates_to_svc2004s_sample_rate(self, tmp_path):
        """200 Hz halved to 100 Hz, so the derived velocity channels agree."""
        sample = parse_sigcomp2011(self._fixture(str(tmp_path)))[
            "sigcomp_Dutch_001"]["genuine"][0]
        # 6 raw rows -> 3 after decimation -> 2 after the pen-up row is dropped.
        assert len(sample) == 2

    def test_drops_pen_up_samples(self, tmp_path):
        """
        SVC2004 Task2 has no zero-pressure points, so neither may this dataset.
        A surviving P = 0 would make pressure mean something different per
        writer_samples for the one encoder that reads both.
        """
        writers = parse_sigcomp2011(self._fixture(str(tmp_path)))
        for record in writers.values():
            for sample in record["genuine"] + record["forged"]:
                assert sample[:, 2].min() > 0.0

    def test_file_with_no_pen_down_samples_raises(self, tmp_path):
        """Silently yielding an empty trajectory would poison a training pair."""
        root = self._fixture(str(tmp_path))
        path = os.path.join(root, "trainingSet", "OnlineSignatures", "Dutch",
                            "TrainingSet", "Online Genuine", "002_01.HWR")
        with open(path, "w") as fh:
            fh.write("100 200 0\n101 201 0\n102 202 0\n")
        with pytest.raises(ValueError, match="fewer than 2 pen-down samples"):
            parse_sigcomp2011(root)

    def test_ignores_offline_images_and_stray_files(self, tmp_path):
        root = self._fixture(str(tmp_path))
        _touch(os.path.join(root, "trainingSet", "OfflineSignatures", "Dutch",
                            "TrainingSet", "Offline Genuine", "001_01.png"))
        _touch(os.path.join(root, "readme.txt"))
        assert len(parse_sigcomp2011(root)) == 4

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_sigcomp2011("/nonexistent/sigcomp2011")


@pytest.mark.skipif(not _have(SIGCOMP2011_DIR), reason="SigComp2011 not downloaded")
def test_real_sigcomp2011_counts():
    """
    The online writer_samples ships 84 writers and 3695 signatures: 64 Dutch (10 in the
    training archive, 54 in the test archive) and 20 Chinese (10 + 10).
    """
    writers = parse_sigcomp2011(SIGCOMP2011_DIR)
    assert len(writers) == 84

    dutch = [w for w in writers if w.startswith("sigcomp_Dutch_")]
    chinese = [w for w in writers if w.startswith("sigcomp_Chinese_")]
    assert len(dutch) == 64 and len(chinese) == 20

    total = sum(len(v["genuine"]) + len(v["forged"]) for v in writers.values())
    assert total == 3695

    # Every writer must be scoreable: skilled pairing needs two genuines and a forgery.
    assert all(len(v["genuine"]) >= 2 and len(v["forged"]) >= 1
               for v in writers.values())

    sample = writers["sigcomp_Dutch_013"]["genuine"][0]
    assert sample.ndim == 2 and sample.shape[1] == 3
    assert np.all(np.isfinite(sample))
    # A real pen sensor, not the flat P = 1.0 of a pressure-blind dataset.
    assert sample[:, 2].std() > 0.0
    assert not np.allclose(sample[:, 2], 1.0)


class TestParseDeepSignDB:
    """
    DeepSignDB: `DeepSignDB/{split}/{modality}/u{id}_{g|s}_*.txt`.

    Every value below is invented. DeepSignDB is licensed and may not be
    redistributed, so no real sample is reproduced here — not even a partial one.
    The azimuth/altitude constants 3600 and 900 are the published WACOM maxima
    already hard-coded in `parsers.py`, not values taken from any signature.

    The fixture carries one writer per source database, so the per-source
    pressure column, the per-file sample rate and the source filter are each
    exercised against a file that would parse wrongly under another source's rules.
    """

    def _fixture(self, root):
        stylus = os.path.join(root, "DeepSignDB", "Development", "stylus")
        os.makedirs(stylus, exist_ok=True)

        # MCYT: 6 columns (x, y, t, azimuth, altitude, P), already 100 Hz.
        # Row 3 is pen-up. Azimuth and altitude sit at their documented maxima so
        # that reading the wrong column yields a pressure far outside [0, 1].
        mcyt = ("4\n"
                "10 20 0 3600 900 512\n"
                "11 21 10 3600 900 1023\n"
                "12 22 20 3600 900 0\n"
                "13 23 30 3600 900 256\n")
        for label in ("g", "s"):
            with open(os.path.join(stylus, f"u0001_{label}_synthetic.txt"), "w") as fh:
                fh.write(mcyt)

        # BiosecurID: 7 columns (x, y, t, button, azimuth, altitude, P) at
        # 200 Hz. Even-indexed rows survive decimation; one of those is pen-up.
        biosecurid = ("6\n"
                      "10 20 0 1 1500 700 512\n"
                      "99 99 5 1 1500 700 999\n"
                      "11 21 10 1 1500 700 1023\n"
                      "99 99 15 1 1500 700 999\n"
                      "12 22 20 1 1500 700 0\n"
                      "99 99 25 1 1500 700 999\n")
        for label in ("g", "s"):
            with open(os.path.join(stylus, f"u0231_{label}_synthetic.txt"), "w") as fh:
                fh.write(biosecurid)

        # e-BioSign: 4 columns (x, y, t, P). Excluded by the default sources.
        with open(os.path.join(stylus, "u1009_g_synthetic_w1.txt"), "w") as fh:
            fh.write("2\n30 40 1000 200\n31 41 1010 300\n")

        # The same integer ID in the other half — a different person.
        evaluation = os.path.join(root, "DeepSignDB", "Evaluation", "stylus")
        os.makedirs(evaluation, exist_ok=True)
        with open(os.path.join(evaluation, "u0001_g_synthetic.txt"), "w") as fh:
            fh.write(mcyt)

        return root

    def test_writer_ids_carry_source_and_split(self, tmp_path):
        writers = parse_deepsigndb(self._fixture(str(tmp_path)))
        assert set(writers) == {"deepsign_MCYT-dev_0001",
                                "deepsign_BiosecurID-dev_0231"}

    def test_g_and_s_become_genuine_and_forged(self, tmp_path):
        record = parse_deepsigndb(self._fixture(str(tmp_path)))["deepsign_MCYT-dev_0001"]
        assert len(record["genuine"]) == 1 and len(record["forged"]) == 1

    def test_default_sources_exclude_ebiosign(self, tmp_path):
        """
        Its five capture devices disagree on the force maximum, so there is no
        fixed divisor to normalize it with. Silently including it would put
        signatures on a different pressure scale than every other writer.
        """
        writers = parse_deepsigndb(self._fixture(str(tmp_path)))
        assert not any("eBioSign" in w for w in writers)

        widened = parse_deepsigndb(self._fixture(str(tmp_path)),
                                   sources=("MCYT", "eBioSign_DS1"))
        assert "deepsign_eBioSign_DS1-dev_1009" in widened

    def test_development_and_evaluation_never_collide(self, tmp_path):
        """
        `Development/u0001` and `Evaluation/u0001` are two different people —
        MCYT's 330 signers are split 230/100 between the halves. One shared
        writer ID would leak a training writer into the held-out set, and it
        would do so silently, by *improving* the reported EER.
        """
        root = self._fixture(str(tmp_path))
        development = parse_deepsigndb(root, split="Development")
        evaluation = parse_deepsigndb(root, split="Evaluation")

        assert "deepsign_MCYT-dev_0001" in development
        assert "deepsign_MCYT-eval_0001" in evaluation
        assert not set(development) & set(evaluation)

    def test_pressure_comes_from_the_source_specific_column(self, tmp_path):
        """
        MCYT keeps pressure in column 6 and BiosecurID in column 7. Reading a
        fixed column would return azimuth instead — 3600, or 3.52 once scaled.
        """
        writers = parse_deepsigndb(self._fixture(str(tmp_path)))
        mcyt = writers["deepsign_MCYT-dev_0001"]["genuine"][0]
        biosecurid = writers["deepsign_BiosecurID-dev_0231"]["genuine"][0]

        assert np.allclose(mcyt[:, 2], [512 / 1023.0, 1.0, 256 / 1023.0], atol=1e-6)
        assert np.allclose(biosecurid[:, 2], [512 / 1023.0, 1.0], atol=1e-6)

    def test_resamples_each_file_by_its_own_rate(self, tmp_path):
        """
        BiosecurID mixes 100 Hz and 200 Hz captures, so the factor is measured
        per file. The 200 Hz fixture halves; the 100 Hz one is left alone.
        """
        writers = parse_deepsigndb(self._fixture(str(tmp_path)))
        # MCYT: 4 rows at 100 Hz, none dropped, minus one pen-up -> 3.
        assert len(writers["deepsign_MCYT-dev_0001"]["genuine"][0]) == 3
        # BiosecurID: 6 rows at 200 Hz -> 3 after halving, minus one pen-up -> 2.
        assert len(writers["deepsign_BiosecurID-dev_0231"]["genuine"][0]) == 2

    def test_drops_pen_up_samples(self, tmp_path):
        """SVC2004 Task2 has no zero-pressure points, so neither may this one."""
        for record in parse_deepsigndb(self._fixture(str(tmp_path))).values():
            for sample in record["genuine"] + record["forged"]:
                assert sample[:, 2].min() > 0.0

    def test_file_with_no_pen_down_samples_raises(self, tmp_path):
        """Silently yielding an empty trajectory would poison a training pair."""
        root = self._fixture(str(tmp_path))
        path = os.path.join(root, "DeepSignDB", "Development", "stylus",
                            "u0002_g_synthetic.txt")
        with open(path, "w") as fh:
            fh.write("3\n10 20 0 3600 900 0\n11 21 10 3600 900 0\n"
                     "12 22 20 3600 900 0\n")
        with pytest.raises(ValueError, match="fewer than 2 pen-down samples"):
            parse_deepsigndb(root)

    def test_column_count_below_the_pressure_column_raises(self, tmp_path):
        """
        A 4-column file in MCYT's ID range means the archive is not laid out as
        documented, and every label below it is then suspect.
        """
        root = self._fixture(str(tmp_path))
        path = os.path.join(root, "DeepSignDB", "Development", "stylus",
                            "u0003_g_synthetic.txt")
        with open(path, "w") as fh:
            fh.write("2\n10 20 0 512\n11 21 10 1023\n")
        with pytest.raises(ValueError, match="does not match its documented source"):
            parse_deepsigndb(root)

    def test_unknown_source_or_split_raises(self, tmp_path):
        root = self._fixture(str(tmp_path))
        with pytest.raises(ValueError, match="Unknown DeepSignDB source"):
            parse_deepsigndb(root, sources=("MCYT", "GPDS960"))
        with pytest.raises(ValueError, match="Unknown DeepSignDB split"):
            parse_deepsigndb(root, split="Testing")

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_deepsigndb("/nonexistent/deepsigndb")

    def test_source_ranges_differ_between_the_two_halves(self):
        """
        u0300 is BiosecurID in one half and BiosecureDS2 in the other, and they
        keep pressure in different columns. The split tag is not decorative.
        """
        assert deepsigndb_source(100, "Development") == "MCYT"
        assert deepsigndb_source(100, "Evaluation") == "MCYT"
        assert deepsigndb_source(231, "Development") == "BiosecurID"
        assert deepsigndb_source(231, "Evaluation") == "BiosecurID"
        assert deepsigndb_source(300, "Development") == "BiosecurID"
        assert deepsigndb_source(300, "Evaluation") == "BiosecureDS2"
        assert deepsigndb_source(9999, "Development") is None


@pytest.mark.skipif(not _have(DEEPSIGNDB_DIR), reason="DeepSignDB not licensed/downloaded")
def test_real_deepsigndb_counts():
    """
    The Development stylus half ships 498 pen writers under the default sources:
    MCYT's 230 (25 genuine + 25 skilled forgeries each) and BiosecurID's 268
    (16 + 12 each), for 19,004 signatures.
    """
    writers = parse_deepsigndb(DEEPSIGNDB_DIR)
    assert len(writers) == 498

    mcyt = {w: v for w, v in writers.items() if w.startswith("deepsign_MCYT-dev_")}
    biosecurid = {w: v for w, v in writers.items()
                  if w.startswith("deepsign_BiosecurID-dev_")}
    assert len(mcyt) == 230 and len(biosecurid) == 268
    assert all(len(v["genuine"]) == 25 and len(v["forged"]) == 25
               for v in mcyt.values())
    assert all(len(v["genuine"]) == 16 and len(v["forged"]) == 12
               for v in biosecurid.values())

    total = sum(len(v["genuine"]) + len(v["forged"]) for v in writers.values())
    assert total == 19004

    sample = writers["deepsign_MCYT-dev_0001"]["genuine"][0]
    assert sample.ndim == 2 and sample.shape[1] == 3
    assert np.all(np.isfinite(sample))
    # A real pen sensor, not the flat P = 1.0 of a pressure-blind dataset.
    assert sample[:, 2].std() > 0.0 and sample[:, 2].min() > 0.0


@pytest.mark.skipif(not _have(DEEPSIGNDB_DIR), reason="DeepSignDB not licensed/downloaded")
def test_real_deepsigndb_evaluation_is_disjoint_from_development():
    """
    The Evaluation half is the authors' held-out benchmark: 372 writers reusing
    Development's integers for different people. Nothing may be shared between
    the halves but the numbers themselves.
    """
    development = parse_deepsigndb(DEEPSIGNDB_DIR, split="Development")
    evaluation = parse_deepsigndb(DEEPSIGNDB_DIR, split="Evaluation")
    assert len(evaluation) == 372
    assert not set(development) & set(evaluation)


class TestMergeDatasets:
    """Merging datasets that share writer IDs must append samples, not replace them."""

    def test_disjoint_writers_are_unioned(self):
        target = {"svc_1": {"genuine": ["a"], "forged": ["b"]}}
        merge_datasets(target, {"mobisig_1": {"genuine": ["c"], "forged": ["d"]}})
        assert set(target) == {"svc_1", "mobisig_1"}

    def test_shared_writer_id_appends_rather_than_overwrites(self):
        """
        SVC2004 Task1 and Task2 are different recordings of the same 40 signers,
        so both parse to `svc_1`..`svc_40`. A dict update would silently discard
        one task's samples entirely.
        """
        target = {"svc_1": {"genuine": ["t1_g"], "forged": ["t1_f"]}}
        merge_datasets(target, {"svc_1": {"genuine": ["t2_g"], "forged": ["t2_f"]}})
        assert target["svc_1"]["genuine"] == ["t1_g", "t2_g"]
        assert target["svc_1"]["forged"] == ["t1_f", "t2_f"]

    def test_returns_the_target_for_chaining(self):
        target = {}
        assert merge_datasets(target, {"svc_1": {"genuine": [], "forged": []}}) is target

    def test_merging_is_order_independent_in_count(self):
        a = {"svc_1": {"genuine": ["g1"], "forged": ["f1"]}}
        b = {"svc_1": {"genuine": ["g2", "g3"], "forged": ["f2"]}}
        forward, backward = {}, {}
        merge_datasets(merge_datasets(forward, a), b)
        merge_datasets(merge_datasets(backward, b), a)
        assert len(forward["svc_1"]["genuine"]) == len(backward["svc_1"]["genuine"]) == 3
        assert len(forward["svc_1"]["forged"]) == len(backward["svc_1"]["forged"]) == 2


class TestTemporalGapChannel:
    """
    The `dt` channel has to mean the same thing in every corpus that supplies it.

    SVC2004 carries a real millisecond clock. SigComp2011 carries none, but it
    samples on a fixed 200 Hz grid and records pen-up travel as force = 0, so the
    time the pen spent in the air is exactly the length of the run the loader
    drops. Recovering it that way is what makes the channel comparable across the
    two corpora instead of a constant that merely identifies which one a sample
    came from.
    """

    # Eight decimated rows: down, down, three in the air, down, down, down.
    # Written at 200 Hz, so exactly every second raw row survives decimation.
    DECIMATED_FORCE = [500, 500, 0, 0, 0, 500, 500, 500]

    def _write_hwr(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            for i, force in enumerate(self.DECIMATED_FORCE):
                for _ in range(SIGCOMP_DECIMATION):
                    fh.write(f"{27000 + i} {24000 + 2 * i} {force}\n")

    def _write_svc_task2(self, path, timestamps):
        with open(path, "w") as fh:
            fh.write(f"{len(timestamps)}\n")
            for i, t in enumerate(timestamps):
                fh.write(f"{3000 + i} {2000 + 2 * i} {t} 1 1800 450 512\n")

    def test_dt_measures_the_dropped_pen_up_run(self, tmp_path):
        """A three-sample lift is a four-row hop, so 40 ms at 10 ms per row."""
        path = os.path.join(tmp_path, "001_01.HWR")
        self._write_hwr(path)

        out = load_sigcomp2011_file(path, extra_channels="dt")

        assert out.shape == (5, 4)
        np.testing.assert_allclose(out[:, 3], [0.0, 0.1, 0.4, 0.1, 0.1], atol=1e-6)

    def test_both_corpora_scale_the_same_pause_identically(self, tmp_path):
        """
        The point of the channel is cross-corpus comparability. A 40 ms pause has
        to arrive at the model as the same number whether it was measured by
        SVC2004's clock or reconstructed from SigComp2011's dropped samples.
        """
        hwr = os.path.join(tmp_path, "001_01.HWR")
        self._write_hwr(hwr)
        svc = os.path.join(tmp_path, "U1S1.TXT")
        self._write_svc_task2(svc, [0, 10, 50, 60, 70])

        from_sigcomp = load_sigcomp2011_file(hwr, extra_channels="dt")[:, 3]
        from_svc = load_svc2004_file(svc, extra_channels="dt")[:, 3]

        np.testing.assert_allclose(from_sigcomp, from_svc, atol=1e-6)

    def test_dt_varies_within_a_signature(self, tmp_path):
        """
        Guards the failure already on the do-not-retry list: padding a corpus
        with a constant channel teaches the model which corpus a sample came
        from, not who wrote it. A dt that never moves is exactly that pad.
        """
        path = os.path.join(tmp_path, "001_01.HWR")
        self._write_hwr(path)

        dt = load_sigcomp2011_file(path, extra_channels="dt")[:, 3]

        assert dt.std() > 0.0

    def test_sigcomp_refuses_tilt(self, tmp_path):
        """
        The archive ships three columns. Tilt cannot be invented from them, and
        silently returning three channels would let a 5-channel run score a
        4-channel corpus without ever failing.
        """
        path = os.path.join(tmp_path, "001_01.HWR")
        self._write_hwr(path)

        with pytest.raises(ValueError, match="tilt"):
            load_sigcomp2011_file(path, extra_channels="tilt")

    def test_default_stays_three_channels(self, tmp_path):
        """Every existing checkpoint reads [L, 3]; the default must not move."""
        path = os.path.join(tmp_path, "001_01.HWR")
        self._write_hwr(path)

        assert load_sigcomp2011_file(path).shape == (5, 3)

    def test_parse_threads_the_channel_through(self, tmp_path):
        """The width has to survive the walk, not just the single-file loader."""
        root = os.path.join(tmp_path, "sigcomp2011")
        self._write_hwr(os.path.join(root, "trainingSet", "OnlineSignatures",
                                     "Dutch", "TrainingSet", "Online Genuine",
                                     "001_01.HWR"))

        writers = parse_sigcomp2011(root, extra_channels="dt")

        assert writers["sigcomp_Dutch_001"]["genuine"][0].shape[1] == 4
