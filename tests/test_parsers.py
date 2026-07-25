"""
Tests for the real-dataset parsers.

Two layers:
  1. Synthetic-fixture tests — always run, pin the naming/label logic.
  2. Real-corpus tests — skipped unless ./data is populated, and assert the
     exact writer/sample counts each dataset is documented to have.
"""

import os

import numpy as np
import pytest

from config.settings import (
    BHSIG260_DIR,
    CEDAR_DIR,
    MOBISIG_DIR,
    SVC2004_TASK1_DIR,
    SVC2004_TASK2_DIR,
    UTSIG_DIR,
)
from src.data.parsers import (
    load_svc2004_file,
    parse_bhsig260,
    parse_cedar,
    parse_mobisig,
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
        """Task1 corpora must be reported as pressure-less so training can warn."""
        assert svc2004_has_pressure(self._fixture(str(tmp_path))) is False


# ─── Real-corpus validation (skipped when ./data is not populated) ──────

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
