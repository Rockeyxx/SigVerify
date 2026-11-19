"""The in-house campaign parser: pairing, OS per file, the app's stroke rule, and the split."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QTabletEvent  # noqa: E402

from src.capture.tablet import SignatureCaptureWidget  # noqa: E402
from src.data.campaign import (inhouse_split, load_inhouse_tablet_file,  # noqa: E402
                              parse_inhouse, write_inhouse_split)
from tools.capture_session import (append_manifest, ink_path, signature_path,  # noqa: E402
                                   write_raw_events)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _events(n=40, t0=1000.0):
    """A plain stroke as the canvas logs it: (t_ms, kind, x, y, p, xtilt, ytilt, inside)."""
    kinds = [int(QEvent.TabletPress)] + [int(QEvent.TabletMove)] * (n - 1)
    rows = [(t0 + 4 * i, k, 100.0 + i, 200.0 + 3 * np.sin(i / 5), 0.5, 0.0, 0.0, 1)
            for i, k in enumerate(kinds)]
    return rows + [(t0 + 4 * n, int(QEvent.TabletRelease), 100.0 + n, 200.0, 0.0, 0.0, 0.0, 1)]


def _tablet(root, code, role, os_name, index=None, forger=None, attempt=None):
    path = signature_path(root, code, "tablet", role, index=index, forger=forger, attempt=attempt)
    write_raw_events(path, _events(), root)
    append_manifest(root, {"file": str(path.relative_to(root)), "participant": code,
                           "modality": "tablet", "role": role, "index": index or "",
                           "forger": forger or "", "attempt": attempt or "", "os": os_name})
    return path


def _paper(root, code, role, index=None, forger=None, attempt=None):
    path = signature_path(root, code, "paper", role, index=index, forger=forger, attempt=attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((40, 60), 255, np.uint8))
    cv2.imwrite(str(ink_path(path)), np.full((20, 30), 255, np.uint8))   # beside every box
    append_manifest(root, {"file": str(path.relative_to(root)), "participant": code,
                           "modality": "paper", "role": role, "index": index or "",
                           "forger": forger or "", "attempt": attempt or "", "os": "Linux"})
    return path


def test_genuines_pair_by_index_and_a_missing_side_stays_none(tmp_path):
    for i in (1, 2, 3):
        _tablet(tmp_path, "P05", "genuine", "Windows", index=i)
    for i in (1, 2, 4):
        _paper(tmp_path, "P05", "genuine", index=i)
    _paper(tmp_path, "P05", "forgercopy")

    genuine = parse_inhouse(str(tmp_path))["inhouse_P05"]["genuine"]
    assert [g["index"] for g in genuine] == [1, 2, 3, 4]
    assert [g["tablet"] is not None for g in genuine] == [True, True, True, False]
    assert [g["paper"] is not None for g in genuine] == [True, True, False, True]
    assert genuine[1]["tablet"].endswith("P05_tablet_genuine_02.csv")
    assert genuine[1]["paper"].endswith("P05_paper_genuine_02.png")      # the raw box, not _ink


def test_forgeries_pair_by_forger_and_attempt(tmp_path):
    _tablet(tmp_path, "P05", "genuine", "Windows", index=1)
    _tablet(tmp_path, "P05", "forgery", "Windows", forger="P02", attempt=1)
    _tablet(tmp_path, "P05", "forgery", "Windows", forger="P02", attempt=2)
    _paper(tmp_path, "P05", "forgery", forger="P02", attempt=2)
    _paper(tmp_path, "P05", "forgery", forger="P03", attempt=1)

    forged = parse_inhouse(str(tmp_path))["inhouse_P05"]["forged"]
    assert [(f["forger"], f["attempt"]) for f in forged] == [("P02", 1), ("P02", 2), ("P03", 1)]
    assert [(f["tablet"] is not None, f["paper"] is not None) for f in forged] == [
        (True, False), (True, True), (False, True)]


def test_os_comes_from_each_tablet_files_own_manifest_row(tmp_path):
    """Windows pressure is on another scale (G10), and a forgery may be recorded elsewhere."""
    _tablet(tmp_path, "P05", "genuine", "Windows", index=1)
    _tablet(tmp_path, "P05", "forgery", "Linux", forger="P02", attempt=1)
    _paper(tmp_path, "P05", "genuine", index=2)

    writer = parse_inhouse(str(tmp_path))["inhouse_P05"]
    assert [g["os"] for g in writer["genuine"]] == ["Windows", None]      # paper-only: no pen OS
    assert writer["forged"][0]["os"] == "Linux"


def test_a_manifest_written_on_windows_still_gives_the_os(tmp_path):
    """The Windows sessions saved backslash paths; they must still match their files."""
    _tablet(tmp_path, "P05", "genuine", "Windows", index=1)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(manifest.read_text().replace("P05/", "P05\\"))
    assert parse_inhouse(str(tmp_path))["inhouse_P05"]["genuine"][0]["os"] == "Windows"


def test_participants_limits_what_is_read(tmp_path):
    _tablet(tmp_path, "P05", "genuine", "Windows", index=1)
    _tablet(tmp_path, "P07", "genuine", "Windows", index=1)
    assert list(parse_inhouse(str(tmp_path), participants=["P07"])) == ["inhouse_P07"]


def test_excluded_participants_are_never_read(tmp_path):
    """The owner dropped some sessions as bad (INHOUSE_EXCLUDED); they enter no number."""
    _tablet(tmp_path, "P05", "genuine", "Windows", index=1)
    _tablet(tmp_path, "P06", "genuine", "Windows", index=1)
    assert list(parse_inhouse(str(tmp_path), exclude=("P06",))) == ["inhouse_P05"]


def test_forgeries_of_no_driver_writers_are_scaled_to_their_genuines(tmp_path):
    """Their genuines went through Windows Ink without the driver, their forgeries through
    Wintab with it (G10: about 0.4 of the pressure); the forgery is brought to the genuines'
    scale, never the other way, so nothing clips."""
    _tablet(tmp_path, "P10", "genuine", "Windows", index=1)
    _tablet(tmp_path, "P10", "forgery", "Windows", forger="P01", attempt=1)
    _tablet(tmp_path, "P14", "forgery", "Windows", forger="P01", attempt=1)
    writers = parse_inhouse(str(tmp_path), no_driver=("P10",), no_driver_scale=0.4)
    assert writers["inhouse_P10"]["genuine"][0]["pressure_scale"] == 1.0
    assert writers["inhouse_P10"]["forged"][0]["pressure_scale"] == 0.4
    assert writers["inhouse_P14"]["forged"][0]["pressure_scale"] == 1.0


def test_pressure_scale_multiplies_the_loaded_pressure(tmp_path):
    path = _tablet(tmp_path, "P10", "genuine", "Windows", index=1)
    full = load_inhouse_tablet_file(str(path))
    scaled = load_inhouse_tablet_file(str(path), pressure_scale=0.4)
    assert np.allclose(scaled[:, 2], 0.4 * full[:, 2])
    assert np.array_equal(np.delete(scaled, 2, axis=1), np.delete(full, 2, axis=1))


def test_a_file_in_another_participants_folder_raises(tmp_path):
    path = _tablet(tmp_path, "P05", "genuine", "Windows", index=1)
    stray = tmp_path / "P07" / path.name
    stray.parent.mkdir()
    path.rename(stray)
    with pytest.raises(ValueError, match="P07"):
        parse_inhouse(str(tmp_path))


def _qt_event(kind, x, y, pressure, ts):
    ev = QTabletEvent(kind, QPointF(x, y), QPointF(x, y), QTabletEvent.Stylus, QTabletEvent.Pen,
                      pressure, 0, 0, 0.0, 0.0, 0, Qt.NoModifier, 1, Qt.LeftButton, Qt.LeftButton)
    ev.setTimestamp(int(ts))
    return ev


def test_a_saved_signature_loads_exactly_as_the_signing_box_made_it(app, tmp_path):
    """Two strokes, a zero-pressure sample inside the box, and a swing out of the box and
    back: the parser must rebuild the same strokes the canvas did, so training data is what
    the app sees."""
    widget = SignatureCaptureWidget()
    canvas = widget.canvas
    w = canvas.width()
    t = 1000
    for k in range(2):
        for i in range(50):
            kind = QEvent.TabletPress if i == 0 else QEvent.TabletMove
            pressure = 0.0 if i == 20 else 0.3 + 0.01 * i
            canvas.tabletEvent(_qt_event(kind, 80 + 40 * k + 2 * i, 120 + 15 * np.sin(i / 6),
                                         pressure, t))
            t += 4
        canvas.tabletEvent(_qt_event(QEvent.TabletRelease, 200, 120, 0.0, t))
        t += 300
    xs = [w - 60 + 10 * i for i in range(12)] + [w - 10 * i for i in range(1, 12)]
    for i, x in enumerate(xs):                                  # leaves the box and comes back
        canvas.tabletEvent(_qt_event(QEvent.TabletPress if i == 0 else QEvent.TabletMove,
                                     x, 150, 0.5, t))
        t += 4
    canvas.tabletEvent(_qt_event(QEvent.TabletRelease, xs[-1], 150, 0.0, t))

    path = tmp_path / "P05" / "P05_tablet_genuine_01.csv"
    write_raw_events(path, canvas.raw_events, tmp_path)
    expected = widget.capture_signature()
    assert np.array_equal(load_inhouse_tablet_file(str(path)), expected)


def test_a_signature_with_no_ink_inside_the_box_raises(tmp_path):
    path = tmp_path / "P05" / "P05_tablet_genuine_01.csv"
    hover = [(1000.0 + 4 * i, int(QEvent.TabletMove), 100.0, 100.0, 0.0, 0.0, 0.0, 1)
             for i in range(10)]
    write_raw_events(path, hover, tmp_path)
    with pytest.raises(ValueError, match="no pen-down"):
        load_inhouse_tablet_file(str(path))


def _writers(codes, forgeries=()):
    """parse_inhouse's shape without files: forgeries are (victim, forger) pairs."""
    out = {f"inhouse_{c}": {"genuine": [], "forged": []} for c in codes}
    for victim, forger in forgeries:
        out[f"inhouse_{victim}"]["forged"].append({"forger": forger, "attempt": 1})
    return out


def test_every_forger_is_a_test_writer():
    """A forger's own hand in training would teach the model their forgeries."""
    codes = [f"P{i:02d}" for i in range(1, 24)]
    forgers = ["P01", "P02", "P03", "P04"]
    split = inhouse_split(_writers(codes, [("P10", f) for f in forgers]))
    assert {f"inhouse_{f}" for f in forgers} <= set(split["test"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert sorted(split["train"] + split["test"]) == sorted(f"inhouse_{c}" for c in codes)


def test_a_writer_keeps_their_side_when_others_join():
    first = inhouse_split(_writers([f"P{i:02d}" for i in range(5, 40, 2)]))
    later = inhouse_split(_writers([f"P{i:02d}" for i in range(1, 40)]))   # some sort between
    for side in ("train", "test"):
        assert set(first[side]) <= set(later[side])


def test_about_half_of_the_volunteers_are_test():
    split = inhouse_split(_writers([f"P{i:03d}" for i in range(400)]))
    assert 0.4 < len(split["test"]) / 400 < 0.6
    assert split["train"], "nobody left to train on"


def test_the_split_record_lists_both_sides_and_its_rule(tmp_path):
    writers = _writers(["P05", "P06", "P07"], [("P05", "P07")])
    record = write_inhouse_split(writers, str(tmp_path / "inhouse_split.json"))
    saved = json.loads((tmp_path / "inhouse_split.json").read_text())
    assert saved == record
    assert "inhouse_P07" in saved["test"] and saved["forgers"] == ["inhouse_P07"]
    assert {"seed", "test_fraction", "rule"} <= set(saved)
