"""The tablet signing window the GUI calls: count, done screen, confirm, start over, cancel."""
import os

import numpy as np
import pytest

from config.settings import TEMPORAL_INPUT_FEATURES

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QTabletEvent  # noqa: E402

from src.capture import collect  # noqa: E402
from src.capture.collect import CollectDialog, collect_signatures  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _event(kind, x, y, pressure, ts):
    ev = QTabletEvent(kind, QPointF(x, y), QPointF(x, y), QTabletEvent.Stylus, QTabletEvent.Pen,
                      pressure, 0, 0, 0.0, 0.0, 0, Qt.NoModifier, 1, Qt.LeftButton, Qt.LeftButton)
    ev.setTimestamp(int(ts))
    return ev


def _sign(dialog):
    """Three strokes at 4 ms inside the box, then 'I approve' (as in test_capture_session)."""
    canvas = dialog.capture.canvas
    for k, t0 in enumerate((1000, 1400, 1800)):
        for i in range(60):
            kind = QEvent.TabletPress if i == 0 else QEvent.TabletMove
            canvas.tabletEvent(_event(kind, 100.0 + 40 * k + 0.6 * i, 200.0 + 30 * np.sin(i / 9),
                                      0.45, t0 + 4 * i))
        canvas.tabletEvent(_event(QEvent.TabletRelease, 300, 200, 0.0, t0 + 240))
    dialog.capture.confirm_signature()


def test_five_signatures_reach_the_done_screen_counting_on_the_way(app):
    d = CollectDialog(count=5)
    seen = []
    for _ in range(5):
        seen.append(d.counter.text())
        assert d.done.isHidden() and not d.capture.isHidden()
        _sign(d)
    assert seen == [f"Signature {i} of 5" for i in range(1, 6)]
    assert d.finished
    assert not d.done.isHidden() and d.capture.isHidden()
    assert d.done.text() == "Thank you, 5/5 done."


def test_confirm_returns_five_arrays_of_the_capture_width(app):
    d = CollectDialog(count=5)
    for _ in range(5):
        _sign(d)
    d.confirm()
    assert d.result() == QtWidgets.QDialog.Accepted
    assert len(d.signatures) == 5
    assert all(s.ndim == 2 and s.shape[1] == TEMPORAL_INPUT_FEATURES for s in d.signatures)


def test_a_rejected_signature_does_not_count(app):
    d = CollectDialog(count=5)
    d.capture.confirm_signature()          # empty box: the widget rejects it
    assert d.signatures == []
    assert d.counter.text() == "Signature 1 of 5"


def test_confirm_before_the_last_signature_does_nothing(app):
    d = CollectDialog(count=5)
    _sign(d)
    d.confirm()
    assert d.result() != QtWidgets.QDialog.Accepted


def test_start_over_drops_everything(app):
    d = CollectDialog(count=5)
    for _ in range(5):
        _sign(d)
    d.start_over()
    assert d.signatures == []
    assert d.counter.text() == "Signature 1 of 5"
    assert d.done.isHidden() and not d.capture.isHidden()


def test_collect_signatures_returns_the_list_or_none(app, monkeypatch):
    def sign_and_confirm(self):
        _sign(self)
        self.confirm()
        return self.result()

    monkeypatch.setattr(collect.CollectDialog, "exec_", sign_and_confirm)
    got = collect_signatures(count=1)
    assert isinstance(got, list) and len(got) == 1          # count=1 is the verification probe
    assert got[0].shape[1] == TEMPORAL_INPUT_FEATURES

    def close(self):
        self.reject()
        return self.result()

    monkeypatch.setattr(collect.CollectDialog, "exec_", close)
    assert collect_signatures(count=5) is None


def test_the_arabic_toggle_translates_and_right_aligns(app):
    d = CollectDialog(count=5)
    d.toggle_language()
    assert d.counter.text() == "التوقيع 1 من 5"
    assert d.counter.layoutDirection() == Qt.RightToLeft
    assert d.capture.btn_confirm.text() == "أوافق"
    d.toggle_language()
    assert d.counter.text() == "Signature 1 of 5"
    assert d.counter.layoutDirection() == Qt.LeftToRight


def test_the_window_uses_the_gui_theme(app):
    from src.gui.styles import STYLE
    d = CollectDialog(count=1)
    assert d.styleSheet() == STYLE
    assert d.btn_confirm.property("role") == "primary"
    assert d.btn_restart.property("role") == "ghost"
    assert d.btn_lang.property("role") == "ghost"
    assert d.counter.property("role") == "title"
    assert d.done.property("role") == "status-success"
