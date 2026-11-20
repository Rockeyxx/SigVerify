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


def _sign_and_confirm(self):
    _sign(self)
    self.confirm()
    return self.result()


def test_collect_signatures_returns_the_signatures_with_os_and_tablet_or_none(app, monkeypatch):
    import platform

    import src.capture.tablet as tablet

    monkeypatch.setattr(tablet, "tablet_names", lambda: ["XPPen Deco 640 Pen"])
    monkeypatch.setattr(collect.CollectDialog, "exec_", _sign_and_confirm)
    got = collect_signatures(count=1)
    assert set(got) == {"signatures", "os", "device_model", "qt_platform"}
    assert len(got["signatures"]) == 1                      # count=1 is the verification probe
    assert got["signatures"][0].shape[1] == TEMPORAL_INPUT_FEATURES
    assert got["os"] == platform.system() and got["device_model"] == "XPPen Deco 640 Pen"

    def close(self):
        self.reject()
        return self.result()

    monkeypatch.setattr(collect.CollectDialog, "exec_", close)
    assert collect_signatures(count=5) is None


def test_a_tablet_plugged_in_after_the_window_opens_is_still_named(app, monkeypatch):
    """ENG_REPORT G16: the name was read only when the window opened, so a tablet plugged in
    afterwards was saved as "no driver" although the pen drew normally."""
    import src.capture.tablet as tablet

    connected = []
    monkeypatch.setattr(tablet, "tablet_names", lambda: list(connected))

    def plug_in_then_sign(self):
        connected.append("XPPen Deco 640 Pen")
        return _sign_and_confirm(self)

    monkeypatch.setattr(collect.CollectDialog, "exec_", plug_in_then_sign)
    assert collect_signatures(count=1)["device_model"] == "XPPen Deco 640 Pen"


def test_two_tablets_ask_which_one_and_cancelling_closes(app, monkeypatch):
    """Qt5 cannot tell which tablet a stroke came from, so the person names it."""
    import src.capture.tablet as tablet

    monkeypatch.setattr(tablet, "tablet_names", lambda: ["XPPen Deco 640 Pen", "Wacom Intuos S Pen"])
    monkeypatch.setattr(collect.CollectDialog, "exec_", _sign_and_confirm)
    offered = []

    def pick_second(parent, title, label, items, current, editable):
        offered.append(list(items))
        return items[1], True

    monkeypatch.setattr(collect.QInputDialog, "getItem", pick_second)
    assert collect_signatures(count=1)["device_model"] == "Wacom Intuos S Pen"
    assert offered == [["XPPen Deco 640 Pen", "Wacom Intuos S Pen"]]

    monkeypatch.setattr(collect.QInputDialog, "getItem", lambda *a: ("", False))
    assert collect_signatures(count=1) is None


def test_one_tablet_asks_nothing(app, monkeypatch):
    import src.capture.tablet as tablet

    monkeypatch.setattr(tablet, "tablet_names", lambda: ["XPPen Deco 640 Pen"])
    monkeypatch.setattr(collect.CollectDialog, "exec_", _sign_and_confirm)

    def must_not_ask(*_a):
        raise AssertionError("asked which tablet with only one connected")

    monkeypatch.setattr(collect.QInputDialog, "getItem", must_not_ask)
    assert collect_signatures(count=1)["device_model"] == "XPPen Deco 640 Pen"


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
    assert d.styleSheet().startswith(STYLE)
    assert d.btn_confirm.property("role") == "primary"
    assert d.btn_lang.property("role") == "ghost"


def test_start_over_is_a_visible_button(app):
    """'ghost' has no background, so Start over read as plain text: it gets the lighter panel colour."""
    from src.gui.styles import ACCENT, PRIMARY_LIGHT
    d = CollectDialog(count=1)
    assert d.btn_restart.property("role") == "secondary"
    rule = (f'QPushButton[role="secondary"] {{ background-color: {PRIMARY_LIGHT}; '
            f'border: 1px solid {ACCENT}; }}')
    assert rule in d.styleSheet()
    assert d.done.property("role") == "status-success"


def test_the_instructions_are_large(app):
    """The signer reads them standing at the tablet: the theme's large roles, not 'muted' 12 px."""
    d = CollectDialog(count=5)
    assert d.counter.property("role") == "hero-title"          # 28 px in STYLE
    assert d.capture.label.property("role") == "title"         # 20 px in STYLE
    assert 'QLabel[role="status-success"] { font-size: 24px; }' in d.styleSheet()


def test_redo_keeps_the_instruction_in_the_chosen_language(app):
    """The widget's Redo writes its own English line; in Arabic the Arabic hint must come back."""
    d = CollectDialog(count=5, lang="ar")
    d.capture.btn_reset.click()
    assert d.capture.label.text() == collect.TEXT["ar"]["hint"]
