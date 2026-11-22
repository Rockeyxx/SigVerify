"""The tablet widget hands the model what pen_stream makes of its strokes, and gates at Confirm."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QTabletEvent  # noqa: E402

from config.settings import TEMPORAL_INPUT_FEATURES  # noqa: E402
from src.capture.tablet import SignatureCaptureWidget  # noqa: E402
from src.data.pen_stream import strokes_to_trajectory  # noqa: E402


def _event(kind, x, y, pressure, ts, xtilt=0, ytilt=0):
    ev = QTabletEvent(kind, QPointF(x, y), QPointF(x, y), QTabletEvent.Stylus, QTabletEvent.Pen,
                      pressure, xtilt, ytilt, 0.0, 0.0, 0, Qt.NoModifier, 1, Qt.LeftButton, Qt.LeftButton)
    ev.setTimestamp(int(ts))
    return ev


def _strokes():
    """Three strokes at the XPPen's 4 ms report interval, in screen pixels, as (t_ms, x, y, p)."""
    out = []
    for k, t0 in enumerate((1000, 1400, 1800)):
        out.append([(float(t0 + 4 * i), 100.0 + 40 * k + 0.6 * i, 200.0 + 30 * np.sin(i / 9), 0.45)
                    for i in range(60)])
    return out


def _sign(canvas, strokes):
    for stroke in strokes:
        for i, (ts, x, y, p) in enumerate(stroke):
            canvas.tabletEvent(_event(QEvent.TabletPress if i == 0 else QEvent.TabletMove, x, y, p, ts))
        ts, x, y, _ = stroke[-1]
        canvas.tabletEvent(_event(QEvent.TabletRelease, x, y, 0.0, ts + 4))


def test_widget_output_is_pen_stream_of_its_strokes(app):
    widget = SignatureCaptureWidget()
    _sign(widget.canvas, _strokes())
    expected = strokes_to_trajectory([np.asarray(s) for s in _strokes()], pressure_max=1.0)
    assert np.allclose(widget.capture_signature(), expected)


def test_a_zero_pressure_press_starts_no_stroke(app):
    """The side button sends a press with pressure 0 (ENG_REPORT G4)."""
    widget = SignatureCaptureWidget()
    widget.canvas.tabletEvent(_event(QEvent.TabletPress, 10, 10, 0.0, 1000))
    widget.canvas.tabletEvent(_event(QEvent.TabletRelease, 10, 10, 0.0, 1004))
    assert widget.canvas.completed_strokes_data == []


def test_confirm_rejects_a_signature_the_gate_refuses(app):
    widget = SignatureCaptureWidget()
    emitted = []
    widget.signature_confirmed.connect(emitted.append)
    _sign(widget.canvas, [_strokes()[0][:6]])          # 20 ms of ink: 3 grid points
    widget.confirm_signature()
    assert emitted == []
    assert widget.label.text().startswith("Rejected")


def test_confirm_emits_a_gated_signature(app):
    widget = SignatureCaptureWidget()
    emitted = []
    widget.signature_confirmed.connect(emitted.append)
    _sign(widget.canvas, _strokes())
    widget.confirm_signature()
    assert len(emitted) == 1
    assert emitted[0].shape[1] == TEMPORAL_INPUT_FEATURES


def test_the_signing_box_is_visible(app):
    """The box painted nothing, so people signed outside it without knowing (ENG_REPORT B10)."""
    widget = SignatureCaptureWidget()
    canvas = widget.canvas
    img = canvas.grab().toImage()
    assert img.pixelColor(canvas.width() // 2, canvas.height() // 2).name() == "#ffffff"
    assert img.pixelColor(0, canvas.height() // 2).lightness() < 128   # a dark border


def test_ink_outside_the_box_is_not_recorded(app):
    """Qt keeps sending a stroke to the canvas after the pen leaves it; a swing toward the
    Confirm button left a stray tail (ENG_REPORT B10). Leaving the box ends the stroke and
    coming back starts a new one."""
    widget = SignatureCaptureWidget()
    canvas = widget.canvas
    w = canvas.width()
    xs = [w - 100 + 10 * i for i in range(20)] + [w - 10 * i for i in range(1, 15)]
    for i, x in enumerate(xs):
        kind = QEvent.TabletPress if i == 0 else QEvent.TabletMove
        canvas.tabletEvent(_event(kind, x, 100, 0.5, 1000 + 4 * i))
    canvas.tabletEvent(_event(QEvent.TabletRelease, xs[-1], 100, 0.0, 1000 + 4 * len(xs)))
    strokes = canvas.completed_strokes_data
    assert len(strokes) == 2
    assert all(0 <= x < w for s in strokes for _, x, _, _ in s)


def test_capture_info_names_device_os_and_qt_platform(app):
    info = SignatureCaptureWidget().capture_info()
    assert set(info) == {"device_model", "os", "qt_platform"}


def _input_node(sys_root, udev_root, event, minor, name, udev_lines):
    """One /sys/class/input/eventN node and its udev record, as Linux lays them out."""
    node = sys_root / event
    (node / "device").mkdir(parents=True)
    (node / "dev").write_text(f"13:{minor}\n")
    (node / "device" / "name").write_text(name + "\n")
    udev_root.mkdir(exist_ok=True)
    (udev_root / f"c13:{minor}").write_text("".join(f"{line}\n" for line in udev_lines))


def test_tablet_names_reads_the_kernel_name_of_the_node_udev_calls_a_tablet(tmp_path):
    """The Deco 640 makes three nodes with one USB ID; only the pen is the tablet. The pad
    flag (ID_INPUT_TABLET_PAD) must not count as a tablet."""
    from src.capture.tablet import tablet_names

    sys_root, udev_root = tmp_path / "input", tmp_path / "udev"
    _input_node(sys_root, udev_root, "event22", 86, "XPPen Deco 640 Mouse", ["E:ID_INPUT_MOUSE=1"])
    _input_node(sys_root, udev_root, "event23", 87, "XPPen Deco 640 Keyboard", ["E:ID_INPUT_KEY=1"])
    _input_node(sys_root, udev_root, "event24", 88, "XPPen Deco 640 Pen",
                ["E:ID_INPUT=1", "E:ID_INPUT_TABLET=1"])
    _input_node(sys_root, udev_root, "event25", 89, "Some Tablet Pad", ["E:ID_INPUT_TABLET_PAD=1"])
    _input_node(sys_root, udev_root, "event3", 67, "AT Translated Set 2 keyboard",
                ["E:ID_INPUT_KEYBOARD=1"])

    assert tablet_names("Linux", sys_root=str(sys_root), udev_root=str(udev_root)) == [
        "XPPen Deco 640 Pen"]


def test_tablet_names_skips_a_node_whose_dev_number_is_not_major_colon_minor(tmp_path):
    """The dev file's text becomes part of a udev path, so only "major:minor" is accepted."""
    from src.capture.tablet import tablet_names

    sys_root, udev_root = tmp_path / "input", tmp_path / "udev"
    _input_node(sys_root, udev_root, "event24", 88, "XPPen Deco 640 Pen", ["E:ID_INPUT_TABLET=1"])
    (sys_root / "event24" / "dev").write_text("../flag\n")       # would read udev/c../flag
    (udev_root / "c..").mkdir()
    (udev_root / "c.." / "flag").write_text("E:ID_INPUT_TABLET=1\n")
    assert tablet_names("Linux", sys_root=str(sys_root), udev_root=str(udev_root)) == []


def test_tablet_names_lists_two_different_tablets_and_one_name_for_two_alike(tmp_path):
    from src.capture.tablet import tablet_names

    sys_root, udev_root = tmp_path / "input", tmp_path / "udev"
    _input_node(sys_root, udev_root, "event24", 88, "XPPen Deco 640 Pen", ["E:ID_INPUT_TABLET=1"])
    _input_node(sys_root, udev_root, "event30", 94, "XPPen Deco 640 Pen", ["E:ID_INPUT_TABLET=1"])
    assert tablet_names("Linux", sys_root=str(sys_root), udev_root=str(udev_root)) == [
        "XPPen Deco 640 Pen"]

    _input_node(sys_root, udev_root, "event31", 95, "Wacom Intuos S Pen", ["E:ID_INPUT_TABLET=1"])
    assert sorted(tablet_names("Linux", sys_root=str(sys_root), udev_root=str(udev_root))) == [
        "Wacom Intuos S Pen", "XPPen Deco 640 Pen"]


def test_tablet_names_is_empty_with_no_tablet_or_no_wintab(tmp_path):
    from src.capture.tablet import tablet_names

    sys_root, udev_root = tmp_path / "input", tmp_path / "udev"
    _input_node(sys_root, udev_root, "event3", 67, "AT Translated Set 2 keyboard",
                ["E:ID_INPUT_KEYBOARD=1"])
    assert tablet_names("Linux", sys_root=str(sys_root), udev_root=str(udev_root)) == []

    def no_driver():
        raise OSError("wintab32.dll not found")

    assert tablet_names("Windows", wintab=no_driver) == []


class _FakeWintab:
    """wintab32.dll's WTInfoW for a driver that reports these device names."""

    def __init__(self, names):
        self.names = names

    def WTInfoW(self, category, index, out):
        if (category, index) == (1, 4):                      # WTI_INTERFACE, IFC_NDEVICES
            out.contents.value = len(self.names)
            return 4
        name = self.names[category - 100]                   # WTI_DEVICES + n, DVC_NAME
        assert index == 1
        if out is None:
            return 2 * (len(name) + 1)
        out.value = name
        return 2 * (len(name) + 1)


def test_tablet_names_reads_each_wintab_device_name():
    from src.capture.tablet import tablet_names

    assert tablet_names("Windows", wintab=lambda: _FakeWintab(["XP-Pen Deco 640"])) == [
        "XP-Pen Deco 640"]
    assert tablet_names("Windows", wintab=lambda: _FakeWintab(["Tablet A", "Tablet B"])) == [
        "Tablet A", "Tablet B"]


def test_capture_info_stores_the_name_read_or_no_driver(app, monkeypatch):
    import src.capture.tablet as tablet
    from config.settings import CAPTURE_NO_DRIVER

    monkeypatch.setattr(tablet, "tablet_names", lambda: ["XPPen Deco 640 Pen"])
    assert SignatureCaptureWidget().capture_info()["device_model"] == "XPPen Deco 640 Pen"

    monkeypatch.setattr(tablet, "tablet_names", lambda: [])
    assert SignatureCaptureWidget().capture_info()["device_model"] == CAPTURE_NO_DRIVER


def test_canvas_keeps_every_event_with_tilt_for_the_logger(app):
    widget = SignatureCaptureWidget()
    widget.canvas.tabletEvent(_event(QEvent.TabletMove, 50, 50, 0.0, 990, xtilt=12, ytilt=-7))  # hover
    _sign(widget.canvas, _strokes())
    events = widget.canvas.raw_events
    assert events[0][4] == 0.0 and events[0][5:7] == (12.0, -7.0)
    assert len(events) == 1 + sum(len(s) + 1 for s in _strokes())   # every move plus each release
    assert all(len(e) == 8 for e in events)


def test_clearing_the_canvas_clears_the_raw_events(app):
    widget = SignatureCaptureWidget()
    _sign(widget.canvas, _strokes())
    widget.canvas.clear_canvas()
    assert widget.canvas.raw_events == []


def test_capture_info_records_the_pen_input_qt_was_started_with(app, monkeypatch):
    """platformName() says only "windows" for both Windows inputs; the manifest needs which."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows:nowmpointer")
    assert SignatureCaptureWidget().capture_info()["qt_platform"] == "windows:nowmpointer"
