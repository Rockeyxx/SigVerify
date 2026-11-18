import sys
import ctypes
import glob
import os
import platform
import re

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QColor, QGuiApplication, QTabletEvent, QPainter, QPen
from PyQt5.QtCore import Qt, QPointF, pyqtSignal

from config.settings import CAPTURE_NO_DRIVER
from src.data.dataset import trajectory_gate
from src.data.pen_stream import strokes_to_trajectory

# Wintab 1.1 WTInfo categories and indexes.
WTI_INTERFACE, IFC_NDEVICES, WTI_DEVICES, DVC_NAME = 1, 4, 100, 1
# A sysfs dev file holds "major:minor"; that text becomes part of the udev record's path.
_DEV_NUMBER = re.compile(r"\d+:\d+")


def _linux_tablet_names(sys_root, udev_root):
    """Kernel names of the input nodes udev flags as tablets. One XPPen makes three nodes with
    one USB ID (Pen, Mouse, Keyboard); only the pen carries ID_INPUT_TABLET."""
    names = []
    for node in sorted(glob.glob(os.path.join(sys_root, "event*"))):
        try:
            with open(os.path.join(node, "dev")) as f:
                major_minor = f.read().strip()
            if not _DEV_NUMBER.fullmatch(major_minor):
                continue
            with open(os.path.join(udev_root, f"c{major_minor}")) as f:  # skylos: ignore[SKY-D215,SKY-D325] fixed udev root, major_minor checked as digits:digits above
                if "E:ID_INPUT_TABLET=1" not in f.read().splitlines():
                    continue
            with open(os.path.join(node, "device", "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name and name not in names:
            names.append(name)
    return names


def _wintab_names(load_wintab):
    """Each device name the tablet driver reports through Wintab; empty without the driver."""
    try:
        dll = load_wintab()
        count = ctypes.c_uint(0)
        if not dll.WTInfoW(WTI_INTERFACE, IFC_NDEVICES, ctypes.pointer(count)):
            return []
        names = []
        for n in range(count.value):
            size = dll.WTInfoW(WTI_DEVICES + n, DVC_NAME, None)   # bytes, with the final NUL
            if not size:
                continue
            buf = ctypes.create_unicode_buffer(size // 2 + 1)
            dll.WTInfoW(WTI_DEVICES + n, DVC_NAME, buf)
            if buf.value.strip() and buf.value.strip() not in names:
                names.append(buf.value.strip())
        return names
    except (AttributeError, OSError):
        return []


def tablet_names(system=None, sys_root="/sys/class/input", udev_root="/run/udev/data",
                 wintab=None):
    """Names of the connected tablets, read from the system: the kernel on Linux, Wintab on
    Windows. Empty when none can be read (no tablet, or Windows without the driver).
    Reading takes milliseconds (3.7 ms for 27 input nodes on Linux)."""
    system = system or platform.system()
    if system == "Linux":
        return _linux_tablet_names(sys_root, udev_root)
    if system == "Windows":
        return _wintab_names(wintab or (lambda: ctypes.WinDLL("wintab32.dll")))
    return []


class TrajectoryGateError(ValueError):
    """Raised when there is nothing to process (no pen-down strokes recorded).
    The quality gate itself runs at Confirm and again in verify()."""
    pass


class SignatureCaptureWidget(QWidget):
    """
    Signature capture container widget for XPPen Deco 640.
    Keeps recording across pen lifts until an explicit Confirm action is triggered.
    Emits `signature_confirmed` with a normalized [L, 4] float32 array on Confirm.
    """
    signature_confirmed = pyqtSignal(object)  # np.ndarray [L, 4] -> (x, y, pressure, dt)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signature Capture — XPPen Deco 640")
        self.resize(800, 500)

        self.layout = QVBoxLayout()
        self.label = QLabel("Sign freely, then click Confirm or Reset.")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.canvas = DrawingCanvas(self)
        self.layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.layout.addSpacing(40)  # keep the buttons away from the box, so a tap on Confirm is not ink

        btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.clicked.connect(self.confirm_signature)
        btn_layout.addWidget(self.btn_confirm)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_capture)
        btn_layout.addWidget(self.btn_reset)

        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)

        # Read once when the window opens, before anyone presses Enter. With two tablets the
        # signing window asks which one (CollectDialog.choose_tablet); until then, the first.
        self.tablets = tablet_names()
        self.device_model = self.tablets[0] if self.tablets else CAPTURE_NO_DRIVER

    def confirm_signature(self):
        try:
            trajectory = self.capture_signature()
        except ValueError as e:  # includes TrajectoryGateError and errors from pen_stream
            print(f"[Rejected] {e}")
            self.label.setText(f"Rejected: {e}")
            return

        passed, details = trajectory_gate(trajectory)
        if not passed:
            reason = details.get("error", "failed the input gate")
            print(f"[Rejected] {reason}")
            self.label.setText(f"Rejected: {reason}")
            return

        print(f"\n[Success] Trajectory Shape: {trajectory.shape}")
        self.label.setText(f"Accepted! Points count: {trajectory.shape[0]}")
        self.signature_confirmed.emit(trajectory)
        self.canvas.clear_canvas()

    def reset_capture(self):
        self.canvas.clear_canvas()
        self.label.setText("Canvas cleared. Start signing again.")

    def capture_signature(self) -> np.ndarray:
        """
        Public API method to retrieve the finalized and normalized signature trajectory.
        Shape: [L, 4] -> (x, y, pressure, dt)
        """
        return self.canvas.trajectory()

    def capture_info(self) -> dict:
        """What enrolment stores beside the template, so verification can insist on the same
        tablet model and operating system (F23; spec decisions D7 and D9). device_model is the
        tablet name read from the system, or CAPTURE_NO_DRIVER."""
        return {"device_model": self.device_model, "os": platform.system(),
                # platformName() is "windows" for both Windows pen inputs; the started spec
                # (e.g. "windows:nowmpointer" for Wintab) tells them apart.
                "qt_platform": os.environ.get("QT_QPA_PLATFORM") or QGuiApplication.platformName()}


class DrawingCanvas(QWidget):
    """
    Canvas widget capturing hardware-level tablet events (hi-res coords, device timestamps, pressure).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(760, 350)
        self.setAttribute(Qt.WA_TabletTracking, True)

        # Display strokes for QPainter rendering
        self.display_strokes = []
        self.current_display_stroke = []

        # Raw pen-down samples per stroke: [(timestamp_ms, x, y, pressure), ...]
        self.completed_strokes_data = []
        self.current_stroke_data = []
        # Every tablet event, for the research logger (tools/capture_session.py):
        # (t_ms, kind, x, y, pressure, xtilt, ytilt, inside). Hover moves have pressure 0.
        self.raw_events = []

    def tabletEvent(self, event: QTabletEvent):
        p = float(event.pressure())
        ts_ms = float(event.timestamp())  # Qt5: device event time, already in milliseconds
        x, y = float(event.hiResGlobalX()), float(event.hiResGlobalY())
        et = event.type()

        # Only samples inside the box count. Qt keeps sending a stroke to the canvas after the
        # pen leaves it, so leaving ends the stroke and coming back starts a new one (B10).
        inside = self.rect().contains(event.pos())
        self.raw_events.append((ts_ms, int(et), x, y, p, float(event.xTilt()),
                                float(event.yTilt()), int(inside)))
        if et == QTabletEvent.TabletPress:
            self._end_stroke()  # a press always starts a fresh stroke
        if et in (QTabletEvent.TabletPress, QTabletEvent.TabletMove):
            if p > 0.0 and inside:
                self.current_display_stroke.append(QPointF(event.posF()))
                self.current_stroke_data.append((ts_ms, x, y, p))
                self.update()
            elif not inside:
                self._end_stroke()
            event.accept()

        elif et == QTabletEvent.TabletRelease:
            # The P=0 release sample is dropped
            self._end_stroke()
            event.accept()

    def _end_stroke(self):
        """Save the current stroke's raw pen-down points, if any."""
        if self.current_stroke_data:
            self.completed_strokes_data.append(list(self.current_stroke_data))
            self.display_strokes.append(list(self.current_display_stroke))
            self.current_stroke_data = []
            self.current_display_stroke = []
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Paint the signing box itself: a plain QWidget subclass ignores a stylesheet background,
        # so without this the box is invisible (B10).
        painter.fillRect(self.rect(), Qt.white)
        painter.setPen(QPen(QColor("#555555"), 4))
        painter.drawRect(self.rect())
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        all_strokes = self.display_strokes + ([self.current_display_stroke] if self.current_display_stroke else [])
        for s in all_strokes:
            for i in range(len(s) - 1):
                painter.drawLine(s[i], s[i + 1])

    def clear_canvas(self):
        self.display_strokes.clear()
        self.current_display_stroke.clear()
        self.completed_strokes_data.clear()
        self.current_stroke_data.clear()
        self.raw_events.clear()
        self.update()

    def trajectory(self) -> np.ndarray:
        strokes = [s for s in self.completed_strokes_data + [self.current_stroke_data] if s]
        if not strokes:
            raise TrajectoryGateError("No strokes recorded.")
        # Qt pressure is already 0..1 (the platform divides by its own maximum). Screen y grows
        # down and the pen corpora grow up, so pen_stream flips it (ENG_REPORT G4).
        return strokes_to_trajectory([np.asarray(s, dtype=np.float64) for s in strokes],
                                     pressure_max=1.0)


def main():
    app = QApplication(sys.argv)
    w = SignatureCaptureWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    