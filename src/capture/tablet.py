import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QTabletEvent, QPainter, QPen
from PyQt5.QtCore import Qt, QPointF, pyqtSignal

from config.settings import TEMPORAL_SAMPLE_PERIOD_MS
from src.data.parsers import normalize_trajectory, MAX_GAP_MS


class TrajectoryGateError(ValueError):
    """Raised when there is nothing to process (no pen-down strokes recorded).
    Quality gates proper are run by verify() -> trajectory_gate(), not here."""
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
        self.layout.addWidget(self.canvas)

        btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.clicked.connect(self.confirm_signature)
        btn_layout.addWidget(self.btn_confirm)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_capture)
        btn_layout.addWidget(self.btn_reset)

        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)

    def confirm_signature(self):
        try:
            tensor = self.capture_signature()
        except ValueError as e:  # includes TrajectoryGateError and errors from normalize_trajectory
            print(f"[Rejected] {e}")
            self.label.setText(f"Rejected: {e}")
            return

        print(f"\n[Success] Trajectory Shape: {tensor.shape}")
        self.label.setText(f"Accepted! Points count: {tensor.shape[0]}")
        self.signature_confirmed.emit(tensor)
        self.canvas.clear_canvas()

    def reset_capture(self):
        self.canvas.clear_canvas()
        self.label.setText("Canvas cleared. Start signing again.")

    def capture_signature(self) -> np.ndarray:
        """
        Public API method to retrieve the finalized and normalized signature trajectory.
        Shape: [L, 4] -> (x, y, pressure, dt)
        """
        return self.canvas.get_final_trajectory_tensor()


class DrawingCanvas(QWidget):
    """
    Canvas widget capturing hardware-level tablet events (hi-res coords, device timestamps, pressure).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(760, 350)
        self.setStyleSheet("background-color: white; border: 2px solid #555;")
        self.setAttribute(Qt.WA_TabletTracking, True)

        # Display strokes for QPainter rendering
        self.display_strokes = []
        self.current_display_stroke = []

        # Raw hardware data per stroke: [(x, y, pressure, timestamp_ms), ...]
        self.completed_strokes_data = []
        self.current_stroke_data = []

    def tabletEvent(self, event: QTabletEvent):
        p = float(event.pressure())
        ts_ms = float(event.timestamp())  # Qt5: device event time, already in milliseconds
        x, y = float(event.hiResGlobalX()), float(event.hiResGlobalY())
        et = event.type()

        if et == QTabletEvent.TabletPress:
            if p > 0.0:
                self.current_display_stroke = [QPointF(event.posF())]
                self.current_stroke_data = [(x, y, p, ts_ms)]
            event.accept()

        elif et == QTabletEvent.TabletMove:
            if p > 0.0:
                self.current_display_stroke.append(QPointF(event.posF()))
                self.current_stroke_data.append((x, y, p, ts_ms))
                self.update()
            event.accept()

        elif et == QTabletEvent.TabletRelease:
            # End of stroke: save raw pen-down points; the P=0 release sample is dropped
            if self.current_stroke_data:
                self.completed_strokes_data.append(list(self.current_stroke_data))
                self.display_strokes.append(list(self.current_display_stroke))
                self.current_stroke_data = []
                self.current_display_stroke = []
                self.update()
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
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
        self.update()

    def _decimate_stroke(self, stroke: list, target_period_ms: float) -> list:
        """
        Decimates high-frequency tablet events down to target sampling period (e.g. 10ms / 100Hz).
        Timestamps are in milliseconds.
        """
        if len(stroke) <= 1:
            return stroke

        decimated = [stroke[0]]
        last_ts = stroke[0][3]

        for pt in stroke[1:]:
            ts = pt[3]
            if (ts - last_ts) >= (target_period_ms - 1.0):  # margin for slight hardware jitter
                decimated.append(pt)
                last_ts = ts

        return decimated

    def get_final_trajectory_tensor(self) -> np.ndarray:
        all_strokes = list(self.completed_strokes_data)
        if self.current_stroke_data:
            all_strokes.append(self.current_stroke_data)

        if not all_strokes:
            raise TrajectoryGateError("No strokes recorded.")

        processed_points = []
        last_stroke_end_time = None

        for raw_stroke in all_strokes:
            if not raw_stroke:
                continue

            # Decimate raw stroke to 100Hz (TEMPORAL_SAMPLE_PERIOD_MS)
            stroke = self._decimate_stroke(raw_stroke, TEMPORAL_SAMPLE_PERIOD_MS)

            # float64: float32 cannot represent large ms timestamps exactly (dt would be corrupted)
            stroke_arr = np.array(stroke, dtype=np.float64)
            xy = stroke_arr[:, :2]
            p = stroke_arr[:, 2]
            ts = stroke_arr[:, 3]  # milliseconds

            dt = np.zeros(len(ts), dtype=np.float64)
            if len(ts) > 1:
                dt[1:] = np.diff(ts)

            # First sample after a pen lift gets the lift duration (ms),
            # measured from the last kept (decimated) sample, as the SigComp/DeepSignDB parsers do
            if last_stroke_end_time is not None:
                dt[0] = max(0.0, ts[0] - last_stroke_end_time)
            else:
                dt[0] = 0.0

            last_stroke_end_time = ts[-1]

            for i in range(len(stroke)):
                processed_points.append((xy[i, 0], xy[i, 1], p[i], dt[i]))

        if not processed_points:
            raise TrajectoryGateError("No pen-down samples recorded.")

        final_arr = np.array(processed_points, dtype=np.float64)
        xy_all = final_arr[:, :2].astype(np.float32)
        p_all = final_arr[:, 2].astype(np.float32)
        dt_all = final_arr[:, 3]

        # Clip deltas to [0, MAX_GAP_MS] and scale to [0, 1] (same contract as the training parsers)
        dt_scaled = (np.clip(dt_all, 0.0, MAX_GAP_MS) / MAX_GAP_MS).astype(np.float32)

        # Quality gates (point count, ranges, ...) are run by verify() -> trajectory_gate()
        return normalize_trajectory(xy_all, p_all, dt_scaled)


def main():
    app = QApplication(sys.argv)
    w = SignatureCaptureWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    