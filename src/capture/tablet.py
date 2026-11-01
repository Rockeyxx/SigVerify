<<<<<<< HEAD
"""
src/capture/tablet.py
======================
Dynamic signature capture layer for the XPPen Deco 640 on Windows,
via PyQt5 Tablet Events.
 
Capture Output Contract (per AGENTS.md):
    Final shape: [L, 4] float32  ->  X, Y, Pressure, dt
    - X, Y      : normalized later via per-signature z-score
    - Pressure  : already scaled to [0.0, 1.0] by Qt
    - dt        : gap since previous sample, clipped at 100ms then scaled to [0,1]
 
Critical note: this layer must call the SAME normalization function used
during training (src.data.parsers.normalize_trajectory), not a separate
copy of it. Any divergent normalization path silently invalidates the
embeddings (explicit rule in AGENTS.md). The function below is a local
placeholder for testing only; when integrating with the real pipeline,
replace _normalize_trajectory_placeholder() with a real call into
src/data/parsers.py.
"""
 
import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QTabletEvent
from PyQt5.QtCore import Qt
 
# ── Contract constants (from AGENTS.md — Capture Contract) ─────────────
MIN_TRAJECTORY_POINTS = 10          # trajectory_gate(): minimum accepted length
MAX_NORMALIZED_XY_MEAN = 0.5        # X/Y centring after normalization
NORMALIZED_XY_STD_RANGE = (0.2, 5.0)
DT_CLIP_MS = 100.0                  # dt clipped at 100ms before scaling to [0,1]
 
 
class TrajectoryGateError(ValueError):
    """Raised when a captured signature fails a quality gate (trajectory_gate)."""
    pass
 
 
def _normalize_trajectory_placeholder(raw: np.ndarray) -> np.ndarray:
    """
     This is a local placeholder for testing ONLY.
 
    When integrating with the real project, this function must be
    replaced with a direct call:
        from src.data.parsers import normalize_trajectory
        return normalize_trajectory(raw)
 
    AGENTS.md states explicitly: any normalization path separate from
    training silently invalidates the embeddings. This local copy exists
    only so this file can be run and inspected standalone, without
    depending on the rest of the repository.
 
    raw: [L, 4] float32 — X, Y in device units, Pressure [0,1], dt in ms
    returns: [L, 4] float32 normalized — X, Y zero-mean/unit-variance per
             axis, Pressure unchanged, dt clipped and scaled to [0,1]
    """
    normalized = raw.copy()
 
    # X, Y: per-axis, per-signature z-score
    for axis in (0, 1):
        col = normalized[:, axis]
        mean, std = col.mean(), col.std()
        if std < 1e-8:
            std = 1e-8  # avoid divide-by-zero if all points are identical
        normalized[:, axis] = (col - mean) / std
 
    # Pressure: left as-is, already [0,1] from Qt
    # dt: clip at 100ms, then scale to [0,1]
    normalized[:, 3] = np.clip(normalized[:, 3], 0.0, DT_CLIP_MS) / DT_CLIP_MS
 
    return normalized
 
 
def pre_normalization_gate(raw: np.ndarray) -> None:
    """
    Preliminary gate — applied to raw data (before normalization).
    Only checks what doesn't depend on coordinate scale: length and
    whether values are finite.
 
     Does NOT check centring here: raw pixel-unit coordinates
    (X~1353, say) are naturally far from zero mean — that's expected
    before normalization, not a defect in the signature.
    """
    if raw.shape[0] < MIN_TRAJECTORY_POINTS:
        raise TrajectoryGateError(
            f"Length gate failed: {raw.shape[0]} points < "
            f"minimum {MIN_TRAJECTORY_POINTS}"
        )
 
    if not np.all(np.isfinite(raw)):
        raise TrajectoryGateError("Finite gate failed: NaN or infinity present")
 
 
def post_normalization_gate(normalized: np.ndarray) -> None:
    """
    Full quality gate — applied only after normalization (X/Y z-score).
    Centring and scale checks are only meaningful once values are on a
    unified scale.
    """
    for axis_name, axis in (("X", 0), ("Y", 1)):
        mean = normalized[:, axis].mean()
        std = normalized[:, axis].std()
        if abs(mean) > MAX_NORMALIZED_XY_MEAN:
            raise TrajectoryGateError(
                f"{axis_name} centring gate failed: mean={mean:.3f} "
                f"exceeds ±{MAX_NORMALIZED_XY_MEAN}"
            )
        low, high = NORMALIZED_XY_STD_RANGE
        if not (low <= std <= high):
            raise TrajectoryGateError(
                f"{axis_name} scale gate failed: std={std:.3f} "
                f"outside [{low}, {high}]"
            )
 
    pressure = normalized[:, 2]
    if pressure.min() < 0.0 or pressure.max() > 1.0:
        raise TrajectoryGateError(
            f"Pressure range gate failed: [{pressure.min():.3f}, "
            f"{pressure.max():.3f}] outside [0, 1]"
        )
 
 
class SignatureCaptureWidget(QWidget):
    """
    Full-signature capture widget: recording starts automatically when
    the pen touches down, and stops automatically on pen-up. It then
    builds an [L, 4] array and passes it through gate + normalize.
    """
 
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signature Capture — Sign with the pen")
        self.resize(800, 500)
 
        self.label = QLabel(
            "Move the pen inside this window and start signing.\n"
            "Recording starts automatically when the pen touches down,\n"
            "and stops when it is lifted."
        )
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
 
        self._reset_state()
 
    def _reset_state(self):
        self._points = []            # each point: (x, y, pressure, timestamp)
        self._last_timestamp = None
        self._is_down = False
 
    def tabletEvent(self, event: QTabletEvent):
        pressure = event.pressure()
        now = time.time()
 
        # Start of contact: first point with actual pressure (> 0) after being idle
        if pressure > 0.0 and not self._is_down:
            self._is_down = True
            self._points = []
            self._last_timestamp = now
            self.label.setText("🖊️ Recording...")
 
        if self._is_down:
            dt_ms = 0.0 if self._last_timestamp is None else (now - self._last_timestamp) * 1000.0
            self._last_timestamp = now
            self._points.append((event.globalX(), event.globalY(), pressure, dt_ms))
 
        # End of contact: pen lifted (pressure back to zero) while recording
        if pressure == 0.0 and self._is_down:
            self._is_down = False
            self._finish_stroke()
 
        event.accept()
 
    def _finish_stroke(self):
        raw = np.array(self._points, dtype=np.float32)  # [L, 4]
        print(f"\n— End of signature: {raw.shape[0]} points captured —")
 
        try:
            pre_normalization_gate(raw)  # preliminary check: length + finite only
        except TrajectoryGateError as e:
            print(f" Rejected before normalization: {e}")
            self.label.setText(f" Signature rejected:\n{e}\n\nTry a longer/clearer signature.")
            return
 
        normalized = _normalize_trajectory_placeholder(raw)
 
        try:
            post_normalization_gate(normalized)  #  full check: centring + scale + pressure
        except TrajectoryGateError as e:
            print(f" Rejected after normalization: {e}")
            self.label.setText(f" Signature rejected:\n{e}")
            return
 
        print(" Signature passed both quality gates (pre- and post-normalization)")
        print(f"Final array shape: {normalized.shape}  (expected: [L, 4])")
        print("First 5 rows after normalization (X, Y, Pressure, dt):")
        print(normalized[:5])
 
        self.label.setText(
            f" Signature accepted — {raw.shape[0]} points\n"
            f"Final shape: {normalized.shape}\n\n"
            "Sign again to try another signature."
        )
 
 
def main():
    app = QApplication(sys.argv)
    window = SignatureCaptureWidget()
    window.show()
    sys.exit(app.exec_())
 
 
if __name__ == "__main__":
    main()
 
=======
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
    
>>>>>>> 2df8a87 (Update tablet capture implementation)
