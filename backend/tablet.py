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
 