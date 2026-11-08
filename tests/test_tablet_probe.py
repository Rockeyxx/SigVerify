"""The probe's analysis: kernel-stream verdicts on Linux, Qt-only verdicts on Windows."""
from tools import tablet_probe as probe


def _kernel_rows(n=60, step="sign_1"):
    return [{"t_ms": 1000.0 + 4.0 * i, "step": step, "x": 8000.0 + 10 * i, "y": 4000.0,
             "p": 3000.0, "tx": float(i % 20 - 10), "ty": 5.0, "pen": 1.0, "rubber": 0.0,
             "touch": 1.0, "button": 0.0} for i in range(n)]


def _qt_rows(n=50, step="sign_1"):
    """Windows Ink pressure arrives in 1/1024 steps (Qt divides by 1024)."""
    return [{"recv_ms": 0.0, "step": step, "type": 87.0, "qt_ts_ms": 1000.0 + 4 * i,
             "gx": 0.0, "gy": 0.0, "lx": 0.0, "ly": 0.0, "pressure": (100 + i) / 1024,
             "xtilt": 0.0, "ytilt": 0.0, "device": 2.0, "pointer": 1.0} for i in range(n)]


def test_kernel_stream_gives_rate_and_live_tilt():
    text = " ".join(probe.analyse(_kernel_rows(), [], {})["verdicts"])
    assert "250 Hz" in text and "Tilt: LIVE" in text


def test_qt_only_stream_reports_rate_and_pressure_step():
    text = " ".join(probe.analyse([], _qt_rows(), {})["verdicts"])
    assert "Qt only" in text and "250 Hz" in text and "1/1024" in text
