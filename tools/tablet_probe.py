#!/usr/bin/env python3
"""XPPen Deco 640 guided channel dump: one task at a time, raw kernel (evdev) and Qt events side by side.

Run from the repo root:  .venv/bin/python <this file>
Follow the big instruction in the window. Each task has its own timer, the next task is
announced 4 s before it starts, and Space finishes a task early. Keep the pen inside the box.
Writes raw_evdev.csv and raw_qt.csv (both tagged with the task) and summary.json.
Re-analyse a saved run:  python <this file> --analyse <run folder>
Reads /dev/input only (needs the 'input' group). Never grabs the device.
"""
import argparse
import bisect
import csv
import json
import os
import statistics
import struct
import sys
import threading
import time
from pathlib import Path

LINUX = sys.platform.startswith("linux")
if LINUX:  # the kernel stream exists only on Linux; elsewhere the probe records Qt alone
    import fcntl
    import select

EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
ABS_NAMES = {0: "x", 1: "y", 24: "p", 26: "tx", 27: "ty"}
KEY_NAMES = {0x140: "pen", 0x141: "rubber", 0x14A: "touch", 0x14B: "button"}
EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux
EVIOCSCLOCKID = (1 << 30) | (4 << 16) | (ord("E") << 8) | 0xA0
CLOCK_MONOTONIC = 1
STATE_KEYS = ["x", "y", "p", "tx", "ty", "pen", "rubber", "touch", "button"]
EV_FIELDS = ["t_ms", "step"] + STATE_KEYS
QT_FIELDS = ["recv_ms", "step", "type", "qt_ts_ms", "gx", "gy", "lx", "ly", "pressure",
             "xtilt", "ytilt", "device", "pointer"]
QT_PRESS, QT_MOVE, QT_RELEASE = 92, 87, 93
P_MAX = 16383
READY_S = 4
STEPS = [  # (task, seconds, instruction)
    ("hover", 10, "Hold the pen about 1 cm ABOVE the box. Do not touch. Move it around slowly."),
    ("button", 10, "Keep hovering, do not touch. Press the side button on the pen 3 times."),
    ("sign_1", 12, "Sign your normal signature inside the box."),
    ("sign_2", 12, "Sign again inside the box (2 of 3)."),
    ("sign_3", 12, "Sign again inside the box (3 of 3)."),
    ("lines", 12, "Draw one long straight line left to right, then one top to bottom."),
    ("tilt_lr", 12, "Put the tip in the middle and keep it still. Lean the pen LEFT, then RIGHT."),
    ("tilt_fb", 12, "Tip still on the tablet. Lean the pen TOWARD you, then AWAY from you."),
    ("hard", 8, "Press the tip down HARD on one spot and hold it."),
    ("light", 10, "Touch as LIGHTLY as you can and draw a slow line."),
    ("still", 6, "Rest the tip on the tablet with normal pressure. Do not move."),
]
current = {"step": "start"}


def find_pen_node():
    for block in Path("/proc/bus/input/devices").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        if not any("Vendor=28bd" in l for l in lines):
            continue
        if not any(l.startswith("N: Name=") and l.rstrip().endswith('Pen"') for l in lines):
            continue
        for l in lines:
            if l.startswith("H: Handlers="):
                for h in l.split("=", 1)[1].split():
                    if h.startswith("event"):
                        return f"/dev/input/{h}"
    return None


class EvdevReader(threading.Thread):
    def __init__(self, path, stop):
        super().__init__(daemon=True)
        self.path, self.stop, self.frames, self.error = path, stop, [], None

    def run(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            self.error = f"cannot open {self.path}: {exc}"
            return
        try:
            fcntl.ioctl(fd, EVIOCSCLOCKID, struct.pack("i", CLOCK_MONOTONIC))
        except OSError:
            pass
        state = dict.fromkeys(STATE_KEYS, 0)
        buf = b""
        while not self.stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            try:
                buf += os.read(fd, EVENT.size * 128)
            except BlockingIOError:
                continue
            while len(buf) >= EVENT.size:
                sec, usec, etype, code, value = EVENT.unpack_from(buf)
                buf = buf[EVENT.size:]
                if etype == EV_ABS and code in ABS_NAMES:
                    state[ABS_NAMES[code]] = value
                elif etype == EV_KEY and code in KEY_NAMES:
                    state[KEY_NAMES[code]] = value
                elif etype == EV_SYN and code == 0:
                    self.frames.append([sec * 1000.0 + usec / 1000.0, current["step"]]
                                       + [state[k] for k in STATE_KEYS])
        os.close(fd)


def run_qt(rows, meta, scale):
    from PyQt5.QtCore import QT_VERSION_STR, QEvent, Qt, QTimer
    from PyQt5.QtGui import QFont, QPainter, QPen
    from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    app = QApplication.instance() or QApplication(sys.argv)
    phases = []
    for i, (key, secs, _) in enumerate(STEPS):
        phases += [("ready", READY_S * scale, i), (key, secs * scale, i)]
    plan = {"idx": 0, "end": 0.0, "skip": False}

    def log(e):
        rows.append([time.monotonic_ns() / 1e6, current["step"], int(e.type()), e.timestamp(),
                     e.hiResGlobalX(), e.hiResGlobalY(), e.posF().x(), e.posF().y(), e.pressure(),
                     e.xTilt(), e.yTilt(), int(e.device()), int(e.pointerType())])

    class Canvas(QWidget):
        def __init__(self):
            super().__init__()
            self.setAttribute(Qt.WA_TabletTracking, True)
            self.setMinimumSize(1100, 600)
            self.dots = []

        def tabletEvent(self, e):
            log(e)
            if e.pressure() > 0:
                self.dots.append(e.posF())
                self.update()
            e.accept()

        def paintEvent(self, _):
            qp = QPainter(self)
            qp.fillRect(self.rect(), Qt.white)
            qp.setPen(QPen(Qt.darkGray, 3))
            qp.drawRect(self.rect().adjusted(1, 1, -2, -2))
            qp.setPen(QPen(Qt.black, 3))
            for d in self.dots:
                qp.drawPoint(d)

    class Window(QWidget):
        def keyPressEvent(self, e):
            if e.key() == Qt.Key_Space:
                plan["skip"] = True

    class Proximity(QWidget):  # proximity events go to the application, not the widget
        def eventFilter(self, obj, e):
            if e.type() in (QEvent.TabletEnterProximity, QEvent.TabletLeaveProximity):
                log(e)
            return False

    window = Window()
    window.setWindowTitle("XPPen probe")
    layout = QVBoxLayout(window)
    title, status, canvas = QLabel(), QLabel(), Canvas()
    title.setWordWrap(True)
    for label, size in ((title, 22), (status, 13)):
        font = QFont()
        font.setPointSize(size)
        font.setBold(label is title)
        label.setFont(font)
    for w in (title, status, canvas):
        layout.addWidget(w)
    proximity = Proximity()
    app.installEventFilter(proximity)

    def start(idx):
        plan.update(idx=idx, skip=False)
        if idx >= len(phases):
            current["step"] = "done"
            window.close()
            app.quit()
            return
        key, secs, _ = phases[idx]
        plan["end"] = time.monotonic() + secs
        current["step"] = key
        if key != "ready":
            canvas.dots.clear()
            canvas.update()

    def tick():
        idx = plan["idx"]
        if idx >= len(phases):
            return
        left = plan["end"] - time.monotonic()
        if left <= 0 or plan["skip"]:
            start(idx + 1)
            return
        key, _, i = phases[idx]
        text = STEPS[i][2]
        if key == "ready":
            title.setText(f"NEXT, task {i + 1} of {len(STEPS)}:\n{text}")
            status.setText(f"Read it now. Starts in {left:.0f} s. Lift the pen away until then.")
        else:
            title.setText(f"NOW: {text}")
            status.setText(f"Task {i + 1} of {len(STEPS)}  |  {left:.0f} s left  |  Space = finish early")

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(100)
    start(0)
    tick()
    window.show()
    app.exec_()
    meta.update(qt_platform=app.platformName(), qt_version=QT_VERSION_STR,
                screens=[dict(name=s.name(), geometry=[s.geometry().x(), s.geometry().y(),
                                                       s.geometry().width(), s.geometry().height()],
                              dpr=s.devicePixelRatio()) for s in app.screens()])


def load(folder):
    def rows(name):
        out = []
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{path} is not a regular file the probe wrote")
        for r in csv.DictReader(open(path)):  # skylos: ignore[SKY-D215] the probe's own CSV in the folder the operator passed, checked above
            out.append({k: (v if k == "step" else float(v)) for k, v in r.items()})
            out[-1].setdefault("step", "all")
        return out
    return rows("raw_evdev.csv"), rows("raw_qt.csv")


def runs(mask):
    out, cur = [], []
    for i, m in enumerate(mask):
        if m:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def stats(v):
    if not v:
        return None
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1)))]
    return {"n": len(v), "median": round(statistics.median(v), 3), "p05": round(q(0.05), 3),
            "p95": round(q(0.95), 3), "max": round(v[-1], 3)}


def slope(xs, ys):
    try:
        return statistics.linear_regression(xs, ys)
    except (statistics.StatisticsError, ValueError):
        return None


def analyse_qt_only(qt, meta):
    """Qt stream alone (Windows, or no access to /dev/input): rate, clock, pressure steps, tilt."""
    out = {"meta": meta, "verdicts": ["Qt only: no kernel stream on this system"], "steps": {}}
    kinds = (QT_PRESS, QT_MOVE, QT_RELEASE)
    order = [s for s in dict.fromkeys(r["step"] for r in qt) if s not in ("start", "ready", "done")]
    for s in order:
        rs = [r for r in qt if r["step"] == s and r["type"] in kinds]
        down = [r for r in rs if r["pressure"] > 0]
        iv = [b["qt_ts_ms"] - a["qt_ts_ms"] for a, b in zip(down, down[1:])
              if 0 <= b["qt_ts_ms"] - a["qt_ts_ms"] < 100]
        out["steps"][s] = {"contact": len(down), "interval_ms": stats(iv),
                           "same_ms_pairs": sum(1 for d in iv if d == 0),
                           "pressure": [min(r["pressure"] for r in down),
                                        max(r["pressure"] for r in down)] if down else None,
                           "qt_press": sum(1 for r in rs if r["type"] == QT_PRESS),
                           "qt_release": sum(1 for r in rs if r["type"] == QT_RELEASE)}
    down = [r for r in qt if r["type"] in kinds and r["pressure"] > 0]
    iv = [b["qt_ts_ms"] - a["qt_ts_ms"] for a, b in zip(down, down[1:])
          if 0 <= b["qt_ts_ms"] - a["qt_ts_ms"] < 100]
    v = out["verdicts"]
    if iv:
        med = statistics.median(iv)
        rate = f"{1000 / med:.0f} Hz" if med > 0 else "undefined"
        v.append(f"Qt rate: {rate} (median {med:.2f} ms between contact events), "
                 f"{sum(1 for d in iv if d == 0)} pairs share one millisecond")
    levels = sorted({r["pressure"] for r in down})
    steps = [b - a for a, b in zip(levels, levels[1:]) if b > a]
    if steps:
        q = min(steps)
        v.append(f"Qt pressure step {q:.6f} (about 1/{1 / q:.0f} of full scale)")
    if down:
        v.append(f"Qt tilt x {min(r['xtilt'] for r in down):.0f}..{max(r['xtilt'] for r in down):.0f}, "
                 f"y {min(r['ytilt'] for r in down):.0f}..{max(r['ytilt'] for r in down):.0f}")
    return out


def analyse(ev, qt, meta):
    if not ev:
        return analyse_qt_only(qt, meta)
    out = {"meta": meta, "verdicts": [], "steps": {}}
    order = [s for s in dict.fromkeys(r["step"] for r in ev) if s not in ("start", "ready", "done")]
    times = [r["t_ms"] for r in ev]
    qc = [r for r in qt if r["type"] in (QT_PRESS, QT_MOVE, QT_RELEASE)]
    pairs = []
    for r in qc:
        j = bisect.bisect_left(times, r["qt_ts_ms"])
        near = [k for k in (j - 1, j) if 0 <= k < len(times)]
        if near:
            k = min(near, key=lambda k: abs(times[k] - r["qt_ts_ms"]))
            pairs.append((abs(times[k] - r["qt_ts_ms"]), k, r))
    same_clock = bool(pairs) and statistics.median(d for d, _, _ in pairs) <= 2.0
    matched = {k: r for d, k, r in pairs if d <= 2.0} if same_clock else {}
    out["qt_clock_is_kernel_monotonic_ms"] = same_clock

    for s in order:
        idx = [i for i, r in enumerate(ev) if r["step"] == s]
        rs = [ev[i] for i in idx]
        cr = runs([r["p"] > 0 for r in rs])
        iv = [rs[b]["t_ms"] - rs[a]["t_ms"] for run in cr for a, b in zip(run, run[1:])]
        cp = [r["p"] for r in rs if r["p"] > 0]
        prox = [r for r in rs if r["pen"] or r["rubber"]]
        btn = [r for r in rs if r["button"]]
        qs = [r for r in qt if r["step"] == s]
        st = {"frames": len(rs), "contact": len(cp), "strokes": len(cr), "interval_ms": stats(iv),
              "pressure": [min(cp), max(cp)] if cp else None,
              "tilt_x": [min(r["tx"] for r in prox), max(r["tx"] for r in prox)] if prox else None,
              "tilt_y": [min(r["ty"] for r in prox), max(r["ty"] for r in prox)] if prox else None,
              "button_frames": len(btn),
              "button_with_contact": sum(1 for r in btn if r["p"] > 0 or r["touch"]),
              "touch_without_pressure": sum(1 for r in rs if r["touch"] and r["p"] == 0),
              "pressure_without_touch": sum(1 for r in rs if r["p"] > 0 and not r["touch"]),
              "qt_press": sum(1 for r in qs if r["type"] == QT_PRESS),
              "qt_release": sum(1 for r in qs if r["type"] == QT_RELEASE),
              "qt_contact_events": sum(1 for r in qs if r["pressure"] > 0 and r["type"] != 171)}
        cidx = [i for i in idx if ev[i]["p"] > 0]
        if same_clock and cidx:
            st["qt_delivered"] = round(sum(1 for i in cidx if i in matched) / len(cidx), 3)
        if s == "still" and len(cp) > 2:
            st["jitter_mm_sd"] = [round(statistics.pstdev([r[a] for r in rs if r["p"] > 0]) / 100, 4)
                                  for a in ("x", "y")]
        out["steps"][s] = st

    v = out["verdicts"]
    work = [r for r in ev if r["step"] not in ("start", "ready", "done")]
    cr = runs([r["p"] > 0 for r in work])
    iv = [work[b]["t_ms"] - work[a]["t_ms"] for run in cr for a, b in zip(run, run[1:])]
    if iv:
        med = statistics.median(iv)
        v.append(f"Report rate: {1000 / med:.0f} Hz (median {med:.2f} ms between samples while "
                 f"touching), {sum(1 for d in iv if d < 0.5)} same-time bursts, "
                 f"{sum(1 for d in iv if d > 2 * med)} gaps over twice the median")
    prox = [r for r in work if r["pen"] or r["rubber"]]
    if prox:
        txs, tys = [r["tx"] for r in prox], [r["ty"] for r in prox]
        live = len(set(txs)) > 3 or len(set(tys)) > 3
        v.append(f"Tilt: {'LIVE' if live else 'NOT MOVING'}, x {min(txs):.0f}..{max(txs):.0f}, "
                 f"y {min(tys):.0f}..{max(tys):.0f} (device range -127..127)")
    cp = [r["p"] for r in work if r["p"] > 0]
    if cp:
        hard = out["steps"].get("hard", {}).get("pressure")
        light = out["steps"].get("light", {}).get("pressure")
        v.append(f"Pressure: max seen {max(cp):.0f} of {P_MAX}, lightest {min(cp):.0f}; "
                 f"hard task max {hard[1] if hard else 'n/a'}, light task min {light[0] if light else 'n/a'}")
    b = out["steps"].get("button")
    if b:
        v.append(f"Side button (hover task): pressed in {b['button_frames']} samples, with tip contact in "
                 f"{b['button_with_contact']} -> "
                 f"{'the button alone makes pen-down samples' if b['button_with_contact'] else 'the button does not write by itself'}")
    signs = [out["steps"][s] for s in order if s.startswith("sign") or s == "all"]
    if signs:
        v.append("Qt per signing task: " + "; ".join(
            f"{st['strokes']} strokes, {st['qt_press']} press / {st['qt_release']} release, "
            f"delivered {st.get('qt_delivered', 'n/a')}" for st in signs))
    if same_clock:
        mp = [(ev[k], r) for k, r in matched.items() if ev[k]["p"] > 0 and r["pressure"] > 0]
        fx = slope([e["x"] for e, _ in mp], [q["gx"] for _, q in mp])
        fy = slope([e["y"] for e, _ in mp], [q["gy"] for _, q in mp])
        fp = slope([e["p"] for e, _ in mp], [q["pressure"] for _, q in mp])
        ft = slope([e["ty"] for e, _ in mp], [q["ytilt"] for _, q in mp])
        if fx and fy:
            px, py = fx.slope * 100, fy.slope * 100
            out["mapping"] = {"px_per_mm": [round(px, 3), round(py, 3)], "aspect_y_over_x": round(py / px, 4),
                              "pairs": len(mp)}
            v.append(f"Screen mapping (Qt vs device): {px:.2f} px/mm across, {py:.2f} px/mm down, "
                     f"aspect {py / px:.3f} (1.000 = shape kept)")
        if fp:
            out["qt_pressure_x_pmax"] = round(fp.slope * P_MAX, 4)
            v.append(f"Qt pressure = device pressure / {P_MAX / fp.slope / P_MAX:.0f}"
                     f" (1.0 x {P_MAX} means scaled by the device max)" if fp.slope else "Qt pressure flat")
        if ft:
            out["qt_ytilt_per_raw"] = round(ft.slope, 4)
            v.append(f"Qt yTilt = {ft.slope:.3f} x raw tilt + {ft.intercept:.1f}")
    else:
        v.append("Qt timestamps are not on the kernel monotonic clock; Qt-vs-device matching skipped")
    st = out["steps"].get("still", {})
    if st.get("jitter_mm_sd"):
        v.append(f"Resting-tip jitter: sd {st['jitter_mm_sd'][0]} mm across, {st['jitter_mm_sd'][1]} mm down")
    return out


def report(summary, folder):
    print("\n".join(["", "VERDICTS"] + ["  - " + x for x in summary["verdicts"]] + ["", "PER TASK"]))
    for s, st in summary["steps"].items():
        if "strokes" not in st:   # Qt-only summary
            print(f"  {s:8s} {json.dumps(st)}")
            continue
        iv = st["interval_ms"] or {}
        print(f"  {s:8s} contact {st['contact']:5d}  strokes {st['strokes']:3d}  median {iv.get('median', '-')} ms"
              f"  pressure {st['pressure']}  tilt x {st['tilt_x']} y {st['tilt_y']}  button {st['button_frames']}"
              f"/{st['button_with_contact']}  qt press {st['qt_press']} rel {st['qt_release']}"
              f"  delivered {st.get('qt_delivered', '-')}")
    print(f"\nsaved to {folder}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every task's time (smoke tests)")
    ap.add_argument("--device", default=None, help="evdev node; default: auto-detect the XPPen pen")
    ap.add_argument("--out", default=None, help="run folder; default: ./probe-run-<time>")
    ap.add_argument("--analyse", default=None, help="re-analyse a saved run folder and exit")
    ap.add_argument("--pen-api", choices=["ink", "wintab"], default=None,
                    help="Windows pen input (default: wintab when the tablet driver is installed, else ink)")
    args = ap.parse_args()

    if args.analyse:
        folder = Path(args.analyse)
        old = json.loads((folder / "summary.json").read_text()).get("meta", {}) if (folder / "summary.json").exists() else {}
        summary = analyse(*load(folder), old)
        (folder / "summary_v2.json").write_text(json.dumps(summary, indent=2))
        report(summary, folder)
        return

    node = args.device or (find_pen_node() if LINUX else None)
    if LINUX and not node:
        sys.exit("XPPen pen node not found in /proc/bus/input/devices; is the tablet plugged in?")
    folder = Path(args.out or Path.cwd() / time.strftime("probe-run-%Y%m%d-%H%M%S"))
    folder.mkdir(parents=True, exist_ok=True)
    if not LINUX:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.capture.pen_input import choose_pen_input
        choose_pen_input(args.pen_api)
    meta = dict(device=node or "none (Qt stream only)", platform=sys.platform,
                pen_input=os.environ.get("QT_QPA_PLATFORM", ""),
                started=time.strftime("%Y-%m-%d %H:%M:%S"), scale=args.scale,
                tasks=[[k, s * args.scale] for k, s, _ in STEPS])
    stop = threading.Event()
    reader = EvdevReader(node, stop) if node else None
    if reader:
        reader.start()
    qt_rows = []
    try:
        run_qt(qt_rows, meta, args.scale)
    except Exception as exc:  # keep the kernel log even if the window fails
        meta["qt_error"] = repr(exc)
    stop.set()
    frames = []
    if reader:
        reader.join(timeout=2)
        frames = reader.frames
        if reader.error:
            meta["evdev_error"] = reader.error
    with open(folder / "raw_evdev.csv", "w", newline="") as f:
        csv.writer(f).writerows([EV_FIELDS] + frames)
    with open(folder / "raw_qt.csv", "w", newline="") as f:
        csv.writer(f).writerows([QT_FIELDS] + qt_rows)
    summary = analyse(*load(folder), meta)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    report(summary, folder)


if __name__ == "__main__":
    main()
