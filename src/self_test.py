"""Whole-system self-test (issue #43): the five tests, run inside the app on a test kit.

    SigVerify --self-test KIT_FOLDER          the installed app (run_test.bat / run_test.sh)
    python -m src.self_test KIT_FOLDER        from source

The kit (tools/make_test_kit.py) holds one signer: paper scans (PNG, blue ink), tablet signings
as the signing window's own event log (CSV: t_ms, kind, x, y, pressure, xtilt, ytilt, inside)
and expected.json, the scores the model gave those files, scored directly, when the kit was
made. Nothing here is a fake model: the app's own window, pages, workers, signing window and
weights, and an encrypted vault in a temporary folder. Only the hardware is replaced: the file
dialog returns the kit's paths, and a virtual tablet plays each recorded signing into the
signing window at its own speed, so the capture code rebuilds the trajectory as it did when the
person signed. On a screen the window is shown and paced so the tester can follow it.
Writes RESULTS_NAME into the kit folder, with no user name, machine name or path.
"""
import base64
import contextlib
import csv
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback

import torch  # noqa: F401 — before PyQt5: on Windows the other order fails to load c10.dll

RESULTS_NAME = "SigVerify-test-results.txt"
PASSWORD = "correct horse battery staple"
VIRTUAL_TABLET = "Virtual tablet (recorded signings)"
DEVICE = {"os": platform.system(), "device_model": VIRTUAL_TABLET, "qt_platform": "self-test"}
MODALITY = {"fused": 0, "paper": 1, "pen": 2}


# ------------------------------------------------------------------ kit and report
def load_kit(kit):
    """(references, probes): references [(png, pen csv, trajectory)], probes [dict with png,
    pen_csv, trajectory, genuine, and the expected paper / pen / fused similarity]."""
    from src.data.campaign import load_inhouse_tablet_file
    with open(os.path.join(kit, "expected.json"), encoding="utf-8") as f:  # skylos: ignore[SKY-D215,SKY-D325] the kit folder the tester chose
        spec = json.load(f)

    def signing(png, pen):
        png, pen = os.path.join(kit, png), os.path.join(kit, pen)
        return png, pen, load_inhouse_tablet_file(pen)
    references = [signing(png, pen) for png, pen in spec["references"]]
    probes = [dict(p, **dict(zip(("png", "pen_csv", "trajectory"), signing(p["png_file"], p["pen_file"]))))
              for p in spec["probes"]]
    return references, probes


class Report:
    def __init__(self):
        self.checks, self.measured = [], {}

    def check(self, test, name, ok, detail=""):
        self.checks.append({"test": test, "name": name, "ok": bool(ok), "detail": detail})

    @property
    def passed(self):
        return bool(self.checks) and all(c["ok"] for c in self.checks)


def _peak_ram_gb():
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        from ctypes import wintypes
        # Types declared: by default ctypes passes the process handle as a 32-bit int, the call
        # failed and the counters stayed 0 (the first Windows runs reported 0.00 GB).
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        c = Counters(cb=ctypes.sizeof(Counters))
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return 0.0
        return c.PeakWorkingSetSize / 1024 ** 3
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2     # KB on Linux


def _machine():
    cpu = platform.processor()
    if not cpu and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "")
    return f"{platform.system()} {platform.release()} ({platform.machine()}), {cpu or 'CPU unknown'}, {os.cpu_count()} threads"


# ------------------------------------------------------------------ the virtual tablet
def _visible():
    from PyQt5.QtGui import QGuiApplication
    return QGuiApplication.platformName() != "offscreen"


def _pause(seconds):
    """Let the tester see a step (on a screen only; offscreen runs do not wait)."""
    from PyQt5.QtWidgets import QApplication
    end = time.perf_counter() + (seconds if _visible() else 0)
    QApplication.processEvents()
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _replay(canvas, pen_csv):
    """Play one recorded signing into the signing box. Each event carries its recorded global
    position and time, which the capture keeps; its local position, used only to draw it and to
    say whether the pen is inside the box, is the recording scaled into this box."""
    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtGui import QTabletEvent
    from PyQt5.QtWidgets import QApplication
    with open(pen_csv, newline="") as f:  # skylos: ignore[SKY-D215,SKY-D325] a pen file of the kit the tester chose
        rows = [(float(r["t_ms"]), int(r["kind"]), float(r["x"]), float(r["y"]), float(r["pressure"]),
                 int(r["inside"])) for r in csv.DictReader(f)]
    ink = [(x, y) for _, _, x, y, _, inside in rows if inside]
    x0, x1 = min(x for x, _ in ink), max(x for x, _ in ink)
    y0, y1 = min(y for _, y in ink), max(y for _, y in ink)
    w, h = canvas.width(), canvas.height()
    scale = 0.85 * min(w / max(x1 - x0, 1), h / max(y1 - y0, 1))
    ox, oy = (w - scale * (x1 - x0)) / 2, (h - scale * (y1 - y0)) / 2
    started, first = time.perf_counter(), rows[0][0]
    for t, kind, x, y, pressure, inside in rows:
        local = QPointF(ox + scale * (x - x0), oy + scale * (y - y0)) if inside else QPointF(-50, -50)
        event = QTabletEvent(kind, local, QPointF(x, y), QTabletEvent.Stylus, QTabletEvent.Pen,
                             pressure, 0, 0, 0.0, 0.0, 0, Qt.NoModifier, 1, Qt.LeftButton, Qt.LeftButton)
        event.setTimestamp(int(t))
        canvas.tabletEvent(event)
        if _visible() and (t - first) / 1000 > time.perf_counter() - started:
            canvas.repaint()
            QApplication.processEvents()
            time.sleep(max(0.0, (t - first) / 1000 - (time.perf_counter() - started)))


@contextlib.contextmanager
def _sandbox():
    """A new install in temporary folders, the hardware replaced; everything put back after."""
    from PyQt5 import QtWidgets
    from src.capture import collect, tablet
    from src.database.db_manager import PASSPHRASE_ENV_VAR
    from src.gui import enrollment_page, interfaces, storage

    root = tempfile.mkdtemp(prefix="sigverify-selftest-")
    home, temp = os.path.join(root, "home"), os.path.join(root, "tmp")
    os.makedirs(temp)
    hw = {"file": None, "pens": None, "dialogs": [], "home": home, "temp": temp}
    saved_env = {k: os.environ.get(k) for k in (storage.HOME_ENV_VAR, PASSPHRASE_ENV_VAR)}
    box = QtWidgets.QMessageBox

    def sign_on_the_virtual_tablet(dialog):
        dialog.show()
        for pen in hw["pens"][:dialog.count]:
            _pause(0.3)
            _replay(dialog.capture.canvas, pen)
            _pause(0.3)
            dialog.capture.confirm_signature()
        _pause(0.5)
        dialog.confirm()
        return dialog.result()

    def show_the_crop(dialog):
        dialog.show()
        _pause(0.6)
        return QtWidgets.QDialog.Accepted

    patches = [
        (tempfile, "tempdir", temp),
        (QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (hw["file"], "")),
        (tablet, "tablet_names", lambda: [VIRTUAL_TABLET]),
        (collect.CollectDialog, "exec_", sign_on_the_virtual_tablet),
        (enrollment_page.CropConfirmDialog, "exec_", show_the_crop),
        (box, "critical", lambda *a: hw["dialogs"].append(a)),
        (box, "information", lambda *a: hw["dialogs"].append(a)),
        (box, "question", lambda *a: box.Yes),
        (interfaces, "_service", None),
    ]
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    try:
        for obj, name, value in patches:
            setattr(obj, name, value)
        os.environ[storage.HOME_ENV_VAR] = home
        os.environ.pop(PASSPHRASE_ENV_VAR, None)
        storage._reset()
        yield hw
    finally:
        storage.logout()
        storage._reset()
        for obj, name, value in originals:
            setattr(obj, name, value)
        for k, v in saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(root, ignore_errors=True)


def _settle(worker):
    """Wait for a worker thread, then deliver its queued signals on the GUI thread."""
    from PyQt5.QtWidgets import QApplication
    worker.wait()
    QApplication.processEvents()


def _open_app():
    from src.gui import app as app_module
    started = time.perf_counter()
    window = app_module.MainWindow()
    window.show()
    _settle(window._warm_up_worker)
    return window, time.perf_counter() - started


def _step(window, text):
    """Say in the title bar what the test does now, and give the tester a moment to see it."""
    window.setWindowTitle(f"SigVerify self-test: {text}")
    _pause(0.8)


def _restart(window):
    from src.gui import storage
    window.close()
    storage.logout()
    storage._reset()
    return _open_app()[0]


def _register(window):
    s = window.register_screen
    s.fullNameInput.setText("Test Owner")
    s.companyInput.setText("Test Co")
    s.usernameInput.setText("tester")
    s.passwordInput.setText(PASSWORD)
    s.confirmPasswordInput.setText(PASSWORD)
    s.createButton.click()


def _login(window):
    window.auth_stack.setCurrentWidget(window.login_screen)
    window.login_screen.usernameInput.setText("tester")
    window.login_screen.passwordInput.setText(PASSWORD)
    window.login_screen._handle_login()
    _settle(window.login_screen._worker)


def _enrol(window, hw, client, references):
    panel = window.enrollment_panel
    window.content_stack.show_page(panel)
    panel.clientIdInput.setText(client)
    for png, _, _ in references:
        hw["file"] = png
        panel.select_reference_image()
        panel.add_image_reference()
    hw["pens"] = [pen for _, pen, _ in references]
    panel.capture_pen_signatures()
    panel.run_enrollment()
    _settle(panel._enroll_worker)
    return panel.statusLabel.text()


def _verify(window, hw, client, probe, modality="fused"):
    """One visit on the Verification page: seconds from Verify to the result on screen, or None
    when the page refused before verifying."""
    panel = window.verification_panel
    window.content_stack.show_page(panel)
    panel.modalitySelector.setCurrentIndex(MODALITY[modality])
    panel.clientIdInput.setText(client)
    if modality != "pen":
        hw["file"] = probe["png"]
        panel.select_probe_image()
    if modality != "paper":
        hw["pens"] = [probe["pen_csv"]]
        panel.capture_pen_probe()
    panel._verify_worker = None
    started = time.perf_counter()
    panel.run_verification()
    if panel._verify_worker is None:
        return None
    _settle(panel._verify_worker)
    seconds = time.perf_counter() - started
    _pause(1.5)                                    # the result stays on screen for the tester
    return seconds


def _verify_rows():
    from src.gui import storage
    return storage._get_db().get_log_events(event="verify")


def _screen_problems(window, row, expected, threshold, routing):
    """What disagrees between the screen, the score bar, the audit row and the model's score."""
    panel, out = window.verification_panel, []
    decision, bar = panel.resultDisplay.text(), panel.scoreBar
    want = "Genuine" if expected >= threshold else "Forged"
    if decision != row["decision"]:
        out.append(f"screen {decision!r}, audit {row['decision']!r}")
    if bar._score != row["confidence"] or bar._genuine != (decision == "Genuine"):
        out.append(f"bar {bar._score}, audit {row['confidence']}")
    if bar._threshold is None or abs(bar._threshold - 100 * row["threshold"]) > 1e-6 or abs(row["threshold"] - threshold) > 1e-6:
        out.append(f"threshold line {bar._threshold}, audit {row['threshold']}, expected {threshold}")
    if row["routing"] != routing or row["degraded"] != (routing != "multimodal"):
        out.append(f"routing {row['routing']} degraded {row['degraded']}")
    if abs(bar._score - 100 * expected) > 0.051:
        out.append(f"score {bar._score} on screen, {100 * expected:.2f} from the model")
    if decision != want:
        out.append(f"decision {decision}, the model says {want}")
    return out


# ------------------------------------------------------------------ the five tests
def _application():
    """The running QApplication, or a new one started as the app starts it (scaling, icon, theme)."""
    from PyQt5.QtWidgets import QApplication
    from src.gui.app import create_application
    return QApplication.instance() or create_application([])


def run(kit, report=None):
    """Run the five tests on the kit. Returns the Report (also written by main())."""
    import src.verification as verification
    from config.settings import (DECISION_THRESHOLD, INFERENCE_RAM_CEILING_GB, LATENCY_SLA_SECONDS,
                                 SPATIAL_DECISION_THRESHOLD, TEMPORAL_DECISION_THRESHOLD)
    from src.database.db_manager import PASSPHRASE_ENV_VAR, _DAMAGED
    from src.gui import app as app_module, interfaces, storage
    from src.gui.workers import WEIGHTS_MESSAGE

    report = report or Report()
    references, probes = load_kit(kit)
    genuine = [p for p in probes if p["genuine"]]
    forged = [p for p in probes if not p["genuine"]]
    if len(genuine) < 6 or not forged:
        raise ValueError("The kit needs 6 genuine probes and a forgery.")
    qt_app = _application()                # kept: an unreferenced QApplication is freed at once
    seconds = []

    with _sandbox() as hw:
        # 1 End to end -------------------------------------------------------------
        window, opened = _open_app()
        report.measured["window opened and models warmed up (s)"] = round(opened, 2)
        _step(window, "creating a test vault")
        _register(window)
        report.check(1, "register a vault and log in as its owner", window.current_employee is not None)
        _step(window, "enrolling a signer: 5 paper scans and 5 tablet signings")
        started = time.perf_counter()
        status = _enrol(window, hw, "Client One", references)
        report.measured["enrolment page, first scan to saved, signing included (s)"] = round(time.perf_counter() - started, 2)
        report.check(1, "enrol 5 scans and 5 pen signings", "Saved enrollment for 'Client One' (5 image, 5 pen" in status, status)
        stored = storage.get_reference_embeddings("Client One")

        def fused_visit(test, name, probe):
            _step(window, name.split(":")[0])
            s = _verify(window, hw, "Client One", probe)
            if s is not None:
                seconds.append(s)
            problems = ["the page refused before verifying"] if s is None else \
                _screen_problems(window, _verify_rows()[-1], probe["fused"], DECISION_THRESHOLD, "multimodal")
            report.check(test, name, not problems, "; ".join(problems) or
                         f"{window.verification_panel.resultDisplay.text()} {window.verification_panel.scoreBar._score}")

        fused_visit(1, "genuine signing, paper + pen: screen, bar and audit agree with the model", genuine[0])
        fused_visit(1, "forgery, paper + pen: screen, bar and audit agree with the model", forged[0])

        window = _restart(window)
        _step(window, "the app was closed and started again: logging in")
        _login(window)
        report.check(1, "close the app, start it again, log in", window.current_employee is not None)
        report.check(1, "references read back unchanged after the restart",
                     storage.get_reference_embeddings("Client One") == stored)
        fused_visit(1, "genuine signing after the restart", genuine[1])
        rows = len(_verify_rows())
        _step(window, "giving the first signing again (a replay)")
        _verify(window, hw, "Client One", genuine[0])
        text = window.verification_panel.resultDisplay.text()
        report.check(1, "the first signing given again is refused as a replay, not logged",
                     text == "Refused: this signing was already used (replay)" and len(_verify_rows()) == rows, text)

        # 3 One kind of signature only (tablet unplugged, or no scan) -------------
        for name, probe, modality, threshold, routing, branch in (
                ("paper only", genuine[2], "paper", SPATIAL_DECISION_THRESHOLD, "spatial_only", "paper"),
                ("pen only", genuine[3], "pen", TEMPORAL_DECISION_THRESHOLD, "temporal_only", "pen")):
            _step(window, f"{name} (as when the tablet is unplugged or there is no scan)")
            s = _verify(window, hw, "Client One", probe, modality)
            if s is not None:
                seconds.append(s)
            problems = ["the page refused before verifying"] if s is None else \
                _screen_problems(window, _verify_rows()[-1], probe[branch], threshold, routing)
            if not window.verification_panel.degradedModeIndicator.text().startswith("⚠ Degraded"):
                problems.append("no degraded message on screen")
            report.check(3, f"{name}: decided on that branch, shown as degraded", not problems, "; ".join(problems))

        # 5 "Sign again" is not logged (step-up forced on for one visit) -----------
        forced = {"VERIFICATION_STEP_UP": True, "STEP_UP_REJECT": 0.0, "STEP_UP_ACCEPT": 1.01}
        before = {k: getattr(verification, k) for k in forced}
        rows = len(_verify_rows())
        _step(window, "a \"Sign again\" round: two signings, one verdict")
        try:
            for k, v in forced.items():
                setattr(verification, k, v)
            _verify(window, hw, "Client One", genuine[4])
            asked = window.verification_panel.resultDisplay.text()
            first_logged = len(_verify_rows()) - rows
            _verify(window, hw, "Client One", genuine[5])
        finally:
            for k, v in before.items():
                setattr(verification, k, v)
        report.check(5, "\"Sign again\" asks once more and is not logged; the verdict after it is",
                     asked == "Please sign once more and give the new signing." and first_logged == 0
                     and len(_verify_rows()) == rows + 1, f"{asked!r}, rows after round 1: +{first_logged}")

        # 2 Latency and RAM ------------------------------------------------------
        report.measured["Verify to result on screen, worst (s)"] = round(max(seconds), 3)
        report.measured["Verify to result on screen, all (s)"] = [round(s, 3) for s in seconds]
        report.check(2, f"every visit under {LATENCY_SLA_SECONDS:g} s from Verify to the result",
                     max(seconds) < LATENCY_SLA_SECONDS, f"worst {max(seconds):.3f} s")

        # 4 The app decides like the model, through the vault ----------------------
        _step(window, f"checking all {len(probes)} signings against the model")
        enrolled = interfaces.enroll_client([r[0] for r in references], [r[2] for r in references])
        ok, message = storage.save_client_enrollment(
            "Client Two", enrolled["dynamic_embeddings"], "tester", spatial_embeddings=enrolled["embeddings"],
            spatial_origin=enrolled["spatial_origin"], capture_device=DEVICE)
        ref_spatial, origin = storage.get_reference_embeddings("Client Two")
        ref_temporal = storage.get_client("Client Two")["dynamic_references"]
        worst, wrong = 0.0, []
        for p in probes:
            result = interfaces.verify_client(ref_spatial, origin, p["png"], reference_temporal_b64=ref_temporal,
                                              probe_trajectory=p["trajectory"])
            worst = max(worst, abs(result["confidence"] / 100 - p["fused"]))
            if result["decision"] != ("Genuine" if p["fused"] >= DECISION_THRESHOLD else "Forged"):
                wrong.append(p["name"])
        report.measured["largest score difference, app vs model"] = float(f"{worst:.2e}")
        report.check(4, f"all {len(probes)} signings through the vault: same decision and score as the model",
                     ok and not wrong and worst < 1e-4,
                     message if not ok else (f"decisions differ: {wrong}; " if wrong else "") + f"largest score difference {worst:.2e}")

        # 5 Data rules (ISO/IEC 24745) --------------------------------------------
        _step(window, "checking the data rules")
        images = [n for n in os.listdir(hw["temp"]) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        report.check(5, "no signature image left in the temp folder", not images, f"{len(images)} left")
        templates = []
        for client in ("Client One", "Client Two"):
            templates += [base64.b64decode(b) for b in storage.get_reference_embeddings(client)[0]]
            templates += [base64.b64decode(b) for b in storage.get_client(client)["dynamic_references"]]
        leaks, locked = [], []
        for folder, _, files in os.walk(hw["home"]):
            for name in files:
                try:
                    with open(os.path.join(folder, name), "rb") as f:  # skylos: ignore[SKY-D215,SKY-D325] the test's own temporary vault folder
                        data = f.read()
                except PermissionError:   # Windows enforces the app's lock on signatures.db.lock (empty)
                    locked.append(name)
                    continue
                leaks += [name for t in templates if t[:32] in data]
        report.check(5, f"none of the {len(templates)} stored templates in plain form in any vault file",
                     not leaks and all(n.endswith(".lock") for n in locked),
                     ", ".join(sorted(set(leaks)) + [f"{n} (locked by the app)" for n in locked]))
        verdicts = [r for r in _verify_rows() if r["client_display_name"] == "Client One"]
        report.check(5, "one audit row per verdict", len(verdicts) == 6, f"{len(verdicts)} rows for 6 verdicts")

        # 3 Faults ------------------------------------------------------------------
        real_service = interfaces._service
        missing = os.path.join(hw["home"], "missing.pth")
        interfaces._service = verification.SignatureService(spatial_weights=missing, temporal_weights=missing)
        rows = len(_verify_rows())
        _step(window, "fault: the model weights are missing")
        _verify(window, hw, "Client One", genuine[6] if len(genuine) > 6 else genuine[0])
        panel = window.verification_panel
        report.check(3, "missing weights: the message, Verify usable again, nothing logged",
                     panel.resultDisplay.text() == WEIGHTS_MESSAGE and panel.verifyButton.isEnabled()
                     and len(_verify_rows()) == rows, panel.resultDisplay.text())
        interfaces._service = real_service

        vault = storage._get_db().db_path
        copy = vault + ".selftest-copy"
        window.close()
        storage.logout()
        storage._reset()
        shutil.copyfile(vault, copy)
        with open(vault, "r+b") as f:
            f.seek(os.path.getsize(vault) // 2)
            f.write(b"\x00" * 64)
        window = _open_app()[0]
        _step(window, "fault: the vault file is damaged")
        _login(window)
        _pause(1.5)
        label = window.login_screen.errorLabel.text()
        report.check(3, "damaged vault file: named on the login screen, the app stays open",
                     window.current_employee is None and label == f"⚠ {_DAMAGED}", label)
        window.close()
        storage._reset()
        shutil.move(copy, vault)

        os.environ[PASSPHRASE_ENV_VAR] = "a passphrase this vault never had"
        try:
            window = _open_app()[0]
            _step(window, "fault: a wrong key")
            _login(window)
            shown = hw["dialogs"][-1] if hw["dialogs"] else None
            report.check(3, "wrong key: an error dialog, the app stays open",
                         window.current_employee is None and shown is not None and shown[1] == app_module.STORAGE_ERROR_TITLE,
                         shown[2] if shown else "no dialog")
            window.close()
        finally:
            os.environ.pop(PASSPHRASE_ENV_VAR, None)

        ram = _peak_ram_gb()
        report.measured["peak RAM of the app (GB)"] = round(ram, 2)
        report.check(2, f"peak RAM at most {INFERENCE_RAM_CEILING_GB:g} GB", 0 < ram <= INFERENCE_RAM_CEILING_GB,
                     f"{ram:.2f} GB" if ram > 0 else "could not be measured")
        qt_app.processEvents()             # closed windows go before their temporary folders do
    return report


# ------------------------------------------------------------------ results file
def write_results(kit, report, crash=None):
    from config.settings import MODEL_VERSION
    from src.app_version import app_version
    lines = ["SigVerify whole-system test (issue #43)", "",
             f"Result:  {'PASS' if report.passed and not crash else 'FAIL'}",
             f"Date:    {time.strftime('%Y-%m-%d %H:%M')}",
             f"App:     {app_version()}  (model {MODEL_VERSION}, {'installed' if getattr(sys, 'frozen', False) else 'source'})",
             f"Machine: {_machine()}", ""]
    for test, title in ((1, "End to end"), (2, "Latency and RAM"), (3, "Faults"),
                        (4, "App decides like the model"), (5, "Data rules")):
        lines.append(f"Test {test}. {title}")
        for c in (c for c in report.checks if c["test"] == test):
            lines.append(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}" + (f"  ({c['detail']})" if c["detail"] else ""))
        lines.append("")
    lines.append("Measured")
    lines += [f"  {k}: {v}" for k, v in report.measured.items()]
    if crash:
        lines += ["", "The test stopped with an error:", crash]
    with open(os.path.join(kit, RESULTS_NAME), "w", encoding="utf-8") as f:  # skylos: ignore[SKY-D215,SKY-D324] the kit folder the tester chose
        f.write("\n".join(lines) + "\n")


def main(argv):
    """argv: [KIT_FOLDER]. Returns 0 when every check passed."""
    kit = os.path.abspath(argv[0] if argv else os.getcwd())
    report = Report()
    try:
        run(kit, report)
        crash = None
    except Exception:
        crash = traceback.format_exc().replace(os.path.expanduser("~"), "~")   # no user name in the file
    write_results(kit, report, crash)
    return 0 if report.passed and not crash else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
