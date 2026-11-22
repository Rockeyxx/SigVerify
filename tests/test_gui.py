"""GUI tests without the model: routing to verify(), temp-file cleanup, messages, pen wiring.

The service is replaced by a fake that records its calls, so these tests need no weights
(issue #23, items 1, 2, 14, 15, 16).
"""
import base64
import os
import sys

import numpy as np
import pytest
import torch  # noqa: F401 — before PyQt5: on Windows the other order fails to load c10.dll

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from config.settings import NUM_ENROLLMENT_REFERENCES  # noqa: E402
from src.database.db_manager import DatabaseManager, PASSPHRASE_ENV_VAR  # noqa: E402
from src.gui import enrollment_page, interfaces, storage, verification_page, workers  # noqa: E402
from tests.vault_file import on_disk  # noqa: E402


def _blob(dim, seed=0):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32).tobytes()


def _b64(dim, seed=0):
    return base64.b64encode(_blob(dim, seed)).decode("ascii")


def _trajectory(seed=0):
    return np.random.default_rng(seed).standard_normal((50, 4)).astype(np.float32)


RESULT = {"decision": "Genuine", "confidence": 90.0, "routing": "multimodal",
          "degraded": False, "reasons": [], "threshold": 0.76,
          "probe_embeddings": {"spatial": [_blob(512, 50)], "temporal": [_blob(128, 50)]}}
OTHER_PROBE = {"spatial": [_blob(512, 51)], "temporal": [_blob(128, 51)]}


class FakeService:
    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.fail_with is not None:
            raise self.fail_with

    def verify(self, **kwargs):
        self._record("verify", **kwargs)
        return dict(RESULT)

    def enroll_multimodal(self, images, trajectories):
        self._record("enroll_multimodal", images=images, trajectories=trajectories)
        return {"spatial_embeddings": [_blob(512, i) for i in range(len(images))],
                "temporal_embeddings": [_blob(128, i) for i in range(len(trajectories))],
                "spatial_origin": "scan"}

    def enroll_unimodal(self, samples, modality):
        self._record("enroll_unimodal", samples=samples, modality=modality)
        dim = 512 if modality == "spatial" else 128
        out = {"embeddings": [_blob(dim, i) for i in range(len(samples))], "modality": modality}
        if modality == "spatial":
            out["spatial_origin"] = "scan"
        return out


@pytest.fixture
def service(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(interfaces, "_get_service", lambda: fake)
    return fake


@pytest.fixture
def temp_images(tmp_path, monkeypatch):
    """_prepare_image without OpenCV: each image becomes a real temp file, as in the app."""
    made = []

    def prepare(path, temp_files):
        out = tmp_path / f"prepared_{len(made)}.png"
        out.write_bytes(b"x")
        made.append(str(out))
        temp_files.append(str(out))
        return str(out)

    monkeypatch.setattr(interfaces, "_prepare_image", prepare)
    return made


# ------------------------------------------------------------------ mode choice
@pytest.mark.parametrize("image, pen, s_refs, t_refs, expected", [
    (True, True, True, True, "multimodal"),
    (True, False, True, True, "spatial"),
    (False, True, True, True, "temporal"),
    (True, True, False, True, "temporal"),     # tablet-only client: pen alone
    (True, True, True, False, "spatial"),      # paper-only client: paper alone
    (True, False, False, True, None),
    (False, False, True, True, None),
])
def test_choose_mode(image, pen, s_refs, t_refs, expected):
    assert interfaces.choose_mode(image, pen, s_refs, t_refs) == expected


# ------------------------------------------------------------------ verify_client
def test_verify_paper_and_pen_is_fused(service, temp_images):
    traj = _trajectory()
    interfaces.verify_client([_b64(512)], "scan", "probe.png",
                             reference_temporal_b64=[_b64(128)], probe_trajectory=traj)
    (name, kw), = service.calls
    assert name == "verify" and kw["mode"] == "multimodal"
    assert kw["probe_trajectory"] is traj and kw["probe_image"] == temp_images[0]
    assert kw["reference_spatial_origin"] == "scan"
    assert len(kw["reference_spatial"]) == 1 and len(kw["reference_temporal"]) == 1


def test_verify_pen_only_never_passes_spatial_references(service, temp_images):
    interfaces.verify_client([], "scan", None, reference_temporal_b64=[_b64(128)],
                             probe_trajectory=_trajectory())
    (name, kw), = service.calls
    assert kw["mode"] == "unimodal"
    assert "reference_spatial" not in kw and "probe_image" not in kw
    assert temp_images == []


def test_verify_paper_only_is_unchanged(service, temp_images):
    interfaces.verify_client([_b64(512)], "scan", "probe.png")
    (name, kw), = service.calls
    assert kw["mode"] == "unimodal" and kw["probe_image"] == temp_images[0]
    assert "probe_trajectory" not in kw


def test_verify_step_up_passes_both_signings_in_order(service, temp_images):
    t1, t2 = _trajectory(1), _trajectory(2)
    interfaces.verify_client([_b64(512)], "scan", ["a.png", "b.png"],
                             reference_temporal_b64=[_b64(128)], probe_trajectory=[t1, t2])
    kw = service.calls[0][1]
    assert kw["probe_image"] == temp_images[:2]
    assert kw["probe_trajectory"][0] is t1 and kw["probe_trajectory"][1] is t2


def test_verify_without_matching_references_raises(service, temp_images):
    with pytest.raises(ValueError, match="no references"):
        interfaces.verify_client([], "scan", "probe.png", reference_temporal_b64=[_b64(128)])
    assert service.calls == []


def test_verify_deletes_temp_images_even_when_it_raises(monkeypatch, temp_images):
    monkeypatch.setattr(interfaces, "_get_service", lambda: FakeService(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        interfaces.verify_client([_b64(512)], "scan", "probe.png")
    assert temp_images and not any(os.path.exists(p) for p in temp_images)


def test_the_employees_own_image_is_never_deleted_even_if_the_crop_hands_it_back(service, tmp_path,
                                                                               monkeypatch):
    original = tmp_path / "scan.png"
    original.write_bytes(b"x")
    monkeypatch.setattr(interfaces, "crop_to_ink", lambda path: path)
    interfaces.verify_client([_b64(512)], "scan", str(original))
    interfaces.enroll_client([str(original)])
    assert original.exists()


# ------------------------------------------------------------------ enroll_client
def test_enroll_paper_and_pen_together(service, temp_images):
    k = NUM_ENROLLMENT_REFERENCES
    trajs = [_trajectory(i) for i in range(k)]
    out = interfaces.enroll_client([f"{i}.png" for i in range(k)], trajs)
    assert service.calls[0][0] == "enroll_multimodal"
    assert len(out["embeddings"]) == k and len(out["dynamic_embeddings"]) == k
    assert out["spatial_origin"] == "scan"
    assert not any(os.path.exists(p) for p in temp_images)


def test_enroll_tablet_only(service, temp_images):
    trajs = [_trajectory(i) for i in range(NUM_ENROLLMENT_REFERENCES)]
    out = interfaces.enroll_client([], trajs)
    (name, kw), = service.calls
    assert (name, kw["modality"]) == ("enroll_unimodal", "temporal")
    assert out["embeddings"] == [] and len(out["dynamic_embeddings"]) == len(trajs)


def test_enroll_paper_only_has_no_pen_embeddings(service, temp_images):
    out = interfaces.enroll_client(["a.png"])
    assert out["dynamic_embeddings"] == [] and len(out["embeddings"]) == 1


def test_enroll_nothing_raises(service):
    with pytest.raises(ValueError):
        interfaces.enroll_client([], [])


def test_pen_gate_rejection_is_named(monkeypatch, temp_images):
    error = ValueError("Enrolment sample 2 failed the temporal input gate: too short.")
    monkeypatch.setattr(interfaces, "_get_service", lambda: FakeService(error))
    with pytest.raises(interfaces.PenSignatureRejectedError):
        interfaces.enroll_client([], [_trajectory()])


# ------------------------------------------------------------------ confirmed signings
def test_confirmed_fused_visit_embeds_paper_and_pen(service, temp_images):
    out = interfaces.embed_confirmed_signing("probe.png", _trajectory())
    assert out["spatial"] and out["temporal"]
    assert len(base64.b64decode(out["temporal"])) == 128 * 4
    assert [kw["modality"] for _, kw in service.calls] == ["spatial", "temporal"]


def test_visit_signings_pairs_first_signing_first():
    t1, t2 = _trajectory(1), _trajectory(2)
    assert verification_page.visit_signings(["a", "b"], [t1, t2]) == [("a", t1), ("b", t2)]
    assert verification_page.visit_signings("a", None) == [("a", None)]
    assert verification_page.visit_signings(None, t1) == [(None, t1)]


# ------------------------------------------------------------------ messages and log
def test_modality_state_records_what_ran():
    text = interfaces.modality_state({"routing": "temporal_only", "degraded": True,
                                      "reasons": ["spatial: too blurry"], "threshold": 0.8})
    assert "Pen only" in text and "spatial: too blurry" in text and "0.8000" in text
    assert interfaces.modality_state(RESULT).startswith("Paper + pen (fused)")


def test_missing_image_is_not_blamed_on_the_weights():
    msg = workers._missing_file_message(FileNotFoundError(2, "No such file", "C:/x/sig.png"))
    assert "C:/x/sig.png" in msg and msg != workers.WEIGHTS_MESSAGE
    assert workers._missing_file_message(
        FileNotFoundError("weights not found: models/weights/x.pth")) == workers.WEIGHTS_MESSAGE


def test_pen_gate_message_is_not_the_photo_advice():
    pen = ValueError("No usable evidence — every supplied probe failed its input gate\n"
                     "  temporal: too short")
    assert workers._gate_message(pen) == workers.PEN_GATE_MESSAGE
    photo = ValueError("spatial: failed the quality gate")
    assert workers._gate_message(photo) == workers.QUALITY_GATE_MESSAGE


def test_enrolment_count_comes_from_settings():
    assert enrollment_page.REQUIRED_IMAGE_REFERENCES == NUM_ENROLLMENT_REFERENCES
    assert enrollment_page.REQUIRED_DYNAMIC_REFERENCES == NUM_ENROLLMENT_REFERENCES


# ------------------------------------------------------------------ panels
@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    """A new install: no account yet."""
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)
    monkeypatch.setenv(storage.HOME_ENV_VAR, str(tmp_path / "home"))
    monkeypatch.delenv(PASSPHRASE_ENV_VAR, raising=False)
    storage._reset()
    yield
    storage._reset()


@pytest.fixture
def db(empty_db):
    """Alice registered the company Test Co: its admin, logged in."""
    assert storage.create_company("Test Co", "Alice A", "alice", "alice-password-1")[0]


@pytest.fixture
def alice_bob(db):
    """Alice is the admin (logged in); she added the employee Bob."""
    assert storage.register_employee("Bob B", "bob", "bob-password-1-long")[0]


class FakeWorker:
    """Stands in for a QThread worker: records its arguments, starts nothing."""
    made = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.finished = _Signal()
        self.failed = _Signal()
        FakeWorker.made.append(self)

    def start(self):
        pass


class _Signal:
    def connect(self, _slot):
        pass


@pytest.fixture
def fake_workers(monkeypatch):
    FakeWorker.made = []
    monkeypatch.setattr(verification_page, "VerifyWorker", FakeWorker)
    monkeypatch.setattr(enrollment_page, "EnrollWorker", FakeWorker)
    return FakeWorker.made


def _panel(employee="alice"):
    panel = verification_page.VerificationPanel()
    panel.set_current_employee(employee)
    return panel


def test_panel_fuses_paper_and_pen(app, db, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel.clientIdInput.setText("c")
    panel.selected_image_path = "probe.png"
    panel.selected_trajectory = _trajectory()
    panel.run_verification()
    worker, = fake_workers
    refs_s, origin, image = worker.args
    assert image == "probe.png" and len(refs_s) == 1 and origin == "scan"
    assert len(worker.kwargs["reference_temporal_b64"]) == 1
    assert worker.kwargs["probe_trajectory"] is panel.selected_trajectory


def test_panel_forgets_the_probe_image_when_the_client_changes(app, db):
    panel = _panel()
    panel.clientIdInput.setText("a")
    panel.selected_image_path = "probe.png"
    panel.probeFileLabel.setText("probe.png")
    panel.clientIdInput.setText("b")
    assert panel.selected_image_path is None
    assert panel.probeFileLabel.text() == "No file selected"


@pytest.mark.parametrize("change", ["modality", "client"])
def test_panel_clears_the_last_result_when_the_modality_or_client_changes(app, db, fake_workers, change):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel.clientIdInput.setText("c")
    panel._pending_client_id = "c"
    panel._pending_images, panel._pending_trajectories = "probe.png", _trajectory()
    panel._on_verify_success(dict(RESULT))
    assert panel.resultDisplay.text() == RESULT["decision"]
    if change == "modality":
        panel.modalitySelector.setCurrentIndex(1)
    else:
        panel.clientIdInput.setText("d")
    assert panel.resultDisplay.text() == "Awaiting verification..."
    assert panel.scoreBar.isHidden() and panel.scoreCaption.isHidden()


def test_panel_pen_only_client_needs_pen_mode(app, db, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[])
    panel = _panel()
    panel.clientIdInput.setText("c")
    panel.selected_image_path = "probe.png"
    panel.selected_trajectory = _trajectory()
    panel.run_verification()                     # default: paper + pen
    assert fake_workers == [] and "no paper references" in panel.resultDisplay.text()
    panel.modalitySelector.setCurrentIndex(2)    # pen only
    panel.clientIdInput.setText("c")
    panel.selected_trajectory = _trajectory()
    panel.run_verification()
    worker, = fake_workers
    assert worker.args[0] == [] and worker.args[2] is None


def test_panel_logs_routing_and_offers_confirm_only_when_fused(app, db, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel._pending_client_id = "c"
    panel._pending_images, panel._pending_trajectories = "probe.png", _trajectory()
    panel._on_verify_success(dict(RESULT))
    assert not panel.confirmButton.isHidden()
    event = storage.get_last_verification_for_client("c", "alice")
    assert event["modality_state"].startswith("Paper + pen (fused)")

    panel._on_verify_success({**RESULT, "routing": "temporal_only", "degraded": True,
                              "probe_embeddings": OTHER_PROBE})
    assert panel.confirmButton.isHidden()


def test_panel_refuses_a_replayed_probe_and_logs_it(app, db, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel._pending_client_id = "c"
    panel._pending_images, panel._pending_trajectories = "probe.png", _trajectory()
    panel._on_verify_success(dict(RESULT))
    assert len(storage.get_verification_log("alice")) == 1

    panel._on_verify_success(dict(RESULT))       # the same probe bytes again
    assert panel.resultDisplay.text() == "Refused: this signing was already used (replay)"
    assert panel.confirmButton.isHidden() and panel.verifyButton.isEnabled()
    assert len(storage.get_verification_log("alice")) == 1
    events = storage._get_db().get_log_events(event=None)
    assert [e["event"] for e in events][-1] == "replay refused"

    # A "Sign again" round is checked too, and its round state is reset.
    panel._awaiting_second, panel._first_probe_path = True, "first.png"
    panel.selectProbeButton.setText("Select Second Signing")
    panel._on_verify_success({**RESULT, "decision": "Sign again"})
    assert panel.resultDisplay.text() == "Refused: this signing was already used (replay)"
    assert not panel._awaiting_second and panel._first_probe_path is None
    assert panel.selectProbeButton.text() == "Select Probe Image"
    assert len(storage.get_verification_log("alice")) == 1


def _audit_hashes():
    import json
    db = storage._get_db()
    with on_disk(db.db_path, db._dek) as conn:
        rows = conn.execute("SELECT event, probe_hashes FROM audit_log ORDER BY id").fetchall()
    return [(e, json.loads(h) if h else None) for e, h in rows]


def test_replay_audit_row_stores_the_hashes_and_a_first_round_sign_again_is_refused(app, db, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel._pending_client_id = "c"
    panel._pending_images, panel._pending_trajectories = "probe.png", _trajectory()
    panel._on_verify_success(dict(RESULT))
    panel.degradedModeIndicator.setText("shown")
    panel._last_accepted_signings, panel._last_accepted_client = ["x"], "c"
    panel._on_verify_success({**RESULT, "decision": "Sign again"})   # first round (_awaiting_second False)
    assert panel.resultDisplay.text() == "Refused: this signing was already used (replay)"
    assert panel.degradedModeIndicator.text() == ""
    assert panel._last_accepted_signings is None and panel._last_accepted_client is None
    events = _audit_hashes()
    assert events[-1][0] == "replay refused" and len(events[-1][1]) == 2
    assert events[-1][1] == [h for e, h in events if e == "verify"][0]


@pytest.mark.parametrize("replay", [False, True])
def test_the_visit_is_logged_before_it_is_shown_and_a_log_error_frees_the_button(app, db, fake_workers,
                                                                              monkeypatch, replay):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel._pending_client_id = "c"
    panel._pending_images, panel._pending_trajectories = "probe.png", _trajectory()
    if replay:
        panel._on_verify_success(dict(RESULT))
    panel.resultDisplay.setText("Processing…")

    def broken(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(storage, "log_replay_refused" if replay else "append_verification_log", broken)
    panel.verifyButton.setEnabled(False)
    panel.verifyButton.setText("Verifying…")
    with pytest.raises(RuntimeError):
        panel._on_verify_success(dict(RESULT))
    assert panel.verifyButton.isEnabled() and panel.verifyButton.text() == "Verify"
    assert panel.resultDisplay.text() == "Processing…"      # nothing shown that was not logged


def test_a_refused_confirm_says_nothing_was_added(app, db):
    panel = _panel()
    panel._last_accepted_client, panel._last_accepted_origin = "gone", "scan"
    panel._last_accepted_signings = ["x"]
    panel._on_confirmed([{"spatial": _b64(512), "temporal": _b64(128)}])
    assert panel.modalityStateLabel.text() == "Nothing was added to the signer's references."


def test_replay_check_fails_loudly_when_the_model_returns_no_probe_embeddings(app, db, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel._pending_client_id = "c"
    broken = {k: v for k, v in RESULT.items() if k != "probe_embeddings"}
    with pytest.raises(KeyError):
        panel._on_verify_success(broken)


def test_bob_verifies_alices_client_and_the_log_row_has_bob_and_the_details(app, alice_bob, fake_workers):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _panel("bob")
    panel.clientIdInput.setText("c")
    panel.selected_image_path = "probe.png"
    panel.selected_trajectory = _trajectory()
    panel.run_verification()
    worker, = fake_workers
    assert len(worker.args[0]) == 1 and "under your account" not in panel.resultDisplay.text()
    panel._on_verify_success(dict(RESULT))
    row = storage.get_last_verification_for_client("c")
    assert row["employee"] == "bob" and row["routing"] == "multimodal" and row["degraded"] is False
    assert row["reasons"] == [] and row["threshold"] == 0.76
    assert row["model_version"] == storage.MODEL_VERSION
    verify_hashes = [h for e, h in _audit_hashes() if e == "verify"]
    assert len(verify_hashes[-1]) == 2


def test_stale_set_is_refused_with_the_message_and_verify_is_not_started(app, db, fake_workers, monkeypatch):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    monkeypatch.setattr(storage, "MODEL_VERSION", "v-next")
    panel = _panel()
    panel.clientIdInput.setText("c")
    panel.selected_image_path = "probe.png"
    panel.selected_trajectory = _trajectory()
    panel.run_verification()
    assert fake_workers == []
    text = panel.resultDisplay.text()
    assert "re-enrol" in text and "\n" in text and panel.verifyButton.isEnabled()
    row = storage.get_audit_log()[-1]            # every refused attempt is logged
    assert (row["event"], row["client_key"], row["employee"]) == ("verify refused", "c", "alice")
    assert row["reasons"] == text.split("\n")
    assert storage.get_verification_stats()["total"] == 0


def test_pen_probe_from_another_tablet_is_refused(app, db, fake_workers, monkeypatch):
    import src.capture.collect as collect
    xppen = {"device_model": "XPPen Deco 640", "os": "Windows", "qt_platform": "windows"}
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=None, capture_device=xppen)
    other = {"signatures": [_trajectory()], "os": "Linux", "device_model": "Wacom One", "qt_platform": "xcb"}
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: other)
    panel = _panel()
    panel.modalitySelector.setCurrentIndex(2)
    panel.clientIdInput.setText("c")
    panel.capture_pen_probe()
    panel.run_verification()
    assert fake_workers == [] and "Wacom One" in panel.resultDisplay.text()
    row = storage.get_audit_log()[-1]
    assert (row["event"], row["employee"]) == ("verify refused", "alice")
    assert row["reasons"] == panel.resultDisplay.text().split("\n")
    same = {**other, "os": "Windows", "device_model": "XPPen Deco 640", "qt_platform": "windows"}
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: same)
    panel.capture_pen_probe()
    panel.run_verification()
    assert len(fake_workers) == 1


def test_confirm_passes_the_origin_of_the_set_used(app, db, fake_workers, monkeypatch):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel.clientIdInput.setText("c")
    panel.selected_image_path = "probe.png"
    panel.selected_trajectory = _trajectory()
    panel.run_verification()
    panel._on_verify_success(dict(RESULT))
    seen = []
    monkeypatch.setattr(storage, "add_confirmed_reference",
                        lambda *a, **k: seen.append(k) or True)
    panel._on_confirmed([{"spatial": _b64(512, 9), "temporal": _b64(128, 9)}])
    assert seen[0]["spatial_origin"] == "scan" and "spatial_origin" not in seen[1]


def test_confirmed_visit_adds_paper_and_pen(app, db):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    panel = _panel()
    panel._last_accepted_client = "c"
    panel._last_accepted_signings = [("probe.png", _trajectory())]
    panel._on_confirmed([{"spatial": _b64(512, 9), "temporal": _b64(128, 9)}])
    record = storage.get_client("c", "alice")
    assert len(record["spatial_embeddings"]) == 2 and len(record["dynamic_references"]) == 2


def test_storage_refuses_a_pen_reference_without_a_pen_set(db):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512)])
    assert storage.add_confirmed_reference("c", _b64(128), "alice", modality="temporal") is False
    with pytest.raises(ValueError):
        storage.add_confirmed_reference("c", _b64(128), "alice", modality="tilt")


def test_enrolment_tablet_button_and_tablet_only_save(app, db, fake_workers, monkeypatch):
    k = NUM_ENROLLMENT_REFERENCES
    captured = {"signatures": [_trajectory(i) for i in range(k)], "os": "Windows",
                "device_model": "XPPen", "qt_platform": "windows"}
    import src.capture.collect as collect
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: captured)
    panel = enrollment_page.EnrollmentPanel()
    panel.set_current_employee("alice")
    assert panel.captureDynamicButton.isEnabled()
    panel.clientIdInput.setText("c")
    panel.load_or_start_new_client()
    panel.capture_pen_signatures()
    assert panel.dynamicCountLabel.text() == f"{k}/{k} pen signatures"
    panel.run_enrollment()
    worker, = fake_workers
    assert worker.args == ([], captured["signatures"])

    panel._on_enroll_success({"embeddings": [], "spatial_origin": "scan",
                              "dynamic_embeddings": [_b64(128, i) for i in range(k)]})
    assert len(storage.get_client("c", "alice")["dynamic_references"]) == k
    assert panel.pen_signatures == []          # raw signatures are not kept


def test_enrolment_closed_tablet_window_changes_nothing(app, db, monkeypatch):
    import src.capture.collect as collect
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: None)
    panel = enrollment_page.EnrollmentPanel()
    panel.pen_signatures = ["kept"]
    panel.capture_pen_signatures()
    assert panel.pen_signatures == ["kept"]


def test_enrolment_delete_refused_shows_why_and_keeps_the_form(app, alice_bob, monkeypatch):
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    storage.save_client_enrollment("c", [], "bob", spatial_embeddings=[_b64(512)])
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a: QtWidgets.QMessageBox.Yes)
    panel = enrollment_page.EnrollmentPanel()
    panel.set_current_employee("bob")
    panel.clientIdInput.setText("c")
    panel.dynamic_references = ["kept"]
    panel.delete_client()
    assert panel.statusLabel.text() == "⚠ Only the owner can delete a signer"
    assert panel.dynamic_references == ["kept"] and storage.client_exists("c")


def test_enrolment_delete_clears_the_deleted_signers_status(app, db, monkeypatch):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512)])
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a: QtWidgets.QMessageBox.Yes)
    panel = _enrol_panel("alice")
    _load(panel, "c")
    assert panel.clientStatusLabel.text().startswith("Loaded existing signer 'c'")
    panel.delete_client()
    assert not storage.client_exists("c")
    assert panel.clientStatusLabel.text() == ""


def _enrol_panel(username):
    panel = enrollment_page.EnrollmentPanel()
    panel.set_current_employee(username, is_admin=(username == "alice"))
    return panel


def _load(panel, client_id):
    panel.clientIdInput.setText(client_id)
    panel.load_or_start_new_client()


def test_enrolment_non_admin_adds_paper_to_a_pen_only_client(app, alice_bob):
    k = NUM_ENROLLMENT_REFERENCES
    storage.save_client_enrollment("c", [_b64(128, i) for i in range(k)], "alice")
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _enrol_panel("bob")
    _load(panel, "c")
    assert panel.selectImageButton.isEnabled() and panel.addImageButton.isEnabled()
    assert not panel.captureDynamicButton.isEnabled()
    assert panel.deleteClientButton.isHidden()
    assert "Only the owner can replace these" in panel.clientStatusLabel.text()
    panel._pending_client_id = "c"
    panel._on_enroll_success({"embeddings": [_b64(512, i) for i in range(k)],
                              "dynamic_embeddings": [], "spatial_origin": "scan"})
    assert panel.statusLabel.text().startswith("✔")
    rec = storage.get_client("c")
    assert len(rec["spatial_sets"]["scan"]["embeddings"]) == k and len(rec["dynamic_references"]) == k


def test_enrolment_non_admin_cannot_replace_a_paper_set(app, alice_bob):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512, i) for i in range(5)])
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _enrol_panel("bob")
    _load(panel, "c")
    assert not panel.selectImageButton.isEnabled() and not panel.addImageButton.isEnabled()
    assert panel.captureDynamicButton.isEnabled()          # no pen set yet: Bob may add one


def test_enrolment_new_client_is_open_to_everyone(app, alice_bob):
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _enrol_panel("bob")
    _load(panel, "new")
    assert panel.selectImageButton.isEnabled() and panel.captureDynamicButton.isEnabled()
    assert panel.deleteClientButton.isHidden()             # Delete is the admin's, new or not


def test_enrolment_lookup_is_company_wide(app, alice_bob):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512, 1)])
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _enrol_panel("bob")
    _load(panel, "c")
    assert panel.clientStatusLabel.text().startswith("Loaded existing signer")
    assert "another member" not in panel.clientStatusLabel.text()


def test_enrolment_admin_replacing_paper_asks_and_keeps_the_pen_rows(app, alice_bob, fake_workers, monkeypatch):
    k = NUM_ENROLLMENT_REFERENCES
    storage.save_client_enrollment("c", [_b64(128, i) for i in range(k)], "alice",
                                   spatial_embeddings=[_b64(512, i) for i in range(k)],
                                   capture_device={"device_model": "XPPen", "os": "Linux"})
    assert storage.add_confirmed_reference("c", _b64(128, 99), "alice", modality="temporal")

    def pen_rows():
        db = storage._get_db()
        with on_disk(db.db_path, db._dek) as conn:
            return conn.execute("SELECT source, model_version, capture_device FROM enrollments "
                                "WHERE modality = 'temporal' ORDER BY id").fetchall()
    before = pen_rows()
    asked = []
    answers = iter([QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes])
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a: asked.append(a[2]) or next(answers))
    panel = _enrol_panel("alice")
    _load(panel, "c")
    assert panel.selectImageButton.isEnabled() and panel.captureDynamicButton.isEnabled()
    new_scan = [_b64(512, 100 + i) for i in range(k)]
    result = {"embeddings": new_scan, "dynamic_embeddings": [], "spatial_origin": "scan"}
    panel.image_references = ["a.png"] * k
    panel.run_enrollment()                      # answer No: no worker, nothing saved
    assert asked == ["This replaces the signer's paper references. Continue?"]
    assert fake_workers == []
    panel.run_enrollment()                      # answer Yes
    assert len(fake_workers) == 1
    panel._pending_client_id = "c"
    panel._on_enroll_success(result)
    assert storage.get_client("c")["spatial_sets"]["scan"]["embeddings"] == new_scan
    assert pen_rows() == before
    assert storage.add_confirmed_reference("c", _b64(128, 100), "alice", modality="temporal") is True


def _ask_log(monkeypatch):
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a: asked.append(a[2]) or QtWidgets.QMessageBox.Yes)
    return asked


def test_enrolment_confirmation_follows_the_id_being_saved_not_the_last_load(app, alice_bob, fake_workers, monkeypatch):
    k = NUM_ENROLLMENT_REFERENCES
    storage.save_client_enrollment("a", [], "alice", spatial_embeddings=[_b64(512, i) for i in range(k)])
    storage.save_client_enrollment("b", [], "alice", spatial_embeddings=[])
    storage.save_client_enrollment("old", [], "alice", spatial_embeddings=[_b64(512, 50 + i) for i in range(k)])
    asked = _ask_log(monkeypatch)
    panel = _enrol_panel("alice")
    _load(panel, "a")                              # (a) loaded A has paper, then B is typed
    panel.clientIdInput.setText("b")
    panel.image_references = ["x.png"] * k
    panel.run_enrollment()
    assert asked == [] and len(fake_workers) == 1
    _load(panel, "brand-new")                      # (b) loaded a new ID, then an existing one is typed
    panel.clientIdInput.setText("old")
    panel.image_references = ["x.png"] * k
    panel.run_enrollment()
    assert asked == ["This replaces the signer's paper references. Continue?"]


def test_enrolment_editing_the_id_after_load_reapplies_permissions(app, alice_bob):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512, i) for i in range(5)])
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _enrol_panel("bob")
    _load(panel, "c")
    assert not panel.selectImageButton.isEnabled()
    panel.clientIdInput.setText("other")
    assert panel.loaded_client is None and panel.selectImageButton.isEnabled()


def test_enrolment_new_pen_passes_the_capture_device(app, db, fake_workers, monkeypatch):
    k = NUM_ENROLLMENT_REFERENCES
    captured = {"signatures": [_trajectory(i) for i in range(k)], "os": "Windows",
                "device_model": "XPPen", "qt_platform": "windows"}
    import src.capture.collect as collect
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: captured)
    panel = _enrol_panel("alice")
    _load(panel, "c")
    panel.capture_pen_signatures()
    panel.run_enrollment()
    panel._on_enroll_success({"embeddings": [], "spatial_origin": "scan",
                              "dynamic_embeddings": [_b64(128, i) for i in range(k)]})
    device = storage.get_client("c")["temporal_set"]["capture_device"]
    assert device == {"os": "Windows", "device_model": "XPPen", "qt_platform": "windows"}


def _pen_result(worker, k):
    """What EnrollWorker would return for the inputs it was given."""
    images, pens = worker.args
    return {"embeddings": [_b64(512, i) for i in range(len(images))], "spatial_origin": "scan",
            "dynamic_embeddings": [_b64(128, i) for i in range(len(pens))]}


def test_enrolment_editing_the_id_drops_the_previous_clients_signings(app, db, fake_workers, monkeypatch):
    k = NUM_ENROLLMENT_REFERENCES
    import src.capture.collect as collect
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: {
        "signatures": [_trajectory(i) for i in range(k)], "os": "Windows",
        "device_model": "XPPen", "qt_platform": "windows"})
    panel = _enrol_panel("alice")
    _load(panel, "a")
    panel.capture_pen_signatures()
    panel.image_references = ["a.png"] * k
    panel.clientIdInput.setText("b")            # A's signings and photos must not go to B
    assert panel.pen_signatures == [] and panel.image_references == []
    assert panel._capture_device is None and panel.imageList.count() == 0
    assert panel.dynamicCountLabel.text() == f"0/{k} pen signatures"
    panel.image_references = ["b.png"] * k
    panel.run_enrollment()
    worker, = fake_workers
    panel._on_enroll_success(_pen_result(worker, k))
    assert storage.get_client("b")["temporal_set"] is None


def test_enrolment_save_uses_what_was_captured_when_save_was_pressed(app, db, fake_workers, monkeypatch):
    k = NUM_ENROLLMENT_REFERENCES
    import src.capture.collect as collect
    devices = iter(["XPPen", "Wacom One"])
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: {
        "signatures": [_trajectory(i) for i in range(k)], "os": "Windows",
        "device_model": next(devices), "qt_platform": "windows"})
    panel = _enrol_panel("alice")
    _load(panel, "c")
    panel.capture_pen_signatures()
    panel.run_enrollment()
    panel.capture_pen_signatures()               # a new capture while the worker runs
    panel._on_enroll_success(_pen_result(fake_workers[0], k))
    assert storage.get_client("c")["temporal_set"]["capture_device"]["device_model"] == "XPPen"

    _load(panel, "d")
    monkeypatch.setattr(collect, "collect_signatures", lambda count, parent=None: {
        "signatures": [_trajectory(i) for i in range(k)], "os": "Linux",
        "device_model": "XPPen", "qt_platform": "xcb"})
    panel.capture_pen_signatures()
    panel.run_enrollment()
    _load(panel, "other")                        # Load while the worker runs
    panel._on_enroll_success(_pen_result(fake_workers[1], k))
    rec = storage.get_client("d")
    assert len(rec["dynamic_references"]) == k and rec["temporal_set"]["capture_device"]["os"] == "Linux"
    assert panel.clientIdInput.text() == "other" and panel.loaded_client is None


def test_enrolment_storage_refusal_is_shown_not_a_success(app, alice_bob, monkeypatch):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512, i) for i in range(5)])
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    panel = _enrol_panel("bob")                  # opened as a new client by mistake: a race
    panel._pending_client_id = "c"
    panel._on_enroll_success({"embeddings": [_b64(512, 7)], "dynamic_embeddings": [],
                              "spatial_origin": "scan"})
    assert "Only the owner can replace" in panel.statusLabel.text()
    assert not panel.statusLabel.text().startswith("✔")


# ------------------------------------------------------------------ app shell, accounts, backup
from PyQt5.QtCore import QTimer  # noqa: E402
from src.gui import app as app_module, employees_page, profile_page, sidebar  # noqa: E402
from src.gui.clients_page import ClientsPage  # noqa: E402
from src.gui.dashboard_page import DashboardScreen  # noqa: E402


@pytest.fixture
def dialogs(monkeypatch):
    """QMessageBox.critical / information are recorded; question answers Yes."""
    shown = []
    box = QtWidgets.QMessageBox
    monkeypatch.setattr(box, "critical", lambda *a: shown.append(a))
    monkeypatch.setattr(box, "information", lambda *a: shown.append(a))
    monkeypatch.setattr(box, "question", lambda *a: box.Yes)
    return shown


@pytest.fixture
def make_window(app, monkeypatch, dialogs):
    monkeypatch.setattr(app_module, "WarmUpWorker", FakeWorker)   # never load the model
    return app_module.MainWindow


def _buttons(widget):
    return [b.text() for b in widget.findChildren(QtWidgets.QPushButton)]


def _login(window, username, password):
    window.auth_stack.setCurrentWidget(window.login_screen)
    window.login_screen.usernameInput.setText(username)
    window.login_screen.passwordInput.setText(password)
    window.login_screen._handle_login()
    _finish_login(window)


def _finish_login(window):
    """Wait for the login worker, then deliver its queued signals to the GUI thread."""
    window.login_screen._worker.wait()
    QtWidgets.QApplication.processEvents()


def _register(screen, company, full_name, username, password):
    screen.fullNameInput.setText(full_name)         # first: it prefills the vault name
    screen.companyInput.setText(company)
    screen.usernameInput.setText(username)
    screen.passwordInput.setText(password)
    screen.confirmPasswordInput.setText(password)
    screen.createButton.click()


def test_first_run_registers_a_company_and_lands_in_it_as_its_admin(empty_db, make_window):
    window = make_window()
    screen = window.auth_stack.currentWidget()
    assert isinstance(screen, app_module.RegisterScreen)
    labels = [label.text() for label in screen.findChildren(QtWidgets.QLabel)]
    assert "Create a vault" in labels and storage.ADMIN_WARNING in labels
    assert "Create Vault" in _buttons(screen) and "I already have a vault: restore it from a backup" in _buttons(screen)
    assert "Back to login" not in _buttons(screen)            # nothing to go back to on a first run

    _register(screen, "", "Alice A", "alice", "alice-password-1")
    assert screen.messageLabel.text() == "⚠ Please fill in all fields" and storage.is_first_run()
    _register(screen, "Acme", "Alice A", "alice", "alice-password-1")
    assert window.current_employee == {"full_name": "Alice A", "username": "alice", "is_admin": True}
    assert window.content_stack.current_page() is window.dashboard_screen
    assert storage.list_companies() == ["Acme"] and storage.current_company() == "Acme"
    assert window.sidebar.companyLabel.text() == "Acme"

    window.sidebar.go_logout.emit()
    assert window.auth_stack.currentWidget() is window.login_screen
    assert _companies(window) == ["Acme"] and window.login_screen.companyInput.currentText() == "Acme"
    assert "Create a new vault" in _buttons(window.login_screen)
    storage._reset()                                          # the app starts again
    later = make_window()
    assert isinstance(later.auth_stack.currentWidget(), app_module.WelcomeScreen)
    _login(later, "alice", "alice-password-1")
    assert later.current_employee["username"] == "alice" and storage.current_company() == "Acme"


def test_typing_a_full_name_fills_the_vault_name_until_the_user_edits_it(empty_db, make_window):
    screen = make_window().auth_stack.currentWidget()
    assert screen.companyInput.placeholderText() == "Vault name (your company, branch, or your own name)"
    screen.fullNameInput.setText("Ali")
    screen.fullNameInput.setText("Alice A")
    assert screen.companyInput.text() == "Alice A"
    screen.companyInput.setText("Acme Bank")                  # the user's own edit
    screen.fullNameInput.setText("Alice Anders")
    assert screen.companyInput.text() == "Acme Bank"


def test_login_hides_the_vault_list_with_one_vault_and_shows_it_with_two(db, make_window):
    storage.logout()
    window = make_window()
    window.auth_stack.setCurrentWidget(window.login_screen)
    assert window.login_screen.companyInput.isHidden()
    _login(window, "alice", "alice-password-1")
    assert window.current_employee["username"] == "alice"
    window.sidebar.go_logout.emit()

    window.login_screen.registerButton.click()
    _register(window.register_screen, "Beta", "Zed Z", "zed", "zed-password-12")
    window.sidebar.go_logout.emit()
    assert not window.login_screen.companyInput.isHidden()
    window.login_screen.companyInput.setCurrentText("Test Co")
    _login(window, "alice", "alice-password-1")
    assert storage.current_company() == "Test Co"


def _companies(window):
    combo = window.login_screen.companyInput
    return [combo.itemText(i) for i in range(combo.count())]


def test_a_company_registered_from_login_sees_none_of_the_other_companys_clients(db, make_window):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    storage.logout()
    window = make_window()
    window.auth_stack.setCurrentWidget(window.login_screen)
    window.login_screen.registerButton.click()
    screen = window.register_screen
    assert window.auth_stack.currentWidget() is screen
    screen.backButton.click()
    assert window.auth_stack.currentWidget() is window.login_screen
    window.login_screen.registerButton.click()
    _register(screen, "Beta", "Zed Z", "zed", "zed-password-12")
    assert window.current_employee == {"full_name": "Zed Z", "username": "zed", "is_admin": True}
    assert window.sidebar.companyLabel.text() == "Beta" and window.clients_page.table.rowCount() == 0

    window.sidebar.go_logout.emit()
    assert _companies(window) == ["Beta", "Test Co"] and window.login_screen.companyInput.currentText() == "Beta"
    _login(window, "zed", "zed-password-12")
    assert storage.list_clients() == [] and window.clients_page.table.rowCount() == 0
    window.sidebar.go_logout.emit()
    _login(window, "alice", "alice-password-1")                # Beta's login screen still says Beta
    assert window.current_employee is None
    window.login_screen.companyInput.setCurrentText("Test Co")
    _login(window, "alice", "alice-password-1")
    assert storage.current_company() == "Test Co" and window.sidebar.companyLabel.text() == "Test Co"
    assert window.clients_page.table.rowCount() == 1


def test_startup_storage_error_shows_it_and_quits(app, empty_db, make_window, dialogs, monkeypatch, tmp_path):
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "companies").write_text("not a folder")       # the company list cannot be read
    later = []
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, slot: later.append(slot))
    window = make_window()
    assert len(dialogs) == 1 and dialogs[0][:2] == (None, "Cannot open the signature data")
    assert "vault folder" in dialogs[0][2]
    assert later == [QtWidgets.QApplication.instance().quit]
    assert window.startup_failed and window.centralWidget() is None


def test_login_storage_error_shows_a_dialog_and_stays(db, make_window, dialogs, monkeypatch):
    storage.logout()
    window = make_window()

    def broken(username, password):
        raise storage.StorageError("The database file cannot be read.")
    monkeypatch.setattr(storage, "authenticate_employee", broken)
    _login(window, "alice", "alice-password-1")
    assert dialogs[-1][1:] == ("Cannot open the signature data", "The database file cannot be read.")
    assert window.current_employee is None
    assert window.auth_stack.currentWidget() is window.login_screen


def test_unexpected_error_shows_a_dialog_and_prints_the_traceback(app, dialogs, capsys):
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        app_module.show_unexpected_error(type(e), e, e.__traceback__)
    assert dialogs[-1][2] == "Something went wrong: boom. The action was not completed."
    assert "RuntimeError: boom" in capsys.readouterr().err


def test_main_installs_the_error_dialog(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)   # restored after the test
    app_module.install_error_dialog()
    assert sys.excepthook is app_module.show_unexpected_error


def test_logout_locks_the_signature_data(db, make_window):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=[_b64(512)])
    storage.logout()
    window = make_window()
    _login(window, "alice", "alice-password-1")
    assert storage.get_client("c") is not None
    window.sidebar.go_logout.emit()
    with pytest.raises(storage.StorageLockedError):
        storage.get_client("c")
    assert window.auth_stack.currentWidget() is window.login_screen


def test_sidebar_shows_employees_to_the_admin_only(alice_bob, make_window):
    storage.logout()
    window = make_window()
    _login(window, "bob", "bob-password-1-long")
    bar = window.sidebar
    assert bar.employeesButton.isHidden()
    bar.toggle_collapse()
    bar.toggle_collapse()
    assert bar.employeesButton.isHidden() and not bar.logoutButton.isHidden()

    bar = sidebar.Sidebar()
    bar.set_admin(True)
    assert not bar.employeesButton.isHidden()
    bar.toggle_collapse()
    assert bar.employeesButton.isHidden()
    bar.toggle_collapse()
    assert not bar.employeesButton.isHidden()


def _employees_page():
    page = employees_page.EmployeesPage()
    page.set_current_employee("alice")
    page.refresh()
    return page


def _select(page, username):
    row = [page.table.item(r, 1).text() for r in range(page.table.rowCount())].index(username)
    page.table.setCurrentCell(row, 0)


def test_employees_page_add_reset_remove_change_only_that_account(app, alice_bob, dialogs):
    page = _employees_page()
    assert [page.table.item(r, 1).text() for r in range(page.table.rowCount())] == ["alice", "bob"]

    page.fullNameInput.setText("Carol C")
    page.usernameInput.setText("carol")
    page.passwordInput.setText("carol-password-1")
    page.confirmPasswordInput.setText("something-else-1")
    page.addButton.click()
    assert page.statusLabel.text() == "⚠ Passwords do not match" and storage.find_employee("carol") is None
    page.confirmPasswordInput.setText("carol-password-1")
    page.addButton.click()
    assert page.statusLabel.text() == "✔ Account created successfully"
    assert storage.find_employee("carol") == {"full_name": "Carol C", "username": "carol", "is_admin": False}
    assert page.table.rowCount() == 3

    _select(page, "bob")
    page.newPasswordInput.setText("bob-new-password-1")
    page.confirmNewPasswordInput.setText("bob-new-password-1")
    page.resetButton.click()
    assert page.statusLabel.text() == "✔ New password set for bob"

    _select(page, "alice")
    page.removeButton.click()
    assert page.statusLabel.text() == "⚠ The owner cannot remove their own account"
    _select(page, "carol")
    page.removeButton.click()
    assert page.statusLabel.text() == "✔ Removed carol"
    assert storage.find_employee("carol") is None and page.table.rowCount() == 2

    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0] is False
    assert storage.authenticate_employee("bob", "bob-new-password-1")[0]
    assert storage.authenticate_employee("alice", "alice-password-1")[0]


def _answer_restore(monkeypatch, path, username, password, company=None):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))
    if company:
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: (company, True))
    from src.gui import employees_page
    monkeypatch.setattr(employees_page, "ask_backup_owner", lambda parent: (username, password))


def test_backup_then_restore_returns_to_login(db, make_window, tmp_path, monkeypatch):
    storage.logout()
    window = make_window()
    _login(window, "alice", "alice-password-1")
    window.sidebar.go_employees.emit()
    page = window.employees_page
    assert window.content_stack.current_page() is page

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    page.backupButton.click()
    files = [f for f in os.listdir(tmp_path) if f.startswith("signatures-backup-")]
    assert len(files) == 1 and page.statusLabel.text() == f"✔ Backup saved to {tmp_path / files[0]}"
    backup = str(tmp_path / files[0])

    _answer_restore(monkeypatch, backup, "alice", "wrong-password-1")
    page.restoreButton.click()
    assert page.statusLabel.text().startswith("⚠ ") and window.content_stack.current_page() is page

    _answer_restore(monkeypatch, backup, "alice", "alice-password-1")
    page.restoreButton.click()
    assert window.auth_stack.currentWidget() is window.login_screen
    assert window.login_screen.errorLabel.text() == "Backup restored. Log in again."
    assert storage.current_employee() is None


def test_first_run_restore_returns_to_login(db, make_window, tmp_path, monkeypatch):
    ok, backup = storage.backup_database(str(tmp_path), "alice")
    assert ok
    storage._reset()
    monkeypatch.setenv(storage.HOME_ENV_VAR, str(tmp_path / "new-install"))
    window = make_window()
    screen = window.auth_stack.currentWidget()
    assert isinstance(screen, app_module.RegisterScreen)
    _answer_restore(monkeypatch, backup, "alice", "wrong-password-1", company="Test Co")
    screen.restoreButton.click()
    assert screen.messageLabel.text().startswith("⚠ ") and storage.list_companies() == []
    _answer_restore(monkeypatch, backup, "alice", "alice-password-1", company="Restored Co")
    screen.restoreButton.click()
    assert screen.messageLabel.text() == "⚠ This backup belongs to vault 'Test Co'."
    _answer_restore(monkeypatch, backup, "alice", "alice-password-1", company="Test Co")
    screen.restoreButton.click()
    assert window.auth_stack.currentWidget() is window.login_screen
    assert window.login_screen.errorLabel.text() == "Backup restored. Log in again."
    _login(window, "alice", "alice-password-1")
    assert window.current_employee["username"] == "alice" and storage.current_company() == "Test Co"


def test_profile_password_change_needs_the_current_password(app, db):
    page = profile_page.ProfilePage()
    page.load_employee(storage.current_employee())
    page.newPasswordInput.setText("alice-new-pass-1")
    page.confirmNewPasswordInput.setText("alice-new-pass-1")
    page.saveButton.click()
    assert page.statusLabel.text() == "⚠ Enter your current password to set a new one"
    page.currentPasswordInput.setText("wrong-password-1")
    page.saveButton.click()
    assert page.statusLabel.text() == "⚠ The current password is incorrect"
    page.currentPasswordInput.setText("alice-password-1")
    page.saveButton.click()
    assert page.statusLabel.text() == "✔ Profile updated successfully"
    storage.logout()
    assert storage.authenticate_employee("alice", "alice-new-pass-1")[0]


def test_profile_new_password_must_match_its_confirmation(app, db, monkeypatch):
    called = []
    monkeypatch.setattr(storage, "update_employee_profile", lambda *a, **k: called.append(a))
    page = profile_page.ProfilePage()
    page.load_employee(storage.current_employee())
    page.newPasswordInput.setText("alice-new-pass-1")
    page.confirmNewPasswordInput.setText("alice-new-pass-2")
    page.currentPasswordInput.setText("alice-password-1")
    page.saveButton.click()
    assert page.statusLabel.text() == "⚠ The new passwords do not match"
    assert called == []


def test_profile_password_is_kept_exactly_as_typed(app, db):
    page = profile_page.ProfilePage()            # no strip, as at login and registration
    page.load_employee(storage.current_employee())
    page.newPasswordInput.setText("alice-new-pass-1 ")
    page.confirmNewPasswordInput.setText("alice-new-pass-1 ")
    page.currentPasswordInput.setText("alice-password-1")
    page.saveButton.click()
    assert page.statusLabel.text() == "✔ Profile updated successfully"
    storage.logout()
    assert storage.authenticate_employee("alice", "alice-new-pass-1")[0] is False
    assert storage.authenticate_employee("alice", "alice-new-pass-1 ")[0]


def _alice_client_and_one_bob_verification():
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    storage.append_verification_log("c", "Genuine", 90.0, "multimodal", "alice")
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    storage.append_verification_log("c", "Forged", 20.0, "multimodal", "bob")


def test_bobs_clients_page_lists_alices_client_and_the_last_verification(app, alice_bob):
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    storage.append_verification_log("c", "Genuine", 90.0, "multimodal", "alice")   # only Alice verifies
    alices = storage.get_last_verification_for_client("c")["timestamp"]
    storage.logout()
    assert storage.authenticate_employee("bob", "bob-password-1-long")[0]
    page = ClientsPage()
    page.refresh("bob")
    assert page.table.rowCount() == 1 and page.table.item(0, 0).text() == "c"
    assert page.table.item(0, 3).text() == alices != "Never"
    expires = storage.get_client("c")["spatial_sets"]["scan"]["expires_at"][:10]
    assert page.table.horizontalHeaderItem(4).text() == "References Expire"
    assert page.table.item(0, 4).text() == expires


def test_dashboard_mine_and_company_counts_differ(app, alice_bob):
    _alice_client_and_one_bob_verification()
    screen = DashboardScreen()
    screen.refresh("bob")
    assert screen.mine.totalClientsTile.valueLabel.text() == "0"
    assert screen.mine.totalVerificationsTile.valueLabel.text() == "1"
    assert screen.company.totalClientsTile.valueLabel.text() == "1"
    assert screen.company.totalVerificationsTile.valueLabel.text() == "2"
    assert screen.mine.newClientsTile.valueLabel.text() == "0"
    assert screen.company.newClientsTile.valueLabel.text() == "1"


def test_dashboard_recent_activity_is_company_wide_and_names_the_employee(app, alice_bob):
    _alice_client_and_one_bob_verification()
    screen = DashboardScreen()
    screen.refresh("bob")
    texts = [w.text() for w in screen.recentTile.findChildren(QtWidgets.QLabel)]
    assert any("alice" in t for t in texts) and any("bob" in t for t in texts)


def test_employees_page_shows_last_backup_and_last_delete(app, alice_bob, tmp_path, monkeypatch):
    page = employees_page.EmployeesPage()
    page.set_current_employee("alice")
    page.refresh()
    assert page.lastBackupLabel.text() == "Last backup: never"
    assert page.lastDeleteLabel.text() == "Last signer deletion: none"

    # Back up
    bak_dir = tmp_path / "backups"
    bak_dir.mkdir()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(bak_dir))
    page.backupButton.click()
    assert "Last backup: " in page.lastBackupLabel.text()
    assert "by alice" in page.lastBackupLabel.text()

    # Delete a client
    storage.save_client_enrollment("client_x", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    storage.delete_client("client_x")
    page.refresh()
    assert "Last signer deletion: client_x at " in page.lastDeleteLabel.text()
    assert "by alice" in page.lastDeleteLabel.text()



# ------------------------------------------------------------------ small screens (issue #23, item 25)
from PyQt5.QtCore import QPoint, QRect  # noqa: E402

SCREENS = [(1024, 700), (1366, 768)]
PAGES = ["dashboard", "enrolment", "verification", "clients", "employees", "profile"]
INPUTS = (QtWidgets.QAbstractButton, QtWidgets.QLineEdit, QtWidgets.QComboBox,
          QtWidgets.QAbstractItemView)
LONG_MESSAGE = "This message is far longer than any window is wide. " * 8


@pytest.fixture
def window(alice_bob, make_window):
    """Alice (admin) is logged in; the signer 'c' has paper and pen references."""
    storage.save_client_enrollment("c", [_b64(128)], "alice", spatial_embeddings=[_b64(512)])
    win = make_window()
    win._handle_login_success(storage.current_employee())
    win.show()
    yield win
    win.close()


def _open_page(win, name):
    if name == "enrolment":
        win._open_client_in_enrollment("c")
        return win.enrollment_panel
    if name == "clients":
        win._open_clients_page()
        return win.clients_page
    if name == "employees":
        win._open_employees_page()
        return win.employees_page
    page = {"dashboard": win.dashboard_screen, "verification": win.verification_panel,
            "profile": win.profile_page}[name]
    win.content_stack.show_page(page)
    return page


def _reachable(app, area, widget):
    """The widget is in view, or scrolling its page's area brings all of it into view."""
    viewport = area.viewport()

    def in_view():
        return viewport.rect().contains(QRect(widget.mapTo(viewport, QPoint(0, 0)), widget.size()))
    if in_view():
        return True
    area.ensureWidgetVisible(widget, 0, 0)
    app.processEvents()
    return in_view()


@pytest.mark.parametrize("size", SCREENS)
@pytest.mark.parametrize("name", PAGES)
def test_every_page_fits_a_small_screen_and_every_control_is_reachable(app, window, name, size):
    width, height = size
    window.resize(width, height)
    page = _open_page(window, name)
    app.processEvents()
    assert window.width() <= width and window.height() <= height     # the page did not force it bigger
    area = window.content_stack.area_of(page)
    controls = [w for w in page.findChildren(INPUTS) if w.isVisible() and w.isEnabled()]
    assert controls or name == "dashboard"                          # the dashboard has none
    unreachable =[f"{type(w).__name__} {getattr(w, 'text', lambda: '')()!r}"
                   for w in controls if not _reachable(app, area, w)]
    assert unreachable == []


@pytest.mark.parametrize("name, label", [("enrolment", "statusLabel"), ("verification", "resultDisplay"),
                                         ("profile", "statusLabel"), ("employees", "statusLabel")])
def test_a_long_message_wraps_instead_of_widening_the_page(app, window, name, label):
    window.resize(*SCREENS[0])
    page = _open_page(window, name)
    getattr(page, label).setText(LONG_MESSAGE)
    app.processEvents()
    area = window.content_stack.area_of(page)
    assert window.width() <= SCREENS[0][0] and area.horizontalScrollBar().maximum() == 0


def test_bob_loading_a_signer_he_cannot_change_keeps_the_page_inside_the_window(app, window):
    storage.logout()
    ok, bob, _ = storage.authenticate_employee("bob", "bob-password-1-long")
    assert ok
    window._handle_login_success(bob)
    window.resize(*SCREENS[0])
    page = _open_page(window, "enrolment")
    app.processEvents()
    assert "Only the owner can replace these" in page.clientStatusLabel.text()
    area = window.content_stack.area_of(page)
    assert window.width() <= SCREENS[0][0] and area.horizontalScrollBar().maximum() == 0


def test_every_password_field_has_the_eye_and_every_new_password_has_generate(db, make_window):
    from src.gui.employees_page import BackupOwnerDialog
    from src.gui.password_field import PasswordField
    storage.logout()
    window = make_window()
    _login(window, "alice", "alice-password-1")
    screens = (window.login_screen, window.register_screen, window.employees_page, window.profile_page,
               BackupOwnerDialog(None))
    for screen in screens:
        fields = [f for f in screen.findChildren(QtWidgets.QLineEdit) if f.echoMode() == QtWidgets.QLineEdit.Password]
        assert fields and all(isinstance(f, PasswordField) for f in fields), type(screen).__name__
    for screen in (window.register_screen, window.employees_page, window.profile_page):
        assert "Generate password" in _buttons(screen), type(screen).__name__
    assert _buttons(window.employees_page).count("Generate password") == 2   # add a member, reset


def test_the_login_title_reads_login(empty_db, make_window):
    labels = [label.text() for label in make_window().login_screen.findChildren(QtWidgets.QLabel)]
    assert "Login" in labels and "Member Login" not in labels


def test_the_side_menu_is_painted_a_different_shade_from_the_page(app):
    from PyQt5.QtGui import QColor
    from src.gui.styles import PRIMARY, PRIMARY_DARK, STYLE
    window = QtWidgets.QWidget()                             # as in the app: a child of the styled window
    window.setStyleSheet(STYLE)
    window.resize(600, 700)
    menu = sidebar.Sidebar()
    menu.setParent(window)
    menu.resize(sidebar.Sidebar.EXPANDED_WIDTH, 700)
    image = window.grab().toImage()
    empty = image.pixelColor(menu.width() // 2, 500)        # between Profile and Logout: background only
    assert empty == QColor(PRIMARY) and empty != QColor(PRIMARY_DARK)


def test_login_runs_off_the_gui_thread_and_shows_it_is_working(db, make_window, monkeypatch):
    import threading
    storage.logout()
    window = make_window()
    seen = {}
    gate = threading.Event()
    real = storage.authenticate_employee

    def slow_check(username, password):
        seen["thread"] = threading.current_thread()
        gate.wait(5)
        return real(username, password)

    monkeypatch.setattr(storage, "authenticate_employee", slow_check)
    screen = window.login_screen
    screen.usernameInput.setText("alice")
    screen.passwordInput.setText("alice-password-1")
    screen.loginButton.click()
    assert not screen.loginButton.isEnabled() and screen.loginButton.text() == "Logging in..."
    assert QtWidgets.QApplication.overrideCursor().shape() == app_module.Qt.WaitCursor
    screen.loginButton.click()                                 # a second press starts nothing
    gate.set()
    _finish_login(window)
    assert seen["thread"] is not threading.main_thread()
    assert window.current_employee["username"] == "alice"
    assert screen.loginButton.isEnabled() and screen.loginButton.text() == "Login"
    assert QtWidgets.QApplication.overrideCursor() is None


def test_a_wrong_password_and_a_storage_error_keep_their_messages_after_the_worker(db, make_window, dialogs,
                                                                                  monkeypatch):
    storage.logout()
    window = make_window()
    _login(window, "alice", "wrong-password-1")
    assert window.login_screen.errorLabel.text().startswith("⚠ ") and window.current_employee is None

    def broken(_company):
        raise storage.StorageError("The vault file could not be read")

    monkeypatch.setattr(storage, "open_company", broken)
    _login(window, "alice", "alice-password-1")
    assert dialogs[-1][2] == "The vault file could not be read"
    assert window.login_screen.loginButton.isEnabled() and QtWidgets.QApplication.overrideCursor() is None


def test_high_dpi_scaling_is_on_before_the_application_is_made(monkeypatch):
    from PyQt5.QtCore import QCoreApplication, Qt
    seen = {}

    class Stop(Exception):
        pass

    class FakeApplication:
        setAttribute = staticmethod(QtWidgets.QApplication.setAttribute)

        def __init__(self, _argv):
            seen["scaling"] = QCoreApplication.testAttribute(Qt.AA_EnableHighDpiScaling)
            seen["pixmaps"] = QCoreApplication.testAttribute(Qt.AA_UseHighDpiPixmaps)
            raise Stop

    monkeypatch.setattr("src.capture.pen_input.choose_pen_input", lambda: None)
    monkeypatch.setattr(app_module, "QApplication", FakeApplication)
    with pytest.raises(Stop):
        app_module.main()
    assert seen == {"scaling": True, "pixmaps": True}


def test_the_window_opens_at_most_of_the_screen_not_a_fixed_size(db, make_window):
    screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
    window = make_window()
    assert window.width() == int(screen.width() * 0.8) and window.height() == int(screen.height() * 0.8)


def test_every_placeholder_on_the_login_and_vault_screens_is_shown_whole(empty_db, make_window):
    from src.gui.password_field import PasswordField
    from src.gui.styles import STYLE
    window = make_window()
    window.setStyleSheet(STYLE)
    for screen in (window.login_screen, window.register_screen):
        for field in screen.findChildren(QtWidgets.QLineEdit):
            field.ensurePolished()
            room = field.width() - 2 * 12 - 2 - (32 if isinstance(field, PasswordField) else 0)   # padding, border, eye
            assert field.fontMetrics().horizontalAdvance(field.placeholderText()) <= room, field.placeholderText()


def test_the_base_text_is_15_px():
    from src.gui.styles import STYLE
    assert "font-size: 15px;" in STYLE.split("QLabel {")[0]


@pytest.mark.parametrize("window_width", [380, 1600])
def test_the_vault_form_shrinks_in_a_narrow_window_and_stops_growing_in_a_wide_one(empty_db, make_window, app,
                                                                                window_width):
    window = make_window()
    window.show()
    window.resize(window_width, 800)
    app.processEvents()
    field = window.register_screen.passwordInput
    if window_width < app_module.FORM_WIDTH + 100:
        assert field.width() < app_module.FORM_WIDTH and field.mapTo(window, field.rect().topRight()).x() < window_width
    else:
        assert field.width() == app_module.FORM_WIDTH
    window.close()


def test_the_login_screen_shows_the_app_version(empty_db, make_window):
    from src.app_version import app_version
    labels = [label.text() for label in make_window().login_screen.findChildren(QtWidgets.QLabel)]
    assert f"Version {app_version()}" in labels
