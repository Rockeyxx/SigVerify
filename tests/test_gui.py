"""GUI tests without the model: routing to verify(), temp-file cleanup, messages, pen wiring.

The service is replaced by a fake that records its calls, so these tests need no weights
(issue #23, items 1, 2, 14, 15, 16).
"""
import base64
import os

import numpy as np
import pytest
import torch  # noqa: F401 — before PyQt5: on Windows the other order fails to load c10.dll

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from config.settings import NUM_ENROLLMENT_REFERENCES  # noqa: E402
from src.database.db_manager import DatabaseManager, PASSPHRASE_ENV_VAR  # noqa: E402
from src.gui import enrollment, interfaces, storage, verification, workers  # noqa: E402


def _blob(dim, seed=0):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32).tobytes()


def _b64(dim, seed=0):
    return base64.b64encode(_blob(dim, seed)).decode("ascii")


def _trajectory(seed=0):
    return np.random.default_rng(seed).standard_normal((50, 4)).astype(np.float32)


RESULT = {"decision": "Genuine", "confidence": 90.0, "routing": "multimodal",
          "degraded": False, "reasons": [], "threshold": 0.76}


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
    assert verification.visit_signings(["a", "b"], [t1, t2]) == [("a", t1), ("b", t2)]
    assert verification.visit_signings("a", None) == [("a", None)]
    assert verification.visit_signings(None, t1) == [(None, t1)]


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
    assert enrollment.REQUIRED_IMAGE_REFERENCES == NUM_ENROLLMENT_REFERENCES
    assert enrollment.REQUIRED_DYNAMIC_REFERENCES == NUM_ENROLLMENT_REFERENCES


# ------------------------------------------------------------------ panels
@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)
    monkeypatch.setenv(storage.HOME_ENV_VAR, str(tmp_path / "home"))
    monkeypatch.delenv(PASSPHRASE_ENV_VAR, raising=False)
    storage._reset()
    assert storage.register_employee("Alice A", "alice", "alice-password-1")[0]
    yield
    storage._reset()


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
    monkeypatch.setattr(verification, "VerifyWorker", FakeWorker)
    monkeypatch.setattr(enrollment, "EnrollWorker", FakeWorker)
    return FakeWorker.made


def _panel(employee="alice"):
    panel = verification.VerificationPanel()
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

    panel._on_verify_success({**RESULT, "routing": "temporal_only", "degraded": True})
    assert panel.confirmButton.isHidden()


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
    panel = enrollment.EnrollmentPanel()
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
    panel = enrollment.EnrollmentPanel()
    panel.pen_signatures = ["kept"]
    panel.capture_pen_signatures()
    assert panel.pen_signatures == ["kept"]
