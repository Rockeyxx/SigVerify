"""QThread workers running enrollment, verification, login, and warm-up off the GUI thread."""

from PyQt5.QtCore import QThread, pyqtSignal

# Track active threads so MainWindow can wait on shutdown (finished signal is reused below).
_started_workers = []


class TrackedThread(QThread):
    def start(self, *args):
        _started_workers[:] = [t for t in _started_workers if t.isRunning()]
        _started_workers.append(self)
        super().start(*args)


def wait_for_workers():
    """Wait for all active worker threads before closing the window."""
    for thread in list(_started_workers):
        if thread.isRunning():
            thread.wait()


WEIGHTS_MESSAGE = "Model weights not found. See README → Model Weights & GitHub Releases."


def _missing_file_message(error: FileNotFoundError) -> str:
    text = str(error)
    if "weights" in text.lower():
        return WEIGHTS_MESSAGE
    name = getattr(error, "filename", None) or text
    return (f"A file could not be found: {name}. "
            "Check that the image still exists and select it again.")


def _is_quality_gate_error(error) -> bool:
    text = str(error).lower()
    return "quality gate" in text or "input gate" in text


QUALITY_GATE_MESSAGE = (
    "The photo is not good enough to check: the signature is too small, too faint or too "
    "blurry. Retake it on white paper with a blue pen, hold the phone steady and close, in "
    "even light, so the signature fills the frame. Then select the new photo."
)


PEN_GATE_MESSAGE = (
    "The pen signature could not be used: it is too short or was not recorded properly. "
    "Ask the signer to sign again on the tablet, inside the box, in one go."
)


def _gate_message(error) -> str:
    text = str(error).lower()
    if "temporal" in text and "spatial" not in text:
        return PEN_GATE_MESSAGE
    return QUALITY_GATE_MESSAGE


class EnrollWorker(TrackedThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, image_paths: list, trajectories: list = None):
        super().__init__()
        self.image_paths = image_paths
        self.trajectories = trajectories

    def run(self):
        import src.gui.interfaces as interfaces
        try:
            result = interfaces.enroll_client(self.image_paths, self.trajectories)
            self.finished.emit(result)
        except FileNotFoundError as e:
            self.failed.emit(_missing_file_message(e))
        except interfaces.ImageRejectedError as e:
            self.failed.emit(str(e))
        except interfaces.PenSignatureRejectedError:
            self.failed.emit(f"One of the pen signatures was rejected. {PEN_GATE_MESSAGE} "
                             "Press 'Capture Pen Signatures' to sign all of them again.")
        except ValueError as e:
            if _is_quality_gate_error(e):
                self.failed.emit(f"One or more reference images were rejected. {QUALITY_GATE_MESSAGE}")
            else:
                self.failed.emit(f"Enrollment failed: {e}")
        except Exception as e:
            self.failed.emit(f"Unexpected error: {e}")


class VerifyWorker(TrackedThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, reference_embeddings_b64: list, spatial_origin: str, probe_image_path,
                 reference_temporal_b64: list = None, probe_trajectory=None):
        super().__init__()
        self.reference_embeddings_b64 = reference_embeddings_b64
        self.spatial_origin = spatial_origin
        self.probe_image_path = probe_image_path
        self.reference_temporal_b64 = reference_temporal_b64
        self.probe_trajectory = probe_trajectory

    def run(self):
        import src.gui.interfaces as interfaces
        try:
            result = interfaces.verify_client(
                self.reference_embeddings_b64, self.spatial_origin, self.probe_image_path,
                reference_temporal_b64=self.reference_temporal_b64,
                probe_trajectory=self.probe_trajectory,
            )
            self.finished.emit(result)
        except FileNotFoundError as e:
            self.failed.emit(_missing_file_message(e))
        except ValueError as e:
            if _is_quality_gate_error(e):
                self.failed.emit(_gate_message(e))
            else:
                self.failed.emit(f"Verification failed: {e}")
        except Exception as e:
            self.failed.emit(f"Unexpected error: {e}")

class LoginWorker(TrackedThread):
    """Open the vault and unwrap the user's key slot (scrypt)."""
    finished = pyqtSignal(bool, object, str)
    failed = pyqtSignal(str)

    def __init__(self, company: str, username: str, password: str):
        super().__init__()
        self.company, self.username, self.password = company, username, password

    def run(self):
        import src.gui.storage as storage
        try:
            storage.open_company(self.company)
            self.finished.emit(*storage.authenticate_employee(self.username, self.password))
        except storage.StorageError as e:
            self.failed.emit(str(e))
        finally:
            self.password = None


class WarmUpWorker(TrackedThread):
    """Load both models once at startup."""
    done = pyqtSignal()

    def run(self):
        import src.gui.interfaces as interfaces
        try:
            interfaces.warm_up()
        except Exception:
            pass  # missing weights surface on the real enroll/verify call
        self.done.emit()


class ConfirmedSigningWorker(TrackedThread):
    """Embed (image_path, trajectory) pairs from a confirmed verification visit."""
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, signings: list):
        super().__init__()
        self.signings = signings

    def run(self):
        import src.gui.interfaces as interfaces
        try:
            self.finished.emit([interfaces.embed_confirmed_signing(image, trajectory)
                                for image, trajectory in self.signings])
        except Exception as e:
            self.failed.emit(f"Could not add the signing: {e}")