"""
workers.py — Signature Verification System

QThread workers that run heavy interfaces.py calls (model inference,
image cleanup) off the main GUI thread, so the window stays responsive
and doesn't show "Not Responding" during enrollment/verification.
"""

from PyQt5.QtCore import QThread, pyqtSignal


class EnrollWorker(QThread):
    finished = pyqtSignal(dict)      # emits: {"embeddings": [...], "spatial_origin": "scan"}
    failed = pyqtSignal(str)         # emits: error message

    def __init__(self, image_paths: list):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        import src.gui.interfaces as interfaces
        try:
            result = interfaces.enroll_client(self.image_paths)
            self.finished.emit(result)
        except FileNotFoundError:
            self.failed.emit("Model weights not found. See README → Model Weights & GitHub Releases.")
        except ValueError as e:
            self.failed.emit(f"One or more images failed the quality gate: {e}")
        except Exception as e:
            self.failed.emit(f"Unexpected error: {e}")


class VerifyWorker(QThread):
    finished = pyqtSignal(dict)      # emits: result dict
    failed = pyqtSignal(str)         # emits: error message

    def __init__(self, reference_embeddings_b64: list, spatial_origin: str, probe_image_path: str):
        super().__init__()
        self.reference_embeddings_b64 = reference_embeddings_b64
        self.spatial_origin = spatial_origin
        self.probe_image_path = probe_image_path

    def run(self):
        import src.gui.interfaces as interfaces
        try:
            result = interfaces.verify_client(
                self.reference_embeddings_b64, self.spatial_origin, self.probe_image_path
            )
            self.finished.emit(result)
        except FileNotFoundError:
            self.failed.emit("Model weights not found. See README → Model Weights & GitHub Releases.")
        except ValueError as e:
            self.failed.emit(f"Verification failed: {e}")
        except Exception as e:
            self.failed.emit(f"Unexpected error: {e}")