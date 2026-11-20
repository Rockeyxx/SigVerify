"""
Enrollment Panel — Signature Verification System

Any employee can load a client and add a set it lacks. Replacing a set the client
already has is a re-enrolment, for the admin only (the controls are disabled for others).
"""

import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QListWidget, QMessageBox, QDialog,
    QDialogButtonBox
)
from src.gui.workers import EnrollWorker
from src.gui.image_cleanup import crop_to_ink, SignatureNotFoundError

import src.gui.storage as storage
import src.gui.interfaces as interfaces
from config.settings import NUM_ENROLLMENT_REFERENCES

# One source of truth for K (config/settings.py), not a literal 5 here.
REQUIRED_IMAGE_REFERENCES = NUM_ENROLLMENT_REFERENCES
REQUIRED_DYNAMIC_REFERENCES = NUM_ENROLLMENT_REFERENCES


class ImagePreviewDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference Preview")
        layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            label.setPixmap(pixmap.scaledToWidth(400))
        else:
            label.setText("Unable to load image")
        layout.addWidget(label)
        self.setLayout(layout)


class CropConfirmDialog(QDialog):
    """
    Shows the cropped/cleaned version of a signature photo and asks the
    employee to confirm it before it's added to the reference list.

    This is the human check the automatic cleanup can't replace: a bad
    crop (e.g. a blank strip from uneven lighting) is obvious to a
    person at a glance, even though it passes the automated quality
    gate (PR review point 3).
    """
    def __init__(self, cropped_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Reference")

        layout = QVBoxLayout()

        instructions = QLabel("Does this look like a clean signature? Confirm or retake.")
        layout.addWidget(instructions)

        preview = QLabel()
        pixmap = QPixmap(cropped_path)
        if not pixmap.isNull():
            preview.setPixmap(pixmap.scaledToWidth(350))
        else:
            preview.setText("Unable to preview this image")
        layout.addWidget(preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        buttons.button(QDialogButtonBox.Yes).setText("Looks Good — Add It")
        buttons.button(QDialogButtonBox.No).setText("Discard — Try Another")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)


class EnrollmentPanel(QWidget):
    client_list_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_employee_username = None
        self.is_admin = False
        self.loaded_client = None     # storage record of the loaded client, None for a new one
        self.image_references = []
        # Stored pen embeddings (base64) of a loaded client, shown in the count label only.
        self.dynamic_references = []
        # What collect_signatures() said about the tablet of the pen signings in pen_signatures.
        self._capture_device = None
        # Raw [L, 4] trajectories captured in this session. Held in memory only until the
        # enrolment is saved, never written anywhere (ISO/IEC 24745).
        self.pen_signatures = []
        self._selected_new_image_path = None
        # What run_enrollment() handed the worker; the save uses these, not the live fields.
        self._pending_client_id = None
        self._pending_new_pen = False
        self._pending_capture_device = None

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("Enrollment")
        title.setProperty("role", "title")
        layout.addWidget(title)

        section1 = QLabel("CLIENT")
        section1.setProperty("role", "section")
        layout.addWidget(section1)

        client_row = QHBoxLayout()
        self.clientIdInput = QLineEdit()
        self.clientIdInput.setPlaceholderText("Client name / ID")
        client_row.addWidget(self.clientIdInput)
        self.clientIdInput.textChanged.connect(self._client_id_edited)

        self.loadClientButton = QPushButton("Load / New")
        self.loadClientButton.clicked.connect(self.load_or_start_new_client)
        client_row.addWidget(self.loadClientButton)
        layout.addLayout(client_row)

        self.clientStatusLabel = QLabel("Enter a client ID and click 'Load / New' to begin")
        self.clientStatusLabel.setProperty("role", "muted")
        self.clientStatusLabel.setWordWrap(True)   # long messages must not widen the window
        layout.addWidget(self.clientStatusLabel)

        section2 = QLabel(f"IMAGE REFERENCES ({REQUIRED_IMAGE_REFERENCES} photos, "
                          "or none for a tablet-only client)")
        section2.setProperty("role", "section")
        layout.addWidget(section2)

        self.selectImageButton = QPushButton("Select Reference Image")
        self.selectImageButton.clicked.connect(self.select_reference_image)
        layout.addWidget(self.selectImageButton)

        self.imageFileLabel = QLabel("No file selected")
        self.imageFileLabel.setProperty("role", "muted")
        layout.addWidget(self.imageFileLabel)

        self.addImageButton = QPushButton("Add Image Reference")
        self.addImageButton.clicked.connect(self.add_image_reference)
        layout.addWidget(self.addImageButton)

        self.imageList = QListWidget()
        self.imageList.setMaximumHeight(90)
        layout.addWidget(self.imageList)

        image_actions_row = QHBoxLayout()
        self.viewImageButton = QPushButton("View")
        self.viewImageButton.clicked.connect(self.view_selected_image)
        self.removeImageButton = QPushButton("Remove")
        self.removeImageButton.clicked.connect(self.remove_selected_image)
        image_actions_row.addWidget(self.viewImageButton)
        image_actions_row.addWidget(self.removeImageButton)
        layout.addLayout(image_actions_row)

        self.imageCountLabel = QLabel(f"0/{REQUIRED_IMAGE_REFERENCES} image references")
        self.imageCountLabel.setProperty("role", "muted")
        layout.addWidget(self.imageCountLabel)

        section3 = QLabel(f"PEN REFERENCES ({REQUIRED_DYNAMIC_REFERENCES} signings on the tablet)")
        section3.setProperty("role", "section")
        layout.addWidget(section3)

        self.captureDynamicButton = QPushButton("Capture Pen Signatures")
        self.captureDynamicButton.clicked.connect(self.capture_pen_signatures)
        layout.addWidget(self.captureDynamicButton)

        self.dynamicCountLabel = QLabel("")
        self.dynamicCountLabel.setProperty("role", "muted")
        layout.addWidget(self.dynamicCountLabel)

        section4 = QLabel("ACTIONS")
        section4.setProperty("role", "section")
        layout.addWidget(section4)

        self.enrollButton = QPushButton("Save Enrollment")
        self.enrollButton.setProperty("role", "primary")
        self.enrollButton.clicked.connect(self.run_enrollment)
        layout.addWidget(self.enrollButton)

        self.deleteClientButton = QPushButton("Delete This Client")
        self.deleteClientButton.setProperty("role", "danger")
        self.deleteClientButton.clicked.connect(self.delete_client)
        layout.addWidget(self.deleteClientButton)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)
        layout.addWidget(self.statusLabel)

        layout.addStretch()
        self.setLayout(layout)

        self._update_counts()
        self._apply_permissions()

    def set_current_employee(self, username: str, is_admin: bool = False):
        self.current_employee_username = username
        self.is_admin = is_admin
        self._apply_permissions()

    def _client_id_edited(self):
        """The loaded record and this session's signings belong to the old ID: forget them."""
        self.loaded_client = None
        self.dynamic_references = []
        self.pen_signatures = []
        self._capture_device = None
        self.image_references = []
        self._refresh_image_list()
        self._update_counts()
        self._apply_permissions()

    def _apply_permissions(self):
        """Non-admin: a set the loaded client already has cannot be replaced; Delete is hidden."""
        record = self.loaded_client
        locked_paper = bool(record and not self.is_admin and record["spatial_sets"].get("scan"))
        locked_pen = bool(record and not self.is_admin and record["temporal_set"])
        for button in (self.selectImageButton, self.addImageButton):
            button.setEnabled(not locked_paper)
        self.captureDynamicButton.setEnabled(not locked_pen)
        self.deleteClientButton.setVisible(self.is_admin)

    def prefill_client(self, client_id: str):
        self.clientIdInput.setText(client_id)
        self.load_or_start_new_client()

    def load_or_start_new_client(self):
        client_id = self.clientIdInput.text().strip()
        if not client_id:
            self.clientStatusLabel.setText("Please enter a client ID first")
            return
        self._load_client(client_id)

    def _load_client(self, client_id: str):
        self.pen_signatures = []   # never carry one client's signatures over to another
        self._capture_device = None
        self.image_references = []   # paths are not stored: only photos picked this session
        record = storage.get_client(client_id)   # company-wide
        self.loaded_client = record
        if record:
            self.dynamic_references = list(record.get("dynamic_references", []))
            n_paper = len(record["spatial_sets"].get("scan", {}).get("embeddings", []))
            n_pen = len((record["temporal_set"] or {}).get("embeddings", []))
            text = (f"Loaded existing client '{client_id}' · Paper: {n_paper} · Pen: {n_pen}. "
                    f"Picking {REQUIRED_IMAGE_REFERENCES} photos again re-enrols the paper set (admin only)")
            if not self.is_admin and (n_paper or n_pen):
                text += ". Only the admin can replace these"
            self.clientStatusLabel.setText(text)
        else:
            self.dynamic_references = []
            self.clientStatusLabel.setText(
                f"New client '{client_id}' — add {REQUIRED_IMAGE_REFERENCES} image references, "
                f"capture {REQUIRED_DYNAMIC_REFERENCES} pen signatures, or both"
            )
        self._apply_permissions()

        self._refresh_image_list()
        self._update_counts()
        self.statusLabel.setText("")

    def select_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Signature", "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self._selected_new_image_path = file_path
            self.imageFileLabel.setText(file_path.split("/")[-1])
        else:
            self.imageFileLabel.setText("No file selected")

    def add_image_reference(self):
        if not self._selected_new_image_path:
            self._set_status_error("Select an image before adding it")
            return

        if len(self.image_references) >= REQUIRED_IMAGE_REFERENCES:
            self._set_status_error(f"Already have {REQUIRED_IMAGE_REFERENCES} references — remove one first")
            return

        # Run the same cleanup pipeline the model will see (crop_to_ink),
        # but here purely so the employee can confirm the result before
        # committing to it. This runs again later, inside interfaces.py's
        # enroll_client(), as part of the real processing pipeline —
        # that duplication is intentional, not a bug: it keeps preview
        # and processing independent and simple.
        try:
            cropped_preview = crop_to_ink(self._selected_new_image_path)
        except SignatureNotFoundError as e:
            self._set_status_error(str(e))
            return
        except FileNotFoundError:
            self._set_status_error(
                "That image file could not be found. Select it again.")
            return

        # The preview is a temporary file: delete it once the employee has answered,
        # whatever the answer (or an error) is.
        try:
            dialog = CropConfirmDialog(cropped_preview, parent=self)
            accepted = dialog.exec_() == QDialog.Accepted
        finally:
            try:
                os.remove(cropped_preview)
            except OSError:
                pass
        if not accepted:
            self.statusLabel.setText("Discarded — select another image")
            return

        self.image_references.append(self._selected_new_image_path)
        self._selected_new_image_path = None
        self.imageFileLabel.setText("No file selected")
        self.statusLabel.setText("")
        self._refresh_image_list()
        self._update_counts()

    def view_selected_image(self):
        row = self.imageList.currentRow()
        if row < 0:
            self.statusLabel.setText("Select an image in the list first")
            return
        dialog = ImagePreviewDialog(self.image_references[row], parent=self)
        dialog.exec_()

    def remove_selected_image(self):
        row = self.imageList.currentRow()
        if row < 0:
            self.statusLabel.setText("Select an image in the list first")
            return
        del self.image_references[row]
        self._refresh_image_list()
        self._update_counts()

    def _refresh_image_list(self):
        self.imageList.clear()
        for path in self.image_references:
            self.imageList.addItem(path.split("/")[-1])

    def _update_counts(self):
        self.imageCountLabel.setText(f"{len(self.image_references)}/{REQUIRED_IMAGE_REFERENCES} image references")
        if self.pen_signatures:
            text = f"{len(self.pen_signatures)}/{REQUIRED_DYNAMIC_REFERENCES} pen signatures"
        elif self.dynamic_references:
            text = (f"{len(self.dynamic_references)} stored pen references "
                    "(kept unless you capture new ones)")
        else:
            text = f"0/{REQUIRED_DYNAMIC_REFERENCES} pen signatures"
        self.dynamicCountLabel.setText(text)

    def capture_pen_signatures(self):
        """Open the tablet window for the K enrolment signings (issue #23, item 2)."""
        from src.capture.collect import collect_signatures
        captured = collect_signatures(count=REQUIRED_DYNAMIC_REFERENCES, parent=self)
        if captured is None:      # window closed: keep whatever was there
            return
        self.pen_signatures = list(captured["signatures"])
        self._capture_device = {k: v for k, v in captured.items() if k != "signatures"}
        self.statusLabel.setText("")
        self._update_counts()

    def run_enrollment(self):
        client_id = self.clientIdInput.text().strip()

        if not client_id:
            self._set_status_error("Please enter a client ID")
            return

        n_images, n_pen = len(self.image_references), len(self.pen_signatures)
        if n_images not in (0, REQUIRED_IMAGE_REFERENCES):
            self._set_status_error(
                f"Exactly {REQUIRED_IMAGE_REFERENCES} image references required "
                f"(currently {n_images}), or none for a tablet-only client"
            )
            return
        if n_images == 0 and n_pen == 0:
            self._set_status_error(
                f"Add {REQUIRED_IMAGE_REFERENCES} image references, capture "
                f"{REQUIRED_DYNAMIC_REFERENCES} pen signatures, or both"
            )
            return

        record = storage.get_client(client_id)   # the ID being saved, not the last one loaded
        replaced = [name for name, new, have in (
            ("paper", n_images, record and record["spatial_sets"].get("scan")),
            ("pen", n_pen, record and record["temporal_set"])) if new and have]
        if replaced and QMessageBox.question(
                self, "Confirm Re-enrolment",
                f"This replaces the client's {' and '.join(replaced)} references. Continue?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        # نعطل الزر ونعرض مؤشر تحميل عشان المستخدم يعرف فيه شي يشتغل
        self.enrollButton.setEnabled(False)
        self.enrollButton.setText("Processing…")
        self.statusLabel.setText("")

        self._pending_client_id = client_id
        self._pending_new_pen = bool(self.pen_signatures)
        self._pending_capture_device = self._capture_device
        # Copies: an edit or a capture while the worker runs must not change its input.
        self._enroll_worker = EnrollWorker(list(self.image_references), list(self.pen_signatures))
        self._enroll_worker.finished.connect(self._on_enroll_success)
        self._enroll_worker.failed.connect(self._on_enroll_failure)
        self._enroll_worker.start()

    def _on_enroll_success(self, result: dict):
        """
        result = interfaces.enroll_client():
        {"embeddings": [...], "dynamic_embeddings": [...], "spatial_origin": "scan"}
        """
        client_id = self._pending_client_id
        embeddings_b64 = result["embeddings"]
        spatial_origin = result["spatial_origin"]
        # None keeps the stored pen set (and its added signings) untouched.
        new_pen = self._pending_new_pen
        pen_b64 = result["dynamic_embeddings"] if new_pen else None

        success, message = storage.save_client_enrollment(
            client_id, pen_b64,
            self.current_employee_username or "unknown",
            spatial_embeddings=embeddings_b64,
            spatial_origin=spatial_origin,
            capture_device=self._pending_capture_device if new_pen else None,
        )

        self.enrollButton.setEnabled(True)
        self.enrollButton.setText("Save Enrollment")

        if not success:
            self._set_status_error(message)
            return

        # Show the saved client as it is now, unless another ID was typed or loaded meanwhile.
        # Loading drops the raw signatures, which are embedded now.
        if self.clientIdInput.text().strip() == client_id:
            self._load_client(client_id)
        self._set_status_success(
            f"Saved enrollment for '{client_id}' ({len(embeddings_b64)} image, "
            f"{len(pen_b64 or [])} pen embeddings)")
        self.client_list_changed.emit()

    def _on_enroll_failure(self, error_message: str):
        self._set_status_error(error_message)
        self.enrollButton.setEnabled(True)
        self.enrollButton.setText("Save Enrollment")

    def delete_client(self):
        client_id = self.clientIdInput.text().strip()
        if not client_id or not storage.client_exists(client_id):
            self._set_status_error("No existing client loaded to delete")
            return

        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete all enrollment data for '{client_id}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            ok, message = storage.delete_client(client_id, self.current_employee_username)
            if not ok:
                self._set_status_error(message)
                return
            self.image_references = []
            self.dynamic_references = []
            self.pen_signatures = []
            self.loaded_client = None
            self._apply_permissions()
            self._refresh_image_list()
            self._update_counts()
            self._set_status_success(message)
            self.client_list_changed.emit()

    def _set_status_error(self, text):
        self.statusLabel.setProperty("role", "status-error")
        self.statusLabel.setText(f"⚠ {text}")
        self._refresh_status_style()

    def _set_status_success(self, text):
        self.statusLabel.setProperty("role", "status-success")
        self.statusLabel.setText(f"✔ {text}")
        self._refresh_status_style()

    def _refresh_status_style(self):
        self.statusLabel.style().unpolish(self.statusLabel)
        self.statusLabel.style().polish(self.statusLabel)