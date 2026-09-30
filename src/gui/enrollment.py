"""
Enrollment Panel — Signature Verification System

Clients are scoped per employee: an employee can only load/edit
clients they personally enrolled.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QListWidget, QMessageBox, QDialog
)
from src.gui.workers import EnrollWorker

import src.gui.storage as storage
import src.gui.interfaces as interfaces

REQUIRED_IMAGE_REFERENCES = 5
REQUIRED_DYNAMIC_REFERENCES = 5


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


class EnrollmentPanel(QWidget):
    client_list_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_employee_username = None
        self.image_references = []
        self.dynamic_references = []
        self._selected_new_image_path = None

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

        self.loadClientButton = QPushButton("Load / New")
        self.loadClientButton.clicked.connect(self.load_or_start_new_client)
        client_row.addWidget(self.loadClientButton)
        layout.addLayout(client_row)

        self.clientStatusLabel = QLabel("Enter a client ID and click 'Load / New' to begin")
        self.clientStatusLabel.setProperty("role", "muted")
        layout.addWidget(self.clientStatusLabel)

        section2 = QLabel(f"IMAGE REFERENCES (exactly {REQUIRED_IMAGE_REFERENCES} required)")
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

        self.imageCountLabel = QLabel("0/5 image references")
        self.imageCountLabel.setProperty("role", "muted")
        layout.addWidget(self.imageCountLabel)

        section3 = QLabel(f"DYNAMIC REFERENCES (exactly {REQUIRED_DYNAMIC_REFERENCES} required)")
        section3.setProperty("role", "section")
        layout.addWidget(section3)

        self.captureDynamicButton = QPushButton("Capture Dynamic Signature (device not connected)")
        self.captureDynamicButton.setEnabled(False)
        layout.addWidget(self.captureDynamicButton)

        self.dynamicCountLabel = QLabel("0/5 dynamic references — pending tablet integration")
        self.dynamicCountLabel.setProperty("role", "warning")
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
        layout.addWidget(self.statusLabel)

        layout.addStretch()
        self.setLayout(layout)

        self._update_counts()

    def set_current_employee(self, username: str):
        self.current_employee_username = username

    def prefill_client(self, client_id: str):
        self.clientIdInput.setText(client_id)
        self.load_or_start_new_client()

    def load_or_start_new_client(self):
        client_id = self.clientIdInput.text().strip()
        if not client_id:
            self.clientStatusLabel.setText("Please enter a client ID first")
            return

        any_owner_record = storage.get_client(client_id, employee_username=None)
        own_record = storage.get_client(client_id, employee_username=self.current_employee_username)

        if any_owner_record and not own_record:
            self.clientStatusLabel.setText(
                "⚠ A client with this name already exists under another employee"
            )
            self.image_references = []
            self.dynamic_references = []
            self._refresh_image_list()
            self._update_counts()
            return

        if own_record:
            self.image_references = list(own_record.get("image_references", []))
            self.dynamic_references = list(own_record.get("dynamic_references", []))
            self.clientStatusLabel.setText(
                f"Loaded existing client '{client_id}' — edit references below"
            )
        else:
            self.image_references = []
            self.dynamic_references = []
            self.clientStatusLabel.setText(
                f"New client '{client_id}' — add {REQUIRED_IMAGE_REFERENCES} image references to enroll"
            )

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

    def run_enrollment(self):
        client_id = self.clientIdInput.text().strip()

        if not client_id:
            self._set_status_error("Please enter a client ID")
            return

        if len(self.image_references) != REQUIRED_IMAGE_REFERENCES:
            self._set_status_error(
                f"Exactly {REQUIRED_IMAGE_REFERENCES} image references required "
                f"(currently {len(self.image_references)})"
            )
            return

        # نعطل الزر ونعرض مؤشر تحميل عشان المستخدم يعرف فيه شي يشتغل
        self.enrollButton.setEnabled(False)
        self.enrollButton.setText("Processing…")
        self.statusLabel.setText("")

        self._pending_client_id = client_id
        self._enroll_worker = EnrollWorker(self.image_references)
        self._enroll_worker.finished.connect(self._on_enroll_success)
        self._enroll_worker.failed.connect(self._on_enroll_failure)
        self._enroll_worker.start()

    def _on_enroll_success(self, result: dict):
        """
        result = {"embeddings": [base64 str, ...], "spatial_origin": "scan"}
        — the shape interfaces.enroll_client() returns since main added
        the spatial_origin requirement.
        """
        client_id = self._pending_client_id
        embeddings_b64 = result["embeddings"]
        spatial_origin = result["spatial_origin"]

        storage.save_client_enrollment(
            client_id, self.image_references, self.dynamic_references,
            self.current_employee_username or "unknown",
            spatial_embeddings=embeddings_b64,
            spatial_origin=spatial_origin,
        )

        self._set_status_success(f"Saved enrollment for '{client_id}' ({len(embeddings_b64)} embeddings)")
        self.client_list_changed.emit()

        self.image_references = []
        self._refresh_image_list()
        self._update_counts()

        self.enrollButton.setEnabled(True)
        self.enrollButton.setText("Save Enrollment")

    def _on_enroll_failure(self, error_message: str):
        self._set_status_error(error_message)
        self.enrollButton.setEnabled(True)
        self.enrollButton.setText("Save Enrollment")

    def delete_client(self):
        client_id = self.clientIdInput.text().strip()
        if not client_id or not storage.client_exists(client_id, self.current_employee_username):
            self._set_status_error("No existing client of yours loaded to delete")
            return

        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete all enrollment data for '{client_id}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            storage.delete_client(client_id)
            self.image_references = []
            self.dynamic_references = []
            self._refresh_image_list()
            self._update_counts()
            self._set_status_success(f"Deleted client '{client_id}'")
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