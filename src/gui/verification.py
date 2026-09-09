"""
Verification Panel — Signature Verification System
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QVBoxLayout, QFileDialog
)

import storage
import interfaces


class SignatureService:
    """
    نقطة مركزية وحدة تجمع منطق التسجيل والتحقق. تعتمد على interfaces.py
    (للموديل، لما يجهز) و storage.py (للتخزين) — هما نقطتا الاستبدال
    الوحيدتان لما يجهز الفريق شغلهم الحقيقي.
    """

    @staticmethod
    def enroll(client_id: str, image_paths: list) -> tuple:
        if not client_id:
            return False, "Please enter a client ID"
        if not image_paths:
            return False, "Please add at least one reference image"

        for image_path in image_paths:
            image_tensor = interfaces.preprocess_image(image_path)
            embedding = interfaces.get_spatial_embedding(image_tensor)
            interfaces.store_embedding(client_id, embedding, modality="spatial")

        return True, "References processed"

    @staticmethod
    def verify(client_id: str, probe_image_path: str) -> tuple:
        """Returns: (success, message, decision, confidence)"""
        if not client_id:
            return False, "Please enter a client ID", "", 0.0

        if not storage.client_exists(client_id):
            return False, f"No enrolled record found for '{client_id}'. Please enroll them first.", "", 0.0

        if not probe_image_path:
            return False, "Please select a probe image", "", 0.0

        image_tensor = interfaces.preprocess_image(probe_image_path)
        probe_embedding = interfaces.get_spatial_embedding(image_tensor)

        references = interfaces.get_references(client_id, modality="spatial")
        if not references:
            return False, f"No reference embeddings found for '{client_id}'", "", 0.0

        distances = [interfaces.compute_distance(probe_embedding, ref) for ref in references]
        s_spatial = min(distances)
        interfaces.fuse_scores(s_spatial, s_temporal=0.0, alpha=1.0)

        decision, confidence = interfaces.mock_verify_decision()
        return True, "", decision, confidence


class VerificationPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.current_employee_username = None

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("Verification")
        title.setProperty("role", "title")
        layout.addWidget(title)

        section1 = QLabel("CLIENT")
        section1.setProperty("role", "section")
        layout.addWidget(section1)

        self.clientIdInput = QLineEdit()
        self.clientIdInput.setPlaceholderText("Enter the client's name / ID")
        layout.addWidget(self.clientIdInput)

        section2 = QLabel("MODALITY")
        section2.setProperty("role", "section")
        layout.addWidget(section2)

        self.modalitySelector = QComboBox()
        self.modalitySelector.addItem("Image only")
        self.modalitySelector.addItem("Dynamic only (device not connected)")
        self.modalitySelector.addItem("Both (device not connected)")
        self.modalitySelector.model().item(1).setEnabled(False)
        self.modalitySelector.model().item(2).setEnabled(False)
        layout.addWidget(self.modalitySelector)

        section3 = QLabel("PROBE SIGNATURE")
        section3.setProperty("role", "section")
        layout.addWidget(section3)

        self.selectProbeButton = QPushButton("Select Probe Image")
        self.selectProbeButton.clicked.connect(self.select_probe_image)
        layout.addWidget(self.selectProbeButton)

        self.probeFileLabel = QLabel("No file selected")
        self.probeFileLabel.setProperty("role", "muted")
        layout.addWidget(self.probeFileLabel)

        self.verifyButton = QPushButton("Verify")
        self.verifyButton.setProperty("role", "primary")
        self.verifyButton.clicked.connect(self.run_verification)
        layout.addWidget(self.verifyButton)

        section4 = QLabel("RESULT")
        section4.setProperty("role", "section")
        layout.addWidget(section4)

        self.resultDisplay = QLabel("Awaiting verification...")
        self.resultDisplay.setProperty("role", "result-idle")
        layout.addWidget(self.resultDisplay)

        self.modalityStateLabel = QLabel("")
        self.modalityStateLabel.setProperty("role", "muted")
        layout.addWidget(self.modalityStateLabel)

        self.degradedModeIndicator = QLabel("")
        self.degradedModeIndicator.setProperty("role", "warning")
        layout.addWidget(self.degradedModeIndicator)

        layout.addStretch()
        self.setLayout(layout)

        self.update_modality_state()

    def set_current_employee(self, username: str):
        self.current_employee_username = username

    def prefill_client(self, client_id: str):
        self.clientIdInput.setText(client_id)

    def update_modality_state(self):
        self.modalityStateLabel.setText("Modality State: Spatial Only (Fallback)")
        self.degradedModeIndicator.setText(
            "⚠ Degraded Mode: no dynamic input device connected"
        )

    def select_probe_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Signature Image", "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.selected_image_path = file_path
            self.probeFileLabel.setText(file_path.split("/")[-1])
        else:
            self.probeFileLabel.setText("No file selected")

    def run_verification(self):
        client_id = self.clientIdInput.text().strip()

        success, message, decision, confidence = SignatureService.verify(
            client_id, self.selected_image_path
        )

        if not success:
            self._show_idle_message(message)
            return

        self.resultDisplay.setText(f"[{decision}, {confidence}%]")
        role = "result-genuine" if decision == "Genuine" else "result-forged"
        self.resultDisplay.setProperty("role", role)
        self._refresh_result_style()

        storage.append_verification_log(
            client_id=client_id,
            decision=decision,
            confidence=confidence,
            modality_state="Spatial Only (Fallback)",
            employee_username=self.current_employee_username or "unknown"
        )

    def _show_idle_message(self, text):
        self.resultDisplay.setText(text)
        self.resultDisplay.setProperty("role", "result-idle")
        self._refresh_result_style()

    def _refresh_result_style(self):
        self.resultDisplay.style().unpolish(self.resultDisplay)
        self.resultDisplay.style().polish(self.resultDisplay)