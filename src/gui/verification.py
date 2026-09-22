"""
Verification Panel — Signature Verification System
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QVBoxLayout, QFileDialog
)
from src.gui.workers import VerifyWorker

import src.gui.storage as storage
import src.gui.interfaces as interfaces


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
        self.resultDisplay.setWordWrap(True)
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

        if not client_id:
            self._show_idle_message("Please enter a client ID")
            return

        client = storage.get_client(client_id, employee_username=self.current_employee_username)
        if client is None:
            self._show_idle_message(
                f"No enrolled record found for '{client_id}' under your account. Please enroll them first."
            )
            return

        embeddings_b64 = client.get("spatial_embeddings", [])
        if not embeddings_b64:
            self._show_idle_message(
                f"'{client_id}' has no stored embeddings — re-save their enrollment."
            )
            return

        if not self.selected_image_path:
            self._show_idle_message("Please select a probe image")
            return

        self.verifyButton.setEnabled(False)
        self.verifyButton.setText("Verifying…")
        self._show_idle_message("Processing…")

        self._pending_client_id = client_id
        self._verify_worker = VerifyWorker(embeddings_b64, self.selected_image_path)
        self._verify_worker.finished.connect(self._on_verify_success)
        self._verify_worker.failed.connect(self._on_verify_failure)
        self._verify_worker.start()

    def _on_verify_success(self, result: dict):
        client_id = self._pending_client_id
        decision = result["decision"]
        confidence = round(result["confidence"], 1)

        self.resultDisplay.setText(f"[{decision}, {confidence}%]")
        role = "result-genuine" if decision == "Genuine" else "result-forged"
        self.resultDisplay.setProperty("role", role)
        self._refresh_result_style()

        if result.get("degraded"):
            reasons = "; ".join(result.get("reasons", [])) or "reduced evidence"
            self.degradedModeIndicator.setText(f"⚠ Degraded: {reasons}")

        storage.append_verification_log(
            client_id=client_id,
            decision=decision,
            confidence=confidence,
            modality_state="Spatial Only (Fallback)",
            employee_username=self.current_employee_username or "unknown"
        )

        self.verifyButton.setEnabled(True)
        self.verifyButton.setText("Verify")

    def _on_verify_failure(self, error_message: str):
        self._show_idle_message(error_message)
        self.verifyButton.setEnabled(True)
        self.verifyButton.setText("Verify")
        
    def _show_idle_message(self, text):
        self.resultDisplay.setText(text)
        self.resultDisplay.setProperty("role", "result-idle")
        self._refresh_result_style()

    def _refresh_result_style(self):
        self.resultDisplay.style().unpolish(self.resultDisplay)
        self.resultDisplay.style().polish(self.resultDisplay)