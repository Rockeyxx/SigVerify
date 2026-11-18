"""Verification Panel for the signature verification graphical interface."""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QVBoxLayout, QFileDialog
)
from src.gui.styles import DANGER, PRIMARY_LIGHT, SUCCESS, TEXT_LIGHT
from src.gui.workers import VerifyWorker, ConfirmedSigningWorker

import src.gui.storage as storage
import src.gui.interfaces as interfaces

# Third decision of step-up verification (src.verification.SIGN_AGAIN).
SIGN_AGAIN = "Sign again"


def threshold_for(result: dict):
    """The threshold (0-100, same scale as the score) verify() decided against, or None.

    verify() returns the threshold it applied as result["threshold"] (0-1), including under
    step-up (STEP_UP_ACCEPT / STEP_UP_TWO), so the line on the bar is always the real one.
    "confidence" is a similarity x 100, not a probability: it is shown as a match score
    against this threshold, never as "% sure".
    """
    threshold = result.get("threshold")
    return None if threshold is None else threshold * 100


class ScoreBar(QWidget):
    """A 0-100 bar: the match score filled in, the decision threshold marked with a line."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(22)
        self._score = 0.0
        self._threshold = None
        self._genuine = False

    def set_values(self, score: float, threshold, genuine: bool):
        self._score, self._threshold, self._genuine = score, threshold, genuine
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(PRIMARY_LIGHT))
        painter.drawRoundedRect(QRectF(0, 7, w, 8), 4, 4)
        fill = max(0.0, min(100.0, self._score)) / 100.0 * w
        painter.setBrush(QColor(SUCCESS if self._genuine else DANGER))
        painter.drawRoundedRect(QRectF(0, 7, fill, 8), 4, 4)
        if self._threshold is not None:
            x = max(0.0, min(100.0, self._threshold)) / 100.0 * w
            painter.setPen(QPen(QColor(TEXT_LIGHT), 2))
            painter.drawLine(int(x), 1, int(x), h - 1)


class VerificationPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.current_employee_username = None
        # Step-up state: the first signing is kept while the second is requested.
        self._first_probe_path = None
        self._awaiting_second = False
        # The last accepted visit, kept so the employee can confirm it.
        self._last_accepted_paths = None
        self._last_accepted_client = None

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
        self.clientIdInput.textChanged.connect(self._reset_visit)
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

        self.scoreBar = ScoreBar()
        self.scoreBar.setVisible(False)
        layout.addWidget(self.scoreBar)

        self.scoreCaption = QLabel("")
        self.scoreCaption.setProperty("role", "muted")
        self.scoreCaption.setVisible(False)
        layout.addWidget(self.scoreCaption)

        self.modalityStateLabel = QLabel("")
        self.modalityStateLabel.setProperty("role", "muted")
        layout.addWidget(self.modalityStateLabel)

        self.degradedModeIndicator = QLabel("")
        self.degradedModeIndicator.setProperty("role", "warning")
        layout.addWidget(self.degradedModeIndicator)

        # Shown only after an accepted, non-degraded result (AGENTS.md: references
        # grow only from confirmed Genuine, non-degraded visits).
        self.confirmButton = QPushButton("Confirm identity and add to references")
        self.confirmButton.clicked.connect(self.confirm_identity)
        self.confirmButton.setVisible(False)
        layout.addWidget(self.confirmButton)

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

    def _reset_visit(self):
        """A new visit: forget any pending first signing and any unconfirmed result."""
        self._first_probe_path = None
        self._awaiting_second = False
        self._last_accepted_paths = None
        self._last_accepted_client = None
        self.confirmButton.setVisible(False)
        self.scoreBar.setVisible(False)
        self.scoreCaption.setVisible(False)
        self.selectProbeButton.setText("Select Probe Image")

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

        # Enrolled references plus confirmed signings added on earlier visits.
        embeddings_b64, spatial_origin = storage.get_reference_embeddings(
            client_id, employee_username=self.current_employee_username)
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
        self.confirmButton.setVisible(False)

        self._pending_client_id = client_id
        if self._awaiting_second:
            # Second round: both signings, first signing first. Never a third.
            probes = [self._first_probe_path, self.selected_image_path]
        else:
            probes = self.selected_image_path
        self._pending_probes = probes
        self._verify_worker = VerifyWorker(embeddings_b64, spatial_origin, probes)
        self._verify_worker.finished.connect(self._on_verify_success)
        self._verify_worker.failed.connect(self._on_verify_failure)
        self._verify_worker.start()

    def _on_verify_success(self, result: dict):
        client_id = self._pending_client_id
        decision = result["decision"]
        confidence = round(result["confidence"], 1)

        if decision == SIGN_AGAIN and not self._awaiting_second:
            # Not a verdict. Neutral request only: no score, no hint of how close it was.
            self._first_probe_path = self.selected_image_path
            self._awaiting_second = True
            self.selected_image_path = None
            self.probeFileLabel.setText("No file selected")
            self.selectProbeButton.setText("Select Second Signing")
            self._show_idle_message("Please sign once more and select the new signing.")
            self.verifyButton.setEnabled(True)
            self.verifyButton.setText("Verify")
            return
        if decision == SIGN_AGAIN:
            # Should never happen (the second call returns only Genuine or Forged).
            # Fail closed rather than asking a third time.
            decision = "Forged"

        accepted_paths = (list(self._pending_probes) if isinstance(self._pending_probes, list)
                          else [self._pending_probes])
        self._first_probe_path = None
        self._awaiting_second = False
        self.selectProbeButton.setText("Select Probe Image")

        threshold = threshold_for(result)
        self.resultDisplay.setText(decision)
        self.scoreBar.set_values(confidence, threshold, decision == "Genuine")
        self.scoreBar.setVisible(True)
        against = f" (line = threshold {threshold:.1f})" if threshold is not None else ""
        self.scoreCaption.setText(
            f"Match score {confidence:.1f}{against}. "
            "The score is a similarity, not a probability.")
        self.scoreCaption.setVisible(True)
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

        if decision == "Genuine" and not result.get("degraded"):
            self._last_accepted_paths = accepted_paths
            self._last_accepted_client = client_id
            self.confirmButton.setVisible(True)

        self.verifyButton.setEnabled(True)
        self.verifyButton.setText("Verify")

    def confirm_identity(self):
        """The employee confirms the person's identity: add this visit to the references."""
        if not self._last_accepted_paths:
            return
        self.confirmButton.setEnabled(False)
        self._confirm_worker = ConfirmedSigningWorker(self._last_accepted_paths)
        self._confirm_worker.finished.connect(self._on_confirmed)
        self._confirm_worker.failed.connect(self._on_confirm_failed)
        self._confirm_worker.start()

    def _on_confirmed(self, embeddings_b64: list):
        for emb in embeddings_b64:
            storage.add_confirmed_reference(
                self._last_accepted_client, emb,
                self.current_employee_username or "unknown")
        self._last_accepted_paths = None
        self.confirmButton.setEnabled(True)
        self.confirmButton.setVisible(False)
        self.degradedModeIndicator.setText("")
        self.modalityStateLabel.setText("Signing added to the client's references.")

    def _on_confirm_failed(self, message: str):
        self.confirmButton.setEnabled(True)
        self._show_idle_message(message)

    def _on_verify_failure(self, error_message: str):
        self._show_idle_message(error_message)
        self.verifyButton.setEnabled(True)
        self.verifyButton.setText("Verify")

    def _show_idle_message(self, text):
        self.scoreBar.setVisible(False)
        self.scoreCaption.setVisible(False)
        self.resultDisplay.setText(text)
        self.resultDisplay.setProperty("role", "result-idle")
        self._refresh_result_style()

    def _refresh_result_style(self):
        self.resultDisplay.style().unpolish(self.resultDisplay)
        self.resultDisplay.style().polish(self.resultDisplay)