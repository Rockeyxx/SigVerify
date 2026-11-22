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

# Modality selector entries: (label, uses paper image, uses pen).
MODALITIES = [
    ("Paper + pen (fused)", True, True),
    ("Paper image only", True, False),
    ("Pen only (tablet)", False, True),
]


def threshold_for(result: dict):
    """The threshold (0-100, same scale as the score) verify() decided against, or None.

    verify() returns the threshold it applied as result["threshold"] (0-1), including under
    step-up (STEP_UP_ACCEPT / STEP_UP_TWO), so the line on the bar is always the real one.
    "confidence" is a similarity x 100, not a probability: it is shown as a match score
    against this threshold, never as "% sure".
    """
    threshold = result.get("threshold")
    return None if threshold is None else threshold * 100


def visit_signings(images, trajectories) -> list:
    """Pair a visit's signings as (image or None, trajectory or None), first signing first.

    images / trajectories: one probe, a list of probes, or None when that kind was not used.
    """
    def as_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, (list, tuple)) else [x]
    images, trajectories = as_list(images), as_list(trajectories)
    count = max(len(images), len(trajectories))
    return [(images[i] if i < len(images) else None,
             trajectories[i] if i < len(trajectories) else None) for i in range(count)]


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
        # The pen signature of this visit: one [L, 4] array, in memory only, never saved.
        self.selected_trajectory = None
        self.current_employee_username = None
        # Step-up state: the first signing (of each kind used) is kept while the second
        # is requested.
        self._first_probe_path = None
        self._first_trajectory = None
        self._awaiting_second = False
        # The last accepted visit, kept so the employee can confirm it:
        # a list of (image path or None, trajectory or None), one per signing.
        self._last_accepted_signings = None
        self._last_accepted_client = None
        # The tablet that made the current pen signing, and the origin of the paper set
        # used for this visit (a confirmed signing goes into that set).
        self._pen_capture_device = None
        self._pending_origin = "scan"
        self._last_accepted_origin = "scan"

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("Verification")
        title.setProperty("role", "title")
        layout.addWidget(title)

        section1 = QLabel("SIGNER")
        section1.setProperty("role", "section")
        layout.addWidget(section1)

        self.clientIdInput = QLineEdit()
        self.clientIdInput.setPlaceholderText("Enter the signer's name / ID")
        self.clientIdInput.textChanged.connect(self._reset_visit)
        layout.addWidget(self.clientIdInput)

        section2 = QLabel("MODALITY")
        section2.setProperty("role", "section")
        layout.addWidget(section2)

        self.modalitySelector = QComboBox()
        for label, _, _ in MODALITIES:
            self.modalitySelector.addItem(label)
        self.modalitySelector.currentIndexChanged.connect(self._on_modality_changed)
        layout.addWidget(self.modalitySelector)

        section3 = QLabel("PROBE SIGNATURE")
        section3.setProperty("role", "section")
        layout.addWidget(section3)

        self.selectProbeButton = QPushButton("Select Probe Image")
        self.selectProbeButton.clicked.connect(self.select_probe_image)
        layout.addWidget(self.selectProbeButton)

        self.probeFileLabel = QLabel("No file selected")
        self.probeFileLabel.setProperty("role", "muted")
        self.probeFileLabel.setWordWrap(True)
        layout.addWidget(self.probeFileLabel)

        self.signPenButton = QPushButton("Sign on Tablet")
        self.signPenButton.clicked.connect(self.capture_pen_probe)
        layout.addWidget(self.signPenButton)

        self.penStatusLabel = QLabel("No pen signature yet")
        self.penStatusLabel.setProperty("role", "muted")
        self.penStatusLabel.setWordWrap(True)
        layout.addWidget(self.penStatusLabel)

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
        self.scoreCaption.setWordWrap(True)
        self.scoreCaption.setVisible(False)
        layout.addWidget(self.scoreCaption)

        self.modalityStateLabel = QLabel("")
        self.modalityStateLabel.setProperty("role", "muted")
        self.modalityStateLabel.setWordWrap(True)
        layout.addWidget(self.modalityStateLabel)

        self.degradedModeIndicator = QLabel("")
        self.degradedModeIndicator.setProperty("role", "warning")
        self.degradedModeIndicator.setWordWrap(True)
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

    # --- modality -----------------------------------------------------------
    def _uses(self):
        """(uses paper image, uses pen) for the selected modality."""
        _, image, pen = MODALITIES[max(0, self.modalitySelector.currentIndex())]
        return image, pen

    def _on_modality_changed(self, _index):
        self._reset_visit()
        self.update_modality_state()

    def update_modality_state(self):
        """Show the inputs the selected modality needs, and what will run."""
        image, pen = self._uses()
        self.selectProbeButton.setVisible(image)
        self.probeFileLabel.setVisible(image)
        self.signPenButton.setVisible(pen)
        self.penStatusLabel.setVisible(pen)
        label = MODALITIES[max(0, self.modalitySelector.currentIndex())][0]
        self.modalityStateLabel.setText(f"Modality: {label}")
        self.degradedModeIndicator.setText(
            "" if (image and pen) else
            "⚠ One modality only: the result is reported as degraded and cannot be "
            "added to the signer's references.")

    def _reset_visit(self):
        """A new visit: forget any pending first signing and any unconfirmed result."""
        self._first_probe_path = None
        self._first_trajectory = None
        self._awaiting_second = False
        self._last_accepted_signings = None
        self._last_accepted_client = None
        self._pen_capture_device = None
        self.selected_trajectory = None
        self.penStatusLabel.setText("No pen signature yet")
        self.selected_image_path = None
        self.probeFileLabel.setText("No file selected")
        self.confirmButton.setVisible(False)
        self._show_idle_message("Awaiting verification...")   # the last result was another visit's
        self.selectProbeButton.setText("Select Probe Image")
        self.signPenButton.setText("Sign on Tablet")

    # --- probes -------------------------------------------------------------
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

    def capture_pen_probe(self):
        """One signing on the tablet (issue #23, item 2). Closing the window changes nothing."""
        from src.capture.collect import collect_signatures
        captured = collect_signatures(count=1, parent=self)
        if captured is None:
            return
        self.selected_trajectory = captured["signatures"][0]
        self._pen_capture_device = {k: v for k, v in captured.items() if k != "signatures"}
        self.penStatusLabel.setText("Pen signature captured")

    # --- verification -------------------------------------------------------
    def run_verification(self):
        client_id = self.clientIdInput.text().strip()

        if not client_id:
            self._show_idle_message("Please enter a signer ID")
            return

        client = storage.get_client(client_id)
        if client is None:
            self._show_idle_message(
                f"No enrolled record found for '{client_id}'. Please enroll them first."
            )
            return

        use_image, use_pen = self._uses()
        # Enrolled references plus confirmed signings added on earlier visits.
        spatial_b64, spatial_origin = storage.get_reference_embeddings(client_id)
        temporal_b64 = list(client.get("dynamic_references") or [])

        if use_image and not spatial_b64:
            self._show_idle_message(
                f"'{client_id}' has no paper references (enrolled on the tablet only). "
                "Choose 'Pen only (tablet)'.")
            return
        if use_pen and not temporal_b64:
            self._show_idle_message(
                f"'{client_id}' has no pen references (enrolled with paper only). "
                "Choose 'Paper image only', or enrol their pen signatures first.")
            return
        if use_image and not self.selected_image_path:
            self._show_idle_message("Please select a probe image")
            return
        if use_pen and self.selected_trajectory is None:
            self._show_idle_message("Please press 'Sign on Tablet' and sign first")
            return

        problems = storage.check_references(
            client_id, spatial_origin if use_image else None, use_pen,
            self._pen_capture_device if use_pen else None)
        if problems:
            storage.log_verify_refused(client_id, self.current_employee_username or "unknown", problems)
            self._show_idle_message("\n".join(problems))
            return

        self.verifyButton.setEnabled(False)
        self.verifyButton.setText("Verifying…")
        self._show_idle_message("Processing…")
        self.confirmButton.setVisible(False)

        image_probe = self.selected_image_path if use_image else None
        pen_probe = self.selected_trajectory if use_pen else None
        if self._awaiting_second:
            # Second round: both signings of each kind, first signing first. Never a third.
            if use_image:
                image_probe = [self._first_probe_path, self.selected_image_path]
            if use_pen:
                pen_probe = [self._first_trajectory, self.selected_trajectory]

        self._pending_client_id = client_id
        self._pending_origin = spatial_origin
        self._pending_images = image_probe
        self._pending_trajectories = pen_probe
        self._verify_worker = VerifyWorker(
            spatial_b64 if use_image else [], spatial_origin, image_probe,
            reference_temporal_b64=temporal_b64 if use_pen else [],
            probe_trajectory=pen_probe,
        )
        self._verify_worker.finished.connect(self._on_verify_success)
        self._verify_worker.failed.connect(self._on_verify_failure)
        self._verify_worker.start()

    def _on_verify_success(self, result: dict):
        try:
            self._show_verify_result(result)
        finally:
            # Even if logging raises, the button must not stay on "Verifying…".
            self.verifyButton.setEnabled(True)
            self.verifyButton.setText("Verify")

    def _show_verify_result(self, result: dict):
        """Log first, then show: nothing is displayed that was not written to the audit log."""
        client_id = self._pending_client_id
        decision = result["decision"]
        confidence = round(result["confidence"], 1)

        # Replay check first, every round: the model returns the embeddings, storage hashes them.
        probes = result["probe_embeddings"]
        fingerprints = [storage.probe_fingerprint(b)
                        for b in probes.get("spatial", []) + probes.get("temporal", [])]
        if storage.is_replayed_probe(client_id, fingerprints):
            storage.log_replay_refused(
                client_id, self.current_employee_username or "unknown", fingerprints)
            self._end_round()
            self._last_accepted_signings = None
            self._last_accepted_client = None
            self.degradedModeIndicator.setText("")
            self._show_idle_message("Refused: this signing was already used (replay)")
            self.confirmButton.setVisible(False)
            return

        if decision == SIGN_AGAIN and not self._awaiting_second:
            # Not a verdict. Neutral request only: no score, no hint of how close it was.
            use_image, use_pen = self._uses()
            self._first_probe_path = self.selected_image_path if use_image else None
            self._first_trajectory = self.selected_trajectory if use_pen else None
            self._awaiting_second = True
            self.selected_image_path = None
            self.selected_trajectory = None
            self.probeFileLabel.setText("No file selected")
            self.penStatusLabel.setText("No pen signature yet")
            self.selectProbeButton.setText("Select Second Signing")
            self.signPenButton.setText("Sign on Tablet Again")
            self._show_idle_message("Please sign once more and give the new signing.")
            return
        if decision == SIGN_AGAIN:
            # Should never happen (the second call returns only Genuine or Forged).
            # Fail closed rather than asking a third time.
            decision = "Forged"

        # What actually ran, why, and the threshold, so the decision can be explained later.
        storage.append_verification_log(
            client_id=client_id,
            decision=decision,
            confidence=confidence,
            modality_state=interfaces.modality_state(result),
            employee_username=self.current_employee_username or "unknown",
            routing=result["routing"],
            degraded=result["degraded"],
            reasons=result["reasons"],
            model_version=storage.MODEL_VERSION,
            threshold=result["threshold"],
            probe_fingerprints=fingerprints,
        )

        signings = visit_signings(self._pending_images, self._pending_trajectories)
        self._end_round()

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

        routing = interfaces.ROUTING_LABELS.get(result.get("routing"), result.get("routing"))
        self.modalityStateLabel.setText(f"Checked: {routing}")
        if result.get("degraded"):
            reasons = "; ".join(result.get("reasons", [])) or "one modality only"
            self.degradedModeIndicator.setText(f"⚠ Degraded: {reasons}")
        else:
            self.degradedModeIndicator.setText("")

        accepted = decision == "Genuine" and not result.get("degraded")
        self._last_accepted_signings = signings if accepted else None
        self._last_accepted_client = client_id if accepted else None
        self._last_accepted_origin = self._pending_origin
        self.confirmButton.setVisible(accepted)

    def _end_round(self):
        """The visit's signings are used: forget the first signing and the pen probe."""
        self._first_probe_path = None
        self._first_trajectory = None
        self._awaiting_second = False
        self.selectProbeButton.setText("Select Probe Image")
        self.signPenButton.setText("Sign on Tablet")
        # A pen signature is used for one verification only.
        self.selected_trajectory = None
        self._pen_capture_device = None
        self.penStatusLabel.setText("No pen signature yet")

    def confirm_identity(self):
        """The employee confirms the person's identity: add this visit to the references."""
        if not self._last_accepted_signings:
            return
        self.confirmButton.setEnabled(False)
        self._confirm_worker = ConfirmedSigningWorker(self._last_accepted_signings)
        self._confirm_worker.finished.connect(self._on_confirmed)
        self._confirm_worker.failed.connect(self._on_confirm_failed)
        self._confirm_worker.start()

    def _on_confirmed(self, embeddings: list):
        employee = self.current_employee_username or "unknown"
        added = 0
        for emb in embeddings:
            for modality in ("spatial", "temporal"):
                extra = {"spatial_origin": self._last_accepted_origin} if modality == "spatial" else {}
                if emb.get(modality) and storage.add_confirmed_reference(
                        self._last_accepted_client, emb[modality], employee, modality=modality, **extra):
                    added += 1
        self._last_accepted_signings = None
        self.confirmButton.setEnabled(True)
        self.confirmButton.setVisible(False)
        self.degradedModeIndicator.setText("")
        self.modalityStateLabel.setText(
            "Signing added to the signer's references." if added
            else "Nothing was added to the signer's references.")

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
