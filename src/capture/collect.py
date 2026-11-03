"""The tablet signing window the GUI calls: n signatures, a thank-you screen, Confirm or Start over.

The GUI owns the model call, the database and the score; this returns raw trajectories only.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from src.capture.tablet import SignatureCaptureWidget
from src.gui.styles import ACCENT, ACCENT_DARK, PRIMARY_LIGHT, STYLE

TEXT = {
    "en": {
        "title": "Sign on the tablet",
        "counter": "Signature {i} of {n}",
        "hint": "Sign inside the box, then press I approve or Redo.",
        "done": "Thank you, {n}/{n} done.",
        "btn_approve": "I approve", "btn_redo": "Redo",
        "btn_confirm": "Confirm", "btn_restart": "Start over",
    },
    "ar": {
        "title": "وقّع على الجهاز اللوحي",
        "counter": "التوقيع {i} من {n}",
        "hint": "وقّع داخل المربع، ثم اضغط «أوافق» أو «إعادة».",
        "done": "شكرًا لك، اكتمل {n} من {n}.",
        "btn_approve": "أوافق", "btn_redo": "إعادة",
        "btn_confirm": "تأكيد", "btn_restart": "البدء من جديد",
    },
}


def _role(widget, role):
    widget.setProperty("role", role)
    return widget


class CollectDialog(QDialog):
    """count signatures on the tablet, then 'Thank you, n/n done' with Confirm / Start over."""

    def __init__(self, count=5, parent=None, lang="en"):
        super().__init__(parent)
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}")
        self.count, self.lang, self.signatures = count, lang, []
        self.setModal(True)
        # The app theme (same look outside the app), plus a larger "Thank you" and a lighter
        # Start over button ('ghost' has no background, so it read as plain text).
        self.setStyleSheet(
            STYLE
            + 'QLabel[role="status-success"] { font-size: 24px; }'
            + f'QPushButton[role="secondary"] {{ background-color: {PRIMARY_LIGHT}; '
              f'border: 1px solid {ACCENT}; }}'
            + f'QPushButton[role="secondary"]:hover {{ background-color: {ACCENT_DARK}; }}')

        self.counter = _role(QLabel(), "hero-title")
        self.capture = SignatureCaptureWidget(self)
        self.capture.signature_confirmed.connect(self._on_signature)
        self.done = _role(QLabel(), "status-success")
        self.btn_confirm = _role(QPushButton(), "primary")
        self.btn_confirm.clicked.connect(self.confirm)
        self.btn_restart = _role(QPushButton(), "secondary")
        self.btn_restart.clicked.connect(self.start_over)
        self.btn_lang = _role(QPushButton(), "ghost")
        self.btn_lang.clicked.connect(self.toggle_language)
        _role(self.capture.label, "title")       # the instruction, read standing at the tablet
        self.capture.label.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(self.counter, 1)
        top.addWidget(self.btn_lang)
        end = QHBoxLayout()
        end.addWidget(self.btn_restart)
        end.addWidget(self.btn_confirm)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.capture)
        layout.addWidget(self.done)
        layout.addLayout(end)
        self._retranslate()

    @property
    def finished(self):
        return len(self.signatures) >= self.count

    # --- flow ---------------------------------------------------------------
    def _on_signature(self, tensor):
        if self.finished:
            return
        self.signatures.append(tensor)
        self.capture.label.setText(self._t("hint"))   # the widget wrote "Accepted!" there
        self._show_stage()

    def confirm(self):
        if self.finished:
            self.accept()

    def start_over(self):
        self.signatures = []
        self.capture.canvas.clear_canvas()
        self.capture.label.setText(self._t("hint"))
        self._show_stage()

    def _show_stage(self):
        done = self.finished
        self.capture.setHidden(done)
        self.counter.setHidden(done)
        for w in (self.done, self.btn_confirm, self.btn_restart):
            w.setHidden(not done)
        self.counter.setText(self._t("counter", i=min(len(self.signatures) + 1, self.count),
                                     n=self.count))
        self.done.setText(self._t("done", n=self.count))

    # --- language -----------------------------------------------------------
    def set_language(self, lang):
        self.lang = lang
        self._retranslate()

    def toggle_language(self):
        self.set_language("ar" if self.lang == "en" else "en")

    def _t(self, key, **values):
        return TEXT[self.lang][key].format(**values)

    def _retranslate(self):
        self.setWindowTitle(self._t("title"))
        for widget, key in ((self.capture.btn_confirm, "btn_approve"),
                            (self.capture.btn_reset, "btn_redo"),
                            (self.btn_confirm, "btn_confirm"),
                            (self.btn_restart, "btn_restart")):
            widget.setText(self._t(key))
        self.btn_lang.setText("English" if self.lang == "ar" else "العربية")
        direction = Qt.RightToLeft if self.lang == "ar" else Qt.LeftToRight
        for label in (self.counter, self.done, self.capture.label):
            label.setLayoutDirection(direction)
            label.setAlignment(Qt.AlignLeading | Qt.AlignVCenter)   # right in Arabic
        self.capture.label.setText(self._t("hint"))
        self._show_stage()


def collect_signatures(count=5, parent=None, lang="en"):
    """Open the signing window. count=5 to enrol, count=1 to verify.

    Returns count arrays [L, TEMPORAL_INPUT_FEATURES] (float32, signing order) after Confirm,
    or None when the window is closed first.
    """
    dialog = CollectDialog(count, parent, lang)
    if dialog.exec_() == QDialog.Accepted:
        return list(dialog.signatures)
    return None
