"""Modal dialog for collecting n tablet signatures with Confirm / Start over."""
import platform

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QInputDialog, QLabel, QPushButton, QVBoxLayout

from config.settings import CAPTURE_NO_DRIVER
from src.capture import tablet
from src.capture.pen_input import wintab_installed
from src.capture.tablet import SignatureCaptureWidget
from src.gui.styles import ACCENT, ACCENT_DARK, PRIMARY_LIGHT, STYLE

TEXT = {
    "en": {
        "title": "Sign on the tablet",
        "counter": "Signature {i} of {n}",
        "hint": "Sign inside the box, then press I approve or Redo.",
        "no_tablet": "No tablet found. Plug it in, or choose Paper image only.",
        "done": "Thank you, {n}/{n} done.",
        "btn_approve": "I approve", "btn_redo": "Redo",
        "btn_confirm": "Confirm", "btn_restart": "Start over",
        "choose_title": "Choose the tablet",
        "choose_label": "More than one tablet is connected. Which one will you sign on?",
    },
    "ar": {
        "title": "وقّع على الجهاز اللوحي",
        "counter": "التوقيع {i} من {n}",
        "hint": "وقّع داخل المربع، ثم اضغط «أوافق» أو «إعادة».",
        "no_tablet": "لم يُعثر على جهاز لوحي. صِله، أو اختر «Paper image only».",
        "done": "شكرًا لك، اكتمل {n} من {n}.",
        "btn_approve": "أوافق", "btn_redo": "إعادة",
        "btn_confirm": "تأكيد", "btn_restart": "البدء من جديد",
        "choose_title": "اختر الجهاز اللوحي",
        "choose_label": "يوجد أكثر من جهاز لوحي متصل. على أيّها ستوقّع؟",
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
        # Secondary button needs an explicit background since 'ghost' renders transparent.
        self.setStyleSheet(
            STYLE
            + 'QLabel[role="status-success"] { font-size: 24px; }'
            + f'QPushButton[role="secondary"] {{ background-color: {PRIMARY_LIGHT}; '
              f'border: 1px solid {ACCENT}; }}'
            + f'QPushButton[role="secondary"]:hover {{ background-color: {ACCENT_DARK}; }}')

        self.counter = _role(QLabel(), "hero-title")
        self.capture = SignatureCaptureWidget(self)
        self.capture.signature_confirmed.connect(self._on_signature)
        # Connected after SignatureCaptureWidget's handler so our localized hint wins.
        self.capture.btn_reset.clicked.connect(lambda: self.capture.label.setText(self._t("hint")))
        self.done = _role(QLabel(), "status-success")
        self.btn_confirm = _role(QPushButton(), "primary")
        self.btn_confirm.clicked.connect(self.confirm)
        self.btn_restart = _role(QPushButton(), "secondary")
        self.btn_restart.clicked.connect(self.start_over)
        self.btn_lang = _role(QPushButton(), "ghost")
        self.btn_lang.clicked.connect(self.toggle_language)
        _role(self.capture.label, "title")
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
        # Without Wintab, Windows Ink still works even when no tablet name can be enumerated.
        self._no_tablet = not self.capture.tablets and (platform.system() != "Windows" or wintab_installed())
        self._retranslate()

    @property
    def finished(self):
        return len(self.signatures) >= self.count

    def _on_signature(self, trajectory):
        if self.finished:
            return
        self.signatures.append(trajectory)
        self.capture.label.setText(self._t("hint"))
        self._show_stage()

    def confirm(self):
        if self.finished:
            self.accept()

    def choose_tablet(self):
        """Prompt user to pick a tablet when multiple are connected; Qt5 does not identify source device."""
        tablets = self.capture.tablets
        if len(tablets) < 2:
            return True
        name, ok = QInputDialog.getItem(self, self._t("choose_title"), self._t("choose_label"),
                                        tablets, 0, False)
        if ok:
            self.capture.device_model = name
        return ok

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
        self.capture.label.setText(self._t("no_tablet" if self._no_tablet else "hint"))
        self._show_stage()


def collect_signatures(count=5, parent=None, lang="en"):
    """Open the signing dialog and return trajectories plus device metadata, or None if cancelled."""
    dialog = CollectDialog(count, parent, lang)
    if dialog.choose_tablet() and dialog.exec_() == QDialog.Accepted:
        info = dialog.capture.capture_info()
        if info["device_model"] == CAPTURE_NO_DRIVER:
            info["device_model"] = (tablet.tablet_names() or [CAPTURE_NO_DRIVER])[0]
        return {"signatures": list(dialog.signatures), **info}
    return None
