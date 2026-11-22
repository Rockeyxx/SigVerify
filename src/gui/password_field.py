"""
password_field.py — The password field every screen uses (#58): an eye button that shows or
hides the text, and for a new password a live rule hint and a Generate button. The rule itself
is checked in DatabaseManager; the hint only repeats it early.
"""

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout
)

from src.database.db_manager import generate_password, password_problem
from src.gui.styles import DANGER, SUCCESS, TEXT_MUTED

NO_RECOVERY_WARNING = ("This password unlocks your copy of the vault's key. It cannot be "
                       "recovered. Write it down or keep it in a password manager now.")


def _eye_icon(crossed: bool) -> QIcon:
    # drawn, not loaded: no icon file or SVG plugin to ship
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(TEXT_MUTED), 2.5))
    painter.drawEllipse(QPointF(16, 16), 13, 7.5)
    painter.setBrush(QColor(TEXT_MUTED))
    painter.drawEllipse(QPointF(16, 16), 4, 4)
    if crossed:
        painter.drawLine(5, 27, 27, 5)
    painter.end()
    return QIcon(pixmap)


class PasswordField(QLineEdit):
    """A password QLineEdit with an eye button. Hidden at start and whenever its page is hidden."""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setEchoMode(QLineEdit.Password)
        self.eye = self.addAction(_eye_icon(crossed=False), QLineEdit.TrailingPosition)
        self.eye.setToolTip("Show password")
        self.eye.triggered.connect(lambda: self._show(self.echoMode() == QLineEdit.Password))

    def _show(self, visible):
        self.setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        self.eye.setIcon(_eye_icon(crossed=visible))
        self.eye.setToolTip("Hide password" if visible else "Show password")

    def hideEvent(self, event):
        if self.echoMode() != QLineEdit.Password:   # Qt also sends this at exit, after Python cleared self.eye
            self._show(False)
        super().hideEvent(event)


class GeneratedPasswordDialog(QDialog):
    def __init__(self, parent, password):
        super().__init__(parent)
        self.setWindowTitle("Generated password")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        warning = QLabel(NO_RECOVERY_WARNING)
        warning.setWordWrap(True)
        warning.setProperty("role", "warning")
        layout.addWidget(warning)
        shown = QLineEdit(password)
        shown.setReadOnly(True)
        layout.addWidget(shown)
        row = QHBoxLayout()
        self.copyButton = QPushButton("Copy")
        self.copyButton.clicked.connect(lambda: QApplication.clipboard().setText(password))
        done = QPushButton("I have saved it")
        done.setProperty("role", "primary")
        done.clicked.connect(self.accept)
        row.addWidget(self.copyButton)
        row.addWidget(done)
        layout.addLayout(row)


def show_generated(parent, password):
    GeneratedPasswordDialog(parent, password).exec_()


def new_password_helpers(field, confirm):
    """Wires a new-password pair. Returns (Generate button, live hint label) to place in a layout."""
    hint = QLabel("")
    hint.setWordWrap(True)

    def update_hint(text):
        problem = password_problem(text) if text else None
        hint.setText(problem or ("Long enough" if text else ""))
        hint.setStyleSheet(f"color: {DANGER if problem else SUCCESS};")

    field.textChanged.connect(update_hint)

    def generate():
        password = generate_password()
        field.setText(password)
        confirm.setText(password)
        show_generated(field.window(), password)

    button = QPushButton("Generate password")
    button.setProperty("role", "ghost")
    button.clicked.connect(generate)
    return button, hint
