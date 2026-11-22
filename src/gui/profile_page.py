"""
profile_page.py — Employee profile editing screen
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
)

import src.gui.storage as storage
from src.gui.password_field import PasswordField, new_password_helpers


class ProfilePage(QWidget):
    profile_updated = pyqtSignal(dict)  # يرسل بيانات الموظف المحدثة

    def __init__(self):
        super().__init__()
        self.current_username = None

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("My Profile")
        title.setProperty("role", "title")
        layout.addWidget(title)

        layout.addWidget(self._section_label("FULL NAME"))
        self.fullNameInput = QLineEdit()
        layout.addWidget(self.fullNameInput)

        layout.addWidget(self._section_label("USER ID"))
        self.usernameInput = QLineEdit()
        layout.addWidget(self.usernameInput)

        layout.addWidget(self._section_label("NEW PASSWORD (leave blank to keep current)"))
        self.newPasswordInput = PasswordField()
        layout.addWidget(self.newPasswordInput)

        layout.addWidget(self._section_label("CONFIRM NEW PASSWORD"))
        self.confirmNewPasswordInput = PasswordField()
        layout.addWidget(self.confirmNewPasswordInput)
        generate, hint = new_password_helpers(self.newPasswordInput, self.confirmNewPasswordInput)
        layout.addWidget(hint)
        layout.addWidget(generate)

        layout.addWidget(self._section_label("CURRENT PASSWORD (needed to set a new one)"))
        self.currentPasswordInput = PasswordField()
        layout.addWidget(self.currentPasswordInput)

        self.saveButton = QPushButton("Save Changes")
        self.saveButton.setProperty("role", "primary")
        self.saveButton.clicked.connect(self._handle_save)
        layout.addWidget(self.saveButton)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)   # long messages must not widen the window
        layout.addWidget(self.statusLabel)

        layout.addStretch()
        self.setLayout(layout)

    def _section_label(self, text):
        label = QLabel(text)
        label.setProperty("role", "section")
        return label

    def load_employee(self, employee: dict):
        self.current_username = employee["username"]
        self.fullNameInput.setText(employee["full_name"])
        self.usernameInput.setText(employee["username"])
        self.newPasswordInput.clear()
        self.confirmNewPasswordInput.clear()
        self.currentPasswordInput.clear()
        self.statusLabel.setText("")

    def _handle_save(self):
        new_full_name = self.fullNameInput.text().strip()
        new_username = self.usernameInput.text().strip()
        new_password = self.newPasswordInput.text() or None   # never stripped, as at login

        if not new_full_name or not new_username:
            self._set_error("Full name and User ID cannot be empty")
            return
        if (new_password or "") != self.confirmNewPasswordInput.text():
            self._set_error("The new passwords do not match")
            return

        success, message, updated_employee = storage.update_employee_profile(
            self.current_username, new_full_name, new_username, new_password,
            current_password=self.currentPasswordInput.text() or None,
        )

        if not success:
            self._set_error(message)
            return

        self.current_username = updated_employee["username"]
        self._set_success(message)
        self.newPasswordInput.clear()
        self.confirmNewPasswordInput.clear()
        self.currentPasswordInput.clear()
        self.profile_updated.emit(updated_employee)

    def _set_error(self, text):
        self.statusLabel.setProperty("role", "status-error")
        self.statusLabel.setText(f"⚠ {text}")
        self._refresh_style()

    def _set_success(self, text):
        self.statusLabel.setProperty("role", "status-success")
        self.statusLabel.setText(f"✔ {text}")
        self._refresh_style()

    def _refresh_style(self):
        self.statusLabel.style().unpolish(self.statusLabel)
        self.statusLabel.style().polish(self.statusLabel)