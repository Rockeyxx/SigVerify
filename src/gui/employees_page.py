"""
employees_page.py — Admin page: employee accounts, backup and restore
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QInputDialog, QMessageBox, QDialog, QDialogButtonBox
)

import src.gui.storage as storage
from src.database.db_manager import EVENT_BACKUP, EVENT_DELETE
from src.gui.password_field import PasswordField, new_password_helpers


class BackupOwnerDialog(QDialog):
    """Asks for an owner of the backup: User ID and password (with the eye button)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Restore from backup")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("An owner of the backup:"))
        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("User ID")
        self.passwordInput = PasswordField("Password")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        for w in (self.usernameInput, self.passwordInput, buttons):
            layout.addWidget(w)


def ask_backup_owner(parent):
    """(User ID, password), or None when cancelled or no User ID was typed."""
    dialog = BackupOwnerDialog(parent)
    if dialog.exec_() != QDialog.Accepted or not dialog.usernameInput.text().strip():
        return None
    return dialog.usernameInput.text().strip(), dialog.passwordInput.text()


def restore_from_backup(parent, new_company=False):
    """Ask for a backup file and an admin of that backup, then restore it over the current
    company, or as a new vault (new_company: its name is asked first).
    Returns None when the employee cancels, otherwise storage's (success, message)."""
    company = None
    if new_company:
        company, ok = QInputDialog.getText(parent, "Restore a vault", "Vault name, as in the backup:")
        if not ok or not company.strip():
            return None
    path, _ = QFileDialog.getOpenFileName(parent, "Choose a backup", "", "Backup (*.db);;All files (*)")
    if not path:
        return None
    owner = ask_backup_owner(parent)
    if owner is None:
        return None
    if new_company:
        return storage.restore_company(company, path, *owner)
    return storage.restore_database(path, *owner)


class EmployeesPage(QWidget):
    restored = pyqtSignal()   # a backup replaced the database: the window logs out

    def __init__(self):
        super().__init__()
        self.admin_username = None

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("Members")
        title.setProperty("role", "title")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Full Name", "User ID", "Owner"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        layout.addWidget(self._section_label("ADD A MEMBER"))
        add_row = QHBoxLayout()
        self.fullNameInput = QLineEdit()
        self.fullNameInput.setPlaceholderText("Full Name")
        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("User ID")
        self.passwordInput = PasswordField("Password")
        self.confirmPasswordInput = PasswordField("Confirm Password")
        self.addButton = QPushButton("Add Member")
        self.addButton.setProperty("role", "primary")
        self.addButton.clicked.connect(self._handle_add)
        generate, hint = new_password_helpers(self.passwordInput, self.confirmPasswordInput)
        for w in (self.fullNameInput, self.usernameInput, self.passwordInput,
                  self.confirmPasswordInput, generate, self.addButton):
            add_row.addWidget(w)
        layout.addLayout(add_row)
        layout.addWidget(hint)

        layout.addWidget(self._section_label("SELECTED MEMBER"))
        selected_row = QHBoxLayout()
        self.newPasswordInput = PasswordField("New Password")
        self.confirmNewPasswordInput = PasswordField("Confirm New Password")
        self.resetButton = QPushButton("Reset Password")
        self.resetButton.setProperty("role", "ghost")
        self.resetButton.clicked.connect(self._handle_reset)
        self.removeButton = QPushButton("Remove Member")
        self.removeButton.setProperty("role", "danger")
        self.removeButton.clicked.connect(self._handle_remove)
        generate, hint = new_password_helpers(self.newPasswordInput, self.confirmNewPasswordInput)
        for w in (self.newPasswordInput, self.confirmNewPasswordInput, generate, self.resetButton,
                  self.removeButton):
            selected_row.addWidget(w)
        layout.addLayout(selected_row)
        layout.addWidget(hint)

        layout.addWidget(self._section_label("BACKUP"))
        backup_row = QHBoxLayout()
        self.backupButton = QPushButton("Back Up to a Folder")
        self.backupButton.setProperty("role", "ghost")
        self.backupButton.clicked.connect(self._handle_backup)
        self.restoreButton = QPushButton("Restore from Backup")
        self.restoreButton.setProperty("role", "danger")
        self.restoreButton.clicked.connect(self._handle_restore)
        backup_row.addWidget(self.backupButton)
        backup_row.addWidget(self.restoreButton)
        backup_row.addStretch()
        layout.addLayout(backup_row)

        self.lastBackupLabel = QLabel("Last backup: never")
        self.lastBackupLabel.setWordWrap(True)
        self.lastDeleteLabel = QLabel("Last signer deletion: none")
        self.lastDeleteLabel.setWordWrap(True)
        layout.addWidget(self.lastBackupLabel)
        layout.addWidget(self.lastDeleteLabel)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)
        layout.addWidget(self.statusLabel)

        layout.addStretch()
        self.setLayout(layout)

    def _section_label(self, text):
        label = QLabel(text)
        label.setProperty("role", "section")
        return label

    def set_current_employee(self, username: str):
        self.admin_username = username

    def refresh(self):
        employees = storage.list_employees()
        self.table.setRowCount(len(employees))
        for row, emp in enumerate(employees):
            self.table.setItem(row, 0, QTableWidgetItem(emp["full_name"]))
            self.table.setItem(row, 1, QTableWidgetItem(emp["username"]))
            self.table.setItem(row, 2, QTableWidgetItem("Yes" if emp["is_admin"] else "No"))
        self._refresh_evidence()

    def _refresh_evidence(self):
        ev_bak = storage.last_event(EVENT_BACKUP)
        if ev_bak and ev_bak.get("timestamp"):
            by = ev_bak.get("employee") or "unknown"
            self.lastBackupLabel.setText(f"Last backup: {ev_bak['timestamp']} by {by}")
        else:
            self.lastBackupLabel.setText("Last backup: never")

        ev_del = storage.last_event(EVENT_DELETE)
        if ev_del and ev_del.get("timestamp"):
            by = ev_del.get("employee") or "unknown"
            client = ev_del.get("client") or "unknown"
            self.lastDeleteLabel.setText(f"Last signer deletion: {client} at {ev_del['timestamp']} by {by}")
        else:
            self.lastDeleteLabel.setText("Last signer deletion: none")

    def _selected_username(self):
        row = self.table.currentRow()
        item = self.table.item(row, 1) if row >= 0 else None
        if item is None:
            self._set_error("Select a member in the list first")
            return None
        return item.text()

    def _handle_add(self):
        full_name = self.fullNameInput.text().strip()
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()
        if not full_name or not username or not password:
            self._set_error("Please fill in all fields")
            return
        if password != self.confirmPasswordInput.text():
            self._set_error("Passwords do not match")
            return
        ok, message = storage.register_employee(full_name, username, password)
        if not ok:
            self._set_error(message)
            return
        for field in (self.fullNameInput, self.usernameInput, self.passwordInput, self.confirmPasswordInput):
            field.clear()
        self.refresh()
        self._set_success(message)

    def _handle_reset(self):
        username = self._selected_username()
        if username is None:
            return
        password = self.newPasswordInput.text()
        if not password:
            self._set_error("Type the new password twice")
            return
        if password != self.confirmNewPasswordInput.text():
            self._set_error("Passwords do not match")
            return
        ok, message = storage.reset_employee_password(self.admin_username, username, password)
        if not ok:
            self._set_error(message)
            return
        self.newPasswordInput.clear()
        self.confirmNewPasswordInput.clear()
        self._set_success(f"New password set for {username}")

    def _handle_remove(self):
        username = self._selected_username()
        if username is None:
            return
        confirm = QMessageBox.question(
            self, "Remove Member",
            f"Remove the account '{username}'? They can no longer log in. "
            "Their signers and log entries stay.",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return
        ok, message = storage.remove_employee(self.admin_username, username)
        if not ok:
            self._set_error(message)
            return
        self.refresh()
        self._set_success(f"Removed {username}")

    def _handle_backup(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the backup")
        if not folder:
            return
        ok, result = storage.backup_database(folder, self.admin_username)
        if not ok:
            self._set_error(result)
            return
        self.refresh()
        self._set_success(f"Backup saved to {result}")

    def _handle_restore(self):
        outcome = restore_from_backup(self)
        if outcome is None:
            return
        ok, message = outcome
        if not ok:
            self._set_error(message)
            return
        self.refresh()
        self.restored.emit()

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
