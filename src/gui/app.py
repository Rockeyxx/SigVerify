"""
Main Application — Signature Verification System
"""

# Must load before PyQt5 — Windows DLL search-path conflict otherwise
import torch  # noqa: F401 (imported here only to force early DLL init)
import cv2    # noqa: F401

import sys
import traceback
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMessageBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QLineEdit
)

from src.gui.enrollment import EnrollmentPanel
from src.gui.verification import VerificationPanel
from src.gui.sidebar import Sidebar
from src.gui.dashboard import DashboardScreen
from src.gui.clients_page import ClientsPage
from src.gui.employees_page import EmployeesPage, restore_from_backup
from src.gui.profile_page import ProfilePage
from src.gui.styles import STYLE
from src.gui.workers import WarmUpWorker, wait_for_workers
import src.gui.storage as storage

STORAGE_ERROR_TITLE = "Cannot open the signature data"
RESTORED_NOTICE = "Backup restored. Log in again."


def show_unexpected_error(exc_type, exc, tb):
    """sys.excepthook: an error no page caught is printed and shown, instead of PyQt5 aborting."""
    traceback.print_exception(exc_type, exc, tb)
    QMessageBox.critical(None, "Unexpected error",
                         f"Something went wrong: {exc}. The action was not completed.")


def install_error_dialog():
    sys.excepthook = show_unexpected_error


def centered(content_layout):
    outer = QVBoxLayout()
    outer.addStretch(1)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addLayout(content_layout)
    row.addStretch(1)
    outer.addLayout(row)
    outer.addStretch(1)
    return outer


class LoginScreen(QWidget):
    def __init__(self, on_login_success):
        super().__init__()
        content = QVBoxLayout()
        content.setSpacing(14)
        content.setAlignment(Qt.AlignHCenter)

        title = QLabel("Employee Login")
        title.setProperty("role", "hero-title")
        title.setAlignment(Qt.AlignHCenter)
        content.addWidget(title)
        content.addSpacing(10)

        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("User ID")
        self.usernameInput.setFixedWidth(260)
        content.addWidget(self.usernameInput, alignment=Qt.AlignHCenter)

        self.passwordInput = QLineEdit()
        self.passwordInput.setPlaceholderText("Password")
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.passwordInput.setFixedWidth(260)
        content.addWidget(self.passwordInput, alignment=Qt.AlignHCenter)

        self.errorLabel = QLabel("")
        self.errorLabel.setProperty("role", "status-error")
        self.errorLabel.setAlignment(Qt.AlignHCenter)
        content.addWidget(self.errorLabel)

        login_button = QPushButton("Login")
        login_button.setProperty("role", "primary")
        login_button.setFixedWidth(260)
        login_button.setFixedHeight(44)
        login_button.clicked.connect(self._handle_login)
        content.addWidget(login_button, alignment=Qt.AlignHCenter)

        self.on_login_success = on_login_success
        self.setLayout(centered(content))

    def _handle_login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()

        try:
            success, employee, message = storage.authenticate_employee(username, password)
        except storage.StorageError as e:
            QMessageBox.critical(self, STORAGE_ERROR_TITLE, str(e))
            return
        if not success:
            self._set_label("status-error", f"⚠ {message}")
            return

        self.errorLabel.setText("")
        self.passwordInput.clear()
        self.on_login_success(employee)

    def show_notice(self, text):
        self._set_label("status-success", text)

    def _set_label(self, role, text):
        self.errorLabel.setProperty("role", role)
        self.errorLabel.setText(text)
        self.errorLabel.style().unpolish(self.errorLabel)
        self.errorLabel.style().polish(self.errorLabel)


class RegisterScreen(QWidget):
    """First run only: creates the admin account (the first account), or restores a backup."""

    def __init__(self, on_register_success, on_restored):
        super().__init__()
        content = QVBoxLayout()
        content.setSpacing(14)
        content.setAlignment(Qt.AlignHCenter)

        title = QLabel("Create the admin account")
        title.setProperty("role", "hero-title")
        title.setAlignment(Qt.AlignHCenter)
        content.addWidget(title)

        warning = QLabel(storage.ADMIN_WARNING)
        warning.setProperty("role", "warning")
        warning.setWordWrap(True)
        warning.setFixedWidth(420)
        warning.setAlignment(Qt.AlignHCenter)
        content.addWidget(warning, alignment=Qt.AlignHCenter)
        content.addSpacing(10)

        self.fullNameInput = QLineEdit()
        self.fullNameInput.setPlaceholderText("Full Name")
        self.fullNameInput.setFixedWidth(260)
        content.addWidget(self.fullNameInput, alignment=Qt.AlignHCenter)

        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("User ID")
        self.usernameInput.setFixedWidth(260)
        content.addWidget(self.usernameInput, alignment=Qt.AlignHCenter)

        self.passwordInput = QLineEdit()
        self.passwordInput.setPlaceholderText("Password")
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.passwordInput.setFixedWidth(260)
        content.addWidget(self.passwordInput, alignment=Qt.AlignHCenter)

        self.confirmPasswordInput = QLineEdit()
        self.confirmPasswordInput.setPlaceholderText("Confirm Password")
        self.confirmPasswordInput.setEchoMode(QLineEdit.Password)
        self.confirmPasswordInput.setFixedWidth(260)
        content.addWidget(self.confirmPasswordInput, alignment=Qt.AlignHCenter)

        self.messageLabel = QLabel("")
        self.messageLabel.setAlignment(Qt.AlignHCenter)
        self.messageLabel.setWordWrap(True)
        self.messageLabel.setFixedWidth(420)
        content.addWidget(self.messageLabel, alignment=Qt.AlignHCenter)

        self.createButton = QPushButton("Create Admin Account")
        self.createButton.setProperty("role", "primary")
        self.createButton.setFixedWidth(260)
        self.createButton.setFixedHeight(44)
        self.createButton.clicked.connect(self._handle_register)
        content.addWidget(self.createButton, alignment=Qt.AlignHCenter)

        self.restoreButton = QPushButton("Restore from backup")
        self.restoreButton.setProperty("role", "ghost")
        self.restoreButton.clicked.connect(self._handle_restore)
        content.addWidget(self.restoreButton, alignment=Qt.AlignHCenter)

        self.on_register_success = on_register_success
        self.on_restored = on_restored
        self.setLayout(centered(content))

    def _handle_restore(self):
        outcome = restore_from_backup(self)
        if outcome is None:
            return
        ok, message = outcome
        if not ok:
            self._set_error(message)
            return
        self.on_restored()

    def _handle_register(self):
        full_name = self.fullNameInput.text().strip()
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()
        confirm = self.confirmPasswordInput.text()

        if not full_name or not username or not password:
            self._set_error("Please fill in all fields")
            return
        if password != confirm:
            self._set_error("Passwords do not match")
            return

        success, message = storage.register_employee(full_name, username, password)
        if not success:
            self._set_error(message)
            return

        self.fullNameInput.clear()
        self.usernameInput.clear()
        self.passwordInput.clear()
        self.confirmPasswordInput.clear()
        self.on_register_success()

    def _set_error(self, text):
        self.messageLabel.setProperty("role", "status-error")
        self.messageLabel.setText(f"⚠ {text}")
        self._refresh_message_style()

    def _refresh_message_style(self):
        self.messageLabel.style().unpolish(self.messageLabel)
        self.messageLabel.style().polish(self.messageLabel)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signature Verification System")
        self.resize(900, 680)
        self.current_employee = None
        self.startup_failed = False
        try:
            first_run = storage.is_first_run()
        except storage.StorageError as e:   # e.g. a database made by an older version
            self.startup_failed = True
            QMessageBox.critical(None, STORAGE_ERROR_TITLE, str(e))
            QTimer.singleShot(0, QApplication.instance().quit)
            return
        self._build_auth_stack(first_run)

        # AGENTS.md: call SignatureService.warm_up() once when the app opens, on a
        # background thread, so the first verification does not pay the model load.
        # Started here (not after login) and kept on self so the thread is not garbage-collected.
        self._warm_up_worker = WarmUpWorker()
        self._warm_up_worker.start()

    def closeEvent(self, event):
        # Closing while a worker (warm-up, enrol, verify, confirm) is still running would print
        # "QThread: Destroyed while thread is still running" or crash, so wait for all of them.
        wait_for_workers()
        super().closeEvent(event)

    def _build_auth_stack(self, first_run=False):
        """First run: the admin-account screen first. Afterwards: Login only, no Register."""
        self.auth_stack = QStackedWidget()
        self.login_screen = LoginScreen(on_login_success=self._handle_login_success)
        if first_run:
            self.auth_stack.addWidget(RegisterScreen(
                on_register_success=lambda: self._handle_login_success(storage.current_employee()),
                on_restored=self._handle_restored,
            ))
        self.auth_stack.addWidget(self.login_screen)
        self.auth_stack.setCurrentIndex(0)
        self.setCentralWidget(self.auth_stack)

    def _show_login(self, notice=""):
        self.current_employee = None
        self._build_auth_stack()
        self.auth_stack.setCurrentWidget(self.login_screen)
        if notice:
            self.login_screen.show_notice(notice)

    def _build_app_stack(self):
        container = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.go_dashboard.connect(lambda: self.content_stack.setCurrentWidget(self.dashboard_screen))
        self.sidebar.go_enrollment.connect(lambda: self.content_stack.setCurrentWidget(self.enrollment_panel))
        self.sidebar.go_verification.connect(lambda: self.content_stack.setCurrentWidget(self.verification_panel))
        self.sidebar.go_clients.connect(self._open_clients_page)
        self.sidebar.go_employees.connect(self._open_employees_page)
        self.sidebar.go_profile.connect(self._open_profile_page)
        self.sidebar.go_logout.connect(self._handle_logout)
        main_layout.addWidget(self.sidebar)

        self.content_stack = QStackedWidget()
        self.dashboard_screen = DashboardScreen()
        self.enrollment_panel = EnrollmentPanel()
        self.verification_panel = VerificationPanel()
        self.clients_page = ClientsPage()
        self.profile_page = ProfilePage()
        self.employees_page = EmployeesPage()

        self.enrollment_panel.client_list_changed.connect(self._refresh_after_change)
        self.clients_page.client_selected.connect(self._open_client_in_enrollment)
        self.profile_page.profile_updated.connect(self._handle_profile_updated)
        self.employees_page.restored.connect(self._handle_restored)

        self.content_stack.addWidget(self.dashboard_screen)
        self.content_stack.addWidget(self.enrollment_panel)
        self.content_stack.addWidget(self.verification_panel)
        self.content_stack.addWidget(self.clients_page)
        self.content_stack.addWidget(self.profile_page)
        self.content_stack.addWidget(self.employees_page)
        self.content_stack.setCurrentWidget(self.dashboard_screen)

        main_layout.addWidget(self.content_stack)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _handle_login_success(self, employee: dict):
        self.current_employee = employee
        self._build_app_stack()
        self.sidebar.set_employee(employee["full_name"], employee["username"])
        self.sidebar.set_admin(employee["is_admin"])
        self.dashboard_screen.set_employee_name(employee["full_name"])
        self.enrollment_panel.set_current_employee(employee["username"])
        self.verification_panel.set_current_employee(employee["username"])
        self.employees_page.set_current_employee(employee["username"])
        self.profile_page.load_employee(employee)
        self._refresh_after_change()

    def _handle_logout(self):
        storage.logout()
        self._show_login()

    def _handle_restored(self):
        # The backup's admin is logged in by the restore; ask for a fresh login instead.
        storage.logout()
        self._show_login(RESTORED_NOTICE)

    def _open_employees_page(self):
        if not self.current_employee["is_admin"]:
            return
        self.employees_page.refresh()
        self.content_stack.setCurrentWidget(self.employees_page)

    def _open_clients_page(self):
        self.clients_page.refresh(self.current_employee["username"])
        self.content_stack.setCurrentWidget(self.clients_page)

    def _open_profile_page(self):
        self.content_stack.setCurrentWidget(self.profile_page)

    def _handle_profile_updated(self, updated_employee: dict):
        self.current_employee = updated_employee
        self.sidebar.set_employee(updated_employee["full_name"], updated_employee["username"])
        self.dashboard_screen.set_employee_name(updated_employee["full_name"])
        self.enrollment_panel.set_current_employee(updated_employee["username"])
        self.verification_panel.set_current_employee(updated_employee["username"])
        self.employees_page.set_current_employee(updated_employee["username"])

    def _open_client_in_enrollment(self, client_id: str):
        self.enrollment_panel.prefill_client(client_id)
        self.content_stack.setCurrentWidget(self.enrollment_panel)

    def _refresh_after_change(self):
        username = self.current_employee["username"]
        self.dashboard_screen.refresh(username)
        self.clients_page.refresh(username)


def main():
    from src.capture.pen_input import choose_pen_input
    choose_pen_input()
    app = QApplication(sys.argv)
    install_error_dialog()
    app.setStyleSheet(STYLE)
    window = MainWindow()
    if not window.startup_failed:
        window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()