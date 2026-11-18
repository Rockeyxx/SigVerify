"""
Main Application — Signature Verification System
"""

# Must load before PyQt5 — Windows DLL search-path conflict otherwise
import torch  # noqa: F401 (imported here only to force early DLL init)
import cv2    # noqa: F401

import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QLineEdit
)

from src.gui.enrollment import EnrollmentPanel
from src.gui.verification import VerificationPanel
from src.gui.sidebar import Sidebar
from src.gui.dashboard import DashboardScreen
from src.gui.clients_page import ClientsPage
from src.gui.profile_page import ProfilePage
from src.gui.styles import STYLE
from src.gui.workers import WarmUpWorker, wait_for_workers
import src.gui.storage as storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "assets", "signature_icon.png")


def create_icon_label(size=160):
    icon_label = QLabel()
    pixmap = QPixmap(ICON_PATH)
    if pixmap.isNull():
        icon_label.setText("✎")
        icon_label.setStyleSheet("font-size: 48px; color: #3d7dfc;")
        print(f"[WARNING] Icon not found at: {ICON_PATH}")
    else:
        scaled = pixmap.scaledToWidth(size, Qt.SmoothTransformation)
        icon_label.setPixmap(scaled)
    icon_label.setAlignment(Qt.AlignHCenter)
    return icon_label


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


class WelcomeScreen(QWidget):
    def __init__(self, go_to_login, go_to_register):
        super().__init__()
        content = QVBoxLayout()
        content.setSpacing(10)
        content.setAlignment(Qt.AlignHCenter)

        content.addWidget(create_icon_label())
        content.addSpacing(8)

        title = QLabel("Signature Verification System")
        title.setProperty("role", "hero-title")
        title.setAlignment(Qt.AlignHCenter)
        content.addWidget(title)

        subtitle = QLabel("TEAM 06  •  EE 499 SENIOR DESIGN PROJECT")
        subtitle.setProperty("role", "hero-subtitle")
        subtitle.setAlignment(Qt.AlignHCenter)
        content.addWidget(subtitle)

        content.addSpacing(30)

        buttons_row = QHBoxLayout()
        login_button = QPushButton("Login")
        login_button.setProperty("role", "primary")
        login_button.setFixedWidth(180)
        login_button.setFixedHeight(46)

        register_button = QPushButton("Register")
        register_button.setProperty("role", "primary")
        register_button.setFixedWidth(180)
        register_button.setFixedHeight(46)

        login_button.clicked.connect(go_to_login)
        register_button.clicked.connect(go_to_register)

        buttons_row.addWidget(login_button)
        buttons_row.addSpacing(16)
        buttons_row.addWidget(register_button)

        content.addLayout(buttons_row)
        self.setLayout(centered(content))


class LoginScreen(QWidget):
    def __init__(self, on_login_success, go_to_register):
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

        register_link = QPushButton("Don't have an account? Register")
        register_link.setProperty("role", "ghost")
        register_link.clicked.connect(go_to_register)
        content.addWidget(register_link, alignment=Qt.AlignHCenter)

        self.on_login_success = on_login_success
        self.setLayout(centered(content))

    def _handle_login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()

        success, employee, message = storage.authenticate_employee(username, password)
        if not success:
            self.errorLabel.setText(f"⚠ {message}")
            return

        self.errorLabel.setText("")
        self.passwordInput.clear()
        self.on_login_success(employee)


class RegisterScreen(QWidget):
    def __init__(self, on_register_success, go_to_login):
        super().__init__()
        content = QVBoxLayout()
        content.setSpacing(14)
        content.setAlignment(Qt.AlignHCenter)

        title = QLabel("Employee Registration")
        title.setProperty("role", "hero-title")
        title.setAlignment(Qt.AlignHCenter)
        content.addWidget(title)
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
        content.addWidget(self.messageLabel)

        register_button = QPushButton("Register")
        register_button.setProperty("role", "primary")
        register_button.setFixedWidth(260)
        register_button.setFixedHeight(44)
        register_button.clicked.connect(self._handle_register)
        content.addWidget(register_button, alignment=Qt.AlignHCenter)

        login_link = QPushButton("Already have an account? Login")
        login_link.setProperty("role", "ghost")
        login_link.clicked.connect(go_to_login)
        content.addWidget(login_link, alignment=Qt.AlignHCenter)

        self.on_register_success = on_register_success
        self.setLayout(centered(content))

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

        self.messageLabel.setProperty("role", "status-success")
        self.messageLabel.setText(f"✔ {message}")
        self._refresh_message_style()
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
        self._build_auth_stack()

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

    def _build_auth_stack(self):
        self.auth_stack = QStackedWidget()

        self.welcome_screen = WelcomeScreen(
            go_to_login=lambda: self.auth_stack.setCurrentWidget(self.login_screen),
            go_to_register=lambda: self.auth_stack.setCurrentWidget(self.register_screen),
        )
        self.login_screen = LoginScreen(
            on_login_success=self._handle_login_success,
            go_to_register=lambda: self.auth_stack.setCurrentWidget(self.register_screen),
        )
        self.register_screen = RegisterScreen(
            on_register_success=lambda: self.auth_stack.setCurrentWidget(self.login_screen),
            go_to_login=lambda: self.auth_stack.setCurrentWidget(self.login_screen),
        )

        self.auth_stack.addWidget(self.welcome_screen)
        self.auth_stack.addWidget(self.login_screen)
        self.auth_stack.addWidget(self.register_screen)
        self.auth_stack.setCurrentWidget(self.welcome_screen)
        self.setCentralWidget(self.auth_stack)

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
        self.sidebar.go_profile.connect(self._open_profile_page)
        self.sidebar.go_logout.connect(self._handle_logout)
        main_layout.addWidget(self.sidebar)

        self.content_stack = QStackedWidget()
        self.dashboard_screen = DashboardScreen()
        self.enrollment_panel = EnrollmentPanel()
        self.verification_panel = VerificationPanel()
        self.clients_page = ClientsPage()
        self.profile_page = ProfilePage()

        self.enrollment_panel.client_list_changed.connect(self._refresh_after_change)
        self.clients_page.client_selected.connect(self._open_client_in_enrollment)
        self.profile_page.profile_updated.connect(self._handle_profile_updated)

        self.content_stack.addWidget(self.dashboard_screen)
        self.content_stack.addWidget(self.enrollment_panel)
        self.content_stack.addWidget(self.verification_panel)
        self.content_stack.addWidget(self.clients_page)
        self.content_stack.addWidget(self.profile_page)
        self.content_stack.setCurrentWidget(self.dashboard_screen)

        main_layout.addWidget(self.content_stack)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _handle_login_success(self, employee: dict):
        self.current_employee = employee
        self._build_app_stack()
        self.sidebar.set_employee(employee["full_name"], employee["username"])
        self.dashboard_screen.set_employee_name(employee["full_name"])
        self.enrollment_panel.set_current_employee(employee["username"])
        self.verification_panel.set_current_employee(employee["username"])
        self.profile_page.load_employee(employee)
        self._refresh_after_change()

    def _handle_logout(self):
        self.current_employee = None
        self._build_auth_stack()

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

    def _open_client_in_enrollment(self, client_id: str):
        self.enrollment_panel.prefill_client(client_id)
        self.content_stack.setCurrentWidget(self.enrollment_panel)

    def _refresh_after_change(self):
        username = self.current_employee["username"]
        self.dashboard_screen.refresh(username)
        self.clients_page.refresh(username)


if __name__ == "__main__":
    from src.capture.pen_input import choose_pen_input
    choose_pen_input()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())