"""Main application entry point for SigVerify."""

# Load torch before PyQt5 to avoid Windows DLL search-path conflicts.
import torch  # noqa: F401 (imported here only to force early DLL init)
import cv2    # noqa: F401

import os
import sys
import traceback
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMessageBox, QFrame, QScrollArea,
    QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QLineEdit, QComboBox
)

from src.gui.enrollment_page import EnrollmentPanel
from src.gui.verification_page import VerificationPanel
from src.gui.sidebar import Sidebar
from src.gui.dashboard_page import DashboardScreen
from src.gui.clients_page import ClientsPage
from src.gui.employees_page import EmployeesPage, restore_from_backup
from src.gui.password_field import PasswordField, new_password_helpers
from src.gui.profile_page import ProfilePage
from src.gui.styles import STYLE
from src.gui.workers import LoginWorker, WarmUpWorker, wait_for_workers
import src.gui.storage as storage
from src.app_version import app_version

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "assets", "signature_icon.png")
STORAGE_ERROR_TITLE = "Cannot open the signature data"
RESTORED_NOTICE = "Backup restored. Log in again."


def show_unexpected_error(exc_type, exc, tb):
    """sys.excepthook: an error no page caught is printed and shown, instead of PyQt5 aborting."""
    traceback.print_exception(exc_type, exc, tb)
    QMessageBox.critical(None, "Unexpected error",
                         f"Something went wrong: {exc}. The action was not completed.")


def install_error_dialog():
    sys.excepthook = show_unexpected_error


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


def centered(content_layout, max_width=None):
    """Centres a column. With max_width the column fills the window up to that width, so it
    shrinks in a narrow window."""
    outer = QVBoxLayout()
    outer.addStretch(1)
    row = QHBoxLayout()
    row.addStretch(1)
    if max_width:
        column = QWidget()
        content_layout.setContentsMargins(0, 0, 0, 0)
        column.setLayout(content_layout)
        column.setMaximumWidth(max_width)
        row.addWidget(column, 100)   # takes the space first, the stretches get what is left
    else:
        row.addLayout(content_layout)
    row.addStretch(1)
    outer.addLayout(row)
    outer.addStretch(1)
    return outer


# Widest the login and create-vault forms get: the longest placeholder fits at the 15 px base text
FORM_WIDTH = 420


class WelcomeScreen(QWidget):
    def __init__(self, go_to_login):
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
        login_button.clicked.connect(go_to_login)
        buttons_row.addWidget(login_button)

        content.addLayout(buttons_row)
        self.setLayout(centered(content))


class LoginScreen(QWidget):
    def __init__(self, on_login_success, go_register):
        super().__init__()
        content = QVBoxLayout()
        content.setSpacing(14)

        title = QLabel("Login")
        title.setProperty("role", "hero-title")
        title.setAlignment(Qt.AlignHCenter)
        content.addWidget(title)
        content.addSpacing(10)

        self.companyInput = QComboBox()            # the last vault used is preselected
        self.companyInput.addItems(storage.list_companies())
        self.companyInput.setCurrentText(storage.current_company() or "")
        self.companyInput.setHidden(self.companyInput.count() == 1)   # one vault: nothing to pick
        content.addWidget(self.companyInput)

        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("User ID")
        content.addWidget(self.usernameInput)

        self.passwordInput = PasswordField("Password")
        content.addWidget(self.passwordInput)

        self.errorLabel = QLabel("")
        self.errorLabel.setWordWrap(True)
        self.errorLabel.setProperty("role", "status-error")
        self.errorLabel.setAlignment(Qt.AlignHCenter)
        content.addWidget(self.errorLabel)

        self.loginButton = QPushButton("Login")
        self.loginButton.setProperty("role", "primary")
        self.loginButton.setFixedHeight(44)
        self.loginButton.clicked.connect(self._handle_login)
        content.addWidget(self.loginButton)
        self._worker = None

        self.registerButton = QPushButton("Create a new vault")
        self.registerButton.setProperty("role", "ghost")
        self.registerButton.clicked.connect(go_register)
        content.addWidget(self.registerButton, alignment=Qt.AlignHCenter)

        version = QLabel(f"Version {app_version()}")
        version.setProperty("role", "muted")
        version.setAlignment(Qt.AlignHCenter)
        content.addWidget(version)

        self.on_login_success = on_login_success
        self.setLayout(centered(content, FORM_WIDTH))

    def _handle_login(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self.loginButton.setEnabled(False)
        self.loginButton.setText("Logging in...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._worker = LoginWorker(self.companyInput.currentText(), self.usernameInput.text().strip(),
                                   self.passwordInput.text())
        self._worker.finished.connect(self._login_done)
        self._worker.failed.connect(self._login_failed)
        self._worker.start()

    def _end_busy(self):
        QApplication.restoreOverrideCursor()
        self.loginButton.setText("Login")
        self.loginButton.setEnabled(True)

    def _login_failed(self, message):
        self._end_busy()
        QMessageBox.critical(self, STORAGE_ERROR_TITLE, message)

    def _login_done(self, success, employee, message):
        self._end_busy()
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
    """Creates a new vault (its own encrypted data; the person registering is its admin),
    or makes one from a backup."""

    def __init__(self, on_register_success, on_restored, go_back=None):
        super().__init__()
        content = QVBoxLayout()
        content.setSpacing(14)

        title = QLabel("Create a vault")
        title.setProperty("role", "hero-title")
        title.setAlignment(Qt.AlignHCenter)
        content.addWidget(title)

        warning = QLabel(storage.ADMIN_WARNING)
        warning.setProperty("role", "warning")
        warning.setWordWrap(True)
        warning.setAlignment(Qt.AlignHCenter)
        content.addWidget(warning)
        content.addSpacing(10)

        self.companyInput = QLineEdit()
        self.companyInput.setPlaceholderText("Vault name (your company, branch, or your own name)")
        content.addWidget(self.companyInput)

        self.fullNameInput = QLineEdit()
        self.fullNameInput.setPlaceholderText("Full Name")
        content.addWidget(self.fullNameInput)
        self._prefilled = ""                       # the vault name follows Full Name until the user edits it
        self.fullNameInput.textChanged.connect(self._prefill_vault_name)

        self.usernameInput = QLineEdit()
        self.usernameInput.setPlaceholderText("User ID")
        content.addWidget(self.usernameInput)

        self.passwordInput = PasswordField("Password")
        content.addWidget(self.passwordInput)

        self.confirmPasswordInput = PasswordField("Confirm Password")
        content.addWidget(self.confirmPasswordInput)

        generate, hint = new_password_helpers(self.passwordInput, self.confirmPasswordInput)
        hint.setAlignment(Qt.AlignHCenter)
        content.addWidget(hint)
        content.addWidget(generate, alignment=Qt.AlignHCenter)

        self.messageLabel = QLabel("")
        self.messageLabel.setAlignment(Qt.AlignHCenter)
        self.messageLabel.setWordWrap(True)
        content.addWidget(self.messageLabel)

        self.createButton = QPushButton("Create Vault")
        self.createButton.setProperty("role", "primary")
        self.createButton.setFixedHeight(44)
        self.createButton.clicked.connect(self._handle_register)
        content.addWidget(self.createButton)

        self.restoreButton = QPushButton("Restore a vault from a backup")
        self.restoreButton.setProperty("role", "ghost")
        self.restoreButton.clicked.connect(self._handle_restore)
        content.addWidget(self.restoreButton, alignment=Qt.AlignHCenter)

        if go_back is not None:                    # not on a first run: there is no login yet
            self.backButton = QPushButton("Back to login")
            self.backButton.setProperty("role", "ghost")
            self.backButton.clicked.connect(go_back)
            content.addWidget(self.backButton, alignment=Qt.AlignHCenter)

        self.on_register_success = on_register_success
        self.on_restored = on_restored
        self.setLayout(centered(content, FORM_WIDTH))

    def _prefill_vault_name(self, full_name):
        if self.companyInput.text() == self._prefilled:
            self.companyInput.setText(full_name)
        self._prefilled = full_name

    def _handle_restore(self):
        outcome = restore_from_backup(self, new_company=True)
        if outcome is None:
            return
        ok, message = outcome
        if not ok:
            self._set_error(message)
            return
        self.on_restored()

    def _handle_register(self):
        company = self.companyInput.text().strip()
        full_name = self.fullNameInput.text().strip()
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()
        confirm = self.confirmPasswordInput.text()

        if not company or not full_name or not username or not password:
            self._set_error("Please fill in all fields")
            return
        if password != confirm:
            self._set_error("Passwords do not match")
            return

        success, message = storage.create_company(company, full_name, username, password)
        if not success:
            self._set_error(message)
            return

        self.companyInput.clear()
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


class PageStack(QStackedWidget):
    """Scroll-wrapped pages so small viewports (e.g. 1366x768) never force window growth."""

    def add_page(self, page):
        area = QScrollArea()
        area.setWidgetResizable(True)          # the page fills the area and scrolls only when it must
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(page)
        self.addWidget(area)

    def show_page(self, page):
        self.setCurrentWidget(self.area_of(page))

    def current_page(self):
        return self.currentWidget().widget()

    def area_of(self, page):
        for i in range(self.count()):
            if self.widget(i).widget() is page:
                return self.widget(i)
        raise ValueError("page is not in this stack")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signature Verification System")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))   # pages scroll when it is small
        self.current_employee = None
        self.startup_failed = False
        try:
            first_run = storage.is_first_run()
        except storage.StorageError as e:   # the vault folder cannot be read
            self.startup_failed = True
            QMessageBox.critical(None, STORAGE_ERROR_TITLE, str(e))
            QTimer.singleShot(0, QApplication.instance().quit)
            return
        self._build_auth_stack(first_run)

        # Warm up on a background thread at launch; keep reference on self to avoid GC.
        self._warm_up_worker = WarmUpWorker()
        self._warm_up_worker.start()

    def closeEvent(self, event):
        wait_for_workers()
        super().closeEvent(event)

    def _build_auth_stack(self, first_run=False):
        """First run (no company yet): Create a vault. Afterwards: Welcome, then Login,
        which also leads to Create a new vault."""
        self.auth_stack = QStackedWidget()
        show_login = lambda: self.auth_stack.setCurrentWidget(self.login_screen)  # noqa: E731
        self.login_screen = LoginScreen(
            on_login_success=self._handle_login_success,
            go_register=lambda: self.auth_stack.setCurrentWidget(self.register_screen),
        )
        self.register_screen = RegisterScreen(
            on_register_success=lambda: self._handle_login_success(storage.current_employee()),
            on_restored=self._handle_restored,
            go_back=None if first_run else show_login,
        )
        screens = [self.register_screen, self.login_screen]
        if not first_run:
            screens.insert(0, WelcomeScreen(go_to_login=show_login))
        for screen in screens:
            self.auth_stack.addWidget(screen)
        self.auth_stack.setCurrentWidget(screens[0])
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
        self.sidebar.go_dashboard.connect(lambda: self.content_stack.show_page(self.dashboard_screen))
        self.sidebar.go_enrollment.connect(lambda: self.content_stack.show_page(self.enrollment_panel))
        self.sidebar.go_verification.connect(lambda: self.content_stack.show_page(self.verification_panel))
        self.sidebar.go_clients.connect(self._open_clients_page)
        self.sidebar.go_employees.connect(self._open_employees_page)
        self.sidebar.go_profile.connect(self._open_profile_page)
        self.sidebar.go_logout.connect(self._handle_logout)
        main_layout.addWidget(self.sidebar)

        self.content_stack = PageStack()
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

        for page in (self.dashboard_screen, self.enrollment_panel, self.verification_panel,
                     self.clients_page, self.profile_page, self.employees_page):
            self.content_stack.add_page(page)
        self.content_stack.show_page(self.dashboard_screen)

        main_layout.addWidget(self.content_stack)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _handle_login_success(self, employee: dict):
        self.current_employee = employee
        self._build_app_stack()
        self.sidebar.set_employee(employee["full_name"], employee["username"])
        self.sidebar.set_company(storage.current_company() or "")
        self.sidebar.set_admin(employee["is_admin"])
        self.dashboard_screen.set_employee_name(employee["full_name"])
        self.enrollment_panel.set_current_employee(employee["username"], employee["is_admin"])
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
        self.content_stack.show_page(self.employees_page)

    def _open_clients_page(self):
        self.clients_page.refresh(self.current_employee["username"])
        self.content_stack.show_page(self.clients_page)

    def _open_profile_page(self):
        self.content_stack.show_page(self.profile_page)

    def _handle_profile_updated(self, updated_employee: dict):
        self.current_employee = updated_employee
        self.sidebar.set_employee(updated_employee["full_name"], updated_employee["username"])
        self.dashboard_screen.set_employee_name(updated_employee["full_name"])
        self.enrollment_panel.set_current_employee(updated_employee["username"], updated_employee["is_admin"])
        self.verification_panel.set_current_employee(updated_employee["username"])
        self.employees_page.set_current_employee(updated_employee["username"])

    def _open_client_in_enrollment(self, client_id: str):
        self.enrollment_panel.prefill_client(client_id)
        self.content_stack.show_page(self.enrollment_panel)

    def _refresh_after_change(self):
        username = self.current_employee["username"]
        self.dashboard_screen.refresh(username)
        self.clients_page.refresh(username)


def create_application(argv):
    """The QApplication as the app starts it: display scaling, icon and theme. The self-test
    (src/self_test.py) starts Qt through this too, so test mode looks like the app."""
    # Qt5 draws at 100% unless told to follow the display scaling (Windows laptops run 125-200%)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(argv)
    app.setWindowIcon(QIcon(ICON_PATH))   # the title bar of every window, dialogs included
    app.setStyleSheet(STYLE)
    return app


def main():
    from src.capture.pen_input import choose_pen_input
    choose_pen_input()
    if sys.platform == "win32":
        # Without its own ID the app is grouped under python.exe, and the taskbar shows Python's icon.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("EE499.Team06.SignatureVerification")
    app = create_application(sys.argv)
    install_error_dialog()
    window = MainWindow()
    if not window.startup_failed:
        window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()