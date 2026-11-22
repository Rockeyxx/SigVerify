"""
sidebar.py — Collapsible persistent left sidebar
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton


class Sidebar(QWidget):
    go_dashboard = pyqtSignal()
    go_enrollment = pyqtSignal()
    go_verification = pyqtSignal()
    go_clients = pyqtSignal()
    go_employees = pyqtSignal()
    go_profile = pyqtSignal()
    go_logout = pyqtSignal()

    EXPANDED_WIDTH = 220
    COLLAPSED_WIDTH = 56

    def __init__(self):
        super().__init__()
        self.setProperty("role", "sidebar")
        self.setAttribute(Qt.WA_StyledBackground)   # a QWidget subclass ignores a stylesheet background without it
        self.is_collapsed = False
        self.is_admin = False

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 16, 10, 16)
        self.main_layout.setSpacing(6)

        # زر الطي/الفتح دائمًا ظاهر
        toggle_row = QHBoxLayout()
        self.toggleButton = QPushButton("☰")
        self.toggleButton.setFixedWidth(32)
        self.toggleButton.setProperty("role", "sidebar-toggle")
        self.toggleButton.clicked.connect(self.toggle_collapse)
        toggle_row.addWidget(self.toggleButton)
        toggle_row.addStretch()
        self.main_layout.addLayout(toggle_row)

        self.main_layout.addSpacing(10)

        # العناصر اللي تختفي وقت الطي
        self.employeeNameLabel = QLabel("—")
        self.employeeNameLabel.setProperty("role", "sidebar-name")
        self.main_layout.addWidget(self.employeeNameLabel)

        self.employeeUsernameLabel = QLabel("—")
        self.employeeUsernameLabel.setProperty("role", "muted")
        self.main_layout.addWidget(self.employeeUsernameLabel)

        self.companyLabel = QLabel("")
        self.companyLabel.setProperty("role", "muted")
        self.companyLabel.setWordWrap(True)
        self.main_layout.addWidget(self.companyLabel)

        self.main_layout.addSpacing(16)

        self.nav_buttons = []
        self._add_nav_button("Home", self.go_dashboard)
        self._add_nav_button("Enrollment", self.go_enrollment)
        self._add_nav_button("Verification", self.go_verification)
        self._add_nav_button("Signers", self.go_clients)
        self.employeesButton = self._add_nav_button("Members", self.go_employees)
        self._add_nav_button("Profile", self.go_profile)
        self.employeesButton.setVisible(False)   # admin only: set_admin(True) shows it

        self.main_layout.addStretch()

        self.logoutButton = QPushButton("Logout")
        self.logoutButton.setProperty("role", "sidebar-nav-danger")
        self.logoutButton.clicked.connect(self.go_logout.emit)
        self.main_layout.addWidget(self.logoutButton)

        self.setLayout(self.main_layout)
        self.setFixedWidth(self.EXPANDED_WIDTH)

        # كل العناصر اللي تختفي وقت الطي (ما عدا زر التوجل نفسه)
        self.collapsible_widgets = [
            self.employeeNameLabel, self.employeeUsernameLabel, self.companyLabel,
            self.logoutButton
        ] + self.nav_buttons

    def _add_nav_button(self, text, signal):
        btn = QPushButton(text)
        btn.setProperty("role", "sidebar-nav")
        btn.clicked.connect(signal.emit)
        self.main_layout.addWidget(btn)
        self.nav_buttons.append(btn)
        return btn

    def toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        if self.is_collapsed:
            self.setFixedWidth(self.COLLAPSED_WIDTH)
            for w in self.collapsible_widgets:
                w.setVisible(False)
        else:
            self.setFixedWidth(self.EXPANDED_WIDTH)
            for w in self.collapsible_widgets:
                w.setVisible(True)
            self.employeesButton.setVisible(self.is_admin)

    def set_admin(self, is_admin: bool):
        """Show the Employees button to the admin only."""
        self.is_admin = bool(is_admin)
        self.employeesButton.setVisible(self.is_admin and not self.is_collapsed)

    def set_employee(self, full_name: str, username: str):
        self.employeeNameLabel.setText(full_name)
        self.employeeUsernameLabel.setText(f"@{username}")

    def set_company(self, name: str):
        self.companyLabel.setText(name)
