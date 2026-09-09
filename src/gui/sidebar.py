"""
sidebar.py — Persistent left sidebar shown after employee login
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget

import storage


class Sidebar(QWidget):
    go_dashboard = pyqtSignal()
    go_enrollment = pyqtSignal()
    go_verification = pyqtSignal()
    go_logout = pyqtSignal()
    client_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        self.setProperty("role", "sidebar")

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(8)

        self.employeeNameLabel = QLabel("—")
        self.employeeNameLabel.setProperty("role", "sidebar-name")
        layout.addWidget(self.employeeNameLabel)

        self.employeeUsernameLabel = QLabel("—")
        self.employeeUsernameLabel.setProperty("role", "muted")
        layout.addWidget(self.employeeUsernameLabel)

        layout.addSpacing(16)

        dash_btn = QPushButton("Dashboard")
        dash_btn.setProperty("role", "sidebar-nav")
        dash_btn.clicked.connect(self.go_dashboard.emit)
        layout.addWidget(dash_btn)

        enroll_btn = QPushButton("Enrollment")
        enroll_btn.setProperty("role", "sidebar-nav")
        enroll_btn.clicked.connect(self.go_enrollment.emit)
        layout.addWidget(enroll_btn)

        verify_btn = QPushButton("Verification")
        verify_btn.setProperty("role", "sidebar-nav")
        verify_btn.clicked.connect(self.go_verification.emit)
        layout.addWidget(verify_btn)

        layout.addSpacing(16)

        clients_label = QLabel("ENROLLED CLIENTS")
        clients_label.setProperty("role", "section")
        layout.addWidget(clients_label)

        self.clientList = QListWidget()
        self.clientList.itemClicked.connect(self._on_client_clicked)
        layout.addWidget(self.clientList)

        layout.addStretch()

        logout_btn = QPushButton("Logout")
        logout_btn.setProperty("role", "sidebar-nav-danger")
        logout_btn.clicked.connect(self.go_logout.emit)
        layout.addWidget(logout_btn)

        self.setLayout(layout)

    def set_employee(self, full_name: str, username: str):
        self.employeeNameLabel.setText(full_name)
        self.employeeUsernameLabel.setText(f"@{username}")

    def refresh_clients(self):
        self.clientList.clear()
        for client_id in storage.list_clients():
            self.clientList.addItem(client_id)

    def _on_client_clicked(self, item):
        self.client_selected.emit(item.text())