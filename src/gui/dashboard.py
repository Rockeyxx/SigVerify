"""
dashboard.py — Landing screen after employee login
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

import storage


class DashboardScreen(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("Dashboard")
        title.setProperty("role", "title")
        layout.addWidget(title)

        self.statsLabel = QLabel("")
        self.statsLabel.setProperty("role", "muted")
        layout.addWidget(self.statsLabel)

        layout.addStretch()
        self.setLayout(layout)

    def refresh(self):
        clients = storage.list_clients()
        events = storage.get_verification_log()
        self.statsLabel.setText(
            f"Enrolled clients: {len(clients)}\n"
            f"Verification events logged: {len(events)}"
        )