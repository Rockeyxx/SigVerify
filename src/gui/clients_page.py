"""
clients_page.py — Table view of every client of the company, whoever enrolled them
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView
)

import src.gui.storage as storage


class ClientsPage(QWidget):
    client_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.employee_username = None

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(10)

        title = QLabel("Signers")
        title.setProperty("role", "title")
        layout.addWidget(title)

        subtitle = QLabel("Double-click a row to open that signer in Enrollment")
        subtitle.setProperty("role", "muted")
        subtitle.setWordWrap(True)   # it set the page's minimum width
        layout.addWidget(subtitle)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Signer Name", "Enrolled On", "References", "Last Verification"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def refresh(self, employee_username: str):
        self.employee_username = employee_username
        clients = storage.list_clients(None)
        self.table.setRowCount(len(clients))

        for row, client in enumerate(clients):
            display_name = client.get("display_name", client.get("key", "—"))
            enrolled_at = client.get("enrolled_at", "—")
            n_scan = len(client["spatial_sets"].get("scan", {}).get("embeddings", []))
            n_pen = len((client["temporal_set"] or {}).get("embeddings", []))

            last_event = storage.get_last_verification_for_client(display_name, None)
            last_verification = last_event.get("timestamp", "—") if last_event else "Never"

            self.table.setItem(row, 0, QTableWidgetItem(display_name))
            self.table.setItem(row, 1, QTableWidgetItem(enrolled_at))
            self.table.setItem(row, 2, QTableWidgetItem(f"scan {n_scan} · pen {n_pen}"))
            self.table.setItem(row, 3, QTableWidgetItem(last_verification))

    def _on_row_double_clicked(self, row, column):
        name_item = self.table.item(row, 0)
        if name_item:
            self.client_selected.emit(name_item.text())