"""
dashboard.py — Combined Home + Stats screen (scoped to current employee)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame

import storage


class StatCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setProperty("role", "stat-card")
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(4)

        self.valueLabel = QLabel("—")
        self.valueLabel.setProperty("role", "stat-value")
        layout.addWidget(self.valueLabel)

        titleLabel = QLabel(title.upper())
        titleLabel.setProperty("role", "stat-title")
        layout.addWidget(titleLabel)

        self.setLayout(layout)

    def set_value(self, value):
        self.valueLabel.setText(str(value))


class DashboardScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.employee_username = None

        layout = QVBoxLayout()
        layout.setContentsMargins(36, 30, 36, 30)
        layout.setSpacing(6)

        self.welcomeLabel = QLabel("Welcome back")
        self.welcomeLabel.setProperty("role", "hero-title")
        layout.addWidget(self.welcomeLabel)

        subtitle = QLabel("Here's a snapshot of your signature verification activity")
        subtitle.setProperty("role", "muted")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        self.newClientsCard = StatCard("New clients this month")
        self.totalClientsCard = StatCard("Total enrolled clients")
        self.totalVerificationsCard = StatCard("Total verifications")
        self.genuineRateCard = StatCard("Genuine rate")

        for card in (self.newClientsCard, self.totalClientsCard,
                     self.totalVerificationsCard, self.genuineRateCard):
            cards_row.addWidget(card)

        layout.addLayout(cards_row)
        layout.addSpacing(20)

        recent_title = QLabel("RECENT VERIFICATION EVENTS")
        recent_title.setProperty("role", "section")
        layout.addWidget(recent_title)

        self.recentEventsLabel = QLabel("No verification events yet.")
        self.recentEventsLabel.setProperty("role", "muted")
        self.recentEventsLabel.setWordWrap(True)
        layout.addWidget(self.recentEventsLabel)

        layout.addStretch()
        self.setLayout(layout)

    def set_employee_name(self, full_name: str):
        self.welcomeLabel.setText(f"Welcome back, {full_name}")

    def refresh(self, employee_username: str):
        self.employee_username = employee_username

        clients = storage.list_clients(employee_username)
        stats = storage.get_verification_stats(employee_username)

        self.newClientsCard.set_value(storage.count_clients_enrolled_this_month(employee_username))
        self.totalClientsCard.set_value(len(clients))
        self.totalVerificationsCard.set_value(stats["total"])
        self.genuineRateCard.set_value(f"{stats['genuine_rate']}%")

        events = storage.get_verification_log(employee_username)
        if not events:
            self.recentEventsLabel.setText("No verification events yet.")
            return

        recent = events[-5:][::-1]
        lines = []
        for e in recent:
            decision = e.get("decision", "?")
            symbol = "✔" if decision == "Genuine" else "✘"
            name = e.get("client_display_name", "Unknown")
            confidence = e.get("confidence", "—")
            timestamp = e.get("timestamp", "—")
            lines.append(f"{symbol}  {name} — {decision} ({confidence}%)  ·  {timestamp}")
        self.recentEventsLabel.setText("\n".join(lines))