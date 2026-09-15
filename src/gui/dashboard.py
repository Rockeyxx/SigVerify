"""
dashboard.py — Bento-style Home + Stats screen (scoped to current employee)
"""

import math
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame

import storage
from styles import ACCENT, BORDER, TEXT_LIGHT, TEXT_MUTED


class ProgressRing(QWidget):
    """دائرة تقدّم رسومية (Bento-style hero tile) — ترسم نفسها بـ QPainter"""
    def __init__(self):
        super().__init__()
        self.percentage = 0
        self.setFixedSize(140, 140)

    def set_percentage(self, value: float):
        self.percentage = max(0, min(100, value))
        self.update()  # يطلب إعادة الرسم

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(10, 10, self.width() - 20, self.height() - 20)

        # الحلقة الخلفية (رمادية خفيفة)
        bg_pen = QPen(QColor(BORDER))
        bg_pen.setWidth(10)
        bg_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # الحلقة الملوّنة (النسبة الفعلية)
        fg_pen = QPen(QColor(ACCENT))
        fg_pen.setWidth(10)
        fg_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(fg_pen)
        span = int(360 * (self.percentage / 100) * 16)
        painter.drawArc(rect, 90 * 16, -span)

        # النص بالمنتصف
        painter.setPen(QColor(TEXT_LIGHT))
        painter.setFont(QFont("Segoe UI", 18, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{int(self.percentage)}%")


class StatTile(QFrame):
    """بلاطة إحصائية صغيرة (Bento tile)"""
    def __init__(self, title: str, icon_text: str = ""):
        super().__init__()
        self.setProperty("role", "stat-card")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        if icon_text:
            icon_label = QLabel(icon_text)
            icon_label.setStyleSheet(f"font-size: 16px; color: {ACCENT};")
            layout.addWidget(icon_label)

        self.valueLabel = QLabel("—")
        self.valueLabel.setProperty("role", "stat-value")
        layout.addWidget(self.valueLabel)

        title_label = QLabel(title.upper())
        title_label.setProperty("role", "stat-title")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        self.setLayout(layout)

    def set_value(self, value):
        self.valueLabel.setText(str(value))


class RecentEventsTile(QFrame):
    """بلاطة عريضة تعرض آخر عمليات التحقق"""
    def __init__(self):
        super().__init__()
        self.setProperty("role", "stat-card")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(6)

        title = QLabel("RECENT VERIFICATIONS")
        title.setProperty("role", "stat-title")
        self.layout.addWidget(title)

        self.rows_container = QVBoxLayout()
        self.layout.addLayout(self.rows_container)
        self.setLayout(self.layout)

    def set_events(self, events: list):
        # نفضي أي صفوف قديمة
        while self.rows_container.count():
            item = self.rows_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not events:
            empty = QLabel("No verification events yet.")
            empty.setProperty("role", "muted")
            self.rows_container.addWidget(empty)
            return

        for e in events[-4:][::-1]:
            row = QHBoxLayout()
            decision = e.get("decision", "?")
            name = e.get("client_display_name", "Unknown")
            confidence = e.get("confidence", "—")

            name_label = QLabel(name)
            name_label.setStyleSheet("font-size: 12px;")
            row.addWidget(name_label)
            row.addStretch()

            role = "status-success" if decision == "Genuine" else "status-error"
            result_label = QLabel(f"{decision} · {confidence}%")
            result_label.setProperty("role", role)
            result_label.setStyleSheet(result_label.styleSheet() + "font-size: 12px;")
            row.addWidget(result_label)

            row_widget = QWidget()
            row_widget.setLayout(row)
            self.rows_container.addWidget(row_widget)


class DashboardScreen(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(36, 30, 36, 30)
        layout.setSpacing(6)

        self.welcomeLabel = QLabel("Welcome back")
        self.welcomeLabel.setProperty("role", "hero-title")
        layout.addWidget(self.welcomeLabel)

        subtitle = QLabel("Here's a snapshot of your signature verification activity")
        subtitle.setProperty("role", "muted")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # ---- شبكة Bento ----
        bento = QGridLayout()
        bento.setSpacing(14)

        # البلاطة الكبيرة: دائرة التقدّم (تمتد صفين)
        ring_tile = QFrame()
        ring_tile.setProperty("role", "stat-card")
        ring_layout = QVBoxLayout()
        ring_layout.setAlignment(Qt.AlignCenter)
        ring_layout.setContentsMargins(16, 20, 16, 20)
        self.progressRing = ProgressRing()
        ring_layout.addWidget(self.progressRing, alignment=Qt.AlignCenter)
        ring_caption = QLabel("GENUINE RATE")
        ring_caption.setProperty("role", "stat-title")
        ring_caption.setAlignment(Qt.AlignCenter)
        ring_layout.addWidget(ring_caption)
        ring_tile.setLayout(ring_layout)

        self.newClientsTile = StatTile("New clients this month")
        self.totalClientsTile = StatTile("Total enrolled clients")
        self.totalVerificationsTile = StatTile("Total verifications")
        self.recentTile = RecentEventsTile()

        bento.addWidget(ring_tile, 0, 0, 2, 1)
        bento.addWidget(self.newClientsTile, 0, 1)
        bento.addWidget(self.totalClientsTile, 0, 2)
        bento.addWidget(self.totalVerificationsTile, 1, 1)
        bento.addWidget(self.recentTile, 1, 2)

        bento.setColumnStretch(0, 1)
        bento.setColumnStretch(1, 1)
        bento.setColumnStretch(2, 1)

        layout.addLayout(bento)
        layout.addStretch()
        self.setLayout(layout)

    def set_employee_name(self, full_name: str):
        self.welcomeLabel.setText(f"Welcome back, {full_name}")

    def refresh(self, employee_username: str):
        clients = storage.list_clients(employee_username)
        stats = storage.get_verification_stats(employee_username)

        self.newClientsTile.set_value(storage.count_clients_enrolled_this_month(employee_username))
        self.totalClientsTile.set_value(len(clients))
        self.totalVerificationsTile.set_value(stats["total"])
        self.progressRing.set_percentage(stats["genuine_rate"])

        events = storage.get_verification_log(employee_username)
        self.recentTile.set_events(events)