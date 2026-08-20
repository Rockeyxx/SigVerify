

# TODO: Implement MainWindow (QMainWindow)
# TODO: Implement navigation between enrollment and verification panels
# TODO: Implement modality selection (image, dynamic, both)
# TODO: Implement degraded-mode indicator for missing pressure
# TODO: Implement deterministic inference mode (model.eval(), frozen dropout/norm, PRNG seeds)

"""
Main Application — Signature Verification System

PyQt5 desktop application entry point. Manages the main window,
routing between enrollment and verification loops, and modality
selection. No CLI execution for end users.
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QStackedWidget
)

from enrollment import EnrollmentPanel
from verification import VerificationPanel


class HomeScreen(QWidget):
    """الشاشة الأولى: زرين للاختيار بين Enrollment و Verification"""
    def __init__(self, go_to_enrollment, go_to_verification):
        super().__init__()
        layout = QVBoxLayout()

        enrollment_button = QPushButton("Enrollment")
        verification_button = QPushButton("Verification")

        # نربط كل زر بدالة "الانتقال" اللي بنستقبلها من MainWindow
        enrollment_button.clicked.connect(go_to_enrollment)
        verification_button.clicked.connect(go_to_verification)

        layout.addWidget(enrollment_button)
        layout.addWidget(verification_button)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signature Verification System")
        self.resize(800, 600)

        # الحاوية الرئيسية اللي تحتوي زر الرجوع + الكومة
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # زر الرجوع (نخفيه بالبداية لأننا بشاشة Home أصلاً)
        self.back_button = QPushButton("← Back to Home")
        self.back_button.clicked.connect(self.show_home)
        self.back_button.setVisible(False)  # مخفي افتراضيًا
        main_layout.addWidget(self.back_button)

        # الكومة (كل الشاشات)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # نجهز كل الشاشات
        self.home_screen = HomeScreen(
            go_to_enrollment=self.show_enrollment,
            go_to_verification=self.show_verification
        )
        self.enrollment_panel = EnrollmentPanel()
        self.verification_panel = VerificationPanel()

        self.stack.addWidget(self.home_screen)
        self.stack.addWidget(self.enrollment_panel)
        self.stack.addWidget(self.verification_panel)

        self.stack.setCurrentWidget(self.home_screen)

    def show_enrollment(self):
        self.stack.setCurrentWidget(self.enrollment_panel)
        self.back_button.setVisible(True)

    def show_verification(self):
        self.stack.setCurrentWidget(self.verification_panel)
        self.back_button.setVisible(True)

    def show_home(self):
        self.stack.setCurrentWidget(self.home_screen)
        self.back_button.setVisible(False)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())