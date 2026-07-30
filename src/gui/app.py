"""
Main Application — Signature Verification System

PyQt5 desktop application entry point. Manages the main window,
routing between enrollment and verification loops, and modality
selection. No CLI execution for end users.
"""

# TODO: Implement MainWindow (QMainWindow)
# TODO: Implement navigation between enrollment and verification panels
# TODO: Implement modality selection (image, dynamic, both)
# TODO: Implement degraded-mode indicator for missing pressure
# TODO: Implement deterministic inference mode (model.eval(), frozen dropout/norm, PRNG seeds)
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signature Verification System")
        self.resize(800, 600)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())