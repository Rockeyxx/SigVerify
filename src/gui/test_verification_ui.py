import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic

class VerificationTest(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/verification_test.ui", self)

        # نربط الزر بدالة تشتغل لما نضغطه
        self.selectImageButton.clicked.connect(self.on_button_click)

    def on_button_click(self):
        self.resultLabel.setText("Button was clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VerificationTest()
    window.show()
    sys.exit(app.exec_())