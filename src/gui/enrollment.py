"""
Enrollment Panel — Signature Verification System

PyQt5 enrollment loop UI. Supports few-shot reference registration
for both modalities (image upload and/or dynamic tablet capture).
Stores generated embeddings only — no backpropagation at enrollment.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFileDialog, QMessageBox
)


class EnrollmentPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None

        layout = QVBoxLayout()

        title = QLabel("Enrollment")
        layout.addWidget(title)

        layout.addWidget(QLabel("User ID:"))
        self.userIdInput = QLineEdit()
        layout.addWidget(self.userIdInput)

        self.selectImageButton = QPushButton("Select Reference Image")
        self.selectImageButton.clicked.connect(self.select_reference_image)
        layout.addWidget(self.selectImageButton)

        self.imageFileLabel = QLabel("No file selected")
        layout.addWidget(self.imageFileLabel)

        self.enrollButton = QPushButton("Enroll")
        self.enrollButton.clicked.connect(self.run_enrollment)
        layout.addWidget(self.enrollButton)

        self.statusLabel = QLabel("")
        layout.addWidget(self.statusLabel)

        self.setLayout(layout)

    def select_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Signature", "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.selected_image_path = file_path
            self.imageFileLabel.setText(file_path.split("/")[-1])
        else:
            self.imageFileLabel.setText("No file selected")

    def run_enrollment(self):
        user_id = self.userIdInput.text()

        if not user_id:
            self.statusLabel.setText("Please enter a User ID")
            return
        if not self.selected_image_path:
            self.statusLabel.setText("Please select a reference image")
            return

        # TODO: لاحقًا يستدعي الموديل الحقيقي لاستخراج الـ embedding
        # ويخزنه بقاعدة البيانات المشفرة عبر فريق الـ DB
        self.statusLabel.setText(f"Enrolled user '{user_id}' successfully (mock)")