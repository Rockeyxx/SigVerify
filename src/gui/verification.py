"""
Verification Panel — Signature Verification System

PyQt5 verification loop UI. Accepts a probe (image, dynamic, or both),
compares against stored reference embeddings, and displays the result
as [Genuine/Forged, Confidence %].
"""

import random
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5 import uic


class VerificationPanel(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/verification_screen.ui", self)

        # نخزن مسار الصورة المختارة هنا مؤقتًا
        self.selected_image_path = None

        # نربط الأزرار بالدوال
        self.selectProbeButton.clicked.connect(self.select_probe_image)
        self.verifyButton.clicked.connect(self.run_verification)

        # نخفي مؤشر degraded-mode افتراضيًا (فاضي لين يصير فعلي)
        self.degradedModeIndicator.setText("⚠ Degraded Mode: Spatial Only (no dynamic input device connected)")

    def select_probe_image(self):
        """يفتح نافذة اختيار ملف حقيقية ويعرض اسم الملف المختار"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Signature Image",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.selected_image_path = file_path
            # نعرض بس اسم الملف، مو المسار الكامل (أوضح للمستخدم)
            file_name = file_path.split("/")[-1]
            self.probeFileLabel.setText(file_name)
        else:
            self.probeFileLabel.setText("No file selected")

    def run_verification(self):
        """
        TODO (لاحقًا): استدعاء الموديل الحقيقي من فريق الـ ML
        حاليًا: نتيجة وهمية للتجربة فقط
        """
        user_id = self.userIdInput.text()

        if not user_id:
            self.resultDisplay.setText("Please enter a User ID")
            return

        if not self.selected_image_path:
            self.resultDisplay.setText("Please select a probe image")
            return

        # ------ نتيجة تجريبية وهمية (Mock) ------
        # لاحقًا هذا الجزء بيستبدل بالاستدعاء الحقيقي لموديل الـ ML
        decision = random.choice(["Genuine", "Forged"])
        confidence = round(random.uniform(60, 99), 1)

        self.resultDisplay.setText(f"[{decision}, {confidence}%]")