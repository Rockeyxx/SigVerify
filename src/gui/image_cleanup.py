"""
image_cleanup.py — Signature Verification System

Automatic ink-cropping and binarization for phone-photo signature
images, applied transparently before every enrollment/verification
call. Ported directly from tools/verify_my_signature.py's
crop_to_ink() — same logic, same reasoning, applied automatically
inside the GUI instead of requiring a manual CLI step.
"""

import os
import tempfile
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps


def crop_to_ink(path: str) -> str:
    """
    Turn a phone photo into something resembling the scans the model
    trained on: fix EXIF rotation, crop to the ink, and binarize.

    Returns the path to a new temp file. The original file is never
    modified. If ink cannot be detected, returns the original path
    unchanged (lets the real quality gate reject it with a clear
    message instead of failing here).
    """
    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grey_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ink = ((hsv[:, :, 1] > 60) & (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 160)).astype(np.uint8)

        if ink.sum() < 800:
            bright = cv2.threshold(grey_full, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
            if count < 2:
                return path  # let the real quality gate reject it
            page = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = cv2.erode((labels == page).astype(np.uint8),
                             np.ones((15, 15), np.uint8))
            paper = float(np.median(grey_full[mask > 0]))
            spread = float(grey_full[mask > 0].std())
            ink = ((mask > 0) & (grey_full < paper - 1.5 * spread)).astype(np.uint8)

        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

        count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 800]
        if not blobs:
            return path  # let the real quality gate reject it

        left = min(stats[i, cv2.CC_STAT_LEFT] for i in blobs)
        top = min(stats[i, cv2.CC_STAT_TOP] for i in blobs)
        right = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in blobs)
        bottom = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in blobs)
        pad = int(0.08 * max(right - left, bottom - top))

        grey = grey_full[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad]
        binary = cv2.threshold(cv2.GaussianBlur(grey, (5, 5), 0), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        destination = os.path.join(tempfile.gettempdir(), f"cropped_{uuid.uuid4().hex}.png")
        cv2.imwrite(destination, binary)
        return destination

    except Exception:
        # Any failure in cleanup falls back to the original file — the real
        # quality gate downstream will give a proper error message.
        return path