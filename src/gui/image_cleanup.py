"""
image_cleanup.py — Signature Verification System

Automatic ink-cropping and binarization for phone-photo signature
images, applied transparently before every enrollment/verification
call. Ported from tools/verify_my_signature.py's crop_to_ink(), with
safeguards added after a blank page was found to pass through as a
false-positive "signature" (PR review point 3), plus an upscaling
guarantee added after discovering the model's own quality_gate
(src/data/dataset.py) measures the INK'S OWN bounding box — not the
overall image size — against MIN_BBOX_PIXELS. Padding the crop with
extra blank paper does nothing for that check, since quality_gate
re-detects the ink itself and ignores surrounding margin; the only
way to satisfy it for a naturally small/flat signature is to upscale
the ink along with the rest of the image.
"""

import os
import tempfile
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

from config.settings import MIN_BBOX_PIXELS


MIN_BLOB_AREA_PX = 800
MIN_ABSOLUTE_CONTRAST = 40
MIN_INK_RATIO = 0.003
MAX_INK_RATIO = 0.35

# Safety margin above the model's own hard minimum (quality_gate checks
# min(bbox_w, bbox_h) >= MIN_BBOX_PIXELS on ITS OWN re-detected ink box).
TARGET_INK_EDGE_PX = MIN_BBOX_PIXELS + 10
MAX_UPSCALE_FACTOR = 6.0   # refuse rather than blow up a tiny/noisy blob absurdly


class SignatureNotFoundError(ValueError):
    """
    Raised when crop_to_ink cannot locate a real signature in the photo,
    or when the signature is too small/low-resolution in the source
    photo to be upscaled to a size the model's quality_gate will accept.
    """
    pass


def _illumination_normalized_contrast(grey: np.ndarray) -> np.ndarray:
    sigma = max(grey.shape) / 20
    background = cv2.GaussianBlur(grey, (0, 0), sigmaX=sigma)
    contrast = background.astype(np.int16) - grey.astype(np.int16)
    return np.clip(contrast, 0, 255).astype(np.uint8)


def crop_to_ink(path: str) -> str:
    """
    Turn a phone photo into something resembling the scans the model
    trained on: fix EXIF rotation, crop to the ink, binarize, and
    upscale if the ink itself is smaller than the model's quality gate
    requires.

    Returns the path to a new temp file. The original file is never
    modified.

    Raises SignatureNotFoundError if no real signature can be located,
    if the ink-to-paper ratio doesn't resemble a signature, or if the
    ink is so small in the source photo that no reasonable upscale can
    reach the model's minimum accepted size.
    """
    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grey_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = grey_full.shape

        ink = ((hsv[:, :, 1] > 60) & (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 160)).astype(np.uint8)

        if ink.sum() < MIN_BLOB_AREA_PX:
            contrast = _illumination_normalized_contrast(grey_full)
            ink = (contrast >= MIN_ABSOLUTE_CONTRAST).astype(np.uint8)

        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > MIN_BLOB_AREA_PX]
        if not blobs:
            raise SignatureNotFoundError(
                "Could not find a signature in this photo. "
                "Retake it on plain paper with even lighting."
            )

        keep = np.isin(labels, blobs).astype(np.uint8)

        left = min(stats[i, cv2.CC_STAT_LEFT] for i in blobs)
        top = min(stats[i, cv2.CC_STAT_TOP] for i in blobs)
        right = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in blobs)
        bottom = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in blobs)

        raw_w = right - left
        raw_h = bottom - top

        pad = int(0.08 * max(raw_w, raw_h))
        py0, py1 = max(0, top - pad), min(img_h, bottom + pad)
        px0, px1 = max(0, left - pad), min(img_w, right + pad)

        # Ink-ratio sanity check on the padded box, matching how the
        # ratio was originally calibrated (a bare ink-only box is nearly
        # 100% ink after MORPH_CLOSE fills stroke gaps, so it must be
        # measured with its normal surrounding margin, not alone).
        ink_ratio = float(keep[py0:py1, px0:px1].mean())
        if not (MIN_INK_RATIO <= ink_ratio <= MAX_INK_RATIO):
            raise SignatureNotFoundError(
                f"This photo doesn't look like a signature (ink covers "
                f"{ink_ratio:.1%} of the detected region). Retake it on plain paper."
            )

        grey_crop = grey_full[py0:py1, px0:px1]

        # The model's quality_gate (src/data/dataset.py) re-detects ink
        # in whatever we hand it and measures ITS OWN bounding box —
        # padding with blank paper does not change that measurement.
        # If the ink itself is smaller than the gate's minimum, the only
        # fix is to upscale the whole crop so the ink grows with it.
        smaller_ink_edge = min(raw_w, raw_h)
        scale = max(1.0, TARGET_INK_EDGE_PX / smaller_ink_edge)

        if scale > MAX_UPSCALE_FACTOR:
            raise SignatureNotFoundError(
                "The signature is too small in this photo to process reliably. "
                "Move closer and retake it so the signature fills more of the frame."
            )

        if scale > 1.0:
            new_size = (int(round(grey_crop.shape[1] * scale)), int(round(grey_crop.shape[0] * scale)))
            grey_crop = cv2.resize(grey_crop, new_size, interpolation=cv2.INTER_CUBIC)

        binary = cv2.threshold(cv2.GaussianBlur(grey_crop, (5, 5), 0), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        destination = os.path.join(tempfile.gettempdir(), f"cropped_{uuid.uuid4().hex}.png")
        cv2.imwrite(destination, binary)
        return destination

    except SignatureNotFoundError:
        raise
    except Exception as e:
        raise SignatureNotFoundError(
            f"Could not process this photo: {e}. "
            "Try a different image or retake it on plain paper."
        )