"""Prepare paper photos/scans for the spatial model: EXIF orient, blue-ink crop, size check."""

import os
import tempfile
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

from config.settings import INK_MIN_CONTRAST, MIN_BBOX_PIXELS, SPATIAL_INPUT_SIZE
from src.capture.ink import ink_crop


MIN_INK_RATIO = 0.003
MAX_INK_RATIO = 0.35

# Margin above MIN_BBOX_PIXELS so quality_gate's own Otsu crop passes cleanly.
TARGET_INK_EDGE_PX = MIN_BBOX_PIXELS + 10
MAX_UPSCALE_FACTOR = 6.0


class SignatureNotFoundError(ValueError):
    """Raised when no valid blue-ink signature can be isolated or upscaled."""
    pass


def crop_to_ink(path: str) -> str:
    """Fix EXIF orientation, crop blue ink, check ink ratio/scale, and write a temp PNG."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        crop = ink_crop(image)
        if crop is None:
            raise SignatureNotFoundError(
                "No signature found. Sign on white paper with a blue pen."
            )
        
        ink = crop < 255 - INK_MIN_CONTRAST
        ys, xs = np.nonzero(ink)
        if len(ys) == 0:
            raise SignatureNotFoundError(
                "No signature found. Sign on white paper with a blue pen."
            )
        ink_ratio = float(ink.mean())
        if not (MIN_INK_RATIO <= ink_ratio <= MAX_INK_RATIO):
            raise SignatureNotFoundError(
                f"This photo doesn't look like a signature (ink covers {ink_ratio:.1%} of the "
                "detected region). Retake it on plain white paper."
            )

        smaller_ink_edge = max(1, min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
        if TARGET_INK_EDGE_PX / smaller_ink_edge > MAX_UPSCALE_FACTOR:
            raise SignatureNotFoundError(
                "The signature is too small in this photo to process reliably. "
                "Move closer and retake it so the signature fills more of the frame."
            )

        # Resize with INTER_AREA here so quality_gate inspects the exact resolution the CNN sees.
        crop = cv2.resize(crop, SPATIAL_INPUT_SIZE, interpolation=cv2.INTER_AREA)

        destination = os.path.join(tempfile.gettempdir(), f"cropped_{uuid.uuid4().hex}.png")
        cv2.imwrite(destination, crop)
        return destination

    except SignatureNotFoundError:
        raise
    except Exception as e:
        raise SignatureNotFoundError(
            f"Could not process this photo: {e}. "
            "Try a different image or retake it on plain paper."
        )