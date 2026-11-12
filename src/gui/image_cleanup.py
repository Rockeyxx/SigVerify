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
from src.capture.ink import ink_crop


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

    Raises FileNotFoundError if the file itself is gone, so callers can say so
    instead of reporting it as a bad photo.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # ink_crop (src/capture/ink.py, ENG_REPORT A6) finds the signature and returns it
        # binarized, ink black on white, or None when the photo holds no signature.
        binary = ink_crop(image)
        if binary is None:
            raise SignatureNotFoundError(
                "No signature found. Sign on white paper with a blue pen."
            )

        # The model's quality_gate (src/data/dataset.py) re-detects the ink and measures ITS OWN
        # bounding box, so padding does not help: if the ink is smaller than the gate's minimum,
        # the whole crop is upscaled so the ink grows with it.
        ys, xs = np.nonzero(binary == 0)
        if len(ys) == 0:
            raise SignatureNotFoundError(
                "No signature found. Sign on white paper with a blue pen."
            )
        smaller_ink_edge = max(1, min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
        scale = max(1.0, TARGET_INK_EDGE_PX / smaller_ink_edge)

        if scale > MAX_UPSCALE_FACTOR:
            raise SignatureNotFoundError(
                "The signature is too small in this photo to process reliably. "
                "Move closer and retake it so the signature fills more of the frame."
            )

        if scale > 1.0:
            new_size = (int(round(binary.shape[1] * scale)), int(round(binary.shape[0] * scale)))
            binary = cv2.threshold(cv2.resize(binary, new_size, interpolation=cv2.INTER_CUBIC),
                                   0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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