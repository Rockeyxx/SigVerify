"""
image_cleanup.py — Signature Verification System

Cleans up a phone photo or scan of a paper signature before it reaches the model, on every
enrollment/verification call (ENG_REPORT A6, issue #23 item 5).

The signature is found by ink_crop (src/capture/ink.py): ink is a pixel darker than its local
paper and blue, and the signature is the cluster of ink pieces holding the most ink. The crop
comes back grey (not black and white), so faint strokes keep their strength. This module adds
what the model's input needs around that:

  * the photo's EXIF rotation is fixed first;
  * a blank-page guard: the ink must cover a plausible share of the crop (a photo of a ruled
    page or a printed blue form is refused instead of being enrolled as a signature);
  * an upscale (enlarge only, no re-thresholding) when the ink is smaller than the model's
    quality_gate (src/data/dataset.py) accepts. That gate measures the INK'S OWN bounding box,
    not the image size, so padding does nothing; the whole crop has to be enlarged;
  * the result is written to a temp file, which the caller deletes after use.

Blue pen on white paper only, by design: ink_crop finds ink by colour, so a black-ink or
grayscale image is reported as no signature (no fallback without asking @Rockeyxx).
"""

import os
import tempfile
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

from config.settings import INK_MIN_CONTRAST, MIN_BBOX_PIXELS, SPATIAL_INPUT_SIZE
from src.capture.ink import ink_crop


# Blank-page guard: share of the crop that is ink (a clearly darker pixel than white paper).
# The values come from the guard of the previous algorithm (PR #17 review point 3).
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


def crop_to_ink(path: str) -> str:
    """
    Turn a phone photo into what the model expects: fix the EXIF rotation, cut out the
    signature with ink_crop (a grey crop, ink dark on white), check that the crop looks like a
    signature (ink share), and enlarge it if the ink itself is smaller than the model's quality
    gate requires.

    Returns the path to a new temp file. The original file is never modified.

    Raises SignatureNotFoundError if ink_crop finds no signature (including black-ink or
    grayscale images), if the ink share of the crop is implausible (blank or ruled page,
    printed form), or if the ink is so small that no reasonable upscale reaches the model's
    minimum size.

    Raises FileNotFoundError if the file itself is gone, so callers can say so
    instead of reporting it as a bad photo.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # ink_crop (src/capture/ink.py, ENG_REPORT A6) finds the signature and returns it as a
        # grey crop (ink dark on white paper), or None when the photo holds no signature.
        crop = ink_crop(image)
        if crop is None:
            raise SignatureNotFoundError(
                "No signature found. Sign on white paper with a blue pen."
            )
        
        # The model's quality_gate (src/data/dataset.py) re-detects the ink and measures ITS OWN
        # bounding box, so padding does not help: if the ink is smaller than the gate's minimum,
        # the whole crop is upscaled so the ink grows with it.
        ink = crop < 255 - INK_MIN_CONTRAST     # the crop is grey: ink is clearly darker than paper
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

        # Resize to the model's input size here, with the same INTER_AREA the model uses
        # (preprocess_image in src/data/dataset.py, SPATIAL_SQUARE_PAD = False). The model's own
        # resize then changes nothing, and quality_gate judges the image the model actually sees,
        # not the soft wide strokes of a full-size phone photo.
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