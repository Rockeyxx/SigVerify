"""GUI interfaces for signature verification.

This module provides a GUI wrapper around SignatureService (src/verification.py).
It converts between GUI storage formats (base64-encoded strings) and raw embedding
byte formats used by the service.
"""

import os
import base64
import shutil
import tempfile
import uuid

import torch  # noqa: E402 — must load before PyQt5 elsewhere; harmless here too
import cv2    # noqa: E402

from src.verification import SignatureService as RealSignatureService
from src.gui.image_cleanup import crop_to_ink

_service = None


def _get_service() -> RealSignatureService:
    global _service
    if _service is None:
        _service = RealSignatureService(device="cpu")
    return _service

def _prepare_image(path: str) -> str:
    """Prepare an image before sending it to the model.

    Steps:
      1. Crop to ink and clean background noise.
      2. Copy to an ASCII-safe temporary path to prevent filename encoding issues.
    """
    cleaned_path = crop_to_ink(path)

    try:
        cleaned_path.encode("ascii")
        return cleaned_path
    except UnicodeEncodeError:
        ext = os.path.splitext(cleaned_path)[1] or ".png"
        safe_path = os.path.join(tempfile.gettempdir(), f"sig_{uuid.uuid4().hex}{ext}")
        shutil.copyfile(cleaned_path, safe_path)
        return safe_path


def enroll_client(image_paths: list) -> dict:
    """Enroll client reference images using the spatial model.

    Returns:
        A dictionary containing:
          - "embeddings": list of base64-encoded embedding strings.
          - "spatial_origin": origin type ("scan").
    """
    safe_paths = [_prepare_image(p) for p in image_paths]
    service = _get_service()
    result = service.enroll_unimodal(safe_paths, modality="spatial")
    return {
        "embeddings": [base64.b64encode(b).decode("ascii") for b in result["embeddings"]],
        "spatial_origin": result["spatial_origin"],
    }


def verify_client(reference_embeddings_b64: list, spatial_origin: str, probe_image_path: str) -> dict:
    """Verify a probe image against stored client reference embeddings."""
    service = _get_service()
    ref_bytes = [base64.b64decode(b) for b in reference_embeddings_b64]
    safe_probe_path = _prepare_image(probe_image_path)
    return service.verify(
        mode="unimodal",
        probe_image=safe_probe_path,
        reference_spatial=ref_bytes,
        reference_spatial_origin=spatial_origin,
    )