"""
interfaces.py — Signature Verification System

Thin GUI-facing wrapper around Omar's real integration facade
(src/verification.py). This module does NOT reimplement any model or
scoring logic — it only converts between the GUI's storage format
(base64 strings in JSON) and the real SignatureService's format
(raw bytes embeddings).

Dynamic (tablet) verification is not wired yet — pending src/capture/.
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
    """
    Full preparation pipeline for any image before it reaches the model:
    1. Crop to ink + binarize (fixes phone-photo framing/lighting issues)
    2. Copy to an ASCII-safe temp path (fixes cv2.imread failing silently
       on non-ASCII paths, e.g. Arabic "Desktop" folder names)
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

def enroll_client(image_paths: list) -> list:
    safe_paths = [_prepare_image(p) for p in image_paths]
    service = _get_service()
    result = service.enroll_unimodal(safe_paths, modality="spatial")
    return [base64.b64encode(b).decode("ascii") for b in result["embeddings"]]


def verify_client(reference_embeddings_b64: list, probe_image_path: str) -> dict:
    service = _get_service()
    ref_bytes = [base64.b64decode(b) for b in reference_embeddings_b64]
    safe_probe_path = _prepare_image(probe_image_path)
    return service.verify(
        mode="unimodal",
        probe_image=safe_probe_path,
        reference_spatial=ref_bytes,
    )