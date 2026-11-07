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

def warm_up() -> None:
    """Load both models and run a blank sample through each.
    Call once when the app opens, on a background thread (AGENTS.md)."""
    _get_service().warm_up()


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


def enroll_client(image_paths: list) -> dict:
    """
    Runs each reference image through the real spatial model.

    Returns a dict (NOT a bare list anymore):
        {
            "embeddings": [base64 str, ...],
            "spatial_origin": "scan",   # these are real scanned/photographed images
        }
    The caller (enrollment.py) must save spatial_origin alongside the
    embeddings — verify_client() requires it later.
    """
    safe_paths = [_prepare_image(p) for p in image_paths]
    service = _get_service()
    result = service.enroll_unimodal(safe_paths, modality="spatial")
    return {
        "embeddings": [base64.b64encode(b).decode("ascii") for b in result["embeddings"]],
        "spatial_origin": result["spatial_origin"],
    }


def verify_client(reference_embeddings_b64: list, spatial_origin: str, probe_image_path) -> dict:
    """Verify a probe image against stored client reference embeddings.

    probe_image_path may be one path or a list of paths (several signings:
    the score is their mean). Under step-up the second call passes both
    signings, first signing first. The result's "decision" may be
    "Sign again" once step-up is on and a fused paper+tablet check is used.
    """
    service = _get_service()
    ref_bytes = [base64.b64decode(b) for b in reference_embeddings_b64]
    if isinstance(probe_image_path, (list, tuple)):
        probe = [_prepare_image(p) for p in probe_image_path]
    else:
        probe = _prepare_image(probe_image_path)
    return service.verify(
        mode="unimodal",
        probe_image=probe,
        reference_spatial=ref_bytes,
        reference_spatial_origin=spatial_origin,
    )


def embed_confirmed_signing(probe_image_path: str) -> str:
    """Embed one confirmed signing for adding to the client's references.
    Same path as enrollment, so the same quality gate applies. Returns base64."""
    result = _get_service().enroll_unimodal([_prepare_image(probe_image_path)], modality="spatial")
    return base64.b64encode(result["embeddings"][0]).decode("ascii")