"""GUI interfaces for signature verification.

This module provides a GUI wrapper around SignatureService (src/verification.py).
It converts between GUI storage formats (base64-encoded strings) and raw embedding
byte formats used by the service.
"""

import os
import base64
import shutil
import tempfile
import threading
import uuid

import torch  # noqa: E402 — must load before PyQt5 elsewhere; harmless here too
import cv2    # noqa: E402

from src.verification import SignatureService as RealSignatureService
from src.gui.image_cleanup import crop_to_ink

_service = None
_service_lock = threading.Lock()


def _get_service() -> RealSignatureService:
    # One SignatureService for the whole session (AGENTS.md): a second instance
    # loads the models again. The lock matters now that warm-up runs on its own
    # thread while the user may already be enrolling or verifying.
    global _service
    with _service_lock:
        if _service is None:
            _service = RealSignatureService(device="cpu")
        return _service

def warm_up() -> None:
    """Load both models and run a blank sample through each.
    Call once when the app opens, on a background thread (AGENTS.md)."""
    _get_service().warm_up()


def _remove_quietly(paths) -> None:
    """Delete temporary files; a file that is already gone is fine."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _prepare_image(path: str, temp_files: list) -> str:
    """Prepare an image before sending it to the model.

    Steps:
      1. Crop to ink and clean background noise.
      2. Copy to an ASCII-safe temporary path to prevent filename encoding issues.

    Every temporary file created is appended to `temp_files`, so the caller can
    delete them in a `finally` (the originals are never touched).
    """
    cleaned_path = crop_to_ink(path)
    temp_files.append(cleaned_path)

    try:
        cleaned_path.encode("ascii")
        return cleaned_path
    except UnicodeEncodeError:
        ext = os.path.splitext(cleaned_path)[1] or ".png"
        safe_path = os.path.join(tempfile.gettempdir(), f"sig_{uuid.uuid4().hex}{ext}")
        shutil.copyfile(cleaned_path, safe_path)
        temp_files.append(safe_path)
        return safe_path


def enroll_client(image_paths: list) -> dict:
    """Enroll client reference images using the spatial model.

    Returns:
        A dictionary containing:
          - "embeddings": list of base64-encoded embedding strings.
          - "spatial_origin": origin type ("scan").
    """
    temp_files = []
    try:
        safe_paths = [_prepare_image(p, temp_files) for p in image_paths]
        service = _get_service()
        result = service.enroll_unimodal(safe_paths, modality="spatial")
        return {
            "embeddings": [base64.b64encode(b).decode("ascii") for b in result["embeddings"]],
            "spatial_origin": result["spatial_origin"],
        }
    finally:
        _remove_quietly(temp_files)


def verify_client(reference_embeddings_b64: list, spatial_origin: str, probe_image_path) -> dict:
    """Verify a probe image against stored client reference embeddings.

    probe_image_path may be one path or a list of paths (several signings:
    the score is their mean). Under step-up the second call passes both
    signings, first signing first. The result's "decision" may be
    "Sign again" once step-up is on and a fused paper+tablet check is used.
    """
    service = _get_service()
    ref_bytes = [base64.b64decode(b) for b in reference_embeddings_b64]
    temp_files = []
    try:
        if isinstance(probe_image_path, (list, tuple)):
            probe = [_prepare_image(p, temp_files) for p in probe_image_path]
        else:
            probe = _prepare_image(probe_image_path, temp_files)
        return service.verify(
            mode="unimodal",
            probe_image=probe,
            reference_spatial=ref_bytes,
            reference_spatial_origin=spatial_origin,
        )
    finally:
        _remove_quietly(temp_files)


def embed_confirmed_signing(probe_image_path: str) -> str:
    """Embed one confirmed signing for adding to the client's references.
    Same path as enrollment, so the same quality gate applies. Returns base64."""
    temp_files = []
    try:
        safe_path = _prepare_image(probe_image_path, temp_files)
        result = _get_service().enroll_unimodal([safe_path], modality="spatial")
        return base64.b64encode(result["embeddings"][0]).decode("ascii")
    finally:
        _remove_quietly(temp_files)