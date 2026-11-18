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

from config.settings import MIN_LAPLACIAN_VARIANCE
from src.verification import SignatureService as RealSignatureService
from src.gui.image_cleanup import SignatureNotFoundError, crop_to_ink

_service = None
_service_lock = threading.Lock()


def _get_service() -> RealSignatureService:
    # One SignatureService for the whole session (AGENTS.md). The lock only makes sure
    # two threads (warm-up and a worker) do not both create the service object; that is
    # instant. The models load lazily inside the service (spatial_model / temporal_model),
    # outside this lock.
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


class ImageRejectedError(ValueError):
    """Reference images were rejected. The message names each image and says why."""


def _is_gate_error(error) -> bool:
    text = str(error).lower()
    return "gate" in text


def _rejection_reason(prepared_path: str) -> str:
    """Why the model's quality gate rejects an image that was already cropped.

    The gate itself only says "failed the quality gate", so the reason is measured here
    with the documented rule (Laplacian variance >= MIN_LAPLACIAN_VARIANCE). It is used only to
    explain a rejection the model has already made, never to reject anything itself.
    """
    gray = cv2.imread(prepared_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return "the image could not be read"
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < MIN_LAPLACIAN_VARIANCE:
        return (f"it is too blurry or the ink is too faint "
                f"(sharpness {sharpness:.0f}, at least {MIN_LAPLACIAN_VARIANCE:.0f} needed)")
    return "the signature is too small or not clear enough"


def _explain_rejections(service, names: list, safe_paths: list, original: ValueError):
    """The batch was rejected: embed each image alone to find which ones, and why.

    Returns an ImageRejectedError naming every rejected image, or `original` if no single
    image reproduces the rejection."""
    problems = []
    for number, (name, path) in enumerate(zip(names, safe_paths), start=1):
        try:
            service.enroll_unimodal([path], modality="spatial")
        except ValueError as one:
            reason = _rejection_reason(path) if _is_gate_error(one) else str(one).splitlines()[0]
            problems.append(f"Image {number} ({name}) was rejected: {reason}.")
    if not problems:
        return original
    return ImageRejectedError(
        " ".join(problems) + " Remove it from the list, retake the photo on white paper with a "
        "blue pen (close, steady, even light) and add it again.")


def enroll_client(image_paths: list) -> dict:
    """Enroll client reference images using the spatial model.

    Returns:
        A dictionary containing:
          - "embeddings": list of base64-encoded embedding strings.
          - "spatial_origin": origin type ("scan").
    """
    temp_files = []
    try:
        names = [os.path.basename(str(p)) for p in image_paths]
        safe_paths = []
        for number, (path, name) in enumerate(zip(image_paths, names), start=1):
            try:
                safe_paths.append(_prepare_image(path, temp_files))
            except SignatureNotFoundError as e:
                raise ImageRejectedError(f"Image {number} ({name}) was rejected: {e}") from e
        service = _get_service()
        try:
            result = service.enroll_unimodal(safe_paths, modality="spatial")
        except ValueError as e:
            if not _is_gate_error(e):
                raise
            explained = _explain_rejections(service, names, safe_paths, e)
            if explained is e:
                raise
            raise explained from e
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