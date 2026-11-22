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
    if os.path.abspath(cleaned_path) != os.path.abspath(path):   # never delete the employee's own file
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


ROUTING_LABELS = {
    "multimodal": "Paper + pen (fused)",
    "spatial_only": "Paper only",
    "temporal_only": "Pen only",
}


def modality_state(result: dict) -> str:
    """The audit-log text for one verify() result: what actually ran, why it was degraded,
    and the threshold applied, so a decision can be explained later (issue #23, item 16)."""
    routing = result.get("routing", "unknown")
    parts = [ROUTING_LABELS.get(routing, routing)]
    if result.get("degraded"):
        reasons = "; ".join(result.get("reasons") or []) or "single modality"
        parts.append(f"degraded: {reasons}")
    if result.get("threshold") is not None:
        parts.append(f"threshold {result['threshold']:.4f}")
    return " | ".join(parts)


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


class PenSignatureRejectedError(ValueError):
    """A tablet signature failed the model's input gate. It has to be signed again."""


def _b64(blobs) -> list:
    return [base64.b64encode(b).decode("ascii") for b in blobs]


def _is_pen_gate_error(error) -> bool:
    return "temporal" in str(error).lower() and _is_gate_error(error)


def enroll_client(image_paths: list, trajectories: list = None) -> dict:
    """Enroll a client from paper images, tablet signatures, or both.

    image_paths: NUM_ENROLLMENT_REFERENCES photos/scans, or empty for a tablet-only client.
    trajectories: NUM_ENROLLMENT_REFERENCES [L, 4] arrays from collect_signatures(), or
        None/empty for a paper-only client. With both, they are enrolled together
        (enroll_multimodal), so the same count is needed for each.

    Returns:
        A dictionary containing:
          - "embeddings": base64 spatial embeddings (512 values each), empty when tablet-only.
          - "dynamic_embeddings": base64 pen embeddings (128 values each), empty when paper-only.
          - "spatial_origin": "scan". Paper images are scans; a tablet-only client has no
            spatial set, and pen signatures are never rendered into one (pen alone measured
            better than pen plus a picture drawn from it).
    The raw trajectories are not kept (ISO/IEC 24745): only the embeddings are returned.
    """
    image_paths = list(image_paths or [])
    trajectories = list(trajectories or [])
    if not image_paths and not trajectories:
        raise ValueError("Nothing to enrol: add paper images, pen signatures, or both.")
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
            if safe_paths and trajectories:
                result = service.enroll_multimodal(safe_paths, trajectories)
                spatial, temporal = result["spatial_embeddings"], result["temporal_embeddings"]
            elif safe_paths:
                result = service.enroll_unimodal(safe_paths, modality="spatial")
                spatial, temporal = result["embeddings"], []
            else:
                result = service.enroll_unimodal(trajectories, modality="temporal")
                spatial, temporal = [], result["embeddings"]
        except ValueError as e:
            if not _is_gate_error(e):
                raise
            if _is_pen_gate_error(e):
                raise PenSignatureRejectedError(str(e)) from e
            explained = _explain_rejections(service, names, safe_paths, e)
            if explained is e:
                raise
            raise explained from e
        return {
            "embeddings": _b64(spatial),
            "dynamic_embeddings": _b64(temporal),
            "spatial_origin": result.get("spatial_origin", "scan"),
        }
    finally:
        _remove_quietly(temp_files)


def choose_mode(has_image: bool, has_pen: bool, has_spatial_refs: bool, has_temporal_refs: bool):
    """Which verify() mode a visit can use, or None if the probes and references do not meet.

    Returns "multimodal" (paper + pen, fused), "spatial" or "temporal" (one branch only).
    A probe whose references are missing is not used: a client enrolled from the tablet only
    is verified on the pen alone, even when a photo is also given.
    """
    use_image = has_image and has_spatial_refs
    use_pen = has_pen and has_temporal_refs
    if use_image and use_pen:
        return "multimodal"
    if use_pen:
        return "temporal"
    if use_image:
        return "spatial"
    return None


def verify_client(reference_embeddings_b64: list, spatial_origin: str, probe_image_path=None,
                  reference_temporal_b64: list = None, probe_trajectory=None) -> dict:
    """Verify a visit against the client's stored references.

    probe_image_path: one image path, a list of them (several signings, or the two signings
        of a step-up, first first), or None for a pen-only visit.
    probe_trajectory: one [L, 4] array from collect_signatures(), a list of them in the same
        order, or None for a paper-only visit.
    With both probes and both reference sets the visit is fused ("multimodal"); otherwise the
    one branch that has a probe and references is used (single branch, reported as degraded).
    The result's "decision" may be "Sign again" once step-up is on and the visit is fused.
    """
    ref_spatial = [base64.b64decode(b) for b in (reference_embeddings_b64 or [])]
    ref_temporal = [base64.b64decode(b) for b in (reference_temporal_b64 or [])]
    mode = choose_mode(probe_image_path is not None, probe_trajectory is not None,
                       bool(ref_spatial), bool(ref_temporal))
    if mode is None:
        raise ValueError(
            "Nothing to compare: this signer has no references for the signature given. "
            "Use the kind of signature the signer was enrolled with.")

    service = _get_service()
    temp_files = []
    try:
        probe_image = None
        if mode in ("multimodal", "spatial"):
            if isinstance(probe_image_path, (list, tuple)):
                probe_image = [_prepare_image(p, temp_files) for p in probe_image_path]
            else:
                probe_image = _prepare_image(probe_image_path, temp_files)
        if mode == "multimodal":
            return service.verify(
                mode="multimodal",
                probe_image=probe_image,
                probe_trajectory=probe_trajectory,
                reference_spatial=ref_spatial,
                reference_temporal=ref_temporal,
                reference_spatial_origin=spatial_origin,
            )
        if mode == "temporal":
            # Do not pass reference_spatial: pen alone, never a picture drawn from the pen.
            return service.verify(
                mode="unimodal",
                probe_trajectory=probe_trajectory,
                reference_temporal=ref_temporal,
            )
        return service.verify(
            mode="unimodal",
            probe_image=probe_image,
            reference_spatial=ref_spatial,
            reference_spatial_origin=spatial_origin,
        )
    finally:
        _remove_quietly(temp_files)


def embed_confirmed_signing(probe_image_path=None, trajectory=None) -> dict:
    """Embed one confirmed signing for adding to the client's references.

    Same path as enrollment, so the same input gates apply. Pass the signing's image, its
    pen trajectory, or both (a fused visit adds both). Returns
    {"spatial": base64 or None, "temporal": base64 or None}.
    """
    temp_files = []
    out = {"spatial": None, "temporal": None}
    try:
        service = _get_service()
        if probe_image_path is not None:
            safe_path = _prepare_image(probe_image_path, temp_files)
            result = service.enroll_unimodal([safe_path], modality="spatial")
            out["spatial"] = _b64(result["embeddings"])[0]
        if trajectory is not None:
            result = service.enroll_unimodal([trajectory], modality="temporal")
            out["temporal"] = _b64(result["embeddings"])[0]
        return out
    finally:
        _remove_quietly(temp_files)