"""
Integration Facade — Signature Verification System

Primary operational abstraction linking backend inference models with application components.
GUI views and database operations import directly from this layer, decoupling downstream consumers
from tensor formatting, distance calculations, and model execution mechanics.

Typical enrollment workflow (persisting embeddings only — ISO/IEC 24745:2022 standard):

    model = load_model("spatial", "models/weights/spatial_cnn.pth")
    blob = embedding_to_bytes(embed(model, "sig.png", "spatial"))   # 2048 bytes
    # ... Database layer encrypts `blob` payload via AES-256-GCM prior to storage

Typical verification workflow:

    refs = [bytes_to_embedding(b) for b in stored_blobs]
    result = verify(spatial=(model, refs, probe_path))
    # {"decision": "Genuine", "confidence": 91.4,
    #  "routing": "spatial_only", "degraded": True}
"""

import os

import numpy as np
import torch

from config.settings import (
    ALPHA_DEFAULT,
    DECISION_THRESHOLD,
    SPATIAL_DECISION_THRESHOLD,
    TEMPORAL_DECISION_THRESHOLD,
    PRESSURE_IMPUTATION_VALUE,
    TEMPORAL_EMBEDDING_DIM,
    TEMPORAL_HIDDEN_SIZE,
    TEMPORAL_NUM_LAYERS,
)
from src.data.dataset import preprocess_image, quality_gate, trajectory_gate
from src.models.siamese import FusionEngine, SpatialSiamese, TemporalSiamese

# Vector representations are serialized as raw little-endian float32 arrays. 512-d spatial → 2048 B,
# 128-d temporal → 512 B. Encrypted byte structures rely on this layout; altering data types invalidates stored templates.
EMBEDDING_DTYPE = np.float32


def load_model(branch, weights_path, device="cpu"):
    """
    Instantiate and initialize a trained model branch for evaluation.

    Maps parameters to CPU by default to ensure GPU-trained weights evaluate cleanly
    on CPU deployment environments. The returned model is set to `.eval()` mode.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")

    if branch == "spatial":
        # Model architecture is reconstructed directly from checkpoint metadata to match original parameters.
        model = SpatialSiamese.from_state_dict(state_dict)
    elif branch == "temporal":
        # Reconstruction reads channel parameters and state dict parameters from stored weights.
        model = TemporalSiamese.from_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown branch '{branch}' — expected 'spatial' or 'temporal'")

    return model.to(device).eval()


def embed(model, sample, branch, device="cpu"):
    """
    Transform raw input sample into continuous feature vector embedding.

    Args:
        sample: Image file path for `spatial` branch, or normalized [L, 3] trajectory array for `temporal`.
        branch: Modality identifier ('spatial' or 'temporal').
        device: Target execution device.

    Returns:
        float32 numpy array — 512-d for spatial, 128-d for temporal.
    """
    if branch == "spatial":
        tensor = torch.from_numpy(preprocess_image(sample)).unsqueeze(0)
    elif branch == "temporal":
        tensor = torch.as_tensor(np.asarray(sample, dtype=np.float32)).unsqueeze(0)
    else:
        raise ValueError(f"Unknown branch '{branch}' — expected 'spatial' or 'temporal'")

    with torch.no_grad():
        return model.encode(tensor.to(device)).squeeze(0).cpu().numpy().astype(EMBEDDING_DTYPE)


def embedding_to_bytes(embedding):
    """
    Convert vector array representation into contiguous byte payload for encrypted storage.

    Accepts numpy array or PyTorch tensor representations.
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    return np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def bytes_to_embedding(blob):
    """
    Reconstruct numpy feature vector from binary byte buffer.

    Returns a writable array copy to prevent read-only buffer mutation errors.
    """
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


def _nearest_reference_score(references, probe_embedding):
    """
    Compute similarity match score between probe embedding and enrolled references.

    Evaluates minimum distance (nearest-reference matching) across registered templates.
    Returns calculated similarity score S = 1 / (1 + d).
    """
    stacked = np.vstack([np.asarray(r, dtype=EMBEDDING_DTYPE) for r in references])
    distances = np.linalg.norm(stacked - probe_embedding[None, :], axis=1)
    return float(1.0 / (1.0 + distances.min()))


def _pressure_is_imputed(trajectory):
    """Check whether input sequence uses flat imputed pressure values."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    return bool(np.allclose(trajectory[:, 2], PRESSURE_IMPUTATION_VALUE))


def verify(spatial=None, temporal=None, engine=None, device="cpu"):
    """
    Perform signature verification evaluating probe inputs against enrolled references.

    Args:
        spatial: Optional tuple of (model, reference_embeddings, probe_image_path).
        temporal: Optional tuple of (model, reference_embeddings, probe_trajectory).
        engine: FusionEngine instance for combining branch scores.

    Returns dictionary containing verification verdict:
        {"decision": str, "confidence": float, "routing": str, "degraded": bool}
    """
    if spatial is None and temporal is None:
        raise ValueError("At least one modality must be provided to verify().")

    # Select single-branch decision threshold or combined fusion threshold based on active inputs.
    if engine is None:
        if spatial is not None and temporal is not None:
            threshold = DECISION_THRESHOLD
        elif spatial is not None:
            threshold = SPATIAL_DECISION_THRESHOLD
        else:
            threshold = TEMPORAL_DECISION_THRESHOLD
        engine = FusionEngine(alpha=ALPHA_DEFAULT, threshold=threshold)

    spatial_score = None
    temporal_score = None
    quality_failed = False
    pressure_imputed = False
    gate_errors = []

    if spatial is not None:
        model, references, probe = spatial
        # Screen spatial image inputs against quality criteria before feature extraction.
        passed, details = quality_gate(probe)
        quality_failed = not passed
        if not passed:
            gate_errors.append(f"spatial: {details.get('error', 'failed the quality gate')}")
        if passed:
            spatial_score = _nearest_reference_score(
                references, embed(model, probe, "spatial", device))

    if temporal is not None:
        model, references, probe = temporal
        # Screen temporal trajectory points against normalization and sequence gates.
        passed, details = trajectory_gate(probe)
        quality_failed = quality_failed or not passed
        if passed:
            pressure_imputed = _pressure_is_imputed(probe)
            temporal_score = _nearest_reference_score(
                references, embed(model, probe, "temporal", device))
        else:
            gate_errors.append(f"temporal: {details.get('error', 'failed the input gate')}")

    if spatial_score is None and temporal_score is None:
        raise ValueError(
            "No usable evidence — every supplied probe failed its input gate, so a "
            "verdict here would rest on nothing:\n  "
            + "\n  ".join(gate_errors or ["no probe was supplied"])
            + "\nSee the README's Capture Contract.")

    result = engine.fuse(spatial_score=spatial_score, temporal_score=temporal_score,
                         pressure_imputed=pressure_imputed)

    if quality_failed:
        result["degraded"] = True

    return result

