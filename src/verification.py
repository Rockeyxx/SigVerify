"""
Integration Facade — Signature Verification System

The single seam between the model workstream and everyone else. The GUI and
database teams import from here and nothing deeper: no distance math, no tensor
construction, no knowledge of which class implements which branch.

Typical enrollment (store embeddings only — ISO/IEC 24745:2022):

    model = load_model("spatial", "models/weights/spatial_cnn.pth")
    blob = embedding_to_bytes(embed(model, "sig.png", "spatial"))   # 2048 bytes
    # ... DB team wraps `blob` in AES-256-GCM and stores it

Typical verification:

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
    PRESSURE_IMPUTATION_VALUE,
    SPATIAL_EMBEDDING_DIM,
    TEMPORAL_EMBEDDING_DIM,
    TEMPORAL_HIDDEN_SIZE,
    TEMPORAL_NUM_LAYERS,
)
from src.data.dataset import preprocess_image, quality_gate
from src.models.siamese import FusionEngine, SpatialSiamese, TemporalSiamese

# Embeddings are persisted as raw little-endian float32. 512-d spatial → 2048 B,
# 128-d temporal → 512 B. The DB team encrypts these bytes; changing this dtype
# silently invalidates every enrolled template already in the database.
EMBEDDING_DTYPE = np.float32


def load_model(branch, weights_path, device="cpu"):
    """
    Load a trained branch ready for inference.

    Always maps to CPU first, so weights trained on a GPU (Colab) load on the
    deployment machine, which has none. Returns the model in `.eval()` mode —
    inference must never run with dropout active.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if branch == "spatial":
        model = SpatialSiamese(embedding_dim=SPATIAL_EMBEDDING_DIM)
    elif branch == "temporal":
        model = TemporalSiamese(
            hidden_size=TEMPORAL_HIDDEN_SIZE,
            num_layers=TEMPORAL_NUM_LAYERS,
            embedding_dim=TEMPORAL_EMBEDDING_DIM,
        )
    else:
        raise ValueError(f"Unknown branch '{branch}' — expected 'spatial' or 'temporal'")

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.to(device).eval()


def embed(model, sample, branch, device="cpu"):
    """
    Turn one raw sample into its embedding vector.

    Args:
        sample: an image path for `spatial`, or an [L, 3] trajectory array for
                `temporal` (already normalized via `parsers.normalize_trajectory`).

    Returns: float32 numpy vector — 512-d spatial, 128-d temporal.
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
    Serialize an embedding for encrypted storage.

    Accepts a numpy array or torch tensor. 512-d → 2048 bytes, 128-d → 512 bytes.
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    return np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def bytes_to_embedding(blob):
    """
    Deserialize a stored embedding.

    Returns a writable copy — `np.frombuffer` would alias read-only memory and
    blow up the first time a caller tried to modify it. A payload whose length
    is not a multiple of 4 raises rather than being reinterpreted as a shorter
    (silently wrong) embedding.
    """
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


def _nearest_reference_score(references, probe_embedding):
    """
    Score a probe against an enrolled reference set.

    Uses **nearest-reference** matching (not the mean): a person's signature
    varies, and the closest enrolled sample is the fairest comparison. This
    choice is fixed at calibration per AGENTS.md — do not switch it to mean
    matching after τ has been tuned.

    Returns S = 1 / (1 + d) ∈ (0, 1], matching the evaluation script's scoring.
    """
    stacked = np.vstack([np.asarray(r, dtype=EMBEDDING_DTYPE) for r in references])
    distances = np.linalg.norm(stacked - probe_embedding[None, :], axis=1)
    return float(1.0 / (1.0 + distances.min()))


def _pressure_is_imputed(trajectory):
    """True when the trajectory carries no real pressure (every P == 1.0)."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    return bool(np.allclose(trajectory[:, 2], PRESSURE_IMPUTATION_VALUE))


def verify(spatial=None, temporal=None, engine=None, device="cpu"):
    """
    Verify a probe signature against an enrolled reference set.

    Args:
        spatial:  (model, reference_embeddings, probe_image_path) or None
        temporal: (model, reference_embeddings, probe_trajectory) or None
        engine:   a configured FusionEngine; defaults to the calibrated settings.

    Supplying one branch routes to it automatically (α collapses to 1 or 0);
    supplying both fuses them. At least one is required.

    Returns the GUI contract:
        {"decision": str, "confidence": float, "routing": str, "degraded": bool}

    `degraded` is True when the result rests on reduced evidence — a missing
    branch, imputed pressure, or a probe image that failed the quality gate.
    The GUI must surface it: a degraded "Genuine" is a weaker claim than a
    clean multimodal one.
    """
    if spatial is None and temporal is None:
        raise ValueError("At least one modality must be provided to verify().")

    engine = engine or FusionEngine(alpha=ALPHA_DEFAULT, threshold=DECISION_THRESHOLD)

    spatial_score = None
    temporal_score = None
    quality_failed = False
    pressure_imputed = False

    if spatial is not None:
        model, references, probe = spatial
        # The quality gate was defined but never called anywhere; this is its
        # one live wiring point. A blurry or near-empty scan still produces an
        # embedding, so without this the result would look fully confident.
        passed, _details = quality_gate(probe)
        quality_failed = not passed
        spatial_score = _nearest_reference_score(references, embed(model, probe, "spatial", device))

    if temporal is not None:
        model, references, probe = temporal
        pressure_imputed = _pressure_is_imputed(probe)
        temporal_score = _nearest_reference_score(references, embed(model, probe, "temporal", device))

    result = engine.fuse(spatial_score=spatial_score, temporal_score=temporal_score,
                         pressure_imputed=pressure_imputed)

    if quality_failed:
        result["degraded"] = True

    return result
