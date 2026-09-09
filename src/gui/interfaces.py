"""
interfaces.py — Signature Verification System

Shared contract between the GUI layer and the ML / Database teams.
Every function here is currently a MOCK (placeholder). Once teammates
finish their real implementations, replace the function body with a
real import — the function signature (name, inputs, outputs) should
stay the same so the GUI code never needs to change.

REQUEST TO SEND TEAMMATES:
Ask Omar and Abdullah to confirm/match these exact function names and
shapes. If they already have different names, we just rename the
import here — nothing else in the GUI changes.
"""

import random
import numpy as np


# ============================================================
# ML PIPELINE — expected from Omar (temporal) & Abdullah (spatial)
# ============================================================

def preprocess_image(image_path: str) -> np.ndarray:
    """
    TODO (Abdullah): replace with real preprocessing.
    Expected output shape: [1, 1, 256, 256] — 8-bit grayscale,
    aspect-ratio-preserving pad, resize, normalize.
    """
    return np.zeros((1, 1, 256, 256), dtype=np.float32)


def preprocess_sequence(raw_sequence: list) -> np.ndarray:
    """
    TODO (Omar): replace with real preprocessing.
    Expected output shape: [1, L, 3] — X, Y, Pressure.
    If pressure is missing, pad with constant P=1.0 (rules.md §5).
    """
    return np.zeros((1, len(raw_sequence) if raw_sequence else 1, 3), dtype=np.float32)


def get_spatial_embedding(image_tensor: np.ndarray) -> np.ndarray:
    """
    TODO (Abdullah): replace with real CNN forward pass.
    Expected output shape: [512]
    """
    return np.random.randn(512).astype(np.float32)


def get_temporal_embedding(sequence_tensor: np.ndarray) -> np.ndarray:
    """
    TODO (Omar): replace with real LSTM forward pass.
    Expected output shape: [128]
    """
    return np.random.randn(128).astype(np.float32)


def compute_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    L2 distance between two embeddings (per rules.md §6).
    This one is simple math — likely doesn't need a teammate's version,
    but keep it here for a single source of truth.
    """
    return float(np.linalg.norm(embedding_a - embedding_b))


def fuse_scores(s_spatial: float, s_temporal: float, alpha: float = 0.5) -> float:
    """
    S_final = alpha * S_spatial + (1 - alpha) * S_temporal  (rules.md §6)
    alpha is tuned later during calibration (currently default 0.5).
    """
    return alpha * s_spatial + (1 - alpha) * s_temporal


# ============================================================
# DATABASE LAYER — expected from Abdullah
# ============================================================

_mock_db = {}  # temporary in-memory store, replaced by real SQLite+AES-256-GCM later

def store_embedding(user_id: str, embedding: np.ndarray, modality: str) -> bool:
    """
    TODO (Abdullah): replace with real encrypted SQLite storage.
    modality: "spatial" or "temporal"
    """
    _mock_db.setdefault(user_id, {}).setdefault(modality, []).append(embedding)
    return True


def get_references(user_id: str, modality: str) -> list:
    """
    TODO (Abdullah): replace with real encrypted retrieval.
    Returns a list of stored embeddings for that user/modality.
    """
    return _mock_db.get(user_id, {}).get(modality, [])


def log_verification_event(user_id: str, decision: str, score: float, modality_state: str) -> None:
    """
    TODO (Abdullah): replace with real plaintext audit log write
    (IEEE 730 traceability requirement — rules.md §8).
    """
    print(f"[AUDIT LOG] user={user_id} decision={decision} "
          f"score={score} modality_state={modality_state}")


# ============================================================
# MOCK DECISION HELPER — used only until real model is connected
# ============================================================

def mock_verify_decision() -> tuple:
    """Temporary random result generator, used only for GUI testing."""
    decision = random.choice(["Genuine", "Forged"])
    confidence = round(random.uniform(60, 99), 1)
    return decision, confidence