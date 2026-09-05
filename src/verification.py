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

import cv2
import numpy as np
import torch

from config.settings import (
    ALPHA_DEFAULT,
    SPATIAL_INPUT_SIZE,
    DECISION_THRESHOLD,
    SPATIAL_DECISION_THRESHOLD,
    TEMPORAL_DECISION_THRESHOLD,
    PRESSURE_IMPUTATION_VALUE,
    INFERENCE_NUM_THREADS,
)
from src.data.dataset import preprocess_image, quality_gate, trajectory_gate
from src.data.renderer import render_trajectory_to_image
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

    # Bound the thread pools before any forward pass. Torch and OpenCV each size
    # themselves to every logical CPU independently, which is fine on an idle box
    # and costs 4-7x on a contended one. Set once at load, not per call.
    torch.set_num_threads(INFERENCE_NUM_THREADS)
    cv2.setNumThreads(INFERENCE_NUM_THREADS)

    # `weights_only=True` refuses to unpickle arbitrary objects. These checkpoints
    # are downloaded from Releases by the install instructions, so a plain
    # `torch.load` would execute whatever a substituted .pth carried. Every
    # artifact this repo ships is a plain tensor state_dict and loads unchanged.
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

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
        tensor = torch.from_numpy(_as_spatial_input(sample)).unsqueeze(0)
    elif branch == "temporal":
        tensor = torch.as_tensor(np.asarray(sample, dtype=np.float32)).unsqueeze(0)
    else:
        raise ValueError(f"Unknown branch '{branch}' — expected 'spatial' or 'temporal'")

    with torch.no_grad():
        return model.encode(tensor.to(device)).squeeze(0).cpu().numpy().astype(EMBEDDING_DTYPE)


def _as_spatial_input(sample):
    """
    Bring a spatial sample to the model's [1, H, W] contract, or refuse it.

    Three inputs are legitimate and only one of them is already canonical:

    * a path — read, resized and inverted by `preprocess_image`;
    * a **raw scan array** (paper bright, ink dark, any size) — same treatment;
    * a **rendered trajectory**, which `render_trajectory_to_image` already
      emits at SPATIAL_INPUT_SIZE with inverted polarity, so it passes straight
      through. Sending it through `preprocess_image` would invert it a second
      time and hand the model a photographic negative.

    Anything else raises. Accepting it quietly is the defect this function
    exists to close: `SpatialSiamese` pools adaptively, so a wrong-sized,
    un-inverted uint8 scan yields a perfectly plausible 512-d embedding with no
    exception — and enrolment then writes that template to the database. The
    inversion alone is worth 4.86% to 0.00% EER on CEDAR, so a silent miss here
    is not cosmetic.
    """
    if not isinstance(sample, np.ndarray):
        return preprocess_image(sample)

    width, height = SPATIAL_INPUT_SIZE
    canonical = (sample.ndim == 3 and sample.shape == (1, height, width)
                 and np.issubdtype(sample.dtype, np.floating))
    if canonical:
        low, high = float(sample.min()), float(sample.max())
        if low < -1e-6 or high > 1.0 + 1e-6:
            raise ValueError(
                f"Spatial array has the canonical shape {sample.shape} but values "
                f"in [{low:.3f}, {high:.3f}] — the contract is [0.0, 1.0] with "
                f"background 0. A raw scan belongs in a 2-D array so it gets "
                f"resized and inverted; this one would reach the model unscaled.")
        return np.ascontiguousarray(sample, dtype=np.float32)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[0] == 1):
        # A raw scan: preprocess_image resizes and inverts it.
        return preprocess_image(sample)

    raise ValueError(
        f"Cannot interpret a spatial sample of shape {sample.shape}. Expected a "
        f"path, a 2-D scan array, or a rendered [1, {height}, {width}] float "
        f"array in [0, 1].")


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


def similarity_from_distance(distance):
    """
    Convert an L2 embedding distance into a bounded similarity score.

    Uses reciprocal scaling S = 1 / (1 + d), mapping [0, inf) to (0, 1]. This
    formula is part of the project contract and cannot be swapped for an exponential
    or cosine form without recalibrating every enrolled reference in the database:
    a stored template enrolled under one curve cannot match probes under another.

    Fixed on 2026-08-18 (P-006): evaluate.py previously defined its own private
    function whose reciprocal denominator held an extra factor of 2. The training
    loss and the verification facade agreed with each other and disagree with
    evaluate.py, so a model benchmarked at one reported threshold could never be
    deployed with it. Single authority now lives here in src/verification.py.

    Never subtract: 1.0 - distance / margin clips at margin and creates a hard floor
    where all impostors beyond margin receive score 0.0, which destroys the ranking
    information evaluate.py needs to sweep thresholds across the tail. Reciprocal
    preserves ordering at every distance.

    Never tune: `similarity_from_distance` is not a hyperparameter. Moving it moves
    the scale every score is reported on, so a test score that 'improved' under a
    calibrated threshold would have kept its old numeric value while silently
    describing a different score distribution.
    """
    return 1.0 / (1.0 + distance)


def _nearest_reference_score(references, probe_embedding):
    """
    Compute similarity match score between probe embedding and enrolled references.

    Evaluates minimum distance (nearest-reference matching) across registered templates.
    """
    stacked = np.vstack([np.asarray(r, dtype=EMBEDDING_DTYPE) for r in references])
    distances = np.linalg.norm(stacked - probe_embedding[None, :], axis=1)
    return float(similarity_from_distance(distances.min()))


def _pressure_is_imputed(trajectory):
    """Check whether input sequence uses flat imputed pressure values."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    return bool(np.allclose(trajectory[:, 2], PRESSURE_IMPUTATION_VALUE))


def verify(spatial=None, temporal=None, engine=None, device="cpu", auto_render_spatial=True):
    """
    Perform signature verification evaluating probe inputs against enrolled references.

    Args:
        spatial: Optional tuple of (model, reference_embeddings, probe_image_path/array).
        temporal: Optional tuple of (model, reference_embeddings, probe_trajectory).
        engine: FusionEngine instance for combining branch scores.
        device: Execution device ('cpu').
        auto_render_spatial: If True and spatial probe is absent while temporal trajectory
            is present, auto-render the spatial stroke image from the trajectory.

    Returns dictionary containing verification verdict:
        {"decision": str, "confidence": float, "routing": str, "degraded": bool}
    """
    if auto_render_spatial and spatial is not None and temporal is not None:
        s_model, s_references, *s_probe_parts = spatial
        s_probe = s_probe_parts[0] if s_probe_parts else None
        if s_probe is None:
            _, _, t_probe = temporal
            if t_probe is not None:
                s_probe = render_trajectory_to_image(t_probe)
                spatial = (s_model, s_references, s_probe)

    if spatial is None and temporal is None:
        raise ValueError("At least one modality must be provided to verify().")

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

    # Threshold follows the routing the gates actually produced, never the shape of
    # the request. Selecting it up-front from which arguments were *passed* meant a
    # probe that ended up scored `spatial_only` — because the trajectory failed its
    # gate — was still compared against the fused DECISION_THRESHOLD. The fused bar
    # is the highest of the three, so the error ran one way only: a genuine signer
    # whose tablet timed out was rejected where the same image submitted alone was
    # accepted, at identical reported confidence.
    if engine is None:
        if spatial_score is not None and temporal_score is not None:
            threshold = DECISION_THRESHOLD
        elif spatial_score is not None:
            threshold = SPATIAL_DECISION_THRESHOLD
        else:
            threshold = TEMPORAL_DECISION_THRESHOLD
        engine = FusionEngine(alpha=ALPHA_DEFAULT, threshold=threshold)

    result = engine.fuse(spatial_score=spatial_score, temporal_score=temporal_score,
                         pressure_imputed=pressure_imputed)

    if quality_failed:
        result["degraded"] = True

    # `degraded: True` on its own cannot tell the GUI whether evidence was dropped
    # for a blurry image, an undersized bounding box, an unnormalized trajectory or
    # imputed pressure. The gate already produced that string; carry it out instead
    # of discarding it on every path except total failure.
    result["reasons"] = list(gate_errors)
    if pressure_imputed:
        result["reasons"].append("temporal: pressure imputed to "
                                 f"{PRESSURE_IMPUTATION_VALUE} — device reported none")

    return result


class TemporalEnsemble:
    """
    Model wrapper ensembling multiple TemporalSiamese checkpoints.

    Evaluates a trajectory across N models and returns their mean embedding.
    Conforms to the `.encode(tensor)` interface expected by embed().
    """

    def __init__(self, models: list):
        self.models = models

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        embs = [m.encode(tensor) for m in self.models]
        return torch.stack(embs, dim=0).mean(dim=0)


class SignatureService:
    """
    High-level operational service for Signature Verification and Enrollment.

    Designed for frontend (GUI) and backend (API/Database) developers to perform
    enrollment and verification without handling raw PyTorch tensors, model loading,
    or embedding byte serialization.

    Supports three verification modes:
      1. 'unimodal': Evaluates either an image probe OR a trajectory probe alone.
      2. 'temporal-multi': Evaluates a digitizer trajectory by extracting dynamics
         and auto-rendering the spatial stroke image, running full late fusion.
      3. 'multimodal': Evaluates both an independent paper scan/photo AND a digitizer
         trajectory under late fusion.

    `use_temporal_ensemble=True` scores the temporal branch across the six `dt`
    seeds and returns their mean embedding, which leaves the 128-d width and every
    stored template unchanged. Measured on 120 unseen DeepSignDB writers under one
    shared K=5 draw: 4.82% EER against 6.92% for a single member. (Averaging the
    per-model scores instead reaches 4.53%, but that cannot be expressed through
    the `.encode()` interface `embed()` calls.) Those numbers are threshold-free
    EER on one draw and are NOT comparable to the calibrated 25-split figures in
    SUMMARY.md; they establish the ordering of the arms, not an operating point.
    """

    def __init__(
        self,
        spatial_weights: str | None = None,
        temporal_weights: str | None = None,
        device: str = "cpu",
        use_temporal_ensemble: bool = False,
    ):
        from config.settings import MODEL_WEIGHTS_DIR
        self.device = device
        self.spatial_weights = spatial_weights or os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth")
        self.temporal_weights = temporal_weights or os.path.join(MODEL_WEIGHTS_DIR, "temporal_lstm.pth")
        self.use_temporal_ensemble = use_temporal_ensemble
        self._spatial_model = None
        self._temporal_model = None

    @property
    def spatial_model(self):
        if self._spatial_model is None:
            self._spatial_model = load_model("spatial", self.spatial_weights, self.device)
        return self._spatial_model

    @property
    def temporal_model(self):
        if self._temporal_model is None:
            if self.use_temporal_ensemble:
                self._temporal_model = self._load_temporal_ensemble()
            else:
                self._temporal_model = load_model("temporal", self.temporal_weights, self.device)
        return self._temporal_model

    def _load_temporal_ensemble(self):
        """
        Load every ensemble member, or refuse.

        Falling back to the single model when a member is missing was the
        original behaviour, and it fails in the worst direction: the caller asked
        for an ensemble, got one model, and nothing said so. The member weights
        are gitignored and absent from the v0.8.0 release, so on any machine
        holding only the release that fallback is the *normal* path, not an edge
        case — the service would quietly run at single-model accuracy while its
        own flag claimed otherwise. Part 9 §12 records four serving defects with
        exactly this shape.
        """
        from config.settings import (MODEL_WEIGHTS_DIR, TEMPORAL_ENSEMBLE_SEEDS,
                                     TEMPORAL_ENSEMBLE_TEMPLATE)

        paths = [os.path.join(MODEL_WEIGHTS_DIR,
                              TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=seed))
                 for seed in TEMPORAL_ENSEMBLE_SEEDS]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"use_temporal_ensemble=True needs all "
                f"{len(TEMPORAL_ENSEMBLE_SEEDS)} member checkpoints; "
                f"{len(missing)} are missing:\n  "
                + "\n  ".join(missing)
                + "\nThese are not in the public release. Either fetch them or "
                  "construct the service with use_temporal_ensemble=False, which "
                  "runs the single shipped model deliberately rather than by "
                  "accident.")
        return TemporalEnsemble([load_model("temporal", p, self.device) for p in paths])

    def _ensure_embeddings(self, refs: list[bytes | np.ndarray] | None) -> list[np.ndarray]:
        if not refs:
            return []
        out = []
        for r in refs:
            if isinstance(r, (bytes, bytearray)):
                out.append(bytes_to_embedding(r))
            else:
                out.append(np.asarray(r, dtype=EMBEDDING_DTYPE))
        return out

    def enroll_from_tablet(self, trajectories: list[np.ndarray]) -> dict[str, list[bytes]]:
        """
        Enroll a user who signs K times on a digitizer tablet (e.g. XPPen Deco 640).

        Extracts 128-d temporal embeddings from trajectories, automatically renders
        each stroke to a canonical 155x220 spatial image, and extracts 512-d spatial
        embeddings.

        Returns:
            {"spatial_embeddings": [bytes...], "temporal_embeddings": [bytes...]}
            ready for SQLite BLOB storage.
        """
        if not trajectories:
            raise ValueError("No trajectories provided for enrollment.")

        spatial_bytes = []
        temporal_bytes = []

        for traj in trajectories:
            t_emb = embed(self.temporal_model, traj, "temporal", self.device)
            temporal_bytes.append(embedding_to_bytes(t_emb))

            rendered_img = render_trajectory_to_image(traj)
            s_emb = embed(self.spatial_model, rendered_img, "spatial", self.device)
            spatial_bytes.append(embedding_to_bytes(s_emb))

        return {
            "spatial_embeddings": spatial_bytes,
            "temporal_embeddings": temporal_bytes,
        }

    def enroll_multimodal(
        self,
        images: list[str | np.ndarray],
        trajectories: list[np.ndarray],
    ) -> dict[str, list[bytes]]:
        """
        Enroll a user with separate paper scans/photos and digitizer trajectories.

        Returns:
            {"spatial_embeddings": [bytes...], "temporal_embeddings": [bytes...]}
        """
        if len(images) != len(trajectories):
            raise ValueError(
                f"Mismatch in enrollment sample count: {len(images)} images vs {len(trajectories)} trajectories."
            )

        spatial_bytes = [
            embedding_to_bytes(embed(self.spatial_model, img, "spatial", self.device))
            for img in images
        ]
        temporal_bytes = [
            embedding_to_bytes(embed(self.temporal_model, traj, "temporal", self.device))
            for traj in trajectories
        ]

        return {
            "spatial_embeddings": spatial_bytes,
            "temporal_embeddings": temporal_bytes,
        }

    def enroll_unimodal(
        self,
        samples: list[str | np.ndarray],
        modality: str,
    ) -> dict[str, list[bytes] | str]:
        """
        Enroll a user in a single modality ('spatial' or 'temporal').

        Returns:
            {"embeddings": [bytes...], "modality": modality}
        """
        if modality not in ("spatial", "temporal"):
            raise ValueError(f"Unknown modality '{modality}'. Expected 'spatial' or 'temporal'.")

        model = self.spatial_model if modality == "spatial" else self.temporal_model
        blobs = [
            embedding_to_bytes(embed(model, s, modality, self.device))
            for s in samples
        ]
        return {"embeddings": blobs, "modality": modality}

    def verify(
        self,
        mode: str,
        probe_trajectory: np.ndarray | None = None,
        probe_image: str | np.ndarray | None = None,
        reference_spatial: list[bytes | np.ndarray] | None = None,
        reference_temporal: list[bytes | np.ndarray] | None = None,
    ) -> dict:
        """
        Verify a signature probe against enrolled reference templates under the specified mode.

        Args:
            mode: 'unimodal', 'temporal-multi', or 'multimodal'.
            probe_trajectory: Optional [L, 4] trajectory array (X, Y, P, dt).
            probe_image: Optional image file path or numpy array.
            reference_spatial: List of enrolled spatial reference templates (bytes or numpy).
            reference_temporal: List of enrolled temporal reference templates (bytes or numpy).

        Returns:
            {"decision": "Genuine"|"Forged", "confidence": float, "routing": str, "degraded": bool, "reasons": list[str]}
        """
        ref_s = self._ensure_embeddings(reference_spatial)
        ref_t = self._ensure_embeddings(reference_temporal)

        if mode == "temporal-multi":
            if probe_trajectory is None:
                raise ValueError("Mode 'temporal-multi' requires probe_trajectory.")
            if not ref_t or not ref_s:
                raise ValueError("Mode 'temporal-multi' requires both reference_temporal and reference_spatial.")

            return verify(
                spatial=(self.spatial_model, ref_s, None),
                temporal=(self.temporal_model, ref_t, probe_trajectory),
                device=self.device,
                auto_render_spatial=True,
            )

        elif mode == "multimodal":
            if probe_trajectory is None or probe_image is None:
                raise ValueError("Mode 'multimodal' requires both probe_image and probe_trajectory.")
            if not ref_s or not ref_t:
                raise ValueError("Mode 'multimodal' requires both reference_spatial and reference_temporal.")

            return verify(
                spatial=(self.spatial_model, ref_s, probe_image),
                temporal=(self.temporal_model, ref_t, probe_trajectory),
                device=self.device,
                auto_render_spatial=False,
            )

        elif mode == "unimodal":
            if probe_trajectory is not None and ref_t:
                return verify(
                    temporal=(self.temporal_model, ref_t, probe_trajectory),
                    device=self.device,
                )
            elif probe_image is not None and ref_s:
                return verify(
                    spatial=(self.spatial_model, ref_s, probe_image),
                    device=self.device,
                )
            else:
                raise ValueError(
                    "Mode 'unimodal' requires either (probe_image + reference_spatial) "
                    "or (probe_trajectory + reference_temporal)."
                )

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'unimodal', 'temporal-multi', or 'multimodal'.")

