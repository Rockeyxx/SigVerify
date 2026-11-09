"""
Integration Facade — Signature Verification System

This module provides the main application programming interface for signature
verification and enrollment. It connects high-level application code (GUI and
database services) with the underlying neural network models.

Enrollment stores continuous embedding vectors in binary format (ISO/IEC 24745:2022).
"""

import os

import cv2
import numpy as np
import torch

from config.settings import (
    ALPHA_DEFAULT,
    DECISION_THRESHOLD,
    SPATIAL_DECISION_THRESHOLD,
    TEMPORAL_DECISION_THRESHOLD,
    PRESSURE_IMPUTATION_VALUE,
    INFERENCE_NUM_THREADS,
    SPATIAL_INPUT_SIZE,
    TEMPORAL_INPUT_FEATURES,
    VERIFICATION_STEP_UP,
    STEP_UP_REJECT,
    STEP_UP_ACCEPT,
    STEP_UP_TWO,
)
from src.data.dataset import as_spatial_input, quality_gate, trajectory_gate
from src.data.renderer import render_trajectory_to_image
from src.models.siamese import FusionEngine, SpatialSiamese, TemporalSiamese

# Embeddings are stored as raw float32 arrays (512-d spatial = 2048 bytes, 128-d temporal = 512 bytes).
EMBEDDING_DTYPE = np.float32


def load_model(branch, weights_path, device="cpu"):
    """
    Load a trained model checkpoint for inference.

    Sets thread counts to prevent CPU contention and initializes the model in eval mode.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    torch.set_num_threads(INFERENCE_NUM_THREADS)
    cv2.setNumThreads(INFERENCE_NUM_THREADS)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if branch == "spatial":
        model = SpatialSiamese.from_state_dict(state_dict)
    elif branch == "temporal":
        model = TemporalSiamese.from_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown branch '{branch}' — expected 'spatial' or 'temporal'")

    return model.to(device).eval()


def embed(model, sample, branch, device="cpu"):
    """
    Generate an embedding vector from an input signature sample.

    Args:
        model: Loaded PyTorch Siamese model or ensemble wrapper.
        sample: Image path / array for spatial, or normalized trajectory for temporal.
        branch: Modality name ('spatial' or 'temporal').
        device: Execution device ('cpu' or 'cuda').

    Returns:
        Numpy float32 embedding vector.
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
    Validate and format a spatial input sample into a [1, H, W] array.
    Rejects raw trajectories to prevent unintended modality conversion.
    """
    return as_spatial_input(sample, allow_trajectory=False)


def embedding_to_bytes(embedding):
    """Convert an embedding array or tensor to a contiguous binary bytes object."""
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    return np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def bytes_to_embedding(blob):
    """Reconstruct a float32 numpy embedding vector from a binary byte buffer."""
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


def similarity_from_distance(distance):
    """
    Convert Euclidean distance d into a similarity score S between 0.0 and 1.0.
    Uses the reciprocal formula S = 1 / (1 + distance).
    """
    return 1.0 / (1.0 + distance)


def _nearest_reference_score(references, probe_embedding):
    """
    Compute similarity between a probe embedding and a set of enrolled references.
    Uses nearest-neighbor matching (minimum distance to enrolled templates).
    """
    if references is None or len(references) == 0:
        raise ValueError(
            "Cannot score a probe against an empty reference list — this identity "
            "has no enrolled templates. numpy reported this as \"need at least one "
            "array to concatenate\" from inside np.vstack, which names numpy's "
            "argument rather than the caller's empty enrolment.")

    stacked = np.vstack([np.asarray(r, dtype=EMBEDDING_DTYPE) for r in references])
    distances = np.linalg.norm(stacked - probe_embedding[None, :], axis=1)
    return float(similarity_from_distance(distances.min()))


def _pressure_is_imputed(trajectory):
    """Check if the trajectory uses fixed constant pressure values."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    return bool(np.allclose(trajectory[:, 2], PRESSURE_IMPUTATION_VALUE))


def _gate_enrollment_sample(sample, modality, index):
    """
    Validate an enrollment sample against quality gates before storing it.
    Prevents bad or corrupt references from entering the enrolled identity set.
    """
    if modality == "spatial":
        passed, details = quality_gate(sample)
        fallback = "failed the quality gate"
    else:
        passed, details = trajectory_gate(sample)
        fallback = "failed the input gate"

    if not passed:
        raise ValueError(
            f"Enrolment sample {index} failed the {modality} input gate: "
            f"{details.get('error', fallback)}. A rejected sample is not enrolled "
            f"— recapture it. See the README's Capture Contract.")


def _signings(probe, modality):
    """One probe, or the list of signings of one kind the app collected (VERIFICATION_SIGNINGS)."""
    probes = list(probe) if isinstance(probe, (list, tuple)) else [probe]
    if not probes:
        raise ValueError(f"{modality}: a list of probes needs at least one signing.")
    return probes


# The third decision of step-up verification (VERIFICATION_STEP_UP): the first signing was too
# close to call, so the GUI asks for one more signing of each kind and calls verify() with both.
SIGN_AGAIN = "Sign again"


def _step_up_round(spatial_probes, temporal_probes):
    """1 for the first signing of each kind, 2 for the first and the one asked for."""
    if len(spatial_probes) != len(temporal_probes):
        raise ValueError(
            "Step-up verification needs the same number of paper and tablet signings; got "
            f"{len(spatial_probes)} paper and {len(temporal_probes)} tablet.")
    if len(spatial_probes) > 2:
        raise ValueError(
            "Step-up verification asks for one more signing at most, never a third; got "
            f"{len(spatial_probes)} signings of each kind.")
    return len(spatial_probes)


def _branch_score(model, references, probe, modality, gate, fallback, device):
    """
    Mean nearest-reference score over the signings, or None and the gate messages.

    If any signing fails its gate the whole branch is dropped: averaging only the
    signings that pass would let a forger hand in one careful forgery and one
    scribble and be scored on the forgery alone.
    """
    probes = _signings(probe, modality)
    errors = []
    for index, sample in enumerate(probes):
        passed, details = gate(sample)
        if not passed:
            label = modality if len(probes) == 1 else f"{modality}, signing {index + 1}"
            errors.append(f"{label}: {details.get('error', fallback)}")
    if errors:
        return None, probes, errors
    scores = [_nearest_reference_score(references, embed(model, sample, modality, device))
              for sample in probes]
    return float(np.mean(scores)), probes, []


def verify(spatial=None, temporal=None, engine=None, device="cpu", auto_render_spatial=True,
           step_up=None):
    """
    Verify a signature probe against enrolled reference templates.

    Args:
        spatial: Optional tuple of (model, reference_embeddings, probe_image).
        temporal: Optional tuple of (model, reference_embeddings, probe_trajectory).
            Either probe may be a list of several signings of that kind; the branch
            score is then their mean, and the same thresholds apply.
        engine: Optional FusionEngine instance.
        device: Device to run inference on.
        auto_render_spatial: If True, renders temporal trajectory into image if spatial probe is missing.
        step_up: Step-up verification; None reads VERIFICATION_STEP_UP. It applies to the
            default fused decision only. One signing of each kind is accepted at
            STEP_UP_ACCEPT, rejected below STEP_UP_REJECT, and anything between returns
            SIGN_AGAIN; two signings of each kind are then decided at STEP_UP_TWO.

    Returns:
        Dictionary with decision, confidence score, routing mode, degraded flag,
        reasons, and the threshold the decision was taken against.
    """
    if auto_render_spatial and spatial is not None and temporal is not None:
        s_model, s_references, *s_probe_parts = spatial
        s_probe = s_probe_parts[0] if s_probe_parts else None
        if s_probe is None:
            _, _, t_probe = temporal
            if t_probe is not None:
                s_probe = ([render_trajectory_to_image(t) for t in t_probe]
                           if isinstance(t_probe, (list, tuple))
                           else render_trajectory_to_image(t_probe))
                spatial = (s_model, s_references, s_probe)

    if spatial is None and temporal is None:
        raise ValueError("At least one modality must be provided to verify().")
    if step_up is None:
        step_up = VERIFICATION_STEP_UP

    spatial_score = None
    temporal_score = None
    pressure_imputed = False
    gate_errors = []

    if spatial is not None:
        model, references, probe = spatial
        spatial_score, spatial_probes, errors = _branch_score(
            model, references, probe, "spatial", quality_gate, "failed the quality gate", device)
        gate_errors.extend(errors)

    if temporal is not None:
        model, references, probe = temporal
        temporal_score, probes, errors = _branch_score(
            model, references, probe, "temporal", trajectory_gate, "failed the input gate", device)
        gate_errors.extend(errors)
        if temporal_score is not None:
            pressure_imputed = any(_pressure_is_imputed(sample) for sample in probes)

    quality_failed = bool(gate_errors)

    if spatial_score is None and temporal_score is None:
        raise ValueError(
            "No usable evidence — every supplied probe failed its input gate, so a "
            "verdict here would rest on nothing:\n  "
            + "\n  ".join(gate_errors or ["no probe was supplied"])
            + "\nSee the README's Capture Contract.")

    # Select operational threshold based on available modality routing.
    step_up_round = None
    if engine is None:
        if spatial_score is not None and temporal_score is not None:
            threshold = DECISION_THRESHOLD
            if step_up:
                step_up_round = _step_up_round(spatial_probes, probes)
                threshold = STEP_UP_ACCEPT if step_up_round == 1 else STEP_UP_TWO
        elif spatial_score is not None:
            threshold = SPATIAL_DECISION_THRESHOLD
        else:
            threshold = TEMPORAL_DECISION_THRESHOLD
        engine = FusionEngine(alpha=ALPHA_DEFAULT, threshold=threshold)

    result = engine.fuse(spatial_score=spatial_score, temporal_score=temporal_score,
                         pressure_imputed=pressure_imputed)
    # The acceptance bar applied, for the audit log. On a step-up first round it is
    # STEP_UP_ACCEPT even when the outcome is SIGN_AGAIN or a reject below STEP_UP_REJECT.
    result["threshold"] = engine.threshold
    # Below STEP_UP_ACCEPT on a first signing: the same fused score against the floor.
    if step_up_round == 1 and result["decision"] == "Forged":
        floor = FusionEngine(alpha=ALPHA_DEFAULT, threshold=STEP_UP_REJECT)
        if floor.fuse(spatial_score=spatial_score, temporal_score=temporal_score)["decision"] == "Genuine":
            result["decision"] = SIGN_AGAIN

    if quality_failed:
        result["degraded"] = True

    result["reasons"] = list(gate_errors)
    if result["decision"] == SIGN_AGAIN:
        result["reasons"].append("step-up: the first signing is too close to call; ask for one more "
                                 "signing of each kind and call verify() again with both")
    if pressure_imputed:
        result["reasons"].append("temporal: pressure imputed to "
                                 f"{PRESSURE_IMPUTATION_VALUE} — device reported none")

    return result


class TemporalEnsemble:
    """
    Ensemble wrapper that averages embeddings across multiple TemporalSiamese checkpoints.
    """

    def __init__(self, models: list):
        self.models = models

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        embs = [m.encode(tensor) for m in self.models]
        return torch.stack(embs, dim=0).mean(dim=0)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        return self.models[index]


def load_temporal_ensemble(
    temporal_lstm_path: str | None = None,
    ensemble_seeds: list[int] | tuple[int, ...] | None = None,
    device: str = "cpu",
) -> TemporalEnsemble | None:
    """
    Load an ensemble of temporal checkpoints from multiple random seeds.
    Returns None if any checkpoint file in the ensemble is missing.
    """
    from config.settings import (MODEL_WEIGHTS_DIR, TEMPORAL_ENSEMBLE_SEEDS,
                                 TEMPORAL_ENSEMBLE_TEMPLATE)

    seeds = ensemble_seeds if ensemble_seeds is not None else TEMPORAL_ENSEMBLE_SEEDS
    if temporal_lstm_path:
        weights_dir = (
            os.path.dirname(temporal_lstm_path)
            if not os.path.isdir(temporal_lstm_path)
            else temporal_lstm_path
        )
    else:
        weights_dir = MODEL_WEIGHTS_DIR

    paths = [
        os.path.join(weights_dir, TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=seed))
        for seed in seeds
    ]
    if not all(os.path.exists(p) for p in paths):
        return None

    models = [load_model("temporal", p, device=device) for p in paths]
    return TemporalEnsemble(models)


class SignatureService:
    """
    Main service interface for signature enrollment and verification.

    Provides three operational verification modes:
      1. unimodal: Verify using either an image probe OR a trajectory probe.
      2. temporal-multi: Verify using a tablet trajectory by evaluating temporal
         dynamics and auto-rendering the spatial stroke image for fusion.
      3. multimodal: Verify using both a scanned image and a tablet trajectory.
    """

    def __init__(
        self,
        spatial_weights: str | None = None,
        temporal_weights: str | None = None,
        device: str = "cpu",
        use_temporal_ensemble: bool = False,
    ):
        from config.settings import SPATIAL_WEIGHTS_PATH, TEMPORAL_WEIGHTS_PATH
        self.device = device
        self.spatial_weights = spatial_weights or SPATIAL_WEIGHTS_PATH
        self.temporal_weights = temporal_weights or TEMPORAL_WEIGHTS_PATH
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

    def warm_up(self) -> None:
        """
        Load both models and run a blank sample through each.

        Call it once when the app opens, on a background thread, so the first Enter
        pays neither the model load nor the first forward pass's kernel setup.
        Calling it again does not reload the models.
        """
        width, height = SPATIAL_INPUT_SIZE
        embed(self.spatial_model, np.zeros((1, height, width), dtype=np.float32),
              "spatial", self.device)
        embed(self.temporal_model, np.zeros((64, TEMPORAL_INPUT_FEATURES), dtype=np.float32),
              "temporal", self.device)

    def _load_temporal_ensemble(self):
        """Load all temporal ensemble checkpoints or raise an error if any are missing."""
        from config.settings import (MODEL_WEIGHTS_DIR, TEMPORAL_ENSEMBLE_SEEDS,
                                     TEMPORAL_ENSEMBLE_TEMPLATE)

        ensemble = load_temporal_ensemble(
            self.temporal_weights,
            ensemble_seeds=TEMPORAL_ENSEMBLE_SEEDS,
            device=self.device,
        )
        if ensemble is None:
            weights_dir = (
                os.path.dirname(self.temporal_weights)
                if not os.path.isdir(self.temporal_weights)
                else self.temporal_weights
            ) or MODEL_WEIGHTS_DIR
            paths = [
                os.path.join(weights_dir, TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=seed))
                for seed in TEMPORAL_ENSEMBLE_SEEDS
            ]
            missing = [p for p in paths if not os.path.exists(p)]
            raise FileNotFoundError(
                f"use_temporal_ensemble=True needs all "
                f"{len(TEMPORAL_ENSEMBLE_SEEDS)} member checkpoints; "
                f"{len(missing)} are missing:\n  "
                + "\n  ".join(missing)
                + "\nThese are not in the public release. Either fetch them or "
                  "construct the service with use_temporal_ensemble=False, which "
                  "runs the single shipped model deliberately rather than by "
                  "accident.")
        return ensemble

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
        Enroll a user from tablet signature trajectories.
        Generates both temporal embeddings and auto-rendered spatial embeddings.
        """
        if not trajectories:
            raise ValueError("No trajectories provided for enrollment.")

        for index, traj in enumerate(trajectories):
            _gate_enrollment_sample(traj, "temporal", index)

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
            "spatial_origin": "render",
        }

    def enroll_multimodal(
        self,
        images: list[str | np.ndarray],
        trajectories: list[np.ndarray],
    ) -> dict[str, list[bytes]]:
        """
        Enroll a user with separate paper scans/photos and digitizer trajectories.
        """
        if len(images) != len(trajectories):
            raise ValueError(
                f"Mismatch in enrollment sample count: {len(images)} images vs {len(trajectories)} trajectories."
            )

        for index, img in enumerate(images):
            _gate_enrollment_sample(img, "spatial", index)
        for index, traj in enumerate(trajectories):
            _gate_enrollment_sample(traj, "temporal", index)

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
            "spatial_origin": "scan",
        }

    def enroll_unimodal(
        self,
        samples: list[str | np.ndarray],
        modality: str,
    ) -> dict[str, list[bytes] | str]:
        """
        Enroll a user in a single modality ('spatial' or 'temporal').
        """
        if modality not in ("spatial", "temporal"):
            raise ValueError(f"Unknown modality '{modality}'. Expected 'spatial' or 'temporal'.")

        for index, sample in enumerate(samples):
            _gate_enrollment_sample(sample, modality, index)

        model = self.spatial_model if modality == "spatial" else self.temporal_model
        blobs = [
            embedding_to_bytes(embed(model, s, modality, self.device))
            for s in samples
        ]
        result = {"embeddings": blobs, "modality": modality}
        if modality == "spatial":
            result["spatial_origin"] = "scan"
        return result

    _SPATIAL_ORIGINS = ("scan", "render")
    _ORIGIN_MISMATCH_NOTE = (
        "A spatial probe and its references must share an origin: measured on 20 paired "
        "SigComp2011 writers, a rendered probe against scanned references scores 57.4% EER "
        "and a scanned probe against rendered references 54.1%, worse than chance, against "
        "22.9% and 24.5% for matched pairs. Enrol the user through the path that matches how "
        "they will be verified, or store both kinds of reference.")

    @classmethod

    def _checked_spatial_origin(cls, origin):
        if origin not in cls._SPATIAL_ORIGINS:
            raise ValueError(
                "reference_spatial requires reference_spatial_origin, one of "
                f"{cls._SPATIAL_ORIGINS}; got {origin!r}. Every enroll_* helper returns it as "
                "\"spatial_origin\": store it with the templates and pass it back. It is "
                "not guessed because both wrong guesses measured worse than chance. "
                + cls._ORIGIN_MISMATCH_NOTE)
        return origin

    def _temporal_only_after_origin_mismatch(self, ref_t, probe_trajectory, probe_origin, reference_origin):
        """Fall back to temporal verification when spatial reference origin does not match probe origin."""
        result = verify(
            temporal=(self.temporal_model, ref_t, probe_trajectory),
            device=self.device,
        )
        result["degraded"] = True
        result["reasons"].append(
            f"spatial: dropped, the probe is a {probe_origin} and the references were "
            f"enrolled as {reference_origin}s. " + self._ORIGIN_MISMATCH_NOTE)
        return result

    def verify(
        self,
        mode: str,
        probe_trajectory: np.ndarray | list[np.ndarray] | None = None,
        probe_image: str | np.ndarray | list[str | np.ndarray] | None = None,
        reference_spatial: list[bytes | np.ndarray] | None = None,
        reference_temporal: list[bytes | np.ndarray] | None = None,
        reference_spatial_origin: str | None = None,
    ) -> dict:
        """
        Verify a signature probe against enrolled reference templates.

        Args:
            mode: Operational mode ('unimodal', 'temporal-multi', or 'multimodal').
            probe_trajectory: Optional [L, 4] trajectory array (X, Y, P, dt), or a list
                of them when the app asks for several signings (VERIFICATION_SIGNINGS).
            probe_image: Optional image path or array, or a list of them.
            reference_spatial: List of enrolled spatial reference templates.
            reference_temporal: List of enrolled temporal reference templates.
            reference_spatial_origin: Source type of spatial references ('scan' or 'render').

        Returns:
            Dictionary with verification decision, confidence percentage, routing, and degraded status.
        """
        ref_s = self._ensure_embeddings(reference_spatial)
        ref_t = self._ensure_embeddings(reference_temporal)
        if ref_s:
            origin = self._checked_spatial_origin(reference_spatial_origin)

        if mode == "temporal-multi":
            if probe_trajectory is None:
                raise ValueError("Mode 'temporal-multi' requires probe_trajectory.")
            if not ref_t or not ref_s:
                raise ValueError("Mode 'temporal-multi' requires both reference_temporal and reference_spatial.")

            if origin != "render":
                return self._temporal_only_after_origin_mismatch(
                    ref_t, probe_trajectory, probe_origin="render", reference_origin=origin)
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

            if origin != "scan":
                return self._temporal_only_after_origin_mismatch(
                    ref_t, probe_trajectory, probe_origin="scan", reference_origin=origin)
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
                if origin != "scan":
                    raise ValueError(
                        "Mode 'unimodal' scores a scanned probe_image, but reference_spatial "
                        f"was enrolled as '{origin}'. " + self._ORIGIN_MISMATCH_NOTE
                        + " There is no temporal branch to fall back to.")
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
