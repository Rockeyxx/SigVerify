"""
Tests for the integration facade consumed by the GUI and database teams.

This is the contract boundary: if these pass, teammates can build against
`src.verification` without touching model internals or distance math.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from src.models.siamese import FusionEngine, SpatialSiamese, TemporalSiamese
from src.verification import (
    bytes_to_embedding,
    embed,
    embedding_to_bytes,
    load_model,
    verify,
)


def _sharp_signature_image():
    """Write a temp image with enough ink and sharpness to pass the quality gate."""
    img = np.full((400, 400), 255, dtype=np.uint8)
    for i in range(50, 350, 5):
        cv2.line(img, (i, 100), (i + 30, 300), 0, 2)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


def _blank_image():
    """A near-empty image that must fail the quality gate."""
    img = np.full((40, 40), 255, dtype=np.uint8)
    cv2.circle(img, (20, 20), 3, 0, 1)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


class TestLoadModel:
    """Contract ①: teammates load weights without knowing the class names."""

    def test_spatial_round_trip(self, tmp_path):
        """load_model must reproduce the exact embedding the saved model produced."""
        original = SpatialSiamese(embedding_dim=512).eval()
        path = tmp_path / "spatial_cnn.pth"
        torch.save(original.state_dict(), path)

        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            expected = original.encode(x)
            actual = load_model("spatial", str(path)).encode(x)

        assert torch.allclose(expected, actual)

    def test_returns_model_in_eval_mode(self, tmp_path):
        """Inference must never run with dropout active."""
        path = tmp_path / "spatial_cnn.pth"
        torch.save(SpatialSiamese().state_dict(), path)
        assert load_model("spatial", str(path)).training is False

    def test_temporal_round_trip(self, tmp_path):
        original = TemporalSiamese(embedding_dim=128).eval()
        path = tmp_path / "temporal_lstm.pth"
        torch.save(original.state_dict(), path)

        x = torch.randn(1, 40, 3)
        with torch.no_grad():
            expected = original.encode(x)
            actual = load_model("temporal", str(path)).encode(x)

        assert torch.allclose(expected, actual)

    def test_unknown_branch_raises(self, tmp_path):
        path = tmp_path / "w.pth"
        torch.save(SpatialSiamese().state_dict(), path)
        with pytest.raises(ValueError, match="branch"):
            load_model("olfactory", str(path))

    def test_missing_weights_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("spatial", "/nonexistent/weights.pth")


class TestEmbeddingBytes:
    """Contract ②: the byte convention the DB team wraps in AES-256-GCM."""

    def test_spatial_embedding_is_2048_bytes(self):
        """512 float32 values → 512 × 4 = 2048 bytes."""
        assert len(embedding_to_bytes(np.zeros(512, dtype=np.float32))) == 2048

    def test_temporal_embedding_is_512_bytes(self):
        """128 float32 values → 128 × 4 = 512 bytes."""
        assert len(embedding_to_bytes(np.zeros(128, dtype=np.float32))) == 512

    def test_round_trip_is_exact(self):
        """Serialization must be lossless — a drifted embedding is a wrong identity."""
        original = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        assert np.array_equal(bytes_to_embedding(embedding_to_bytes(original)), original)

    def test_accepts_torch_tensor(self):
        """Callers hold tensors straight off the model; don't force manual conversion."""
        assert len(embedding_to_bytes(torch.zeros(512))) == 2048

    def test_decoded_embedding_is_writable(self):
        """np.frombuffer aliases read-only memory — callers must get a real array."""
        decoded = bytes_to_embedding(embedding_to_bytes(np.zeros(128, dtype=np.float32)))
        decoded[0] = 1.0  # must not raise

    def test_corrupt_payload_raises(self):
        """A truncated blob is not silently reinterpreted as a shorter embedding."""
        with pytest.raises(ValueError):
            bytes_to_embedding(b"\x00\x00\x00")


class TestEmbed:
    """Preprocessing + encoding in one call, so callers never build tensors."""

    def test_spatial_embed_from_path(self):
        path = _sharp_signature_image()
        embedding = embed(SpatialSiamese(embedding_dim=512).eval(), path, "spatial")
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        os.unlink(path)

    def test_temporal_embed_from_array(self):
        trajectory = np.random.default_rng(0).standard_normal((60, 3)).astype(np.float32)
        embedding = embed(TemporalSiamese(embedding_dim=128).eval(), trajectory, "temporal")
        assert embedding.shape == (128,)


class TestVerify:
    """Contract ③/④: one entrypoint returning the GUI's dict, no distance math outside."""

    def _spatial_setup(self):
        model = SpatialSiamese(embedding_dim=512).eval()
        path = _sharp_signature_image()
        references = [embed(model, path, "spatial")]
        return model, references, path

    def _temporal_setup(self):
        model = TemporalSiamese(embedding_dim=128).eval()
        trajectory = np.random.default_rng(1).standard_normal((50, 3)).astype(np.float32)
        references = [embed(model, trajectory, "temporal")]
        return model, references, trajectory

    def test_returns_the_gui_contract(self):
        model, references, path = self._spatial_setup()
        result = verify(spatial=(model, references, path))
        assert set(result) == {"decision", "confidence", "routing", "degraded"}
        os.unlink(path)

    def test_spatial_only_is_degraded(self):
        """A single modality is reduced evidence and must be flagged as such."""
        model, references, path = self._spatial_setup()
        result = verify(spatial=(model, references, path))
        assert result["routing"] == "spatial_only"
        assert result["degraded"] is True
        os.unlink(path)

    def test_identical_probe_scores_near_maximum(self):
        """Probing with the enrolled signature itself gives d≈0, so S=1/(1+d)≈1."""
        model, references, path = self._spatial_setup()
        result = verify(spatial=(model, references, path),
                        engine=FusionEngine(threshold=0.5))
        assert result["decision"] == "Genuine"
        assert result["confidence"] > 95.0
        os.unlink(path)

    def test_multimodal_routing(self):
        smodel, srefs, spath = self._spatial_setup()
        tmodel, trefs, traj = self._temporal_setup()
        result = verify(spatial=(smodel, srefs, spath), temporal=(tmodel, trefs, traj))
        assert result["routing"] == "multimodal"
        os.unlink(spath)

    def test_failed_quality_gate_marks_degraded(self):
        """
        The orphaned quality gate must actually run on the spatial path.

        Uses the multimodal route deliberately: that path is otherwise
        `degraded=False`, so a True here can only come from the quality gate.
        """
        smodel, srefs, good_path = self._spatial_setup()
        tmodel, trefs, traj = self._temporal_setup()
        bad_path = _blank_image()

        clean = verify(spatial=(smodel, srefs, good_path), temporal=(tmodel, trefs, traj))
        assert clean["degraded"] is False

        gated = verify(spatial=(smodel, srefs, bad_path), temporal=(tmodel, trefs, traj))
        assert gated["routing"] == "multimodal"
        assert gated["degraded"] is True

        os.unlink(good_path)
        os.unlink(bad_path)

    def test_nearest_reference_matching(self):
        """
        With several enrolled references the best match wins, so an unrelated
        extra reference cannot drag a genuine probe below threshold.
        """
        model, references, path = self._spatial_setup()
        noisy = np.random.default_rng(9).standard_normal(512).astype(np.float32) * 50
        result = verify(spatial=(model, references + [noisy], path),
                        engine=FusionEngine(threshold=0.5))
        assert result["decision"] == "Genuine"
        os.unlink(path)

    def test_no_modality_raises(self):
        with pytest.raises(ValueError):
            verify()

    def test_accepts_decoded_reference_bytes(self):
        """End-to-end DB path: stored bytes → embedding → verify, no reshaping by callers."""
        model, references, path = self._spatial_setup()
        stored = [embedding_to_bytes(r) for r in references]
        result = verify(spatial=(model, [bytes_to_embedding(b) for b in stored], path),
                        engine=FusionEngine(threshold=0.5))
        assert result["decision"] == "Genuine"
        os.unlink(path)
