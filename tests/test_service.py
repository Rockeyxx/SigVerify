"""
Tests for SignatureService (Option C High-Level Operational Facade).
"""

import os
import numpy as np
import pytest

from config.settings import (
    MODEL_WEIGHTS_DIR,
    TEMPORAL_ENSEMBLE_SEEDS,
    TEMPORAL_ENSEMBLE_TEMPLATE,
)
from src.verification import SignatureService, bytes_to_embedding


class TestSignatureService:
    @pytest.fixture
    def service(self):
        return SignatureService(device="cpu")

    def _sample_trajectory(self, num_points=100):
        t = np.linspace(0, 4 * np.pi, num_points)
        x = (t * np.cos(t)).astype(np.float32)
        y = (t * np.sin(t)).astype(np.float32)
        p = (0.3 + 0.6 * (np.sin(t) ** 2)).astype(np.float32)
        dt = np.full(num_points, 0.01, dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)
        return np.stack([x, y, p, dt], axis=1)

    def test_enroll_from_tablet_creates_both_modalities(self, service):
        trajs = [self._sample_trajectory(80 + i * 5) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)

        assert "spatial_embeddings" in enrollment
        assert "temporal_embeddings" in enrollment
        assert len(enrollment["spatial_embeddings"]) == 5
        assert len(enrollment["temporal_embeddings"]) == 5

        # Verify byte formats: 512-d float32 = 2048 bytes, 128-d float32 = 512 bytes
        for s_blob in enrollment["spatial_embeddings"]:
            assert isinstance(s_blob, bytes)
            assert len(s_blob) == 2048
            arr = bytes_to_embedding(s_blob)
            assert arr.shape == (512,)

        for t_blob in enrollment["temporal_embeddings"]:
            assert isinstance(t_blob, bytes)
            assert len(t_blob) == 512
            arr = bytes_to_embedding(t_blob)
            assert arr.shape == (128,)

    def test_verify_temporal_multi_mode(self, service):
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)
        probe = self._sample_trajectory(95)

        result = service.verify(
            mode="temporal-multi",
            probe_trajectory=probe,
            reference_spatial=enrollment["spatial_embeddings"],
            reference_temporal=enrollment["temporal_embeddings"],
        )

        assert result["routing"] == "multimodal"
        assert result["decision"] in ("Genuine", "Forged")
        assert 0.0 <= result["confidence"] <= 100.0
        assert not result["degraded"]

    def test_verify_unimodal_temporal_mode(self, service):
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)
        probe = self._sample_trajectory(95)

        result = service.verify(
            mode="unimodal",
            probe_trajectory=probe,
            reference_temporal=enrollment["temporal_embeddings"],
        )

        assert result["routing"] == "temporal_only"
        assert result["decision"] in ("Genuine", "Forged")

    def test_verify_invalid_mode_raises(self, service):
        probe = self._sample_trajectory(95)
        with pytest.raises(ValueError, match="Unknown mode"):
            service.verify(mode="invalid-mode", probe_trajectory=probe)

    def test_verify_missing_required_inputs_raises(self, service):
        with pytest.raises(ValueError, match="requires probe_trajectory"):
            service.verify(mode="temporal-multi", probe_trajectory=None)

    def test_enroll_and_verify_multimodal(self, service):
        from src.data.renderer import render_trajectory_to_image
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        imgs = [render_trajectory_to_image(t) for t in trajs]

        enrollment = service.enroll_multimodal(images=imgs, trajectories=trajs)
        assert len(enrollment["spatial_embeddings"]) == 5
        assert len(enrollment["temporal_embeddings"]) == 5

        probe_traj = self._sample_trajectory(95)
        probe_img = render_trajectory_to_image(probe_traj)

        result = service.verify(
            mode="multimodal",
            probe_image=probe_img,
            probe_trajectory=probe_traj,
            reference_spatial=enrollment["spatial_embeddings"],
            reference_temporal=enrollment["temporal_embeddings"],
        )
        assert result["routing"] == "multimodal"
        assert result["decision"] in ("Genuine", "Forged")

    def test_enroll_unimodal(self, service):
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_unimodal(samples=trajs, modality="temporal")
        assert enrollment["modality"] == "temporal"
        assert len(enrollment["embeddings"]) == 5

    @pytest.mark.skipif(
        not all(
            os.path.exists(os.path.join(MODEL_WEIGHTS_DIR, TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=s)))
            for s in TEMPORAL_ENSEMBLE_SEEDS
        ),
        reason="Temporal ensemble weights not available in CI (optional, not in releases)",
    )
    def test_temporal_ensemble_mode(self):
        service = SignatureService(device="cpu", use_temporal_ensemble=True)
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)
        probe = self._sample_trajectory(95)

        result = service.verify(
            mode="temporal-multi",
            probe_trajectory=probe,
            reference_spatial=enrollment["spatial_embeddings"],
            reference_temporal=enrollment["temporal_embeddings"],
        )
        assert result["routing"] == "multimodal"
        assert result["decision"] in ("Genuine", "Forged")
        assert not result["degraded"]

    def test_load_temporal_ensemble_returns_models_when_present(self):
        from src.verification import load_temporal_ensemble

        have_ensemble = all(
            os.path.exists(os.path.join(MODEL_WEIGHTS_DIR, TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=s)))
            for s in TEMPORAL_ENSEMBLE_SEEDS
        )
        if not have_ensemble:
            pytest.skip("Ensemble weights not available locally")

        ensemble = load_temporal_ensemble()
        assert ensemble is not None
        assert len(ensemble) == 6
        assert hasattr(ensemble, "encode")

    def test_load_temporal_ensemble_returns_none_when_missing(self, monkeypatch):
        import src.verification as verification
        from src.verification import load_temporal_ensemble

        monkeypatch.setattr(verification.os.path, "exists", lambda _p: False)
        ensemble = load_temporal_ensemble()
        assert ensemble is None

    def test_missing_ensemble_member_raises_instead_of_downgrading(self, monkeypatch):
        """
        Asking for the ensemble and silently getting one model is the failure
        mode that matters here.

        The member checkpoints are gitignored and absent from the public release,
        so on a machine holding only the release the fallback was the *normal*
        path: the service would run at single-model accuracy while its own flag
        said otherwise, and nothing would report it.
        """
        import src.verification as verification

        service = SignatureService(device="cpu", use_temporal_ensemble=True)
        monkeypatch.setattr(verification.os.path, "exists", lambda _p: False)

        with pytest.raises(FileNotFoundError, match="member checkpoints"):
            service._load_temporal_ensemble()

    def test_ensemble_members_come_from_config_not_literals(self):
        """
        The seed list and filename pattern belong in config/settings.py, so the
        shipped ensemble cannot drift from what the service loads.
        """
        assert len(TEMPORAL_ENSEMBLE_SEEDS) == 6
        assert TEMPORAL_ENSEMBLE_TEMPLATE.format(seed=42) == "temporal_dt_s42.pth"

