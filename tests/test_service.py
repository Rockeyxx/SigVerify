"""
Tests for SignatureService (Option C High-Level Operational Facade).
"""

import os
import cv2
import numpy as np
import pytest

from config.settings import (
    MIN_BBOX_PIXELS,
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
            reference_spatial_origin=enrollment["spatial_origin"],
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

        # enroll_multimodal labels its images as scans; these are renders, so the
        # test declares the truth and checks the guard lets a matched pair through.
        result = service.verify(
            mode="multimodal",
            probe_image=probe_img,
            probe_trajectory=probe_traj,
            reference_spatial=enrollment["spatial_embeddings"],
            reference_temporal=enrollment["temporal_embeddings"],
            reference_spatial_origin=enrollment["spatial_origin"],
        )
        assert result["routing"] == "multimodal"
        assert not result["degraded"]
        assert result["decision"] in ("Genuine", "Forged")

    # ── spatial reference origin ──────────────────────────────────────────
    # Measured 2026-09-18 on 20 paired SigComp2011 writers, shipped weights:
    # render/render 22.9% EER, scan/scan 24.5%, render probe vs scan refs 57.4%,
    # scan probe vs render refs 54.1%. A render sits ~3.6x further from its own
    # scan than two scans of one writer sit apart, so a mixed pair carries no
    # identity and fusion would average noise into the verdict. Every enrolment
    # therefore declares how its spatial references were made, and verify()
    # refuses to score a spatial probe against references of the other origin.

    def test_enrolments_declare_spatial_origin(self, service):
        from src.data.renderer import render_trajectory_to_image
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        imgs = [render_trajectory_to_image(t) for t in trajs]

        assert service.enroll_from_tablet(trajs)["spatial_origin"] == "render"
        assert service.enroll_multimodal(images=imgs, trajectories=trajs)["spatial_origin"] == "scan"
        assert service.enroll_unimodal(samples=imgs, modality="spatial")["spatial_origin"] == "scan"
        assert "spatial_origin" not in service.enroll_unimodal(samples=trajs, modality="temporal")

    def test_temporal_multi_drops_spatial_when_references_are_scans(self, service):
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)

        result = service.verify(
            mode="temporal-multi",
            probe_trajectory=self._sample_trajectory(95),
            reference_spatial=enrollment["spatial_embeddings"],
            reference_temporal=enrollment["temporal_embeddings"],
            reference_spatial_origin="scan",
        )
        assert result["routing"] == "temporal_only"
        assert result["degraded"]
        assert any("render" in r and "scan" in r for r in result["reasons"])

    def test_multimodal_drops_spatial_when_references_are_renders(self, service):
        from src.data.renderer import render_trajectory_to_image
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)
        probe_traj = self._sample_trajectory(95)

        result = service.verify(
            mode="multimodal",
            probe_image=render_trajectory_to_image(probe_traj),
            probe_trajectory=probe_traj,
            reference_spatial=enrollment["spatial_embeddings"],
            reference_temporal=enrollment["temporal_embeddings"],
            reference_spatial_origin=enrollment["spatial_origin"],
        )
        assert result["routing"] == "temporal_only"
        assert result["degraded"]
        assert any("render" in r and "scan" in r for r in result["reasons"])

    def test_unimodal_spatial_refuses_rendered_references(self, service):
        from src.data.renderer import render_trajectory_to_image
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)

        with pytest.raises(ValueError, match="render"):
            service.verify(
                mode="unimodal",
                probe_image=render_trajectory_to_image(self._sample_trajectory(95)),
                reference_spatial=enrollment["spatial_embeddings"],
                reference_spatial_origin=enrollment["spatial_origin"],
            )

    def test_spatial_references_without_an_origin_are_refused(self, service):
        """
        An undeclared origin must not be guessed: guessing 'scan' would fuse a
        scan probe with rendered references, guessing 'render' the reverse, and
        both measured worse than chance. The error names the enrolment key.
        """
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_from_tablet(trajs)

        with pytest.raises(ValueError, match="spatial_origin"):
            service.verify(
                mode="temporal-multi",
                probe_trajectory=self._sample_trajectory(95),
                reference_spatial=enrollment["spatial_embeddings"],
                reference_temporal=enrollment["temporal_embeddings"],
            )
        with pytest.raises(ValueError, match="spatial_origin"):
            service.verify(
                mode="temporal-multi",
                probe_trajectory=self._sample_trajectory(95),
                reference_spatial=enrollment["spatial_embeddings"],
                reference_temporal=enrollment["temporal_embeddings"],
                reference_spatial_origin="photo",
            )

    def test_temporal_references_need_no_origin(self, service):
        trajs = [self._sample_trajectory(90 + i * 2) for i in range(5)]
        enrollment = service.enroll_unimodal(samples=trajs, modality="temporal")
        result = service.verify(
            mode="unimodal",
            probe_trajectory=self._sample_trajectory(95),
            reference_temporal=enrollment["embeddings"],
        )
        assert result["routing"] == "temporal_only"

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
            reference_spatial_origin=enrollment["spatial_origin"],
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


    def _degenerate_trajectory(self):
        """Raw tablet units — `trajectory_gate` refuses it, exactly as `verify()` does."""
        traj = self._sample_trajectory(80)
        traj[:, 0] = traj[:, 0] * 1200 + 8000
        traj[:, 1] = traj[:, 1] * 1200 + 6000
        return traj

    def _low_quality_image(self):
        """
        Correctly shaped and worthless.

        It clears the MIN_BBOX_PIXELS floor `as_spatial_input` enforces on both
        axes, so the shape check passes it and only `quality_gate` can catch it.
        That is the whole gap: enrolment already rejected grossly wrong *shapes*
        and accepted any *content*.
        """
        img = np.full((MIN_BBOX_PIXELS * 2, MIN_BBOX_PIXELS * 2), 255, dtype=np.uint8)
        cv2.circle(img, (MIN_BBOX_PIXELS, MIN_BBOX_PIXELS), 3, 0, 1)
        return img

    def test_enroll_from_tablet_refuses_a_gate_failing_trajectory(self, service):
        """
        Enrolment used to embed whatever it was handed.

        `verify()` gates every probe it scores; the three `enroll_*` methods
        called `embed()` directly and gated nothing, so a trajectory still in
        raw device units became a stored template. Every later verification
        against that template was meaningless, and the diagnostic — if any —
        pointed at the probe rather than at the reference it was compared to.
        """
        trajs = [self._sample_trajectory(80), self._degenerate_trajectory()]

        with pytest.raises(ValueError) as excinfo:
            service.enroll_from_tablet(trajs)

        message = str(excinfo.value)
        assert "sample 1" in message, "the error must name which sample failed"
        assert "z-scored" in message, "and carry the gate's own reason, not a generic one"

    def test_enroll_multimodal_refuses_a_low_quality_scan(self, service):
        """The spatial half of the same hole: shape was checked, content was not."""
        with pytest.raises(ValueError, match="spatial input gate"):
            service.enroll_multimodal(images=[self._low_quality_image()],
                                      trajectories=[self._sample_trajectory(80)])

    def test_enroll_unimodal_refuses_a_gate_failing_sample_in_either_modality(self, service):
        with pytest.raises(ValueError, match="temporal input gate"):
            service.enroll_unimodal(samples=[self._degenerate_trajectory()],
                                    modality="temporal")

        with pytest.raises(ValueError, match="spatial input gate"):
            service.enroll_unimodal(samples=[self._low_quality_image()],
                                    modality="spatial")
