"""Tests for the Siamese model architectures, loss function, and fusion engine."""

import torch
import pytest


class TestSpatialSiamese:
    """Test the CNN backbone for offline signature verification."""

    def test_encode_output_shape(self):
        """[B, 1, 256, 256] → [B, 512]"""
        from src.models.siamese import SpatialSiamese
        model = SpatialSiamese(embedding_dim=512)
        x = torch.randn(2, 1, 256, 256)
        out = model.encode(x)
        assert out.shape == (2, 512)

    def test_forward_returns_two_embeddings(self):
        from src.models.siamese import SpatialSiamese
        model = SpatialSiamese(embedding_dim=512)
        x1 = torch.randn(2, 1, 256, 256)
        x2 = torch.randn(2, 1, 256, 256)
        emb1, emb2 = model.forward(x1, x2)
        assert emb1.shape == (2, 512)
        assert emb2.shape == (2, 512)

    def test_weight_sharing(self):
        """Both branches must share the same weights (Siamese property)."""
        from src.models.siamese import SpatialSiamese
        model = SpatialSiamese()
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        emb_a = model.encode(x)
        emb_b = model.encode(x)
        assert torch.allclose(emb_a, emb_b)

    def test_no_batchnorm(self):
        """Architecture must use GroupNorm/LayerNorm, never BatchNorm."""
        from src.models.siamese import SpatialSiamese
        model = SpatialSiamese()
        for module in model.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))


class TestTemporalSiamese:
    """Test the LSTM backbone for online signature verification."""

    def test_encode_output_shape(self):
        """[B, L, 3] → [B, 128]"""
        from src.models.siamese import TemporalSiamese
        model = TemporalSiamese(embedding_dim=128)
        x = torch.randn(2, 50, 3)  # 2 samples, 50 timesteps, 3 features
        out = model.encode(x)
        assert out.shape == (2, 128)

    def test_variable_sequence_length(self):
        """Must handle variable-length sequences."""
        from src.models.siamese import TemporalSiamese
        model = TemporalSiamese(embedding_dim=128)
        short = torch.randn(1, 20, 3)
        long = torch.randn(1, 200, 3)
        assert model.encode(short).shape == (1, 128)
        assert model.encode(long).shape == (1, 128)

    def test_pressure_dropout_training(self):
        """During training, pressure channel should be dropped for some samples."""
        from src.models.siamese import TemporalSiamese
        model = TemporalSiamese(pressure_dropout=1.0)  # Force 100% dropout
        model.train()
        x = torch.randn(4, 30, 3)
        x[:, :, 2] = torch.rand(4, 30) * 5  # Non-trivial pressure values
        emb = model.encode(x.clone())
        assert emb.shape == (4, 128)

    def test_pressure_dropout_eval(self):
        """During eval, pressure dropout must NOT be applied."""
        from src.models.siamese import TemporalSiamese
        model = TemporalSiamese(pressure_dropout=1.0)
        model.eval()
        x = torch.randn(2, 30, 3)
        x[:, :, 2] = 5.0  # Set specific pressure
        emb1 = model.encode(x.clone())
        emb2 = model.encode(x.clone())
        assert torch.allclose(emb1, emb2)


class TestContrastiveLoss:
    """Test the contrastive loss function."""

    def test_genuine_pair_loss(self):
        """Genuine pairs (label=1): loss should be d²."""
        from src.models.siamese import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[0.0, 1.0]])
        label = torch.tensor([1.0])
        loss = loss_fn(emb1, emb2, label)
        expected_d = torch.sqrt(torch.tensor(2.0))  # L2 distance
        expected_loss = expected_d ** 2
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_impostor_pair_within_margin(self):
        """Impostor pairs (label=0) within margin: loss = max(0, m-d)²."""
        from src.models.siamese import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=2.0)
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[0.5, 0.0]])
        label = torch.tensor([0.0])
        loss = loss_fn(emb1, emb2, label)
        expected = (2.0 - 0.5) ** 2  # (margin - distance)²
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_impostor_pair_beyond_margin(self):
        """Impostor pairs beyond margin: loss should be 0."""
        from src.models.siamese import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[2.0, 0.0]])  # distance = 2.0 > margin 1.0
        label = torch.tensor([0.0])
        loss = loss_fn(emb1, emb2, label)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_gradient_flows(self):
        """Gradients must flow back through the loss to both embeddings."""
        from src.models.siamese import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.randn(4, 512, requires_grad=True)
        emb2 = torch.randn(4, 512, requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(emb1, emb2, labels)
        loss.backward()
        assert emb1.grad is not None
        assert emb2.grad is not None


class TestFusionEngine:
    """Test the late-fusion score combination."""

    def test_multimodal_fusion(self):
        from src.models.siamese import FusionEngine
        engine = FusionEngine(alpha=0.6, threshold=0.5)
        decision, confidence, routing = engine.fuse(
            spatial_score=0.8, temporal_score=0.7
        )
        assert routing == "multimodal"
        assert isinstance(decision, str)
        assert decision in ("Genuine", "Forged")

    def test_spatial_only_fallback(self):
        from src.models.siamese import FusionEngine
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        decision, confidence, routing = engine.fuse(spatial_score=0.8)
        assert routing == "spatial_only"

    def test_temporal_only_fallback(self):
        from src.models.siamese import FusionEngine
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        decision, confidence, routing = engine.fuse(temporal_score=0.6)
        assert routing == "temporal_only"

    def test_no_modality_raises(self):
        from src.models.siamese import FusionEngine
        engine = FusionEngine()
        with pytest.raises(ValueError):
            engine.fuse()

    def test_threshold_decision_boundary(self):
        """Score exactly at threshold → Genuine. Below → Forged."""
        from src.models.siamese import FusionEngine
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        decision_at, _, _ = engine.fuse(spatial_score=0.5)
        decision_below, _, _ = engine.fuse(spatial_score=0.49)
        assert decision_at == "Genuine"
        assert decision_below == "Forged"
