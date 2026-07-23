"""Tests for the Siamese model architectures, loss function, and fusion engine."""

import pytest
import torch
import torch.optim as optim

from src.models.siamese import (
    ContrastiveLoss,
    FusionEngine,
    SpatialSiamese,
    TemporalSiamese,
)


class TestSpatialSiamese:
    """Test suite for the Siamese CNN offline signature verification backbone."""

    def test_encode_output_shape(self):
        """Verify feature encoder maps input tensor [B, 1, 256, 256] → embedding vector [B, 512]."""
        model = SpatialSiamese(embedding_dim=512)
        x = torch.randn(2, 1, 256, 256)
        out = model.encode(x)
        assert out.shape == (2, 512)

    def test_forward_returns_two_embeddings(self):
        """Ensure pair forward pass extracts embeddings for both signature inputs."""
        model = SpatialSiamese(embedding_dim=512)
        x1 = torch.randn(2, 1, 256, 256)
        x2 = torch.randn(2, 1, 256, 256)
        emb1, emb2 = model.forward(x1, x2)
        assert emb1.shape == (2, 512)
        assert emb2.shape == (2, 512)

    def test_weight_sharing(self):
        """Verify Siamese twin branches share identical parameters (deterministic encoding)."""
        model = SpatialSiamese()
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        emb_a = model.encode(x)
        emb_b = model.encode(x)
        assert torch.allclose(emb_a, emb_b)

    def test_no_batchnorm(self):
        """Verify network uses GroupNorm/LayerNorm, adhering to pair-sampling metric learning rules."""
        model = SpatialSiamese()
        for module in model.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))


class TestTemporalSiamese:
    """Test suite for the Siamese LSTM online signature verification backbone."""

    def test_encode_output_shape(self):
        """Verify trajectory encoder maps sequence tensor [B, L, 3] → embedding vector [B, 128]."""
        model = TemporalSiamese(embedding_dim=128)
        x = torch.randn(2, 50, 3)  # 2 signature samples, 50 trajectory points, 3 features (X, Y, P)
        out = model.encode(x)
        assert out.shape == (2, 128)

    def test_variable_sequence_length(self):
        """Ensure model cleanly handles variable sequence lengths (different stroke counts)."""
        model = TemporalSiamese(embedding_dim=128)
        short = torch.randn(1, 20, 3)
        long = torch.randn(1, 200, 3)
        assert model.encode(short).shape == (1, 128)
        assert model.encode(long).shape == (1, 128)

    def test_pressure_dropout_training(self):
        """Verify pressure channel substitution (P=1.0) executes during training phase."""
        model = TemporalSiamese(pressure_dropout=1.0)  # Force 100% pressure dropout
        model.train()
        x = torch.randn(4, 30, 3)
        x[:, :, 2] = torch.rand(4, 30) * 5  # Non-trivial raw pressure readings
        emb = model.encode(x.clone())
        assert emb.shape == (4, 128)

    def test_pressure_dropout_eval(self):
        """Ensure pressure dropout is strictly disabled during inference evaluation."""
        model = TemporalSiamese(pressure_dropout=1.0)
        model.eval()
        x = torch.randn(2, 30, 3)
        x[:, :, 2] = 5.0  # Constant pressure value
        emb1 = model.encode(x.clone())
        emb2 = model.encode(x.clone())
        assert torch.allclose(emb1, emb2)


class TestContrastiveLoss:
    """Test suite for pair metric learning contrastive loss calculation."""

    def test_genuine_pair_loss(self):
        """Verify genuine pair loss evaluates strictly to Euclidean distance square (d^2)."""
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[0.0, 1.0]])
        label = torch.tensor([1.0])
        loss = loss_fn(emb1, emb2, label)
        expected_d = torch.sqrt(torch.tensor(2.0))  # L2 distance = sqrt(1^2 + 1^2)
        expected_loss = expected_d ** 2
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_impostor_pair_within_margin(self):
        """Verify forgery loss within margin evaluates to (margin - d)^2."""
        loss_fn = ContrastiveLoss(margin=2.0)
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[0.5, 0.0]])
        label = torch.tensor([0.0])
        loss = loss_fn(emb1, emb2, label)
        expected = (2.0 - 0.5) ** 2  # (margin - distance)^2
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_impostor_pair_beyond_margin(self):
        """Verify forgery pair beyond margin yields zero loss (no gradient penalty)."""
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[2.0, 0.0]])  # distance = 2.0 > margin 1.0
        label = torch.tensor([0.0])
        loss = loss_fn(emb1, emb2, label)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_gradient_flows(self):
        """Verify backpropagation gradients propagate to both embedding branches."""
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.randn(4, 512, requires_grad=True)
        emb2 = torch.randn(4, 512, requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(emb1, emb2, labels)
        loss.backward()
        assert emb1.grad is not None
        assert emb2.grad is not None


class TestFusionEngine:
    """Test suite for late-fusion decision score combining."""

    def test_multimodal_fusion(self):
        """Test dual-branch multimodal score weighted sum decision."""
        engine = FusionEngine(alpha=0.6, threshold=0.5)
        decision, confidence, routing = engine.fuse(
            spatial_score=0.8, temporal_score=0.7
        )
        assert routing == "multimodal"
        assert isinstance(decision, str)
        assert decision in ("Genuine", "Forged")

    def test_spatial_only_fallback(self):
        """Test automatic fallback routing when temporal sequence is absent."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        decision, confidence, routing = engine.fuse(spatial_score=0.8)
        assert routing == "spatial_only"

    def test_temporal_only_fallback(self):
        """Test automatic fallback routing when signature image is absent."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        decision, confidence, routing = engine.fuse(temporal_score=0.6)
        assert routing == "temporal_only"

    def test_no_modality_raises(self):
        """Verify ValueError is raised if neither modality score is supplied."""
        engine = FusionEngine()
        with pytest.raises(ValueError):
            engine.fuse()

    def test_threshold_decision_boundary(self):
        """Verify decision boundary classification around threshold tau."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        decision_at, _, _ = engine.fuse(spatial_score=0.5)
        decision_below, _, _ = engine.fuse(spatial_score=0.49)
        assert decision_at == "Genuine"
        assert decision_below == "Forged"


class TestWeightsSaveLoad:
    """Verify PyTorch state_dict save and load round-trip fidelity."""

    def test_spatial_save_load_roundtrip(self, tmp_path):
        """Verify SpatialSiamese model weights save/load without embedding deviation."""
        model = SpatialSiamese(embedding_dim=512)
        x = torch.randn(1, 1, 256, 256)

        model.eval()
        with torch.no_grad():
            emb_before = model.encode(x)

        path = tmp_path / "spatial.pth"
        torch.save(model.state_dict(), path)

        model2 = SpatialSiamese(embedding_dim=512)
        model2.load_state_dict(torch.load(path, map_location="cpu"))
        model2.eval()
        with torch.no_grad():
            emb_after = model2.encode(x)

        assert torch.allclose(emb_before, emb_after)

    def test_temporal_save_load_roundtrip(self, tmp_path):
        """Verify TemporalSiamese model weights save/load without embedding deviation."""
        model = TemporalSiamese(embedding_dim=128)
        x = torch.randn(1, 50, 3)

        model.eval()
        with torch.no_grad():
            emb_before = model.encode(x)

        path = tmp_path / "temporal.pth"
        torch.save(model.state_dict(), path)

        model2 = TemporalSiamese(embedding_dim=128)
        model2.load_state_dict(torch.load(path, map_location="cpu"))
        model2.eval()
        with torch.no_grad():
            emb_after = model2.encode(x)

        assert torch.allclose(emb_before, emb_after)

    def test_checkpoint_resume(self, tmp_path):
        """Verify training checkpoint serialization (model state, optimizer state, epoch, loss)."""
        model = SpatialSiamese(embedding_dim=512)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epoch = 42
        loss = 0.123

        path = tmp_path / "checkpoint.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, path)

        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["epoch"] == 42
        assert abs(checkpoint["loss"] - 0.123) < 1e-6

        model2 = SpatialSiamese(embedding_dim=512)
        model2.load_state_dict(checkpoint["model_state_dict"])
        assert sum(p.numel() for p in model2.parameters()) == sum(p.numel() for p in model.parameters())

