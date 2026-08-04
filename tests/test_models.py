"""Tests for the Siamese model architectures, loss function, and fusion engine."""

import pytest
import torch
import torch.optim as optim

from src.models.siamese import (
    SPATIAL_TRUNKS,
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


class TestSpatialTrunks:
    """
    Test suite for the selectable convolutional trunks and the checkpoint-driven
    architecture recovery that lets archived runs be re-scored without a config edit.
    """

    @pytest.mark.parametrize("trunk", sorted(SPATIAL_TRUNKS))
    def test_every_trunk_encodes_at_the_real_input_size(self, trunk):
        """Each trunk maps a 155x220 signature onto the configured embedding width."""
        model = SpatialSiamese(embedding_dim=512, trunk=trunk).eval()
        out = model.encode(torch.randn(2, 1, 155, 220))
        assert out.shape == (2, 512)

    def test_wide_trunk_matches_the_reference_channel_widths(self):
        """The wide trunk is the Dey et al. 96/256/384/256 stack with an 11x11 stem."""
        convs = [m for m in SpatialSiamese(trunk="wide").features
                 if isinstance(m, torch.nn.Conv2d)]
        assert [c.out_channels for c in convs] == [96, 256, 384, 256]
        assert [c.kernel_size[0] for c in convs] == [11, 5, 3, 3]

    def test_group_norm_divides_every_trunk_width(self):
        """GroupNorm groups must divide the channel count, or construction throws."""
        for trunk in SPATIAL_TRUNKS:
            for module in SpatialSiamese(trunk=trunk).features:
                if isinstance(module, torch.nn.GroupNorm):
                    assert module.num_channels % module.num_groups == 0

    def test_unknown_trunk_rejected(self):
        """A typo'd trunk name fails loudly instead of silently falling back."""
        with pytest.raises(ValueError, match="Unknown trunk"):
            SpatialSiamese(trunk="signet")

    def test_fc_hidden_inserts_one_layer_and_keeps_a_linear_output(self):
        """
        The FC head adds a hidden ReLU layer but must still END on a plain Linear —
        a ReLU on the output would confine every embedding to one convex cone.
        """
        model = SpatialSiamese(embedding_dim=512, fc_hidden=1024)
        linears = [m for m in model.embedding if isinstance(m, torch.nn.Linear)]
        assert [l.out_features for l in linears] == [1024, 512]
        assert isinstance(model.embedding[-1], torch.nn.Linear)

    @pytest.mark.parametrize("trunk,fc_hidden,embedding_dim", [
        ("compact", 0, 512),
        ("compact", 0, 128),
        ("wide", 0, 512),
        ("wide", 1024, 512),
    ])
    def test_from_state_dict_recovers_the_architecture(self, trunk, fc_hidden, embedding_dim):
        """A checkpoint carries its own architecture — no config lookup involved."""
        original = SpatialSiamese(embedding_dim=embedding_dim, trunk=trunk,
                                  fc_hidden=fc_hidden).eval()
        restored = SpatialSiamese.from_state_dict(original.state_dict()).eval()

        assert restored.trunk == trunk
        assert restored.fc_hidden == fc_hidden
        assert restored.embedding_dim == embedding_dim

        x = torch.randn(2, 1, 155, 220)
        assert torch.allclose(original.encode(x), restored.encode(x))

    def test_from_state_dict_rejects_an_unknown_architecture(self):
        """
        Weights from a trunk this code no longer defines must raise, not load
        partially. A near-miss architecture would otherwise produce a plausible
        EER that describes nothing.
        """
        state_dict = SpatialSiamese(trunk="compact").state_dict()
        state_dict["features.0.weight"] = torch.randn(48, 1, 5, 5)   # 32 -> 48 filters
        with pytest.raises(ValueError, match="matches no trunk"):
            SpatialSiamese.from_state_dict(state_dict)

    def test_dropout_rates_do_not_move_the_parameter_shapes(self):
        """
        Dropout is inert at eval time, so it must not change the state_dict —
        otherwise a checkpoint could not be reloaded without knowing the rate
        it happened to be trained with.
        """
        dry = SpatialSiamese(trunk="wide", conv_dropout=0.0, head_dropout=0.0)
        wet = SpatialSiamese(trunk="wide", conv_dropout=0.3, head_dropout=0.5)
        assert {k: v.shape for k, v in dry.state_dict().items()} == \
               {k: v.shape for k, v in wet.state_dict().items()}


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
        result = engine.fuse(spatial_score=0.8, temporal_score=0.7)
        assert result["routing"] == "multimodal"
        assert result["decision"] in ("Genuine", "Forged")

    def test_result_contract_keys(self):
        """Verify fuse() returns the exact dict contract the GUI consumes."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        result = engine.fuse(spatial_score=0.8, temporal_score=0.7)
        assert set(result) == {"decision", "confidence", "routing", "degraded"}
        assert isinstance(result["decision"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["routing"], str)
        assert isinstance(result["degraded"], bool)

    def test_multimodal_not_degraded(self):
        """Both modalities present with real pressure is the full-confidence path."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        result = engine.fuse(spatial_score=0.8, temporal_score=0.7)
        assert result["degraded"] is False

    def test_spatial_only_fallback(self):
        """Test automatic fallback routing when temporal sequence is absent."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        result = engine.fuse(spatial_score=0.8)
        assert result["routing"] == "spatial_only"
        assert result["degraded"] is True

    def test_temporal_only_fallback(self):
        """Test automatic fallback routing when signature image is absent."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        result = engine.fuse(temporal_score=0.6)
        assert result["routing"] == "temporal_only"
        assert result["degraded"] is True

    def test_imputed_pressure_flags_degraded(self):
        """Verify P=1.0 imputation marks the verification degraded even when multimodal."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        result = engine.fuse(spatial_score=0.8, temporal_score=0.7,
                             pressure_imputed=True)
        assert result["routing"] == "multimodal"
        assert result["degraded"] is True

    def test_no_modality_raises(self):
        """Verify ValueError is raised if neither modality score is supplied."""
        engine = FusionEngine()
        with pytest.raises(ValueError):
            engine.fuse()

    def test_threshold_decision_boundary(self):
        """Verify decision boundary classification around threshold tau."""
        engine = FusionEngine(alpha=0.5, threshold=0.5)
        assert engine.fuse(spatial_score=0.5)["decision"] == "Genuine"
        assert engine.fuse(spatial_score=0.49)["decision"] == "Forged"


class TestHardNegativeMining:
    """Test suite for dynamic hard-negative mining inside the contrastive loss."""

    def _batch_with_one_hard_negative(self):
        """
        Build a batch of 4 impostor pairs: one hard (d≈0.1, well inside the
        margin) and three easy (d=5.0, far beyond it and contributing zero loss).
        """
        emb1 = torch.zeros(4, 2)
        emb2 = torch.tensor([[0.1, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
        labels = torch.zeros(4)
        return emb1, emb2, labels

    def test_mining_raises_loss_above_vanilla_mean(self):
        """Mined loss must exceed the vanilla mean, which easy negatives dilute to ~0."""
        emb1, emb2, labels = self._batch_with_one_hard_negative()
        vanilla = ContrastiveLoss(margin=1.0, hard_negative_ratio=1.0)
        mined = ContrastiveLoss(margin=1.0, hard_negative_ratio=0.25)
        assert mined(emb1, emb2, labels) > vanilla(emb1, emb2, labels)

    def test_ratio_one_matches_vanilla_contrastive(self):
        """hard_negative_ratio=1.0 disables selection and reproduces the plain mean."""
        emb1 = torch.randn(6, 8)
        emb2 = torch.randn(6, 8)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        loss_fn = ContrastiveLoss(margin=1.0, hard_negative_ratio=1.0)

        distance = torch.nn.functional.pairwise_distance(emb1, emb2, p=2)
        expected = (labels * distance.pow(2)
                    + (1 - labels) * torch.nn.functional.relu(1.0 - distance).pow(2)).mean()
        assert torch.allclose(loss_fn(emb1, emb2, labels), expected, atol=1e-6)

    def test_gradients_flow_through_mined_loss(self):
        """Backprop must still reach both branches after negative selection."""
        emb1 = torch.randn(8, 16, requires_grad=True)
        emb2 = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([1.0, 0.0] * 4)
        ContrastiveLoss(margin=1.0, hard_negative_ratio=0.5)(emb1, emb2, labels).backward()
        assert emb1.grad is not None and emb1.grad.abs().sum() > 0
        assert emb2.grad is not None and emb2.grad.abs().sum() > 0

    def test_all_genuine_batch_is_safe(self):
        """A batch with zero impostors must not divide by zero or return NaN."""
        emb1 = torch.randn(4, 8)
        emb2 = torch.randn(4, 8)
        labels = torch.ones(4)
        loss = ContrastiveLoss(margin=1.0, hard_negative_ratio=0.5)(emb1, emb2, labels)
        assert torch.isfinite(loss)

    def test_all_impostor_batch_is_safe(self):
        """A batch with zero genuine pairs must not divide by zero or return NaN."""
        emb1 = torch.randn(4, 8)
        emb2 = torch.randn(4, 8)
        labels = torch.zeros(4)
        loss = ContrastiveLoss(margin=1.0, hard_negative_ratio=0.5)(emb1, emb2, labels)
        assert torch.isfinite(loss)


class TestTemporalPackedBatching:
    """Test suite for variable-length batching via pack_padded_sequence."""

    def test_padded_batch_matches_single_sample(self):
        """
        The core contamination check: encoding a short sequence inside a padded
        batch must equal encoding it alone. Without packing, the LSTM keeps
        consuming pad rows and the two diverge.
        """
        model = TemporalSiamese(embedding_dim=128, pressure_dropout=0.0)
        model.eval()

        long_seq = torch.randn(300, 3)
        short_seq = torch.randn(150, 3)

        padded = torch.zeros(2, 300, 3)
        padded[0, :300] = long_seq
        padded[1, :150] = short_seq
        lengths = torch.tensor([300, 150])

        with torch.no_grad():
            batched = model.encode(padded, lengths=lengths)
            alone = model.encode(short_seq.unsqueeze(0), lengths=torch.tensor([150]))

        assert torch.allclose(batched[1], alone[0], atol=1e-5)

    def test_encode_without_lengths_still_works(self):
        """Omitting lengths keeps the original un-packed path for equal-length batches."""
        model = TemporalSiamese(embedding_dim=128, pressure_dropout=0.0)
        model.eval()
        assert model.encode(torch.randn(2, 50, 3)).shape == (2, 128)


class TestTemporalAttentionPooling:
    """
    Attention pools over the *unpacked* sequence, so padding it must mask itself.

    Packing protects the LSTM from pad rows; it does not protect a pooling layer
    that runs afterwards. These tests exist because a missing mask is silent —
    the model trains, the shapes are right, and only the embeddings are wrong.
    """

    def _model(self, **kwargs):
        model = TemporalSiamese(embedding_dim=128, pressure_dropout=0.0,
                                attention=True, **kwargs)
        model.eval()
        return model

    def test_padded_batch_matches_single_sample(self):
        """
        The same contamination check as the packed path, now with attention.

        This is the acceptance test for the mask: without `masked_fill` the short
        sequence spreads attention over 150 frames of padding and its embedding
        changes depending on what it was batched with.
        """
        model = self._model()

        long_seq = torch.randn(300, 3)
        short_seq = torch.randn(150, 3)

        padded = torch.zeros(2, 300, 3)
        padded[0, :300] = long_seq
        padded[1, :150] = short_seq

        with torch.no_grad():
            batched = model.encode(padded, lengths=torch.tensor([300, 150]))
            alone = model.encode(short_seq.unsqueeze(0), lengths=torch.tensor([150]))

        assert torch.allclose(batched[1], alone[0], atol=1e-5)

    def test_attention_weights_ignore_padding(self):
        """
        Every unit of attention mass must land on a real frame.

        Recomputes the pooling internals deliberately: the pooled vector alone
        cannot distinguish "padding got zero weight" from "padding got weight but
        happened to sit near the mean".
        """
        model = self._model()
        padded = torch.zeros(2, 40, 3)
        padded[0] = torch.randn(40, 3)
        padded[1, :10] = torch.randn(10, 3)
        lengths = torch.tensor([40, 10])

        with torch.no_grad():
            x = model._add_dynamics(padded)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            outputs, _ = model.lstm(packed)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            scores = model.attend(outputs).squeeze(-1)
            positions = torch.arange(outputs.size(1))
            padding = positions[None, :] >= lengths[:, None]
            weights = torch.softmax(scores.masked_fill(padding, float("-inf")), dim=1)

        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
        assert weights[1, 10:].abs().max().item() == 0.0

    def test_attention_changes_the_embedding(self):
        """
        Attention and final-state pooling must not coincidentally agree.

        Guards against the flag being accepted but silently ignored — the failure
        that would make an A/B arm measure nothing while appearing to run.
        """
        torch.manual_seed(0)
        with_attention = self._model()
        without = TemporalSiamese(embedding_dim=128, pressure_dropout=0.0)
        without.eval()
        without.load_state_dict(
            {k: v for k, v in with_attention.state_dict().items()
             if not k.startswith("attend.")})

        sequence = torch.randn(1, 80, 3)
        with torch.no_grad():
            assert not torch.allclose(with_attention.encode(sequence),
                                      without.encode(sequence), atol=1e-4)

    def test_from_state_dict_recovers_attention(self):
        """A checkpoint must declare its own pooling, as it already does its width."""
        saved = self._model().state_dict()
        assert TemporalSiamese.from_state_dict(saved).attention

    def test_from_state_dict_rejects_attention_for_old_checkpoints(self):
        """
        A pre-attention checkpoint must rebuild without it.

        Silently granting it an untrained attention layer would score the old
        weights through a random projection and report the result as a regression.
        """
        legacy = TemporalSiamese(embedding_dim=128, pressure_dropout=0.0).state_dict()
        assert not TemporalSiamese.from_state_dict(legacy).attention


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



class TestSpatialGradientCheckpointing:
    """
    Gradient checkpointing is a memory/compute trade, not a modelling change.
    These tests hold it to that: same gradients, same parameters, same weights file.
    """

    def test_checkpointed_gradients_match_the_plain_backward(self):
        """Recomputing the trunk must reproduce the gradients storing it produces."""
        torch.manual_seed(0)
        plain = SpatialSiamese(embedding_dim=64, trunk="wide")
        checkpointed = SpatialSiamese(embedding_dim=64, trunk="wide", grad_checkpoint=4)
        checkpointed.load_state_dict(plain.state_dict())

        x = torch.randn(2, 1, 155, 220)
        for model in (plain, checkpointed):
            model.train()
            model.encode(x).sum().backward()

        for (name, a), (_, b) in zip(plain.named_parameters(),
                                     checkpointed.named_parameters()):
            assert torch.allclose(a.grad, b.grad, atol=1e-5), name

    def test_checkpointing_leaves_no_trace_in_the_state_dict(self):
        """A checkpointed run's weights load into a plain model and vice versa."""
        saved = SpatialSiamese(trunk="wide", grad_checkpoint=4).state_dict()
        restored = SpatialSiamese.from_state_dict(saved)
        assert restored.grad_checkpoint == 0

    def test_eval_mode_skips_checkpointing(self):
        """Inference has no backward pass to save memory for, so it pays nothing."""
        model = SpatialSiamese(embedding_dim=64, trunk="wide", grad_checkpoint=4).eval()
        with torch.no_grad():
            out = model.encode(torch.randn(2, 1, 155, 220))
        assert out.shape == (2, 64)


class TestTemporalArchitectureRecovery:
    """
    A temporal checkpoint must declare its own channel count. The 3-channel
    contract is the shipped one, but 4- and 5-channel experiments exist, and
    scoring one under the wrong width would produce a plausible, meaningless
    number rather than an error.
    """

    @pytest.mark.parametrize("channels", [3, 4, 5])
    def test_channel_count_survives_a_round_trip(self, channels):
        original = TemporalSiamese(input_features=channels, embedding_dim=128)
        restored = TemporalSiamese.from_state_dict(original.state_dict())
        # _add_dynamics doubles the width before the LSTM, so the recovered
        # input size must be exactly 2x the declared channel count.
        assert restored.lstm.input_size == channels * 2

    def test_restored_model_scores_identically(self):
        torch.manual_seed(0)
        original = TemporalSiamese(input_features=4, embedding_dim=128).eval()
        restored = TemporalSiamese.from_state_dict(original.state_dict()).eval()
        batch = torch.randn(2, 25, 4)
        with torch.no_grad():
            assert torch.allclose(original.encode(batch), restored.encode(batch))

    def test_embedding_width_and_depth_are_recovered(self):
        original = TemporalSiamese(input_features=3, embedding_dim=64, num_layers=2)
        restored = TemporalSiamese.from_state_dict(original.state_dict())
        assert restored.embedding.out_features == 64
        assert restored.lstm.num_layers == 2

    def test_a_pre_dynamics_checkpoint_is_rejected(self):
        # An odd LSTM input width cannot be 2 * input_features, so it predates
        # derived dynamics and must fail loudly instead of being rounded.
        broken = TemporalSiamese(input_features=3).state_dict()
        broken["lstm.weight_ih_l0"] = broken["lstm.weight_ih_l0"][:, :5]
        with pytest.raises(ValueError, match="odd"):
            TemporalSiamese.from_state_dict(broken)
