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

    def test_group_norm_uses_four_channels_per_group_capped_at_32(self):
        """
        Pin `_group_norm`'s actual rule, not divisibility.

        The previous version asserted `num_channels % num_groups == 0`, which
        `nn.GroupNorm.__init__` already raises on, so it could not fail: a
        constructor that always returned `groups=1` would have satisfied it.
        """
        expected = {"compact": [8, 16, 32, 32, 32], "wide": [24, 32, 32, 32]}
        for trunk, want in expected.items():
            got = [m.num_groups for m in SpatialSiamese(trunk=trunk).features
                   if isinstance(m, torch.nn.GroupNorm)]
            assert got == want, f"{trunk}: {got} != {want}"

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
        # Shape alone cannot show the pressure channel was touched. At rate 1.0
        # every sample must be flattened to the imputation value.
        assert torch.allclose(model._apply_pressure_dropout(x.clone())[:, :, 2],
                              torch.ones(4, 30))

    def test_pressure_dropout_eval(self):
        """
        Pressure dropout must not fire under `.eval()`.

        The previous version encoded the same input twice and asserted the two
        embeddings matched. At `pressure_dropout=1.0` the substitution is
        deterministic, so both calls agreed whether or not the `self.training`
        gate existed: verified by deleting `and self.training` from
        `_apply_pressure_dropout`, after which that test still passed.

        Compare against a model that cannot substitute instead. Eval mode must
        reproduce the `pressure_dropout=0.0` result exactly, and must differ from
        what the same weights produce in training mode.
        """
        model = TemporalSiamese(pressure_dropout=1.0)
        reference = TemporalSiamese(pressure_dropout=0.0)
        reference.load_state_dict(model.state_dict())

        x = torch.randn(2, 30, 3)
        x[:, :, 2] = 0.25          # NOT the imputation value, so substitution shows

        model.eval(); reference.eval()
        with torch.no_grad():
            assert torch.allclose(model.encode(x.clone()), reference.encode(x.clone())), \
                "eval-mode encode must match a model with dropout off"

            model.train()
            assert not torch.allclose(model.encode(x.clone()), reference.encode(x.clone())), \
                "training-mode encode must differ, or the substitution never fired"


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
        model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
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
        model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
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

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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


class TestDeriveDeltas:
    """
    `_add_dynamics` appends first differences to whatever it is given. That is
    right for the 3-channel capture contract, where velocity has to be derived,
    and wrong for the derived feature set in `src/data/features.py`, which
    already carries dx, dy, ddx and ddy — appending deltas there makes ddx a
    *third* derivative of raw tablet coordinates, which is amplified noise.

    The width alone cannot say which case a checkpoint is: a 6-wide LSTM is
    either 3 channels with deltas or 6 channels without. So the flag has to be
    stored in the checkpoint, not inferred, or every load path guesses.
    """

    def test_disabling_deltas_halves_the_lstm_input_width(self):
        with_deltas = TemporalSiamese(input_features=16)
        without = TemporalSiamese(input_features=16, derive_deltas=False)
        assert with_deltas.lstm.input_size == 32
        assert without.lstm.input_size == 16

    def test_the_flag_survives_a_checkpoint_round_trip(self):
        """The ambiguity this exists to resolve: same width, different meaning."""
        for channels, derive in ((3, True), (6, False), (16, False), (8, True)):
            original = TemporalSiamese(input_features=channels, derive_deltas=derive)
            restored = TemporalSiamese.from_state_dict(original.state_dict())
            assert restored.derive_deltas is derive
            assert restored.lstm.input_size == original.lstm.input_size

    def test_six_wide_is_not_silently_read_as_three_channels(self):
        """
        Without a stored flag, a 6-channel no-delta model and a 3-channel
        with-delta model are indistinguishable and one would be scored as the
        other — a plausible wrong number, the exact failure this contract exists
        to prevent.
        """
        no_deltas = TemporalSiamese(input_features=6, derive_deltas=False)
        restored = TemporalSiamese.from_state_dict(no_deltas.state_dict())
        assert restored.derive_deltas is False

    def test_a_checkpoint_without_the_flag_defaults_to_deriving(self):
        """Every existing checkpoint predates the flag and used deltas."""
        legacy = TemporalSiamese(input_features=3).state_dict()
        legacy.pop("derive_deltas_flag", None)
        restored = TemporalSiamese.from_state_dict(legacy)
        assert restored.derive_deltas is True
        assert restored.lstm.input_size == 6

    def test_restored_model_scores_identically_without_deltas(self):
        torch.manual_seed(0)
        original = TemporalSiamese(input_features=16, derive_deltas=False).eval()
        restored = TemporalSiamese.from_state_dict(original.state_dict()).eval()
        batch = torch.randn(2, 25, 16)
        with torch.no_grad():
            assert torch.allclose(original.encode(batch), restored.encode(batch))

    def test_padded_batch_still_matches_a_single_sample(self):
        """
        The invariant that guards every temporal change. Packing must still hide
        padding when deltas are off, or a sample's embedding depends on which
        batch it landed in.
        """
        torch.manual_seed(0)
        model = TemporalSiamese(input_features=16, derive_deltas=False).eval()
        short, long = torch.randn(1, 12, 16), torch.randn(1, 30, 16)
        padded = torch.zeros(2, 30, 16)
        padded[0, :12], padded[1] = short[0], long[0]
        lengths = torch.tensor([12, 30])
        with torch.no_grad():
            batched = model.encode(padded, lengths)
            alone = model.encode(short, torch.tensor([12]))
        assert torch.allclose(batched[0], alone[0], atol=1e-6)
