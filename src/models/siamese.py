"""
Dual-Siamese Signature Verification — Model Architectures

This module defines the neural network components for signature verification:
  - SpatialSiamese: CNN for offline (scanned image) signatures
  - TemporalSiamese: LSTM for online (pen trajectory) signatures
  - ContrastiveLoss: Distance-based loss with hard-negative mining
  - FusionEngine: Late-fusion combining spatial and temporal scores
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from config.settings import (
    SPATIAL_CONV_DROPOUT, SPATIAL_EMBEDDING_DIM, SPATIAL_FC_HIDDEN,
    SPATIAL_HEAD_DROPOUT, SPATIAL_TRUNK,
    TEMPORAL_EMBEDDING_DIM, TEMPORAL_HIDDEN_SIZE, TEMPORAL_INPUT_FEATURES,
    TEMPORAL_NUM_LAYERS, PRESSURE_DROPOUT_RATE,
    TEMPORAL_CONV_FILTERS, TEMPORAL_CONV_KERNEL,
    ALPHA_DEFAULT, DECISION_THRESHOLD,
)


# Convolutional trunk configurations for the spatial branch.
# Each stage specifies (out_channels, kernel_size, pool_after, dropout_after).
#   - compact: 5 convolutional stages with pooling after each stage.
#   - wide: 4 convolutional stages based on the SigNet architecture.
SPATIAL_TRUNKS = {
    "compact": (
        (32, 5, True, False),
        (64, 5, True, False),
        (128, 3, True, False),
        (256, 3, True, False),
        (256, 3, True, False),
    ),
    "wide": (
        (96, 11, True, False),
        (256, 5, True, True),
        (384, 3, False, False),
        (256, 3, True, True),
    ),
}


def _group_norm(channels):
    """
    Create a GroupNorm layer with roughly 4 channels per group (up to 32 groups).
    GroupNorm works well with small batch sizes in Siamese training.
    """
    groups = min(32, channels // 4)
    return nn.GroupNorm(groups, channels)


def _adaptive_windows(length, grid):
    """Calculate pooling window start and end coordinates along one dimension."""
    return [(i * length // grid, -(-(i + 1) * length // grid)) for i in range(grid)]


def _pool_windows(x, grid, reduce):
    """Apply a reduction function over a 2D grid of pooling windows."""
    rows, cols = _adaptive_windows(x.shape[-2], grid), _adaptive_windows(x.shape[-1], grid)
    return torch.stack([torch.stack([reduce(x[..., r0:r1, c0:c1]) for c0, c1 in cols], dim=-1)
                        for r0, r1 in rows], dim=-2)


def adaptive_max_pool2d_deterministic(x, grid):
    """
    Adaptive max pooling with a deterministic backward pass.
    Standard PyTorch CUDA adaptive max pooling can introduce non-determinism.
    """
    return _pool_windows(x, grid, lambda win: win.amax(dim=(-2, -1)))


def adaptive_avg_pool2d_deterministic(x, grid):
    """
    Adaptive average pooling with a deterministic backward pass.
    Produces repeatable results across different runs.
    """
    return _pool_windows(x, grid, lambda win: win.mean(dim=(-2, -1)))


class SpatialSiamese(nn.Module):
    """
    Siamese CNN for offline signature verification.

    Accepts grayscale signature images [B, 1, H, W] and projects them into
    a continuous embedding space for distance comparison.
    """

    def __init__(self, embedding_dim=SPATIAL_EMBEDDING_DIM, head_dropout=SPATIAL_HEAD_DROPOUT,
                 trunk=SPATIAL_TRUNK, conv_dropout=SPATIAL_CONV_DROPOUT,
                 fc_hidden=SPATIAL_FC_HIDDEN, grad_checkpoint=0,
                 pool_grid=4, pool_channels=None):
        super().__init__()
        if trunk not in SPATIAL_TRUNKS:
            raise ValueError(f"Unknown trunk '{trunk}' — expected one of {sorted(SPATIAL_TRUNKS)}")

        self.embedding_dim = embedding_dim
        self.trunk = trunk
        self.fc_hidden = fc_hidden
        self.grad_checkpoint = grad_checkpoint

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel, pool, drop in SPATIAL_TRUNKS[trunk]:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                _group_norm(out_channels),
                nn.ReLU(inplace=not grad_checkpoint),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            if drop:
                layers.append(nn.Dropout2d(conv_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # Regional pooling grid size (pool_grid x pool_grid regions).
        self.pool_grid = int(pool_grid)

        # Optional 1x1 convolution before pooling to reduce feature channels.
        self.pool_channels = None if pool_channels is None else int(pool_channels)
        if self.pool_channels is not None:
            self.reduce = nn.Conv2d(in_channels, self.pool_channels, kernel_size=1)
            in_channels = self.pool_channels

        self.pool_avg = nn.AdaptiveAvgPool2d((self.pool_grid, self.pool_grid))
        self.pool_max = nn.AdaptiveMaxPool2d((self.pool_grid, self.pool_grid))

        # Combine average and max pooling features to capture both density and peaks.
        pooled_features = 2 * in_channels * self.pool_grid * self.pool_grid

        # Linear projection layers to produce the final embedding vector.
        head: list[nn.Module] = [nn.Dropout(head_dropout)]
        if fc_hidden:
            head += [
                nn.Linear(pooled_features, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(fc_hidden, embedding_dim),
            ]
        else:
            head.append(nn.Linear(pooled_features, embedding_dim))
        self.embedding = nn.Sequential(*head)

    @classmethod
    def from_state_dict(cls, state_dict, **overrides):
        """
        Reconstruct a SpatialSiamese model using parameters saved in state_dict.
        Recovers the convolutional trunk, pooling grid, and head dimensions.
        """
        convs = sorted(
            (int(key.split(".")[1]), tuple(value.shape))
            for key, value in state_dict.items()
            if key.startswith("features.") and key.endswith(".weight") and value.dim() == 4
        )
        signature = tuple((shape[0], shape[2]) for _, shape in convs)

        trunk = next(
            (name for name, spec in SPATIAL_TRUNKS.items()
             if tuple((channels, kernel) for channels, kernel, _, _ in spec) == signature),
            None,
        )
        if trunk is None:
            raise ValueError(
                f"Checkpoint convolution signature {signature} matches no trunk in "
                f"SPATIAL_TRUNKS. It was trained with an architecture this code no "
                f"longer defines — re-add the trunk rather than scoring it as another."
            )

        # Detect whether the checkpoint uses a two-layer FC head or a single linear layer.
        if "embedding.4.weight" in state_dict:
            fc_hidden, embedding_dim = state_dict["embedding.4.weight"].shape[::-1]
        elif "embedding.1.weight" in state_dict:
            fc_hidden, embedding_dim = 0, state_dict["embedding.1.weight"].shape[0]
        else:
            raise ValueError(
                "Checkpoint has no recognizable embedding head. It was saved before "
                "the head became a Sequential, which also means before the 4x4 pooling "
                "grid and the inverted preprocessing — it cannot be scored against "
                "the current pipeline at all."
            )

        # Recover pooling grid from the linear input dimension.
        in_features = state_dict["embedding.1.weight"].shape[1]
        pool_channels = (int(state_dict["reduce.weight"].shape[0])
                         if "reduce.weight" in state_dict else None)
        trunk_out = pool_channels or SPATIAL_TRUNKS[trunk][-1][0]
        cells = in_features / (2 * trunk_out)
        pool_grid = int(round(cells ** 0.5))
        if pool_grid < 1 or 2 * trunk_out * pool_grid ** 2 != in_features:
            raise ValueError(
                f"Checkpoint projection width {in_features} is not "
                f"2 x {trunk_out} x grid^2 for any integer grid, so the pooling "
                f"grid it was trained with cannot be recovered. Scoring it under "
                f"the default grid would silently reshape the embedding.")

        model = cls(embedding_dim=int(embedding_dim), trunk=trunk,
                    fc_hidden=int(fc_hidden), pool_grid=pool_grid,
                    pool_channels=pool_channels, **overrides)
        model.load_state_dict(state_dict)
        return model.eval()

    def encode(self, x):
        """
        Extract the continuous embedding vector for an input image.
        Input shape: [B, 1, H, W], output shape: [B, embedding_dim].
        """
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            x = checkpoint_sequential(self.features, self.grad_checkpoint, x,
                                      use_reentrant=False)
        else:
            x = self.features(x)
        if self.pool_channels is not None:
            x = self.reduce(x)
        pooled = torch.cat(
            [torch.flatten(adaptive_avg_pool2d_deterministic(x, self.pool_grid), 1),
             torch.flatten(adaptive_max_pool2d_deterministic(x, self.pool_grid), 1)],
            dim=1,
        )
        return self.embedding(pooled)

    def forward(self, x1, x2):
        """Forward pass for a pair of images. Returns embeddings for both branches."""
        return self.encode(x1), self.encode(x2)


class TemporalSiamese(nn.Module):
    """
    Siamese LSTM for online (dynamic) signature verification.

    Processes trajectory sequences containing coordinates and pressure.
    Extracts time dynamics and maps signatures to continuous embeddings.
    """

    def __init__(self, input_features=TEMPORAL_INPUT_FEATURES, hidden_size=TEMPORAL_HIDDEN_SIZE,
                 num_layers=TEMPORAL_NUM_LAYERS, embedding_dim=TEMPORAL_EMBEDDING_DIM,
                 pressure_dropout=PRESSURE_DROPOUT_RATE, derive_deltas=True,
                 bidirectional=True, conv_filters=TEMPORAL_CONV_FILTERS,
                 conv_kernel=TEMPORAL_CONV_KERNEL):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bidirectional = bool(bidirectional)
        self.pressure_dropout = pressure_dropout
        self.derive_deltas = derive_deltas

        # Save delta flag in state_dict buffer to preserve model configuration.
        self.register_buffer("derive_deltas_flag",
                             torch.tensor(float(derive_deltas)), persistent=True)

        # When derive_deltas is True, per-frame velocity deltas double the input features.
        lstm_features = input_features * 2 if derive_deltas else input_features

        # Research arm (off by default): local stroke context. The conv output is joined to
        # each sample, so the LSTM keeps every channel it has today. No normalisation layer:
        # its statistics would differ between a padded batch and a single signature.
        self.conv = None
        if conv_filters:
            if conv_kernel % 2 == 0:
                raise ValueError(f"conv_kernel must be odd to keep the length, got {conv_kernel}")
            self.conv = nn.Conv1d(lstm_features, conv_filters, conv_kernel,
                                  padding=conv_kernel // 2)
            lstm_features += conv_filters

        # Bidirectional LSTM to process trajectory sequence in both time directions.
        self.lstm = nn.LSTM(
            input_size=lstm_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        # Linear projection from final hidden state to embedding space.
        self.embedding = nn.Linear(hidden_size * (2 if self.bidirectional else 1), embedding_dim)

    @classmethod
    def from_state_dict(cls, state_dict, **overrides):
        """
        Reconstruct a TemporalSiamese model using parameters saved in state_dict.
        Recovers input features, hidden size, layers, and delta settings.
        """
        weight = state_dict["lstm.weight_ih_l0"]
        hidden_size = weight.shape[0] // 4
        conv_weight = state_dict.get("conv.weight")
        conv_filters = conv_weight.shape[0] if conv_weight is not None else 0
        width = weight.shape[1] - conv_filters

        derive_deltas = bool(state_dict.get("derive_deltas_flag", torch.tensor(1.0)).item())

        if derive_deltas and width % 2:
            raise ValueError(
                f"LSTM input width {width} is odd, so it cannot be "
                "2 * input_features. This checkpoint predates derived dynamics "
                "and cannot be scored by this class."
            )

        layers = 1 + max(
            (int(k.rsplit("_l", 1)[-1].replace("_reverse", ""))
             for k in state_dict if k.startswith("lstm.weight_ih_l")),
            default=0,
        )
        params = {
            "input_features": width // 2 if derive_deltas else width,
            "hidden_size": hidden_size,
            "num_layers": layers,
            "embedding_dim": state_dict["embedding.weight"].shape[0],
            "derive_deltas": derive_deltas,
            "bidirectional": "lstm.weight_ih_l0_reverse" in state_dict,
            "conv_filters": conv_filters,
            "conv_kernel": conv_weight.shape[2] if conv_weight is not None else TEMPORAL_CONV_KERNEL,
        }
        params.update(overrides)

        model = cls(**params)
        if "derive_deltas_flag" not in state_dict:
            state_dict = {**state_dict,
                          "derive_deltas_flag": torch.tensor(float(derive_deltas))}
        model.load_state_dict(state_dict)
        return model.eval()

    @staticmethod
    def _add_dynamics(x):
        """Calculate per-frame deltas and concatenate them with the input trajectory."""
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return torch.cat([x, delta], dim=-1)

    def _apply_pressure_dropout(self, x):
        """
        Randomly set pressure values to 1.0 during training to improve robustness
        when evaluated on devices without pressure sensors.
        """
        if self.training and self.pressure_dropout > 0:
            batch_size = x.size(0)
            mask = torch.rand(batch_size, device=x.device) < self.pressure_dropout
            if mask.any():
                x = x.clone()
                x[mask, :, 2] = 1.0
        return x

    def encode(self, x, lengths=None):
        """
        Extract the continuous embedding from a pen trajectory sequence.
        Input shape: [B, L, C], output shape: [B, embedding_dim].
        """
        x = self._apply_pressure_dropout(x)

        if self.derive_deltas:
            x = self._add_dynamics(x)

        if self.conv is not None:
            if lengths is not None:
                # Zero everything past each signature's end: the appended deltas are not zero
                # there, and the conv must read what a single signature's own padding gives.
                keep = (torch.arange(x.size(1), device=x.device)[None, :]
                        < lengths.to(x.device)[:, None])
                x = x * keep[:, :, None]
            local = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
            x = torch.cat([x, local], dim=-1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_combined = torch.cat([h_n[-2], h_n[-1]], dim=1)   # last layer, both directions
        else:
            h_combined = h_n[-1]
        return self.embedding(h_combined)

    def forward(self, x1, x2, lengths1=None, lengths2=None):
        """Forward pass for a pair of trajectory sequences."""
        return self.encode(x1, lengths1), self.encode(x2, lengths2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese metric learning.

    Minimizes Euclidean distance for genuine pairs and pushes impostor
    pairs beyond the contrastive margin threshold.
    Supports hard-negative mining to focus on difficult forgeries.
    """

    def __init__(self, margin=1.0, hard_negative_ratio=0.5):
        super().__init__()
        self.margin = margin
        self.hard_negative_ratio = hard_negative_ratio

    def forward(self, embedding1, embedding2, label):
        distance = F.pairwise_distance(embedding1, embedding2, p=2)

        genuine_loss = label * distance.pow(2)
        impostor_loss = (1 - label) * F.relu(self.margin - distance).pow(2)

        if self.hard_negative_ratio >= 1.0:
            return (genuine_loss + impostor_loss).mean()

        is_impostor = label < 0.5
        num_impostor = int(is_impostor.sum())

        kept = [genuine_loss[~is_impostor]]

        if num_impostor > 0:
            num_mined = max(1, math.ceil(self.hard_negative_ratio * num_impostor))
            hardest = torch.topk(impostor_loss[is_impostor], num_mined).values
            kept.append(hardest)

        return torch.cat(kept).mean()


class FusionEngine:
    """
    Combines similarity scores from spatial and temporal branches.

    Formula: S_final = alpha * S_spatial + (1 - alpha) * S_temporal
    Falls back to single-branch verification if one modality is absent.
    """

    def __init__(self, alpha=ALPHA_DEFAULT, threshold=DECISION_THRESHOLD,
                 spatial_norm=None, temporal_norm=None):
        self.alpha = alpha
        self.threshold = threshold
        self.spatial_norm = spatial_norm
        self.temporal_norm = temporal_norm

    def _normalize(self, score, norm_params):
        """Normalize score using mean and standard deviation if available."""
        if norm_params is None:
            return score
        mean, std = norm_params
        return (score - mean) / std if std != 0 else score - mean

    def fuse(self, spatial_score=None, temporal_score=None, pressure_imputed=False):
        """
        Combine branch scores and compare against the decision threshold.

        Returns a dictionary with the verification verdict, confidence percentage,
        routing mode, and degraded status indicator.
        """
        if spatial_score is not None and temporal_score is not None:
            s = self.alpha * self._normalize(spatial_score, self.spatial_norm) + \
                (1 - self.alpha) * self._normalize(temporal_score, self.temporal_norm)
            routing = "multimodal"
        elif spatial_score is not None:
            s = self._normalize(spatial_score, self.spatial_norm)
            routing = "spatial_only"
        elif temporal_score is not None:
            s = self._normalize(temporal_score, self.temporal_norm)
            routing = "temporal_only"
        else:
            raise ValueError("At least one modality score must be provided for fusion.")

        return {
            "decision": "Genuine" if s >= self.threshold else "Forged",
            "confidence": min(max(s * 100, 0.0), 100.0),
            "routing": routing,
            "degraded": bool(routing != "multimodal" or pressure_imputed),
        }
