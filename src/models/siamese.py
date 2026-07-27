"""
Dual-Siamese Signature Verification — Model Architecture

Consolidated module containing:
  - SpatialSiamese:   CNN backbone for offline (image) signature verification
  - TemporalSiamese:  LSTM backbone for online (dynamic) signature verification
  - ContrastiveLoss:  Contrastive loss with hard-negative mining
  - FusionEngine:     Late-fusion score combination with unimodal fallback

Architecture follows the approved Term-1 baseline (Alternative 3 — Late-Fusion
Dual Siamese Networks). See AGENTS.md for hard constraints.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


# Convolutional trunks the spatial branch can be built with. Each stage is
# (out_channels, kernel_size, pool_after, dropout_after) and expands to
# Conv2d → GroupNorm → ReLU → [MaxPool2d] → [Dropout2d].
#
#   compact — the project's own trunk. Five narrow stages, every one pooled, so
#             a 155x220 input reaches 256x4x6 before the pooling grid.
#   wide    — the Dey et al. Siamese reference (SigNet): a 96-filter 11x11 stem,
#             then 256x5x5, 384x3x3 and 256x3x3, with dropout after the second
#             and fourth stages. Four stages instead of five and only three
#             pools, so the map stays larger and far wider through the trunk.
#
# Adding a trunk here is enough to train with it; `from_state_dict` recovers
# which one a checkpoint was trained with from the convolution shapes alone.
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
    GroupNorm sized to keep four channels per group, capped at 32 groups.

    Reproduces the hand-written groups the compact trunk shipped with (8 for 32
    channels, 16 for 64, 32 for 128 and 256) and extends the same rule to the
    wider trunks, where 96 -> 24 groups and 384 -> 32.
    """
    groups = min(32, channels // 4)
    return nn.GroupNorm(groups, channels)


class SpatialSiamese(nn.Module):
    """
    Siamese CNN for offline signature verification.

    Uses GroupNorm instead of BatchNorm to ensure stable normalization
    when processing small, pair-sampled batches during metric learning.

    Forward pass accepts two grayscale images [B, 1, H, W] and extracts
    their 512-dimensional continuous embeddings for distance comparison.

    `trunk` selects a convolutional stack from SPATIAL_TRUNKS; `conv_dropout`
    and `head_dropout` are training-time regularizers that leave the parameter
    shapes untouched, and `fc_hidden` inserts the reference's FC-1024 layer
    ahead of the embedding projection.
    """

    def __init__(self, embedding_dim=512, head_dropout=0.0,
                 trunk="compact", conv_dropout=0.0, fc_hidden=0,
                 grad_checkpoint=0):
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
                # In-place is a free memory saving normally, but under gradient
                # checkpointing a ReLU that lands on a segment boundary has its
                # output overwritten before the recompute reads it, which autograd
                # reports as "a variable needed for gradient computation has been
                # modified". Whether that happens depends on the module count
                # lining up with the segment count, so do not rely on it.
                nn.ReLU(inplace=not grad_checkpoint),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            if drop:
                # Dropout2d, not Dropout: zeroing whole feature maps is what
                # regularizes a convnet, since neighbouring pixels in one map are
                # correlated enough that dropping them individually barely
                # removes information. The module is built whatever the rate, so
                # the layer indices — and therefore the state_dict keys — do not
                # depend on whether dropout happens to be switched on.
                layers.append(nn.Dropout2d(conv_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # A 4x4 grid keeps 16 regions per channel, which preserves the coarse
        # stroke geometry that identifies a signer. Pooling every position into
        # one channel mean would describe ink density instead of shape.
        self.pool_grid = 4
        self.pool_avg = nn.AdaptiveAvgPool2d((self.pool_grid, self.pool_grid))
        self.pool_max = nn.AdaptiveMaxPool2d((self.pool_grid, self.pool_grid))

        # avg gives regional ink density, max the strongest filter response there.
        # Ink covers roughly 1% of the canvas, so averaging on its own dilutes the
        # stroke evidence against a background level that is nearly identical for
        # every signer. The max branch ignores that floor.
        pooled_features = 2 * in_channels * self.pool_grid * self.pool_grid

        # The projection ends on a plain Linear, with or without the hidden
        # layer. A ReLU on the *output* would confine every embedding to a single
        # convex cone, capping angular spread and pushing discrimination into
        # vector magnitude instead of direction. A ReLU on a hidden layer before
        # it is fine — the final linear map can still reach the whole space.
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
        Rebuild the architecture a checkpoint was trained with, then load it.

        Trunk width, embedding width and the presence of the FC hidden layer are
        all readable off the saved parameter shapes, so a checkpoint carries its
        own architecture and no caller has to remember which config produced it.
        Without this, benchmarking an archived run means editing `config/` first,
        and scoring a checkpoint against the wrong architecture is a silent
        wrong number rather than an error.

        `overrides` pass through to __init__ for the settings that leave no trace
        in the parameters — the dropout rates, which are inert at eval time.
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

        # Head shapes: an FC-1024 variant carries a second Linear at index 4.
        if "embedding.4.weight" in state_dict:
            fc_hidden, embedding_dim = state_dict["embedding.4.weight"].shape[::-1]
        elif "embedding.1.weight" in state_dict:
            fc_hidden, embedding_dim = 0, state_dict["embedding.1.weight"].shape[0]
        else:
            # Pre-dating the Dropout-wrapped head, so the projection was a bare
            # `embedding.weight`. Those checkpoints also pre-date the 4x4 pooling
            # grid and the inverted preprocessing, so there is no version of this
            # code they are valid under — say so instead of raising a KeyError.
            raise ValueError(
                "Checkpoint has no recognizable embedding head. It was saved before "
                "the head became a Sequential, which also means before the 4x4 pooling "
                "grid and the inverted preprocessing — it cannot be scored against "
                "the current pipeline at all."
            )

        model = cls(embedding_dim=int(embedding_dim), trunk=trunk,
                    fc_hidden=int(fc_hidden), **overrides)
        model.load_state_dict(state_dict)
        return model

    def encode(self, x):
        """
        Extract the raw (un-normalized) embedding from a single image.
        Tensor flow (compact trunk at 155x220): [B, 1, 155, 220] → [B, 256, 4, 6]
                     → 2x [B, 256, 4, 4] → [B, 8192] → [B, 512].

        Embeddings are NOT L2-normalized. Projecting onto the unit sphere makes
        CONTRASTIVE_MARGIN scale-free, which is tidier, but it measured slightly
        worse on held-out EER, so the raw projection stands. Do not assume ‖e‖ = 1.
        """
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            # Trade compute for memory: store only the segment boundaries and
            # recompute each segment during the backward pass. The arithmetic is
            # unchanged — checkpointing restores the RNG state before recomputing,
            # so dropout draws the same masks — which is what lets the wide trunk
            # be compared against the compact one at the same batch size on a
            # 4 GB card instead of at a batch size only it could afford.
            x = checkpoint_sequential(self.features, self.grad_checkpoint, x,
                                      use_reentrant=False)
        else:
            x = self.features(x)
        pooled = torch.cat(
            [torch.flatten(self.pool_avg(x), 1), torch.flatten(self.pool_max(x), 1)],
            dim=1,
        )
        return self.embedding(pooled)

    def forward(self, x1, x2):
        """Forward pass for a pair of images. Returns embeddings for both input branches."""
        return self.encode(x1), self.encode(x2)


class TemporalSiamese(nn.Module):
    """
    Siamese LSTM for online (dynamic) signature verification.

    Processes trajectory sequences containing (X, Y, Pressure) coordinates.
    Supports pressure-channel dropout during training: a percentage of samples
    have their real pressure replaced with P=1.0 so degraded mode (no pressure)
    remains inside the model's learned feature distribution.
    """

    def __init__(self, input_features=3, hidden_size=256,
                 num_layers=2, embedding_dim=128, pressure_dropout=0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pressure_dropout = pressure_dropout

        # The capture device reports position and pressure, but what separates a
        # signer from a forger is mostly how the pen moves: a forger can copy the
        # shape while tracing it far more slowly and evenly. Velocity is the most
        # discriminative feature family in the online-verification literature, so
        # it is derived here from consecutive frames rather than being demanded
        # from the tablet. The capture contract stays (X, Y, P); the LSTM sees
        # (X, Y, P, dX, dY, dP).
        lstm_features = input_features * 2

        # Bidirectional LSTM to capture stroke dynamics forward and backward in time
        self.lstm = nn.LSTM(
            input_size=lstm_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Concatenated final hidden states (2 * hidden_size) projected to 128-d
        # embedding space by a single linear layer.
        #
        # No ReLU and no dropout on this path. Dropout inside a Siamese head is
        # actively harmful: the two branches draw independent masks, so the same
        # signature encoded twice lands at two different points. That puts a
        # random floor under the genuine-pair distance the loss is trying to
        # drive to zero. A ReLU would also trap every embedding in one convex
        # cone, pushing discrimination into magnitude rather than direction.
        self.embedding = nn.Linear(hidden_size * 2, embedding_dim)

    @staticmethod
    def _add_dynamics(x):
        """
        Append per-frame deltas, turning [B, L, 3] into [B, L, 6].

        The first frame has no predecessor, so its delta is zero. Deltas at
        indices past a sequence's true length read into padding, but packing in
        `encode` drops those positions before the LSTM sees them.
        """
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return torch.cat([x, delta], dim=-1)

    def _apply_pressure_dropout(self, x):
        """
        Randomly substitute the pressure coordinate (index 2) with 1.0 for training samples.
        This forces the network to learn geometric trajectory features without over-relying on pressure.
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
        Extract the raw (un-normalized) embedding from a trajectory sequence.
        Input shape: [B, L, 3] where L is sequence length, 3 features are (X, Y, P).
        Output shape: [B, 128].

        `lengths` (optional [B] int tensor of true sequence lengths) makes the
        LSTM skip padding via pack_padded_sequence, so a padded batch produces
        the same embedding as the sample encoded alone.
        """
        x = self._apply_pressure_dropout(x)

        # Deltas are taken after pressure dropout so a sample whose pressure was
        # flattened to 1.0 also reports a flat dP, which is what a real capture
        # without pressure looks like.
        x = self._add_dynamics(x)

        if lengths is not None:
            # Packing hides the padded tail from both LSTM directions. Without it
            # the backward pass would start on pad rows and contaminate h_n.
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # Extract last layer forward state (h_n[-2]) and backward state (h_n[-1])
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        return self.embedding(h_combined)

    def forward(self, x1, x2, lengths1=None, lengths2=None):
        """Forward pass for a pair of trajectory sequences. Returns both embeddings."""
        return self.encode(x1, lengths1), self.encode(x2, lengths2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese pair metric learning.

    Formula:
        L = y · d² + (1 - y) · max(0, m - d)²

    where:
        - y = 1 for genuine pairs (encourages distance d → 0)
        - y = 0 for forgery/impostor pairs (enforces separation distance d ≥ m)
        - d is the pairwise L2 Euclidean distance
        - m is the contrastive margin threshold

    Hard-negative mining (AGENTS.md requirement): most impostor pairs sit beyond
    the margin and contribute exactly zero loss, so averaging over the whole
    batch dilutes the gradient signal toward zero as training progresses. Each
    batch therefore keeps only the `hard_negative_ratio` fraction of impostors
    with the largest loss — equivalently the smallest distances, the ones the
    model currently finds most confusable. Set the ratio to 1.0 to disable
    mining and recover the plain contrastive mean.
    """

    def __init__(self, margin=1.0, hard_negative_ratio=0.5):
        super().__init__()
        self.margin = margin
        self.hard_negative_ratio = hard_negative_ratio

    def forward(self, embedding1, embedding2, label):
        # Calculate pairwise L2 Euclidean distance between embedding vectors
        distance = F.pairwise_distance(embedding1, embedding2, p=2)

        # Loss component for genuine pairs: minimize square of distance
        genuine_loss = label * distance.pow(2)

        # Loss component for forgery pairs: push distance above margin m
        impostor_loss = (1 - label) * F.relu(self.margin - distance).pow(2)

        if self.hard_negative_ratio >= 1.0:
            return (genuine_loss + impostor_loss).mean()

        is_impostor = label < 0.5
        num_impostor = int(is_impostor.sum())

        # Genuine terms are always kept in full — mining only prunes impostors
        kept = [genuine_loss[~is_impostor]]

        if num_impostor > 0:
            num_mined = max(1, math.ceil(self.hard_negative_ratio * num_impostor))
            hardest = torch.topk(impostor_loss[is_impostor], num_mined).values
            kept.append(hardest)

        return torch.cat(kept).mean()


class FusionEngine:
    """
    Late-fusion score combination with unimodal fallback.

    Combines calibrated spatial and temporal similarity scores using a weighted sum:
        S_final = α · S_spatial + (1 − α) · S_temporal

    If either modality is missing, the engine automatically routes inference
    to the available branch (setting α = 1.0 or α = 0.0).
    """

    def __init__(self, alpha=0.5, threshold=0.5,
                 spatial_norm=None, temporal_norm=None):
        self.alpha = alpha
        self.threshold = threshold
        self.spatial_norm = spatial_norm
        self.temporal_norm = temporal_norm

    def _normalize(self, score, norm_params):
        """Standardize raw score using validation mean and standard deviation."""
        if norm_params is None:
            return score
        mean, std = norm_params
        return (score - mean) / std if std != 0 else score - mean

    def fuse(self, spatial_score=None, temporal_score=None, pressure_imputed=False):
        """
        Fuse available branch scores and compare against decision threshold τ.

        Args:
            spatial_score:    calibrated spatial similarity, or None if unavailable.
            temporal_score:   calibrated temporal similarity, or None if unavailable.
            pressure_imputed: True when the temporal sample had no pressure channel
                              and P = 1.0 was substituted (AGENTS.md degraded mode).

        Returns a dict — this is the contract the GUI renders against:
            {
              "decision":   "Genuine" if S_final >= τ else "Forged",
              "confidence": score scaled to [0.0, 100.0] percentage,
              "routing":    "multimodal" | "spatial_only" | "temporal_only",
              "degraded":   True when the verification ran on reduced evidence —
                            either a branch was missing or pressure was imputed.
                            The GUI must surface this; a degraded "Genuine" is
                            not the same claim as a multimodal one.
            }
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

