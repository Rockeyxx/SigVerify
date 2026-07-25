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


class SpatialSiamese(nn.Module):
    """
    Siamese CNN for offline signature verification.

    Uses GroupNorm instead of BatchNorm to ensure stable normalization
    when processing small, pair-sampled batches during metric learning.
    
    Forward pass accepts two grayscale images [B, 1, 256, 256] and extracts
    their 512-dimensional continuous embeddings for distance comparison.
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 5-stage feature extractor: downsamples input from [B, 1, 256, 256] down to [B, 256, 8, 8]
        self.features = nn.Sequential(
            # Block 1: [B, 1, 256, 256] → [B, 32, 128, 128]
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: [B, 32, 128, 128] → [B, 64, 64, 64]
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: [B, 64, 64, 64] → [B, 128, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: [B, 128, 32, 32] → [B, 256, 16, 16]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: [B, 256, 16, 16] → [B, 256, 8, 8]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling reduces spatial dims from [B, 256, H', W'] to [B, 256, 1, 1]
        # This reduces parameter count in the FC projection layer and provides spatial invariance.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected projection block mapping pooled features to a 512-d embedding space
        self.embedding = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
        )

    def encode(self, x):
        """
        Extract the raw (un-normalized) embedding from a single image.
        Tensor flow: [B, 1, 256, 256] → [B, 256, 8, 8] → [B, 256, 1, 1] → [B, 512].

        Embeddings are NOT L2-normalized — magnitudes are free to vary, which
        is what the contrastive margin operates on. Do not assume ‖e‖ = 1.
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten feature map to vector [B, 256]
        return self.embedding(x)

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

        # Bidirectional LSTM to capture stroke dynamics forward and backward in time
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Concatenated final hidden states (2 * hidden_size) projected to 128-d embedding space
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
        )

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

