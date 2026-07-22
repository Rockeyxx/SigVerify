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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSiamese(nn.Module):
    """
    Siamese CNN for offline signature verification.

    Uses GroupNorm instead of BatchNorm for stable pair-sampled
    metric learning. Forward pass accepts two images and returns
    their embeddings. For single-image embedding, use encode().
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            # Block 1: 256×256 → 128×128
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 128×128 → 64×64
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64×64 → 32×32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 32×32 → 16×16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 16×16 → 8×8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.embedding = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
        )

    def encode(self, x):
        """Extract embedding from a single image [B, 1, 256, 256] → [B, 512]."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embedding(x)

    def forward(self, x1, x2):
        """Forward pass for a pair of images. Returns both embeddings."""
        return self.encode(x1), self.encode(x2)


class TemporalSiamese(nn.Module):
    """
    Siamese LSTM for online (dynamic) signature verification.

    Supports pressure-channel dropout during training: for a configurable
    fraction of samples, the real pressure channel is replaced with P=1.0
    so the no-pressure path is in-distribution.
    """

    def __init__(self, input_features=3, hidden_size=256,
                 num_layers=2, embedding_dim=128, pressure_dropout=0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pressure_dropout = pressure_dropout

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
        )

    def _apply_pressure_dropout(self, x):
        """Replace pressure channel with 1.0 for a fraction of batch samples."""
        if self.training and self.pressure_dropout > 0:
            batch_size = x.size(0)
            mask = torch.rand(batch_size, device=x.device) < self.pressure_dropout
            if mask.any():
                x = x.clone()
                x[mask, :, 2] = 1.0
        return x

    def encode(self, x):
        """Extract embedding from a sequence [B, L, 3] → [B, 128]."""
        x = self._apply_pressure_dropout(x)
        output, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        return self.embedding(h_combined)

    def forward(self, x1, x2):
        """Forward pass for a pair of sequences. Returns both embeddings."""
        return self.encode(x1), self.encode(x2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese training.

    L = y · d² + (1 - y) · max(0, m - d)²

    y=1 for genuine pairs, y=0 for impostor/forgery pairs,
    d is L2 distance, m is the margin.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        genuine_loss = label * distance.pow(2)
        impostor_loss = (1 - label) * F.relu(self.margin - distance).pow(2)
        return (genuine_loss + impostor_loss).mean()


class FusionEngine:
    """
    Late-fusion score combination with automatic unimodal fallback.

    S_final = α · S_spatial + (1 − α) · S_temporal

    If a modality is absent, its branch is dropped.
    Decision: S_final ≥ τ → Genuine, else Forged.
    """

    def __init__(self, alpha=0.5, threshold=0.5,
                 spatial_norm=None, temporal_norm=None):
        self.alpha = alpha
        self.threshold = threshold
        self.spatial_norm = spatial_norm
        self.temporal_norm = temporal_norm

    def _normalize(self, score, norm_params):
        if norm_params is None:
            return score
        mean, std = norm_params
        return (score - mean) / std if std != 0 else score - mean

    def fuse(self, spatial_score=None, temporal_score=None):
        """
        Fuse per-branch scores into a final decision.

        Returns:
            (decision: str, confidence: float, routing: str)
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
            raise ValueError("At least one modality score must be provided.")

        decision = "Genuine" if s >= self.threshold else "Forged"
        confidence = min(max(s * 100, 0.0), 100.0)
        return decision, confidence, routing
