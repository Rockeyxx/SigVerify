"""Sig2Vec model and average-precision ranking loss for synthetic trajectories."""

import numpy as np
import torch
import torch.nn as nn

_EPS = 1e-8

# Table 2 of the paper: (out_channels, kernel, pool_after).
_TRUNK = (
    (64, 7, True),
    (64, 3, False),
    (128, 3, True),
    (128, 3, False),
    (256, 3, True),
    (256, 3, False),
)

EMBEDDING_DIM = 512


def velocity_features(trajectory):
    """Return standardized `(vx, vy, |v|)` channels (equations 20 to 22) via central differences."""
    xy = np.asarray(trajectory, dtype=np.float64)[:, :2]
    n = len(xy)
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)

    vx = np.zeros(n)
    vy = np.zeros(n)
    vx[1:-1] = (xy[2:, 0] - xy[:-2, 0]) / 2.0
    vy[1:-1] = (xy[2:, 1] - xy[:-2, 1]) / 2.0
    vx[0], vy[0] = vx[1], vy[1]
    vx[-1], vy[-1] = vx[-2], vy[-2]
    speed = np.sqrt(vx ** 2 + vy ** 2)

    out = np.stack([vx, vy, speed], axis=1)
    # Guard stationary pens where channel std is zero.
    std = out.std(axis=0)
    out = (out - out.mean(axis=0)) / np.where(std < _EPS, 1.0, std)
    return out.astype(np.float32)


def time_functions(trajectory):
    """Return 12 standardized time functions `(vx, vy, v, theta, cos, sin, z, v_dot, theta_dot, rho, c, a)`."""
    arr = np.asarray(trajectory, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return np.zeros((n, 12), dtype=np.float32)

    def _d(series):
        """Central differences with endpoints copied."""
        out = np.zeros(n)
        out[1:-1] = (series[2:] - series[:-2]) / 2.0
        out[0], out[-1] = out[1], out[-2]
        return out

    x, y = arr[:, 0], arr[:, 1]
    z = arr[:, 2] if arr.shape[1] > 2 else np.zeros(n)

    vx, vy = _d(x), _d(y)
    v = np.sqrt(vx ** 2 + vy ** 2)
    # Wrapped angle is the channel; the derivative is taken on the unwrapped
    # signal, or every crossing of +/-pi would register as a huge false rotation.
    theta = np.arctan2(vy, vx)
    theta_dot = _d(np.unwrap(theta))
    v_dot = _d(v)
    rho = np.log((v + _EPS) / (np.abs(theta_dot) + _EPS))
    c = v * theta_dot
    a = np.sqrt(v_dot ** 2 + c ** 2)

    out = np.stack([vx, vy, v, theta, np.cos(theta), np.sin(theta), z,
                    v_dot, theta_dot, rho, c, a], axis=1)
    std = out.std(axis=0)
    out = (out - out.mean(axis=0)) / np.where(std < _EPS, 1.0, std)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


FEATURES = {"velocity": (velocity_features, 3), "full": (time_functions, 12)}


class Sig2Vec(nn.Module):
    """Six-layer 1D CNN over velocity sequences (Table 2) with SELU and masked average pooling."""

    def __init__(self, in_channels=3, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Use ModuleList so padded frames can be re-zeroed between layers in forward().
        self.blocks = nn.ModuleList()
        self.pools = []
        channels = in_channels
        self.pool_factor = 1
        for out_channels, kernel, pool in _TRUNK:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels, out_channels, kernel_size=kernel,
                          stride=1, padding=kernel // 2),
                nn.SELU(inplace=True)))
            self.pools.append(bool(pool))
            if pool:
                self.pool_factor *= 2
            channels = out_channels
        self.pool = nn.MaxPool1d(2, 2)
        self.embedding = nn.Linear(channels, embedding_dim)

    @staticmethod
    def _mask(lengths, width, device, dtype):
        index = torch.arange(width, device=device)[None, :]
        return (index < lengths[:, None].to(device)).to(dtype)[:, None, :]

    def forward(self, x, lengths=None):
        """Encode `[B, C, L]` velocity features into `[B, embedding_dim]`, masking padded tails."""
        h = x
        if lengths is None:
            for block, pool in zip(self.blocks, self.pools):
                h = block(h)
                if pool:
                    h = self.pool(h)
            return self.embedding(h.mean(dim=2))

        # Re-zero the padded tail after each block so Conv1d bias never leaks across batch padding.
        current = lengths.to(h.device)
        for block, pool in zip(self.blocks, self.pools):
            h = block(h) * self._mask(current, h.shape[2], h.device, h.dtype)
            if pool:
                h = self.pool(h)
                current = torch.clamp(torch.div(current, 2, rounding_mode="floor"),
                                      min=1, max=h.shape[2])
                h = h * self._mask(current, h.shape[2], h.device, h.dtype)

        mask = self._mask(current, h.shape[2], h.device, h.dtype)
        pooled = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        return self.embedding(pooled)


def average_precision_loss(g1_similarity, g2_similarity, temperature=0.01):
    """Differentiable `1 - AP` loss ranking `g1_similarity` above `g2_similarity` via sigmoid temperature."""
    if g1_similarity.numel() == 0 or g2_similarity.numel() == 0:
        raise ValueError("average precision is undefined for an empty pool")

    positives = g1_similarity.reshape(-1)
    negatives = g2_similarity.reshape(-1)

    # Soft count of items ranked at or above each positive.
    pos_vs_pos = torch.sigmoid((positives[None, :] - positives[:, None]) / temperature)
    pos_vs_neg = torch.sigmoid((negatives[None, :] - positives[:, None]) / temperature)

    # Positives ranked at or above index i (Prec@i numerator).
    positives_above = 1.0 + (pos_vs_pos.sum(dim=1) - torch.diagonal(pos_vs_pos))
    # Rank of positive i: itself, plus everything scoring above it.
    rank = positives_above + pos_vs_neg.sum(dim=1)

    precision_at = positives_above / rank.clamp(min=_EPS)
    return (1.0 - precision_at.mean()).clamp(0.0, 1.0)


def template_similarity(template_embedding, sample_embeddings):
    """Cosine similarity of each sample embedding to `template_embedding` (equation 16)."""
    t = torch.nn.functional.normalize(template_embedding.reshape(1, -1), dim=1)
    s = torch.nn.functional.normalize(sample_embeddings, dim=1)
    return (s @ t.T).reshape(-1)
