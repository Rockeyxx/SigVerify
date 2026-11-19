"""Sig2Vec model and average-precision ranking loss for synthetic trajectories.

Learns representations from genuine signature templates and their synthetic
distortions (G1 and G2) without requiring skilled forgery samples during training.
"""

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
    """
    The paper's three input channels: `vx`, `vy`, `|v|` — equations 20-22.

    Note what is absent: **pressure**. SynSig2Vec keys entirely on how the pen
    moved, not how hard it pressed, so this is a strictly narrower input than the
    project's `(X, Y, P)` contract and needs no capture change at all.

    Central differences, then per-channel standardisation to zero mean and unit
    variance, as the paper specifies before the CNN.
    """
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
    # A stationary pen has zero variance on every channel; divide by one rather
    # than clean up the NaN afterwards.
    std = out.std(axis=0)
    out = (out - out.mean(axis=0)) / np.where(std < _EPS, 1.0, std)
    return out.astype(np.float32)


def time_functions(trajectory):
    """
    The 12 time functions this lab's later work uses — DsDTW (Jiang et al.).

    The AAAI SynSig2Vec paper feeds the CNN three channels (`velocity_features`
    above, equations 20-22). Its own authors moved to a richer 12-function set
    in the follow-up work, and this project's temporal branch is already
    diagnosed as channel-starved, so the wider set is offered as an alternative
    input rather than a replacement — both are measurable.

    In order: `vx, vy, v, theta, cos(theta), sin(theta), z, v_dot, theta_dot,
    rho, c, a`, where `rho = log(v / |theta_dot|)` is the log curvature radius,
    `c = v * theta_dot` the centripetal acceleration and `a` the total.

    **Pressure is `z`.** SigmaLambda reconstructs `x, y` only, so a synthetic
    trajectory arrives here two columns wide and its pressure channel is zero
    unless the caller attaches the template's own profile first. That is a real
    deviation from the synthesis model, which does not describe pressure at all.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return np.zeros((n, 12), dtype=np.float32)

    def _d(series):
        """Central differences, endpoints copied — same convention as above."""
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
    """
    Six-layer 1D CNN over velocity sequences, Table 2 of the paper.

    SELU rather than ReLU, and **no batch normalisation**: a training batch here
    is one signature plus its own synthetics, which are highly correlated, so
    normalising across the batch would mix a sample with near-copies of itself.
    SELU self-normalises instead.

    Masked average pooling collapses variable length to a fixed 512-d vector.
    The mask is the load-bearing part — without it a padded batch would let a
    sample's embedding depend on the longest sequence beside it.
    """

    def __init__(self, in_channels=3, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Held as a ModuleList rather than a Sequential so the padded region can
        # be re-zeroed between layers — see `forward`.
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
        """
        Args:
            x:       `[B, 3, L]` velocity features.
            lengths: optional `[B]` true lengths, before pooling. Required for a
                     padded batch, or padded frames contaminate the average.

        Returns: `[B, embedding_dim]`.
        """
        h = x
        if lengths is None:
            for block, pool in zip(self.blocks, self.pools):
                h = block(h)
                if pool:
                    h = self.pool(h)
            return self.embedding(h.mean(dim=2))

        # Re-zero the padded tail after **every** layer, not just before pooling.
        # Conv1d carries a bias, so conv(0) = bias != 0: the padded region stops
        # being zero after the first layer and then bleeds one kernel-radius
        # further inward per layer. Masking only at the pooling stage (as the
        # paper describes) leaves a sample's embedding dependent on the longest
        # sequence batched beside it — measured, and the reason
        # `test_padded_batch_matches_a_single_sample` failed before this.
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
    """
    `1 - AP` for ranking the low-distortion pool above the median-distortion one.

    `g1_similarity` and `g2_similarity` are each sample's similarity to the
    template. G1 samples are the positives; ranking every one of them above
    every G2 sample gives AP = 1 and a loss of 0.

    AP counts, for each positive, how many items outrank it — a step function of
    score differences, and therefore not differentiable. The step is replaced by
    a sigmoid of width `temperature`, so the count becomes a soft one and
    gradients flow. Smaller temperature is closer to true AP and harder to
    optimise; 0.01 keeps the relaxation tight on cosine-scale inputs.
    """
    if g1_similarity.numel() == 0 or g2_similarity.numel() == 0:
        raise ValueError("average precision is undefined for an empty pool")

    positives = g1_similarity.reshape(-1)
    negatives = g2_similarity.reshape(-1)

    # Soft count of items ranked at or above each positive.
    pos_vs_pos = torch.sigmoid((positives[None, :] - positives[:, None]) / temperature)
    pos_vs_neg = torch.sigmoid((negatives[None, :] - positives[:, None]) / temperature)

    # Positives at or above i — the numerator of Prec@i.
    positives_above = 1.0 + (pos_vs_pos.sum(dim=1) - torch.diagonal(pos_vs_pos))
    # Rank of positive i: itself, plus everything scoring above it.
    rank = positives_above + pos_vs_neg.sum(dim=1)

    precision_at = positives_above / rank.clamp(min=_EPS)
    return (1.0 - precision_at.mean()).clamp(0.0, 1.0)


def template_similarity(template_embedding, sample_embeddings):
    """
    Cosine similarity of each sample to the template — equation 16.

    Cosine rather than Euclidean because the verifier normalises embeddings
    before comparing them (equation 23), so the ranking the loss optimises and
    the distance the verifier thresholds measure the same geometry.
    """
    t = torch.nn.functional.normalize(template_embedding.reshape(1, -1), dim=1)
    s = torch.nn.functional.normalize(sample_embeddings, dim=1)
    return (s @ t.T).reshape(-1)
