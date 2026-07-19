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

# TODO: Implement SpatialSiamese (nn.Module)
#   - CNN backbone with GroupNorm (not BatchNorm)
#   - Input: [B, 1, 256, 256], Output: [B, 512]

# TODO: Implement TemporalSiamese (nn.Module)
#   - LSTM backbone with pressure-channel dropout
#   - Input: [B, L, 3], Output: [B, 128]

# TODO: Implement ContrastiveLoss (nn.Module)
#   - L = y·d² + (1−y)·max(0, m−d)²
#   - Support dynamic hard-negative mining

# TODO: Implement FusionEngine
#   - S_final = α · S_spatial + (1 − α) · S_temporal
#   - Unimodal fallback when a modality is absent
