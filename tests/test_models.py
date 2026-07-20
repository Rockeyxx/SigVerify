"""Tests for the Siamese model architectures, loss function, and fusion engine."""

# TODO: Test SpatialSiamese forward pass shape [B, 1, 256, 256] → [B, 512]
# TODO: Test TemporalSiamese forward pass shape [B, L, 3] → [B, 128]
# TODO: Test ContrastiveLoss gradient flow
# TODO: Test FusionEngine unimodal fallback routing
# TODO: Test pressure-channel dropout behavior in training vs eval mode
