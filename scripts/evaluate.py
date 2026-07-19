"""
Evaluation Script — Signature Verification System

Standalone evaluation and calibration script. Imports from src.models
and runs analysis independently of the training pipeline.

Implements:
  - EER, FAR, FRR computation under skilled-forgery protocol
  - ROC/DET curve generation
  - Confusion matrix reporting
  - α grid search over [0, 1] in 0.05 steps (validation set)
  - τ (threshold) calibration on validation, frozen for test
  - Tri-point reporting: spatial-only EER, temporal-only EER, fused EER
"""

# TODO: Implement EER computation from score distributions
# TODO: Implement ROC and DET curve generation (matplotlib)
# TODO: Implement confusion matrix reporting
# TODO: Implement α grid search with α-vs-EER sweep plot
# TODO: Implement τ threshold calibration
# TODO: Implement tri-point reporting (spatial, temporal, fused)
# TODO: Implement asymmetric FAR/FRR analysis
