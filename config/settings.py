"""
Project Configuration — Signature Verification System (EE 499)

Central configuration for hyperparameters, thresholds, paths, and
system-wide constants. All tunable values live here.

Note on Model Weights Distribution:
  Compiled `.pth` model weights files are excluded from Git version control
  via `.gitignore` and hosted on GitHub Releases for downloading.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset roots. `data/` is gitignored — datasets are fetched locally, never committed.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CEDAR_DIR = os.path.join(DATA_DIR, "cedar")          # full_org/ + full_forg/
BHSIG260_DIR = os.path.join(DATA_DIR, "bhsig260")    # Bengali/ + Hindi/
UTSIG_DIR = os.path.join(DATA_DIR, "utsig")          # Genuine/ + Forgery/

# SVC2004 ships two tasks. Both are kept so the temporal branch can be trained
# twice and the contribution of the pressure channel measured directly.
SVC2004_TASK1_DIR = os.path.join(DATA_DIR, "svc2004_task1")   # 4-col, no pressure
SVC2004_TASK2_DIR = os.path.join(DATA_DIR, "svc2004_task2")   # 7-col, real pressure
SVC2004_DIR = SVC2004_TASK2_DIR                               # default: pressure-aware

# ─── DataLoader ───────────────────────────────────────────────────────
NUM_WORKERS = 4                         # 0 on Windows/notebook if workers hang
PIN_MEMORY = False                      # CPU-only training — pinning buys nothing

# ─── Spatial Branch (Offline — Siamese CNN) ───────────────────────────
SPATIAL_INPUT_SIZE = (256, 256)          # H × W → [1, 1, 256, 256] tensor
SPATIAL_EMBEDDING_DIM = 512
SPATIAL_CHANNELS = 1                    # Grayscale

# Quality gate thresholds (effective-resolution proxies)
MIN_BBOX_PIXELS = 64                    # Minimum signature bounding-box dimension
MIN_LAPLACIAN_VARIANCE = 50.0           # Sharpness threshold

# ─── Temporal Branch (Online — Siamese LSTM) ──────────────────────────
TEMPORAL_INPUT_FEATURES = 3             # (X, Y, Pressure)
TEMPORAL_EMBEDDING_DIM = 128
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_NUM_LAYERS = 2
PRESSURE_DROPOUT_RATE = 0.4             # Replace P with 1.0 for 40% of training samples
PRESSURE_IMPUTATION_VALUE = 1.0

# ─── Pair Generation ─────────────────────────────────────────────────
# Per-writer pair budgets for a real training run (the 5/5 demo defaults are
# far too small: 55 CEDAR writers × 5 would yield only ~550 pairs total).
PAIRS_GENUINE_PER_WRITER = 40
PAIRS_IMPOSTOR_PER_WRITER = 40

# ─── Loss & Training ─────────────────────────────────────────────────
CONTRASTIVE_MARGIN = 1.0
HARD_NEGATIVE_RATIO = 0.5               # Fraction of impostor terms kept by mining
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4

# Early Stopping hyperparameters (monitors validation loss to halt training)
EARLY_STOPPING_PATIENCE = 10            # Stop after 10 epochs without validation improvement
EARLY_STOPPING_MIN_DELTA = 1e-4         # Minimum loss decrease required to trigger improvement

# ─── Fusion & Decision ───────────────────────────────────────────────
ALPHA_DEFAULT = 0.5                     # Tuned on validation via grid search
ALPHA_GRID_STEP = 0.05
DECISION_THRESHOLD = 0.5               # τ: S_final ≥ τ → Genuine

# ─── Evaluation Targets ──────────────────────────────────────────────
TARGET_FAR = 0.05
TARGET_FRR = 0.05
TARGET_ACCURACY = 0.95

