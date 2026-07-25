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

# MOBISIG: 83 users, finger-drawn on a capacitive touchscreen rather than a pen
# tablet, so its "pressure" comes from contact area. Used to pre-train the
# temporal branch, not to train the shipped model, since the deployment device
# is a pen tablet. https://www.ms.sapientia.ro/~manyi/mobisig.html
MOBISIG_DIR = os.path.join(DATA_DIR, "MOBISIG")

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
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Dropout on the 8192-d pooled vector feeding the spatial embedding projection.
# Off by default. At 0.4 it costs about a point of held-out EER and barely
# narrows the train/validation gap, so head capacity is not what limits
# generalization here.
SPATIAL_HEAD_DROPOUT = 0.0

# Writer-independent split ratios (train, val, test).
# The image corpora have 430 writers between them, so 70/15/15 still leaves 65
# test writers. SVC2004 has only 40, where the same ratio yields 6 test writers
# and roughly 480 pairs, far too few to rank configurations on. Halving the
# training share buys 10 validation and 10 test writers instead.
SPATIAL_SPLIT_RATIOS = (0.7, 0.15, 0.15)
TEMPORAL_SPLIT_RATIOS = (0.5, 0.25, 0.25)

# Pairs drawn per writer for the temporal branch. Twenty training writers give
# only 1600 pairs at the shared default of 40+40, and the LSTM overfits that
# within about fifteen epochs. Each SVC2004 writer supplies 20 genuine and 20
# forged samples, which allow 190 genuine and 400 impostor combinations, so
# sampling more of them costs nothing but time.
TEMPORAL_PAIRS_GENUINE_PER_WRITER = 180
TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER = 180

# Early Stopping hyperparameters (monitors validation loss to halt training)
EARLY_STOPPING_PATIENCE = 5            # Stop after 10 epochs without validation improvement
EARLY_STOPPING_MIN_DELTA = 1e-4         # Minimum loss decrease required to trigger improvement

# ─── Fusion & Decision ───────────────────────────────────────────────
ALPHA_DEFAULT = 0.5                     # Tuned on validation via grid search
ALPHA_GRID_STEP = 0.05
# τ: S_final ≥ τ → Genuine. Tuned on the validation split, never on test, and
# re-derive it whenever the spatial weights change. Similarity is S = 1/(1+d),
# so a threshold sits directly on a distance: the old default of 0.5 meant
# d = 1.0, which is exactly CONTRASTIVE_MARGIN. That put the decision boundary
# on the loss's own target with no slack, and every impostor the model had not
# pushed fully past the margin landed on the accept side.
DECISION_THRESHOLD = 0.7143

# ─── Evaluation Targets ──────────────────────────────────────────────
TARGET_FAR = 0.05
TARGET_FRR = 0.05
TARGET_ACCURACY = 0.95

