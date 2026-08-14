"""
Configuration Core — Signature Verification System (EE 499)

Consolidated hyperparameter definitions, system boundaries, filesystem targets,
and global operational constants. Tunable parameters are defined here.

Model Weights Storage Note:
  Binary `.pth` parameter files are excluded from git tracking through `.gitignore`,
  and are distributed via GitHub Releases for manual retrieval.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Root data directories. `data/` is gitignored — local retrieval is required; no datasets are checked into git.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CEDAR_DIR = os.path.join(DATA_DIR, "cedar")          # full_org/ + full_forg/
BHSIG260_DIR = os.path.join(DATA_DIR, "bhsig260")    # Bengali/ + Hindi/
UTSIG_DIR = os.path.join(DATA_DIR, "utsig")          # Genuine/ + Forgery/

# SVC2004 provides two distinct task variants. Retaining both permits training
# duplicate temporal models to evaluate the specific isolation impact of pen pressure.
SVC2004_TASK1_DIR = os.path.join(DATA_DIR, "svc2004_task1")   # 4-col, no pressure
SVC2004_TASK2_DIR = os.path.join(DATA_DIR, "svc2004_task2")   # 7-col, real pressure
SVC2004_DIR = SVC2004_TASK2_DIR                               # default: pressure-aware

# MOBISIG: 83 subjects utilizing touch interface interaction (finger strokes) rather
# than stylus input, deriving "pressure" from touch contact bounding region. Used for temporal
# initialization pre-training rather than target evaluation because the target hardware uses a tablet stylus.
# https://www.ms.sapientia.ro/~manyi/mobisig.html
MOBISIG_DIR = os.path.join(DATA_DIR, "MOBISIG")

# SigComp2011 (ICDAR benchmark competition): 84 dynamic signers — 64 Dutch, 20
# Chinese — recorded via WACOM Intuos3 A3 digitizing tablet at 200 Hz reporting 0–1023
# force values. Shares digitizer architecture with SVC2004, placing signers within
# the target hardware domain. Archives unpack directly here; parser logic traverses contents dynamically.
# Refer to README for download instructions.
SIGCOMP2011_DIR = os.path.join(DATA_DIR, "sigcomp2011")

# DeepSignDB (BiDA Lab, Universidad Autónoma de Madrid): Combined collection of MCYT,
# BiosecurID, BiosecureDS2, and e-BioSign, partitioned by original authors into Development
# and Evaluation subsets. The stylus partition provides the primary online dataset volume—
# offering 498 Development subjects compared to 124 in baseline sets.
#
# **Restricted License Dataset.** Access requires a signed institutional agreement (refer
# to README DeepSignDB notes). Datasets, extracted representations, and trained weights
# are restricted to academic evaluation and are excluded from public repositories.
DEEPSIGNDB_DIR = os.path.join(DATA_DIR, "deepsigndb")

# ─── DataLoader ───────────────────────────────────────────────────────
NUM_WORKERS = 4                         # Set to 0 if worker process hanging occurs on Windows
PIN_MEMORY = False                      # CPU-bound execution — memory pinning provides no benefit

# ─── Spatial Branch (Offline — Siamese CNN) ───────────────────────────
SPATIAL_INPUT_SIZE = (220, 155)          # cv2 (width, height) → [1, 1, 155, 220] tensor
SPATIAL_EMBEDDING_DIM = 512
SPATIAL_CHANNELS = 1                    # Grayscale

# Image spatial mapping behavior.
#   True  — Pre-pad image to square prior to scaling. Maintains aspect ratio; wide
#           BHSig260 ribbons maintain proportion but allocate canvas area to padding.
#   False — Direct bilinear scaling to SPATIAL_INPUT_SIZE. Modifies aspect ratio while
#           maximizing signature canvas occupancy, following Dey et al. Siamese CNN standard (155x220).
# Verified via empirical evaluation — see docs/SOLUTIONS.md for comparison data.
SPATIAL_SQUARE_PAD = False

# Quality filtering limits (proxies for effective image resolution)
MIN_BBOX_PIXELS = 64                    # Minimum signature bounding box edge dimension
MIN_LAPLACIAN_VARIANCE = 50.0           # Minimum Laplacian sharpness threshold

# ─── Temporal input gate ─────────────────────────────────────────────
# Quality screening for trajectory input streams. `normalize_trajectory` executes
# inside dataset parsers but not on direct inference paths, requiring boundary verification
# to prevent unnormalized raw tablet inputs from reaching models trained on normalized coordinates.
#
# `normalize_trajectory` applies standard scaling (z-score) per axis, producing zero-mean,
# unit-variance distributions. Thresholds reject unscaled digitizer coordinates while accepting standard variance.
MIN_TRAJECTORY_POINTS = 10              # Trajectories below 10 points lack sufficient trajectory detail
MAX_NORMALIZED_XY_MEAN = 0.5            # Un-normalized inputs exceed mean threshold
NORMALIZED_XY_STD_RANGE = (0.2, 5.0)    # Standard deviation outer boundaries

# ─── Temporal Branch (Online — Siamese LSTM) ──────────────────────────
TEMPORAL_INPUT_FEATURES = 3             # (X, Y, Pressure)
TEMPORAL_EMBEDDING_DIM = 128
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_NUM_LAYERS = 1
# Evaluated 2026-07-29 and adjusted to 0.0. Training with pressure channel dropout was evaluated
# to measure degraded mode robustness; empirical scoring showed higher error rates on both standard
# and pressure-blind inputs. Re-activating dropout requires fresh empirical validation.
PRESSURE_DROPOUT_RATE = 0.0
PRESSURE_IMPUTATION_VALUE = 1.0

# ─── Pair Generation ─────────────────────────────────────────────────
# Per-subject pair sampling limits for active model training (the 5/5 demonstration default
# is too restrictive for full convergence).
# 276 = C(24,2), representing total distinct genuine combinations available from a 24-sample sign-set per writer.
PAIRS_GENUINE_PER_WRITER = 276
PAIRS_IMPOSTOR_PER_WRITER = 276




# Proportion of negative pair allocations assigned to **random impostors** —
# genuine signatures from non-matching signers — rather than targeted skilled forgeries.
# Evaluated during training phase only; benchmark scoring continues to evaluate strictly against skilled forgeries (refer to AGENTS.md).
RANDOM_IMPOSTOR_RATIO = 0.5

# ─── Loss & Training ─────────────────────────────────────────────────
CONTRASTIVE_MARGIN = 1.0
HARD_NEGATIVE_RATIO = 0.5               # Fraction of impostor terms kept by mining
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Regularization dropout applied to feature vectors prior to spatial embedding projection.
# Disabled by default based on validation metrics.
SPATIAL_HEAD_DROPOUT = 0.0

# Convolutional backbone selection for offline spatial processing — refer to `src/models/siamese.py::SPATIAL_TRUNKS`.
# "compact" designates the custom 32-64-128-256-256 pipeline; "wide" specifies the Dey et al. Siamese reference backbone.
SPATIAL_TRUNK = "wide"
SPATIAL_CONV_DROPOUT = 0.0
SPATIAL_FC_HIDDEN = 0

# Number of gradient checkpointing segments during backward pass to reduce peak memory usage.
# Setting 0 turns off activation recomputation. Wide trunks utilize 4 segments to fit target mini-batches within 4 GB VRAM limits.
SPATIAL_GRAD_CHECKPOINT = 4

# Writer-independent partition distribution ratios (train, val, test).
SPATIAL_SPLIT_RATIOS = (0.7, 0.15, 0.15)
TEMPORAL_SPLIT_RATIOS = (0.5, 0.25, 0.25)

# Per-writer pair generation budget for dynamic temporal processing.
TEMPORAL_PAIRS_GENUINE_PER_WRITER = 180
TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER = 180

# ─── Enrollment Evaluation Protocol ──────────────────────────────────
# Operational protocol for dynamic signature verification: subjects register K genuine sample
# signatures, and dynamic probes are evaluated against closest reference embeddings.
# K = 5 was chosen because the SVC2004 competition and real-world deployment enroll 5 genuine signatures.
# See AGENTS.md (Enrollment Protocol) and SUMMARY.md (Part 7 — Measuring how good it is).
NUM_ENROLLMENT_REFERENCES = 5

# Number of independent reference draws averaged during evaluation to minimize sample variation bias.
ENROLLMENT_TRIALS = 10

# Dynamic data augmentation flag for temporal sequences (spatial rotation, jitter, speed variation).
TEMPORAL_AUGMENT = False

# Spatial data augmentation flag: mild affine transformation and stroke width perturbation applied to training set only.
SPATIAL_AUGMENT = True

# Early stopping configuration (monitors validation loss convergence)
EARLY_STOPPING_PATIENCE = 5            # Epoch patience threshold before stopping training
EARLY_STOPPING_MIN_DELTA = 1e-4         # Minimum loss decrement threshold

# ─── Fusion & Decision ───────────────────────────────────────────────
# Modal combination weighting factor for decision fusion score calculation: S_final = α · S_spatial + (1 − α) · S_temporal
# ALPHA_DEFAULT = 0.5 is provisional equal weighting as required by AGENTS.md until in-house multimodal dataset is collected.
ALPHA_DEFAULT = 0.5                     # PROVISIONAL — equal branch weighting
ALPHA_GRID_STEP = 0.05

# Decision confidence threshold settings (S_final >= tau -> Genuine, else Forged).
# Thresholds are calibrated on validation set under enrollment protocol (K=5) at EER balance point (FAR = FRR).
# See AGENTS.md (Evaluation Protocol) and SUMMARY.md (Part 7: What it looks like & threshold calibration).
SPATIAL_DECISION_THRESHOLD = 0.6299     # Calibrated spatial branch threshold (EER on validation)
TEMPORAL_DECISION_THRESHOLD = 0.7900    # Calibrated temporal branch threshold (EER on validation)
DECISION_THRESHOLD = 0.8196             # Calibrated threshold tau for late-fused model evaluation

# ─── Evaluation Targets ──────────────────────────────────────────────
# Target error metrics specified in Term-1 design requirements (AGENTS.md Hard Constraints).
TARGET_FAR = 0.05                       # 5% target False Acceptance Rate
TARGET_FRR = 0.05                       # 5% target False Rejection Rate
TARGET_ACCURACY = 0.95                  # 95% target accuracy (100% - EER)


