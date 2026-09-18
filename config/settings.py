"""
Configuration Core — Signature Verification System (EE 499)

Consolidated hyperparameter definitions, system boundaries, paths,
and global operational constants.

Model Weights Storage Note:
  Binary `.pth` weight files are excluded from git tracking via `.gitignore`
  and are distributed through GitHub Releases for local setup.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Root data directories. `data/` is gitignored — local download is required; no datasets are checked into git.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CEDAR_DIR = os.path.join(DATA_DIR, "cedar")          # full_org/ + full_forg/
BHSIG260_DIR = os.path.join(DATA_DIR, "bhsig260")    # Bengali/ + Hindi/
UTSIG_DIR = os.path.join(DATA_DIR, "utsig")          # Genuine/ + Forgery/

# SVC2004 provides two task variants. We keep both to train separate temporal models and measure the impact of pen pressure.
SVC2004_TASK1_DIR = os.path.join(DATA_DIR, "svc2004_task1")   # 4-col, no pressure
SVC2004_TASK2_DIR = os.path.join(DATA_DIR, "svc2004_task2")   # 7-col, real pressure
SVC2004_DIR = SVC2004_TASK2_DIR                               # default: pressure-aware

# MOBISIG: 83 users signing with a finger on touchscreens rather than a stylus. Used for temporal pre-training rather than target evaluation since our target hardware is a digitizer tablet.
# https://www.ms.sapientia.ro/~manyi/mobisig.html
MOBISIG_DIR = os.path.join(DATA_DIR, "MOBISIG")

# SigComp2011 (ICDAR benchmark): 84 online signers (64 Dutch, 20 Chinese) recorded on a Wacom Intuos3 tablet at 200 Hz with 0–1023 force levels. Matches the pen digitizer domain of SVC2004. See README for download details.
SIGCOMP2011_DIR = os.path.join(DATA_DIR, "sigcomp2011")

# SigComp2011 offline (scanned images) — the spatial sibling of the online set.
# Three files re-derived this as a raw "data/sigcomp2011_offline" string because
# no constant existed, which only resolved when cwd happened to be the repo root.
SIGCOMP2011_OFFLINE_DIR = os.path.join(DATA_DIR, "sigcomp2011_offline")

# DeepSignDB (BiDA Lab, UAM): Aggregates MCYT, BiosecurID, BiosecureDS2, and e-BioSign. The stylus partition supplies 498 Development signers (compared to 124 in baseline sets).
# **Restricted License Dataset:** Requires a signed agreement. Datasets, extracted features, and trained weights are restricted to academic evaluation and are excluded from public releases.
DEEPSIGNDB_DIR = os.path.join(DATA_DIR, "deepsigndb")

# ─── DataLoader ───────────────────────────────────────────────────────
NUM_WORKERS = 4                         # Set to 0 if worker process hanging occurs on Windows
PIN_MEMORY = False                      # CPU-bound execution — memory pinning provides no benefit

# ─── Spatial Branch (Offline — Siamese CNN) ───────────────────────────
SPATIAL_INPUT_SIZE = (220, 155)          # cv2 (width, height) → [1, 1, 155, 220] tensor
SPATIAL_EMBEDDING_DIM = 512

# Image spatial mapping behavior.
#   True  — Pre-pad image to square prior to scaling. Maintains aspect ratio; wide BHSig260 ribbons maintain proportion but allocate canvas area to padding.
#   False — Direct scaling to SPATIAL_INPUT_SIZE (155x220). Modifies aspect ratio while maximizing signature canvas occupancy, following Dey et al.
# Verified by measurement: direct scaling won on every corpus. Square-padding
# spends canvas on padding for the wide BHSig260 ribbons and measured worse.
SPATIAL_SQUARE_PAD = False

# Quality filtering limits (proxies for effective image resolution)
MIN_BBOX_PIXELS = 64                    # Minimum signature bounding box edge dimension
MIN_LAPLACIAN_VARIANCE = 50.0           # Minimum Laplacian sharpness threshold

# ─── Trajectory rasterizer (src/data/renderer.py) ────────────────────
# Renders a pen trajectory into the spatial branch's input so one signing event
# can feed both branches. Every constant lives here rather than as a literal in
# the renderer: a duplicated `(220, 155)` would silently disagree with
# SPATIAL_INPUT_SIZE the moment either moved, which is the same failure a
# hardcoded channel count already caused on the temporal side.
RENDER_MARGIN_RATIO = 0.12              # Fraction of each edge left blank, so strokes never clip
RENDER_BASE_THICKNESS = 2.0             # Stroke width in pixels at full pressure
RENDER_MIN_THICKNESS = 1                # Thinnest drawable stroke; cv2.line needs >= 1
# A sample is pen-up when pressure vanishes OR the clock jumps. SigComp2011
# records pen-up travel as force = 0, but SVC2004 and DeepSignDB mark a lift by
# the time gap alone, so pressure on its own would draw a line across every lift
# in those corpora.
RENDER_PEN_UP_PRESSURE = 1e-4           # Pressure at or below this is not touching
# The gap is judged against the signature's OWN median sampling period, not an
# absolute number of seconds. Sampling rates differ per corpus -- SVC2004 ~100 Hz,
# SigComp2011 200 Hz, DeepSignDB resampled to 10 ms -- so a fixed cutoff such as
# 0.05 s calls every sample a pen lift on any corpus slower than 20 Hz, and
# catches none on a fast one.
RENDER_PEN_UP_DT_RATIO = 3.0            # A gap this many times the median period is a lift

# ─── Temporal ensemble (SignatureService use_temporal_ensemble=True) ──
# Averaging these six checkpoints' embeddings measured 4.82% EER against 6.92%
# for a single member, on 120 unseen DeepSignDB writers under one shared K=5
# draw. Averaging their per-model SCORES instead reaches 4.53%, but that cannot
# be expressed through the `.encode()` interface `embed()` calls, so the
# embedding mean ships and gives up 0.29 points for a far smaller change.
# These weights are gitignored, but they ARE published as release assets from
# v0.9.0 on (all six `temporal_dt_s4*.pth`), so a machine that only has the
# release CAN form the ensemble. The previous note here said the opposite,
# which is why CI fetched 2 of the 8 assets and the one test that loads a real
# ensemble skipped on every run.
TEMPORAL_ENSEMBLE_SEEDS = (42, 43, 44, 45, 46, 47)
TEMPORAL_ENSEMBLE_TEMPLATE = "temporal_dt_s{seed}.pth"

# ─── Temporal input gate ─────────────────────────────────────────────
# Quality screening for trajectory streams. `normalize_trajectory` runs inside dataset parsers but not on direct inference paths, requiring boundary checks to prevent unnormalized raw tablet inputs from reaching models trained on normalized coordinates.
# `normalize_trajectory` applies z-score normalization per axis (zero-mean, unit-variance).
MIN_TRAJECTORY_POINTS = 10              # Trajectories below 10 points lack sufficient detail
MAX_NORMALIZED_XY_MEAN = 0.5            # Unnormalized inputs exceed mean threshold
NORMALIZED_XY_STD_RANGE = (0.2, 5.0)    # Standard deviation bounds

# ─── Temporal Branch (Online — Siamese LSTM) ──────────────────────────
# The shipped temporal input contract. `dt` is the inter-sample time gap: 10 ms while writing, longer across pen lifts, exposing pause rhythm to the model. Adding `dt` improved pooled EER from 10.45% to 8.97%. See SUMMARY.md Part 9 §10.
# Capture hardware must supply timestamps (channel 3 of standard 7-channel tablet data).
TEMPORAL_CHANNELS = "dt"                # "xyp" for the legacy 3-channel weights
# Input feature dimension per channel mode. Used to build model input width dynamically.
TEMPORAL_CHANNEL_WIDTHS = {"xyp": 3, "dt": 4, "tilt": 5}
TEMPORAL_INPUT_FEATURES = TEMPORAL_CHANNEL_WIDTHS[TEMPORAL_CHANNELS]
TEMPORAL_EMBEDDING_DIM = 128
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_NUM_LAYERS = 1
# Pressure dropout rate during training. Set to 0.0 based on validation results showing higher error rates when pressure dropout was active.
PRESSURE_DROPOUT_RATE = 0.0
PRESSURE_IMPUTATION_VALUE = 1.0

# ─── Pair Generation ─────────────────────────────────────────────────
# Per-subject pair sampling limits for active model training.
# 276 = C(24,2), representing total distinct genuine combinations available from a 24-sample set per writer.
PAIRS_GENUINE_PER_WRITER = 276
PAIRS_IMPOSTOR_PER_WRITER = 276

# Proportion of negative pairs drawn as random impostors (different signers) vs skilled forgeries during training.
# Benchmark evaluation always scores strictly against skilled forgeries per AGENTS.md.
RANDOM_IMPOSTOR_RATIO = 0.5

# ─── Loss & Training ─────────────────────────────────────────────────
CONTRASTIVE_MARGIN = 1.0
HARD_NEGATIVE_RATIO = 0.5               # Fraction of impostor terms kept by hard-negative mining
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Regularization dropout applied to feature vectors prior to spatial embedding projection.
SPATIAL_HEAD_DROPOUT = 0.0

# Convolutional backbone selection for spatial branch ("compact" = 32-64-128-256-256; "wide" = Dey et al. reference backbone).
SPATIAL_TRUNK = "wide"
SPATIAL_CONV_DROPOUT = 0.0
SPATIAL_FC_HIDDEN = 0

# Gradient checkpointing segments to reduce peak memory usage. Set 0 to disable. Wide trunk uses 4 segments to fit mini-batches in 4 GB VRAM.
SPATIAL_GRAD_CHECKPOINT = 4

# Writer-independent partition ratios (train, val, test).
SPATIAL_SPLIT_RATIOS = (0.7, 0.15, 0.15)
TEMPORAL_SPLIT_RATIOS = (0.5, 0.25, 0.25)

# Per-writer pair generation budget for dynamic temporal processing.
TEMPORAL_PAIRS_GENUINE_PER_WRITER = 180
TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER = 180

# ─── Enrollment Evaluation Protocol ──────────────────────────────────
# Operational protocol: subjects register K=5 genuine signatures, and probe signatures match against nearest enrolled reference embedding.
# K = 5 aligns with standard benchmark protocols (SVC2004). See AGENTS.md and SUMMARY.md.
NUM_ENROLLMENT_REFERENCES = 5

# Number of independent reference draws averaged during evaluation to minimize sample variation.
ENROLLMENT_TRIALS = 10

# Data augmentation flags.
TEMPORAL_AUGMENT = False
SPATIAL_AUGMENT = True

# Augmentation strengths. These were function defaults inside `src/data/dataset.py`
# and restated a second time at the call site, so retuning one left the other stale.
# Ranges stay deliberately small: forgery detection keys on fine stroke geometry,
# and warping past that teaches tolerance to the very differences it must detect.
SPATIAL_AUG_MAX_ROTATION_DEG = 6.0
SPATIAL_AUG_MAX_SHEAR = 0.08
SPATIAL_AUG_MAX_SCALE = 0.08
SPATIAL_AUG_MAX_TRANSLATE = 0.04
SPATIAL_AUG_STROKE_PROB = 0.3
TEMPORAL_AUG_MAX_ROTATION_DEG = 6.0
TEMPORAL_AUG_JITTER = 0.02
TEMPORAL_AUG_MAX_SPEED_CHANGE = 0.15

# Target sampling rate for online trajectories, and the period that follows from
# it. Everything inside src/ derives from these two: features.py's Nyquist term,
# and parsers.py's SIGCOMP_SAMPLE_MS and DEEPSIGN_TARGET_PERIOD_MS, which are the
# rates the two pen corpora are decimated *onto*. Change the rate here and the
# resampling follows; a literal 10.0 left behind in a parser would silently
# resample one corpus to a different clock than the model was trained on.
#
# parsers.MAX_GAP_MS is NOT part of this family. It is 100 *milliseconds* — the
# cap on an inter-sample gap — and shares the number 100 with the rate in hertz
# by coincidence. Wiring them together would couple the dt clip to the sample
# rate and break the `dt` channel's [0, 1] contract.
#
# tools/sigma_lognormal.py restates the rate deliberately: it is a clean-room
# implementation with no project imports. Keep the two in step by hand.
TEMPORAL_SAMPLE_RATE_HZ = 100.0
TEMPORAL_SAMPLE_PERIOD_MS = 1000.0 / TEMPORAL_SAMPLE_RATE_HZ

# Early stopping configuration (monitors validation loss convergence).
EARLY_STOPPING_PATIENCE = 5            # Epoch patience threshold before stopping training
EARLY_STOPPING_MIN_DELTA = 1e-4         # Minimum loss decrement threshold

# ─── Fusion & Decision ───────────────────────────────────────────────
# Modal combination score calculation: S_final = α · S_spatial + (1 − α) · S_temporal
# ALPHA_DEFAULT = 0.5 was measured, not assumed. On paired SigComp2011 writers
# (25 writer-disjoint splits, alpha fitted on 10 writers and scored on the other
# 10) the fixed 0.5 scored 13.85% against 14.57% for a per-split grid search;
# the fitted value was 0.548 +/- 0.122, statistically the default. Cascade, AND,
# likelihood-ratio and band rules all measured worse than the plain average
# (SUMMARY.md Part 9, sections 7 and 25). Do not grid-search alpha on small
# calibration sets; it overfits.
ALPHA_DEFAULT = 0.5                     # Measured: equal weighting is the optimum
ALPHA_GRID_STEP = 0.05

# Decision threshold settings (S_final >= tau -> Genuine, else Forged).
# Calibrated on validation set under enrollment protocol (K=5) at EER balance point (FAR = FRR).
# See AGENTS.md and SUMMARY.md.
SPATIAL_DECISION_THRESHOLD = 0.7711     # Calibrated spatial branch threshold (EER on validation)
TEMPORAL_DECISION_THRESHOLD = 0.7900    # Calibrated temporal branch threshold (EER on validation)
DECISION_THRESHOLD = 0.8196             # Calibrated threshold tau for late-fused model evaluation

# ─── Runtime Budget (AGENTS.md Hard Constraints) ─────────────────────
# These were documented in AGENTS.md as hard constraints but existed nowhere in
# code, so nothing could enforce or even report them. Declared here so
# `tools/profile_latency.py` and the test suite can assert against the same
# numbers the spec states.
LATENCY_SLA_SECONDS = 3.0               # End-to-end verification decision budget
INFERENCE_RAM_CEILING_GB = 4.0          # Peak RSS ceiling for a single verification

# Thread ceiling applied at model load. Torch and OpenCV each default to one
# worker per logical CPU with no coordination between them. Measured on an 8-core
# box: an identical spatial forward pass took 41 ms idle and 172-322 ms while one
# other CPU-bound process ran — a 4-7x swing on the same code. Capping does not
# improve the idle case (1 thread measured 221 ms vs 48 ms at 8); it bounds the
# contended case, which is the "consumer hardware" the 3 s budget is written for.
INFERENCE_NUM_THREADS = 4

# ─── Evaluation Targets ──────────────────────────────────────────────
# Term-1 target error metrics. NOT from AGENTS.md's Hard Constraints table,
# which lists only RAM, latency, execution, platform and build steps. README's
# device-tier table is the source, and it is tiered (primary <=5%, secondary
# <=10%, tertiary <=20%); these flat constants encode the primary tier only.
TARGET_FAR = 0.05                       # 5% target False Acceptance Rate
TARGET_FRR = 0.05                       # 5% target False Rejection Rate
TARGET_ACCURACY = 0.95                  # 95% target accuracy (100% - EER)



