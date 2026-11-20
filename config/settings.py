"""
Configuration settings and constants for signature verification system.
Contains model hyperparameters, paths, hardware constraints, and decision thresholds.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
# The shipped weights (GitHub Releases), each with a <name>.split.json record
# of the corpora and writer partition it was trained on.
SPATIAL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth")
TEMPORAL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "temporal_lstm.pth")
# Release tag of the two shipped weights above; both are byte-identical to that
# release's assets. Storage saves it beside every template, and a template whose
# tag differs was embedded by other weights and needs re-enrolment. Bump it
# whenever either shipped weight changes (ship-model-release step 5). CI downloads
# the weights of this release, so it always tests what ships.
MODEL_VERSION = "v0.10.0"
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

# SigComp2009 / NFI: paired online (.hwr) and offline (.png) of the same signing
# events, used only as an evaluation corpus (fusion replication, cross-corpus
# spatial checks). Not a training corpus for any shipped weight.
SIGCOMP2009_DIR = os.path.join(DATA_DIR, "sigcomp2009_dl", "extracted")

# DeepSignDB (BiDA Lab, UAM): Aggregates MCYT, BiosecurID, BiosecureDS2, and e-BioSign. The stylus partition supplies 498 Development signers (compared to 124 in baseline sets).
# **Restricted License Dataset:** Requires a signed agreement. Datasets, extracted features, and trained weights are restricted to academic evaluation and are excluded from public releases.
DEEPSIGNDB_DIR = os.path.join(DATA_DIR, "deepsigndb")

# BiosecurID-SONOF (BiDA Lab, UAM): 132 writers, a pen recording (OnlineReal/) and a scan
# (OfflineReal/) of the same signing. **Restricted License Dataset**, as DeepSignDB:
# evaluation and calibration only; no weight trained on it ships.
BIOSECURID_SONOF_DIR = os.path.join(DATA_DIR, "biosecurid_sonof")
INHOUSE_DIR = os.path.join(DATA_DIR, "inhouse")      # XPPen campaign: <participant>/ + manifest.csv
# In-house writer split (src/data/campaign.py): a volunteer goes to test when a hash of SEED and
# their code falls under this fraction, so nobody changes side as volunteers are added. Forger
# codes name separate people who signed no genuines, so they decide nothing. The record lists
# participant codes only and is tracked in git.
INHOUSE_TEST_FRACTION = 0.5
# Sessions the owner dropped (P06, P16, P18 have no forgeries; P19 was never recorded).
INHOUSE_EXCLUDED = ("P06", "P16", "P18", "P19")
# Genuines recorded on Windows without the XPPen driver (Windows Ink), forgeries with it (Wintab).
# G10, one signer both ways: without the driver peak 0.22 / median 0.18, with it 0.55 / 0.47, so
# a Wintab forgery of these writers is scaled by 0.4 to their genuines' scale.
INHOUSE_NO_DRIVER = ("P10", "P11", "P12", "P13")
INHOUSE_NO_DRIVER_PRESSURE_SCALE = 0.4
INHOUSE_SPLIT_PATH = os.path.join(PROJECT_ROOT, "config", "inhouse_split.json")

# The corpora each branch trains on, which are the ones the shipped weights
# trained on. Scoring a shipped checkpoint takes its corpora and partition from
# its split record instead, and tests/test_leakage.py fails when the two
# disagree. Not "every image corpus the parsers can read": that list includes the
# evaluation-only sigcomp2011_offline, and a split built from it scored the shipped
# spatial weight on 5 of its own training writers.
SPATIAL_TRAIN_DIRS = (CEDAR_DIR, BHSIG260_DIR, UTSIG_DIR)
TEMPORAL_TRAIN_DIRS = (SVC2004_TASK2_DIR, SIGCOMP2011_DIR)

# ─── DataLoader ───────────────────────────────────────────────────────
NUM_WORKERS = 4                         # Set to 0 if worker process hanging occurs on Windows
PIN_MEMORY = False                      # CPU-bound execution — memory pinning provides no benefit

# ─── Spatial Branch (Offline — Siamese CNN) ───────────────────────────
def spatial_input_size_from_env(value, default):
    """
    Parse the image size override from SIGVER_SPATIAL_INPUT_SIZE (e.g. '220x155').
    Used during input size sweeps. Raises ValueError if format is invalid.
    """
    if value is None or value == "":
        return default
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"SIGVER_SPATIAL_INPUT_SIZE must be WxH in pixels, e.g. 220x155; got {value!r}")
    return (int(parts[0]), int(parts[1]))


SPATIAL_INPUT_SIZE = spatial_input_size_from_env(
    os.environ.get("SIGVER_SPATIAL_INPUT_SIZE"), (220, 155))   # cv2 (width, height) → [1, 1, 155, 220] tensor
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
RENDER_BASE_THICKNESS = 3.0             # Stroke width in pixels at full pressure; 3 px measured
                                        # 4.8 points better than 2 px on 20 SigComp2011 writers
                                        # (rendered-only EER, 10/10 draws), 0.6 better on NFI
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
TRAJECTORY_DT_MEDIAN_TOLERANCE = 0.10   # Median dt may sit at most 10% off one 10 ms grid step
WINDOWS_PEN_API = "wintab"               # Windows pen input: "wintab" (tablet driver, full pressure range) or "ink" (Windows Ink, 1024 steps; G10)
CAPTURE_NO_DRIVER = "no driver"         # device_model when no tablet name can be read (no tablet, or Windows without Wintab); enrol and verify on one tablet model (F23)
CAMPAIGN_GENUINE_PER_MODALITY = 12      # Genuine signatures per volunteer on tablet and on paper
CAMPAIGN_PRACTICE_SIGNINGS = 3          # Tablet practice signatures first; never saved
CAMPAIGN_BREAK_AFTER = 6                # Short break after this many tablet genuines
CAMPAIGN_FORGERY_ATTEMPTS = 2           # A forger's attempts per victim; "Sign again" before the 2nd
SHEET_GRID = (4, 2)                     # Paper sheet boxes: rows, columns (8 per A4 sheet)
SHEET_DPI = 300                         # Printed sheet and scanned PDFs are drawn at this resolution
SHEET_MARGIN = 0.05                     # Fraction of each box cut away, so the printed border stays out
# Ink crop for scanned paper boxes (tools/capture_session.py, ENG_REPORT A6). The app's crop_to_ink
# drops faint ballpoint strokes; this one scores ink against the local paper, so shadows cancel.
INK_BACKGROUND_KERNEL = 51              # Median blur size (px) that estimates the paper under the ink
INK_MIN_CONTRAST = 18                   # Ink is at least this much darker than its local paper (0-255)
INK_MIN_SATURATION = 10                 # ...and coloured: grey shadows, creases and page edges are not (0 lets folds in)
INK_HUE_RANGE = (95, 165)               # Blue ballpoint, OpenCV hue 0-179
INK_CLUSTER_GAP = 61                    # Ink pieces closer than this (px) belong to one signature
INK_MIN_PIXELS = 200                    # Fewer ink pixels than this: no signature in the box
INK_CROP_PAD = 0.08                     # Margin around the ink, as a fraction of its longer side
INK_LINE_MIN_FRACTION = 0.5             # A straight run this long (share of the box side) may be print
INK_LINE_MAX_BLUE = 0.25                # ...and is removed when less than this share of it is blue

# ─── Temporal Branch (Online — Siamese LSTM) ──────────────────────────
# The shipped temporal input contract. `dt` is the inter-sample time gap: 10 ms while writing, longer across pen lifts, exposing pause rhythm to the model. Adding `dt` improved pooled EER from 10.45% to 8.97%.
# Capture hardware must supply timestamps (channel 3 of standard 7-channel tablet data).
TEMPORAL_CHANNELS = "dt"                # "xyp" for the legacy 3-channel weights
# Input feature dimension per channel mode. Used to build model input width dynamically.
TEMPORAL_CHANNEL_WIDTHS = {"xyp": 3, "dt": 4, "tilt": 5,
                           # Research only (AGENTS.md exception): dt + tilt, with a fill
                           # value for corpora without tilt. The fill is in the name
                           # because a checkpoint records its width, not its fill.
                           "dt_tilt0": 6, "dt_tilt1": 6, "dt_tilt_flag": 7}
# Tilt mode -> fill value for corpora that carry no tilt.
TILT_CHANNEL_MODES = {"dt_tilt0": 0.0, "dt_tilt1": 1.0, "dt_tilt_flag": 0.0}
# dt_tilt_flag only: chance a training-side SVC2004 sequence has its tilt hidden.
TEMPORAL_TILT_HIDE_PROB = 0.5
TEMPORAL_INPUT_FEATURES = TEMPORAL_CHANNEL_WIDTHS[TEMPORAL_CHANNELS]
TEMPORAL_EMBEDDING_DIM = 128
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_NUM_LAYERS = 1
# Research arm, off in the shipped model: a Conv1d stage whose output is joined to each sample
# before the LSTM, so the LSTM also sees a short piece of stroke around the sample.
TEMPORAL_CONV_FILTERS = 0               # 0 = no conv stage (shipped)
TEMPORAL_CONV_KERNEL = 5                # Samples the conv spans; 5 is 50 ms at 100 Hz. Odd.
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

# Seed for the writer split, pair sampling and training. Every published number uses 42.
SEED = 42

# Writer-independent partition ratios (train, val, test).
SPATIAL_SPLIT_RATIOS = (0.7, 0.15, 0.15)
TEMPORAL_SPLIT_RATIOS = (0.5, 0.25, 0.25)

# Per-writer pair generation budget for dynamic temporal processing.
TEMPORAL_PAIRS_GENUINE_PER_WRITER = 180
TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER = 180

# ─── Enrollment Evaluation Protocol ──────────────────────────────────
# Operational protocol: subjects register K=5 genuine signatures, and probe signatures match against nearest enrolled reference embedding.
# K = 5 aligns with standard benchmark protocols (SVC2004). See AGENTS.md.
NUM_ENROLLMENT_REFERENCES = 5
# Most references one client keeps per modality: the enrolled ones plus confirmed signings
# added after a Genuine verification, oldest added dropped first. The measured arms held 6 to 8
# (AGENTS.md); nothing beyond 8 was measured. The GUI storage and the database both read this.
MAX_REFERENCES = 8

# Number of independent reference draws averaged during evaluation to minimize sample variation.
ENROLLMENT_TRIALS = 10

# Data augmentation flags.
TEMPORAL_AUGMENT = False
SPATIAL_AUGMENT = True

# Training reads the SVC2004 and DeepSignDB files as float32, the way the loaders once
# did; every scorer reads them as float64. float32 cannot hold every millisecond past
# 2**24 ms, and SVC2004's clock runs to 7.5e7 ms, so on most of its files 10 ms steps
# come out as 8, 12 or 16 ms. That rounding is the only noise the shipped dt recipe
# trains with: on exact timestamps the same recipe is 2.8 EER points worse in-domain.
TEMPORAL_TRAIN_FLOAT32_PARSE = True

# Augmentation strengths. These were function defaults inside `src/data/dataset.py`
# and restated a second time at the call site, so retuning one left the other stale.
# Ranges stay deliberately small: forgery detection keys on fine stroke geometry,
# and warping past that teaches tolerance to the very differences it must detect.
SPATIAL_AUG_MAX_ROTATION_DEG = 6.0
SPATIAL_AUG_MAX_SHEAR = 0.08
SPATIAL_AUG_MAX_SCALE = 0.08
SPATIAL_AUG_MAX_TRANSLATE = 0.04
SPATIAL_AUG_STROKE_PROB = 0.0           # Measured 2026-09-26: stroke dilate/erode jitter costs 1.23 pooled EER points (report L9)
# Scanner effects (E3, report F9 follow-up): off by default, so the shipped recipe draws
# the same random stream and every released seed reproduces. Switched on per run with
# --aug-blur, --aug-ink-gain-min and --aug-noise.
SPATIAL_AUG_BLUR_SIGMA = 0.0            # Max Gaussian blur sigma in pixels
SPATIAL_AUG_INK_GAIN_MIN = 1.0          # Ink intensity scaled by U(min, 1); 1.0 is off
SPATIAL_AUG_NOISE_STD = 0.0             # Max additive noise sigma on the [0, 1] image
TEMPORAL_AUG_MAX_ROTATION_DEG = 6.0
TEMPORAL_AUG_JITTER = 0.02
TEMPORAL_AUG_MAX_SPEED_CHANGE = 0.15
# Device-spec augmentation (E2, --device-augment): a training trajectory re-recorded as
# another tablet inside the capture spec (README hardware table: >= 200 Hz, >= 1024
# pressure levels) would deliver it after ingest. The ranges are the spec's envelope, not
# measured device statistics: report rates common tablets ship with, and pressure curves
# from soft to firm.
TEMPORAL_DEVICE_AUG_RATE_HZ = (200.0, 300.0)
TEMPORAL_DEVICE_AUG_PRESSURE_GAMMA = (0.67, 1.5)
TEMPORAL_DEVICE_AUG_PRESSURE_GAIN = (0.85, 1.0)
TEMPORAL_DEVICE_AUG_PRESSURE_LEVELS = 1024

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
# Chosen on the 132 BiosecurID-SONOF paired writers (lowest fused EER, K=5, 10 draws),
# then confirmed on the 14 NFI writers, who took no part in choosing it.
ALPHA_DEFAULT = 0.4                     # Measured: SONOF optimum; equal weighting (0.5) scored worse on both
ALPHA_GRID_STEP = 0.05
# The 14 screened SigComp2009 (NFI) writers that check the fused alpha and tau after they
# are chosen on SONOF (ENG_REPORT K1, F14, F16). Neither branch trained on NFI, and each
# signing has a scan and a pen file with the same name, so the pairing is exact.
NFI_FUSION_WRITERS = ("nfi_002", "nfi_008", "nfi_016", "nfi_018", "nfi_024", "nfi_035",
                      "nfi_044", "nfi_063", "nfi_070", "nfi_071", "nfi_077", "nfi_085",
                      "nfi_086", "nfi_089")

# Signings asked for at verification, of each kind (paper and tablet). The GUI reads this as its
# default and may offer it as a toggle (1 or 2); verify() averages however many it is given, and
# every signing must pass its input gate or that kind of evidence is dropped. The thresholds below
# are shared: at a fixed threshold averaging can only lower FAR. Measured 2026-10-21, pen, K = 5,
# 140 BiosecureDS2 writers enrolled at one sitting: two signings took EER from 16.06% to 13.70% a
# session later and from 8.42% to 6.04% at the same sitting.
VERIFICATION_SIGNINGS = 1

# Step-up verification, off until the GUI can show "Sign again" (AGENTS.md). One signing of each
# kind first: a fused score of at least STEP_UP_ACCEPT is accepted, one below STEP_UP_REJECT is
# rejected, and anything between asks for one more signing of each kind, whose mean with the first
# must reach STEP_UP_TWO. Fused decisions only, with VERIFICATION_SIGNINGS left at 1. Calibrated on
# 132 BiosecurID-SONOF writers three sessions after enrolment, forgers assumed to repeat their first
# attempt (the worst case for false accepts), at most 30% asked twice: 11.4% of genuine signers
# refused at 5.1% of forgeries passed, against 15.7% for one signing at 5%; 18% asked twice.
# Kept off after the in-house campaign (F43): at one sitting it refused no genuine pair but passed
# 2 of 18 forger pairs (11.1%), against 1 of 36 forgeries (2.78%) for one signing at
# DECISION_THRESHOLD; STEP_UP_ACCEPT is below that threshold. Recalibrate on time-gap data first.
VERIFICATION_STEP_UP = False
STEP_UP_REJECT = 0.4229
STEP_UP_ACCEPT = 0.7514
STEP_UP_TWO = 0.7470

# Decision threshold settings (S_final >= tau -> Genuine, else Forged).
# Calibrated on validation set under enrollment protocol (K=5) at EER balance point (FAR = FRR).
# Paper-alone threshold, set by campaign rule D6 (issue #44 item 11): the EER point of the scan
# score on 18 in-house writers, paper through the app's grey crop (ENG_REPORT F44). At 0.7637,
# the EER point on the public validation writers, paper alone accepted about half of the forgeries.
# No threshold meets 5% / 5% for paper alone on our scans or on faint photos; fusion does.
SPATIAL_DECISION_THRESHOLD = 0.8387     # Measured paper-alone threshold (EER point on the campaign)
# Pen-alone threshold. 0.7381 (the FAR 5% point on SONOF) was tried and reverted: AGENTS.md keeps FAR
# low first, and at 0.7381 DeepSignDB MCYT and BiosecureDS2 accepted 11.6% and 12.0% of skilled
# forgeries. No fixed pen-alone threshold meets FAR 5% on every new collection (their FAR 5% points
# run from 0.72 to 0.83); the deployment's own writers set it.
TEMPORAL_DECISION_THRESHOLD = 0.7900    # Calibrated temporal branch threshold (EER on validation)
# Fused threshold, set by campaign rule D6 on our own XPPen tablet and paper (phone app and flatbed): the EER point
# of 0.4*s + 0.6*t on 18 in-house writers at one sitting (FAR 2.78%, FRR 1.60%, 36 skilled
# forgeries; ENG_REPORT F43). The SONOF value let 4 of 36 forgeries through there. On the
# public paired sets it trades FRR for FAR: SONOF FAR 2.53%, FRR 10.17%; NFI 2.23%, 6.92%.
DECISION_THRESHOLD = 0.7807             # Measured fused threshold (EER point on the in-house campaign)
# The EER point on the 132 SONOF paired writers (F16); scripts/calibrate_fusion.py reproduces it.
SONOF_FUSED_THRESHOLD = 0.7564

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



