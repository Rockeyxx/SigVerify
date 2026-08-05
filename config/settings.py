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

# SigComp2011 (ICDAR competition dataset): 84 online writers — 64 Dutch, 20
# Chinese — captured on a WACOM Intuos3 A3 pen tablet at 200 Hz with real 0–1023
# pen force. Same capture family as SVC2004, so unlike MOBISIG these writers are
# in the target pen domain. Both archives unzip straight into this directory and
# the parser walks whatever is present. See README for the download recipe.
SIGCOMP2011_DIR = os.path.join(DATA_DIR, "sigcomp2011")

# DeepSignDB (BiDA Lab, Universidad Autónoma de Madrid): an aggregate of MCYT,
# BiosecurID, BiosecureDS2 and e-BioSign, split by its authors into a Development
# set to train on and an Evaluation set that published results are reported on.
# Its stylus half is by far the largest pen-tablet corpus this project can reach —
# 498 Development writers against the 124 it had before.
#
# **Licensed, not public.** Held under a signed research-use agreement (see the
# README's DeepSignDB section). The data, any features derived from it, and any
# weights trained on it are research-only, never committed and never
# redistributed. `data/` is gitignored, which is what keeps that true by default.
DEEPSIGNDB_DIR = os.path.join(DATA_DIR, "deepsigndb")

# ─── DataLoader ───────────────────────────────────────────────────────
NUM_WORKERS = 4                         # 0 on Windows/notebook if workers hang
PIN_MEMORY = False                      # CPU-only training — pinning buys nothing

# ─── Spatial Branch (Offline — Siamese CNN) ───────────────────────────
SPATIAL_INPUT_SIZE = (220, 155)          # cv2 (width, height) → [1, 1, 155, 220] tensor
SPATIAL_EMBEDDING_DIM = 512
SPATIAL_CHANNELS = 1                    # Grayscale

# How a scan is mapped onto the network input.
#   True  — pad to a square first, then resize. Preserves aspect ratio; a wide
#           BHSig260 ribbon keeps its shape but spends most of the canvas on pad.
#   False — resize straight to SPATIAL_INPUT_SIZE. Distorts aspect, but every
#           signature fills the frame. This is what the Dey et al. Siamese CNN
#           reference does at 155x220.
# Measured rather than assumed — see docs/SOLUTIONS.md for the comparison.
SPATIAL_SQUARE_PAD = False

# Quality gate thresholds (effective-resolution proxies)
MIN_BBOX_PIXELS = 64                    # Minimum signature bounding-box dimension
MIN_LAPLACIAN_VARIANCE = 50.0           # Sharpness threshold

# ─── Temporal Branch (Online — Siamese LSTM) ──────────────────────────
TEMPORAL_INPUT_FEATURES = 3             # (X, Y, Pressure)
TEMPORAL_EMBEDDING_DIM = 128
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_NUM_LAYERS = 2
# Measured 2026-07-29 and set to 0. Pressure dropout was meant to buy degraded-mode
# robustness by training on flat P for a share of samples. It bought the opposite:
# at 0.4 it cost 2.47 pooled EER points on normal input (12.73% → 10.26%) *and* 3.00
# points on pressure-blind input (23.32% → 20.32%). It lost on both paths at once.
# Raising it again needs a measurement, not an argument from robustness.
PRESSURE_DROPOUT_RATE = 0.0
PRESSURE_IMPUTATION_VALUE = 1.0

# ─── Pair Generation ─────────────────────────────────────────────────
# Per-writer pair budgets for a real training run (the 5/5 demo defaults are
# far too small: 55 CEDAR writers × 5 would yield only ~550 pairs total).
# 276 = C(24,2), the number of distinct genuine pairs a 24-signature CEDAR writer
# supports, and the value the Dey et al. Siamese CNN reference draws. The old 40
# used 14% of the available supervision while the encoder memorised its training
# writers by epoch 4 (train 0.067 against val 0.155). The temporal branch already
# learned this lesson: 40 -> 180 pairs took it from 33.25% to 25.50% EER.
PAIRS_GENUINE_PER_WRITER = 276
PAIRS_IMPOSTOR_PER_WRITER = 276

# Fraction of each writer's negative budget spent on **random impostors** —
# another writer's genuine signature — instead of a skilled forgery of this
# writer. Training-time only; every reported metric still pairs against skilled
# forgeries alone (AGENTS.md eval protocol).
#
# Measured 2026-08-04 (full write-up in `docs/SOLUTIONS.md`): at 0.0 the encoder
# never sees a cross-writer negative, and on CEDAR it stopped separating writers
# entirely — a different person's genuine signature sat at mean distance 0.048
# against 0.038 for the same person's, while that writer's own skilled forgery
# sat at 2.043. The model had learned a genuine/forged axis, not identity, and
# scored 36.64% EER against random impostors while scoring 0.00% against skilled
# ones. Every dataset showed the same effect more weakly.
RANDOM_IMPOSTOR_RATIO = 0.0

# ─── Loss & Training ─────────────────────────────────────────────────
CONTRASTIVE_MARGIN = 1.0
HARD_NEGATIVE_RATIO = 0.5               # Fraction of impostor terms kept by mining
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Dropout on the pooled vector feeding the spatial embedding projection.
# Off by default. At 0.4 it costs about a point of held-out EER and barely
# narrows the train/validation gap, so head capacity is not what limits
# generalization here.
SPATIAL_HEAD_DROPOUT = 0.0

# Which convolutional trunk the spatial branch is built from — see
# `src/models/siamese.py::SPATIAL_TRUNKS`. "compact" is this project's own
# 32-64-128-256-256 stack; "wide" is the Dey et al. Siamese reference
# (96x11x11 stem, then 256, 384, 256). `SPATIAL_CONV_DROPOUT` and
# `SPATIAL_FC_HIDDEN` are the reference's other two departures: Dropout2d after
# the second and fourth stages, and an FC-1024 layer before the embedding.
#
# These are training-time settings only. Benchmarking and inference recover the
# architecture from the checkpoint itself via `SpatialSiamese.from_state_dict`,
# so changing them here can never mis-score an existing weights file.
# Measured 2026-07-28 as three separate arms, all at --enroll 5 on the same 65
# held-out writers: "wide" alone reached 11.78% pooled against "compact"'s
# 12.31% (utsig 33.76% → 30.41%), and is what ships. Adding the reference's
# dropout cost 6.78 points and adding its FC-1024 head on top collapsed the
# model. Leave the next two at 0: at 301 training writers the failure is data
# scarcity, not excess capacity, so regularizing removes capacity the model
# needs. Dey et al. can afford it because they train on GPDS300.
SPATIAL_TRUNK = "wide"
SPATIAL_CONV_DROPOUT = 0.0
SPATIAL_FC_HIDDEN = 0

# Segments to recompute the trunk in during the backward pass instead of storing
# its activations. 0 is off. This is a memory/compute trade, not a modelling
# choice — the gradients are identical. The compact trunk peaks at 1.5 GB at
# batch 32 and does not need it; the wide trunk needs 4 segments to fit the same
# batch on a 4 GB card, which it must, since batch size sets how many impostors
# hard-negative mining has to choose from.
SPATIAL_GRAD_CHECKPOINT = 4

# Writer-independent split ratios (train, val, test).
# The image datasets have 430 writers between them, so 70/15/15 still leaves 65
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

# ─── Enrollment Evaluation Protocol ──────────────────────────────────
# How the field measures online signature verification, and what
# `src/verification.py` actually does at inference: a user enrolls K genuine
# signatures and a probe is scored against the nearest of them.
#
# The SVC2004 competition enrolled exactly 5 genuine signatures per user, never
# showed skilled forgeries during enrollment, and averaged over several random
# enrollment draws. Its best Task 2 result was 2.89% EER under that protocol.
# Scoring isolated pairs instead measures a different, harder task, so published
# numbers were never comparable to this project's.
NUM_ENROLLMENT_REFERENCES = 5

# Independent reference draws to average. Which 5 signatures a user happens to
# enroll moves the result, and with only 10-13 held-out SVC2004 writers a single
# draw is a small sample. Averaging over draws — and reporting the spread —
# separates a real effect from enrollment luck.
ENROLLMENT_TRIALS = 10

# Perturb training trajectories with small rotation, jitter and speed changes.
# Off by default: measured, it cost 3.6 EER points (25.50% -> 29.12%) even
# though validation loss improved. Augmentation multiplies samples, but
# writer-independent verification has to learn what differs between *people*,
# and no perturbation of 20 signers' trajectories invents a 21st signer.
# Kept available because it should pay off once more writers are in the dataset.
TEMPORAL_AUGMENT = False

# Spatial augmentation: small affine warps plus occasional stroke-width change.
# Unlike the temporal case above, this one has a real argument behind it. Every
# spatial run best-validates at epoch 1 on 149,280 pairs, so the ceiling is the
# *variety* the 301 training writers hold, not the pair count. Warping images is
# the only way to add variety without new writers, and it is what the Siamese
# references rely on at this dataset size. Training split only — `build_loader`
# gates it on `shuffle`, so validation and test are never augmented.
SPATIAL_AUGMENT = True

# Early Stopping hyperparameters (monitors validation loss to halt training)
EARLY_STOPPING_PATIENCE = 5            # Stop after 10 epochs without validation improvement
EARLY_STOPPING_MIN_DELTA = 1e-4         # Minimum loss decrease required to trigger improvement

# ─── Fusion & Decision ───────────────────────────────────────────────
ALPHA_DEFAULT = 0.5                     # Tuned on validation via grid search
ALPHA_GRID_STEP = 0.05
# τ: S_final ≥ τ → Genuine. Tuned on the validation split, never on test, and
# re-derive it whenever the weights change. Similarity is S = 1/(1+d), so a
# threshold sits directly on a distance: the old default of 0.5 meant d = 1.0,
# which is exactly CONTRASTIVE_MARGIN. That put the decision boundary on the
# loss's own target with no slack, and every impostor the model had not pushed
# fully past the margin landed on the accept side.
#
# **Calibrate under the protocol you deploy.** Until 2026-08-04 τ was tuned on
# *pair-scoring* validation scores while `src/verification.py` scores under
# *enrolment* (nearest of K references). Taking a minimum over K references
# shrinks every distance, so scores shift up — genuine further than impostor —
# and a τ frozen at the pair-scoring EER point sits below the deployed
# distribution and accepts almost everything. Measured at the old 0.7143:
#
#   branch    pair-scoring            enrolment K=5
#   spatial   FAR 17.01/FRR 17.38     FAR 19.32/FRR  2.65
#   temporal  FAR 11.74/FRR 11.39     FAR 25.00/FRR  2.18
#
# A 36-probe MOBISIG sample independently measured FAR 22.2%/FRR 0.0%.
#
# Both values below are the FAR<=10% operating point, chosen on validation under
# enrolment at K=5 and then reported on test. FAR<=10% is the knee of the curve:
# below it the FRR cost per point of FAR roughly doubles (spatial FAR 10->5 costs
# +17 FRR), and FAR<=5% would reject one genuine user in three.
#
#   spatial  τ=0.7868 -> test FAR 9.25% / FRR 11.17%
#   temporal τ=0.8524 -> test FAR 7.97% / FRR 14.51%
#            sigcomp_Dutch, the XPPen deployment condition: FAR 1.80% / FRR 10.26%
#
# The two branches need different τ because their score distributions differ;
# one global value cannot serve both. DECISION_THRESHOLD remains the fused-path
# default, where scores from both branches are mixed.
SPATIAL_DECISION_THRESHOLD = 0.7868
TEMPORAL_DECISION_THRESHOLD = 0.8524
DECISION_THRESHOLD = 0.8196          # midpoint, fused path only

# ─── Evaluation Targets ──────────────────────────────────────────────
TARGET_FAR = 0.05
TARGET_FRR = 0.05
TARGET_ACCURACY = 0.95

