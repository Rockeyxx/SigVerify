"""Project configuration: paths, model hyperparameters, thresholds, and runtime limits."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
SPATIAL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "spatial_cnn.pth")
TEMPORAL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "temporal_lstm.pth")
# Patch version counts commits since this line last changed (src/app_version.py).
APP_VERSION = "1.0"
# Release tag and SHA-256 digests of the shipped weights; update all three together.
MODEL_VERSION = "v0.10.0"
SPATIAL_WEIGHTS_SHA256 = "d4c52c9401d5b1633ddedc266cb7c7d210a79fc536f501da4c4b7a324b1480bf"
TEMPORAL_WEIGHTS_SHA256 = "231f99568ddd4684e9cf61240f960d52b11fb2340c47d662a923eb296cf6b0b1"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Datasets (gitignored; local download required)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CEDAR_DIR = os.path.join(DATA_DIR, "cedar")
BHSIG260_DIR = os.path.join(DATA_DIR, "bhsig260")
UTSIG_DIR = os.path.join(DATA_DIR, "utsig")
SVC2004_TASK1_DIR = os.path.join(DATA_DIR, "svc2004_task1")   # 4 columns, no pressure
SVC2004_TASK2_DIR = os.path.join(DATA_DIR, "svc2004_task2")   # 7 columns, with pressure
SVC2004_DIR = SVC2004_TASK2_DIR
MOBISIG_DIR = os.path.join(DATA_DIR, "MOBISIG")
SIGCOMP2011_DIR = os.path.join(DATA_DIR, "sigcomp2011")
SIGCOMP2011_OFFLINE_DIR = os.path.join(DATA_DIR, "sigcomp2011_offline")
SIGCOMP2009_DIR = os.path.join(DATA_DIR, "sigcomp2009_dl", "extracted")
# DeepSignDB and BiosecurID-SONOF are restricted-license datasets (evaluation only; weights do not ship).
DEEPSIGNDB_DIR = os.path.join(DATA_DIR, "deepsigndb")
BIOSECURID_SONOF_DIR = os.path.join(DATA_DIR, "biosecurid_sonof")
INHOUSE_DIR = os.path.join(DATA_DIR, "inhouse")
INHOUSE_TEST_FRACTION = 0.5
INHOUSE_EXCLUDED = ("P06", "P16", "P18", "P19")
# Writers whose genuines were recorded via Windows Ink and forgeries via Wintab; scale forgeries to match.
INHOUSE_NO_DRIVER = ("P10", "P11", "P12", "P13")
INHOUSE_NO_DRIVER_PRESSURE_SCALE = 0.4
INHOUSE_SPLIT_PATH = os.path.join(PROJECT_ROOT, "config", "inhouse_split.json")

# Training corpora for the shipped checkpoints.
SPATIAL_TRAIN_DIRS = (CEDAR_DIR, BHSIG260_DIR, UTSIG_DIR)
TEMPORAL_TRAIN_DIRS = (SVC2004_TASK2_DIR, SIGCOMP2011_DIR)

# DataLoader
NUM_WORKERS = 4                         # Set to 0 if worker processes hang on Windows
PIN_MEMORY = False

# Spatial branch (Siamese CNN)
def spatial_input_size_from_env(value, default):
    """Parse WxH override from SIGVER_SPATIAL_INPUT_SIZE."""
    if value is None or value == "":
        return default
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"SIGVER_SPATIAL_INPUT_SIZE must be WxH in pixels, e.g. 220x155; got {value!r}")
    return (int(parts[0]), int(parts[1]))


SPATIAL_INPUT_SIZE = spatial_input_size_from_env(
    os.environ.get("SIGVER_SPATIAL_INPUT_SIZE"), (220, 155))   # cv2 (width, height) -> [1, 1, 155, 220]
SPATIAL_EMBEDDING_DIM = 512
SPATIAL_SQUARE_PAD = False              # Resize directly onto SPATIAL_INPUT_SIZE without square padding
MIN_BBOX_PIXELS = 64
MIN_LAPLACIAN_VARIANCE = 50.0

# Trajectory rasterizer (src/data/renderer.py)
RENDER_MARGIN_RATIO = 0.12
RENDER_BASE_THICKNESS = 3.0             # Stroke width in px at full pressure
RENDER_MIN_THICKNESS = 1                # Minimum cv2.line thickness
RENDER_PEN_UP_PRESSURE = 1e-4           # Pressure at or below this marks a pen lift
RENDER_PEN_UP_DT_RATIO = 3.0            # Gap >= ratio * median_dt also marks a pen lift

# Temporal ensemble (research only; the app ships and runs a single temporal checkpoint)
TEMPORAL_ENSEMBLE_SEEDS = (42, 43, 44, 45, 46, 47)
TEMPORAL_ENSEMBLE_TEMPLATE = "temporal_dt_s{seed}.pth"

# Temporal input gate and capture settings
MIN_TRAJECTORY_POINTS = 10
MAX_NORMALIZED_XY_MEAN = 0.5
NORMALIZED_XY_STD_RANGE = (0.2, 5.0)
TRAJECTORY_DT_MEDIAN_TOLERANCE = 0.10   # Allowed drift around one 10 ms grid step
WINDOWS_PEN_API = "wintab"              # "wintab" (tablet driver) or "ink" (Windows Ink)
CAPTURE_NO_DRIVER = "no driver"
CAMPAIGN_GENUINE_PER_MODALITY = 12
CAMPAIGN_PRACTICE_SIGNINGS = 3
CAMPAIGN_BREAK_AFTER = 6
CAMPAIGN_FORGERY_ATTEMPTS = 2
SHEET_GRID = (4, 2)
SHEET_DPI = 300
SHEET_MARGIN = 0.05
# Ink crop thresholds for scanned paper boxes (tools/capture_session.py).
INK_BACKGROUND_KERNEL = 51
INK_MIN_CONTRAST = 18
INK_MIN_SATURATION = 10
INK_HUE_RANGE = (95, 165)               # Blue ballpoint in OpenCV HSV (0-179)
INK_CLUSTER_GAP = 61
INK_MIN_PIXELS = 200
INK_CROP_PAD = 0.08
INK_LINE_MIN_FRACTION = 0.5
INK_LINE_MAX_BLUE = 0.25

# Temporal branch (Siamese LSTM)
TEMPORAL_CHANNELS = "dt"
TEMPORAL_CHANNEL_WIDTHS = {"xyp": 3, "dt": 4, "tilt": 5,
                           # Research-only modes: dt + tilt with constant fill when tilt is absent.
                           "dt_tilt0": 6, "dt_tilt1": 6, "dt_tilt_flag": 7}
TILT_CHANNEL_MODES = {"dt_tilt0": 0.0, "dt_tilt1": 1.0, "dt_tilt_flag": 0.0}
TEMPORAL_TILT_HIDE_PROB = 0.5
TEMPORAL_INPUT_FEATURES = TEMPORAL_CHANNEL_WIDTHS[TEMPORAL_CHANNELS]
TEMPORAL_EMBEDDING_DIM = 128
TEMPORAL_HIDDEN_SIZE = 256
TEMPORAL_NUM_LAYERS = 1
TEMPORAL_CONV_FILTERS = 0               # 0 disables the optional pre-LSTM Conv1d stage
TEMPORAL_CONV_KERNEL = 5
PRESSURE_DROPOUT_RATE = 0.0
PRESSURE_IMPUTATION_VALUE = 1.0

# Pair sampling
PAIRS_GENUINE_PER_WRITER = 276          # C(24, 2) distinct genuine pairs per writer
PAIRS_IMPOSTOR_PER_WRITER = 276
RANDOM_IMPOSTOR_RATIO = 0.5             # Training mix; evaluation always uses skilled forgeries

# Loss and training
CONTRASTIVE_MARGIN = 1.0
HARD_NEGATIVE_RATIO = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

SPATIAL_HEAD_DROPOUT = 0.0
SPATIAL_TRUNK = "wide"                  # "compact" or "wide" (see src/models/siamese.py)
SPATIAL_CONV_DROPOUT = 0.0
SPATIAL_FC_HIDDEN = 0
SPATIAL_GRAD_CHECKPOINT = 4             # Checkpoint segments to stay within 4 GB VRAM

SEED = 42
SPATIAL_SPLIT_RATIOS = (0.7, 0.15, 0.15)
TEMPORAL_SPLIT_RATIOS = (0.5, 0.25, 0.25)
TEMPORAL_PAIRS_GENUINE_PER_WRITER = 180
TEMPORAL_PAIRS_IMPOSTOR_PER_WRITER = 180

# Enrolment and retention
NUM_ENROLLMENT_REFERENCES = 5
MAX_REFERENCES = 8
TEMPLATE_RETENTION_DAYS = 730           # Reference set expiry in days (ISO/IEC 24745)

# Template store (src/database, src/gui/storage.py)
MIN_PASSWORD_LENGTH = 15                # NIST SP 800-63B single-factor minimum
COMMON_PASSWORDS_FILE = os.path.join(PROJECT_ROOT, "src", "database", "common_passwords.txt")
GENERATED_PASSWORD_LENGTH = 20
REPLAY_WINDOW_DAYS = 365
MAX_VAULT_BYTES = 1 << 30               # Rejects oversized backup files before reading into memory

ENROLLMENT_TRIALS = 10

# Augmentation
TEMPORAL_AUGMENT = False
SPATIAL_AUGMENT = True
# Training parses SVC2004/DeepSignDB timestamps as float32 to preserve the quantization noise
# the shipped dt model expects; evaluation always reads exact float64 timestamps.
TEMPORAL_TRAIN_FLOAT32_PARSE = True

SPATIAL_AUG_MAX_ROTATION_DEG = 6.0
SPATIAL_AUG_MAX_SHEAR = 0.08
SPATIAL_AUG_MAX_SCALE = 0.08
SPATIAL_AUG_MAX_TRANSLATE = 0.04
SPATIAL_AUG_STROKE_PROB = 0.0
SPATIAL_AUG_BLUR_SIGMA = 0.0
SPATIAL_AUG_INK_GAIN_MIN = 1.0
SPATIAL_AUG_NOISE_STD = 0.0
TEMPORAL_AUG_MAX_ROTATION_DEG = 6.0
TEMPORAL_AUG_JITTER = 0.02
TEMPORAL_AUG_MAX_SPEED_CHANGE = 0.15
TEMPORAL_DEVICE_AUG_RATE_HZ = (200.0, 300.0)
TEMPORAL_DEVICE_AUG_PRESSURE_GAMMA = (0.67, 1.5)
TEMPORAL_DEVICE_AUG_PRESSURE_GAIN = (0.85, 1.0)
TEMPORAL_DEVICE_AUG_PRESSURE_LEVELS = 1024

# Target sample rate for online trajectories. parsers.MAX_GAP_MS is a 100 ms clip and independent.
TEMPORAL_SAMPLE_RATE_HZ = 100.0
TEMPORAL_SAMPLE_PERIOD_MS = 1000.0 / TEMPORAL_SAMPLE_RATE_HZ

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

# Fusion and thresholds: S_final = alpha * S_spatial + (1 - alpha) * S_temporal
ALPHA_DEFAULT = 0.4
ALPHA_GRID_STEP = 0.05
NFI_FUSION_WRITERS = ("nfi_002", "nfi_008", "nfi_016", "nfi_018", "nfi_024", "nfi_035",
                      "nfi_044", "nfi_063", "nfi_070", "nfi_071", "nfi_077", "nfi_085",
                      "nfi_086", "nfi_089")

VERIFICATION_SIGNINGS = 1
VERIFICATION_STEP_UP = False
STEP_UP_REJECT = 0.4229
STEP_UP_ACCEPT = 0.7514
STEP_UP_TWO = 0.7470

SPATIAL_DECISION_THRESHOLD = 0.8387
TEMPORAL_DECISION_THRESHOLD = 0.7900
DECISION_THRESHOLD = 0.7807
SONOF_FUSED_THRESHOLD = 0.7564

# Runtime constraints
LATENCY_SLA_SECONDS = 3.0
INFERENCE_RAM_CEILING_GB = 4.0
INFERENCE_NUM_THREADS = 4               # Caps Torch and OpenCV threads to limit CPU contention

# Evaluation targets (primary tier)
TARGET_FAR = 0.05
TARGET_FRR = 0.05
TARGET_ACCURACY = 0.95




