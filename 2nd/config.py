from pathlib import Path

# =====================================
# CHANGE ECG LEADS HERE
# =====================================
SELECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# =====================================

# Point this to the extracted PTB-XL directory containing ptbxl_database.csv.
PTBXL_ROOT = Path("data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

# PTB-XL provides 100 Hz low-resolution and 500 Hz high-resolution records.
SAMPLING_RATE = 100  # choose 100 or 500

ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_CLASSES = ["Normal", "Myocardial Infarction", "ST/T Abnormality"]
SUPERCLASS_TO_TARGET = {"NORM": 0, "MI": 1, "STTC": 2}

# PTB-XL records can contain multiple diagnostic superclasses. This priority
# converts them into one softmax label without using any non-PTB-XL labels.
CLASS_PRIORITY = ["MI", "STTC", "NORM"]

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]

RECORD_SECONDS = 10
SIGNAL_LENGTH_BY_RATE = {100: 1000, 500: 5000}

RANDOM_SEED = 42
BATCH_SIZE = 16
CLASSIFIER_EPOCHS = 30
ANOMALY_EPOCHS = 15
LEARNING_RATE = 1e-4
ANOMALY_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.02
MIXED_PRECISION = True

# Preprocessing
REMOVE_BASELINE_WANDER = True
BANDPASS_FILTER = True
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 3
ZSCORE_EPS = 1e-6

# Optional training-time augmentation. Reconstruction targets stay clean, so
# the reconstruction head behaves like a denoising autoencoder when enabled.
NOISE_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.45
GAUSSIAN_NOISE_STD_RANGE = (0.01, 0.08)
BASELINE_WANDER_AMPLITUDE_RANGE = (0.02, 0.15)
BASELINE_WANDER_FREQ_RANGE = (0.15, 0.45)
LEAD_DROPOUT_PROBABILITY = 0.05

# Model
CNN_FILTERS = [64, 128, 192, 256]
CNN_KERNEL_SIZE = 7
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_KEY_DIM = 64
TRANSFORMER_MLP_DIM = 384
EMBEDDING_DIM = 256
DROPOUT = 0.15

CLASSIFIER_PHASE_LOSS_WEIGHTS = {
    "classification": 1.0,
    "noise": 0.15,
    "reconstruction": 0.0,
}
MULTITASK_PHASE_LOSS_WEIGHTS = {
    "classification": 0.75,
    "noise": 0.15,
    "reconstruction": 0.8,
}

# Outputs
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"
EXPLAINABILITY_DIR = OUTPUT_DIR / "explainability"

CLASSIFIER_WEIGHTS_PATH = CHECKPOINT_DIR / "classifier_best.weights.h5"
FINAL_WEIGHTS_PATH = CHECKPOINT_DIR / "research_model_final.weights.h5"
MAHALANOBIS_STATS_PATH = OUTPUT_DIR / "mahalanobis_stats.npz"


def signal_length(sampling_rate: int = SAMPLING_RATE) -> int:
    if sampling_rate not in SIGNAL_LENGTH_BY_RATE:
        raise ValueError(f"Unsupported sampling rate {sampling_rate}. Use 100 or 500.")
    return SIGNAL_LENGTH_BY_RATE[sampling_rate]


def num_leads() -> int:
    return len(SELECTED_LEADS)


def validate_config() -> None:
    unknown = [lead for lead in SELECTED_LEADS if lead not in ALL_LEADS]
    if unknown:
        raise ValueError(f"Unknown leads in SELECTED_LEADS: {unknown}. Valid leads: {ALL_LEADS}")
    if len(set(SELECTED_LEADS)) != len(SELECTED_LEADS):
        raise ValueError("SELECTED_LEADS contains duplicates.")
    if not SELECTED_LEADS:
        raise ValueError("SELECTED_LEADS must contain at least one ECG lead.")
    signal_length(SAMPLING_RATE)


def make_output_dirs() -> None:
    for path in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, EXPLAINABILITY_DIR]:
        path.mkdir(parents=True, exist_ok=True)
