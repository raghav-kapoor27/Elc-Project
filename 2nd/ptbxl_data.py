import ast
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, medfilt
from tensorflow.keras.utils import Sequence

import config


NOISE_COLUMNS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]


def load_ptbxl_metadata(
    root: Path = config.PTBXL_ROOT,
    sampling_rate: int = config.SAMPLING_RATE,
) -> pd.DataFrame:
    root = Path(root)
    database_path = root / "ptbxl_database.csv"
    scp_path = root / "scp_statements.csv"

    if not database_path.exists() or not scp_path.exists():
        raise FileNotFoundError(
            "PTB-XL files not found. Set config.PTBXL_ROOT to the extracted PTB-XL directory "
            "containing ptbxl_database.csv and scp_statements.csv."
        )

    metadata = pd.read_csv(database_path, index_col="ecg_id")
    metadata["scp_codes"] = metadata["scp_codes"].apply(ast.literal_eval)

    scp_statements = pd.read_csv(scp_path, index_col=0)
    diagnostic_scp = scp_statements[scp_statements["diagnostic"] == 1]

    metadata["diagnostic_superclasses"] = metadata["scp_codes"].apply(
        lambda codes: _aggregate_diagnostic_superclasses(codes, diagnostic_scp)
    )
    metadata["target_id"] = metadata["diagnostic_superclasses"].apply(_assign_target_id)
    metadata = metadata.dropna(subset=["target_id"]).copy()
    metadata["target_id"] = metadata["target_id"].astype(int)

    metadata[["signal_quality_score", "noise_score"]] = metadata.apply(
        _derive_noise_targets, axis=1, result_type="expand"
    )
    metadata["filename"] = metadata["filename_lr"] if sampling_rate == 100 else metadata["filename_hr"]
    metadata["strat_fold"] = metadata["strat_fold"].astype(int)
    return metadata


def split_metadata(metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = metadata[metadata["strat_fold"].isin(config.TRAIN_FOLDS)].copy()
    val = metadata[metadata["strat_fold"].isin(config.VAL_FOLDS)].copy()
    test = metadata[metadata["strat_fold"].isin(config.TEST_FOLDS)].copy()
    return train, val, test


def load_record_from_metadata(
    row: pd.Series,
    root: Path = config.PTBXL_ROOT,
    sampling_rate: int = config.SAMPLING_RATE,
    selected_leads: Optional[Iterable[str]] = None,
) -> np.ndarray:
    record_path = Path(root) / row["filename"]
    return load_record(record_path, sampling_rate=sampling_rate, selected_leads=selected_leads)


def load_record(
    record_path: Path,
    sampling_rate: int = config.SAMPLING_RATE,
    selected_leads: Optional[Iterable[str]] = None,
) -> np.ndarray:
    selected = list(selected_leads or config.SELECTED_LEADS)
    signal, fields = wfdb.rdsamp(str(record_path))
    signal = signal.astype(np.float32)

    lead_names = fields.get("sig_name", config.ALL_LEADS)
    missing = [lead for lead in selected if lead not in lead_names]
    if missing:
        raise ValueError(f"Record {record_path} is missing requested leads: {missing}")

    lead_indices = [lead_names.index(lead) for lead in selected]
    signal = signal[:, lead_indices]
    return _fix_length(signal, config.signal_length(sampling_rate))


def preprocess_ecg(signal: np.ndarray, sampling_rate: int = config.SAMPLING_RATE) -> np.ndarray:
    x = signal.astype(np.float32)
    if config.REMOVE_BASELINE_WANDER:
        x = remove_baseline_wander(x, sampling_rate)
    if config.BANDPASS_FILTER:
        x = bandpass_filter(x, sampling_rate)
    return zscore_normalize(x)


def load_and_preprocess_record(
    record_path: Path,
    sampling_rate: int = config.SAMPLING_RATE,
    selected_leads: Optional[Iterable[str]] = None,
) -> np.ndarray:
    raw = load_record(record_path, sampling_rate=sampling_rate, selected_leads=selected_leads)
    return preprocess_ecg(raw, sampling_rate=sampling_rate)


def remove_baseline_wander(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    kernel_200ms = _odd_kernel(0.2 * sampling_rate)
    kernel_600ms = _odd_kernel(0.6 * sampling_rate)
    filtered = np.empty_like(signal, dtype=np.float32)

    for lead_idx in range(signal.shape[1]):
        baseline = medfilt(signal[:, lead_idx], kernel_size=kernel_200ms)
        baseline = medfilt(baseline, kernel_size=kernel_600ms)
        filtered[:, lead_idx] = signal[:, lead_idx] - baseline

    return filtered


def bandpass_filter(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    nyquist = sampling_rate * 0.5
    low = max(config.BANDPASS_LOW_HZ / nyquist, 1e-5)
    high = min(config.BANDPASS_HIGH_HZ / nyquist, 0.999)
    if low >= high:
        return signal.astype(np.float32)

    b, a = butter(config.BANDPASS_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return ((signal - mean) / (std + config.ZSCORE_EPS)).astype(np.float32)


def augment_ecg(
    clean_signal: np.ndarray,
    sampling_rate: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    if rng.random() > config.AUGMENTATION_PROBABILITY:
        return clean_signal.astype(np.float32), 0.0

    x = clean_signal.copy()
    severity = 0.0

    noise_std = rng.uniform(*config.GAUSSIAN_NOISE_STD_RANGE)
    x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    severity = max(severity, min(1.0, noise_std / max(config.GAUSSIAN_NOISE_STD_RANGE)))

    amplitude = rng.uniform(*config.BASELINE_WANDER_AMPLITUDE_RANGE)
    frequency = rng.uniform(*config.BASELINE_WANDER_FREQ_RANGE)
    t = np.arange(x.shape[0], dtype=np.float32) / float(sampling_rate)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, x.shape[1]))
    wander = amplitude * np.sin(2.0 * np.pi * frequency * t[:, None] + phase)
    x += wander.astype(np.float32)
    severity = max(severity, min(1.0, amplitude / max(config.BASELINE_WANDER_AMPLITUDE_RANGE)))

    if x.shape[1] > 1 and rng.random() < config.LEAD_DROPOUT_PROBABILITY:
        lead_idx = int(rng.integers(0, x.shape[1]))
        x[:, lead_idx] = 0.0
        severity = max(severity, 0.9)

    return x.astype(np.float32), float(severity)


class PTBXLSequence(Sequence):
    def __init__(
        self,
        metadata: pd.DataFrame,
        root: Path = config.PTBXL_ROOT,
        sampling_rate: int = config.SAMPLING_RATE,
        batch_size: int = config.BATCH_SIZE,
        augment: bool = False,
        shuffle: bool = False,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.metadata = metadata.reset_index(drop=False)
        self.root = Path(root)
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.metadata))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.metadata) / float(self.batch_size)))

    def __getitem__(self, batch_index: int):
        batch_indices = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        batch_size = len(batch_indices)
        x = np.zeros(
            (batch_size, config.signal_length(self.sampling_rate), len(config.SELECTED_LEADS)),
            dtype=np.float32,
        )
        reconstruction_target = np.zeros_like(x, dtype=np.float32)
        y_class = np.zeros((batch_size, len(config.TARGET_CLASSES)), dtype=np.float32)
        y_noise = np.zeros((batch_size, 2), dtype=np.float32)

        for out_idx, meta_idx in enumerate(batch_indices):
            row = self.metadata.iloc[meta_idx]
            raw = load_record_from_metadata(
                row,
                root=self.root,
                sampling_rate=self.sampling_rate,
                selected_leads=config.SELECTED_LEADS,
            )
            clean = preprocess_ecg(raw, sampling_rate=self.sampling_rate)
            model_input = clean

            noise_score = float(row["noise_score"])
            if self.augment and config.NOISE_AUGMENTATION:
                model_input, augmented_noise = augment_ecg(clean, self.sampling_rate, self.rng)
                noise_score = max(noise_score, augmented_noise)

            x[out_idx] = model_input
            reconstruction_target[out_idx] = clean
            y_class[out_idx, int(row["target_id"])] = 1.0
            y_noise[out_idx] = np.array([1.0 - noise_score, noise_score], dtype=np.float32)

        return x, {
            "classification": y_class,
            "noise": y_noise,
            "reconstruction": reconstruction_target,
        }

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def _aggregate_diagnostic_superclasses(scp_codes: dict, diagnostic_scp: pd.DataFrame) -> list:
    classes = set()
    for code in scp_codes.keys():
        if code in diagnostic_scp.index:
            diagnostic_class = diagnostic_scp.loc[code, "diagnostic_class"]
            if pd.notna(diagnostic_class):
                classes.add(str(diagnostic_class))
    return sorted(classes)


def _assign_target_id(diagnostic_superclasses: list) -> Optional[int]:
    for superclass in config.CLASS_PRIORITY:
        if superclass in diagnostic_superclasses:
            return config.SUPERCLASS_TO_TARGET[superclass]
    return None


def _derive_noise_targets(row: pd.Series) -> Tuple[float, float]:
    available_columns = [column for column in NOISE_COLUMNS if column in row.index]
    if not available_columns:
        return 1.0, 0.0

    noisy_count = sum(_has_noise_annotation(row[column]) for column in available_columns)
    noise_score = float(noisy_count) / float(len(available_columns))
    signal_quality = 1.0 - noise_score
    return signal_quality, noise_score


def _has_noise_annotation(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "false", "0", "0.0"}


def _fix_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[0] == target_length:
        return signal.astype(np.float32)
    if signal.shape[0] > target_length:
        return signal[:target_length].astype(np.float32)

    pad_width = target_length - signal.shape[0]
    return np.pad(signal, ((0, pad_width), (0, 0)), mode="edge").astype(np.float32)


def _odd_kernel(value: float) -> int:
    kernel = max(3, int(round(value)))
    return kernel if kernel % 2 == 1 else kernel + 1
