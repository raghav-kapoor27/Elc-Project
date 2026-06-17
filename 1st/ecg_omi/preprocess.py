from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt
import scipy.signal as signal
import wfdb

from ecg_omi.config import DEFAULT_SAMPLE_RATE


def load_wfdb_lead_i(record_path: str | Path) -> tuple[np.ndarray, int]:
    record = wfdb.rdrecord(str(record_path))
    fs = int(record.fs)
    names = [name.upper().replace(" ", "") for name in (record.sig_name or [])]
    lead_idx = 0
    for candidate in ("I", "LEADI", "MLI"):
        if candidate in names:
            lead_idx = names.index(candidate)
            break
    x = np.asarray(record.p_signal[:, lead_idx], dtype=np.float32)
    return x, fs


def resample_ecg(x: np.ndarray, src_fs: int, dst_fs: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    if src_fs == dst_fs:
        return x.astype(np.float32)
    n = int(round(len(x) * dst_fs / src_fs))
    return signal.resample_poly(x, dst_fs, src_fs)[:n].astype(np.float32)


def remove_baseline_wander(x: np.ndarray, fs: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    kernel = max(3, int(0.2 * fs) | 1)
    baseline = signal.medfilt(x, kernel_size=kernel)
    kernel = max(3, int(0.6 * fs) | 1)
    baseline = signal.medfilt(baseline, kernel_size=kernel)
    return (x - baseline).astype(np.float32)


def bandpass_filter(
    x: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def wavelet_denoise(x: np.ndarray, wavelet: str = "db6", level: int = 4) -> np.ndarray:
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = sigma * np.sqrt(2 * np.log(max(len(x), 2)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    y = pywt.waverec(coeffs, wavelet, mode="symmetric")
    return y[: len(x)].astype(np.float32)


def normalize_signal(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad
    if scale < eps:
        scale = np.std(x) + eps
    return ((x - median) / scale).astype(np.float32)


def preprocess_ecg(x: np.ndarray, src_fs: int, dst_fs: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    y = resample_ecg(np.nan_to_num(x).astype(np.float32), src_fs, dst_fs)
    y = remove_baseline_wander(y, dst_fs)
    y = bandpass_filter(y, dst_fs)
    y = wavelet_denoise(y)
    return normalize_signal(y)


def fixed_windows(x: np.ndarray, window_samples: int, stride_samples: int | None = None) -> np.ndarray:
    stride = stride_samples or window_samples
    if len(x) < window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: len(x)] = x
        return padded[None, :]
    starts = range(0, len(x) - window_samples + 1, stride)
    return np.stack([x[s : s + window_samples] for s in starts]).astype(np.float32)
