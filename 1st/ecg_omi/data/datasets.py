from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_omi.augment import NoiseAugmenter
from ecg_omi.config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SAMPLES
from ecg_omi.data.acquisition import DatasetRegistry
from ecg_omi.preprocess import fixed_windows, load_wfdb_lead_i, preprocess_ecg


@dataclass
class ECGSample:
    signal: torch.Tensor
    clean: torch.Tensor
    quality: torch.Tensor
    record: str


class ECGWindowDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        dataset_name: str,
        registry: DatasetRegistry | None = None,
        max_records: int | None = None,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        stride_samples: int | None = None,
        augment: bool = False,
    ) -> None:
        self.registry = registry or DatasetRegistry()
        self.dataset_name = dataset_name
        requested_records = self.registry.discover_records(dataset_name, max_records=max_records)
        self.registry.ensure(dataset_name, records=requested_records)
        self.records = requested_records
        self.window_samples = window_samples
        self.stride_samples = stride_samples
        self.augment = augment
        self.augmenter = NoiseAugmenter(DEFAULT_SAMPLE_RATE)
        self.index = self._build_index()

    def _build_index(self) -> list[tuple[str, int]]:
        index: list[tuple[str, int]] = []
        for record in self.records:
            try:
                x, fs = load_wfdb_lead_i(self.registry.record_path(self.dataset_name, record))
                y = preprocess_ecg(x, fs)
                n_windows = len(fixed_windows(y, self.window_samples, self.stride_samples))
            except Exception:
                continue
            index.extend((record, i) for i in range(n_windows))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record, window_idx = self.index[idx]
        x, fs = load_wfdb_lead_i(self.registry.record_path(self.dataset_name, record))
        clean = preprocess_ecg(x, fs)
        windows = fixed_windows(clean, self.window_samples, self.stride_samples)
        clean_window = windows[window_idx]
        if self.augment:
            noisy, quality = self.augmenter(clean_window)
        else:
            noisy, quality = clean_window, {"noise_score": 0.0, "signal_quality": 1.0}
        return {
            "signal": torch.from_numpy(np.asarray(noisy, dtype=np.float32)).unsqueeze(0),
            "clean": torch.from_numpy(clean_window.astype(np.float32)).unsqueeze(0),
            "signal_quality": torch.tensor([quality["signal_quality"]], dtype=torch.float32),
            "noise_score": torch.tensor([quality["noise_score"]], dtype=torch.float32),
            "record": record,
        }
