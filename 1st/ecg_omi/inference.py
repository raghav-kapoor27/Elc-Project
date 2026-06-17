from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ecg_omi.config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW_SAMPLES
from ecg_omi.models.foundation import ECGFoundationModel
from ecg_omi.models.zero_shot import DeepSVDDMahalanobis
from ecg_omi.preprocess import preprocess_ecg


class RealTimeOMIInference:
    def __init__(
        self,
        checkpoint: str | Path,
        detector_path: str | Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        device: str = "cpu",
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.buffer: deque[float] = deque(maxlen=window_samples)
        self.device = torch.device(device)
        self.model = ECGFoundationModel().to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        state_dict = state.get("state_dict", state)
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.detector = DeepSVDDMahalanobis.load(detector_path)

    def push(self, sample: float) -> dict[str, float] | None:
        self.buffer.append(float(sample))
        if len(self.buffer) < self.window_samples:
            return None
        return self.predict(np.asarray(self.buffer, dtype=np.float32), self.sample_rate)

    def predict(self, signal: np.ndarray, src_fs: int) -> dict[str, float]:
        x = preprocess_ecg(signal, src_fs, self.sample_rate)[-self.window_samples :]
        if len(x) < self.window_samples:
            pad = np.zeros(self.window_samples, dtype=np.float32)
            pad[-len(x) :] = x
            x = pad
        tensor = torch.from_numpy(x).float().view(1, 1, -1).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        scores = self.detector.score(out["embedding"].cpu().numpy())
        return {
            "signal_quality": float(out["signal_quality"].item()),
            "noise_score": float(out["noise_score"].item()),
            "anomaly_score": float(scores["anomaly_score"][0]),
            "omi_suspicion_score": float(scores["omi_suspicion_score"][0]),
            "confidence": float(scores["confidence"][0]),
        }
