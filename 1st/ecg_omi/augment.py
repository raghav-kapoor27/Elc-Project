from __future__ import annotations

import numpy as np


class NoiseAugmenter:
    def __init__(self, fs: int = 250, rng: np.random.Generator | None = None) -> None:
        self.fs = fs
        self.rng = rng or np.random.default_rng()

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        y = x.astype(np.float32).copy()
        noise_score = 0.0
        y, s = self.gaussian_noise(y)
        noise_score += s
        y, s = self.motion_artifact(y)
        noise_score += s
        y, s = self.baseline_drift(y)
        noise_score += s
        y, s = self.powerline(y)
        noise_score += s
        y, s = self.dropout(y)
        noise_score += s
        noise_score = float(np.clip(noise_score / 5.0, 0.0, 1.0))
        return y.astype(np.float32), {"noise_score": noise_score, "signal_quality": 1.0 - noise_score}

    def gaussian_noise(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        amp = float(self.rng.uniform(0.0, 0.12))
        return x + self.rng.normal(0.0, amp, size=x.shape), amp / 0.12

    def motion_artifact(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        amp = float(self.rng.uniform(0.0, 0.25))
        n = len(x)
        knots = max(4, n // self.fs)
        low = np.interp(np.arange(n), np.linspace(0, n - 1, knots), self.rng.normal(0, amp, knots))
        return x + low.astype(np.float32), amp / 0.25

    def baseline_drift(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        amp = float(self.rng.uniform(0.0, 0.2))
        freq = float(self.rng.uniform(0.05, 0.4))
        t = np.arange(len(x)) / self.fs
        return x + amp * np.sin(2 * np.pi * freq * t), amp / 0.2

    def powerline(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        amp = float(self.rng.uniform(0.0, 0.05))
        freq = float(self.rng.choice([50.0, 60.0]))
        t = np.arange(len(x)) / self.fs
        return x + amp * np.sin(2 * np.pi * freq * t), amp / 0.05

    def dropout(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        p = float(self.rng.uniform(0.0, 0.08))
        mask = self.rng.random(len(x)) > p
        return x * mask.astype(np.float32), p / 0.08
