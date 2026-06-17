from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DeepSVDDMahalanobis:
    center: np.ndarray
    covariance_inv: np.ndarray
    svdd_radius: float
    score_scale: float

    @classmethod
    def fit(cls, embeddings: np.ndarray, quantile: float = 0.95) -> "DeepSVDDMahalanobis":
        embeddings = np.asarray(embeddings, dtype=np.float64)
        center = embeddings.mean(axis=0)
        diff = embeddings - center
        covariance = np.cov(diff, rowvar=False) + np.eye(diff.shape[1]) * 1e-4
        covariance_inv = np.linalg.pinv(covariance)
        distances = np.sqrt(np.sum((diff @ covariance_inv) * diff, axis=1))
        radius = float(np.quantile(distances, quantile))
        score_scale = float(np.std(distances) + 1e-6)
        return cls(center, covariance_inv, radius, score_scale)

    def score(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        diff = embeddings - self.center
        mahal = np.sqrt(np.sum((diff @ self.covariance_inv) * diff, axis=1))
        svdd = np.maximum(0.0, mahal - self.svdd_radius)
        anomaly = mahal + svdd
        suspicion = 1.0 / (1.0 + np.exp(-(anomaly - self.svdd_radius) / self.score_scale))
        confidence = np.clip(np.abs(suspicion - 0.5) * 2.0, 0.0, 1.0)
        return {
            "mahalanobis": mahal.astype(np.float32),
            "deep_svdd": svdd.astype(np.float32),
            "anomaly_score": anomaly.astype(np.float32),
            "omi_suspicion_score": suspicion.astype(np.float32),
            "confidence": confidence.astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            center=self.center,
            covariance_inv=self.covariance_inv,
            svdd_radius=self.svdd_radius,
            score_scale=self.score_scale,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeepSVDDMahalanobis":
        data = np.load(path)
        return cls(
            center=data["center"],
            covariance_inv=data["covariance_inv"],
            svdd_radius=float(data["svdd_radius"]),
            score_scale=float(data["score_scale"]),
        )
