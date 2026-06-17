from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ecg_omi.data.datasets import ECGWindowDataset
from ecg_omi.export import load_model
from ecg_omi.models.zero_shot import DeepSVDDMahalanobis
from ecg_omi.training.embed import collect_embeddings


def evaluate_detector(
    checkpoint: str | Path,
    detector_path: str | Path,
    dataset_name: str,
    max_records: int | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    dataset = ECGWindowDataset(dataset_name, max_records=max_records, augment=False)
    model = load_model(checkpoint, device=device)
    detector = DeepSVDDMahalanobis.load(detector_path)
    embeddings = collect_embeddings(model, dataset, device=device)
    scores = detector.score(embeddings)
    result = {
        "mean_anomaly_score": float(np.mean(scores["anomaly_score"])),
        "mean_omi_suspicion_score": float(np.mean(scores["omi_suspicion_score"])),
        "n_windows": float(len(embeddings)),
    }
    labels = getattr(dataset, "labels", None)
    if labels is not None and len(labels) == len(embeddings):
        result["roc_auc"] = float(roc_auc_score(labels, scores["omi_suspicion_score"]))
    return result
