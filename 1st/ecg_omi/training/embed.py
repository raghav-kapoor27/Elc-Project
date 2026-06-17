from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_omi.data.datasets import ECGWindowDataset
from ecg_omi.models.foundation import ECGFoundationModel


def collect_embeddings(
    model: ECGFoundationModel,
    dataset: ECGWindowDataset,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding ECG windows"):
            x = batch["signal"].to(device)
            chunks.append(model.embed(x).cpu().numpy())
    if not chunks:
        raise RuntimeError("No embeddings collected. Check dataset cache and record parsing.")
    return np.concatenate(chunks, axis=0)
