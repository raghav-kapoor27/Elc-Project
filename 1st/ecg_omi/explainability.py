from __future__ import annotations

import torch

from ecg_omi.models.foundation import ECGFoundationModel


def saliency_map(model: ECGFoundationModel, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    out = model(x)
    score = out["embedding"].norm(dim=-1).sum()
    score.backward()
    return x.grad.detach().abs()


def attention_proxy(model: ECGFoundationModel, x: torch.Tensor) -> torch.Tensor:
    """Token-importance proxy for transformer attention maps.

    PyTorch's stock TransformerEncoder does not expose per-head weights without
    replacing layers. For research visualization, token energy is a stable proxy
    that can be upsampled to the ECG timeline.
    """

    model.eval()
    with torch.no_grad():
        tokens = model.encode_tokens(x)
        importance = tokens.norm(dim=-1)
        importance = importance / (importance.amax(dim=-1, keepdim=True) + 1e-6)
        return torch.nn.functional.interpolate(
            importance.unsqueeze(1),
            size=x.shape[-1],
            mode="linear",
            align_corners=False,
        )


def highlighted_regions(score: torch.Tensor, threshold: float = 0.75) -> list[tuple[int, int]]:
    binary = (score.squeeze().detach().cpu() >= threshold).numpy()
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, active in enumerate(binary):
        if active and start is None:
            start = i
        elif not active and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(binary)))
    return regions
