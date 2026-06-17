from __future__ import annotations

import torch
import torch.nn.functional as F
import lightning as L

from ecg_omi.models.foundation import ECGFoundationModel


class PathologyAdaptationModule(L.LightningModule):
    """Adapts embeddings to normal, MI, and ST/T abnormality targets.

    Batch labels are expected as:
    0 normal ECG pattern, 1 myocardial infarction pattern, 2 ST/T abnormality.
    """

    def __init__(self, model: ECGFoundationModel | None = None, lr: float = 5e-5) -> None:
        super().__init__()
        self.model = model or ECGFoundationModel()
        self.lr = lr

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["signal"])
        labels = batch.get("label")
        if labels is None:
            labels = torch.zeros(out["pathology_logits"].shape[0], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(out["pathology_logits"], labels.long())
        self.log("train/pathology_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
