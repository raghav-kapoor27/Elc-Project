from __future__ import annotations

import torch
import torch.nn.functional as F
import lightning as L

from ecg_omi.config import ModelConfig
from ecg_omi.models.foundation import ECGFoundationModel


def random_mask(x: torch.Tensor, ratio: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(x) < ratio
    masked = x.masked_fill(mask, 0.0)
    return masked, mask.float()


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = torch.cat([F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)], dim=0)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-1e9)
    n = z1.shape[0]
    labels = torch.arange(2 * n, device=z.device)
    labels = (labels + n) % (2 * n)
    return F.cross_entropy(logits, labels)


class SelfSupervisedECGModule(L.LightningModule):
    def __init__(self, config: ModelConfig | None = None, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ECGFoundationModel(config)
        self.lr = lr

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        clean = batch["clean"]
        noisy = batch["signal"]
        masked, mask = random_mask(clean)
        out_masked = self.model(masked)
        out_noisy = self.model(noisy)
        out_clean = self.model(clean)
        masked_loss = (F.l1_loss(out_masked["reconstruction"], clean, reduction="none") * mask).mean()
        denoise_loss = F.l1_loss(out_noisy["reconstruction"], clean)
        contrastive_loss = nt_xent(
            out_noisy["projection"],
            out_clean["projection"],
            self.model.config.contrastive_temperature,
        )
        quality_target = torch.cat([batch["signal_quality"], batch["noise_score"]], dim=1)
        quality_loss = F.mse_loss(torch.cat([out_noisy["signal_quality"], out_noisy["noise_score"]], dim=1), quality_target)
        loss = masked_loss + denoise_loss + contrastive_loss + quality_loss
        self.log_dict(
            {
                "train/loss": loss,
                "train/masked_recon": masked_loss,
                "train/denoise": denoise_loss,
                "train/contrastive": contrastive_loss,
                "train/quality": quality_loss,
            },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
