from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ecg_omi.config import ModelConfig


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGFoundationModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        channels = (1, *self.config.cnn_channels)
        self.cnn = nn.Sequential(
            *[CNNBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.proj = nn.Conv1d(self.config.cnn_channels[-1], self.config.embed_dim, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.embed_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.config.transformer_layers)
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose1d(self.config.embed_dim, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.GELU(),
            nn.Linear(self.config.embed_dim, 128),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(self.config.embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        self.pathology_head = nn.Linear(self.config.embed_dim, 3)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.cnn(x)).transpose(1, 2)
        return self.transformer(z)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x)
        return tokens.mean(dim=1)

    def reconstruct(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.reconstruction_head(tokens.transpose(1, 2))
        return F.interpolate(y, size=target_len, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.encode_tokens(x)
        embedding = tokens.mean(dim=1)
        quality = self.quality_head(embedding)
        return {
            "tokens": tokens,
            "embedding": embedding,
            "projection": F.normalize(self.projection_head(embedding), dim=-1),
            "reconstruction": self.reconstruct(tokens, x.shape[-1]),
            "signal_quality": quality[:, 0:1],
            "noise_score": quality[:, 1:2],
            "pathology_logits": self.pathology_head(embedding),
        }
