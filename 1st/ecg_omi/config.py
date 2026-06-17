from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "ecg_data"
DEFAULT_SAMPLE_RATE = 250
DEFAULT_WINDOW_SECONDS = 10
DEFAULT_WINDOW_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_SECONDS


@dataclass(frozen=True)
class ModelConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    window_samples: int = DEFAULT_WINDOW_SAMPLES
    embed_dim: int = 256
    cnn_channels: tuple[int, int, int, int] = (64, 128, 192, 256)
    transformer_layers: int = 6
    attention_heads: int = 8
    dropout: float = 0.1
    contrastive_temperature: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    max_epochs: int = 10
