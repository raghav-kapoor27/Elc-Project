from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from ecg_omi.data.datasets import ECGWindowDataset
from ecg_omi.training.self_supervised import SelfSupervisedECGModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="icentia11k")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="checkpoints/self_supervised.ckpt")
    args = parser.parse_args()
    dataset = ECGWindowDataset(args.dataset, max_records=args.max_records, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    module = SelfSupervisedECGModule()
    trainer = L.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto")
    trainer.fit(module, loader)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.out)


if __name__ == "__main__":
    main()
