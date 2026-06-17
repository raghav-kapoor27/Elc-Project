from __future__ import annotations

import argparse
import ast
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg_omi.data.datasets import ECGWindowDataset
from ecg_omi.export import load_model
from ecg_omi.training.pathology import PathologyAdaptationModule


class PTBXLPathologyDataset(ECGWindowDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_by_record = self._load_labels()

    def _load_labels(self) -> dict[str, int]:
        metadata = self.registry.path("ptb-xl") / "ptbxl_database.csv"
        if not metadata.exists():
            return {}
        df = pd.read_csv(metadata)
        labels: dict[str, int] = {}
        for _, row in df.iterrows():
            codes = ast.literal_eval(row.get("scp_codes", "{}"))
            code_names = {str(k).upper() for k, v in codes.items() if float(v) > 0}
            label = 0
            if any("MI" in code for code in code_names):
                label = 1
            elif code_names.intersection({"STTC", "NST_", "ISC_", "ISCAL", "ISCAS", "ISCIN", "ISCIL"}):
                label = 2
            for column in ("filename_lr", "filename_hr"):
                value = row.get(column)
                if isinstance(value, str):
                    labels[value.replace("\\", "/")] = label
        return labels

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        record = str(item["record"]).replace("\\", "/")
        label = self.labels_by_record.get(record, 0)
        item["label"] = torch.tensor(label, dtype=torch.long)
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="checkpoints/pathology.ckpt")
    args = parser.parse_args()
    dataset = PTBXLPathologyDataset("ptb-xl", max_records=args.max_records, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = load_model(args.pretrained) if args.pretrained else None
    module = PathologyAdaptationModule(model=model)
    trainer = L.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto")
    trainer.fit(module, loader)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.out)


if __name__ == "__main__":
    main()
