from __future__ import annotations

import argparse

from ecg_omi.data.datasets import ECGWindowDataset
from ecg_omi.export import load_model
from ecg_omi.models.zero_shot import DeepSVDDMahalanobis
from ecg_omi.training.embed import collect_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/zero_shot_detector.npz")
    args = parser.parse_args()
    dataset = ECGWindowDataset("ptb-xl", max_records=args.max_records, augment=False)
    model = load_model(args.checkpoint, device=args.device)
    embeddings = collect_embeddings(model, dataset, batch_size=args.batch_size, device=args.device)
    detector = DeepSVDDMahalanobis.fit(embeddings)
    detector.save(args.out)
    print(f"saved zero-shot detector to {args.out}")


if __name__ == "__main__":
    main()
