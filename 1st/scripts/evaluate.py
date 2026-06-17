from __future__ import annotations

import argparse
import json

from ecg_omi.evaluation import evaluate_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--dataset", default="staffiii")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_detector(args.checkpoint, args.detector, args.dataset, args.max_records, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
