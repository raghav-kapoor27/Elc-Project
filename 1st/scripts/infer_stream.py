from __future__ import annotations

import argparse
import json

from ecg_omi.inference import RealTimeOMIInference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--input", required=True, help="CSV/text file with one ECG sample per line")
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    engine = RealTimeOMIInference(args.checkpoint, args.detector, sample_rate=args.sample_rate, device=args.device)
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sample = float(line.split(",")[0])
            result = engine.push(sample)
            if result is not None:
                print(json.dumps(result))


if __name__ == "__main__":
    main()
