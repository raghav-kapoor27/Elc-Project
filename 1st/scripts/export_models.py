from __future__ import annotations

import argparse

from ecg_omi.export import export_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detector", default=None, help="Detector is already NumPy based; pass for workflow parity.")
    parser.add_argument("--out-dir", default="exports")
    args = parser.parse_args()
    export_all(args.checkpoint, args.out_dir)
    print(f"exports written to {args.out_dir}")


if __name__ == "__main__":
    main()
