from __future__ import annotations

import argparse

from ecg_omi.data.acquisition import DatasetRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "ptb-xl", "icentia11k", "staffiii", "mitdb"])
    parser.add_argument("--cache-dir", default=".cache/ecg_data")
    args = parser.parse_args()
    registry = DatasetRegistry(args.cache_dir)
    names = list(registry.SPECS) if args.dataset == "all" else [args.dataset]
    for name in names:
        path = registry.ensure(name)
        records = registry.records(name)
        print(f"{name}: cached at {path} with {len(records)} discovered records")


if __name__ == "__main__":
    main()
