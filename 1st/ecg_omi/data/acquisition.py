from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import wfdb

from ecg_omi.config import DEFAULT_CACHE_DIR


@dataclass(frozen=True)
class ECGDatasetSpec:
    name: str
    source: str
    physionet_slug: str | None = None
    version: str | None = None


class DatasetRegistry:
    """Lazy dataset discovery and acquisition.

    Files are cached under `.cache/ecg_data/<name>`. Existing directories are
    reused. WFDB is the primary acquisition path for the requested PhysioNet
    datasets.
    """

    SPECS: dict[str, ECGDatasetSpec] = {
        "ptb-xl": ECGDatasetSpec("ptb-xl", "physionet", "ptb-xl", "1.0.3"),
        "icentia11k": ECGDatasetSpec(
            "icentia11k",
            "physionet",
            "icentia11k-continuous-ecg",
            "1.0",
        ),
        "staffiii": ECGDatasetSpec("staffiii", "physionet", "staffiii", "1.0.0"),
        "mitdb": ECGDatasetSpec("mitdb", "physionet", "mitdb", "1.0.0"),
    }

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def spec(self, name: str) -> ECGDatasetSpec:
        key = name.lower()
        if key not in self.SPECS:
            raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(self.SPECS)}")
        return self.SPECS[key]

    def path(self, name: str) -> Path:
        return self.cache_dir / self.spec(name).name

    def ensure(self, name: str, records: list[str] | None = None) -> Path:
        spec = self.spec(name)
        target = self.path(name)
        if records and self._has_records(target, records):
            return target
        if not records and self._looks_downloaded(target):
            return target
        if spec.source == "physionet":
            return self._download_physionet(spec, target, records)
        if spec.source == "kaggle":
            return self._download_kaggle(spec, target)
        if spec.source == "huggingface":
            return self._download_huggingface(spec, target)
        raise ValueError(f"Unsupported source: {spec.source}")

    def records(self, name: str, max_records: int | None = None) -> list[str]:
        spec = self.spec(name)
        local_records = self._local_records(self.path(name))
        if not local_records and spec.physionet_slug:
            local_records = wfdb.get_record_list(spec.physionet_slug)
        return local_records[:max_records] if max_records else local_records

    def discover_records(self, name: str, max_records: int | None = None) -> list[str]:
        spec = self.spec(name)
        if spec.physionet_slug:
            records = wfdb.get_record_list(spec.physionet_slug)
        else:
            records = self._local_records(self.path(name))
        return records[:max_records] if max_records else records

    def record_path(self, name: str, record: str) -> str:
        return str(self.path(name) / record)

    @staticmethod
    def _looks_downloaded(path: Path) -> bool:
        return path.exists() and any(path.rglob("*.hea"))

    @staticmethod
    def _has_records(path: Path, records: list[str]) -> bool:
        return all((path / f"{record}.hea").exists() for record in records)

    @staticmethod
    def _local_records(path: Path) -> list[str]:
        records: list[str] = []
        if not path.exists():
            return records
        for header in path.rglob("*.hea"):
            records.append(str(header.with_suffix("").relative_to(path)).replace("\\", "/"))
        return sorted(records)

    def _download_physionet(
        self,
        spec: ECGDatasetSpec,
        target: Path,
        records: list[str] | None,
    ) -> Path:
        if not spec.physionet_slug:
            raise ValueError(f"No PhysioNet slug configured for {spec.name}")
        target.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(spec.physionet_slug, dl_dir=str(target), records=records)
        return target

    @staticmethod
    def _download_kaggle(spec: ECGDatasetSpec, target: Path) -> Path:
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            raise RuntimeError("Kaggle credentials require KAGGLE_USERNAME and KAGGLE_KEY")
        raise NotImplementedError(f"No Kaggle dataset configured for {spec.name}")

    @staticmethod
    def _download_huggingface(spec: ECGDatasetSpec, target: Path) -> Path:
        if not os.getenv("HF_TOKEN"):
            raise RuntimeError("Private HuggingFace datasets require HF_TOKEN")
        raise NotImplementedError(f"No HuggingFace dataset configured for {spec.name}")
