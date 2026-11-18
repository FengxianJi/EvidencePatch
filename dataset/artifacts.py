"""Index local code, data, experiment outputs, and environment files by hash."""
from __future__ import annotations

import hashlib
from pathlib import Path


def index_artifacts(paths: list[Path]) -> list[dict[str, str | int]]:
    rows = []
    for entry in paths:
        files = [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.is_file())
        for file in files:
            raw = file.read_bytes()
            suffix = file.suffix.lower()
            role = "code" if suffix in {".py", ".r", ".jl", ".sh", ".ipynb"} else "result" if suffix in {".csv", ".json", ".jsonl", ".txt", ".log"} else "data" if suffix in {".parquet", ".feather", ".pkl", ".npy", ".npz"} else "environment" if file.name in {"requirements.txt", "environment.yml", "pyproject.toml", "Dockerfile"} else "other"
            rows.append({"path": str(file), "role": role, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return rows
