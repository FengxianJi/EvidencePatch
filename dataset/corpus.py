from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_cohort(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = root / config["dataset_catalog"]
    text_dir = root / config["dataset_text_dir"]
    eligible: list[dict[str, Any]] = []
    with catalog.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("task_role") != config["cohort_role"]:
                continue
            rel = row.get("md", "")
            path = root / "paper_fin" / rel
            if not rel or not path.is_file() or path.parent != text_dir:
                continue
            raw = path.read_bytes()
            row["_path"] = path
            row["_sha256"] = sha256(raw)
            row["_text"] = raw.decode("utf-8", errors="replace")
            pdf_rel = row.get("pdf", "")
            pdf_path = root / "paper_fin" / pdf_rel if pdf_rel else None
            row["_pdf_path"] = pdf_path if pdf_path and pdf_path.is_file() else None
            row["_pdf_sha256"] = sha256(row["_pdf_path"].read_bytes()) if row["_pdf_path"] else ""
            eligible.append(row)
    if len(eligible) < config["sample_size"]:
        raise RuntimeError(f"Cohort has {len(eligible)} papers; need {config['sample_size']}.")
    random.Random(config["sample_seed"]).shuffle(eligible)
    return eligible[: config["sample_size"]]


def select_relevant_text(text: str, limit: int) -> str:
    """Deterministically retain the beginning, sampled middle, and conclusion."""
    text = text.replace("\x00", " ")
    if len(text) <= limit:
        return text
    intro = text[: min(6500, limit // 3)]
    tail_budget = min(3000, limit // 6)
    tail = text[-tail_budget:]
    middle_budget = limit - len(intro) - len(tail) - 80
    remaining = text[len(intro) : -len(tail)]
    chunk_size = max(1200, middle_budget // 5)
    chunks = []
    for i in range(5):
        start = max(0, int((len(remaining) - chunk_size) * i / 4))
        chunks.append(remaining[start : start + chunk_size])
    middle = "\n[...document excerpt... ]\n".join(chunks)
    return (intro + "\n[...middle sections sampled... ]\n" + middle + "\n[...conclusion... ]\n" + tail)[:limit]


def write_manifest(papers: list[dict[str, Any]], root: Path, output: Path) -> None:
    fields = ["paper_id", "title", "task_role", "document_kind", "source_url", "pdf_path", "pdf_sha256", "md_path", "sha256", "bytes"]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            writer.writerow({
                "paper_id": p.get("id", ""), "title": p.get("title", ""),
                "task_role": p.get("task_role", ""), "document_kind": p.get("document_kind", ""),
                "source_url": p.get("source_url", ""),
                "pdf_path": str(p["_pdf_path"].relative_to(root)) if p.get("_pdf_path") else "",
                "pdf_sha256": p.get("_pdf_sha256", ""),
                "md_path": str(p["_path"].relative_to(root)), "sha256": p["_sha256"],
                "bytes": p["_path"].stat().st_size,
            })
