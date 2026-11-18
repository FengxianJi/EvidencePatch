"""Versioned PDF/LaTeX/Markdown ingestion and heading-based segmentation."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from dataset.corpus import sha256


def parse_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        import pymupdf
    except ImportError:
        pymupdf = None
    if pymupdf is not None:
        doc = pymupdf.open(stream=raw, filetype="pdf")
        pages = [page.get_text() for page in doc]
        parser = "PyMuPDF"
    elif shutil.which("mutool"):
        result = subprocess.run(["mutool", "draw", "-q", "-F", "txt", "-o", "-", str(path)],
                                check=True, capture_output=True, text=True)
        pages = result.stdout.split("\f")
        if pages and not pages[-1].strip():
            pages.pop()
        parser = "MuPDF CLI"
    else:
        raise RuntimeError("PDF parsing requires PyMuPDF or the MuPDF `mutool` command.")
    text = "\n\n".join(f"## Page {i + 1}\n\n{page}" for i, page in enumerate(pages))
    if not text.strip():
        raise RuntimeError(f"No embedded PDF text found in {path}; OCR may be required.")
    return text, {"path": str(path), "sha256": sha256(raw), "pages": len(pages), "parser": parser, "ocr": False}


def parse_latex(path: Path) -> tuple[str, dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("*.tex"))
    if not files:
        raise FileNotFoundError(f"No .tex files found under {path}")
    chunks = []
    versions = []
    for file in files:
        raw = file.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        # Preserve commands and math so the raw scientific source is auditable.
        chunks.append(f"% source-file: {file.name}\n{text}")
        versions.append({"path": str(file), "sha256": sha256(raw), "bytes": len(raw)})
    return "\n\n".join(chunks), {"path": str(path), "files": versions, "parser": "raw-LaTeX-preserving"}


def ingest_paper(*, markdown_path: Path | None = None, pdf_path: Path | None = None,
                 latex_path: Path | None = None) -> dict[str, Any]:
    """Load the preferred source plus all supplied source hashes."""
    if markdown_path:
        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        selected = {"format": "markdown", "path": str(markdown_path), "sha256": sha256(markdown_path.read_bytes())}
    elif latex_path:
        text, selected = parse_latex(latex_path)
        selected["format"] = "latex"
    elif pdf_path:
        text, selected = parse_pdf(pdf_path)
        selected["format"] = "pdf"
    else:
        raise ValueError("Supply a Markdown, LaTeX, or PDF source")
    versions = {"selected": selected}
    if pdf_path and pdf_path.exists():
        versions["pdf_sha256"] = sha256(pdf_path.read_bytes())
    if latex_path and latex_path.exists() and not selected.get("files"):
        versions["latex_sha256"] = sha256(latex_path.read_bytes()) if latex_path.is_file() else None
    return {"text": text, "source_versions": versions, "segments": segment_markdown(text)}


def segment_markdown(text: str) -> list[dict[str, Any]]:
    """Split on Markdown and LaTeX section headings with line provenance.

    Fenced Markdown examples stay inside their current section. LaTeX
    commands are recognized in raw source without rewriting it, preserving
    exact line locators for downstream evidence retrieval.
    """
    segments: list[dict[str, Any]] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    latex_heading_re = re.compile(
        r"\\(part|chapter|section|subsection|subsubsection)\*?"
        r"(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
    )
    latex_levels = {"part": 1, "chapter": 1, "section": 2,
                    "subsection": 3, "subsubsection": 4}
    current = {"section_id": "document-preamble", "heading": "Preamble", "level": 0, "start_line": 1, "lines": []}
    fence_char: str | None = None
    fence_size = 0

    def flush() -> None:
        current["text"] = "\n".join(current.pop("lines"))
        segments.append(dict(current))

    for line_number, line in enumerate(text.splitlines(), 1):
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = None, 0
            current["lines"].append(line)
            continue

        match = None if fence_char is not None else heading_re.match(line)
        heading = None
        level = None
        if match:
            heading, level = match.group(2), len(match.group(1))
        elif fence_char is None and not line.lstrip().startswith("%"):
            latex_match = latex_heading_re.search(line)
            if latex_match:
                command = latex_match.group(1)
                heading = re.sub(r"\s+", " ", latex_match.group(2)).strip()
                level = latex_levels[command]

        if heading is not None:
            if current["lines"] or not segments:
                flush()
            current = {"section_id": f"line-{line_number}", "heading": heading,
                       "level": level, "start_line": line_number, "lines": [line]}
        else:
            current["lines"].append(line)
    if current["lines"] or not segments:
        flush()
    return segments
