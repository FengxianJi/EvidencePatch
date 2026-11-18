"""Dependency-free BM25 retrieval over exact manuscript passages."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from bisect import bisect_right
import math
import re
from typing import Iterable

from dataset.parse import segment_markdown
from verifier.provenance import normalized_contains


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d+(?:\.\d+)?")
STOPWORDS = frozenset("""a an and are as at be been being by can could did do does for from had has have if in into is it its may might must of on or our paper should than that the their them then there these they this those to was we were what when where which while who will with would you your""".split())
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9([{])")


@dataclass(frozen=True)
class Passage:
    passage_id: str
    start_char: int
    end_char: int
    section: str
    text: str


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


def _section_boundaries(text: str) -> tuple[list[int], list[str]]:
    """Map parser section starts to exact character offsets in the source."""
    line_offsets = [0]
    for match in re.finditer("\n", text):
        line_offsets.append(match.end())
    parsed = segment_markdown(text)
    starts: list[int] = []
    names: list[str] = []
    for section in parsed:
        line_index = max(0, min(section["start_line"] - 1, len(line_offsets) - 1))
        start = line_offsets[line_index]
        if starts and start == starts[-1]:
            # An empty preamble can share the first line with the first
            # heading. Prefer the actual heading's name at that offset.
            names[-1] = section["heading"]
        else:
            starts.append(start)
            names.append(section["heading"])
    if not starts:
        return [0], ["Full manuscript"]
    if len(starts) == 1 and names == ["Preamble"]:
        names[0] = "Full manuscript"
    return starts, names


def _section_at(position: int, starts: list[int], names: list[str]) -> str:
    index = max(0, bisect_right(starts, position) - 1)
    return names[index]


def split_passages(text: str, max_words: int = 110, overlap_sentences: int = 1) -> list[Passage]:
    """Split source text into exact-offset passages with parsed section labels."""
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    passages: list[Passage] = []
    section_starts, section_names = _section_boundaries(text)
    for block_match in re.finditer(r"\S(?:.|\n)*?(?=\n\s*\n|\Z)", text):
        raw = block_match.group(0)
        stripped = raw.strip()
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", stripped)
        if heading and block_match.start() in section_starts:
            continue
        if not stripped or stripped.startswith("<!--"):
            continue
        section = _section_at(block_match.start(), section_starts, section_names)
        sentence_spans = []
        previous = 0
        for boundary in SENTENCE_RE.finditer(stripped):
            if stripped[previous:boundary.start()].strip():
                sentence_spans.append((previous, boundary.start()))
            previous = boundary.end()
        if stripped[previous:].strip():
            sentence_spans.append((previous, len(stripped)))
        if not sentence_spans:
            sentence_spans = [(0, len(stripped))]
        current: list[tuple[int, int]] = []
        current_words = 0
        chunks: list[tuple[int, int]] = []
        for sentence_start, sentence_end in sentence_spans:
            sentence = stripped[sentence_start:sentence_end]
            word_count = len(sentence.split())
            if current and current_words + word_count > max_words:
                chunks.append((current[0][0], current[-1][1]))
                carry = current[-overlap_sentences:] if overlap_sentences else []
                current = list(carry)
                current_words = sum(len(stripped[a:b].split()) for a, b in current)
            current.append((sentence_start, sentence_end))
            current_words += word_count
        if current:
            chunks.append((current[0][0], current[-1][1]))
        leading = len(raw) - len(raw.lstrip())
        for chunk_start, chunk_end in chunks:
            local_start = leading + chunk_start
            local_end = leading + chunk_end
            passage_text = raw[local_start:local_end].strip()
            absolute_start = block_match.start() + local_start
            absolute_end = block_match.start() + local_end
            passages.append(Passage(f"P{len(passages)+1:05d}", absolute_start, absolute_end, section, passage_text))
    return passages


def split_sections(text: str) -> list[Passage]:
    """Keep complete parsed sections intact for high-recall retrieval."""
    section_starts, section_names = _section_boundaries(text)
    sections: list[Passage] = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append(Passage(f"S{len(sections)+1:05d}", start, end, section_names[i], content))
    return sections


class BM25Index:
    def __init__(self, passages: Iterable[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = list(passages)
        self.k1 = k1
        self.b = b
        self.tokens = [_tokens(p.text) for p in self.passages]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tokens in self.tokens:
            df.update(set(tokens))
        self.idf = {term: math.log(1 + (len(self.passages) - freq + 0.5) / (freq + 0.5))
                    for term, freq in df.items()}

    def search(self, query: str, k: int = 10) -> list[tuple[Passage, float]]:
        if k < 1:
            raise ValueError("k must be positive")
        query_terms = Counter(_tokens(query))
        scored = []
        for passage, tokens, length in zip(self.passages, self.tokens, self.lengths):
            frequencies = Counter(tokens)
            score = 0.0
            for term, qtf in query_terms.items():
                if term not in self.idf:
                    continue
                tf = frequencies[term]
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.average_length) if self.average_length else 1
                score += self.idf[term] * tf * (self.k1 + 1) / norm * min(qtf, 2)
            if score > 0:
                scored.append((passage, score))
        return sorted(scored, key=lambda x: (-x[1], x[0].start_char))[:k]

    def search_with_char_budget(
        self, query: str, *, top_k: int, max_context_chars: int | None,
    ) -> list[tuple[Passage, float]]:
        """Select ranked passages without exceeding a per-query text budget.

        Oversized candidates are skipped rather than truncated, so every
        returned passage remains an exact source span. This is useful for
        complete-section retrieval, where a single section may be long.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_context_chars is not None and max_context_chars < 1:
            raise ValueError("max_context_chars must be positive when set")
        limit = len(self.passages) if max_context_chars is not None else top_k
        ranked = self.search(query, max(1, limit))
        selected: list[tuple[Passage, float]] = []
        used = 0
        for passage, score in ranked:
            size = len(passage.text)
            if max_context_chars is not None and used + size > max_context_chars:
                continue
            selected.append((passage, score))
            used += size
            if len(selected) >= top_k:
                break
        return selected


def retrieve_for_reviews(text: str, reviews: list[str], top_k: int = 8, max_words: int = 110) -> list[dict[str, object]]:
    passages = split_passages(text, max_words=max_words)
    index = BM25Index(passages)
    results = []
    for review_number, review in enumerate(reviews, 1):
        results.append({"review_number": review_number,
                        "passages": [{"passage_id": p.passage_id, "section": p.section, "text": p.text,
                                      "start_char": p.start_char, "end_char": p.end_char, "score": score}
                                     for p, score in index.search(review, top_k)]})
    return results


def retrieved_quote_recall(index: BM25Index, review: str, source_quote: str, ks: tuple[int, ...] = (3, 5, 8, 10)) -> dict[int, bool]:
    """Diagnostic only: whether exact source text is in a review's retrieved passages."""
    return {k: any(normalized_contains(p.text, source_quote) for p, _ in index.search(review, k)) for k in ks}
