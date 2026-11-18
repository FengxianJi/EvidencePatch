from __future__ import annotations

import re
from typing import Any

from planner.budget import render_evidence_index


def _normalized_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace/case while retaining each normalized char's source span."""
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            if chars and end < len(text):
                chars.append(" ")
                spans.append((index, end))
            index = end
            continue
        folded = char.casefold()
        for normalized_char in folded:
            chars.append(normalized_char)
            spans.append((index, index + 1))
        index += 1
    return "".join(chars), spans


def find_quote_span(haystack: str, quote: str) -> dict[str, int] | None:
    """Return an exact, whitespace-normalized quote's source offsets and lines."""
    normalized_haystack, spans = _normalized_with_offsets(haystack)
    return _find_quote_span_in_index(haystack, normalized_haystack, spans, quote)


def _find_quote_span_in_index(
    haystack: str, normalized_haystack: str, spans: list[tuple[int, int]], quote: str,
) -> dict[str, int] | None:
    normalized_quote, _ = _normalized_with_offsets(quote)
    if not normalized_quote:
        return None
    match_start = normalized_haystack.find(normalized_quote)
    if match_start < 0:
        return None
    match_end = match_start + len(normalized_quote)
    start_char = spans[match_start][0]
    end_char = spans[match_end - 1][1]
    return {
        "start_char": start_char,
        "end_char": end_char,
        "start_line": haystack.count("\n", 0, start_char) + 1,
        "end_line": haystack.count("\n", 0, end_char - 1) + 1,
    }


def normalized_contains(haystack: str, quote: str) -> bool:
    return find_quote_span(haystack, quote) is not None


def verify_patch(patch: dict[str, Any], original: str, max_chars: int = 8000) -> dict[str, Any]:
    concerns = patch.get("concerns")
    if not isinstance(concerns, list):
        concerns = []
    checks = []
    safe_items = []
    normalized_original, original_spans = _normalized_with_offsets(original)
    for concern in concerns:
        quotes = concern.get("source_quotes", []) if isinstance(concern, dict) else []
        clean_quotes = [str(q).strip() for q in quotes if str(q).strip()]
        usable_quotes = [q for q in clean_quotes if len(q.split()) <= 25]
        quote_locations = [
            _find_quote_span_in_index(original, normalized_original, original_spans, quote)
            for quote in clean_quotes
        ]
        all_quotes_short = bool(clean_quotes) and len(usable_quotes) == len(clean_quotes)
        quote_checks = [location is not None for location in quote_locations]
        allowed_action = isinstance(concern, dict) and concern.get("action") in ("evidence_pointer", "clarification")
        item_valid = all_quotes_short and bool(quote_checks) and all(quote_checks) and allowed_action
        checks.append({
            "concern_id": concern.get("id", "") if isinstance(concern, dict) else "",
            "action": concern.get("action", "") if isinstance(concern, dict) else "",
            "has_source_quote": bool(clean_quotes),
            "all_quotes_found": bool(quote_checks) and all(quote_checks),
            "all_quotes_at_most_25_words": all_quotes_short,
            "source_quote_locations": quote_locations,
            "included_in_patch": item_valid,
        })
        if item_valid:
            safe_items.extend((str(concern.get("id", "")), str(concern.get("claim", "")), q) for q in clean_quotes)
    appendix, selected, deferred = render_evidence_index(safe_items, max_chars=max_chars)
    return {
        "valid": bool(selected), "appendix_chars": len(appendix), "checks": checks,
        "safe_appendix_markdown": appendix, "selected_pointers": len(selected),
        "deferred_by_budget": len(deferred),
    }
