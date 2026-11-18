"""Conservative review-to-source evidence locator generation and rendering."""
from __future__ import annotations

import re
from typing import Any

from planner.budget import render_evidence_index
from verifier.provenance import normalized_contains


ALLOWED_ACTIONS = {"evidence_pointer", "clarification"}


def _section_before(text: str, quote: str) -> str:
    at = text.casefold().find(quote.casefold())
    if at < 0:
        return ""
    prefix = text[:at]
    headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", prefix))
    return headings[-1].group(1).strip() if headings else "Preamble"


def verify_and_render(patch: dict[str, Any], manuscript: str, reviews: list[str], max_chars: int = 2500,
                      max_concerns: int = 3) -> dict[str, Any]:
    """Accept only one concise review excerpt paired to exact manuscript text.

    Model-generated paraphrases and explanations are deliberately excluded from
    the rendering because quote presence does not verify their entailment.
    """
    by_review: dict[int, dict[str, Any]] = {}
    checks = []
    items = patch.get("concerns") if isinstance(patch.get("concerns"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            review_n = int(item.get("review_number"))
        except (TypeError, ValueError):
            review_n = 0
        review_excerpt = str(item.get("review_excerpt", "")).strip()
        sources = item.get("source_quotes", [])
        if not isinstance(sources, list):
            sources = []
        clean_sources = [str(q).strip() for q in sources if str(q).strip() and len(str(q).split()) <= 25]
        review_exists = 1 <= review_n <= len(reviews)
        review_match = review_exists and len(review_excerpt.split()) <= 30 and normalized_contains(reviews[review_n - 1], review_excerpt)
        source_matches = [normalized_contains(manuscript, q) for q in clean_sources]
        action_ok = item.get("action") in ALLOWED_ACTIONS
        eligible = bool(review_match and action_ok and clean_sources and all(source_matches))
        checks.append({"review_number": review_n, "review_excerpt_found": bool(review_match),
                       "source_quote_count": len(clean_sources), "all_source_quotes_found": bool(source_matches) and all(source_matches),
                       "allowed_action": action_ok, "eligible": eligible})
        if eligible and review_n not in by_review:
            by_review[review_n] = {"review_number": review_n, "review_excerpt": review_excerpt,
                                   "source_quotes": clean_sources, "action": item["action"]}
    selected = [by_review[k] for k in sorted(by_review)[:max_concerns]]
    appendix_blocks = ["## Review-to-manuscript evidence locator", "",
        "Each entry pairs a verbatim review excerpt with text already present in the frozen manuscript. This locator does not claim by itself that the concern is resolved."]
    for item in selected:
        appendix_blocks.extend(["", f"### Review {item['review_number']}: cited concern", "> " + item["review_excerpt"]])
        for quote in item["source_quotes"]:
            section = _section_before(manuscript, quote)
            appendix_blocks.extend([f"Potentially relevant manuscript passage ({section}):", "> " + quote])
    appendix = "\n".join(appendix_blocks) if selected else ""
    if len(appendix) > max_chars:
        # Keep the review-to-source pair atomic; do not truncate quotations.
        kept = []
        for item in selected:
            candidate = kept + [item]
            blocks = ["## Review-to-manuscript evidence locator", "",
                "Each entry pairs a verbatim review excerpt with text already present in the frozen manuscript. This locator does not claim by itself that the concern is resolved."]
            for chosen in candidate:
                blocks.extend(["", f"### Review {chosen['review_number']}: cited concern", "> " + chosen["review_excerpt"]])
                for quote in chosen["source_quotes"]:
                    blocks.extend([f"Potentially relevant manuscript passage ({_section_before(manuscript, quote)}):", "> " + quote])
            if len("\n".join(blocks)) <= max_chars:
                kept = candidate
        selected = kept
        appendix_blocks = ["## Review-to-manuscript evidence locator", "",
            "Each entry pairs a verbatim review excerpt with text already present in the frozen manuscript. This locator does not claim by itself that the concern is resolved."]
        for item in selected:
            appendix_blocks.extend(["", f"### Review {item['review_number']}: cited concern", "> " + item["review_excerpt"]])
            for quote in item["source_quotes"]:
                appendix_blocks.extend([f"Potentially relevant manuscript passage ({_section_before(manuscript, quote)}):", "> " + quote])
        appendix = "\n".join(appendix_blocks) if selected else ""
    return {"valid": bool(selected), "checks": checks, "selected_entries": selected,
            "safe_appendix_markdown": appendix, "appendix_chars": len(appendix),
            "candidate_count": len(items), "selected_count": len(selected),
            "eligible_review_count": len(by_review), "deferred_by_budget": max(0, len(by_review) - len(selected))}
