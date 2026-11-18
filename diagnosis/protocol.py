import json
import re
from typing import Any

REVIEW_SYSTEM = """You are a careful scientific reviewer. Evaluate only claims supported by the supplied manuscript. Do not infer quality from writing polish or length. The score is an ordinal recommendation: 1 strong reject, 2 reject, 3 borderline, 4 accept, 5 strong accept. Return only valid JSON matching the requested schema."""
PATCH_SYSTEM = """You diagnose review concerns and draft a strictly evidence-only appendix. Never invent methods, results, citations, data, code, or experiments. Use only facts explicitly present in the manuscript text. If a concern cannot be addressed from existing material, mark it unresolved and state what evidence is missing. Cite exact short source quotations (max 25 words each) so a local verifier can check provenance. Return JSON only."""


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result
