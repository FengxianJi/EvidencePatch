from __future__ import annotations

import json
import random
from typing import Any, Callable


def result(method_id: str, appendix: str = "", **extra: Any) -> dict[str, Any]:
    return {"method_id": method_id, "appendix_markdown": appendix.strip(), **extra}


def call_model(context: dict[str, Any], system: str, user: str, purpose: str) -> str:
    callback = context.get("llm_call")
    if not callable(callback):
        raise RuntimeError(f"{purpose} requires context['llm_call']")
    value = callback(system=system, user=user, purpose=purpose,
                     paper_id=str(context.get("paper_id", "unknown")))
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response contains no JSON object")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def render_entries(entries: list[dict[str, Any]], heading: str = "Evidence notes") -> str:
    lines = [f"## {heading}"]
    for item in entries:
        claim = str(item.get("claim", "")).strip()
        quote = str(item.get("quote", "")).strip()
        response = str(item.get("response", "")).strip()
        if not (claim or quote or response):
            continue
        lines.append("")
        if claim:
            lines.append(f"**Concern:** {claim}")
        if response:
            lines.append(response)
        if quote:
            lines.append(f"> {quote}")
    return "\n".join(lines) if len(lines) > 1 else ""


def exact_source_quotes(entries: list[dict[str, Any]], manuscript: str) -> list[dict[str, Any]]:
    verified = []
    normalized_source = " ".join(manuscript.split()).casefold()
    for item in entries:
        quote = str(item.get("quote", "")).strip()
        if quote and " ".join(quote.split()).casefold() in normalized_source:
            verified.append({**item, "quote": quote})
    return verified


def planner_select(context: dict[str, Any], policy: str) -> list[str]:
    """Delegate to the first experiment's audited planner implementation."""
    candidates = context.get("candidates", [])
    budget = float(context.get("budget", float("inf")))
    api = context.get("planner_api")
    if api is None:
        raise RuntimeError("budget baselines require context['planner_api']")
    converted = [api.PatchCandidate(
        candidate_id=str(c["candidate_id"]), cost=float(c.get("cost", 0)),
        expected_benefit=float(c.get("expected_benefit", 0)),
        uncertainty=float(c.get("uncertainty", 0)), risk=float(c.get("risk", 0)),
        dependencies=tuple(c.get("dependencies", ())), conflicts=tuple(c.get("conflicts", ())),
        shared_costs=c.get("shared_costs", {}), claim_benefits=c.get("claim_benefits", {}))
        for c in candidates]
    if policy == "random":
        return api.random_budget_plan(converted, budget, int(context.get("seed", 0)))
    if policy == "cost_greedy":
        return api.greedy_cost_plan(converted, budget)
    if policy == "all_feasible":
        return api.all_feasible_plan(converted, budget)
    if policy == "oracle":
        return api.optimal_budget_plan(converted, budget)
    raise ValueError(f"Unknown planner policy: {policy}")


def selected_evidence(context: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {str(c["candidate_id"]): c for c in context.get("candidates", [])}
    return [{"claim": str(by_id[cid].get("claim", cid)),
             "quote": str(by_id[cid].get("quote", "")),
             "response": str(by_id[cid].get("response", "")),
             "candidate_id": cid} for cid in ids if cid in by_id]
