"""Independent hard-constraint checks for model-proposed budget plans."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from planner.budget import PatchCandidate, plan_cost, plan_utility


def audit_budget_plan(
    selected_ids: Iterable[str], candidates: Sequence[PatchCandidate], budget: float, *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0,
) -> dict[str, Any]:
    """Return score and explicit violations; never repair a model proposal silently."""
    ids = list(selected_ids)
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    known = set(candidate_ids)
    selected = set(ids)
    violations: list[dict[str, Any]] = []

    if len(candidate_ids) != len(known):
        raise ValueError("candidate_id values must be unique")
    if len(ids) != len(selected):
        violations.append({"type": "duplicate_candidate_id"})
    unknown = sorted(selected - known)
    if unknown:
        violations.append({"type": "unknown_candidate_id", "candidate_ids": unknown})
        return {
            "selected_ids": sorted(selected), "cost": None, "utility": None,
            "within_budget": False, "missing_dependencies": [],
            "violated_conflicts": [], "feasible": False, "violations": violations,
        }

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    missing_dependencies = sorted(
        f"{candidate_id}->{dependency}"
        for candidate_id in selected
        for dependency in by_id[candidate_id].dependencies
        if dependency not in selected
    )
    violated_conflicts = sorted({
        tuple(sorted((candidate_id, conflict)))
        for candidate_id in selected
        for conflict in by_id[candidate_id].conflicts
        if conflict in selected
    })
    if missing_dependencies:
        violations.append({"type": "missing_dependency", "edges": missing_dependencies})
    if violated_conflicts:
        violations.append({"type": "conflicting_candidates", "pairs": [list(pair) for pair in violated_conflicts]})

    cost = plan_cost(selected, candidates)
    valid_budget = math.isfinite(budget) and budget >= 0
    within_budget = bool(valid_budget and (cost <= budget or math.isclose(cost, budget, rel_tol=1e-12, abs_tol=1e-12)))
    if not valid_budget:
        violations.append({"type": "invalid_budget"})
    elif not within_budget:
        violations.append({"type": "budget_exceeded", "cost": cost, "budget": budget})
    utility = plan_utility(
        selected, candidates,
        lambda_uncertainty=lambda_uncertainty, mu_risk=mu_risk,
    )
    return {
        "selected_ids": sorted(selected), "cost": cost, "utility": utility,
        "within_budget": within_budget,
        "missing_dependencies": missing_dependencies,
        "violated_conflicts": [list(pair) for pair in violated_conflicts],
        "feasible": not violations, "violations": violations,
    }
