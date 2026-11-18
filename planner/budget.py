from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PatchCandidate:
    """A plan-time estimate for one evidence action.

    ``cost`` is action-specific. ``shared_costs`` captures resources paid once
    for the selected set (for example, a single shared training run).
    ``claim_benefits`` assigns expected benefit to concern IDs; when multiple
    actions address the same concern, the planner counts the largest estimate
    once rather than summing duplicate coverage. ``expected_benefit`` remains
    the action's non-overlapping benefit. Dependencies and conflicts use
    candidate IDs. Scores are estimates only; downstream adjudication must
    replace them with observed outcomes.
    """

    candidate_id: str
    cost: float
    expected_benefit: float
    uncertainty: float = 0.0
    risk: float = 0.0
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    shared_costs: Mapping[str, float] = field(default_factory=dict)
    claim_benefits: Mapping[str, float] = field(default_factory=dict)

    def utility(self, lambda_uncertainty: float = 1.0, mu_risk: float = 1.0) -> float:
        return (self.expected_benefit + sum(self.claim_benefits.values())
                - lambda_uncertainty * self.uncertainty - mu_risk * self.risk)


def _candidate_map(candidates: Sequence[PatchCandidate]) -> dict[str, PatchCandidate]:
    result = {candidate.candidate_id: candidate for candidate in candidates}
    if len(result) != len(candidates):
        raise ValueError("candidate_id values must be unique")
    for candidate in candidates:
        numeric = [candidate.cost, candidate.expected_benefit, candidate.uncertainty, candidate.risk,
                   *candidate.shared_costs.values(), *candidate.claim_benefits.values()]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Candidate values must be finite: {candidate.candidate_id}")
        if (candidate.cost < 0 or candidate.uncertainty < 0 or candidate.risk < 0
                or any(value < 0 for value in candidate.shared_costs.values())
                or any(value < 0 for value in candidate.claim_benefits.values())):
            raise ValueError(f"Costs, uncertainty, and risk must be non-negative: {candidate.candidate_id}")
        if any(not isinstance(claim_id, str) or not claim_id for claim_id in candidate.claim_benefits):
            raise ValueError(f"Claim benefit keys must be non-empty strings: {candidate.candidate_id}")
        missing = set(candidate.dependencies + candidate.conflicts) - set(result)
        if missing:
            raise ValueError(f"Unknown candidate references from {candidate.candidate_id}: {sorted(missing)}")
    return result


def _closure(ids: Iterable[str], by_id: Mapping[str, PatchCandidate]) -> set[str]:
    result = set(ids)
    pending = list(result)
    while pending:
        current = pending.pop()
        for dependency in by_id[current].dependencies:
            if dependency not in result:
                result.add(dependency)
                pending.append(dependency)
    return result


def _is_compatible(ids: set[str], by_id: Mapping[str, PatchCandidate]) -> bool:
    for candidate_id in ids:
        candidate = by_id[candidate_id]
        if not set(candidate.dependencies).issubset(ids):
            return False
        if set(candidate.conflicts) & ids:
            return False
    return True


def plan_cost(ids: Iterable[str], candidates: Sequence[PatchCandidate]) -> float:
    by_id = _candidate_map(candidates)
    selected = set(ids)
    unknown = selected - set(by_id)
    if unknown:
        raise ValueError(f"Unknown selected candidates: {sorted(unknown)}")
    return _cost_from_map(selected, by_id)


def _cost_from_map(ids: set[str], by_id: Mapping[str, PatchCandidate]) -> float:
    shared: dict[str, float] = {}
    total = 0.0
    for candidate_id in ids:
        candidate = by_id[candidate_id]
        total += candidate.cost
        for resource_id, cost in candidate.shared_costs.items():
            shared[resource_id] = max(shared.get(resource_id, 0.0), cost)
    return total + sum(shared.values())


def plan_utility(
    ids: Iterable[str], candidates: Sequence[PatchCandidate], *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0,
) -> float:
    by_id = _candidate_map(candidates)
    selected = set(ids)
    unknown = selected - set(by_id)
    if unknown:
        raise ValueError(f"Unknown selected candidates: {sorted(unknown)}")
    _validate_objective(0.0, lambda_uncertainty, mu_risk)
    return _utility_from_map(selected, by_id, lambda_uncertainty, mu_risk)


def _utility_from_map(
    ids: set[str], by_id: Mapping[str, PatchCandidate],
    lambda_uncertainty: float, mu_risk: float,
) -> float:
    additive_benefit = 0.0
    claim_benefit: dict[str, float] = {}
    penalty = 0.0
    for candidate_id in ids:
        candidate = by_id[candidate_id]
        additive_benefit += candidate.expected_benefit
        penalty += lambda_uncertainty * candidate.uncertainty + mu_risk * candidate.risk
        for claim_id, benefit in candidate.claim_benefits.items():
            claim_benefit[claim_id] = max(claim_benefit.get(claim_id, 0.0), benefit)
    return additive_benefit + sum(claim_benefit.values()) - penalty


def _validated_set(
    ids: Iterable[str], by_id: Mapping[str, PatchCandidate], budget: float,
) -> set[str] | None:
    selected = _closure(ids, by_id)
    total_cost = _cost_from_map(selected, by_id)
    within_budget = total_cost <= budget or math.isclose(total_cost, budget, rel_tol=1e-12, abs_tol=1e-12)
    if not _is_compatible(selected, by_id) or not within_budget:
        return None
    return selected


def _validate_objective(budget: float, lambda_uncertainty: float, mu_risk: float) -> None:
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    if not math.isfinite(lambda_uncertainty) or lambda_uncertainty < 0:
        raise ValueError("lambda_uncertainty must be finite and non-negative")
    if not math.isfinite(mu_risk) or mu_risk < 0:
        raise ValueError("mu_risk must be finite and non-negative")


def greedy_cost_plan(
    candidates: Sequence[PatchCandidate], budget: float, *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0,
) -> list[str]:
    """Cost-greedy plan by marginal risk-adjusted utility per marginal cost."""
    by_id = _candidate_map(candidates)
    _validate_objective(budget, lambda_uncertainty, mu_risk)
    selected: set[str] = set()
    while True:
        current_utility = _utility_from_map(selected, by_id, lambda_uncertainty, mu_risk)
        current_cost = _cost_from_map(selected, by_id)
        options = []
        for candidate_id in sorted(set(by_id) - selected):
            proposed = _validated_set(selected | {candidate_id}, by_id, budget)
            if proposed is None or proposed == selected:
                continue
            delta_cost = _cost_from_map(proposed, by_id) - current_cost
            delta_utility = _utility_from_map(proposed, by_id, lambda_uncertainty, mu_risk) - current_utility
            if delta_cost < 0 or delta_utility <= 0:
                continue
            utility_per_cost = math.inf if delta_cost == 0 else delta_utility / delta_cost
            options.append((utility_per_cost, delta_utility, -delta_cost, candidate_id, proposed))
        if not options:
            break
        selected = max(options, key=lambda item: item[:4])[4]
    return sorted(selected)


def random_budget_plan(candidates: Sequence[PatchCandidate], budget: float, seed: int) -> list[str]:
    """Random feasible patch baseline; order is reproducible for a fixed seed."""
    by_id = _candidate_map(candidates)
    _validate_objective(budget, 0.0, 0.0)
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    selected: set[str] = set()
    for candidate_id in order:
        proposed = _validated_set(selected | {candidate_id}, by_id, budget)
        if proposed is not None:
            selected = proposed
    return sorted(selected)


def all_feasible_plan(candidates: Sequence[PatchCandidate], budget: float) -> list[str]:
    """Add every dependency-compatible candidate that fits, in stable ID order."""
    by_id = _candidate_map(candidates)
    _validate_objective(budget, 0.0, 0.0)
    selected: set[str] = set()
    for candidate_id in sorted(by_id):
        proposed = _validated_set(selected | {candidate_id}, by_id, budget)
        if proposed is not None:
            selected = proposed
    return sorted(selected)


def optimal_budget_plan(
    candidates: Sequence[PatchCandidate], budget: float, *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0, max_candidates: int = 22,
) -> list[str]:
    """Exact subset search for small candidate sets, useful as a planner oracle."""
    by_id = _candidate_map(candidates)
    _validate_objective(budget, lambda_uncertainty, mu_risk)
    if len(candidates) > max_candidates:
        raise ValueError(f"Exact search is limited to {max_candidates} candidates")
    best: tuple[float, float, tuple[str, ...]] = (0.0, 0.0, ())
    ordered = sorted(by_id)
    for size in range(1, len(ordered) + 1):
        for subset in combinations(ordered, size):
            ids = set(subset)
            if not _is_compatible(ids, by_id) or _cost_from_map(ids, by_id) > budget:
                continue
            utility = _utility_from_map(ids, by_id, lambda_uncertainty, mu_risk)
            cost = _cost_from_map(ids, by_id)
            candidate_result = (utility, -cost, tuple(sorted(ids)))
            if candidate_result > best:
                best = candidate_result
    return list(best[2])


def render_evidence_index(items: Iterable[tuple[str, str, str]], max_chars: int) -> tuple[str, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Select whole verified pointers in model order under a hard character budget."""
    selected: list[tuple[str, str, str]] = []
    deferred: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        cid, claim, quote = item
        if (cid, quote) in seen:
            continue
        seen.add((cid, quote))
        candidate = selected + [item]
        appendix = _render(candidate)
        if len(appendix) <= max_chars:
            selected.append(item)
        else:
            deferred.append(item)
    return _render(selected) if selected else "", selected, deferred


def _render(items: list[tuple[str, str, str]]) -> str:
    blocks = ["## Evidence locator appendix", "", "The entries below reproduce short passages from the frozen manuscript. They add no new experiments, analyses, or scientific claims."]
    for cid, claim, quote in items:
        blocks.extend(["", f"### {cid}: manuscript evidence pointer", f"Review concern (M3-generated): {claim}", "Exact source text:", f"> {quote}"])
    return "\n".join(blocks)
