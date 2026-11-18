"""Multi-resource budget planning with shared-cost and overlap handling.

This complements ``budget.py``'s scalar-cost API. Resource dimensions remain
separate; the planner never converts tokens, CPU time, wall time, or appendix
length into a post-hoc weighted scalar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import random
from typing import Iterable, Mapping, Sequence

from .budget import PatchCandidate, _candidate_map, _closure, _is_compatible, _utility_from_map


@dataclass(frozen=True)
class ResourceCandidate:
    candidate_id: str
    resource_costs: Mapping[str, float]
    expected_benefit: float
    uncertainty: float = 0.0
    risk: float = 0.0
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    shared_resource_costs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    claim_benefits: Mapping[str, float] = field(default_factory=dict)


def _prepare(
    candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float],
) -> tuple[dict[str, ResourceCandidate], dict[str, PatchCandidate], dict[str, float]]:
    caps = dict(budgets)
    if not caps or any(not isinstance(k, str) or not k for k in caps):
        raise ValueError("budgets must map non-empty resource names to finite caps")
    if any(not math.isfinite(v) or v < 0 for v in caps.values()):
        raise ValueError("resource budget caps must be finite and non-negative")
    by_id = {c.candidate_id: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate_id values must be unique")
    proxies = []
    for c in candidates:
        if not isinstance(c.candidate_id, str) or not c.candidate_id:
            raise ValueError("candidate_id values must be non-empty strings")
        values = list(c.resource_costs.values())
        for vector in c.shared_resource_costs.values():
            if not isinstance(vector, Mapping):
                raise ValueError(f"shared resource costs must be resource vectors: {c.candidate_id}")
            values.extend(vector.values())
        if any(not isinstance(dim, str) or not dim for dim in c.resource_costs):
            raise ValueError(f"resource names must be non-empty strings: {c.candidate_id}")
        if any(not isinstance(dim, str) or not dim for vector in c.shared_resource_costs.values() for dim in vector):
            raise ValueError(f"shared resource names must be non-empty strings: {c.candidate_id}")
        if any(not isinstance(shared_id, str) or not shared_id for shared_id in c.shared_resource_costs):
            raise ValueError(f"shared resource IDs must be non-empty strings: {c.candidate_id}")
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f"resource costs must be finite and non-negative: {c.candidate_id}")
        dimensions = set(c.resource_costs)
        for shared in c.shared_resource_costs.values():
            dimensions.update(shared)
        if dimensions - set(caps):
            raise ValueError(f"missing budget caps for resource dimensions: {sorted(dimensions - set(caps))}")
        proxies.append(PatchCandidate(
            candidate_id=c.candidate_id, cost=0.0,
            expected_benefit=c.expected_benefit, uncertainty=c.uncertainty,
            risk=c.risk, dependencies=c.dependencies, conflicts=c.conflicts,
            claim_benefits=c.claim_benefits,
        ))
    proxy_map = _candidate_map(proxies)
    return by_id, proxy_map, caps


def _resource_costs_from_map(
    ids: Iterable[str], by_id: Mapping[str, ResourceCandidate], caps: Mapping[str, float],
) -> dict[str, float]:
    selected = set(ids)
    if selected - set(by_id):
        raise ValueError(f"unknown selected candidates: {sorted(selected - set(by_id))}")
    total = {dimension: 0.0 for dimension in caps}
    shared: dict[str, dict[str, float]] = {}
    for candidate_id in selected:
        c = by_id[candidate_id]
        for dim, amount in c.resource_costs.items():
            total[dim] += amount
        for shared_id, vector in c.shared_resource_costs.items():
            target = shared.setdefault(shared_id, {})
            for dim, amount in vector.items():
                target[dim] = max(target.get(dim, 0.0), amount)
    for vector in shared.values():
        for dim, amount in vector.items():
            total[dim] += amount
    return total


def resource_costs(ids: Iterable[str], candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float]) -> dict[str, float]:
    by_id, _, caps = _prepare(candidates, budgets)
    return _resource_costs_from_map(ids, by_id, caps)


def resource_utility(
    ids: Iterable[str], candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float], *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0,
) -> float:
    by_id, proxy_map, _ = _prepare(candidates, budgets)
    selected = set(ids)
    if selected - set(by_id):
        raise ValueError(f"unknown selected candidates: {sorted(selected - set(by_id))}")
    if not math.isfinite(lambda_uncertainty) or lambda_uncertainty < 0 or not math.isfinite(mu_risk) or mu_risk < 0:
        raise ValueError("uncertainty and risk weights must be finite and non-negative")
    return _utility_from_map(selected, proxy_map, lambda_uncertainty, mu_risk)


def _compatible_budget_set(ids: Iterable[str], by_id: Mapping[str, ResourceCandidate], caps: Mapping[str, float]) -> set[str] | None:
    selected = _closure(ids, by_id)
    for candidate_id in selected:
        candidate = by_id[candidate_id]
        if not set(candidate.dependencies).issubset(selected) or set(candidate.conflicts) & selected:
            return None
    used = _resource_costs_from_map(selected, by_id, caps)
    if any(used[d] > cap and not math.isclose(used[d], cap, rel_tol=1e-12, abs_tol=1e-12) for d, cap in caps.items()):
        return None
    return selected


def random_resource_plan(candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float], seed: int) -> list[str]:
    by_id, _, caps = _prepare(candidates, budgets)
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    selected: set[str] = set()
    for candidate_id in order:
        proposed = _compatible_budget_set(selected | {candidate_id}, by_id, caps)
        if proposed is not None:
            selected = proposed
    return sorted(selected)


def all_feasible_resource_plan(candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float]) -> list[str]:
    by_id, _, caps = _prepare(candidates, budgets)
    selected: set[str] = set()
    for candidate_id in sorted(by_id):
        proposed = _compatible_budget_set(selected | {candidate_id}, by_id, caps)
        if proposed is not None:
            selected = proposed
    return sorted(selected)


def greedy_resource_plan(
    candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float], *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0,
) -> list[str]:
    """Greedy by net utility per largest fraction of configured resource caps."""
    by_id, proxy_map, caps = _prepare(candidates, budgets)
    if not math.isfinite(lambda_uncertainty) or lambda_uncertainty < 0 or not math.isfinite(mu_risk) or mu_risk < 0:
        raise ValueError("uncertainty and risk weights must be finite and non-negative")
    selected: set[str] = set()
    while True:
        current_utility = _utility_from_map(selected, proxy_map, lambda_uncertainty, mu_risk)
        current_cost = _resource_costs_from_map(selected, by_id, caps)
        choices = []
        for candidate_id in sorted(set(by_id) - selected):
            proposed = _compatible_budget_set(selected | {candidate_id}, by_id, caps)
            if proposed is None or proposed == selected:
                continue
            delta_utility = _utility_from_map(proposed, proxy_map, lambda_uncertainty, mu_risk) - current_utility
            if delta_utility <= 0:
                continue
            new_cost = _resource_costs_from_map(proposed, by_id, caps)
            fractions = []
            for dim, cap in caps.items():
                delta = new_cost[dim] - current_cost[dim]
                if cap == 0 and delta > 1e-12:
                    fractions.append(math.inf)
                elif cap > 0:
                    fractions.append(delta / cap)
            marginal_fraction = max(fractions, default=0.0)
            ratio = math.inf if marginal_fraction <= 1e-15 else delta_utility / marginal_fraction
            choices.append((ratio, delta_utility, -marginal_fraction, candidate_id, proposed))
        if not choices:
            break
        selected = max(choices, key=lambda item: item[:4])[4]
    return sorted(selected)


def optimal_resource_plan(
    candidates: Sequence[ResourceCandidate], budgets: Mapping[str, float], *,
    lambda_uncertainty: float = 1.0, mu_risk: float = 1.0, max_candidates: int = 22,
) -> list[str]:
    """Exact multi-cap subset search for small action pools."""
    by_id, proxy_map, caps = _prepare(candidates, budgets)
    if not math.isfinite(lambda_uncertainty) or lambda_uncertainty < 0 or not math.isfinite(mu_risk) or mu_risk < 0:
        raise ValueError("uncertainty and risk weights must be finite and non-negative")
    if len(by_id) > max_candidates:
        raise ValueError(f"exact search is limited to {max_candidates} candidates")
    best_key: tuple[float, float, tuple[str, ...]] = (0.0, 0.0, ())
    ordered = sorted(by_id)
    for size in range(1, len(ordered) + 1):
        for subset in combinations(ordered, size):
            selected = set(subset)
            if not _is_compatible(selected, proxy_map):
                continue
            costs = _resource_costs_from_map(selected, by_id, caps)
            if any(costs[d] > cap and not math.isclose(costs[d], cap, rel_tol=1e-12, abs_tol=1e-12) for d, cap in caps.items()):
                continue
            utility = _utility_from_map(selected, proxy_map, lambda_uncertainty, mu_risk)
            normalized_cost = sum(costs[d] / caps[d] for d in caps if caps[d] > 0)
            key = (utility, -normalized_cost, tuple(sorted(selected)))
            if key > best_key:
                best_key = key
    return list(best_key[2])
