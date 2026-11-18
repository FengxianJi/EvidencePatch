from __future__ import annotations

import random
import math


def bootstrap_mean_ci(values: list[float], seed: int = 1729, reps: int = 10000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(reps)]
    means.sort()
    return [means[int(0.025 * reps)], means[min(reps - 1, int(0.975 * reps))]]


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    radius = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_signed_rank_p(values: list[int]):
    """Two-sided exact Wilcoxon signed-rank randomization p-value."""
    nonzero = [(abs(v), v) for v in values if v]
    if not nonzero:
        return {"n_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "p_two_sided": 1.0}
    ordered = sorted(range(len(nonzero)), key=lambda i: nonzero[i][0])
    doubled_ranks = [0] * len(nonzero)
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and nonzero[ordered[end]][0] == nonzero[ordered[pos]][0]:
            end += 1
        rank2 = (pos + 1) + end
        for k in range(pos, end):
            doubled_ranks[ordered[k]] = rank2
        pos = end
    wplus2 = sum(doubled_ranks[i] for i, (_, value) in enumerate(nonzero) if value > 0)
    total2 = sum(doubled_ranks)
    counts = [1]
    for rank2 in doubled_ranks:
        next_counts = counts + [0] * rank2
        for subtotal, count in enumerate(counts):
            next_counts[subtotal + rank2] += count
        counts = next_counts
    distance2 = abs(2 * wplus2 - total2)
    extreme = sum(count for subtotal, count in enumerate(counts) if abs(2 * subtotal - total2) >= distance2)
    return {"n_nonzero": len(nonzero), "w_plus": wplus2 / 2, "w_minus": (total2 - wplus2) / 2,
            "p_two_sided": extreme / (2 ** len(nonzero))}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values, preserving keys and capping at 1."""
    if not p_values:
        return {}
    if any(not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p-values must be numeric and within [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * p_value))
        adjusted[name] = running_max
    return adjusted


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts."""
    if discordant_a < 0 or discordant_b < 0:
        raise ValueError("discordant counts must be non-negative")
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(lower + 1)) / (2 ** n))
