#!/usr/bin/env python3
"""Pair primary-300 reviewer-score deltas with corrected resolution proxy v3."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCORE_CSV = HERE / "dataset/primary300_analysis_bundle_v1/paper_scores.csv"
PRIMARY_STATE = HERE / "results/confirmatory_primary_300/experiment/state.json"
PRIMARY_CONTRASTS = HERE / "results/completed_experiments_2026-11-18/primary300_contrasts.csv"
SECONDARY_JSON = HERE / "results/confirmatory_primary_300/experiment/predeclared_secondary_analysis.json"
RUN_DIR = HERE / "results/primary300_resolution_proxy_m3_v3"
RESOLUTION_CSV = RUN_DIR / "corrected_paper_metrics.csv"
CONCERN_CSV = RUN_DIR / "corrected_concern_records.csv"
OUT = HERE / "results/primary300_score_resolution_alignment_v2"
BOOTSTRAPS = 10_000
SEED = 20261118


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = ((i + 1) + j) / 2
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def corr(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(dx * dy) if dx and dy else 0.0


def population_sd(xs: list[float]) -> float:
    center = mean(xs)
    return math.sqrt(mean([(x - center) ** 2 for x in xs]))


def spearman(xs: list[float], ys: list[float]) -> float:
    return corr(ranks(xs), ranks(ys))


def main() -> None:
    score_rows = list(csv.DictReader(SCORE_CSV.open(encoding="utf-8-sig", newline="")))
    contrast_rows = list(csv.DictReader(PRIMARY_CONTRASTS.open(encoding="utf-8-sig", newline="")))
    secondary = json.loads(SECONDARY_JSON.read_text(encoding="utf-8"))
    primary_state = json.loads(PRIMARY_STATE.read_text(encoding="utf-8"))["papers"]
    resolution_rows = list(csv.DictReader(RESOLUTION_CSV.open(encoding="utf-8-sig", newline="")))
    scores = {row["cohort_id"]: row for row in score_rows}
    resolutions = {row["case_id"]: row for row in resolution_rows}
    if len(scores) != 300 or len(resolutions) != 300 or scores.keys() != resolutions.keys():
        raise RuntimeError("Expected exactly matched, unique 300-paper input coverage")
    pairs = []
    excluded = 0
    for case_id, resolution in resolutions.items():
        score = scores[case_id]
        if resolution["source_sha256"] != score["source_sha256"]:
            raise RuntimeError(f"Source hash mismatch: {case_id}")
        if resolution["review_text_sha256"] != score["review_text_sha256"]:
            raise RuntimeError(f"Review hash mismatch: {case_id}")
        if int(resolution["valid_repairable_concern_count"]) < 1:
            excluded += 1
            continue
        pairs.append((float(score["targeted_concern_evidence_minus_baseline"]),
                      float(resolution["C5_minus_C0"])))

    sx = [x for x, _ in pairs]
    ry = [y for _, y in pairs]
    rho = spearman(sx, ry)
    pearson = corr(sx, ry)
    rng = random.Random(SEED)
    boot = []
    for _ in range(BOOTSTRAPS):
        idx = [rng.randrange(len(pairs)) for _ in pairs]
        boot.append(spearman([sx[i] for i in idx], [ry[i] for i in idx]))
    boot.sort()
    signs = Counter((sign(x), sign(y)) for x, y in pairs)
    groups = {}
    for name, predicate in (("resolution_proxy_increased", lambda y: y > 0),
                            ("resolution_proxy_unchanged", lambda y: y == 0),
                            ("resolution_proxy_decreased", lambda y: y < 0)):
        group = [(x, y) for x, y in pairs if predicate(y)]
        groups[name] = {
            "papers": len(group),
            "mean_score_delta": mean([x for x, _ in group]) if group else None,
            "score_delta_sign_counts": {str(k): v for k, v in Counter(sign(x) for x, _ in group).items()},
        }

    paper_descriptives = []
    for case_id, paper in primary_state.items():
        arm_scores = paper["scores"]
        arm_means = {
            arm: mean([float(arm_scores[repeat][arm]["score"]) for repeat in arm_scores])
            for arm in ("baseline", "placebo", "targeted_concern_evidence")
        }
        proxy = resolutions[case_id]
        raw_delta = proxy["C5_minus_C0"]
        patch = paper["patches"]["targeted_concern_evidence"]
        exact_rate = patch["exact_source_quote_rate"]
        paper_descriptives.append({
            "baseline_score": arm_means["baseline"],
            "score_delta": arm_means["targeted_concern_evidence"] - arm_means["baseline"],
            "score_vs_placebo_delta": arm_means["targeted_concern_evidence"] - arm_means["placebo"],
            "resolution_delta": float(raw_delta) if raw_delta else None,
            "patch_chars": len(patch["appendix"]),
            "patch_entries": int(patch["selected_count"]),
            "source_quote_match_rate": float(exact_rate) if exact_rate not in (None, "") else 0.0,
            "patch_empty": not bool(patch["appendix"].strip()),
        })
    discordant = [row for row in paper_descriptives
                  if row["resolution_delta"] is not None and row["resolution_delta"] > 0 and row["score_delta"] < 0]
    empty_patches = [row for row in paper_descriptives if row["patch_empty"]]
    nonempty_patches = [row for row in paper_descriptives if not row["patch_empty"]]

    concerns = list(csv.DictReader(CONCERN_CSV.open(encoding="utf-8-sig", newline="")))
    eligible = [row for row in concerns if row["paired_analysis_eligible"] == "True"]
    transitions = Counter((row["C0_resolution"], row["C5_resolution"]) for row in eligible)
    quote_counts = {}
    for arm in ("C0", "C5"):
        quoted = [row for row in eligible if row[f"{arm}_quote"]]
        quote_counts[arm] = {
            "nonempty_quotes": len(quoted),
            "exact_quotes": sum(row[f"{arm}_quote_exact_recomputed"] == "True" for row in quoted),
        }
    repairability_counts = Counter(row["repairability"] for row in concerns)
    validity_counts = Counter(row["validity"] for row in concerns)
    actionability = {}
    for action in ("existing_evidence", "clarification", "reanalysis"):
        action_rows = [row for row in eligible if row["repairability"] == action]
        upgraded = sum(row["C0_resolution"] == "unresolved"
                       and row["C5_resolution"] in ("partially_resolved", "fully_resolved")
                       for row in action_rows)
        actionability[action] = {"eligible_concerns": len(action_rows), "unresolved_to_resolved": upgraded,
                                 "upgrade_rate": upgraded / len(action_rows) if action_rows else None}

    baseline_strata = {}
    for name, predicate in (("low_lt_3", lambda x: x < 3),
                            ("middle_3_to_lt4", lambda x: 3 <= x < 4),
                            ("high_ge4", lambda x: x >= 4)):
        stratum = [row for row in paper_descriptives if predicate(row["baseline_score"])]
        with_proxy = [row for row in stratum if row["resolution_delta"] is not None]
        baseline_strata[name] = {
            "papers": len(stratum),
            "mean_C5_minus_C0_score": mean([row["score_delta"] for row in stratum]) if stratum else None,
            "mean_C5_minus_placebo_score": mean([row["score_vs_placebo_delta"] for row in stratum]) if stratum else None,
            "papers_with_proxy": len(with_proxy),
            "mean_resolution_proxy_delta": mean([row["resolution_delta"] for row in with_proxy]) if with_proxy else None,
        }
    patch_correlations = {}
    for feature in ("baseline_score", "patch_chars", "patch_entries", "source_quote_match_rate"):
        patch_correlations[feature] = corr([row["score_delta"] for row in paper_descriptives],
                                           [row[feature] for row in paper_descriptives])
    repeatability = {}
    for arm in ("baseline", "placebo", "clarification", "reproducibility", "targeted_concern_evidence"):
        per_paper_sd = [population_sd([float(paper["scores"][repeat][arm]["score"])
                                       for repeat in paper["scores"]])
                        for paper in primary_state.values()]
        repeatability[arm] = {"mean_within_paper_sd_across_three_calls": mean(per_paper_sd)}
    feedback_patterns = {
        "explicit_no_or_insufficient_evidence": r"(no|lack of|lacking|insufficient|missing)\s+(\w+\s+){0,3}evidence",
        "unsupported_or_overclaim": r"\b(unsupported|unsubstantiated|overclaim|not substantiated)\b",
        "experimental_gap_terms": r"\b(experiment|evaluation|dataset|baseline|ablation|statistic)\w*\b",
        "clarity_terms": r"\b(unclear|confusing|clarity|readability)\w*\b",
    }
    feedback_scan = {}
    for label, pattern in feedback_patterns.items():
        paper_flags = {}
        for arm in ("baseline", "placebo", "targeted_concern_evidence"):
            paper_flags[arm] = []
            for paper in primary_state.values():
                text = " ".join(
                    weakness
                    for repeat in paper["scores"]
                    for weakness in paper["scores"][repeat][arm].get("weaknesses", [])
                )
                paper_flags[arm].append(bool(re.search(pattern, text, re.IGNORECASE)))
        feedback_scan[label] = {
            "papers_with_keyword_match_by_arm": {arm: sum(flags) for arm, flags in paper_flags.items()},
            "C0_only_papers": sum(a and not b for a, b in zip(
                paper_flags["baseline"], paper_flags["targeted_concern_evidence"])),
            "C5_only_papers": sum(b and not a for a, b in zip(
                paper_flags["baseline"], paper_flags["targeted_concern_evidence"])),
        }

    summary = {
        "analysis_id": "primary300-score-resolution-alignment-v2",
        "status": "complete_exploratory_descriptive",
        "cohort_papers": 300,
        "paired_papers_with_valid_repairable_concern": len(pairs),
        "excluded_papers_without_proxy_denominator": excluded,
        "score_delta": {"mean": mean(sx), "metric": "M3 holistic 1-5 paper score, C5 minus C0"},
        "resolution_proxy_delta": {"mean": mean(ry), "metric": "paper-macro concern status proxy, C5 minus C0"},
        "alignment": {
            "pearson_r": pearson,
            "spearman_rho": rho,
            "spearman_paper_bootstrap_ci95": [boot[int(.025 * BOOTSTRAPS)], boot[int(.975 * BOOTSTRAPS)]],
            "bootstrap_replicates": BOOTSTRAPS,
            "bootstrap_seed": SEED,
            "direction_counts_score_sign_then_resolution_sign": {
                f"{a:+d},{b:+d}": n for (a, b), n in sorted(signs.items())
            },
            "resolution_groups": groups,
        },
        "concern_level": {
            "paired_concerns": len(eligible),
            "all_extracted_concern_validity_counts": dict(validity_counts),
            "all_extracted_concern_repairability_counts": dict(repairability_counts),
            "C0_to_C5_transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
            "resolution_upgrade_rate_by_actionability": actionability,
            "eligible_exact_quote_counts": quote_counts,
        },
        "primary_score_contrasts": {
            row["contrast"]: {k: row[k] for k in ("mean_delta", "ci95_low", "ci95_high")}
            for row in contrast_rows if row["contrast"] in {
                "C1 placebo vs C0 baseline", "C2 clarification vs C0 baseline",
                "C3 reproducibility vs C0 baseline", "C5 targeted evidence vs C0 baseline (primary)",
                "C5 targeted evidence vs C1 placebo",
            }
        },
        "predeclared_C5_minus_placebo": secondary["contrasts"]["C5_targeted_concern_evidence_minus_C1_placebo"],
        "patch_outcome_descriptives": {
            "papers_with_empty_c5_appendix": len(empty_patches),
            "empty_patch_mean_score_delta": mean([row["score_delta"] for row in empty_patches]) if empty_patches else None,
            "nonempty_patch_mean_score_delta": mean([row["score_delta"] for row in nonempty_patches]) if nonempty_patches else None,
            "proxy_up_score_down_papers": len(discordant),
            "proxy_up_score_down_mean_c5_patch_chars": mean([row["patch_chars"] for row in discordant]) if discordant else None,
            "score_delta_pearson_r_with_patch_and_baseline_features": patch_correlations,
            "baseline_score_strata": baseline_strata,
            "mean_within_paper_score_sd_across_three_calls": repeatability,
            "weakness_feedback_keyword_screen": feedback_scan,
        },
        "input_sha256": {"score_csv": sha(SCORE_CSV), "primary_state": sha(PRIMARY_STATE),
                         "primary_contrast_csv": sha(PRIMARY_CONTRASTS),
                         "primary_secondary_analysis": sha(SECONDARY_JSON),
                         "corrected_v3_paper_csv": sha(RESOLUTION_CSV),
                         "corrected_v3_concern_csv": sha(CONCERN_CSV)},
        "interpretation_limit": "Same-family model proxies and post-hoc descriptive alignment; not expert validity, causal attribution, or human concern-resolution truth.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f'''# Primary-300 score and concern-resolution alignment (v3 inputs)

Status: complete, paired, post-hoc descriptive analysis. This recomputes the prior alignment using corrected v3 concern records; it does not replace either underlying experiment.

## Paired result

- Matched cohort: 300/300; paired analysis includes {len(pairs)} papers with at least one model-labeled valid and repairable concern. The other {excluded} have no eligible proxy denominator and are excluded, not assigned zero.
- Mean holistic score change (C5−C0): {mean(sx):+.4f}/5.
- Mean concern-resolution proxy change (C5−C0): {mean(ry):+.4f}.
- Pearson r: {pearson:.4f}; Spearman rho: {rho:.4f}, paper bootstrap 95% CI [{summary['alignment']['spearman_paper_bootstrap_ci95'][0]:.4f}, {summary['alignment']['spearman_paper_bootstrap_ci95'][1]:.4f}].
- Paper-level direction counts (score sign, resolution-proxy sign): {dict(signs)}.
- Among {groups['resolution_proxy_increased']['papers']} papers with a positive resolution proxy change, score changes were negative for {groups['resolution_proxy_increased']['score_delta_sign_counts'].get('-1', 0)}, tied for {groups['resolution_proxy_increased']['score_delta_sign_counts'].get('0', 0)}, and positive for {groups['resolution_proxy_increased']['score_delta_sign_counts'].get('1', 0)}.

## Concern-level decomposition

The 1,745 extracted concerns include 717 classified as requiring a new experiment and 92 as a core-method/claim defect; neither class enters the paired proxy. The 873 eligible concern pairs transition as follows: {dict(transitions)}. Thus the positive mean proxy comes from model-assigned unresolved-to-partial/full transitions; it is not a broad shift over every concern. Upgrade rates differ by action type: {actionability}. Exact quote matching among eligible concerns: C0 {quote_counts['C0']['exact_quotes']}/{quote_counts['C0']['nonempty_quotes']} and C5 {quote_counts['C5']['exact_quotes']}/{quote_counts['C5']['nonempty_quotes']}. Exact string match verifies provenance only, not semantic support or resolution.

## Evidence-based attribution

1. **The metrics target different constructs and are given different context.** The primary scoring prompt receives the title, frozen manuscript, and supplement, but not the original reviews; it asks for a holistic 1–5 recommendation. The resolution prompt receives the original reviews and asks the model to extract and label individual concerns against both anonymous supplements. This protocol mismatch is directly confirmed in the frozen code/prompts. The proxy also averages only concerns MiniMax calls valid and repairable using existing evidence, clarification, or reanalysis; it excludes concerns requiring a new experiment, core-method/claim defects, and other ineligible labels. A narrow response can change a local concern label while leaving the overall recommendation unchanged or lower.
2. **The observed per-paper alignment is weak.** The latest v3-input Spearman estimate is near zero and its paper bootstrap interval spans negative and positive associations. Of the papers whose resolution proxy increased, fewer than one fifth had a positive score change. This directly shows that score movement is not a usable stand-in for this proxy in this cohort; it does not reveal which underlying mechanism caused the mismatch.
3. **A generic supplementary-material penalty and a C5-specific penalty are both visible in the score proxy.** C1 placebo (-0.071), C2 clarification (-0.054), C3 reproducibility (-0.092), and C5 (-0.233) all score below C0 on average. The predeclared, prompt-token-matched C5−C1 contrast is {secondary['contrasts']['C5_targeted_concern_evidence_minus_C1_placebo']['mean']:+.3f} (95% CI [{secondary['contrasts']['C5_targeted_concern_evidence_minus_C1_placebo']['paper_level_bootstrap_ci95'][0]:+.3f}, {secondary['contrasts']['C5_targeted_concern_evidence_minus_C1_placebo']['paper_level_bootstrap_ci95'][1]:+.3f}]). This confirms that this M3 scoring setup rated targeted material lower than a neutral appendix beyond a simple length difference. The negative C5−C0 direction also appears in the existing Qwen3 and Mistral panels (-0.857 and -0.133, respectively); these are still AI score proxies and have no matched concern-resolution panels. The observed score decline is therefore not unique to the M3 call, but its human meaning is unknown. Twenty-five C5 appendices are empty and copy the baseline score by protocol; among the 275 nonempty patches the mean score delta is -0.2545, so empty patches do not account for the negative mean.
4. **The measurement protocols also differ in replication and scale.** Holistic score averages three model calls per condition. The resolution outcome comes from one paired call that extracts up to six concerns and assigns full/partial/unresolved labels. The latter has a restricted denominator and more aggregation ties, so differences in stochastic noise, prompt framing, and scale may contribute.
5. **Evidence matching is not evidence sufficiency, and the proxy has a possible same-family leniency risk.** Exact quote checks establish lexical provenance only. They do not establish that the quote entails the response, is relevant to the concern, or supplies missing results. Among eligible concerns, no C5 item is labeled worse than C0; 75 move from unresolved to partial/full. That could reflect legitimate added support, but it could also reflect the same-model paired judge's preference or asymmetric decision boundary. Two quotes exceed the output word limit overall. None of these checks establishes expert-confirmed resolution.
6. **Remaining patch-level explanations are hypotheses.** The supplement may be too narrow, fail to address major concerns that need new experiments, introduce unsupported or weakly relevant claims, or draw attention to gaps. Judge-family style preferences and context effects may also matter. Existing aggregate data do not isolate these causes; expert-coded discordant cases are needed.

## Further exploratory checks

- **Actionability:** The proxy upgrade rate is highest when a concern is labeled addressable with existing evidence ({actionability['existing_evidence']['unresolved_to_resolved']}/{actionability['existing_evidence']['eligible_concerns']}, {actionability['existing_evidence']['upgrade_rate']:.1%}); it is lower for clarification ({actionability['clarification']['unresolved_to_resolved']}/{actionability['clarification']['eligible_concerns']}, {actionability['clarification']['upgrade_rate']:.1%}) and reanalysis ({actionability['reanalysis']['unresolved_to_resolved']}/{actionability['reanalysis']['eligible_concerns']}, {actionability['reanalysis']['upgrade_rate']:.1%}). This is consistent with evidence pointers helping most when the needed evidence already exists, but these action labels are model-generated.
- **Baseline-score strata:** C5−C0 averaged {baseline_strata['low_lt_3']['mean_C5_minus_C0_score']:+.3f} for the low-baseline stratum ({baseline_strata['low_lt_3']['papers']} papers), {baseline_strata['middle_3_to_lt4']['mean_C5_minus_C0_score']:+.3f} for the middle stratum ({baseline_strata['middle_3_to_lt4']['papers']}), and {baseline_strata['high_ge4']['mean_C5_minus_C0_score']:+.3f} for the high stratum ({baseline_strata['high_ge4']['papers']}). The high stratum is too small for inference. The middle-stratum decline also appears against placebo; subgroup findings are exploratory and not a causal explanation.
- **Patch metadata:** Score deltas have descriptive Pearson correlations of {patch_correlations['patch_chars']:+.3f} with appendix length, {patch_correlations['patch_entries']:+.3f} with selected entry count, {patch_correlations['source_quote_match_rate']:+.3f} with candidate source-quote match rate, and {patch_correlations['baseline_score']:+.3f} with baseline score. The score-delta/baseline correlation is mathematically coupled because baseline is subtracted from the outcome; it is not evidence of baseline quality causing the decline. These are unadjusted associations; paper difficulty and number of concerns can confound patch metadata. They do not show that longer or better-matched patches lower scores.
- **Repeat variation:** Mean within-paper standard deviation across three M3 calls was {repeatability['baseline']['mean_within_paper_sd_across_three_calls']:.3f} for C0 and {repeatability['targeted_concern_evidence']['mean_within_paper_sd_across_three_calls']:.3f} for C5. Repeated scoring reduces noise in the paper mean but does not address the missing review context or the one-call resolution proxy.
- **Score-rationale screen:** A literal keyword scan of the three-call weakness text found explicit “no/insufficient evidence” wording in {feedback_scan['explicit_no_or_insufficient_evidence']['papers_with_keyword_match_by_arm']['baseline']} C0 papers, {feedback_scan['explicit_no_or_insufficient_evidence']['papers_with_keyword_match_by_arm']['targeted_concern_evidence']} C5 papers, and {feedback_scan['explicit_no_or_insufficient_evidence']['papers_with_keyword_match_by_arm']['placebo']} placebo papers. “Unsupported/overclaim” terms appeared in {feedback_scan['unsupported_or_overclaim']['papers_with_keyword_match_by_arm']['baseline']} C0 versus {feedback_scan['unsupported_or_overclaim']['papers_with_keyword_match_by_arm']['targeted_concern_evidence']} C5 papers. This crude lexical screen does not semantically code the rationale; it gives no evidence that unsupported claims broadly increased under C5 and cannot explain the numerical score shift.

## Interpretation

This analysis supports reporting reviewer-score change and concern-resolution proxy as separate outcomes. It does not establish expert-confirmed concern resolution or attribute the negative score change to a specific patch defect. The next attribution study should have blinded experts label concern validity, evidence entailment/relevance, resolution, and paper score separately; stratify by concern type and actionability; then audit discordant cases (proxy up/score down and proxy flat/score up) with reason codes. Preserve an untouched held-out subset for confirmatory evaluation.

Inputs and reproducible aggregation are pinned by `summary.json` and `evaluation/analyze_primary300_score_resolution_alignment_v2.py`.
'''
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
