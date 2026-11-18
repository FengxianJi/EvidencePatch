#!/usr/bin/env python3
"""Independent, no-API validation and corrected aggregation for resolution proxy v3."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
RUN = HERE / "results" / "primary300_resolution_proxy_m3_v3"
PRIMARY = HERE / "results" / "confirmatory_primary_300" / "experiment"
COHORT = HERE / "results" / "reviewarena_openreview300"
REVIEWS_PATH = HERE / "results" / "confirmatory_600" / "review_packets.json"
OUT_VALIDATION = RUN / "integrity_validation.json"
OUT_METRICS = RUN / "corrected_metrics.json"
OUT_REPORT = RUN / "report.md"
OUT_PAPER_CSV = RUN / "corrected_paper_metrics.csv"
OUT_CONCERN_CSV = RUN / "corrected_concern_records.csv"
SEED = 20261022
B = 10_000
ELIGIBLE_REPAIRABILITY = {"existing_evidence", "clarification", "reanalysis"}
SCORE = {"fully_resolved": 1.0, "partially_resolved": 0.5,
         "unresolved": 0.0, "introduced_unsupported_claim": 0.0}
EXPECTED_API_BUDGET = {
    "max_scoring_api_events": 500,
    "max_reported_tokens": 8_500_000,
    "max_scoring_events_per_call": 10,
    "request_token_reserve": 75_000,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def boot(diffs: list[float], rng: random.Random) -> tuple[float | None, float | None]:
    if not diffs:
        return None, None
    means = []
    for _ in range(B):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(.025 * B)], means[int(.975 * B)]


def check_api_budget(metrics: dict[str, Any], state: dict[str, Any],
                     events: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    metrics_budget = metrics.get("api_budget") or {}
    state_budget = state.get("api_budget") or {}
    for key, expected in EXPECTED_API_BUDGET.items():
        if metrics_budget.get(key) != expected:
            issues.append(f"metrics api_budget.{key}={metrics_budget.get(key)!r}; expected {expected!r}")
        if state_budget.get(key) != expected:
            issues.append(f"state api_budget.{key}={state_budget.get(key)!r}; expected {expected!r}")
    total_tokens = sum(int(event.get("total_tokens") or 0) for event in events)
    if len(events) > EXPECTED_API_BUDGET["max_scoring_api_events"]:
        issues.append(f"scoring API events {len(events)} exceed cap")
    if total_tokens > EXPECTED_API_BUDGET["max_reported_tokens"]:
        issues.append(f"reported tokens {total_tokens} exceed cap")
    credentials = sorted({int(event["credential_index"]) for event in events
                          if event.get("credential_index") is not None})
    if any(index not in (2, 3) for index in credentials):
        issues.append(f"unexpected credential indices in scoring ledger: {credentials}")
    return {"status": "passed" if not issues else "failed", "issues": issues,
            "scoring_api_events": len(events), "reported_tokens": total_tokens,
            "credential_indices": credentials, "caps": EXPECTED_API_BUDGET}


def check_run_coverage(metrics: dict[str, Any], state_ids: set[str],
                       manifest_ids: set[str]) -> list[str]:
    status = metrics.get("status")
    scored = metrics.get("papers_scored")
    issues = []
    if status not in ("complete", "incomplete"):
        issues.append(f"unexpected original run status: {status!r}")
    if scored != len(state_ids):
        issues.append("original metrics papers_scored differs from saved state coverage")
    if status == "complete" and state_ids != manifest_ids:
        issues.append(f"complete state/manifest ID mismatch: missing={len(manifest_ids-state_ids)}, extra={len(state_ids-manifest_ids)}")
    if status == "incomplete" and not state_ids < manifest_ids:
        issues.append(f"incomplete state is not a strict manifest subset: saved={len(state_ids)}, manifest={len(manifest_ids)}")
    return issues


def main() -> None:
    metrics_original = json.loads((RUN / "metrics.json").read_text())
    state_path = RUN / "state.json"
    state = json.loads(state_path.read_text())
    manifest_path = PRIMARY / "primary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_state = json.loads((PRIMARY / "state.json").read_text())["papers"]
    review_rows = json.loads(REVIEWS_PATH.read_text())
    usage_path = RUN / "api_usage.jsonl"
    events = [json.loads(line) for line in usage_path.read_text().splitlines() if line.strip()]
    budget_validation = check_api_budget(metrics_original, state, events)
    reviews_by_id = {str(x["cohort_id"]): x for x in review_rows}
    manifest_by_id = {str(x["cohort_id"]): x for x in manifest}
    errors: list[str] = []
    if len(manifest) != 300 or len(manifest_by_id) != 300:
        errors.append(f"manifest rows/unique IDs not 300: {len(manifest)}/{len(manifest_by_id)}")
    state_ids = set(state.get("papers", {}))
    manifest_ids = set(manifest_by_id)
    partial_run = metrics_original.get("status") == "incomplete"
    errors.extend(check_run_coverage(metrics_original, state_ids, manifest_ids))

    source_hash_mismatches: list[str] = []
    review_hash_mismatches: list[str] = []
    supplement_hash_mismatches: list[str] = []
    order_errors: list[str] = []
    schema_errors: list[str] = []
    output_constraint_violations: list[str] = []
    condition_field_mismatches: list[str] = []
    randomization_mismatches: list[str] = []
    quote_checks = 0
    exact_quote_count = 0
    nonempty_quote_count = 0
    stored_quote_flag_mismatches = 0
    quote_flag_mismatch_examples: list[dict[str, Any]] = []
    concern_total = 0
    valid_total = 0
    invalid_total = 0
    uncertain_total = 0
    requires_experiment = 0
    repairable_pairs: list[tuple[str, float, float]] = []
    by_paper: dict[str, list[tuple[float, float]]] = {}
    corrected_concern_rows: list[dict[str, Any]] = []

    allowed_validity = {"valid", "not_valid", "uncertain"}
    allowed_repairability = {"existing_evidence", "clarification", "reanalysis",
                             "requires_new_experiment", "core_method_or_claim_defect",
                             "not_actionable", "uncertain"}
    for case_id in sorted(state_ids & manifest_ids):
        rec = state["papers"][case_id]
        row = manifest_by_id[case_id]
        source_path = Path(row["paper_path"])
        full_paper = source_path.read_text(encoding="utf-8", errors="replace")
        full_bytes = source_path.read_bytes()
        if sha(full_bytes) != row.get("paper_sha256") or rec.get("source_sha256") != row.get("paper_sha256"):
            source_hash_mismatches.append(case_id)
        base_rec = source_state.get(case_id, {})
        if rec.get("main_text_sha256") != base_rec.get("main_text_sha256"):
            source_hash_mismatches.append(case_id + ":main_text")
        reviews = [x for x in reviews_by_id.get(case_id, {}).get("reviews", [])
                   if x.get("has_text", bool(str(x.get("review_text") or "").strip()))]
        joined_reviews = "\n\n".join(str(x.get("review_text") or "").strip() for x in reviews)
        if sha(joined_reviews.encode()) != rec.get("review_text_sha256") or rec.get("review_text_sha256") != base_rec.get("review_text_sha256"):
            review_hash_mismatches.append(case_id)
        patch = str(base_rec.get("patches", {}).get("targeted_concern_evidence", {}).get("appendix") or "")
        expected_patch_hash = sha(patch.encode())
        if rec.get("supplement_sha256") != expected_patch_hash:
            supplement_hash_mismatches.append(case_id)

        order = rec.get("variant_order", [])
        if len(order) != 2 or set(order) != {"baseline", "targeted_concern_evidence"}:
            order_errors.append(case_id)
            continue
        expected_order = ["baseline", "targeted_concern_evidence"]
        random.Random(SEED ^ sum(map(ord, case_id))).shuffle(expected_order)
        if order != expected_order:
            randomization_mismatches.append(case_id)
        # The raw response is A/B. Correctly map those labels using the randomized input order.
        expected_a_condition, expected_b_condition = order[0], order[1]
        if rec.get("conditions_are_randomized_opaque_A_B") is not True:
            condition_field_mismatches.append(case_id + ":paper_randomization_marker")

        paper_pairs: list[tuple[float, float]] = []
        concerns = rec.get("concerns", [])
        if len(concerns) > 6:
            schema_errors.append(case_id + ":more_than_six_concerns")
        seen_ids: set[str] = set()
        for concern in concerns:
            concern_total += 1
            concern_id = str(concern.get("concern_id") or "")
            # The runner stores A/B condition metadata on each concern, not on the
            # paper record. Validate every stored judgment against the frozen order.
            if (concern.get("variant_A_condition") != expected_a_condition
                    or concern.get("variant_B_condition") != expected_b_condition):
                condition_field_mismatches.append(case_id + ":" + (concern_id or "missing_concern_id"))
            if not concern_id or concern_id in seen_ids:
                schema_errors.append(case_id + ":missing_or_duplicate_concern_id")
            seen_ids.add(concern_id)
            validity = concern.get("validity")
            repairability = concern.get("repairability")
            if validity not in allowed_validity:
                schema_errors.append(case_id + ":validity")
            if repairability not in allowed_repairability:
                schema_errors.append(case_id + ":repairability")
            try:
                confidence = float(concern.get("confidence"))
                if not 0.0 <= confidence <= 1.0:
                    schema_errors.append(case_id + ":confidence_out_of_range")
            except (TypeError, ValueError):
                schema_errors.append(case_id + ":confidence_missing_or_invalid")
            review_index = concern.get("review_index")
            if not isinstance(review_index, int) or not 1 <= review_index <= len(reviews):
                schema_errors.append(case_id + ":review_index_out_of_range")
            valid_total += validity == "valid"
            invalid_total += validity == "not_valid"
            uncertain_total += validity == "uncertain" or repairability == "uncertain"
            requires_experiment += repairability == "requires_new_experiment"

            # Quote A is checked against whichever condition was actually placed in A.
            # This corrects the v2 runner's post-hoc quote checker, which assumed A=C0/B=C5.
            quote_by_condition: dict[str, tuple[str, bool, int]] = {}
            for label, condition in (("A", expected_a_condition), ("B", expected_b_condition)):
                quote = str(concern.get(f"variant_{label}_support_quote") or "").strip()
                word_count = len(quote.split())
                if quote and word_count > 20:
                    output_constraint_violations.append(case_id + ":quote_over_20_words")
                saved_flag = concern.get(f"variant_{label}_support_quote_exact")
                appendix = patch if condition == "targeted_concern_evidence" else ""
                haystack = full_paper + "\n" + appendix
                exact = bool(quote and quote.casefold() in haystack.casefold())
                quote_by_condition[condition] = (quote, exact, word_count)
                quote_checks += 1
                nonempty_quote_count += bool(quote)
                exact_quote_count += exact
                if saved_flag is not exact:
                    stored_quote_flag_mismatches += 1
                    if len(quote_flag_mismatch_examples) < 20:
                        quote_flag_mismatch_examples.append({
                            "case_id": case_id, "concern_id": concern.get("concern_id"),
                            "variant": label, "condition_in_that_variant": condition,
                            "stored_flag": saved_flag, "recomputed_flag": exact,
                        })

            status_by_condition = {
                expected_a_condition: concern.get("variant_A_resolution"),
                expected_b_condition: concern.get("variant_B_resolution"),
            }
            value_by_condition = {condition: SCORE.get(label)
                                  for condition, label in status_by_condition.items()}
            c0_quote, c0_exact, c0_words = quote_by_condition.get("baseline", ("", False, 0))
            c5_quote, c5_exact, c5_words = quote_by_condition.get("targeted_concern_evidence", ("", False, 0))
            paired_eligible = validity == "valid" and repairability in ELIGIBLE_REPAIRABILITY
            corrected_concern_rows.append({
                "case_id": case_id, "concern_id": concern_id, "validity": validity,
                "repairability": repairability, "confidence": concern.get("confidence"),
                "review_index": review_index,
                "C0_resolution": status_by_condition.get("baseline"),
                "C5_resolution": status_by_condition.get("targeted_concern_evidence"),
                "C0_proxy_value": value_by_condition.get("baseline"),
                "C5_proxy_value": value_by_condition.get("targeted_concern_evidence"),
                "paired_analysis_eligible": paired_eligible,
                "C0_quote": c0_quote, "C0_quote_exact_recomputed": c0_exact,
                "C0_quote_word_count": c0_words,
                "C5_quote": c5_quote, "C5_quote_exact_recomputed": c5_exact,
                "C5_quote_word_count": c5_words,
                "source_sha256": rec.get("source_sha256"),
                "review_text_sha256": rec.get("review_text_sha256"),
                "supplement_sha256": rec.get("supplement_sha256"),
            })

            if validity == "valid" and repairability in ELIGIBLE_REPAIRABILITY:
                status_a = concern.get("variant_A_resolution")
                status_b = concern.get("variant_B_resolution")
                if status_a not in SCORE or status_b not in SCORE:
                    schema_errors.append(case_id + ":resolution")
                    continue
                score_by_condition = {expected_a_condition: SCORE[status_a],
                                      expected_b_condition: SCORE[status_b]}
                baseline = score_by_condition["baseline"]
                patch_value = score_by_condition["targeted_concern_evidence"]
                paper_pairs.append((baseline, patch_value))
                repairable_pairs.append((case_id, baseline, patch_value))
        if paper_pairs:
            by_paper[case_id] = paper_pairs

    # Recompute the same paper-cluster bootstrap used in the run, independently from saved metrics.
    macro_pairs = [(mean([a for a, _ in xs]), mean([b for _, b in xs]))
                   for xs in by_paper.values()]
    macro_diffs = [b - a for a, b in macro_pairs]
    micro_pairs = [(a, b) for xs in by_paper.values() for a, b in xs]
    rng = random.Random(SEED)
    macro_ci = boot(macro_diffs, rng)
    micro_diffs = [b - a for a, b in micro_pairs]
    micro_ci = boot(micro_diffs, rng)
    # Match the runner's RNG sequence: its paper bootstrap consumes RNG before concern bootstrap.
    check_rng = random.Random(SEED)
    _ = boot(macro_diffs, check_rng)
    micro_ci = boot(micro_diffs, check_rng)

    improved = sum(d > 0 for d in macro_diffs)
    equal = sum(d == 0 for d in macro_diffs)
    worsened = sum(d < 0 for d in macro_diffs)
    macro_mean = mean(macro_diffs) if macro_diffs else None
    corrected = {
        "run_id": state.get("run_id"),
        "status": ("incomplete_validated" if partial_run and not errors else
                   "incomplete_with_integrity_issues" if partial_run else
                   "complete_validated" if not errors else "complete_with_integrity_issues"),
        "papers_expected": 300, "papers_verified": len(state_ids & manifest_ids),
        "concern_count": concern_total, "valid_concerns": valid_total, "invalid_concerns": invalid_total,
        "uncertain_concerns_or_repairability": uncertain_total,
        "requires_new_experiment_concerns": requires_experiment,
        "repairable_valid_concerns_in_paired_analysis": len(micro_pairs),
        "papers_with_at_least_one_repairable_valid_concern": len(macro_pairs),
        "paper_macro_c5_minus_c0_mean": macro_mean,
        "paper_macro_paired_bootstrap_ci95": macro_ci,
        "concern_micro_paired_bootstrap_ci95": micro_ci,
        "paper_count_patch_improved_equal_worsened": {"improved": improved, "equal": equal, "worsened": worsened},
        "corrected_quote_exactness": {"nonempty_quotes": nonempty_quote_count,
                                      "exact_quotes": exact_quote_count,
                                      "exact_quote_rate": exact_quote_count / nonempty_quote_count if nonempty_quote_count else None},
        "quote_flag_mismatches_vs_correct_randomized_condition": stored_quote_flag_mismatches,
        "requested_quote_word_limit_violations": len(output_constraint_violations),
        "runner_condition_field_mismatches": len(condition_field_mismatches),
        "bootstrap": {"unit": "paper cluster", "replicates": B, "seed": SEED},
        "model": metrics_original.get("evaluator"),
        "interpretation_limit": "Post-hoc, same-family MiniMax-M3 proxy; not expert adjudication, not a causal quality claim.",
        "rights_limit": "Local-only; per-article rights not individually cleared.",
        "gpu_used": False,
        "source_hashes": {"primary_manifest": sha(manifest_path.read_bytes()),
                          "primary_state": sha((PRIMARY / "state.json").read_bytes()),
                          "review_packets": sha(REVIEWS_PATH.read_bytes()),
                          "proxy_state": sha(state_path.read_bytes()),
                          "api_usage": sha((RUN / "api_usage.jsonl").read_bytes())},
    }
    source_fingerprint_components = {
        "primary_manifest": metrics_original.get("source_fingerprints", {}).get("primary_manifest") == sha(manifest_path.read_bytes()),
        "primary_state": metrics_original.get("source_fingerprints", {}).get("primary_state") == sha((PRIMARY / "state.json").read_bytes()),
        "review_packets": metrics_original.get("source_fingerprints", {}).get("review_packets") == sha(REVIEWS_PATH.read_bytes()),
        "runner": metrics_original.get("source_fingerprints", {}).get("runner") == sha((HERE / "run_primary300_resolution_proxy_v3.py").read_bytes()),
        "corpus_selector": metrics_original.get("source_fingerprints", {}).get("corpus_selector") == sha((HERE / "dataset/corpus.py").read_bytes()),
        "api_client": metrics_original.get("source_fingerprints", {}).get("api_client") == sha((HERE / "executor/minimax.py").read_bytes()),
        "json_parser": metrics_original.get("source_fingerprints", {}).get("json_parser") == sha((HERE / "diagnosis/protocol.py").read_bytes()),
    }
    source_fingerprint_runner_matches = source_fingerprint_components["runner"]
    source_fingerprints_match = all(source_fingerprint_components.values())
    core_integrity_passed = not (
        errors or budget_validation["issues"] or source_hash_mismatches or review_hash_mismatches or supplement_hash_mismatches
        or order_errors or randomization_mismatches or schema_errors or not source_fingerprints_match
    )
    mapping_integrity_passed = not condition_field_mismatches and stored_quote_flag_mismatches == 0
    status = (
        "incomplete_validated" if core_integrity_passed and mapping_integrity_passed
        and partial_run and not output_constraint_violations
        else "incomplete_with_output_constraint_violations"
        if core_integrity_passed and mapping_integrity_passed and partial_run
        else "passed" if core_integrity_passed and mapping_integrity_passed and not output_constraint_violations
        else "passed_with_output_constraint_violations"
        if core_integrity_passed and mapping_integrity_passed
        else "failed"
    )
    validation = {
        "status": status,
        "partial_run": partial_run,
        "paper_id_coverage_exact": state_ids == manifest_ids,
        "manifest_count": len(manifest), "state_count": len(state_ids),
        "source_hash_mismatches": source_hash_mismatches,
        "review_hash_mismatches": review_hash_mismatches,
        "supplement_hash_mismatches": supplement_hash_mismatches,
        "randomized_order_errors": order_errors,
        "deterministic_randomization_mismatches": randomization_mismatches,
        "schema_errors": schema_errors[:100], "schema_error_count": len(schema_errors),
        "output_constraint_violations": output_constraint_violations[:100],
        "output_constraint_violation_count": len(output_constraint_violations),
        "randomized_condition_field_mismatch_count": len(condition_field_mismatches),
        "condition_field_mismatch_examples": condition_field_mismatches[:12],
        "quote_flags_recomputed": quote_checks,
        "quote_flag_mismatch_count": stored_quote_flag_mismatches,
        "quote_flag_mismatch_examples": quote_flag_mismatch_examples,
        "api_events": len(events),
        "api_budget_validation": budget_validation,
        "api_error_attempts": 0,
        "api_error_attempt_details": [],
        "api_errors_recovered": 0,
        "api_errors_unrecovered": 0,
        "source_fingerprint_runner_matches_current": source_fingerprint_runner_matches,
        "source_fingerprint_components_match_current": source_fingerprint_components,
        "corrected_paper_csv_sha256": None,
        "corrected_metrics_sha256": sha((json.dumps(corrected, ensure_ascii=False, indent=2) + "\n").encode()),
        "validator_sha256": sha(Path(__file__).read_bytes()),
    }
    err_events = [x for x in events if x.get("status") == "error"]
    recovered, unrecovered = [], []
    for i, item in enumerate(events):
        if item.get("status") != "error":
            continue
        ok = next((x for x in events[i+1:] if x.get("paper_id") == item.get("paper_id")
                   and x.get("purpose") == item.get("purpose") and x.get("status") != "error"), None)
        (recovered if ok else unrecovered).append(item)
    validation["api_error_attempts"] = len(err_events)
    validation["api_error_attempt_details"] = [{"paper_id": x.get("paper_id"), "type": x.get("error_type"),
                                                "elapsed_seconds": x.get("elapsed_seconds"),
                                                "recovered": x in recovered} for x in err_events]
    validation["api_errors_recovered"] = len(recovered)
    validation["api_errors_unrecovered"] = len(unrecovered)
    validation["api_total_reported_tokens"] = sum(x.get("total_tokens", 0) or 0 for x in events)
    paper_rows = []
    for case_id in sorted(state_ids & manifest_ids):
        rec = state["papers"][case_id]
        xs = by_paper.get(case_id, [])
        a_condition, b_condition = rec.get("variant_order", [None, None])
        paper_rows.append({
            "case_id": case_id,
            "variant_A_condition": a_condition,
            "variant_B_condition": b_condition,
            "concern_count": len(rec.get("concerns", [])),
            "valid_repairable_concern_count": len(xs),
            "C0_mean_resolution_proxy": mean([a for a, _ in xs]) if xs else None,
            "C5_mean_resolution_proxy": mean([b for _, b in xs]) if xs else None,
            "C5_minus_C0": mean([b-a for a, b in xs]) if xs else None,
            "source_sha256": rec.get("source_sha256"),
            "review_text_sha256": rec.get("review_text_sha256"),
            "supplement_sha256": rec.get("supplement_sha256"),
        })
    with OUT_PAPER_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paper_rows[0]))
        writer.writeheader()
        writer.writerows(paper_rows)
    validation["corrected_paper_csv_sha256"] = sha(OUT_PAPER_CSV.read_bytes())
    with OUT_CONCERN_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(corrected_concern_rows[0]))
        writer.writeheader()
        writer.writerows(corrected_concern_rows)
    with OUT_CONCERN_CSV.open(encoding="utf-8", newline="") as stream:
        normalized_rows = list(csv.DictReader(stream))
    normalized_eligible = [row for row in normalized_rows if row["paired_analysis_eligible"] == "True"]
    normalized_valid = (len(normalized_rows) == concern_total
                        and len(normalized_eligible) == len(micro_pairs)
                        and all(row["C0_resolution"] and row["C5_resolution"]
                                and row["C0_proxy_value"] and row["C5_proxy_value"]
                                for row in normalized_eligible))
    validation["corrected_concern_csv_sha256"] = sha(OUT_CONCERN_CSV.read_bytes())
    validation["corrected_concern_rows"] = len(normalized_rows)
    validation["corrected_concern_eligible_pairs"] = len(normalized_eligible)
    validation["corrected_concern_export_roundtrip_valid"] = normalized_valid
    validation["raw_state_sha256_after_export"] = sha(state_path.read_bytes())
    validation["raw_state_unchanged"] = validation["raw_state_sha256_after_export"] == corrected["source_hashes"]["proxy_state"]
    if (not normalized_valid or not validation["raw_state_unchanged"]
            or not source_fingerprints_match or not mapping_integrity_passed):
        validation["status"] = "failed"
    corrected["corrected_concern_csv_sha256"] = validation["corrected_concern_csv_sha256"]
    corrected["corrected_concern_rows"] = len(normalized_rows)
    OUT_METRICS.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n")
    OUT_VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    OUT_REPORT.write_text("\n".join([
        "# Primary-300 concern-resolution proxy audit (MiniMax M3)", "",
        "Post-hoc, same-family automated audit of randomized A/B views of the frozen C0/C5 packets. This is not expert adjudication and cannot establish causal improvement in scientific quality.", "",
        f"- Coverage: {corrected['papers_verified']}/300 papers; {concern_total} extracted concerns (maximum six per paper).",
        f"- Model-labeled valid / invalid / uncertain concerns: {valid_total} / {invalid_total} / {uncertain_total}.",
        f"- Valid, repairable paired concerns: {len(micro_pairs)} across {len(macro_pairs)} papers.",
        f"- Paper-macro C5−C0 resolution proxy: {macro_mean:.4f} (paper-cluster bootstrap 95% CI {macro_ci}).",
        f"- Improved/tied/worsened papers: {improved}/{equal}/{worsened}.",
        f"- Corrected exact quote checks: {exact_quote_count}/{nonempty_quote_count} ({corrected['corrected_quote_exactness']['exact_quote_rate']:.3f}).",
        f"- Integrity findings: {stored_quote_flag_mismatches} saved quote flags disagreed after randomized A/B remapping; {len(output_constraint_violations)} generated quotes exceeded the requested 20-word maximum.",
        f"- Input integrity: source/review/supplement mismatches = {len(source_hash_mismatches)}/{len(review_hash_mismatches)}/{len(supplement_hash_mismatches)}.",
        f"- Per-concern A/B condition metadata mismatches: {len(condition_field_mismatches)}; paired outcomes were recomputed using the randomized condition order.",
        f"- API attempts: {len(events)}; recovered errors: {len(recovered)}; unrecovered errors: {len(unrecovered)}; reported tokens: {validation['api_total_reported_tokens']:,}.",
        "- No GPU used. Per-article rights remain unaudited; source materials are local only.",
        "- The row-level `corrected_concern_records.csv` maps raw A/B responses to C0/C5 using each paper's saved randomized order and recomputes quote provenance flags; it does not alter the raw runner state.",
        "- Corrected metrics are in `corrected_metrics.json`; paper-level summaries are in `corrected_paper_metrics.csv`; detailed integrity findings are in `integrity_validation.json`. The original `metrics.json` and `state.json` remain unchanged.", ""
    ]), encoding="utf-8")
    print(json.dumps({"validation_status": validation["status"],
                      "coverage": f"{len(state_ids & manifest_ids)}/300",
                      "macro_delta": macro_mean, "macro_ci95": macro_ci,
                      "quote_exactness": corrected["corrected_quote_exactness"],
                      "condition_field_mismatch_count": len(condition_field_mismatches),
                      "api_errors_recovered_unrecovered": [len(recovered), len(unrecovered)],
                      "source_review_supplement_mismatches": [len(source_hash_mismatches), len(review_hash_mismatches), len(supplement_hash_mismatches)]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
