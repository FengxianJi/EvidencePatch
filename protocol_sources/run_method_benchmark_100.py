#!/usr/bin/env python3
"""Run and blindly score every registered method on a fixed 100-paper cohort.

The run is resumable. API-backed methods and the blinded M3 evaluator share one
rate-limited MiniMax client; no GPU is used. Method names are hidden from the
evaluator behind randomized opaque packet keys.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from dataset.corpus import select_relevant_text
from diagnosis.protocol import PATCH_SYSTEM, REVIEW_SYSTEM, parse_json
from evaluation.statistics import bootstrap_mean_ci, exact_signed_rank_p, holm_adjust
from executor.minimax import MiniMaxClient
from method import get_method, list_methods
from baseline_method_adapter import run_baseline
from verifier.provenance import normalized_contains

COHORT = HERE / "results" / "re2_cohort_005_main120"
PRIOR_METHOD_RUN = HERE / "results" / "re2_method_v2_main120"
OUT = HERE / "results" / "baseline_methods_m3_100_v1"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CONFIG.update(experiment_id="evidencepatch-10-baselines-m3-100-v2-budgeted", min_seconds_between_requests=10.5)
STATE_PATH = OUT / "state.json"
USAGE_PATH = OUT / "api_usage.jsonl"
MANIFEST_PATH = OUT / "sample_manifest.csv"
MAX_PAPERS = 100
MAX_API_CALLS = 2400
MAX_REPORTED_TOKENS = 40_000_000
APPENDIX_CHAR_BUDGET = int(CONFIG.get("appendix_char_budget", 8000))
MAX_ATTEMPTS_PER_METHOD_ROUND = 2
PLACEBO = ("## Supplementary navigation\n\nThis note adds no factual claims, data, analyses, or results. "
           "Consult the manuscript for the study evidence.")
METHOD_IDS = list_methods()
CLIENT = MiniMaxClient(CONFIG, ROOT, USAGE_PATH)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rows() -> list[dict[str, Any]]:
    rows = json.loads((COHORT / "reviews.json").read_text(encoding="utf-8"))
    if len(rows) < MAX_PAPERS:
        raise RuntimeError(f"Expected at least {MAX_PAPERS} rows, found {len(rows)}")
    return rows[:MAX_PAPERS]


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("experiment_id") != CONFIG["experiment_id"]:
            if state.get("experiment_id") == "evidencepatch-10-baselines-m3-100-v1":
                state["superseded_experiment_id"] = state["experiment_id"]
                state["experiment_id"] = CONFIG["experiment_id"]
            else:
                raise RuntimeError("Checkpoint belongs to a different experiment")
        if state.get("method_ids") != METHOD_IDS:
            raise RuntimeError("Registered method list changed after the run started")
        return state
    return {"experiment_id": CONFIG["experiment_id"], "model": CONFIG["model"],
            "dataset_revision": json.loads((COHORT / "sample_manifest.json").read_text()).get("revision"),
            "methods": METHOD_IDS, "method_ids": METHOD_IDS, "papers": {},
            "created_at": datetime.now(timezone.utc).isoformat()}


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def api_events() -> list[dict[str, Any]]:
    if not USAGE_PATH.exists():
        return []
    result = []
    for line in USAGE_PATH.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def budgeted_complete(messages: list[dict[str, str]], purpose: str, case_id: str) -> str:
    events = api_events()
    total_tokens = sum(x.get("total_tokens", 0) or 0 for x in events)
    if len(events) >= MAX_API_CALLS or total_tokens >= MAX_REPORTED_TOKENS:
        raise RuntimeError(f"Run budget reached ({len(events)} calls, {total_tokens} tokens); checkpoint saved")
    return CLIENT.complete(messages, purpose, case_id)["content"]


def request_json(messages: list[dict[str, str]], purpose: str, case_id: str) -> dict[str, Any]:
    content = budgeted_complete(messages, purpose, case_id)
    try:
        return parse_json(content)
    except (ValueError, json.JSONDecodeError):
        repaired_text = [*messages, {"role": "assistant", "content": content[:12000]},
                         {"role": "user", "content": "Return the same answer as valid JSON matching the requested schema. Do not add or change claims."}]
        return parse_json(budgeted_complete(repaired_text, purpose + "_json_repair", case_id))


def call_model(*, system: str, user: str, purpose: str, paper_id: str) -> str:
    obj = request_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                       purpose, paper_id)
    return json.dumps(obj, ensure_ascii=False)


def score_packet(paper: str, appendix: str, packet_key: str) -> dict[str, Any]:
    prompt = f"""Blindly evaluate this manuscript packet. Packet key: {packet_key}
MANUSCRIPT (frozen):\n{paper}\n\nSUPPLEMENTARY MATERIAL:\n{appendix or '[No supplementary material supplied.]'}
Return JSON: {{"score":1,"strengths":["..."],"weaknesses":["..."],"confidence":0.0,"would_materially_change_recommendation":false}}.
Score from 1 to 5 based only on the evidence supplied. Do not reward length. Treat added text as unverified unless the packet supports it."""
    obj = request_json([{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": prompt}],
                       "baseline_method_blind_score", packet_key)
    score = obj.get("score")
    try:
        score = int(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Blind evaluator returned no integer 1-5 score") from exc
    if score not in range(1, 6):
        raise ValueError(f"Blind evaluator score out of range: {score}")
    obj["score"] = score
    return obj


def render_review_text(reviews: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"REVIEW {r['review_number']}:\n{r['review_text']}" for r in reviews)


def create_candidates(prior: dict[str, Any], manuscript: str) -> list[dict[str, Any]]:
    locator = prior.get("locator", {})
    entries = locator.get("verification", {}).get("selected_entries", [])
    candidates = []
    for index, entry in enumerate(entries):
        claim = str(entry.get("review_excerpt", "")).strip()
        source_quotes = entry.get("source_quotes", [])
        if not isinstance(source_quotes, list):
            source_quotes = [source_quotes]
        quote = next((str(q).strip() for q in source_quotes
                      if str(q).strip() and normalized_contains(manuscript, str(q).strip())), "")
        if not quote:
            continue
        cid = f"e{index + 1}"
        candidates.append({"candidate_id": cid, "claim": claim, "quote": quote,
            "response": "This exact passage is provided as a locator; it is not by itself a resolution.",
            "cost": float(max(1, len(claim) + len(quote) + 72)),
            "expected_benefit": 1.0, "uncertainty": 0.0, "risk": 0.0,
            "review_number": entry.get("review_number")})
    return candidates


def action_budget(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    costs = sorted(float(x["cost"]) for x in candidates)
    # A shared per-paper budget forces selectors to make choices when there
    # are multiple candidate actions, while allowing the cheapest action.
    return max(costs[0], 0.6 * sum(costs))


def provenance_summary(method_output: dict[str, Any], manuscript: str) -> dict[str, Any]:
    quotes: list[str] = []
    raw = method_output.get("raw") or {}
    entries = method_output.get("entries") or raw.get("entries") or []
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get("source_quotes", [])
            quotes.extend(str(x).strip() for x in value if str(x).strip())
    for concern in method_output.get("concerns", []) or raw.get("concerns", []) or []:
        if not isinstance(concern, dict):
            continue
        for key in ("source_quotes", "evidence_quote"):
            value = concern.get(key, [])
            if isinstance(value, str):
                value = [value]
            if isinstance(value, list):
                quotes.extend(str(x).strip() for x in value if str(x).strip())
    # Retrieval and budget-selector appendices render evidence as block quotes.
    for line in str(method_output.get("appendix_markdown", "")).splitlines():
        if line.lstrip().startswith(">"):
            quote = line.lstrip()[1:].strip()
            if quote:
                quotes.append(quote)
    return {"quoted_evidence_count": len(quotes),
            "exact_source_quote_count": sum(normalized_contains(manuscript, q) for q in quotes),
            "exact_source_quote_rate": (sum(normalized_contains(manuscript, q) for q in quotes) / len(quotes)) if quotes else None}


def bound_appendix(output: dict[str, Any]) -> dict[str, Any]:
    """Apply one fixed supplement budget to every method before blind scoring."""
    appendix = str(output.get("appendix_markdown", "")).strip()
    original_chars = output.get("uncapped_appendix_chars", len(appendix))
    output.setdefault("uncapped_appendix_chars", original_chars)
    if len(appendix) <= APPENDIX_CHAR_BUDGET:
        output.setdefault("appendix_truncated_to_budget", False)
        return output
    output.setdefault("uncapped_appendix_markdown", appendix)
    suffix = "\n\n[Text truncated at the shared appendix character budget.]"
    limit = max(0, APPENDIX_CHAR_BUDGET - len(suffix))
    head = appendix[:limit]
    # Prefer a paragraph boundary, then a sentence boundary, to avoid a partial
    # claim at the cut point. The hard character limit always takes priority.
    boundary = max(head.rfind("\n\n"), head.rfind(". "), head.rfind("。"))
    if boundary >= int(limit * 0.6):
        head = head[:boundary + (1 if head[boundary:boundary + 1] in ".。" else 0)].rstrip()
    output["appendix_markdown"] = (head + suffix)[:APPENDIX_CHAR_BUDGET]
    output["appendix_truncated_to_budget"] = True
    return output


def migrate_to_budgeted_scoring(state: dict[str, Any]) -> None:
    """Retain old outputs, but supersede scores made without a shared char cap."""
    for record in state.get("papers", {}).values():
        if record.get("scoring_protocol") == "shared_appendix_char_budget_v2":
            continue
        outputs = record.get("outputs", {})
        for output in outputs.values():
            bound_appendix(output)
        if record.get("scores"):
            record["scores_uncapped_superseded"] = record["scores"]
        record["scores"] = {}
        record["score_order"] = []
        record["packet_keys"] = {}
        record["processing_complete"] = False
        record["scoring_protocol"] = "shared_appendix_char_budget_v2"
        record["appendix_char_budget"] = APPENDIX_CHAR_BUDGET


def process_paper(row: dict[str, Any], state: dict[str, Any], prior_state: dict[str, Any]) -> None:
    pid = str(row["paper_id"])
    source_path = COHORT / "papers" / pid / "Initial_manuscript.md"
    source = source_path.read_text(encoding="utf-8", errors="replace")
    paper = select_relevant_text(source, CONFIG.get("max_paper_chars", 26000))
    reviews = row["reviews"]
    prior = prior_state.get(pid)
    if not prior or "locator" not in prior:
        raise RuntimeError(f"No frozen prior locator for {pid}; cannot define a shared candidate pool")
    source_sha, paper_sha = digest(source), digest(paper)
    review_sha = digest(render_review_text(reviews))
    record = state["papers"].setdefault(pid, {
        "venue_year": row.get("conference_year_track", ""), "source_sha256": source_sha,
        "main_text_sha256": paper_sha, "review_text_sha256": review_sha,
        "outputs": {}, "scores": {}, "score_order": [], "packet_keys": {},
        "scoring_protocol": "shared_appendix_char_budget_v2", "appendix_char_budget": APPENDIX_CHAR_BUDGET})
    if (record["source_sha256"], record["main_text_sha256"], record["review_text_sha256"]) != (source_sha, paper_sha, review_sha):
        raise RuntimeError(f"Frozen input hash changed for {pid}")
    candidates = record.setdefault("candidate_pool", create_candidates(prior, source))
    if record.get("candidate_pool") != candidates:
        raise RuntimeError(f"Candidate pool changed for {pid}")
    evidence_candidates = [{"claim": x["claim"], "quote": x["quote"], "response": x["response"]} for x in candidates]
    context = {"paper_id": pid, "manuscript": paper, "reviews": render_review_text(reviews),
        "claims": [{"review_number": x.get("review_number"), "claim": x["claim"]} for x in candidates],
        "evidence_candidates": evidence_candidates, "candidates": candidates,
        "budget": action_budget(candidates), "seed": 20261021 ^ sum(map(ord, pid)),
        "top_k": 3, "appendix_char_budget": APPENDIX_CHAR_BUDGET, "verification_feedback":
            "Audit all proposed quotations against the frozen manuscript. Keep only exact quotations; explicitly abstain from claims requiring new experiments or missing evidence.",
        "llm_call": call_model}

    for method_id in METHOD_IDS:
        if method_id in record["outputs"]:
            continue
        try:
            output = run_baseline(method_id, context)
            output = bound_appendix(output)
            appendix = str(output.get("appendix_markdown", "")).strip()
            record["outputs"][method_id] = {**output, "appendix_markdown": appendix,
                "provenance_audit": provenance_summary(output, source)}
            record.setdefault("method_errors", {}).pop(method_id, None)
            save_state(state)
        except Exception as exc:
            record.setdefault("method_errors", {})[method_id] = {
                "error_type": type(exc).__name__, "message": str(exc)[:500],
                "time": datetime.now(timezone.utc).isoformat()}
            save_state(state)
            raise

    if not record["score_order"]:
        order = list(METHOD_IDS)
        random.Random(20261021 ^ sum(map(ord, pid))).shuffle(order)
        record["score_order"] = order
        save_state(state)
    deferred_noop = []
    for method_id in record["score_order"]:
        if method_id in record["scores"]:
            continue
        appendix = record["outputs"][method_id].get("appendix_markdown", "")
        # Diagnostic-only verification has no appendix intervention. Empty
        # generated patches are no-ops; copy the exact frozen-original score.
        if method_id != "frozen_original" and not appendix:
            if "frozen_original" in record["scores"]:
                copied = dict(record["scores"]["frozen_original"])
                copied["no_patch_feasible_baseline_copy"] = True
                copied["score_reused_by_design"] = True
                record["scores"][method_id] = copied
            else:
                deferred_noop.append(method_id)
            save_state(state)
            continue
        opaque = hashlib.sha256(f"{CONFIG['experiment_id']}|{pid}|{method_id}|20261021".encode()).hexdigest()[:16]
        record["packet_keys"][method_id] = opaque
        record["scores"][method_id] = score_packet(paper, appendix, opaque)
        record["scores"][method_id]["opaque_packet_key"] = opaque
        save_state(state)
    if deferred_noop:
        if "frozen_original" not in record["scores"]:
            opaque = hashlib.sha256(f"{CONFIG['experiment_id']}|{pid}|frozen_original|20261021".encode()).hexdigest()[:16]
            record["packet_keys"]["frozen_original"] = opaque
            record["scores"]["frozen_original"] = score_packet(paper, "", opaque)
        for method_id in deferred_noop:
            copied = dict(record["scores"]["frozen_original"])
            copied["no_patch_feasible_baseline_copy"] = True
            copied["score_reused_by_design"] = True
            record["scores"][method_id] = copied
        save_state(state)
    record["processing_complete"] = all(mid in record["scores"] for mid in METHOD_IDS)
    save_state(state)


def mean_ci(values: list[int], seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "bootstrap_ci95": None,
                "signed_rank": exact_signed_rank_p([])}
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "bootstrap_ci95": bootstrap_mean_ci(values, seed=seed),
            "signed_rank": exact_signed_rank_p(values)}


def summarize(state: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_rows = []
    for row in rows:
        pid = str(row["paper_id"])
        rec = state["papers"].get(pid, {})
        source = (COHORT / "papers" / pid / "Initial_manuscript.md").read_text(encoding="utf-8", errors="replace")
        scores = rec.get("scores", {})
        base = scores.get("frozen_original", {}).get("score")
        out = {"paper_id": pid, "venue_year": rec.get("venue_year"),
               "source_sha256": rec.get("source_sha256"), "main_text_sha256": rec.get("main_text_sha256"),
               "shared_candidate_count": len(rec.get("candidate_pool", [])),
               "estimated_action_budget": action_budget(rec.get("candidate_pool", []))}
        for method_id in METHOD_IDS:
            score = scores.get(method_id, {}).get("score")
            out[method_id + "_score"] = score
            out[method_id + "_delta"] = score - base if score is not None and base is not None else None
            method_output = rec.get("outputs", {}).get(method_id, {})
            audit = provenance_summary(method_output, source)
            out[method_id + "_appendix_chars"] = len(method_output.get("appendix_markdown", ""))
            out[method_id + "_exact_quote_rate"] = audit.get("exact_source_quote_rate")
            out[method_id + "_action_count"] = len(method_output.get("actions", []))
        output_rows.append(out)
    if output_rows:
        with (OUT / "paper_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
            writer.writeheader(); writer.writerows(output_rows)
    by_method = {}
    complete_papers = sum(bool(state["papers"].get(str(x["paper_id"]), {}).get("processing_complete")) for x in rows)
    for i, mid in enumerate(METHOD_IDS):
        valid = [r for r in output_rows if r.get(mid + "_score") is not None and r.get("frozen_original_score") is not None]
        deltas = [int(r[mid + "_delta"]) for r in valid]
        outputs = [state["papers"].get(str(r["paper_id"]), {}).get("outputs", {}).get(mid, {}) for r in valid]
        quote_rates = [r[mid + "_exact_quote_rate"] for r in valid if r.get(mid + "_exact_quote_rate") is not None]
        by_method[mid] = {"paired_score_metrics": mean_ci(deltas, 92000 + i),
            "scored_papers": len(valid), "nonempty_patches": sum(bool(x.get("appendix_markdown", "").strip()) for x in outputs),
            "mean_patch_chars": statistics.mean([len(x.get("appendix_markdown", "")) for x in outputs]) if outputs else None,
            "exact_quote_rates_available": len(quote_rates),
            "mean_exact_quote_rate": statistics.mean(quote_rates) if quote_rates else None,
            "no_op_scores_reused": sum(bool(state["papers"].get(str(r["paper_id"]), {}).get("scores", {}).get(mid, {}).get("score_reused_by_design")) for r in valid)}
        if mid == "peerify_verify":
            verifications = [v for output in outputs for v in (output.get("verifications") or []) if isinstance(v, dict)]
            by_method[mid]["verification_instances"] = len(verifications)
            by_method[mid]["status_counts"] = {status: sum(v.get("status") == status for v in verifications)
                for status in ("supported", "partly_supported", "not_supported", "not_determined")}
            by_method[mid]["verification_accuracy"] = None  # No expert claim-level gold labels in this cohort.
    by_method["frozen_original"]["holm_adjusted_p"] = None
    p_values = {mid: by_method[mid]["paired_score_metrics"]["signed_rank"]["p_two_sided"]
                for mid in METHOD_IDS if mid != "frozen_original"
                and by_method[mid]["scored_papers"] > 0}
    adjusted = holm_adjust(p_values)
    for mid in METHOD_IDS:
        if mid != "frozen_original":
            by_method[mid]["holm_adjusted_p"] = adjusted.get(mid)
    events = api_events()
    token_values = [x.get("total_tokens") for x in events if isinstance(x.get("total_tokens"), int)]
    metrics = {"experiment_id": CONFIG["experiment_id"], "status": "complete" if complete_papers == MAX_PAPERS else "in_progress",
        "dataset": str(COHORT.relative_to(HERE)), "dataset_revision": state["dataset_revision"],
        "requested_papers": MAX_PAPERS, "papers_in_state": len(state["papers"]),
        "papers_with_all_methods_scored": complete_papers, "method_ids": METHOD_IDS,
        "model": CONFIG["model"], "scoring_repeats": 1, "reviewer_model_families": [CONFIG["model"]],
        "condition_order_randomized_per_paper": True, "method_names_sent_to_blind_scorer": False,
        "planner_pool_source": "frozen, exact-quote-verified selected entries from prior v2 locator output; identical pool shared by the four selectors",
        "appendix_char_budget": APPENDIX_CHAR_BUDGET,
        "planner_budget": "per-paper 60% of summed verified action costs, floored at cheapest single action",
        "methods": by_method, "api_calls_logged": len(events), "api_total_tokens": sum(token_values) if token_values else None,
        "failed_api_attempts": sum(x.get("status") == "error" for x in events),
        "api_call_cap": MAX_API_CALLS, "reported_token_cap": MAX_REPORTED_TOKENS,
        "expert_resolution_labels": "not available; M3 score is a proxy",
        "independent_model_replication": "not run; only MiniMax-M3 requested",
        "interpretation": "Exploratory single-judge baseline benchmark; no causal or expert-confirmed resolution claim."}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# Ten-method MiniMax M3 benchmark (100 papers)", "",
        f"- Status: **{metrics['status']}**; {complete_papers}/{MAX_PAPERS} papers have every method scored.",
        f"- Dataset revision: `{state['dataset_revision']}`; first 100 rows of the frozen 120-paper Re² main cohort.",
        f"- API use: {len(events)} logged attempts; {metrics['api_total_tokens']} reported tokens.",
        "- Method names were omitted from the scoring prompt; score order was deterministically randomized per paper.",
        "- Empty patches and diagnostic-only verification copy the frozen-original score by design.",
        "- All planner selectors share one hash-checked, exact-quote-verified candidate pool and the same per-paper budget.",
        "- Scores are one MiniMax-M3 judge pass. Expert adjudication and cross-family replication are not included.", "",
        "## Paired mean score change versus frozen original", "", "| Method | n | Mean Δ | Median Δ | 95% bootstrap CI | Holm p | Nonempty patches |", "|---|---:|---:|---:|---|---:|---:|"]
    for mid, result in by_method.items():
        st = result["paired_score_metrics"]
        report.append(f"| `{mid}` | {st['n']} | {st['mean']} | {st['median']} | {st['bootstrap_ci95']} | {result.get('holm_adjusted_p')} | {result['nonempty_patches']} |")
    (OUT / "statistical_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def write_manifest(rows: list[dict[str, Any]]) -> None:
    prior_state = json.loads((PRIOR_METHOD_RUN / "state.json").read_text(encoding="utf-8"))["papers"]
    fields = ["paper_id", "venue_year", "source_sha256", "main_text_sha256", "review_text_sha256", "prior_locator_hash"]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            pid = str(row["paper_id"])
            source = (COHORT / "papers" / pid / "Initial_manuscript.md").read_text(encoding="utf-8", errors="replace")
            paper = select_relevant_text(source, CONFIG.get("max_paper_chars", 26000))
            reviews = render_review_text(row["reviews"])
            prior = prior_state[pid]
            writer.writerow({"paper_id": pid, "venue_year": row.get("conference_year_track", ""),
                "source_sha256": digest(source), "main_text_sha256": digest(paper),
                "review_text_sha256": digest(reviews),
                "prior_locator_hash": digest(json.dumps(prior["locator"], ensure_ascii=False, sort_keys=True))})


def write_run_manifest(state: dict[str, Any]) -> None:
    method_hashes = {}
    for mid in METHOD_IDS:
        source = ROOT / "method" / "methods" / mid / "method.py"
        method_hashes[mid] = digest(source.read_text(encoding="utf-8"))
    files = [Path(__file__), ROOT / "method" / "common.py", ROOT / "method" / "registry.py",
             HERE / "baseline_method_adapter.py", COHORT / "sample_manifest.json"]
    (OUT / "run_manifest.json").write_text(json.dumps({
        "experiment_id": CONFIG["experiment_id"], "run_protocol": "shared_appendix_char_budget_v2",
        "model": CONFIG["model"], "endpoint": CONFIG["api_endpoint"],
        "requested_papers": MAX_PAPERS, "cohort_selection": "first 100 rows of frozen Re2 Main-120 manifest",
        "method_ids": METHOD_IDS, "appendix_char_budget": APPENDIX_CHAR_BUDGET,
        "blind_score_order_randomized_per_paper": True, "method_names_disclosed_to_scorer": False,
        "score_repeats": 1, "planner_candidate_pool": "frozen exact-source-verified v2 locator entries",
        "planner_budget_rule": "60% of summed verified action cost, floored at cheapest single action",
        "superseded_uncapped_score_count": sum(len(r.get("scores_uncapped_superseded", {})) for r in state["papers"].values()),
        "superseded_prior_run_id": state.get("superseded_experiment_id"),
        "method_source_sha256": method_hashes,
        "protocol_file_sha256": {str(path.relative_to(ROOT)): digest(path.read_text(encoding="utf-8")) for path in files},
        "gpu_used": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    prior_state = json.loads((PRIOR_METHOD_RUN / "state.json").read_text(encoding="utf-8"))["papers"]
    if len(prior_state) < 120:
        raise RuntimeError("The frozen v2 method run is incomplete; refuses to fabricate a planner candidate pool")
    write_manifest(rows)
    state = load_state()
    migrate_to_budgeted_scoring(state)
    save_state(state)
    write_run_manifest(state)
    for round_number in range(1, MAX_ATTEMPTS_PER_METHOD_ROUND + 1):
        errors = 0
        for index, row in enumerate(rows, start=1):
            pid = str(row["paper_id"])
            rec = state["papers"].get(pid, {})
            if rec.get("processing_complete"):
                continue
            try:
                process_paper(row, state, prior_state)
            except Exception as exc:
                errors += 1
                print(json.dumps({"event": "paper_error", "round": round_number, "index": index,
                    "paper_id": pid, "error_type": type(exc).__name__, "message": str(exc)[:300]}, ensure_ascii=False), flush=True)
                if "budget reached" in str(exc):
                    break
            if index % 5 == 0:
                complete = sum(bool(state["papers"].get(str(x["paper_id"]), {}).get("processing_complete")) for x in rows)
                events = api_events()
                print(json.dumps({"event": "progress", "round": round_number, "paper_index": index,
                    "papers_complete": complete, "api_calls": len(events),
                    "tokens": sum(x.get("total_tokens", 0) or 0 for x in events)}, ensure_ascii=False), flush=True)
        metrics = summarize(state, rows)
        if metrics["status"] == "complete":
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            return
        if errors == 0:
            break
        save_state(state)
        time.sleep(3)
    print(json.dumps(summarize(state, rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
