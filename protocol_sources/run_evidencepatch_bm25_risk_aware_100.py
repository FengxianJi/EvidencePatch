#!/usr/bin/env python3
"""Exploratory M3 run: deterministic BM25 candidates plus risk-aware planning.

This separate 100-paper Main-120 development run tests whether local exact-span
retrieval addresses candidate provenance failures in the frozen v2 locator.
It does not alter prior method runs or the confirmatory 300-paper cohort.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import run_method_benchmark_100 as benchmark
from diagnosis.evidence_retrieval import BM25Index, split_passages
from diagnosis.protocol import REVIEW_SYSTEM, parse_json
from evaluation.statistics import bootstrap_mean_ci, exact_signed_rank_p
from executor.minimax import MiniMaxClient
from method.methods.evidencepatch_bm25_risk_aware.method import run as run_method
from verifier.provenance import normalized_contains

OUT = HERE / "results" / "evidencepatch_bm25_riskaware_m3_100_v1"
OUT.mkdir(parents=True, exist_ok=True)
STATE_PATH, USAGE_PATH = OUT / "state.json", OUT / "api_usage.jsonl"
METRICS_PATH, MANIFEST_PATH = OUT / "metrics.json", OUT / "run_manifest.json"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CONFIG.update(experiment_id="evidencepatch-bm25-riskaware-m3-100-v1",
              min_seconds_between_requests=10.5,
              credential_indices=[2, 3])
CLIENT = MiniMaxClient(CONFIG, ROOT, USAGE_PATH)
COHORT = HERE / "results" / "re2_cohort_005_main120"
METHOD_ID = "evidencepatch_bm25_risk_aware"
N_PAPERS, MAX_ATTEMPTS, MAX_TOKENS = 100, 900, 28_000_000
TOP_K_PER_REVIEW = 3
MAX_SOURCE_WORDS = 25
MAX_REVIEW_EXCERPT_WORDS = 25
MAX_REVIEW_CONTEXT_CHARS = 2500
APPENDIX_CHAR_BUDGET = 8000


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events() -> list[dict]:
    if not USAGE_PATH.exists():
        return []
    return [json.loads(line) for line in USAGE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def exhausted_credential_indices() -> set[int]:
    exhausted = set()
    for event in events():
        meta = event.get("response_metadata", {})
        # MiniMax reports plan depletion as either 2056/2067 and a depleted
        # pay-as-you-go balance as 1008. All three make a credential unusable
        # for this run until its allowance is restored.
        code = meta.get("base_status_code", event.get("http_status"))
        if event.get("status") == "error" and code in (2056, 2067, 1008, "2056", "2067", "1008"):
            try:
                exhausted.add(int(event["credential_index"]) - 1)
            except (KeyError, TypeError, ValueError):
                continue
    return exhausted


def allowed_credential_indices() -> list[int]:
    """Return the configured 1-based credential allowlist for this run."""
    key_path = ROOT / CONFIG["key_file"]
    credential_count = sum(bool(line.strip()) for line in key_path.read_text(encoding="utf-8").splitlines())
    indices = sorted({int(index) for index in CONFIG.get("credential_indices", range(1, credential_count + 1))})
    if not indices or any(index < 1 or index > credential_count for index in indices):
        raise RuntimeError("Configured MiniMax credential_indices are invalid")
    return indices


def activate_available_credential() -> None:
    allowed = allowed_credential_indices()
    # Historical exhaustion is not permanent: Token Plan limits can reset and
    # prepaid balances can be replenished between resumptions. Let the client
    # probe the configured keys on each resumed request; it records fresh
    # provider errors and short-circuits after all keys fail for that request.
    if CLIENT.active_key_index >= len(allowed):
        CLIENT.active_key_index = 0


def complete(messages: list[dict[str, str]], purpose: str, paper_id: str) -> str:
    logged = events()
    tokens = sum(int(item.get("total_tokens") or 0) for item in logged)
    if len(logged) >= MAX_ATTEMPTS or tokens >= MAX_TOKENS:
        raise RuntimeError(f"Run budget reached: attempts={len(logged)}, tokens={tokens}")
    activate_available_credential()
    return CLIENT.complete(messages, purpose, paper_id)["content"]


def llm_call(*, system: str, user: str, purpose: str, paper_id: str) -> str:
    raw = complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                   purpose, paper_id)
    return json.dumps(parse_json(raw), ensure_ascii=False)


def score(manuscript: str, appendix: str, packet_key: str, paper_id: str) -> dict:
    prompt = f"""Blindly evaluate this manuscript packet. Packet key: {packet_key}
MANUSCRIPT (frozen):\n{manuscript}\n\nSUPPLEMENTARY MATERIAL:\n{appendix or '[No supplementary material supplied.]'}
Return JSON: {{"score":1,"strengths":["..."],"weaknesses":["..."],"confidence":0.0,"would_materially_change_recommendation":false}}.
Score from 1 to 5 based only on evidence supplied. Do not reward length. Treat added text as unverified unless the packet supports it."""
    obj = parse_json(complete([{"role": "system", "content": REVIEW_SYSTEM},
                               {"role": "user", "content": prompt}],
                              "evidencepatch_bm25_blind_score", paper_id))
    value = int(obj.get("score"))
    if value not in range(1, 6):
        raise ValueError(f"Invalid score for {paper_id}")
    obj["score"] = value
    return obj


def build_candidate_pool(reviews: list[dict], source: str) -> tuple[list[dict], dict]:
    source_passages = [p for p in split_passages(source, max_words=MAX_SOURCE_WORDS,
                                                 overlap_sentences=0)
                       if 0 < len(p.text.split()) <= MAX_SOURCE_WORDS]
    source_index = BM25Index(source_passages)
    candidates: list[dict] = []
    rejected = {"no_source_passage_hit": 0, "no_review_excerpt_hit": 0,
                "duplicate_source_quote": 0, "invalid_source_offset": 0}
    source_quote_seen: set[tuple[int, str]] = set()
    spans_checked = 0
    for review in reviews:
        review_no = int(review["review_number"])
        review_text = str(review["review_text"])
        review_passages = [p for p in split_passages(review_text, max_words=MAX_REVIEW_EXCERPT_WORDS,
                                                     overlap_sentences=0)
                           if 0 < len(p.text.split()) <= MAX_REVIEW_EXCERPT_WORDS]
        if not review_passages:
            rejected["no_review_excerpt_hit"] += 1
            continue
        review_index = BM25Index(review_passages)
        hits = source_index.search(review_text, TOP_K_PER_REVIEW)
        if not hits:
            rejected["no_source_passage_hit"] += 1
            continue
        for rank, (source_passage, score_value) in enumerate(hits, 1):
            quote = source_passage.text.strip()
            if len(quote.split()) > MAX_SOURCE_WORDS:
                rejected["invalid_source_offset"] += 1
                continue
            start, end = source_passage.start_char, source_passage.end_char
            if source[start:end].strip() != quote or not normalized_contains(source, quote):
                rejected["invalid_source_offset"] += 1
                continue
            spans_checked += 1
            key = (review_no, " ".join(quote.split()).casefold())
            if key in source_quote_seen:
                rejected["duplicate_source_quote"] += 1
                continue
            source_quote_seen.add(key)
            review_hits = review_index.search(quote, 1)
            if not review_hits:
                rejected["no_review_excerpt_hit"] += 1
                continue
            review_passage, review_score = review_hits[0]
            excerpt = review_passage.text.strip()
            rs, re = review_passage.start_char, review_passage.end_char
            if (len(excerpt.split()) > MAX_REVIEW_EXCERPT_WORDS
                    or review_text[rs:re].strip() != excerpt
                    or not normalized_contains(review_text, excerpt)):
                rejected["invalid_source_offset"] += 1
                continue
            cid = f"r{review_no}_p{rank}"
            candidates.append({
                "candidate_id": cid,
                "review_number": review_no,
                "claim": excerpt,
                "review_context": review_text[:MAX_REVIEW_CONTEXT_CHARS],
                "quote": quote,
                "section": source_passage.section,
                "cost": float(max(1, len(excerpt) + len(quote) + 72)),
                "expected_benefit": 1.0,
                "uncertainty": 0.0,
                "risk": 0.0,
                "bm25_score": float(score_value),
                "review_excerpt_bm25_score": float(review_score),
                "source_start_char": start,
                "source_end_char": end,
                "review_start_char": rs,
                "review_end_char": re,
                "source_quote_sha256": digest(quote),
                "review_excerpt_sha256": digest(excerpt),
            })
    meta = {"source_passages": len(source_passages), "source_spans_checked": spans_checked,
            "candidate_count": len(candidates), "candidates_by_review": {
                str(n): sum(c["review_number"] == n for c in candidates)
                for n in sorted({int(r["review_number"]) for r in reviews})},
            "rejected_counts": rejected}
    return candidates, meta


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("experiment_id") != CONFIG["experiment_id"]:
            raise RuntimeError("Checkpoint belongs to another experiment")
        return state
    revision = json.loads((COHORT / "sample_manifest.json").read_text(encoding="utf-8")).get("revision")
    return {"experiment_id": CONFIG["experiment_id"], "dataset_revision": revision,
            "model": CONFIG["model"], "papers": {}, "created_at": datetime.now(timezone.utc).isoformat()}


def save_state(state: dict) -> None:
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATE_PATH)


def process(row: dict, state: dict, planner) -> None:
    pid = str(row["paper_id"])
    record = state["papers"].setdefault(pid, {"scores": {}, "complete": False})
    source_path = COHORT / "papers" / pid / "Initial_manuscript.md"
    source = source_path.read_text(encoding="utf-8", errors="replace")
    manuscript = benchmark.select_relevant_text(source, CONFIG.get("max_paper_chars", 26000))
    review_text = benchmark.render_review_text(row["reviews"])
    hashes = {"source_sha256": digest(source), "main_text_sha256": digest(manuscript),
              "review_text_sha256": digest(review_text)}
    if record.get("hashes") not in (None, hashes):
        raise RuntimeError(f"Frozen input hash mismatch: {pid}")
    record["hashes"] = hashes
    if "candidates" not in record:
        candidates, retrieval_meta = build_candidate_pool(row["reviews"], source)
        record["candidates"] = candidates
        record["retrieval_meta"] = retrieval_meta
        save_state(state)
    else:
        candidates = record["candidates"]
    review_by_number = {int(item["review_number"]): str(item["review_text"])
                        for item in row["reviews"]}
    for c in candidates:
        if (not normalized_contains(source, str(c["quote"]))
                or int(c["review_number"]) not in review_by_number
                or not normalized_contains(review_by_number[int(c["review_number"])], str(c["claim"]))):
            raise RuntimeError(f"Reconstructed candidate span failed source check: {pid}/{c['candidate_id']}")

    if "output" not in record:
        context = {"paper_id": pid, "manuscript": manuscript, "reviews": review_text,
                   "candidates": candidates, "budget": benchmark.action_budget(candidates),
                   "seed": 20261024 ^ sum(map(ord, pid)), "planner_api": planner,
                   "llm_call": llm_call}
        output = run_method(context)
        appendix = str(output.get("appendix_markdown", "")).strip()
        if len(appendix) > APPENDIX_CHAR_BUDGET:
            raise ValueError(f"Appendix character budget exceeded: {pid}")
        action_set = set(map(str, output.get("actions", [])))
        by_id = {str(c["candidate_id"]): c for c in candidates}
        if not action_set.issubset(by_id):
            raise ValueError(f"Unknown selected candidate ID: {pid}")
        for cid in action_set:
            c = by_id[cid]
            review = review_by_number[int(c["review_number"])]
            if not normalized_contains(source, str(c["quote"])) or not normalized_contains(review, str(c["claim"])):
                raise ValueError(f"Selected pointer failed exact-source audit: {pid}/{cid}")
        record["output"] = {**output, "appendix_markdown": appendix}
        save_state(state)

    if not record.get("score_order"):
        order = ["frozen_original", METHOD_ID]
        random.Random(20261024 ^ sum(map(ord, pid))).shuffle(order)
        record["score_order"] = order
        save_state(state)
    for condition in record["score_order"]:
        if condition in record["scores"]:
            continue
        appendix = "" if condition == "frozen_original" else record["output"]["appendix_markdown"]
        key = hashlib.sha256(f"{CONFIG['experiment_id']}|{pid}|{condition}".encode()).hexdigest()[:16]
        record["scores"][condition] = score(manuscript, appendix, key, pid)
        record["scores"][condition]["opaque_packet_key"] = key
        save_state(state)
    record["complete"] = set(record["scores"]) == {"frozen_original", METHOD_ID}
    save_state(state)


def summarize(state: dict, rows: list[dict]) -> dict:
    records = [state["papers"].get(str(row["paper_id"]), {}) for row in rows]
    complete_records = [r for r in records if r.get("complete")]
    deltas = [r["scores"][METHOD_ID]["score"] - r["scores"]["frozen_original"]["score"]
              for r in complete_records]
    usage = events()
    candidates = [c for r in records for c in r.get("candidates", [])]
    selected = [cid for r in records for cid in r.get("output", {}).get("actions", [])]
    metric = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete" if len(complete_records) == len(rows) else "in_progress",
        "model": CONFIG["model"], "endpoint": CONFIG["api_endpoint"], "gpu_used": False,
        "credential_indices": allowed_credential_indices(),
        "requested_papers": len(rows), "completed_papers": len(complete_records),
        "method_id": METHOD_ID, "conditions": ["frozen_original", METHOD_ID],
        "method_output_count": sum("output" in r for r in records),
        "candidate_count": len(candidates),
        "candidate_papers": sum(bool(r.get("candidates")) for r in records),
        "selected_action_count": len(selected),
        "selected_papers": sum(bool(r.get("output", {}).get("actions")) for r in records),
        "source_quote_exact_rate": sum(bool(c.get("quote")) for c in candidates) / len(candidates) if candidates else None,
        "retrieval": {"kind": "deterministic BM25 over exact source passages",
                      "top_k_per_review": TOP_K_PER_REVIEW,
                      "max_source_words": MAX_SOURCE_WORDS,
                      "max_review_excerpt_words": MAX_REVIEW_EXCERPT_WORDS,
                      "source_spans_checked": sum(r.get("retrieval_meta", {}).get("source_spans_checked", 0)
                                                   for r in records),
                      "rejected_counts": {key: sum(r.get("retrieval_meta", {}).get("rejected_counts", {}).get(key, 0)
                                                   for r in records)
                                          for key in ("no_source_passage_hit", "no_review_excerpt_hit",
                                                      "duplicate_source_quote", "invalid_source_offset")}},
        "effect_vs_frozen_original": {
            "n": len(deltas), "mean": statistics.mean(deltas) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "bootstrap_ci95": bootstrap_mean_ci(deltas, seed=10242026) if deltas else None,
            "signed_rank_p": exact_signed_rank_p(deltas),
            "positive": sum(x > 0 for x in deltas), "unchanged": sum(x == 0 for x in deltas),
            "negative": sum(x < 0 for x in deltas)},
        "api_attempts": len(usage), "api_tokens": sum(int(x.get("total_tokens") or 0) for x in usage),
        "api_errors": sum(x.get("status") == "error" for x in usage),
        "api_attempt_cap": MAX_ATTEMPTS, "api_token_cap": MAX_TOKENS,
        "interpretation": "Exploratory same-family M3 score proxy on the frozen development cohort; no expert labels.",
    }
    METRICS_PATH.write_text(json.dumps(metric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metric


def write_manifest(state: dict, rows: list[dict]) -> None:
    method_path = ROOT / "method/methods/evidencepatch_bm25_risk_aware/method.py"
    paths = [Path(__file__), method_path, ROOT / "method/common.py",
             HERE / "executor/minimax.py",
             HERE / "run_method_benchmark_100.py", HERE / "diagnosis/evidence_retrieval.py",
             HERE / "planner/budget.py",
             HERE / "evaluation/validate_evidencepatch_bm25_risk_aware_100.py",
             COHORT / "sample_manifest.json"]
    n_complete = sum(bool(state.get("papers", {}).get(str(r["paper_id"]), {}).get("complete")) for r in rows)
    manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete" if n_complete == len(rows) else "running",
        "model": CONFIG["model"], "endpoint": CONFIG["api_endpoint"], "gpu_used": False,
        "credential_indices": allowed_credential_indices(),
        "requested_papers": len(rows), "cohort": "first 100 records of frozen Re2 Main-120 development cohort",
        "dataset_revision": state.get("dataset_revision"),
        "conditions": ["frozen_original", METHOD_ID],
        "generation_method": "local BM25 over exact passages; MiniMax-M3 risk estimator; greedy budget planner",
        "top_k_per_review": TOP_K_PER_REVIEW, "max_source_words": MAX_SOURCE_WORDS,
        "max_review_excerpt_words": MAX_REVIEW_EXCERPT_WORDS,
        "review_context_chars": MAX_REVIEW_CONTEXT_CHARS,
        "appendix_char_budget": APPENDIX_CHAR_BUDGET,
        "score_order_randomized": True, "method_name_hidden_from_scorer": True,
        "independent_expert_labels": False,
        "protocol_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in paths},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows = benchmark.load_rows()
    if len(rows) != N_PAPERS:
        raise RuntimeError(f"Expected exactly {N_PAPERS} papers, found {len(rows)}")
    planner_path = HERE / "planner/budget.py"
    spec = importlib.util.spec_from_file_location("evidencepatch_bm25_budget_planner", planner_path)
    planner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = planner
    spec.loader.exec_module(planner)
    state = load_state()
    write_manifest(state, rows)
    for index, row in enumerate(rows, 1):
        if state.get("papers", {}).get(str(row["paper_id"]), {}).get("complete"):
            continue
        try:
            process(row, state, planner)
        except Exception as exc:
            print(json.dumps({"event": "paper_error", "index": index,
                              "paper_id": str(row["paper_id"]),
                              "error_type": type(exc).__name__, "message": str(exc)[:300]},
                             ensure_ascii=False), flush=True)
            save_state(state)
            metrics = summarize(state, rows)
            write_manifest(state, rows)
            allowed = allowed_credential_indices()
            key_count = len(allowed)
            exhausted = exhausted_credential_indices().intersection(index - 1 for index in allowed)
            if key_count and len(exhausted) >= key_count:
                exhausted_events = [event for event in events()
                                    if event.get("status") == "error"
                                    and int(event.get("credential_index", 0)) - 1 in exhausted]
                failure_codes = {}
                for event in exhausted_events:
                    code = event.get("response_metadata", {}).get("base_status_code", event.get("http_status"))
                    if code is not None:
                        failure_codes.setdefault(str(event["credential_index"]), set()).add(str(code))
                metrics["status"] = "incomplete_api_credentials"
                metrics["blocker"] = {"reason": "all_configured_credentials_reported_quota_or_balance_exhaustion",
                    "provider_status_codes_by_credential": {key: sorted(value) for key, value in failure_codes.items()},
                    "exhausted_credential_indices": sorted(i + 1 for i in exhausted),
                    "configured_credential_indices": allowed,
                    "configured_credential_count": key_count,
                    "last_failed_request": {"paper_id": str(row["paper_id"]),
                                            "purpose": events()[-1].get("purpose") if events() else None}}
                METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                manifest["status"] = metrics["status"]
                manifest["blocker"] = metrics["blocker"]
                MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        done = sum(bool(state["papers"].get(str(r["paper_id"]), {}).get("complete")) for r in rows)
        if done % 5 == 0:
            print(json.dumps({"event": "paper_complete", "completed": done, "total": len(rows),
                              "api_attempts": len(events()), "api_tokens": sum(int(x.get("total_tokens") or 0) for x in events())}), flush=True)
    metrics = summarize(state, rows)
    write_manifest(state, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
