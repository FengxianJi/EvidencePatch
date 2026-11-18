from __future__ import annotations

from method.common import call_model, parse_json_object, result


SYSTEM = """You are the diagnosis and evidence-value estimator in EvidencePatch. Use the exact manuscript quote and its retrieved source section as evidence. Abstain when the quote does not address the review concern. Never invent evidence or claim that a pointer proves a result."""


def run(context):
    candidates = context.get("candidates", [])
    if not candidates:
        return result("evidencepatch_section_context", "", actions=[], ratings=[], calls=0,
                      interpretation="No exact-source candidate was available; abstain.")
    compact = []
    for candidate in candidates:
        retrieved = candidate.get("retrieved_context", [])
        compact.append({
            "candidate_id": str(candidate["candidate_id"]),
            "review_concern": str(candidate.get("claim", "")),
            "exact_manuscript_quote": str(candidate.get("quote", "")),
            "retrieved_source_sections": [str(x) for x in retrieved],
        })
    prompt = f"""Assess each candidate independently for a minimal evidence-pointer appendix. The quoted passage is an exact substring of the frozen manuscript. The retrieved sections are additional context selected from that same frozen manuscript; they are not independent evidence. Decide whether the quote directly helps a reviewer check the concern. A topic mention alone is insufficient. Do not propose experiments or additional facts.

For every candidate return: candidate_id, decision ('select' or 'abstain'), p_concern_valid, p_quote_relevant, uncertainty, risk, and a short pointer_note. Probabilities must be numbers from 0 to 1. p_concern_valid estimates whether the concern is a real, checkable issue. p_quote_relevant estimates whether the exact quote directly helps inspect it in the context of the retrieved sections. uncertainty represents uncertainty in those estimates. risk represents the chance that including the pointer could mislead, overstate resolution, or distract. The note may only guide inspection and must not claim resolution. Return all candidates exactly once as JSON.

Candidates:
{compact}

Return JSON only: {{"ratings":[{{"candidate_id":"e1","decision":"select","p_concern_valid":0.0,"p_quote_relevant":0.0,"uncertainty":0.0,"risk":0.0,"pointer_note":"..."}}]}}."""
    raw = call_model(context, SYSTEM, prompt, "evidencepatch_section_context_risk_estimation")
    obj = parse_json_object(raw)
    by_id = {str(c["candidate_id"]): c for c in candidates}
    raw_ratings = obj.get("ratings")
    if not isinstance(raw_ratings, list):
        raise ValueError("EvidencePatch ratings must be a list")

    def prob(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    ratings = {}
    for item in raw_ratings:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", ""))
        if cid not in by_id or cid in ratings:
            continue
        valid, relevant = prob(item.get("p_concern_valid")), prob(item.get("p_quote_relevant"))
        ratings[cid] = {"candidate_id": cid,
                        "decision": "select" if item.get("decision") == "select" else "abstain",
                        "p_concern_valid": valid, "p_quote_relevant": relevant,
                        "expected_benefit": valid * relevant,
                        "uncertainty": prob(item.get("uncertainty")),
                        "risk": prob(item.get("risk")),
                        "pointer_note": str(item.get("pointer_note", ""))[:280]}
    for cid in by_id:
        ratings.setdefault(cid, {"candidate_id": cid, "decision": "abstain",
                                 "p_concern_valid": 0.0, "p_quote_relevant": 0.0,
                                 "expected_benefit": 0.0, "uncertainty": 1.0,
                                 "risk": 1.0, "pointer_note": ""})

    planner_api = context.get("planner_api")
    if planner_api is None:
        raise RuntimeError("EvidencePatch requires the audited budget planner")
    scored = []
    for cid, candidate in by_id.items():
        rating = ratings[cid]
        benefit = rating["expected_benefit"] if rating["decision"] == "select" else 0.0
        scored.append(planner_api.PatchCandidate(candidate_id=cid,
                       cost=float(candidate.get("cost", 0.0)), expected_benefit=0.0,
                       uncertainty=rating["uncertainty"], risk=rating["risk"],
                       claim_benefits={cid: benefit}))
    selected_ids = planner_api.greedy_cost_plan(
        scored, float(context.get("budget", 0.0)), lambda_uncertainty=1.0, mu_risk=1.0)
    blocks = ["## Budgeted evidence pointers", "",
              "Entries point to exact text already present in the frozen manuscript. Retrieved context informed selection but is not added as new evidence."]
    for cid in selected_ids:
        candidate = by_id[cid]
        quote = str(candidate.get("quote", "")).strip()
        if not quote:
            continue
        blocks.extend(["", f"### {cid}: existing manuscript evidence",
                       f"Review concern: {str(candidate.get('claim', '')).strip()}",
                       "Inspect the quoted passage to determine whether it addresses the concern.",
                       "This is a locator only; it does not by itself establish that the concern is resolved.",
                       f"> {quote}"])
    appendix = "\n".join(blocks) if selected_ids else ""
    return result("evidencepatch_section_context", appendix, actions=selected_ids,
                  ratings=list(ratings.values()), calls=1,
                  planner="risk-adjusted marginal utility per cost using retrieved section context",
                  retrieval={"granularity": "complete_sections", "top_k": 5,
                             "max_context_chars_per_claim": 20_000},
                  abstained_candidates=sum(x["decision"] != "select" for x in ratings.values()))
