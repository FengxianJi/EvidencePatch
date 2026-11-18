from __future__ import annotations

from method.common import call_model, parse_json_object, result


SYSTEM = """You are a conservative reviewer-claim and evidence-value assessor. The review excerpt and manuscript quotation are verbatim spans from their frozen sources. Decide whether the quotation directly helps inspect the excerpt in the context of the full review. Do not claim the concern is resolved. Never invent evidence or suggest that a locator proves a result."""


def run(context):
    candidates = context.get("candidates", [])
    if not candidates:
        return result("evidencepatch_bm25_risk_aware", "", actions=[], ratings=[], calls=0,
                      interpretation="No exact BM25 review/source pair; abstain.")

    compact = [{
        "candidate_id": str(c["candidate_id"]),
        "review_number": c.get("review_number"),
        "exact_review_excerpt": str(c.get("claim", "")),
        "full_review_context": str(c.get("review_context", ""))[:5000],
        "exact_manuscript_quote": str(c.get("quote", "")),
        "manuscript_section": str(c.get("section", "")),
        "bm25_score": c.get("bm25_score"),
    } for c in candidates]
    prompt = f"""Assess each exact review-excerpt/manuscript-quote pair independently for a minimal evidence-pointer appendix. BM25 only proposes candidate relevance; it is not proof. A topic mention alone is not enough. Do not propose new experiments or additional facts.

For every candidate return candidate_id, decision ('select' or 'abstain'), p_concern_valid, p_quote_relevant, uncertainty, risk, and a short pointer_note. Probabilities must be numbers from 0 to 1. The concern-validity probability is about whether the excerpt is a real, checkable issue. Quote relevance is whether this exact passage directly helps a reader inspect that concern. Uncertainty is epistemic uncertainty; risk is the chance that a pointer misleads, overstates resolution, or distracts. A note can only guide inspection and must not claim resolution. Include every candidate exactly once.

Candidates:
{compact}

Return JSON only: {{"ratings":[{{"candidate_id":"r1_p1","decision":"select","p_concern_valid":0.0,"p_quote_relevant":0.0,"uncertainty":0.0,"risk":0.0,"pointer_note":"..."}}]}}."""
    obj = parse_json_object(call_model(context, SYSTEM, prompt, "evidencepatch_bm25_risk_estimation"))
    rating_rows = obj.get("ratings")
    if not isinstance(rating_rows, list):
        raise ValueError("Risk estimates must include a ratings list")
    by_id = {str(c["candidate_id"]): c for c in candidates}

    def prob(value):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    ratings = {}
    for item in rating_rows:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", ""))
        if cid not in by_id or cid in ratings:
            continue
        valid = prob(item.get("p_concern_valid"))
        relevant = prob(item.get("p_quote_relevant"))
        ratings[cid] = {
            "candidate_id": cid,
            "decision": "select" if item.get("decision") == "select" else "abstain",
            "p_concern_valid": valid,
            "p_quote_relevant": relevant,
            "expected_benefit": valid * relevant,
            "uncertainty": prob(item.get("uncertainty")),
            "risk": prob(item.get("risk")),
            "pointer_note": str(item.get("pointer_note", ""))[:280],
        }
    for cid in by_id:
        ratings.setdefault(cid, {"candidate_id": cid, "decision": "abstain",
                                 "p_concern_valid": 0.0, "p_quote_relevant": 0.0,
                                 "expected_benefit": 0.0, "uncertainty": 1.0,
                                 "risk": 1.0, "pointer_note": ""})

    planner = context.get("planner_api")
    if planner is None:
        raise RuntimeError("An audited budget planner is required")
    plan_candidates = []
    for cid, candidate in by_id.items():
        rating = ratings[cid]
        benefit = rating["expected_benefit"] if rating["decision"] == "select" else 0.0
        plan_candidates.append(planner.PatchCandidate(
            candidate_id=cid, cost=float(candidate.get("cost", 0.0)), expected_benefit=0.0,
            uncertainty=rating["uncertainty"], risk=rating["risk"],
            claim_benefits={cid: benefit}))
    actions = planner.greedy_cost_plan(plan_candidates, float(context.get("budget", 0.0)),
                                       lambda_uncertainty=1.0, mu_risk=1.0)
    blocks = ["## Budgeted evidence pointers", "",
              "These entries pair exact text from the review with passages already present in the frozen manuscript. A locator is not evidence that the concern is resolved."]
    for cid in actions:
        candidate = by_id[cid]
        blocks.extend(["", f"### Review {candidate.get('review_number')}: existing evidence",
                       f"Review excerpt: {candidate.get('claim', '')}",
                       f"Inspect the quoted passage in {candidate.get('section') or 'the manuscript'} to assess the concern.",
                       f"> {candidate.get('quote', '')}"])
    appendix = "\n".join(blocks) if actions else ""
    return result("evidencepatch_bm25_risk_aware", appendix, actions=actions,
                  ratings=list(ratings.values()), calls=1,
                  planner="risk-adjusted marginal utility per cost",
                  candidate_count=len(candidates),
                  abstained_candidates=sum(x["decision"] != "select" for x in ratings.values()))
