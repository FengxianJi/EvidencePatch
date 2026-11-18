from __future__ import annotations

from method.common import call_model, parse_json_object, result


SYSTEM = """You are the diagnosis and evidence-value estimator in EvidencePatch. Use calibrated probabilities, abstain when the supplied manuscript quote does not address the review concern, and never invent evidence or claim that a pointer proves a result."""


def run(context):
    candidates = context.get("candidates", [])
    if not candidates:
        return result("evidencepatch_risk_aware", "", actions=[], ratings=[], calls=0,
                      interpretation="No exact-source candidate was available; abstain.")

    compact = [{"candidate_id": str(c["candidate_id"]),
                "review_concern": str(c.get("claim", "")),
                "exact_manuscript_quote": str(c.get("quote", ""))}
               for c in candidates]
    prompt = f"""Assess each candidate independently for a minimal evidence-pointer appendix. The quote is already an exact substring of the frozen manuscript. Decide whether the quote directly helps a reviewer check the stated concern. A quote that merely mentions the topic is not enough. Do not propose new experiments or additional facts.

For each candidate return: candidate_id, decision ('select' or 'abstain'), p_concern_valid, p_quote_relevant, uncertainty, risk, and a short pointer_note. All probabilities are numbers from 0 to 1. p_concern_valid estimates whether the review concern is a real, checkable issue. p_quote_relevant estimates whether this exact quote directly helps inspect it. uncertainty represents uncertainty in those estimates. risk represents the chance that including the pointer could mislead, overstate resolution, or distract. The note may only say where/what to inspect, and must not claim the concern is resolved.

Candidates:
{compact}

Return JSON only: {{"ratings":[{{"candidate_id":"e1","decision":"select","p_concern_valid":0.0,"p_quote_relevant":0.0,"uncertainty":0.0,"risk":0.0,"pointer_note":"..."}}]}}. Include every candidate exactly once."""
    raw = call_model(context, SYSTEM, prompt, "evidencepatch_risk_estimation")
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
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in by_id or candidate_id in ratings:
            continue
        valid = prob(item.get("p_concern_valid"))
        relevant = prob(item.get("p_quote_relevant"))
        uncertainty = prob(item.get("uncertainty"))
        risk = prob(item.get("risk"))
        ratings[candidate_id] = {
            "candidate_id": candidate_id,
            "decision": "select" if item.get("decision") == "select" else "abstain",
            "p_concern_valid": valid,
            "p_quote_relevant": relevant,
            "expected_benefit": valid * relevant,
            "uncertainty": uncertainty,
            "risk": risk,
            "pointer_note": str(item.get("pointer_note", ""))[:280],
        }

    # Missing or malformed model rows are conservative abstentions.
    for candidate_id in by_id:
        ratings.setdefault(candidate_id, {
            "candidate_id": candidate_id, "decision": "abstain",
            "p_concern_valid": 0.0, "p_quote_relevant": 0.0,
            "expected_benefit": 0.0, "uncertainty": 1.0, "risk": 1.0,
            "pointer_note": ""})

    planner_api = context.get("planner_api")
    if planner_api is None:
        raise RuntimeError("EvidencePatch requires the audited budget planner")
    scored_candidates = []
    for candidate_id, candidate in by_id.items():
        rating = ratings[candidate_id]
        benefit = rating["expected_benefit"] if rating["decision"] == "select" else 0.0
        scored_candidates.append(planner_api.PatchCandidate(
            candidate_id=candidate_id,
            cost=float(candidate.get("cost", 0.0)),
            expected_benefit=0.0,
            uncertainty=rating["uncertainty"],
            risk=rating["risk"],
            claim_benefits={candidate_id: benefit},
        ))
    selected_ids = planner_api.greedy_cost_plan(
        scored_candidates, float(context.get("budget", 0.0)),
        lambda_uncertainty=1.0, mu_risk=1.0)

    blocks = ["## Budgeted evidence pointers", "",
              "Selected entries point to exact text already present in the frozen manuscript. They add no experiments, results, or new scientific claims."]
    for candidate_id in selected_ids:
        candidate = by_id[candidate_id]
        quote = str(candidate.get("quote", "")).strip()
        if not quote:
            continue
        blocks.extend(["", f"### {candidate_id}: existing manuscript evidence",
                       f"Review concern: {str(candidate.get('claim', '')).strip()}"])
        blocks.append("Inspect the quoted passage to determine whether it addresses the concern.")
        blocks.extend(["This is a locator only; it does not by itself establish that the concern is resolved.", f"> {quote}"])
    appendix = "\n".join(blocks) if selected_ids else ""
    return result("evidencepatch_risk_aware", appendix,
                  actions=selected_ids, ratings=list(ratings.values()), calls=1,
                  planner="risk-adjusted marginal utility per cost",
                  abstained_candidates=sum(v["decision"] != "select" for v in ratings.values()))
