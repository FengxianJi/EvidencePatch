from method.common import call_model, parse_json_object, result

SYSTEM = "You are an evidence-grounded rebuttal assistant. Do not fabricate results or citations."


def run(context):
    prompt = f"""Apply a four-stage pipeline: (1) decompose review text into atomic concerns; (2) retrieve exact evidence from the manuscript; (3) choose respond/clarify/needs-new-evidence/abstain; (4) draft short appendix entries.
MANUSCRIPT:\n{context.get('manuscript', '')}\nREVIEWS:\n{context.get('reviews', '')}
Return JSON {{\"concerns\":[{{\"claim\":\"...\",\"evidence_quote\":\"exact quote or empty\",\"plan\":\"...\",\"response\":\"...\"}}],\"appendix_markdown\":\"...\"}}. Keep the appendix within {context.get('appendix_char_budget', 8000)} characters."""
    obj = parse_json_object(call_model(context, SYSTEM, prompt, "baseline_drpg_style"))
    return result("drpg_style", str(obj.get("appendix_markdown", "")),
                  concerns=obj.get("concerns", []), calls=1,
                  provenance="DRPG-inspired prompt adaptation; not original DRPG reproduction")
