from method.common import call_model, parse_json_object, result

SYSTEM = "You draft grounded academic supplementary material. Never fabricate evidence or claim a new experiment was run."


def run(context):
    manuscript, reviews = str(context.get("manuscript", "")), str(context.get("reviews", ""))
    first = parse_json_object(call_model(context, SYSTEM,
        f"Manuscript:\n{manuscript}\nReviews:\n{reviews}\nReturn JSON with appendix_markdown and unresolved. Keep the appendix within {context.get('appendix_char_budget', 8000)} characters.",
        "baseline_iterative_draft"))
    draft = str(first.get("appendix_markdown", ""))
    feedback = str(context.get("verification_feedback", "No independent feedback supplied."))
    revised = parse_json_object(call_model(context, SYSTEM,
        f"Manuscript:\n{manuscript}\nReviews:\n{reviews}\nDraft:\n{draft}\nAudit feedback:\n{feedback}\n"
        f"Revise only to remove unsupported or unclear statements. Keep the final appendix within {context.get('appendix_char_budget', 8000)} characters. Return JSON with appendix_markdown and unresolved.",
        "baseline_iterative_revision"))
    return result("iterative_generic", str(revised.get("appendix_markdown", "")),
                  unresolved=revised.get("unresolved", first.get("unresolved", [])), calls=2,
                  draft_chars=len(draft))
