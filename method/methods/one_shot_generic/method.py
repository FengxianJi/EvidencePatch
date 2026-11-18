from method.common import call_model, parse_json_object, result

SYSTEM = "You draft a cautious academic appendix. Never invent analyses, results, citations, or facts."


def run(context):
    user = f"""Draft a generic supplementary appendix from this frozen manuscript and its original human reviews.
Use only facts, definitions, analyses, and results stated in the manuscript. Do not invent experiments, measurements, citations, code, or results. Do not claim a concern is resolved unless the supplied text supports the response. If a concern needs new evidence, say so and do not fabricate it.
FROZEN MANUSCRIPT:\n{context.get('manuscript', '')}\nORIGINAL HUMAN REVIEWS:\n{context.get('reviews', '')}
Return JSON only: {{\"appendix_markdown\":\"short appendix text\",\"entries\":[{{\"review_number\":1,\"review_excerpt\":\"verbatim excerpt <=30 words\",\"response\":\"short response using only manuscript facts\",\"source_quotes\":[\"verbatim manuscript text, <=25 words\"]}}]}}. At most three entries total. Keep the appendix within {context.get('appendix_char_budget', 8000)} characters."""
    raw = call_model(context, SYSTEM, user, "baseline_one_shot_generic")
    obj = parse_json_object(raw)
    entries = obj.get("entries", []) if isinstance(obj.get("entries"), list) else []
    return result("one_shot_generic", str(obj.get("appendix_markdown", "")),
                  unresolved=obj.get("unresolved", []), entries=entries, raw=obj, calls=1)
