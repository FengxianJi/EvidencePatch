from method.common import render_entries, result


def run(context):
    entries = list(context.get("evidence_candidates", []))
    limit = int(context.get("top_k", 5))
    chosen = entries[:limit]
    appendix = render_entries(chosen, "Retrieved manuscript passages")
    return result("retrieval_only", appendix, entries=chosen, calls=0,
                  interpretation="retrieval only; no entailment or concern-resolution judgment")
