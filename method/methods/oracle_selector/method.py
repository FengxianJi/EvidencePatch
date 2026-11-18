from method.common import planner_select, render_entries, result, selected_evidence


def run(context):
    ids = planner_select(context, "oracle")
    entries = selected_evidence(context, ids)
    return result("oracle_selector", render_entries(entries), actions=ids,
                  cost_policy="exact subset optimum; diagnostic upper bound only",
                  budget=context.get("budget"), diagnostic_only=True)
