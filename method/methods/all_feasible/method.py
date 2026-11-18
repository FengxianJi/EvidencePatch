from method.common import planner_select, render_entries, result, selected_evidence


def run(context):
    ids = planner_select(context, "all_feasible")
    entries = selected_evidence(context, ids)
    return result("all_feasible", render_entries(entries), actions=ids, cost_policy="all compatible actions fitting budget",
                  budget=context.get("budget"))
