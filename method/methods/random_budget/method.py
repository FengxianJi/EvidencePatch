from method.common import planner_select, render_entries, result, selected_evidence


def run(context):
    ids = planner_select(context, "random")
    entries = selected_evidence(context, ids)
    return result("random_budget", render_entries(entries), actions=ids, cost_policy="random feasible",
                  budget=context.get("budget"), seed=context.get("seed", 0))
