from method.common import planner_select, render_entries, result, selected_evidence


def run(context):
    ids = planner_select(context, "cost_greedy")
    entries = selected_evidence(context, ids)
    return result("cost_greedy", render_entries(entries), actions=ids, cost_policy="marginal utility per cost",
                  budget=context.get("budget"))
