# Random feasible budget selector

Internal cost-matched planner baseline; delegates to the first experiment budget planner with fixed seed.

## Interface

`run(context: dict) -> dict`. The shared contract is documented in `method/README.md`. API credentials are supplied only by the calling experiment.
