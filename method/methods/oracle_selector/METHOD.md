# Exact small-set oracle

Diagnostic upper bound only, not a deployable baseline. Uses exact subset search and is limited by the planner to small candidate pools.

## Interface

`run(context: dict) -> dict`. The shared contract is documented in `method/README.md`. API credentials are supplied only by the calling experiment.
