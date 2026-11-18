# One-shot generic LLM appendix

Local first-experiment baseline: a single M3 generation call, no structured evidence planner. See the existing comparison design in show/experiment/[1]first experiment/run_re2_baseline_comparison_v4.py.

## Interface

`run(context: dict) -> dict`. The shared contract is documented in `method/README.md`. API credentials are supplied only by the calling experiment.
