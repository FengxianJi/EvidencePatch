# DRPG-style rebuttal pipeline

Inspired by DRPG, Decompose, Retrieve, Plan, Generate (arXiv:2601.18081). Paper: https://arxiv.org/abs/2601.18081 ; official repository: https://github.com/ulab-uiuc/DRPG-RebuttalAgent . Local implementation is a one-call prompt adaptation, not an original-code reproduction.

## Interface

`run(context: dict) -> dict`. The shared contract is documented in `method/README.md`. API credentials are supplied only by the calling experiment.
