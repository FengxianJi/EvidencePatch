# Peerify-style claim verification

Inspired by Peerify (arXiv:2609.25046), https://arxiv.org/abs/2609.25046 ; related official claim-verification codebase: https://github.com/Reviewerly-Inc/Peerify . Local adapter is an M3 verifier interface, not a reproduced benchmark result.

## Interface

`run(context: dict) -> dict`. The shared contract is documented in `method/README.md`. API credentials are supplied only by the calling experiment.
