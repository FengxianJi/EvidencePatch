from __future__ import annotations

from importlib import import_module

METHODS = {
    "frozen_original": "frozen_original",
    "one_shot_generic": "one_shot_generic",
    "iterative_generic": "iterative_generic",
    "drpg_style": "drpg_style",
    "retrieval_only": "retrieval_only",
    "peerify_verify": "peerify_verify",
    "random_budget": "random_budget",
    "cost_greedy": "cost_greedy",
    "all_feasible": "all_feasible",
    "oracle_selector": "oracle_selector",
}


def list_methods() -> list[str]:
    return list(METHODS)


def get_method(method_id: str):
    try:
        module = import_module(f".methods.{METHODS[method_id]}.method", package=__package__)
    except KeyError as exc:
        raise KeyError(f"Unknown baseline {method_id!r}; choose from {list_methods()}") from exc
    return module.run
