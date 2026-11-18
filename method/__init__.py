"""Baseline methods used by project5's first experiment.

Each method exposes ``run(context) -> dict``.  The package is deliberately
independent of experiment data and credentials; the experiment supplies its
frozen paper/reviews and, for API methods, a callable MiniMax adapter.
"""

from .registry import METHODS, get_method, list_methods

__all__ = ["METHODS", "get_method", "list_methods"]
