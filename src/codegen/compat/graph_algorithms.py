"""Optional re-export module for graph algorithm utilities.

This module keeps an alternate import path available while canonical
implementations live in :mod:`codegen.graph_algorithms`.
"""

from ..graph_algorithms import (
    topological_sort,
    condensation_execution_order,
    has_cycle,
)

__all__ = [
    "topological_sort",
    "condensation_execution_order",
    "has_cycle",
]
