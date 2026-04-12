"""Optional alternative import namespace for graph algorithm utilities."""

from .graph_algorithms import (
    topological_sort,
    condensation_execution_order,
    has_cycle,
)

__all__ = [
    "topological_sort",
    "condensation_execution_order",
    "has_cycle",
]
