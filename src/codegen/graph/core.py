"""
Unified graph abstraction for layer dependency analysis.

Single source of truth for layer graph queries:
- Adjacency relationships
- Strongly connected components (SCCs)
- Cycle detection
- Execution ordering (acyclic or cyclic)
- Topological properties

Replaces:
- Scattered build_layer_graph() calls in regionizer.py
- Redundant adjacency dict construction
- Manual SCC computation
"""

from typing import Dict, Set, List
from functools import cached_property
import logging

from ..types import BaseLayer, NetworkIR

logger = logging.getLogger(__name__)


class LayerGraph:
    """
    Unified interface for layer dependency analysis.

    Wraps NetworkIR and provides:
    - Lazy-computed graph properties (adjacency, SCCs, execution order)
    - Memoization to avoid redundant computation
    - Clear API for graph queries

    **Key Design:** Immutable after construction. Properties are computed once and cached.
    """

    def __init__(self, network_ir: NetworkIR):
        """
        Args:
            network_ir: NetworkIR instance to analyze
        """
        self.ir = network_ir

    # ========== Core Graph Properties (Cached) ==========

    @cached_property
    def adjacency(self) -> Dict[str, Set[str]]:
        """
        Layer dependency graph: producer -> consumers.

        Built from tensor flow relationships:
        - If Layer A outputs tensor T
        - And Layer B inputs tensor T
        - Then add edge A -> B

        Returns:
            Dict[layer_name -> set of successor layer names]
        """
        from ..graph_algorithms import build_layer_graph

        adj, _ = build_layer_graph(
            self.ir.layers,
            self.ir.tensor_producers,
            self.ir.tensor_consumers,
        )
        return adj

    @cached_property
    def reverse_adjacency(self) -> Dict[str, Set[str]]:
        """
        Reverse dependency graph: consumer -> producers.

        Returns:
            Dict[layer_name -> set of predecessor layer names]
        """
        from ..graph_algorithms import build_layer_graph

        _, rev_adj = build_layer_graph(
            self.ir.layers,
            self.ir.tensor_producers,
            self.ir.tensor_consumers,
        )
        return rev_adj

    @cached_property
    def strongly_connected_components(self) -> List[List[str]]:
        """
        Compute SCCs using Tarjan's algorithm.

        Each component is a maximal set of mutually reachable nodes:
        - Acyclic regions = single-node components
        - Recurrent regions = multi-node components

        Returns:
            List of components, each as list of layer names
        """
        from ..graph_algorithms import tarjan_scc

        return tarjan_scc(self.adjacency)

    # ========== Analysis Query Methods ==========

    def has_cycle(self) -> bool:
        """
        Check if graph contains any cycles.

        Returns:
            True if any SCCs have >1 node or any self-loops exist
        """
        for component in self.strongly_connected_components:
            if len(component) > 1:
                return True
            # Check self-loop
            node = component[0]
            if node in self.adjacency.get(node, set()):
                return True
        return False

    def get_sccs_with_recurrence(self) -> List[Set[str]]:
        """
        Get all multi-node SCCs (actual cycles).

        Used to identify recurrent regions.

        Returns:
            List of sets, each containing nodes forming a cycle
        """
        return [set(scc) for scc in self.strongly_connected_components if len(scc) > 1]

    def get_self_loops(self) -> Set[str]:
        """
        Get all layers with self-dependencies (A -> A).

        These form single-node recurrent components (like state variables).

        Returns:
            Set of layer names with self-loops
        """
        loops = set()
        for layer_name in self.ir.layers:
            if layer_name in self.adjacency.get(layer_name, set()):
                loops.add(layer_name)
        return loops

    def is_acyclic(self) -> bool:
        """
        Check if graph is a pure DAG (no cycles).

        Returns:
            True if no cycles exist anywhere
        """
        return not self.has_cycle()

    def get_layer_dependencies(self, layer_name: str) -> Set[str]:
        """
        Get all layers that the given layer depends on (predecessors).

        Args:
            layer_name: Name of layer to query

        Returns:
            Set of layer names this layer depends on
        """
        dependencies = set()
        layer = self.ir.layers.get(layer_name)
        if not layer:
            return dependencies

        for input_tensor in layer.inputs:
            # Skip network inputs (not produced by any layer)
            if input_tensor in self.ir.input_tensors:
                continue

            producer = self.ir.tensor_producers.get(input_tensor)
            if producer and producer in self.ir.layers:
                dependencies.add(producer)

        return dependencies

    def get_layer_dependents(self, layer_name: str) -> Set[str]:
        """
        Get all layers that depend on the given layer (successors).

        Args:
            layer_name: Name of layer to query

        Returns:
            Set of layer names that depend on this layer
        """
        dependents = set()
        layer = self.ir.layers.get(layer_name)
        if not layer:
            return dependents

        for output_tensor in layer.outputs:
            consumers = self.ir.tensor_consumers.get(output_tensor, [])
            dependents.update(consumers)

        return dependents

    def get_execution_order(self) -> List[str]:
        """
        Get execution order for all layers.

        For acyclic graphs:  Standard topological sort
        For cyclic graphs:   SCC-based linearization (best-effort ordering)

        Returns:
            List of layer names in execution order
        """
        from ..graph_algorithms import (
            topological_sort,
            condensation_execution_order,
            has_cycle,
        )

        # Check if acyclic
        if not has_cycle(
            self.ir.layers, self.ir.tensor_producers, self.ir.input_tensors
        ):
            return topological_sort(
                self.ir.layers,
                self.ir.tensor_producers,
                self.ir.input_tensors,
            )
        else:
            # Fall back to SCC-based ordering
            return condensation_execution_order(
                self.ir.layers,
                self.ir.tensor_producers,
                self.ir.input_tensors,
            )

    # ========== SCC Analysis ==========

    def get_condensation_graph(self) -> Dict[int, Set[int]]:
        """
        Build condensation graph (DAG of SCCs).

        Each SCC is collapsed to a single node in a DAG of components.
        Used for component-level topological ordering.

        Returns:
            Dict[component_id -> set of successor component_ids]
        """
        sccs = self.strongly_connected_components
        node_to_component = {node: cid for cid, scc in enumerate(sccs) for node in scc}

        component_edges: Dict[int, Set[int]] = {cid: set() for cid in range(len(sccs))}

        for src, neighbors in self.adjacency.items():
            src_cid = node_to_component[src]
            for dst in neighbors:
                dst_cid = node_to_component[dst]
                if src_cid != dst_cid:
                    component_edges[src_cid].add(dst_cid)

        return component_edges

    def get_ordered_sccs(self) -> List[List[str]]:
        """
        Get SCCs in topological order (dependencies first).

        Returns:
            List of SCCs, each as list of layer names, ordered topologically
        """
        from ..graph_algorithms import topological_order_components

        sccs = self.strongly_connected_components
        component_edges = self.get_condensation_graph()
        ordered_cids = topological_order_components(component_edges)

        return [sccs[cid] for cid in ordered_cids]

    # ========== Utility Methods ==========

    def layer_count(self) -> int:
        """Get total number of layers."""
        return len(self.ir.layers)

    def is_empty(self) -> bool:
        """Check if graph has no layers."""
        return self.layer_count() == 0

    def __repr__(self) -> str:
        """Debug representation."""
        acyclic = "ACYCLIC" if self.is_acyclic() else "CYCLIC"
        scc_count = len(self.strongly_connected_components)
        return f"LayerGraph(layers={self.layer_count()}, sccs={scc_count}, {acyclic})"
