"""
Reusable graph algorithms for IR manipulation.

Provides utilities for:
- Topological sorting (Kahn's algorithm)
- Strongly connected component (SCC) detection (Tarjan's algorithm)
- Component-based graph condensation
"""

import logging
from typing import Dict, List, Set, Tuple
from collections import deque, defaultdict

from codegen.types import BaseLayer

logger = logging.getLogger(__name__)


def topological_sort(
    layers: Dict[str, BaseLayer],
    tensor_producers: Dict[str, str],
    input_tensors: tuple,
) -> List[str]:
    """
    Perform topological sort on the layer graph using Kahn's algorithm.

    Validates that the graph is acyclic. Raises ValueError if a cycle is detected.

    Args:
        layers: Dictionary of IR layer objects keyed by layer name
        tensor_producers: Mapping of tensor names to producing layer names
        input_tensors: Network input tensor names (not produced by any layer)

    Returns:
        List of layer names in execution order

    Raises:
        ValueError: If a cycle is detected in the graph
    """
    adj_list = defaultdict(list)
    in_degree = {name: 0 for name in layers.keys()}

    for layer_name, layer in layers.items():
        for input_tensor in layer.inputs:
            if input_tensor in input_tensors:
                continue

            if input_tensor in tensor_producers:
                producer = tensor_producers[input_tensor]
                if producer != layer_name:
                    adj_list[producer].append(layer_name)
                    in_degree[layer_name] += 1

    queue = deque([name for name, degree in in_degree.items() if degree == 0])
    sorted_order = []

    while queue:
        current = queue.popleft()
        sorted_order.append(current)

        for neighbor in adj_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_order) != len(layers):
        missing = set(layers.keys()) - set(sorted_order)
        raise ValueError(f"Cycle detected in layer graph: {missing}")

    return sorted_order


def build_layer_graph(
    layers: Dict[str, "BaseLayer"],
    tensor_producers: Dict[str, str],
    tensor_consumers: Dict[str, List[str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build producer->consumer and reverse adjacency from tensor flow.

    Args:
        layers: Dictionary of layer objects
        tensor_producers: Tensor name -> producing layer name
        tensor_consumers: Tensor name -> [consuming layer names]

    Returns:
        Tuple of (adjacency, reverse_adjacency) where each maps layer names to sets of layer names
    """
    adjacency: Dict[str, Set[str]] = {name: set() for name in layers.keys()}
    reverse_adjacency: Dict[str, Set[str]] = {name: set() for name in layers.keys()}

    for producer in layers:
        layer = layers[producer]
        for out_tensor in layer.outputs:
            for consumer in tensor_consumers.get(out_tensor, []):
                if consumer in layers and consumer != producer:
                    adjacency[producer].add(consumer)
                    reverse_adjacency[consumer].add(producer)

    return adjacency, reverse_adjacency


def tarjan_scc(adjacency: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Compute strongly connected components using Tarjan's algorithm.

    Returns components in reverse topological order (dependencies come after dependents).

    Args:
        adjacency: Dictionary mapping node names to sets of successor node names

    Returns:
        List of strongly connected components, each as a list of node names
    """
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    components: List[List[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, set()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == node:
                    break
            components.append(component)

    for node in adjacency:
        if node not in indices:
            strongconnect(node)

    return components


def topological_order_components(component_edges: Dict[int, Set[int]]) -> List[int]:
    """
    Topologically sort a component DAG (condensation graph).

    Args:
        component_edges: Dictionary mapping component ID to set of successor component IDs

    Returns:
        List of component IDs in topological order
    """
    indegree: Dict[int, int] = {cid: 0 for cid in component_edges}
    for src in component_edges:
        for dst in component_edges[src]:
            indegree[dst] += 1

    queue = deque(sorted([cid for cid, deg in indegree.items() if deg == 0]))
    order: List[int] = []

    while queue:
        cid = queue.popleft()
        order.append(cid)
        for dst in sorted(component_edges[cid]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)

    if len(order) != len(component_edges):
        # Defensive fallback; condensation graph should be acyclic.
        logger.warning(
            "Component condensation graph has cycles; returning fallback sort"
        )
        return sorted(component_edges.keys())
    return order
