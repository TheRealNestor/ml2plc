"""
Utilities for working with NetworkIR.

Provides helper functions for manipulating and querying IR structures.
Centralizes common IR operations to avoid duplication across the pipeline.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from .types import BaseLayer, NetworkIR


def build_tensor_maps(
    layers: Dict[str, BaseLayer],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build tensor producer/consumer maps from layers.

    Extracts graph structure (tensor flow) from a collection of layers.
    A centralized function used throughout the pipeline:
    - After parsing (converter.py does this incrementally for efficiency)
    - After optimization (optimizer.py rebuilds when layers are removed)
    - After merging regions (regionizer.py rebuilds for merged components)

    Args:
        layers: Dictionary of layer objects

    Returns:
        Tuple of (tensor_producers, tensor_consumers) where:
        - tensor_producers: Dict[tensor_name] -> producing_layer_name
        - tensor_consumers: Dict[tensor_name] -> [consuming_layer_names]
    """
    tensor_producers: Dict[str, str] = {}
    tensor_consumers: Dict[str, List[str]] = defaultdict(list)

    for layer in layers.values():
        for inp in layer.inputs:
            tensor_consumers[inp].append(layer.name)
        for out in layer.outputs:
            tensor_producers[out] = layer.name

    return tensor_producers, dict(tensor_consumers)


def filter_tensor_maps_for_nodes(
    tensor_producers: Dict[str, str],
    tensor_consumers: Dict[str, List[str]],
    component_nodes: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Extract tensor maps filtered to a subset of nodes (a component).

    Used when creating subgraphs for regions. Keeps only tensors that are:
    - Produced by nodes in the component
    - Consumed by nodes in the component

    Args:
        tensor_producers: Global tensor producer map
        tensor_consumers: Global tensor consumer map
        component_nodes: Set of layer names to filter to

    Returns:
        Tuple of (filtered_producers, filtered_consumers)
    """
    filtered_producers = {
        tensor: producer
        for tensor, producer in tensor_producers.items()
        if producer in component_nodes
    }

    filtered_consumers = {
        tensor: [c for c in consumers if c in component_nodes]
        for tensor, consumers in tensor_consumers.items()
        if any(c in component_nodes for c in consumers)
    }

    return filtered_producers, filtered_consumers


def extract_component_input_tensors(
    graph: NetworkIR,
    component_nodes: Set[str],
    filtered_producers: Dict[str, str],
) -> Set[str]:
    """
    Extract input tensors for a component.

    Includes:
    1. Global network inputs consumed by the component
    2. Tensors consumed but not produced internally (external inputs)

    This handles state back-edges (outputs that loop back as inputs).

    Args:
        graph: Full network graph
        component_nodes: Set of layer names in the component
        filtered_producers: Tensor producers internal to component

    Returns:
        Set of input tensor names for the component
    """
    input_tensors_set: Set[str] = set()

    # Global network inputs consumed by component
    for t in graph.input_tensors:
        if any(c in component_nodes for c in graph.tensor_consumers.get(t, [])):
            input_tensors_set.add(t)

    # External inputs (consumed but not produced internally)
    for layer_name in component_nodes:
        layer = graph.layers[layer_name]
        for in_tensor in layer.inputs:
            if in_tensor not in filtered_producers:
                input_tensors_set.add(in_tensor)

    return input_tensors_set


def extract_component_output_tensors(
    graph: NetworkIR,
    component_nodes: Set[str],
) -> Tuple[str, ...]:
    """
    Extract output tensors produced by a component.

    Returns global network outputs that are produced by this component.

    Args:
        graph: Full network graph
        component_nodes: Set of layer names in the component

    Returns:
        Tuple of output tensor names for the component
    """
    return tuple(
        t
        for t in graph.output_tensors
        if graph.tensor_producers.get(t) in component_nodes
    )


def extract_component_state_tensors(
    graph: NetworkIR,
    component_nodes: Set[str],
) -> Tuple[str, ...]:
    """
    Extract state tensors relevant to a component.

    Returns state tensors that are produced by this component
    (detected by the converter from RNN operators).

    Args:
        graph: Full network graph
        component_nodes: Set of layer names in the component

    Returns:
        Tuple of state tensor names for the component
    """
    return tuple(
        t
        for t in graph.state_tensors.keys()
        if t in graph.tensor_producers and graph.tensor_producers[t] in component_nodes
    )
