"""
Utilities for working with NetworkIR.

Provides helper classes and functions for manipulating and querying IR structures.
Centralizes common IR operations to avoid duplication across the pipeline.

Key Components:
  - TensorMapBuilder: Unified class for constructing and querying tensor maps
  - Helper functions: Maintain backward compatibility with legacy code
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from .types import BaseLayer, NetworkIR


class TensorMapBuilder:
    """
    Unified builder for constructing and querying tensor producer/consumer maps.

    This consolidates all tensor mapping operations from across the codebase:
    - converter.py: Initial map construction
    - ir_utils.py: Map rebuilding and filtering
    - regionizer.py: Component-specific map extraction

    Architecture: Single source of truth for tensor graph structure.
    """

    def __init__(self):
        """Initialize empty builder."""
        self.producers: Dict[str, str] = {}
        self.consumers: Dict[str, List[str]] = defaultdict(list)

    @staticmethod
    def build(layers: Dict[str, BaseLayer]) -> "TensorMapBuilder":
        """
        Build complete producer/consumer maps from a layers dictionary.

        Used throughout the pipeline:
        - After parsing (converter.py)
        - After optimization (optimizer.py)
        - After merging regions (regionizer.py)

        Args:
            layers: Dictionary of layer objects

        Returns:
            TensorMapBuilder instance with populated maps
        """
        builder = TensorMapBuilder()
        for layer in layers.values():
            for inp in layer.inputs:
                builder.consumers[inp].append(layer.name)
            for out in layer.outputs:
                builder.producers[out] = layer.name
        return builder

    def extract_for_nodes(self, component_nodes: Set[str]) -> "TensorMapBuilder":
        """
        Extract a filtered builder containing only tensors for a component subset.

        Used when creating subgraphs for regions. Keeps only tensors that are:
        - Produced by nodes in the component
        - Consumed by nodes in the component

        Args:
            component_nodes: Set of layer names to filter to

        Returns:
            New TensorMapBuilder with filtered maps
        """
        filtered = TensorMapBuilder()

        filtered.producers = {
            tensor: producer
            for tensor, producer in self.producers.items()
            if producer in component_nodes
        }

        filtered.consumers = {
            tensor: [c for c in consumers if c in component_nodes]
            for tensor, consumers in self.consumers.items()
            if any(c in component_nodes for c in consumers)
        }

        return filtered

    def extract_input_tensors(
        self, graph: NetworkIR, component_nodes: Set[str]
    ) -> Set[str]:
        """
        Extract input tensors for a component.

        Includes:
        1. Global network inputs consumed by the component
        2. Tensors consumed but not produced internally (external inputs)

        This handles state back-edges (outputs that loop back as inputs).

        Args:
            graph: Full network graph (for network inputs)
            component_nodes: Set of layer names in the component

        Returns:
            Set of input tensor names for the component
        """
        input_tensors: Set[str] = set()

        # Global network inputs consumed by component
        for t in graph.input_tensors:
            if any(c in component_nodes for c in self.consumers.get(t, [])):
                input_tensors.add(t)

        # External inputs (consumed but not produced internally)
        for layer_name in component_nodes:
            layer = graph.layers[layer_name]
            for in_tensor in layer.inputs:
                if in_tensor not in self.producers:
                    input_tensors.add(in_tensor)

        return input_tensors

    def extract_output_tensors(
        self, graph: NetworkIR, component_nodes: Set[str]
    ) -> Tuple[str, ...]:
        """
        Extract output tensors produced by a component.

        Returns global network outputs that are produced by this component.

        Args:
            graph: Full network graph (for network outputs)
            component_nodes: Set of layer names in the component

        Returns:
            Tuple of output tensor names for the component
        """
        return tuple(
            t for t in graph.output_tensors if self.producers.get(t) in component_nodes
        )

    def extract_state_tensors(
        self, graph: NetworkIR, component_nodes: Set[str]
    ) -> Tuple[str, ...]:
        """
        Extract state tensors relevant to a component.

        Returns state tensors that are produced by this component
        (detected by the converter from RNN operators).

        Args:
            graph: Full network graph (for state tensor info)
            component_nodes: Set of layer names in the component

        Returns:
            Tuple of state tensor names for the component
        """
        return tuple(
            t
            for t in graph.state_tensors.keys()
            if t in self.producers and self.producers[t] in component_nodes
        )

    def get_producer(self, tensor: str) -> str | None:
        """Get the layer name that produces this tensor."""
        return self.producers.get(tensor)

    def get_consumers(self, tensor: str) -> List[str]:
        """Get the layer names that consume this tensor."""
        return self.consumers.get(tensor, [])

    def as_tuple(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """Return maps as tuple for backward compatibility."""
        return (self.producers, dict(self.consumers))


# ============================================================================
# Backward-Compatibility Functions
# ============================================================================


def build_tensor_maps(
    layers: Dict[str, BaseLayer],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Legacy function for backward compatibility.

    Use TensorMapBuilder.build().as_tuple() instead in new code.
    """
    builder = TensorMapBuilder.build(layers)
    return builder.as_tuple()


def filter_tensor_maps_for_nodes(
    tensor_producers: Dict[str, str],
    tensor_consumers: Dict[str, List[str]],
    component_nodes: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Legacy function for backward compatibility.

    Use TensorMapBuilder.extract_for_nodes() instead in new code.
    """
    builder = TensorMapBuilder()
    builder.producers = tensor_producers
    builder.consumers = defaultdict(list, tensor_consumers)
    filtered = builder.extract_for_nodes(component_nodes)
    return filtered.as_tuple()


def extract_component_input_tensors(
    graph: NetworkIR,
    component_nodes: Set[str],
    filtered_producers: Dict[str, str],
) -> Set[str]:
    """
    Legacy function for backward compatibility.

    Use TensorMapBuilder.extract_input_tensors() instead in new code.
    """
    builder = TensorMapBuilder()
    builder.producers = filtered_producers
    return builder.extract_input_tensors(graph, component_nodes)


def extract_component_output_tensors(
    graph: NetworkIR,
    component_nodes: Set[str],
) -> Tuple[str, ...]:
    """
    Legacy function for backward compatibility.

    Build a TensorMapBuilder from graph and use extract_output_tensors() instead.
    """
    builder = TensorMapBuilder()
    builder.producers = graph.tensor_producers
    return builder.extract_output_tensors(graph, component_nodes)


def extract_component_state_tensors(
    graph: NetworkIR,
    component_nodes: Set[str],
) -> Tuple[str, ...]:
    """
    Legacy function for backward compatibility.

    Build a TensorMapBuilder from graph and use extract_state_tensors() instead.
    """
    builder = TensorMapBuilder()
    builder.producers = graph.tensor_producers
    return builder.extract_state_tensors(graph, component_nodes)
