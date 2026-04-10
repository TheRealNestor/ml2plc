"""
Model regionization utilities.

Partitions a NetworkIR into typed regions (acyclic, recurrent, loop) based on
strongly connected component (SCC) analysis.
"""

import logging
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from ..types import (
    NetworkIR,
    ModelIR,
    AcyclicRegionIR,
    RecurrentRegionIR,
    LoopRegionIR,
    RegionKind,
)
from ..graph.core import LayerGraph
from ..ir_utils import (
    TensorMapBuilder,
    build_tensor_maps,  # Legacy, kept for compatibility
)

logger = logging.getLogger(__name__)

_CONTROL_FLOW_OPS = {"Loop", "Scan", "If"}


@dataclass
class RegionClassification:
    """
    Result of classifying a region.

    Single responsibility: Hold the result of region classification.
    Decouples classification logic from region construction.
    """

    kind: RegionKind
    state_inputs: Tuple[str, ...] = ()
    state_outputs: Tuple[str, ...] = ()
    loop_inputs: Tuple[str, ...] = ()
    loop_outputs: Tuple[str, ...] = ()


class RegionClassifier:
    """
    Unified classification of regions based on graph properties.

    Single responsibility: Determine region kind and extract relevant parameters.

    This consolidates:
    - _classify_region_kind(): Determines region kind (Loop, Recurrent, Acyclic)
    - _infer_state_tensors(): Extracts state input/output tensors
    - _infer_loop_tensors(): Extracts loop input/output tensors

    All region-specific logic is now in one place, making it easier to understand
    the relationship between region type and parameter extraction.
    """

    def __init__(self, graph: NetworkIR, layer_graph: LayerGraph):
        """
        Args:
            graph: Full NetworkIR being regionized
            layer_graph: LayerGraph for efficient graph queries
        """
        self.graph = graph
        self.layer_graph = layer_graph

    def classify(self, component_nodes: Set[str]) -> RegionClassification:
        """
        Classify a component into a region type and extract parameters.

        Determines region kind (acyclic, recurrent, or loop) and extracts
        relevant parameters (state tensors for recurrent, loop tensors for loops).

        Args:
            component_nodes: Set of layer names forming the component

        Returns:
            RegionClassification with kind and component-specific parameters
        """
        kind = self._determine_kind(component_nodes)

        if kind == RegionKind.ACYCLIC:
            return RegionClassification(kind=kind)

        elif kind == RegionKind.RECURRENT:
            state_inputs, state_outputs = self._infer_state_tensors(component_nodes)
            return RegionClassification(
                kind=kind,
                state_inputs=state_inputs,
                state_outputs=state_outputs,
            )

        elif kind == RegionKind.LOOP:
            loop_inputs, loop_outputs = self._infer_loop_tensors(component_nodes)
            return RegionClassification(
                kind=kind,
                loop_inputs=loop_inputs,
                loop_outputs=loop_outputs,
            )

        # Fallback (should not reach)
        logger.warning(
            f"Unable to classify region {component_nodes}; defaulting to ACYCLIC"
        )
        return RegionClassification(kind=RegionKind.ACYCLIC)

    def _determine_kind(self, component_nodes: Set[str]) -> RegionKind:
        """
        Determine region kind from component structure.

        Priority order:
        1. If contains ONNX Loop/Scan/If operators → LOOP region
        2. If component has multiple nodes → RECURRENT region (contains cycle)
        3. If single node with self-loop → RECURRENT region
        4. Otherwise → ACYCLIC region

        Args:
            component_nodes: Set of layer names to analyze

        Returns:
            RegionKind (LOOP, RECURRENT, or ACYCLIC)
        """
        # Check for explicit control flow operators (highest priority)
        has_control_flow = any(
            self.graph.layers[n].op_type in _CONTROL_FLOW_OPS for n in component_nodes
        )
        if has_control_flow:
            return RegionKind.LOOP

        # Check for multiple nodes (indicating feedback within component)
        if len(component_nodes) > 1:
            return RegionKind.RECURRENT

        # Check for self-loop on single node
        single_node = next(iter(component_nodes))
        if single_node in self.layer_graph.adjacency.get(single_node, set()):
            return RegionKind.RECURRENT

        # No cycles, no control flow → purely acyclic
        return RegionKind.ACYCLIC

    def _infer_state_tensors(
        self, component_nodes: Set[str]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """
        Infer state input/output tensors for a recurrent region.

        Two-tier strategy:
        1. **Annotated state (preferred):** From converter's StateDetector
           - Based on ONNX spec (e.g., LSTM initial_h at input index 5)
           - High confidence, self-documenting
           - Ground truth source

        2. **Topology-based (fallback):** Back-edge detection via circular deps
           - For older ONNX or partially-supported operators
           - Infers state by finding cycles in component
           - Used only when annotated state is not available

        Returns:
            (state_inputs, state_outputs) tensor name tuples
        """
        state_inputs: List[str] = []
        state_outputs: List[str] = []

        # Build builder for component to query tensors
        full_builder = TensorMapBuilder.from_graph(self.graph)
        component_builder = full_builder.extract_for_nodes(component_nodes)

        # Strategy 1: Use explicitly detected state tensors (from converter)
        # This is the primary mechanism and is reliable when available
        annotated_state_tensors = list(
            component_builder.extract_state_tensors(self.graph, component_nodes)
        )

        if annotated_state_tensors:
            # Use annotated state information (ground truth)
            state_outputs.extend(annotated_state_tensors)
            state_inputs.extend(annotated_state_tensors)
            logger.debug(
                f"Region {component_nodes}: using {len(annotated_state_tensors)} "
                f"annotated state tensors: {annotated_state_tensors}"
            )
            return (tuple(state_inputs), tuple(state_outputs))

        # Strategy 2: Fall back to topology-based inference for back-edges
        # This handles cases where state detection wasn't possible
        logger.debug(
            f"Region {component_nodes}: no annotated state tensors; "
            f"falling back to topology analysis"
        )

        for node_name in component_nodes:
            layer = self.graph.layers[node_name]

            for out_tensor in layer.outputs:
                # Check if this output is consumed by nodes in the same component
                consumers = self.graph.tensor_consumers.get(out_tensor, [])
                component_consumers = [c for c in consumers if c in component_nodes]

                if not component_consumers:
                    continue

                # This is a potential state tensor (internal to component)
                # It's a state output (produced here)
                if out_tensor not in state_outputs:
                    state_outputs.append(out_tensor)

                # For each consumer in this component, it's a state input
                for consumer in component_consumers:
                    if out_tensor not in layer.inputs:
                        # Only add as state input if it's actually consumed
                        if out_tensor not in state_inputs:
                            state_inputs.append(out_tensor)

        return (tuple(state_inputs), tuple(state_outputs))

    def _infer_loop_tensors(
        self, component_nodes: Set[str]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """
        Infer loop input/output tensors for a Loop region.

        For ONNX Loop/Scan operators, identify:
        - loop_inputs: tensors iterated over or loop state inputs
        - loop_outputs: tensors produced by loop body

        Args:
            component_nodes: Set of layer names forming the loop region

        Returns:
            (loop_inputs, loop_outputs) tensor name tuples
        """
        loop_layer = None
        for node_name in component_nodes:
            layer = self.graph.layers[node_name]
            if layer.op_type in _CONTROL_FLOW_OPS:
                loop_layer = layer
                break

        if not loop_layer:
            return ((), ())

        # Loop layer inputs and outputs are the loop-related tensors
        loop_inputs = tuple(
            t for t in loop_layer.inputs if t not in self.graph.input_tensors
        )
        loop_outputs = tuple(loop_layer.outputs)

        return (loop_inputs, loop_outputs)


def _subgraph_for_component(
    graph: NetworkIR,
    component_nodes: Set[str],
) -> NetworkIR:
    """
    Create a region-local graph for a component.

    Extracts a subgraph containing only nodes and tensors relevant to the
    component, preserving graph structure and state information.
    """
    # Extract layers belonging to this component
    layers = {
        name: graph.layers[name]
        for name in graph.execution_order
        if name in component_nodes
    }

    # Build tensor maps using centralized factory (from_graph)
    full_builder = TensorMapBuilder.from_graph(graph)
    component_builder = full_builder.extract_for_nodes(component_nodes)
    tensor_producers, tensor_consumers = component_builder.as_tuple()

    # Compute I/O tensors using builder
    input_tensors_set = component_builder.extract_input_tensors(graph, component_nodes)
    input_tensors = tuple(sorted(input_tensors_set))
    output_tensors = component_builder.extract_output_tensors(graph, component_nodes)

    # Extract state tensor information (ground truth from converter)
    state_tensors = {
        tensor: role
        for tensor, role in graph.state_tensors.items()
        if tensor in tensor_producers or tensor in input_tensors_set
    }

    # Preserve execution order within component
    execution_order = [
        name for name in graph.execution_order if name in component_nodes
    ]

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors=state_tensors,
    )


def _classify_region_kind(
    component_nodes: Set[str],
    layer_graph: LayerGraph,
) -> RegionKind:
    """
    [DEPRECATED] Use RegionClassifier.classify() instead.

    This function is kept for backward compatibility. The RegionClassifier
    class consolidates all classification logic in one place.

    Migration:
        Old: kind = _classify_region_kind(nodes, layer_graph)
        New: classifier = RegionClassifier(graph, layer_graph)
             classification = classifier.classify(nodes)
             kind = classification.kind

    Classify a component as one of: Loop, Recurrent, or Acyclic.

    Classification logic (priority order):
    1. If contains ONNX Loop/Scan/If operators → LOOP region
    2. If multiple nodes or self-loop → RECURRENT region
    3. Otherwise → ACYCLIC region
    """
    # Check for explicit control flow operators (highest priority)
    has_control_flow = any(
        layer_graph.ir.layers[n].op_type in _CONTROL_FLOW_OPS for n in component_nodes
    )
    if has_control_flow:
        return RegionKind.LOOP

    # Check for multiple nodes (indicating feedback within component)
    if len(component_nodes) > 1:
        return RegionKind.RECURRENT

    # Check for self-loop on single node
    single_node = next(iter(component_nodes))
    if single_node in layer_graph.adjacency.get(single_node, set()):
        return RegionKind.RECURRENT

    # No cycles, no control flow → purely acyclic
    return RegionKind.ACYCLIC


def _infer_state_tensors(
    component_nodes: Set[str],
    graph: NetworkIR,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    [DEPRECATED] Use RegionClassifier.classify() instead.

    This function is kept for backward compatibility. The RegionClassifier
    class consolidates all classification logic in one place, including
    state tensor inference.

    Migration:
        Old: state_in, state_out = _infer_state_tensors(nodes, graph)
        New: classifier = RegionClassifier(graph, layer_graph)
             classification = classifier.classify(nodes)
             state_in = classification.state_inputs
             state_out = classification.state_outputs

    Infer state input/output tensors for a recurrent region.

    Strategy: Use detected state tensors (from converter) as ground truth where
    available, but fall back to topology analysis for cases where state detection
    wasn't possible (e.g., older ONNX versions without operator annotations).

    State tensors are outputs that:
    1. Are marked as "state" in the converter (preferred approach)
    2. Are produced by nodes in the component
    3. Are consumed by nodes in the component
    4. Create back-edges (appear as inputs to their own producers)

    Returns:
        (state_inputs, state_outputs) tensor name tuples
    """
    state_inputs: List[str] = []
    state_outputs: List[str] = []

    # Build builder for component to query state tensors
    full_builder = TensorMapBuilder.build(graph.layers)
    component_builder = full_builder.extract_for_nodes(component_nodes)

    # Strategy 1: Use explicitly detected state tensors (from converter)
    # This is the primary mechanism and is reliable when available
    annotated_state_tensors = list(
        component_builder.extract_state_tensors(graph, component_nodes)
    )

    if annotated_state_tensors:
        # Use annotated state information
        state_outputs.extend(annotated_state_tensors)
        state_inputs.extend(annotated_state_tensors)
        logger.debug(
            f"Region {component_nodes}: using {len(annotated_state_tensors)} "
            f"annotated state tensors: {annotated_state_tensors}"
        )
        return (tuple(state_inputs), tuple(state_outputs))

    # Strategy 2: Fall back to topology-based inference for back-edges
    # This handles cases where state detection wasn't possible
    logger.debug(
        f"Region {component_nodes}: no annotated state tensors; "
        f"falling back to topology analysis"
    )

    for node_name in component_nodes:
        layer = graph.layers[node_name]

        for out_tensor in layer.outputs:
            # Check if this output is consumed by nodes in the same component
            consumers = graph.tensor_consumers.get(out_tensor, [])
            component_consumers = [c for c in consumers if c in component_nodes]

            if not component_consumers:
                continue

            # This is a potential state tensor (internal to component)
            # It's a state output (produced here)
            if out_tensor not in state_outputs:
                state_outputs.append(out_tensor)

            # For each consumer in this component, it's a state input
            for consumer in component_consumers:
                if out_tensor not in layer.inputs:
                    # Only add as state input if it's actually consumed
                    if out_tensor not in state_inputs:
                        state_inputs.append(out_tensor)

    return (tuple(state_inputs), tuple(state_outputs))


def _infer_loop_tensors(
    component_nodes: Set[str],
    graph: NetworkIR,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    [DEPRECATED] Use RegionClassifier.classify() instead.

    This function is kept for backward compatibility. The RegionClassifier
    class consolidates all classification logic in one place, including
    loop tensor inference.

    Migration:
        Old: loop_in, loop_out = _infer_loop_tensors(nodes, graph)
        New: classifier = RegionClassifier(graph, layer_graph)
             classification = classifier.classify(nodes)
             loop_in = classification.loop_inputs
             loop_out = classification.loop_outputs

    Infer loop input/output tensors for a Loop region.

    For ONNX Loop/Scan operators, identify:
    - loop_inputs: tensors iterated over or loop state inputs
    - loop_outputs: tensors produced by loop body

    Returns:
        (loop_inputs, loop_outputs) tensor name tuples
    """
    loop_layer = None
    for node_name in component_nodes:
        layer = graph.layers[node_name]
        if layer.op_type in _CONTROL_FLOW_OPS:
            loop_layer = layer
            break

    if not loop_layer:
        return ((), ())

    # Loop layer inputs and outputs are the loop-related tensors
    loop_inputs = tuple(t for t in loop_layer.inputs if t not in graph.input_tensors)
    loop_outputs = tuple(loop_layer.outputs)

    return (loop_inputs, loop_outputs)


def _rebuild_merged_graph_structure(
    layers: Dict[str, object],
    execution_order: List[str],
    state_tensors_from_regions: Dict[str, str],
) -> NetworkIR:
    """
    Rebuild graph structure for merged acyclic regions.

    When merging regions, we need to recompute tensor producers/consumers
    since the merged region has different boundaries than the originals.
    Delegates to centralized tensor map builder for consistency.
    """
    tensor_producers, tensor_consumers = build_tensor_maps(layers)

    # Compute I/O tensors from tensor boundary sets.
    # Inputs: consumed in this graph but not produced by any layer in this graph.
    # Outputs: produced by this graph but not consumed by any layer in this graph.
    input_tensors = tuple(
        t for t in tensor_consumers.keys() if t not in tensor_producers
    )
    output_tensors = tuple(
        t for t in tensor_producers.keys() if t not in tensor_consumers
    )

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors=state_tensors_from_regions,
    )


def _merge_consecutive_acyclic_regions(regions: List) -> List:
    """
    Merge consecutive acyclic regions into single regions.

    Optimization: Reduces trivial region boundaries in pure feed-forward networks.
    Preserves recurrent and loop regions as natural boundaries.

    Algorithm:
    - Iterate through regions
    - Accumulate layers from consecutive ACYCLIC regions
    - When encountering a non-acyclic region, flush accumulated acyclic regions
    - Rebuild tensor structure for merged acyclic region

    Args:
        regions: List of region objects (AcyclicRegionIR, RecurrentRegionIR, LoopRegionIR)

    Returns:
        List of merged regions (fewer acyclic regions, same non-acyclic regions)
    """
    if not regions:
        return regions

    merged = []
    current_acyclic_layers = {}
    current_acyclic_order = []
    accumulated_state_tensors = {}

    for region in regions:
        if region.kind == RegionKind.ACYCLIC:
            # Accumulate layers and state info from this acyclic region
            current_acyclic_layers.update(region.graph.layers)
            current_acyclic_order.extend(region.graph.execution_order)
            accumulated_state_tensors.update(region.graph.state_tensors)

        else:
            # Non-acyclic region encountered: flush accumulated acyclic regions
            if current_acyclic_layers:
                merged_graph = _rebuild_merged_graph_structure(
                    current_acyclic_layers,
                    current_acyclic_order,
                    accumulated_state_tensors,
                )
                merged.append(
                    AcyclicRegionIR(
                        region_id="",  # Will be re-IDed later
                        kind=RegionKind.ACYCLIC,
                        graph=merged_graph,
                    )
                )

                # Reset accumulators for next batch of acyclic regions
                current_acyclic_layers = {}
                current_acyclic_order = []
                accumulated_state_tensors = {}

            # Add the non-acyclic region as-is
            merged.append(region)

    # Flush any remaining acyclic regions at the end
    if current_acyclic_layers:
        merged_graph = _rebuild_merged_graph_structure(
            current_acyclic_layers,
            current_acyclic_order,
            accumulated_state_tensors,
        )
        merged.append(
            AcyclicRegionIR(
                region_id="",  # Will be re-IDed later
                kind=RegionKind.ACYCLIC,
                graph=merged_graph,
            )
        )

    return merged


def regionize_network_ir(network_ir) -> ModelIR:
    """
    Partition graph into SCC-based typed regions.

    High-level flow:
    1. Analyze graph structure with LayerGraph
    2. Compute strongly connected components (SCCs)
    3. Classify each SCC using RegionClassifier
    4. Extract region-specific metadata (state tensors, loop tensors)
    5. Merge consecutive acyclic regions (optimization)
    6. Return partitioned ModelIR

    Args:
        network_ir: NetworkIR to regionize

    Returns:
        ModelIR with regions tuple and global I/O metadata
    """
    graph = NetworkIR(
        layers=network_ir.layers,
        execution_order=network_ir.execution_order,
        tensor_producers=network_ir.tensor_producers,
        tensor_consumers=network_ir.tensor_consumers,
        input_tensors=network_ir.input_tensors,
        output_tensors=network_ir.output_tensors,
        state_tensors=network_ir.state_tensors,
    )

    if not graph.layers:
        region = AcyclicRegionIR(region_id="r0", kind=RegionKind.ACYCLIC, graph=graph)
        return ModelIR(
            regions=(region,),
            input_tensors=graph.input_tensors,
            output_tensors=graph.output_tensors,
            metadata={
                "regionizer": "scc_partitioner",
                "region_count": "1",
                "scc_count": "1",
            },
        )

    # Analyze graph structure using LayerGraph (single source of truth)
    layer_graph = LayerGraph(graph)
    sccs = layer_graph.strongly_connected_components
    ordered_sccs = layer_graph.get_ordered_sccs()
    classifier = RegionClassifier(graph, layer_graph)

    regions = []

    for ridx, scc in enumerate(ordered_sccs):
        component_nodes = set(scc)

        # Classify component (unified logic in RegionClassifier)
        classification = classifier.classify(component_nodes)

        # Extract subgraph for this region
        subgraph = _subgraph_for_component(graph, component_nodes)
        region_id = f"r{ridx}"

        # Create region based on classification
        if classification.kind == RegionKind.ACYCLIC:
            region = AcyclicRegionIR(
                region_id=region_id, kind=classification.kind, graph=subgraph
            )

        elif classification.kind == RegionKind.RECURRENT:
            region = RecurrentRegionIR(
                region_id=region_id,
                kind=classification.kind,
                graph=subgraph,
                state_inputs=classification.state_inputs,
                state_outputs=classification.state_outputs,
            )

        else:  # LOOP
            region = LoopRegionIR(
                region_id=region_id,
                kind=classification.kind,
                graph=subgraph,
                loop_inputs=classification.loop_inputs,
                loop_outputs=classification.loop_outputs,
            )

        regions.append(region)

    # Merge consecutive acyclic regions (optimization: reduces trivial region boundaries)
    regions = _merge_consecutive_acyclic_regions(regions)

    # Re-ID regions after merging
    for idx, region in enumerate(regions):
        if isinstance(region, AcyclicRegionIR):
            region = AcyclicRegionIR(
                region_id=f"r{idx}", kind=region.kind, graph=region.graph
            )
        elif isinstance(region, RecurrentRegionIR):
            region = RecurrentRegionIR(
                region_id=f"r{idx}",
                kind=region.kind,
                graph=region.graph,
                state_inputs=region.state_inputs,
                state_outputs=region.state_outputs,
            )
        else:
            region = LoopRegionIR(
                region_id=f"r{idx}",
                kind=region.kind,
                graph=region.graph,
                loop_inputs=region.loop_inputs,
                loop_outputs=region.loop_outputs,
            )
        regions[idx] = region

    return ModelIR(
        regions=tuple(regions),
        input_tensors=graph.input_tensors,
        output_tensors=graph.output_tensors,
        metadata={
            "regionizer": "scc_partitioner",
            "region_count": str(len(regions)),
            "scc_count": str(len(sccs)),
        },
    )
