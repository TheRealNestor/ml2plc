"""
Model regionization utilities.

Partitions a NetworkIR into typed regions (acyclic, recurrent, loop) based on
strongly connected component (SCC) analysis.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from ..types import (
    NetworkIR,
    ModelIR,
    AcyclicRegionIR,
    RecurrentRegionIR,
    LoopRegionIR,
    RegionKind,
)
from ..graph_algorithms import tarjan_scc, topological_order_components
from ..ir_utils import (
    build_tensor_maps,
    filter_tensor_maps_for_nodes,
    extract_component_input_tensors,
    extract_component_output_tensors,
    extract_component_state_tensors,
)

logger = logging.getLogger(__name__)

_CONTROL_FLOW_OPS = {"Loop", "Scan", "If"}


def _build_layer_graph(
    graph: NetworkIR,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build producer->consumer and reverse adjacency over layer names."""
    adjacency: Dict[str, Set[str]] = {name: set() for name in graph.layers}
    reverse_adjacency: Dict[str, Set[str]] = {name: set() for name in graph.layers}

    for producer in graph.layers:
        layer = graph.layers[producer]
        for out_tensor in layer.outputs:
            for consumer in graph.tensor_consumers.get(out_tensor, []):
                if consumer in graph.layers and consumer != producer:
                    adjacency[producer].add(consumer)
                    reverse_adjacency[consumer].add(producer)

    return adjacency, reverse_adjacency


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

    # Extract internal tensor flow (using utility)
    tensor_producers, tensor_consumers = filter_tensor_maps_for_nodes(
        graph.tensor_producers,
        graph.tensor_consumers,
        component_nodes,
    )

    # Compute I/O tensors (using utilities)
    input_tensors_set = extract_component_input_tensors(
        graph, component_nodes, tensor_producers
    )
    input_tensors = tuple(sorted(input_tensors_set))
    output_tensors = extract_component_output_tensors(graph, component_nodes)

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
    adjacency: Dict[str, Set[str]],
    graph: NetworkIR,
) -> RegionKind:
    """
    Classify a component as one of: Loop, Recurrent, or Acyclic.

    Classification logic (priority order):
    1. If contains ONNX Loop/Scan/If operators → LOOP region
    2. If multiple nodes or self-loop → RECURRENT region
    3. Otherwise → ACYCLIC region
    """
    # Check for explicit control flow operators (highest priority)
    has_control_flow = any(
        graph.layers[n].op_type in _CONTROL_FLOW_OPS for n in component_nodes
    )
    if has_control_flow:
        return RegionKind.LOOP

    # Check for multiple nodes (indicating feedback within component)
    if len(component_nodes) > 1:
        return RegionKind.RECURRENT

    # Check for self-loop on single node
    single_node = next(iter(component_nodes))
    if single_node in adjacency.get(single_node, set()):
        return RegionKind.RECURRENT

    # No cycles, no control flow → purely acyclic
    return RegionKind.ACYCLIC


def _infer_state_tensors(
    component_nodes: Set[str],
    graph: NetworkIR,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
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
    # Strategy 1: Use explicitly detected state tensors (from converter)
    # This is the primary mechanism and is reliable when available
    annotated_state_tensors = list(
        extract_component_state_tensors(graph, component_nodes)
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

    # Compute I/O tensors: inputs are those produced nowhere, outputs are those consumed nowhere
    input_tensors = tuple(
        t for t in tensor_producers.keys() if t not in tensor_consumers
    )
    output_tensors = tuple(
        t for t in tensor_consumers.keys() if t not in tensor_producers
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
    """Partition graph into SCC-based typed regions."""
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

    adjacency, _ = _build_layer_graph(graph)
    sccs = tarjan_scc(adjacency)

    node_to_component: Dict[str, int] = {}
    for cid, component in enumerate(sccs):
        for node in component:
            node_to_component[node] = cid

    component_edges: Dict[int, Set[int]] = {cid: set() for cid in range(len(sccs))}
    for src, neighbors in adjacency.items():
        src_cid = node_to_component[src]
        for dst in neighbors:
            dst_cid = node_to_component[dst]
            if src_cid != dst_cid:
                component_edges[src_cid].add(dst_cid)

    ordered_component_ids = topological_order_components(component_edges)
    regions = []

    for ridx, cid in enumerate(ordered_component_ids):
        component_nodes = set(sccs[cid])
        kind = _classify_region_kind(component_nodes, adjacency, graph)
        subgraph = _subgraph_for_component(graph, component_nodes)
        region_id = f"r{ridx}"

        if kind == RegionKind.ACYCLIC:
            region = AcyclicRegionIR(region_id=region_id, kind=kind, graph=subgraph)
        elif kind == RegionKind.RECURRENT:
            state_inputs, state_outputs = _infer_state_tensors(component_nodes, graph)
            region = RecurrentRegionIR(
                region_id=region_id,
                kind=kind,
                graph=subgraph,
                state_inputs=state_inputs,
                state_outputs=state_outputs,
            )
        else:  # LOOP
            loop_inputs, loop_outputs = _infer_loop_tensors(component_nodes, graph)
            region = LoopRegionIR(
                region_id=region_id,
                kind=kind,
                graph=subgraph,
                loop_inputs=loop_inputs,
                loop_outputs=loop_outputs,
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
