"""
Model regionization utilities.

Partitions a NetworkIR into typed regions (acyclic, recurrent, loop) based on
strongly connected component (SCC) analysis.
"""

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
    """Create a region-local graph for a component."""
    layers = {
        name: graph.layers[name]
        for name in graph.execution_order
        if name in component_nodes
    }

    tensor_producers = {
        tensor: producer
        for tensor, producer in graph.tensor_producers.items()
        if producer in component_nodes
    }

    tensor_consumers = {
        tensor: [c for c in consumers if c in component_nodes]
        for tensor, consumers in graph.tensor_consumers.items()
        if any(c in component_nodes for c in consumers)
    }

    input_tensors = tuple(
        t
        for t in graph.input_tensors
        if any(c in component_nodes for c in graph.tensor_consumers.get(t, []))
    )
    output_tensors = tuple(
        t
        for t in graph.output_tensors
        if graph.tensor_producers.get(t) in component_nodes
    )

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
    )


def _classify_region_kind(
    component_nodes: Set[str],
    adjacency: Dict[str, Set[str]],
    graph: NetworkIR,
) -> RegionKind:
    """Classify a component as acyclic/recurrent/loop."""
    if any(graph.layers[n].op_type in _CONTROL_FLOW_OPS for n in component_nodes):
        return RegionKind.LOOP

    if len(component_nodes) > 1:
        return RegionKind.RECURRENT

    only = next(iter(component_nodes))
    if only in adjacency.get(only, set()):
        return RegionKind.RECURRENT

    return RegionKind.ACYCLIC


def _infer_state_tensors(
    component_nodes: Set[str],
    graph: NetworkIR,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Infer state input/output tensors for a recurrent region.

    State tensors are outputs that:
    1. Are produced by nodes in the component
    2. Are consumed by nodes in the component
    3. Create back-edges (appear as inputs to their own producers)

    Returns:
        (state_inputs, state_outputs) tensor name tuples
    """
    state_inputs: List[str] = []
    state_outputs: List[str] = []

    # For each node in component, check if any outputs create back-edges
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


def _merge_consecutive_acyclic_regions(regions: List) -> List:
    """
    Merge consecutive acyclic regions into single regions.

    Optimization: Reduces trivial region boundaries in pure feed-forward networks.
    Preserves recurrent and loop regions as boundaries.

    Args:
        regions: List of region objects (may be AcyclicRegionIR, RecurrentRegionIR, LoopRegionIR)

    Returns:
        List of merged regions
    """
    if not regions:
        return regions

    merged = []
    current_acyclic_layers = {}
    current_acyclic_order = []

    for region in regions:
        if region.kind == RegionKind.ACYCLIC:
            # Accumulate acyclic layers
            current_acyclic_layers.update(region.graph.layers)
            current_acyclic_order.extend(region.graph.execution_order)
        else:
            # Non-acyclic region: flush any accumulated acyclic regions
            if current_acyclic_layers:
                # Create merged acyclic region
                merged_graph = NetworkIR(
                    layers=current_acyclic_layers,
                    execution_order=current_acyclic_order,
                    tensor_producers=region.graph.tensor_producers.copy(),  # Will be updated
                    tensor_consumers=region.graph.tensor_consumers.copy(),  # Will be updated
                    input_tensors=(),  # Will be computed
                    output_tensors=(),  # Will be computed
                )
                # Rebuild tensor maps for merged region
                tensor_producers = {}
                tensor_consumers = {}
                for layer in current_acyclic_layers.values():
                    for out_t in layer.outputs:
                        tensor_producers[out_t] = layer.name
                    for in_t in layer.inputs:
                        if in_t not in tensor_consumers:
                            tensor_consumers[in_t] = []
                        tensor_consumers[in_t].append(layer.name)

                merged_graph = NetworkIR(
                    layers=current_acyclic_layers,
                    execution_order=current_acyclic_order,
                    tensor_producers=tensor_producers,
                    tensor_consumers=tensor_consumers,
                    input_tensors=tuple(
                        t for t in tensor_producers.keys() if t not in tensor_consumers
                    ),
                    output_tensors=tuple(
                        t for t in tensor_consumers.keys() if t not in tensor_producers
                    ),
                )
                merged.append(
                    AcyclicRegionIR(
                        region_id="",  # Will be re-IDed later
                        kind=RegionKind.ACYCLIC,
                        graph=merged_graph,
                    )
                )
                current_acyclic_layers = {}
                current_acyclic_order = []

            # Add the non-acyclic region
            merged.append(region)

    # Flush any remaining acyclic regions
    if current_acyclic_layers:
        tensor_producers = {}
        tensor_consumers = {}
        for layer in current_acyclic_layers.values():
            for out_t in layer.outputs:
                tensor_producers[out_t] = layer.name
            for in_t in layer.inputs:
                if in_t not in tensor_consumers:
                    tensor_consumers[in_t] = []
                tensor_consumers[in_t].append(layer.name)

        merged_graph = NetworkIR(
            layers=current_acyclic_layers,
            execution_order=current_acyclic_order,
            tensor_producers=tensor_producers,
            tensor_consumers=tensor_consumers,
            input_tensors=tuple(
                t for t in tensor_producers.keys() if t not in tensor_consumers
            ),
            output_tensors=tuple(
                t for t in tensor_consumers.keys() if t not in tensor_producers
            ),
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
