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
            region = RecurrentRegionIR(
                region_id=region_id,
                kind=kind,
                graph=subgraph,
                state_inputs=(),
                state_outputs=(),
            )
        else:
            region = LoopRegionIR(
                region_id=region_id,
                kind=kind,
                graph=subgraph,
                loop_inputs=(),
                loop_outputs=(),
            )
        regions.append(region)

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
