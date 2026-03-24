"""
Model regionization utilities.
"""

from ..types import (
    ModelIR,
    RegionKind,
    AcyclicRegionIR,
    network_ir_to_graph_ir,
    NetworkIR,
)


def regionize_network_ir(network_ir: NetworkIR) -> ModelIR:
    """Convert NetworkIR into a regioned ModelIR."""
    graph_ir = network_ir_to_graph_ir(network_ir)
    region = AcyclicRegionIR(
        region_id="region_0", kind=RegionKind.ACYCLIC, graph=graph_ir
    )

    return ModelIR(
        regions=(region,),
        input_tensors=network_ir.input_tensors,
        output_tensors=network_ir.output_tensors,
        metadata={"regionizer": "single_acyclic_region"},
    )
