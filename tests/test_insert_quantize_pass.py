"""Tests for InsertQuantizePass."""

from codegen.ir_optimizer.passes.insert_quantize import InsertQuantizePass
from codegen.types import FusedLinearLayer, NetworkIR, ModelIR, AcyclicRegionIR, RegionKind


def test_insert_quantize_before_fused_linear():
    fused = FusedLinearLayer(
        layer_id=0,
        name="fused",
        op_type="FusedLinear",
        input_size=4,
        output_size=2,
        inputs=("x",),
        outputs=("y",),
        weights=None,
        bias=None,
        activation=None,
    )

    layers = {fused.name: fused}
    execution_order = [fused.name]
    tensor_producers = {"y": fused.name}
    tensor_consumers = {"x": []}

    network = NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=("x",),
        output_tensors=("y",),
    )

    region = AcyclicRegionIR(region_id="r0", kind=RegionKind.ACYCLIC, graph=network)
    model = ModelIR(regions=(region,))

    # Run pass on the network of the region directly
    pass_inst = InsertQuantizePass()
    pass_inst.optimize(network)

    # Check that quantize/dequantize layers were inserted
    names = list(network.layers.keys())
    assert any("Quantize" in n for n in names)
    assert any("Dequantize" in n for n in names)
