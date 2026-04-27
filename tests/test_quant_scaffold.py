"""Tests for the quantization scaffold module."""

import numpy as np
from codegen.types import (
    FusedLinearLayer,
    NetworkIR,
    AcyclicRegionIR,
    RegionKind,
    ModelIR,
)
from codegen.quantization.scaffold import apply_post_training_quantization


def test_apply_post_training_quantization_sets_scale_and_zp():
    # Create a small weight matrix and fused layer
    weights = np.array([[ -1.0, 0.0], [0.5, 2.0]], dtype=float)

    fused = FusedLinearLayer(
        layer_id=0,
        name="fused",
        op_type="FusedLinear",
        input_size=2,
        output_size=2,
        inputs=("x",),
        outputs=("y",),
        weights=weights,
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

    annotated = apply_post_training_quantization(model)

    annotated_layer = annotated.regions[0].graph.layers["fused"]

    assert hasattr(annotated_layer, "weight_scale")
    assert hasattr(annotated_layer, "weight_zero_point")
    assert float(annotated_layer.weight_scale) > 0
