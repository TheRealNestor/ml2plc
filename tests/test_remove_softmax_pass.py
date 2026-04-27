"""Unit tests for RemoveSoftmaxPass (Softmax -> ArgMax opt-in pass)."""

from codegen.types import (
    ActivationLayer,
    ActivationType,
    ArgMaxLayer,
    NetworkIR,
)
from codegen.ir_optimizer import IROptimizer
from codegen.ir_optimizer.passes.remove_softmax import RemoveSoftmaxPass


def _make_network_ir_for_softmax():
    """Create a minimal NetworkIR containing a single softmax activation."""
    softmax = ActivationLayer(
        layer_id=0,
        name="softmax",
        op_type="Activation",
        input_size=1,
        output_size=1,
        inputs=("pred",),
        outputs=("prob",),
        activation=ActivationType.SOFTMAX,
    )

    layers = {softmax.name: softmax}
    execution_order = [softmax.name]
    tensor_producers = {"prob": softmax.name}
    tensor_consumers = {"pred": []}

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=("pred",),
        output_tensors=("prob",),
    )


def test_remove_softmax_replaced_by_argmax():
    ir = _make_network_ir_for_softmax()

    optimizer = IROptimizer(ir, passes=[RemoveSoftmaxPass()])
    result = optimizer.optimize()
    optimized_ir = result.ir

    # Find ArgMaxLayer in optimized IR
    argmax_layers = [l for l in optimized_ir.layers.values() if isinstance(l, ArgMaxLayer)]

    assert len(argmax_layers) == 1, "Expected a single ArgMaxLayer after pass"

    argmax = argmax_layers[0]
    assert argmax.inputs == ("pred",)
    assert argmax.outputs == ("prob",)
    assert argmax.axis == -1


def test_remove_softmax_from_fused_linear():
    """FusedLinearLayer with softmax should be split into linear + ArgMax."""
    from codegen.types import FusedLinearLayer

    # Build fused layer (weights can be dummy)
    fused = FusedLinearLayer(
        layer_id=0,
        name="fused",
        op_type="FusedLinear",
        input_size=10,
        output_size=5,
        inputs=("x",),
        outputs=("prob",),
        weights=None,
        bias=None,
        activation=ActivationType.SOFTMAX,
    )

    layers = {fused.name: fused}
    execution_order = [fused.name]
    tensor_producers = {"prob": fused.name}
    tensor_consumers = {"x": []}

    ir = NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=("x",),
        output_tensors=("prob",),
    )

    optimizer = IROptimizer(ir, passes=[RemoveSoftmaxPass()])
    result = optimizer.optimize()
    optimized_ir = result.ir

    # Expect one FusedLinearLayer (activation NONE) and one ArgMaxLayer
    fused_linear_layers = [l for l in optimized_ir.layers.values() if isinstance(l, FusedLinearLayer) and l.activation == ActivationType.NONE]
    argmax_layers = [l for l in optimized_ir.layers.values() if isinstance(l, ArgMaxLayer)]

    assert len(fused_linear_layers) == 1, "Expected a single linear layer without activation"
    assert len(argmax_layers) == 1, "Expected a single ArgMaxLayer"

    # ArgMax should produce the original output name
    assert argmax_layers[0].outputs == ("prob",)

