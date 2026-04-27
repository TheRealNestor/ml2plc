"""Tests for DeadVariableEliminationPass."""

from codegen.ir_optimizer.passes.dead_variable_elimination import DeadVariableEliminationPass
from codegen.types import BaseLayer, NetworkIR


def _layer(name, inputs=(), outputs=()):
    return BaseLayer(
        layer_id=0,
        name=name,
        op_type="Op",
        input_size=len(inputs),
        output_size=len(outputs),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def test_dead_variable_elim_removes_unconsumed_layer():
    a = _layer("A", inputs=("x",), outputs=("a",))
    b = _layer("B", inputs=("a",), outputs=("out",))
    c = _layer("C", inputs=("z",), outputs=("unused",))

    layers = {"A": a, "B": b, "C": c}
    execution_order = ["A", "B", "C"]
    tensor_producers = {"a": "A", "out": "B", "unused": "C"}
    tensor_consumers = {"x": ["A"], "a": ["B"], "z": []}

    net = NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=("x", "z"),
        output_tensors=("out",),
    )

    p = DeadVariableEliminationPass()
    p.optimize(net)

    assert p.should_remove("C")
    assert not p.should_remove("A")
    assert not p.should_remove("B")
