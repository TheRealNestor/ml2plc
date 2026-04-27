"""Tests for prune, fold-quantized, loop-unrolling and buffer-min passes."""

import numpy as np
from codegen.ir_optimizer.passes.prune_weights import PruneWeightsPass
from codegen.ir_optimizer.passes.fold_quantized_weights import FoldQuantizedWeightsPass
from codegen.ir_optimizer.passes.loop_unrolling import LoopUnrollingPass
from codegen.ir_optimizer.passes.buffer_minimization import BufferMinimizationPass
from codegen.types import LinearLayer, NetworkIR


def test_prune_weights_zeros_small_values():
    w = np.array([0.5, 0.0001, -0.00001, 1.0], dtype=np.float32)
    layer = LinearLayer(
        layer_id=0,
        name="lin",
        op_type="Linear",
        input_size=4,
        output_size=1,
        inputs=("x",),
        outputs=("y",),
        weights=w,
        bias=None,
    )

    net = type("N", (), {"layers": {"lin": layer}})()
    p = PruneWeightsPass(threshold=1e-3)
    p.optimize(net)

    new_w = net.layers["lin"].weights
    assert new_w.dtype == w.dtype
    assert np.isclose(new_w[1], 0.0)
    assert np.isclose(new_w[2], 0.0)


def test_fold_quantized_weights_produces_uint8():
    w = np.array([0.0, 1.0], dtype=np.float32)
    layer = LinearLayer(
        layer_id=0,
        name="lin2",
        op_type="Linear",
        input_size=2,
        output_size=1,
        inputs=("x",),
        outputs=("y",),
        weights=w,
        bias=None,
    )
    object.__setattr__(layer, "weight_scale", np.array(0.01))
    object.__setattr__(layer, "weight_zero_point", np.array(128))

    net = type("N", (), {"layers": {"lin2": layer}})()
    p = FoldQuantizedWeightsPass()
    p.optimize(net)

    new_w = net.layers["lin2"].weights
    assert new_w.dtype == np.uint8


def test_loop_unrolling_marks_small_loops():
    loop = type("L", (), {"op_type": "Loop", "trip": 3, "name": "loop1"})()
    net = type("N", (), {"layers": {"loop1": loop}})()
    p = LoopUnrollingPass(max_trip_count=5)
    p.optimize(net)
    assert getattr(net.layers["loop1"], "unrolled", False) is True


def test_buffer_minimization_groups_equal_sizes():
    # Create fake layers with output_shape attribute
    a = type("A", (), {"name": "a", "outputs": ("t1",), "output_shape": (2, 2)})()
    b = type("B", (), {"name": "b", "outputs": ("t2",), "output_shape": (2, 2)})()

    layers = {"a": a, "b": b}
    execution_order = ["a", "b"]
    tensor_producers = {"t1": "a", "t2": "b"}
    tensor_consumers = {}


    net = NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=(),
        output_tensors=("t1", "t2"),
    )

    p = BufferMinimizationPass()
    p.optimize(net)

    # Both tensors should map into some buffers (same size -> possibly same buffer)
    assert isinstance(p.buffer_assignments, dict)
    assert "t1" in p.buffer_assignments and "t2" in p.buffer_assignments
