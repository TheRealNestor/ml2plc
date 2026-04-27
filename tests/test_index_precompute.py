"""Tests for IndexPrecomputationPass."""

import numpy as np
from codegen.ir_optimizer.passes.index_precomputation import IndexPrecomputationPass
from codegen.types import GatherLayer, SliceLayer


def test_gather_precomputes_indices():
    g = GatherLayer(
        layer_id=0,
        name="g",
        op_type="Gather",
        input_size=1,
        output_size=1,
        inputs=("x",),
        outputs=("y",),
        gather_axis=0,
        indices=np.array([0, 2, 3]),
    )

    net = type("N", (), {"layers": {"g": g}})()
    p = IndexPrecomputationPass()
    p.optimize(net)

    assert hasattr(net.layers["g"], "precomputed_indices")
    assert np.array_equal(net.layers["g"].precomputed_indices, np.array([0, 2, 3]))


def test_slice_precomputes_slice():
    s = SliceLayer(
        layer_id=0,
        name="s",
        op_type="Slice",
        input_size=1,
        output_size=1,
        inputs=("x",),
        outputs=("y",),
        starts=[0],
        ends=[10],
        axes=[0],
    )

    net = type("N", (), {"layers": {"s": s}})()
    p = IndexPrecomputationPass()
    p.optimize(net)

    assert hasattr(net.layers["s"], "precomputed_slice")
    ps = net.layers["s"].precomputed_slice
    assert (ps["starts"][0] == 0) and (ps["ends"][0] == 10)
