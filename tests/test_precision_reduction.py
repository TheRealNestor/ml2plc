"""Tests for PrecisionReductionPass."""

import numpy as np
from codegen.ir_optimizer.passes.precision_reduction import PrecisionReductionPass
from codegen.types import LinearLayer


def test_precision_reduction_rounds_weights_and_preserves_dtype():
    w = np.array([0.123456789, -1.987654321], dtype=np.float64)
    b = np.array([0.00098765], dtype=np.float32)

    layer = LinearLayer(
        layer_id=0,
        name="lin",
        op_type="Linear",
        input_size=2,
        output_size=1,
        inputs=("x",),
        outputs=("y",),
        weights=w,
        bias=b,
    )

    class Net:
        def __init__(self, layers):
            self.layers = layers

    net = Net({"lin": layer})

    p = PrecisionReductionPass(decimals=3)
    p.optimize(net)

    new_w = net.layers["lin"].weights
    new_b = net.layers["lin"].bias

    assert new_w.dtype == w.dtype
    assert new_b.dtype == b.dtype

    assert np.allclose(new_w, np.round(w, 3))
    assert np.allclose(new_b, np.round(b, 3))
