"""Tests for TransposeWeightsPass."""

import numpy as np
from codegen.ir_optimizer.passes.transpose_weights import TransposeWeightsPass
from codegen.types import LinearLayer


def test_transpose_weights_applies_to_linear():
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2,3)
    layer = LinearLayer(
        layer_id=0,
        name="lin",
        op_type="Linear",
        input_size=3,
        output_size=2,
        inputs=("x",),
        outputs=("y",),
        weights=w,
        bias=None,
    )

    network = type("Net", (), {"layers": {"lin": layer}})()

    # The pass operates on NetworkIR; provide minimal mapping
    class DummyNet:
        def __init__(self, layers):
            self.layers = layers

    net = DummyNet({"lin": layer})

    p = TransposeWeightsPass()
    p.optimize(net)

    new_w = net.layers["lin"].weights
    assert new_w.shape == (3, 2)
    assert np.allclose(new_w, w.T)
