"""Tests for ConstantFoldingPass."""

import numpy as np
from codegen.ir_optimizer.passes.constant_folding import ConstantFoldingPass
from codegen.types import BinaryElementwiseLayer, NetworkIR


def test_constant_folding_attaches_folded_constant():
    # Create a binary elementwise layer with rhs_const and no producers for inputs
    layer = BinaryElementwiseLayer(
        layer_id=0,
        name="be",
        op_type="Mul",
        input_size=1,
        output_size=1,
        inputs=("const_in",),
        outputs=("out",),
        operation="Mul",
        rhs_const=np.array([2.0]),
    )

    net = NetworkIR(layers={"be": layer}, execution_order=["be"], tensor_producers={}, tensor_consumers={}, input_tensors=("const_in",), output_tensors=("out",))

    p = ConstantFoldingPass()
    p.optimize(net)

    assert hasattr(net.layers["be"], "folded_constant")
    assert np.array_equal(net.layers["be"].folded_constant, np.array([2.0]))
