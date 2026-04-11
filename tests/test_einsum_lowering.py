import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir.converter import onnx_to_ir
from codegen.onnx_to_ir.einsum_lowering import lower_supported_einsum_layers


def test_lower_abcd_cde_to_abe_rewrites_to_core_ops():
    class _AnalyzerStub:
        def __init__(self):
            self.tensor_info = {
                "x": {"shape": [1, 2, 3, 4]},
                "w": {"shape": [3, 4, 5]},
            }
            self.weights = {"w": np.random.randn(3, 4, 5).astype(np.float32)}

    analyzer = _AnalyzerStub()
    layers = [
        {
            "name": "einsum0",
            "op_type": "Einsum",
            "inputs": ["x", "w"],
            "outputs": ["y"],
            "attributes": {"equation": "abcd,cde->abe"},
        }
    ]
    constants = {}

    lowered, report = lower_supported_einsum_layers(layers, analyzer, constants)

    assert report.lowered_count == 1
    assert report.skipped_count == 0
    assert [l["op_type"] for l in lowered] == ["Reshape", "MatMul", "Reshape"]

    shape_values = {
        tuple(np.asarray(v).tolist())
        for v in constants.values()
        if np.asarray(v).ndim == 1
    }
    assert (2, 12) in shape_values
    assert (1, 2, 5) in shape_values
    # RHS reshaped weight should be materialized as a constant matrix
    assert any(v.shape == (12, 5) for v in constants.values())


def test_onnx_to_ir_handles_supported_einsum_equation_via_lowering():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 5])

    w_arr = np.random.randn(3, 4, 5).astype(np.float32)
    w = helper.make_tensor("w", TensorProto.FLOAT, [3, 4, 5], w_arr.flatten().tolist())

    einsum = helper.make_node(
        "Einsum",
        inputs=["x", "w"],
        outputs=["y"],
        name="einsum_main",
        equation="abcd,cde->abe",
    )

    graph = helper.make_graph([einsum], "einsum_graph", [x], [y], initializer=[w])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="einsum-lowering-test",
    )
    onnx.checker.check_model(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "einsum.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        network_ir = onnx_to_ir(analyzer)
        op_types = [
            network_ir.layers[name].op_type for name in network_ir.execution_order
        ]

        assert "Einsum" not in op_types
        assert "MatMul" in op_types
        assert op_types.count("Reshape") >= 2


def test_lower_abcd_cde_to_abe_uses_output_rhs_shape_fallback_when_lhs_missing():
    class _AnalyzerStub:
        def __init__(self):
            # No x shape available -> triggers fallback path.
            self.tensor_info = {
                "y": {"shape": [1, 20, 32]},
                "w": {"shape": [4, 8, 32]},
            }
            self.weights = {"w": np.random.randn(4, 8, 32).astype(np.float32)}

    analyzer = _AnalyzerStub()
    layers = [
        {
            "name": "einsum_fallback",
            "op_type": "Einsum",
            "inputs": ["x", "w"],
            "outputs": ["y"],
            "attributes": {"equation": "abcd,cde->abe"},
        }
    ]
    constants = {}

    lowered, report = lower_supported_einsum_layers(layers, analyzer, constants)

    assert report.lowered_count == 1
    assert [l["op_type"] for l in lowered] == ["Reshape", "MatMul", "Reshape"]
    # Reconstructed from y=(1,20,32) and w=(4,8,32): x=(1,20,4,8)
    assert any(v.shape == (32, 32) for v in constants.values())
