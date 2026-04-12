import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from codegen.ir_to_st.codegen_core import translate_ir_to_st
from codegen.ir_to_st.layers.data_movement import generate_einsum_code
from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir.layer_extractors import _extract_einsum_layer
from codegen.onnx_to_ir.tensor_resolution import ResolvedTensor
from codegen.onnx_to_ir.converter import onnx_to_ir
from codegen.types import EinsumLayer, ReduceProdLayer, RuntimeMatMulLayer


def _build_runtime_rhs_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])

    t = helper.make_node(
        "Transpose", inputs=["x"], outputs=["xt"], name="xt", perm=[1, 0]
    )
    mm = helper.make_node(
        "MatMul", inputs=["x", "xt"], outputs=["y"], name="runtime_mm"
    )

    graph = helper.make_graph([t, mm], "runtime_rhs_matmul", [x], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="runtime-matmul-test",
    )
    onnx.checker.check_model(model)
    return model


def _build_invalid_runtime_rhs_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [32])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

    mm = helper.make_node(
        "MatMul", inputs=["x", "rhs"], outputs=["y"], name="runtime_mm_bad"
    )

    graph = helper.make_graph([mm], "runtime_rhs_matmul_invalid", [x, rhs], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="runtime-matmul-test",
    )
    onnx.checker.check_model(model)
    return model


def _build_reduce_prod_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1])

    node = helper.make_node(
        "ReduceProd",
        inputs=["x"],
        outputs=["y"],
        name="rp",
        axes=[-1],
        keepdims=1,
    )

    graph = helper.make_graph([node], "reduce_prod_model", [x], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="reduce-prod-test",
    )
    onnx.checker.check_model(model)
    return model


def _build_supported_einsum_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 32])

    rhs_data = np.random.randn(4, 8, 32).astype(np.float32)
    rhs = helper.make_tensor("rhs", TensorProto.FLOAT, [4, 8, 32], rhs_data.flatten())

    node = helper.make_node(
        "Einsum",
        inputs=["x", "rhs"],
        outputs=["y"],
        name="einsum_abcd_cde_to_abe",
        equation="abcd,cde->abe",
    )

    graph = helper.make_graph([node], "einsum_model", [x], [y], [rhs])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="einsum-test",
    )
    onnx.checker.check_model(model)
    return model


def test_runtime_rhs_matmul_extracts_and_generates_st():
    model = _build_runtime_rhs_matmul_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "runtime_rhs_mm.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        network_ir = onnx_to_ir(analyzer)
        layers = [network_ir.layers[name] for name in network_ir.execution_order]

        runtime_mm = next(l for l in layers if l.name == "runtime_mm")
        assert isinstance(runtime_mm, RuntimeMatMulLayer)
        assert runtime_mm.input_shape == (2, 3)
        assert runtime_mm.rhs_shape == (3, 2)
        assert runtime_mm.output_shape == (2, 2)

        st_code = translate_ir_to_st(network_ir, fb_name="RuntimeMatMul")
        assert "Runtime MatMul" in st_code
        assert "MOD 2" in st_code


def test_runtime_rhs_matmul_invalid_shapes_fail_early():
    model = _build_invalid_runtime_rhs_matmul_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "runtime_rhs_mm_bad.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        with pytest.raises(ValueError, match=r"incompatible shapes \(1,\) @ \(32,\)"):
            onnx_to_ir(analyzer)


def test_reduce_prod_extracts_and_generates_st():
    model = _build_reduce_prod_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "reduce_prod.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        network_ir = onnx_to_ir(analyzer)
        layers = [network_ir.layers[name] for name in network_ir.execution_order]

        reduce_prod = layers[0]
        assert isinstance(reduce_prod, ReduceProdLayer)
        assert reduce_prod.output_shape == (1, 2, 1)

        st_code = translate_ir_to_st(network_ir, fb_name="ReduceProdModel")
        assert "ReduceProd" in st_code
    assert "output_data[j] := 1.0;" in st_code
    assert "output_data[j] := output_data[j] * input_data[i];" in st_code


def test_einsum_extracts_and_generates_st():
    rhs = np.random.randn(4, 8, 32).astype(np.float32)
    enriched = {
        "name": "einsum_fallback",
        "op_type": "Einsum",
        "attributes": {"equation": "abcd,cde->abe"},
        "resolved_inputs": [
            ResolvedTensor(
                name="lhs",
                shape=(1,),
                dtype="float32",
                size=1,
                value=None,
                is_weight=False,
            ),
            ResolvedTensor(
                name="rhs",
                shape=rhs.shape,
                dtype="float32",
                size=int(rhs.size),
                value=rhs,
                is_weight=True,
            ),
        ],
        "resolved_outputs": [
            ResolvedTensor(
                name="y",
                shape=(),
                dtype="float32",
                size=32,
                value=None,
                is_weight=False,
            )
        ],
        "inputs": ["lhs", "rhs"],
        "outputs": ["y"],
    }

    einsum_layer = _extract_einsum_layer(enriched, 17, analyzer=None)
    assert isinstance(einsum_layer, EinsumLayer)
    assert einsum_layer.output_shape == (1, 1, 32)

    st = str(generate_einsum_code(einsum_layer, "lhs_buf", "out_buf"))
    assert "Einsum abcd,cde->abe" in st
    assert "einsum_rhs_17" in st
