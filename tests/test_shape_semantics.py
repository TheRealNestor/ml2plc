import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir.converter import onnx_to_ir
from codegen.onnx_to_ir.shape import infer_layer_shapes
from codegen.onnx_to_ir.tensor_resolution import ResolvedTensor
from codegen.shape_semantics import ShapeSemanticsTracker, TensorRole


def test_shape_semantics_role_propagation_for_shape_chain():
    semantics = ShapeSemanticsTracker(
        {
            "idx": np.array(0, dtype=np.int64),
            "const_dim": np.array([8], dtype=np.int64),
        }
    )

    semantics.record_layer(
        {
            "name": "Shape__1",
            "op_type": "Shape",
            "inputs": ["x"],
            "outputs": ["x_shape"],
            "resolved_inputs": [],
        }
    )
    semantics.record_layer(
        {
            "name": "Gather__2",
            "op_type": "Gather",
            "inputs": ["x_shape", "idx"],
            "outputs": ["batch_dim"],
            "resolved_inputs": [],
        }
    )
    semantics.record_layer(
        {
            "name": "Concat__3",
            "op_type": "Concat",
            "inputs": ["batch_dim", "const_dim"],
            "outputs": ["target_shape"],
            "resolved_inputs": [],
        }
    )
    semantics.record_layer(
        {
            "name": "Reshape__4",
            "op_type": "Reshape",
            "inputs": ["x", "target_shape"],
            "outputs": ["x_reshaped"],
            "resolved_inputs": [],
        }
    )

    assert semantics.role_of("x_shape") == TensorRole.SHAPE
    assert semantics.role_of("batch_dim") == TensorRole.SHAPE
    assert semantics.role_of("target_shape") == TensorRole.SHAPE
    assert semantics.role_of("x_reshaped") == TensorRole.VALUE


def _build_invalid_runtime_rhs_matmul_with_lineage_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [32])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

    rhs_passthrough = helper.make_node(
        "Identity",
        inputs=["rhs"],
        outputs=["rhs_aliased"],
        name="Identity__rhs",
    )
    mm = helper.make_node(
        "MatMul",
        inputs=["x", "rhs_aliased"],
        outputs=["y"],
        name="MatMul__bad",
    )

    graph = helper.make_graph(
        [rhs_passthrough, mm],
        "runtime_rhs_matmul_invalid_with_lineage",
        [x, rhs],
        [y],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="shape-semantics-test",
    )
    onnx.checker.check_model(model)
    return model


def test_runtime_matmul_failure_contains_lineage(tmp_path):
    model = _build_invalid_runtime_rhs_matmul_with_lineage_model()
    model_path = tmp_path / "runtime_rhs_mm_bad_lineage.onnx"
    onnx.save(model, model_path)

    analyzer = ONNXModel(model_path)
    assert analyzer.load_model()

    with pytest.raises(ValueError) as exc_info:
        onnx_to_ir(analyzer)

    msg = str(exc_info.value)
    assert "incompatible shapes (1,) @ (32,)" in msg
    assert "Lineage (lhs='x')" in msg
    assert "Lineage (rhs='rhs_aliased')" in msg
    assert "Identity__rhs (Identity)" in msg


def test_transpose_on_shape_tensor_fails_when_perm_rank_mismatch():
    semantics = ShapeSemanticsTracker()
    semantics.record_layer(
        {
            "name": "Shape__seed",
            "op_type": "Shape",
            "inputs": ["x"],
            "outputs": ["x_shape"],
            "resolved_inputs": [],
        }
    )

    layer = {
        "name": "Transpose__shape_bad",
        "op_type": "Transpose",
        "attributes": {"perm": [0, 2]},
        "inputs": ["x_shape"],
        "outputs": ["x_shape_t"],
        "resolved_inputs": [
            ResolvedTensor(
                name="x_shape",
                shape=(3,),
                dtype="int64",
                size=3,
                value=None,
                is_weight=False,
            )
        ],
        "resolved_outputs": [],
        "_shape_semantics": semantics,
    }

    with pytest.raises(ValueError, match=r"Transpose 'Transpose__shape_bad':"):
        infer_layer_shapes(layer)


def test_transpose_on_value_tensor_keeps_backward_compatible_fallback():
    semantics = ShapeSemanticsTracker()
    layer = {
        "name": "Transpose__value_bad",
        "op_type": "Transpose",
        "attributes": {"perm": [0, 1]},
        "inputs": ["x"],
        "outputs": ["x_t"],
        "resolved_inputs": [
            ResolvedTensor(
                name="x",
                shape=(3,),
                dtype="float32",
                size=3,
                value=None,
                is_weight=False,
            )
        ],
        "resolved_outputs": [],
        "_shape_semantics": semantics,
    }

    in_shape, out_shape = infer_layer_shapes(layer)
    assert in_shape == (3,)
    assert out_shape == (3,)


def test_unsqueeze_on_shape_tensor_requires_axes():
    semantics = ShapeSemanticsTracker()
    semantics.record_layer(
        {
            "name": "Shape__seed_unsq",
            "op_type": "Shape",
            "inputs": ["x"],
            "outputs": ["x_shape"],
            "resolved_inputs": [],
        }
    )

    layer = {
        "name": "Unsqueeze__shape_bad",
        "op_type": "Unsqueeze",
        "attributes": {},
        "inputs": ["x_shape"],
        "outputs": ["x_shape_u"],
        "resolved_inputs": [
            ResolvedTensor(
                name="x_shape",
                shape=(3,),
                dtype="int64",
                size=3,
                value=None,
                is_weight=False,
            )
        ],
        "resolved_outputs": [],
        "_shape_semantics": semantics,
    }

    with pytest.raises(ValueError, match=r"Unsqueeze 'Unsqueeze__shape_bad':"):
        infer_layer_shapes(layer)


def test_expand_on_shape_tensor_fails_on_incompatible_broadcast():
    semantics = ShapeSemanticsTracker({"target": np.array([2, 3], dtype=np.int64)})
    semantics.record_layer(
        {
            "name": "Shape__seed_expand",
            "op_type": "Shape",
            "inputs": ["x"],
            "outputs": ["x_shape"],
            "resolved_inputs": [],
        }
    )

    layer = {
        "name": "Expand__shape_bad",
        "op_type": "Expand",
        "attributes": {},
        "inputs": ["x_shape", "target"],
        "outputs": ["x_shape_e"],
        "resolved_inputs": [
            ResolvedTensor(
                name="x_shape",
                shape=(4,),
                dtype="int64",
                size=4,
                value=None,
                is_weight=False,
            ),
            ResolvedTensor(
                name="target",
                shape=(2,),
                dtype="int64",
                size=2,
                value=np.array([2, 3], dtype=np.int64),
                is_weight=True,
            ),
        ],
        "resolved_outputs": [],
        "_shape_semantics": semantics,
    }

    with pytest.raises(ValueError, match=r"Expand 'Expand__shape_bad':"):
        infer_layer_shapes(layer)


def test_unsqueeze_on_value_tensor_keeps_backward_compatible_fallback_without_axes():
    semantics = ShapeSemanticsTracker()
    layer = {
        "name": "Unsqueeze__value_no_axes",
        "op_type": "Unsqueeze",
        "attributes": {},
        "inputs": ["x"],
        "outputs": ["x_u"],
        "resolved_inputs": [
            ResolvedTensor(
                name="x",
                shape=(3,),
                dtype="float32",
                size=3,
                value=None,
                is_weight=False,
            )
        ],
        "resolved_outputs": [],
        "_shape_semantics": semantics,
    }

    in_shape, out_shape = infer_layer_shapes(layer)
    assert in_shape == (3,)
    assert out_shape == (3,)
