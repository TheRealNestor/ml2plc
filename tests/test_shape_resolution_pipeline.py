import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir.shape import validate_model_shapes


def _make_dynamic_batch_matmul_model() -> onnx.ModelProto:
    """Create a tiny model with symbolic/dynamic batch for validation tests."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])

    w_arr = np.random.randn(4, 3).astype(np.float32)
    w = helper.make_tensor("W", TensorProto.FLOAT, [4, 3], w_arr.flatten().tolist())

    node = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"], name="mm")
    graph = helper.make_graph([node], "dynamic_batch_mm", [x], [y], initializer=[w])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="shape-resolution-test",
    )
    onnx.checker.check_model(model)
    return model


def test_validate_model_shapes_resolves_dynamic_batch_in_place():
    model = _make_dynamic_batch_matmul_model()

    ok, model_copy, changes, diagnostics = validate_model_shapes(model)
    if model_copy is not None:
        model = model_copy

    x_dims = model.graph.input[0].type.tensor_type.shape.dim
    y_dims = model.graph.output[0].type.tensor_type.shape.dim

    assert x_dims[0].dim_value == 1
    assert not x_dims[0].dim_param
    assert y_dims[0].dim_value == 1
    assert not y_dims[0].dim_param


def test_analyzer_refresh_after_shape_mutation_rebuilds_tensor_cache():
    model = _make_dynamic_batch_matmul_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "dynamic_mm.onnx"
        onnx.save(model, model_path)

        analyzer = ONNXModel(model_path)
        assert analyzer.load_model()

        # Before validation+refresh, symbolic batch should still be present in tensor_info.
        before_shape = analyzer.tensor_info["x"]["shape"]
        assert before_shape[0] == "batch"

        ok, model_copy, changes, diagnostics = validate_model_shapes(analyzer.model)
        if model_copy is not None:
            # The analyzer holds the original model; ensure we don't mutate it
            # in this test. The test checks that refresh_after_model_mutation()
            # rebuilds caches after an external mutation.
            analyzer.model = model_copy

        # Cache is stale until explicit refresh.
        stale_shape = analyzer.tensor_info["x"]["shape"]
        assert stale_shape[0] == "batch"

        analyzer.refresh_after_model_mutation()

        refreshed_shape = analyzer.tensor_info["x"]["shape"]
        assert refreshed_shape[0] == 1


def _make_gru_dynamic_hidden_state_model(hidden_size: int = 4) -> onnx.ModelProto:
    """Create a tiny GRU model where initial_h has a dynamic hidden dimension."""
    seq_len = 3
    batch = 1
    input_size = 2
    num_directions = 1

    x = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [seq_len, batch, input_size]
    )
    initial_h = helper.make_tensor_value_info(
        "initial_h", TensorProto.FLOAT, [num_directions, batch, "dyn_hidden"]
    )

    y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [seq_len, num_directions, batch, hidden_size]
    )
    y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [num_directions, batch, hidden_size]
    )

    w_arr = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(
        np.float32
    )
    r_arr = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(
        np.float32
    )
    b_arr = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32)

    w = helper.make_tensor("W", TensorProto.FLOAT, w_arr.shape, w_arr.flatten())
    r = helper.make_tensor("R", TensorProto.FLOAT, r_arr.shape, r_arr.flatten())
    b = helper.make_tensor("B", TensorProto.FLOAT, b_arr.shape, b_arr.flatten())

    gru_node = helper.make_node(
        "GRU",
        inputs=["X", "W", "R", "B", "", "initial_h"],
        outputs=["Y", "Y_h"],
        name="gru_dynamic_state",
        hidden_size=hidden_size,
    )

    graph = helper.make_graph(
        [gru_node],
        "gru_dynamic_state_model",
        [x, initial_h],
        [y, y_h],
        initializer=[w, r, b],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="shape-resolution-test",
    )
    onnx.checker.check_model(model)
    return model


def test_validate_model_shapes_resolves_recurrent_hidden_dim_from_hidden_size():
    model = _make_gru_dynamic_hidden_state_model(hidden_size=7)

    ok, model_copy, changes, diagnostics = validate_model_shapes(model)
    if model_copy is not None:
        model = model_copy

    initial_h_dims = model.graph.input[1].type.tensor_type.shape.dim
    assert initial_h_dims[2].dim_value == 7
    assert not initial_h_dims[2].dim_param


def test_validate_model_shapes_resolves_recurrent_hidden_dim_through_transpose():
    hidden_size = 5
    seq_len = 2
    batch = 1
    input_size = 3
    num_directions = 1

    x = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [seq_len, batch, input_size]
    )
    initial_h = helper.make_tensor_value_info(
        "initial_h", TensorProto.FLOAT, [num_directions, batch, "dyn_hidden"]
    )
    y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [seq_len, num_directions, batch, hidden_size]
    )
    y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [num_directions, batch, hidden_size]
    )

    w_arr = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(
        np.float32
    )
    r_arr = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(
        np.float32
    )
    b_arr = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32)

    w = helper.make_tensor("W", TensorProto.FLOAT, w_arr.shape, w_arr.flatten())
    r = helper.make_tensor("R", TensorProto.FLOAT, r_arr.shape, r_arr.flatten())
    b = helper.make_tensor("B", TensorProto.FLOAT, b_arr.shape, b_arr.flatten())

    transpose_state = helper.make_node(
        "Transpose",
        inputs=["initial_h"],
        outputs=["state_t"],
        name="state_passthrough",
        perm=[0, 1, 2],
    )

    gru_node = helper.make_node(
        "GRU",
        inputs=["X", "W", "R", "B", "", "state_t"],
        outputs=["Y", "Y_h"],
        name="gru_dynamic_state_indirect",
        hidden_size=hidden_size,
    )

    graph = helper.make_graph(
        [transpose_state, gru_node],
        "gru_dynamic_state_indirect_model",
        [x, initial_h],
        [y, y_h],
        initializer=[w, r, b],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="shape-resolution-test",
    )
    onnx.checker.check_model(model)

    ok, model_copy, changes, diagnostics = validate_model_shapes(model)
    if model_copy is not None:
        model = model_copy

    initial_h_dims = model.graph.input[1].type.tensor_type.shape.dim
    assert initial_h_dims[2].dim_value == hidden_size
    assert not initial_h_dims[2].dim_param


def test_validate_model_shapes_resolves_recurrent_axis1_batch_to_one():
    hidden_size = 6
    seq_len = 2
    batch = 1
    input_size = 3
    num_directions = 1

    x = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [seq_len, batch, input_size]
    )
    # Dynamic axis 1 simulates sequence-major tensors where batch is not axis 0.
    state = helper.make_tensor_value_info(
        "state", TensorProto.FLOAT, [1, "dyn_batch", hidden_size]
    )
    y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [seq_len, num_directions, batch, hidden_size]
    )
    y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [num_directions, batch, hidden_size]
    )

    w_arr = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(
        np.float32
    )
    r_arr = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(
        np.float32
    )
    b_arr = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32)

    w = helper.make_tensor("W", TensorProto.FLOAT, w_arr.shape, w_arr.flatten())
    r = helper.make_tensor("R", TensorProto.FLOAT, r_arr.shape, r_arr.flatten())
    b = helper.make_tensor("B", TensorProto.FLOAT, b_arr.shape, b_arr.flatten())

    gru_node = helper.make_node(
        "GRU",
        inputs=["X", "W", "R", "B", "", "state"],
        outputs=["Y", "Y_h"],
        name="gru_dynamic_axis1_batch",
        hidden_size=hidden_size,
    )

    graph = helper.make_graph(
        [gru_node],
        "gru_dynamic_axis1_batch_model",
        [x, state],
        [y, y_h],
        initializer=[w, r, b],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="shape-resolution-test",
    )
    onnx.checker.check_model(model)

    ok, model_copy, changes, diagnostics = validate_model_shapes(model)
    if model_copy is not None:
        model = model_copy

    state_dims = model.graph.input[1].type.tensor_type.shape.dim
    assert state_dims[1].dim_value == 1
    assert not state_dims[1].dim_param
