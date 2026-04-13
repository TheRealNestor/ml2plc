import numpy as np

from codegen.memory_check.memory_analyzer import (
    _compute_layer_weights,
    _estimate_activation_memory,
)
from codegen.types import LSTMLayer, MatMulLayer, NetworkIR
from codegen.types import GRULayer


def _mk_ir(layers, execution_order, input_tensors=("x",), output_tensors=("y",)):
    tensor_producers = {}
    tensor_consumers = {}

    for layer_name in execution_order:
        layer = layers[layer_name]
        for out in layer.outputs:
            tensor_producers[out] = layer_name
        for inp in layer.inputs:
            tensor_consumers.setdefault(inp, []).append(layer_name)

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )


def test_compute_layer_weights_lstm_uses_explicit_tensor_bytes():
    w = np.zeros((1, 16, 8), dtype=np.float32)
    r = np.zeros((1, 16, 4), dtype=np.float32)
    b = np.zeros((1, 32), dtype=np.float32)

    layer = LSTMLayer(
        layer_id=1,
        name="lstm_0",
        op_type="LSTM",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=2,
        W=w,
        R=r,
        B=b,
    )

    weights_bytes, bias_bytes = _compute_layer_weights(layer)

    assert weights_bytes == w.nbytes + r.nbytes
    assert bias_bytes == b.nbytes


def test_activation_estimate_recurrent_includes_workspace():
    lstm = LSTMLayer(
        layer_id=1,
        name="lstm_0",
        op_type="LSTM",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=3,
        W=np.zeros((1, 16, 8), dtype=np.float32),
        R=np.zeros((1, 16, 4), dtype=np.float32),
        B=np.zeros((1, 32), dtype=np.float32),
    )

    ir = _mk_ir(
        layers={"lstm_0": lstm},
        execution_order=["lstm_0"],
        input_tensors=("x",),
        output_tensors=("y",),
    )

    estimated = _estimate_activation_memory(ir)
    plain_input_plus_output = (8 * 4) + (4 * 4)

    assert estimated > plain_input_plus_output


def test_activation_estimate_attention_named_matmul_is_more_conservative():
    attn_mm = MatMulLayer(
        layer_id=1,
        name="self_attn_qk_matmul",
        op_type="MatMul",
        input_size=8,
        output_size=8,
        inputs=("x",),
        outputs=("attn",),
        input_shape=(8,),
        output_shape=(8,),
        input_type="tensor(float)",
        output_type="tensor(float)",
        weights=np.zeros((8, 8), dtype=np.float32),
        bias=np.zeros((8,), dtype=np.float32),
    )

    ir = _mk_ir(
        layers={"attn": attn_mm},
        execution_order=["attn"],
        input_tensors=("x",),
        output_tensors=("attn",),
    )

    estimated = _estimate_activation_memory(ir)

    # baseline (input + output) = 32 + 32 = 64 bytes
    assert estimated > 64


def test_activation_estimate_gru_includes_recurrent_workspace():
    gru = GRULayer(
        layer_id=1,
        name="gru_0",
        op_type="GRU",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=3,
        W=np.zeros((1, 12, 8), dtype=np.float32),
        R=np.zeros((1, 12, 4), dtype=np.float32),
        B=np.zeros((1, 24), dtype=np.float32),
    )

    ir = _mk_ir(
        layers={"gru_0": gru},
        execution_order=["gru_0"],
        input_tensors=("x",),
        output_tensors=("y",),
    )

    estimated = _estimate_activation_memory(ir)
    plain_input_plus_output = (8 * 4) + (4 * 4)

    assert estimated > plain_input_plus_output


def test_compute_layer_weights_gru_uses_explicit_tensor_bytes():
    w = np.zeros((1, 12, 8), dtype=np.float32)
    r = np.zeros((1, 12, 4), dtype=np.float32)
    b = np.zeros((1, 24), dtype=np.float32)

    layer = GRULayer(
        layer_id=1,
        name="gru_0",
        op_type="GRU",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=2,
        W=w,
        R=r,
        B=b,
    )

    weights_bytes, bias_bytes = _compute_layer_weights(layer)

    assert weights_bytes == w.nbytes + r.nbytes
    assert bias_bytes == b.nbytes


def test_activation_with_buffer_allocations_keeps_conservative_peak():
    lstm = LSTMLayer(
        layer_id=1,
        name="lstm_0",
        op_type="LSTM",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=3,
        W=np.zeros((1, 16, 8), dtype=np.float32),
        R=np.zeros((1, 16, 4), dtype=np.float32),
        B=np.zeros((1, 32), dtype=np.float32),
    )

    ir = _mk_ir(
        layers={"lstm_0": lstm},
        execution_order=["lstm_0"],
        input_tensors=("x",),
        output_tensors=("y",),
    )

    peak_only = _estimate_activation_memory(ir)
    with_buffers = _estimate_activation_memory(ir, buffer_allocations={"y": "buf0"})

    assert with_buffers >= peak_only


def test_bidirectional_lstm_estimate_exceeds_forward_lstm():
    forward = LSTMLayer(
        layer_id=1,
        name="lstm_forward",
        op_type="LSTM",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=3,
        direction="forward",
        W=np.zeros((1, 16, 8), dtype=np.float32),
        R=np.zeros((1, 16, 4), dtype=np.float32),
        B=np.zeros((1, 32), dtype=np.float32),
    )

    bidirectional = LSTMLayer(
        layer_id=2,
        name="lstm_bidir",
        op_type="LSTM",
        input_size=8,
        output_size=4,
        inputs=("x",),
        outputs=("y",),
        input_shape=(1, 8),
        output_shape=(1, 4),
        input_type="tensor(float)",
        output_type="tensor(float)",
        hidden_size=4,
        sequence_length=3,
        direction="bidirectional",
        W=np.zeros((2, 16, 8), dtype=np.float32),
        R=np.zeros((2, 16, 4), dtype=np.float32),
        B=np.zeros((2, 32), dtype=np.float32),
    )

    forward_ir = _mk_ir(
        layers={"forward": forward},
        execution_order=["forward"],
        input_tensors=("x",),
        output_tensors=("y",),
    )
    bidir_ir = _mk_ir(
        layers={"bidir": bidirectional},
        execution_order=["bidir"],
        input_tensors=("x",),
        output_tensors=("y",),
    )

    forward_est = _estimate_activation_memory(forward_ir)
    bidir_est = _estimate_activation_memory(bidir_ir)

    assert bidir_est > forward_est
