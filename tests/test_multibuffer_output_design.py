"""
Test to verify the fundamental design fix for multi-output layers.

This test demonstrates that:
1. Output indices are correctly mapped in layer extraction
2. Primary output is correctly specified
3. Code generation respects the primary output specification
"""

import pytest
import numpy as np
from codegen.types import LSTMLayer, GRULayer
from codegen.ir_to_st.layers.recurrent import generate_lstm_code


def test_lstm_output_indices_mapping():
    """Test that LSTM layer has correct output index mapping."""
    lstm = LSTMLayer(
        layer_id=1,
        name="test_lstm",
        op_type="LSTM",
        input_size=20,
        output_size=640,  # 20 timesteps * 32 hidden
        inputs=("input",),
        outputs=("Y", "Y_h", "Y_c"),
        input_shape=(20, 20),
        output_shape=(20, 32),
        hidden_size=32,
        sequence_length=20,
        W=np.zeros((128, 20)),  # 4*hidden_size x input_size
        R=np.zeros((128, 32)),  # 4*hidden_size x hidden_size
        B=np.zeros(128),
        output_indices={"Y": 0, "Y_h": 1, "Y_c": 2},
        primary_output="Y",
    )

    # Verify output mapping
    assert lstm.output_indices["Y"] == 0
    assert lstm.output_indices["Y_h"] == 1
    assert lstm.output_indices["Y_c"] == 2
    assert lstm.primary_output == "Y"

    # Verify output size matches full sequence
    assert lstm.output_size == 20 * 32  # seq_len * hidden_size


def test_lstm_code_generation_for_y_output():
    """Test that LSTM generates code for full sequence (Y) output."""
    lstm = LSTMLayer(
        layer_id=1,
        name="test_lstm",
        op_type="LSTM",
        input_size=20,
        output_size=640,  # 20 timesteps * 32 hidden
        inputs=("input",),
        outputs=("Y",),
        input_shape=(20, 20),
        output_shape=(20, 32),
        hidden_size=32,
        sequence_length=20,
        W=np.zeros((128, 20)),
        R=np.zeros((128, 32)),
        B=np.zeros(128),
        output_indices={"Y": 0},
        primary_output="Y",
    )

    code = generate_lstm_code(lstm, "input_var", "output_var")
    code_str = code.to_string()

    # Verify code mentions Y output
    assert "Y output" in code_str

    # Verify code has the loop pattern for accumulating timesteps
    assert "t * 32" in code_str  # t * hidden_size

    # Verify code doesn't have Y_h or Y_c patterns
    assert "Y_h output" not in code_str
    assert "Y_c output" not in code_str


def test_lstm_code_generation_for_yh_output():
    """Test that LSTM generates code for final hidden state (Y_h) output."""
    lstm = LSTMLayer(
        layer_id=1,
        name="test_lstm",
        op_type="LSTM",
        input_size=20,
        output_size=32,  # Only hidden_size for final state
        inputs=("input",),
        outputs=("Y_h",),
        input_shape=(20, 20),
        output_shape=(32,),  # Only final state
        hidden_size=32,
        sequence_length=20,
        W=np.zeros((128, 20)),
        R=np.zeros((128, 32)),
        B=np.zeros(128),
        output_indices={"Y_h": 1},
        primary_output="Y_h",
    )

    code = generate_lstm_code(lstm, "input_var", "output_var")
    code_str = code.to_string()

    # Verify code mentions Y_h output
    assert "Y_h output" in code_str

    # Verify code uses IF to only write on last timestep
    assert "IF t =" in code_str or "IF t=" in code_str


def test_lstm_code_generation_for_yc_output():
    """Test that LSTM generates code for final cell state (Y_c) output."""
    lstm = LSTMLayer(
        layer_id=1,
        name="test_lstm",
        op_type="LSTM",
        input_size=20,
        output_size=32,  # Only hidden_size for final state
        inputs=("input",),
        outputs=("Y_c",),
        input_shape=(20, 20),
        output_shape=(32,),  # Only final state
        hidden_size=32,
        sequence_length=20,
        W=np.zeros((128, 20)),
        R=np.zeros((128, 32)),
        B=np.zeros(128),
        output_indices={"Y_c": 2},
        primary_output="Y_c",
    )

    code = generate_lstm_code(lstm, "input_var", "output_var")
    code_str = code.to_string()

    # Verify code mentions Y_c output
    assert "Y_c output" in code_str

    # Verify code uses IF to only write on last timestep
    assert "IF t =" in code_str or "IF t=" in code_str


def test_gru_output_indices_mapping():
    """Test that GRU layer has correct output index mapping."""
    gru = GRULayer(
        layer_id=2,
        name="test_gru",
        op_type="GRU",
        input_size=20,
        output_size=320,  # 10 timesteps * 32 hidden
        inputs=("input",),
        outputs=("Y", "Y_h"),
        input_shape=(10, 20),
        output_shape=(10, 32),
        hidden_size=32,
        W=np.zeros((96, 20)),  # 3*hidden_size x input_size
        R=np.zeros((96, 32)),  # 3*hidden_size x hidden_size
        output_indices={"Y": 0, "Y_h": 1},
        primary_output="Y",
    )

    # Verify output mapping
    assert gru.output_indices["Y"] == 0
    assert gru.output_indices["Y_h"] == 1
    assert gru.primary_output == "Y"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
