"""
Recurrent layer code generators (LSTM, GRU).

Handles sequence processing with internal state management.
"""

import logging
from ...types import LSTMLayer, GRULayer
from ..st_code import STCode, STCodeBuilder

logger = logging.getLogger(__name__)


def generate_lstm_code(layer: LSTMLayer, input_var: str, output_var: str) -> STCode:
    """Generate LSTM layer code with full temporal unrolling."""
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    seq_len = layer.sequence_length
    lid = layer.layer_id

    builder.add_line(
        f"(* Layer {lid}: LSTM (hidden_size={h_size}, input_size={x_size}, seq_len={seq_len}) *)"
    )

    # Determine which output to generate
    primary_output = layer.primary_output or "Y"
    builder.add_line(f"(* Generating output: {primary_output} *)")

    # Debug logging
    logger.debug(
        f"Generating LSTM code: layer_id={lid}, h_size={h_size}, x_size={x_size}, "
        f"seq_len={seq_len}, primary_output={primary_output}, "
        f"input_var={input_var}, output_var={output_var}"
    )
    logger.debug(
        f"  W shape={layer.W.shape if layer.W is not None else 'None'}, "
        f"R shape={layer.R.shape if layer.R is not None else 'None'}, "
        f"B shape={layer.B.shape if layer.B is not None else 'None'}"
    )

    # Initialize states
    builder.add_line("")
    builder.add_line(f"(* Initialize LSTM states to zero *)")
    builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
    with builder.indent():
        builder.add_line(f"h_state_{lid}[j] := 0.0;")
        builder.add_line(f"c_state_{lid}[j] := 0.0;")
    builder.add_line("END_FOR;")

    # Timestep loop
    builder.add_line("")
    builder.add_line(f"(* Process {seq_len} timesteps *)")
    builder.add_line(f"FOR t := 0 TO {seq_len - 1} DO")
    with builder.indent():
        # Helper to emit gate computation
        def emit_gate(gate_name, var_name, activation):
            builder.add_line("")
            builder.add_line(f"(* {gate_name} *)")
            builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
            with builder.indent():
                builder.add_line("sum := 0.0;")
                builder.add_line(f"FOR i := 0 TO {x_size - 1} DO")
                with builder.indent():
                    gate_letter = var_name.split("_")[0]
                    builder.add_line(
                        f"sum := sum + {input_var}[t * {x_size} + i] "
                        f"* weights_{lid}_{gate_letter}[j * {x_size} + i];"
                    )
                builder.add_line("END_FOR;")
                builder.add_line(f"FOR i := 0 TO {h_size - 1} DO")
                with builder.indent():
                    builder.add_line(
                        f"sum := sum + h_state_{lid}[i] "
                        f"* recurrent_{lid}_{gate_letter}[j * {h_size} + i];"
                    )
                builder.add_line("END_FOR;")
                builder.add_line(f"sum := sum + bias_{lid}_{gate_letter}[j];")
                if activation == "sigmoid":
                    builder.add_line(f"{var_name}_{lid}[j] := 1.0 / (1.0 + EXP(-sum));")
                elif activation == "tanh":
                    builder.add_line(f"exp_val := EXP(2.0 * sum);")
                    builder.add_line(
                        f"{var_name}_{lid}[j] := (exp_val - 1.0) / (exp_val + 1.0);"
                    )
            builder.add_line("END_FOR;")

        emit_gate("Input Gate", "i_gate", "sigmoid")
        emit_gate("Forget Gate", "f_gate", "sigmoid")
        emit_gate("Cell Gate", "g_gate", "tanh")
        emit_gate("Output Gate", "o_gate", "sigmoid")

        # Cell state update
        builder.add_line("")
        builder.add_line(f"(* Cell State: c_t = f_t * c_{{t-1}} + i_t * g_t *)")
        builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"c_state_{lid}[j] := f_gate_{lid}[j] * c_state_{lid}[j] "
                f"+ i_gate_{lid}[j] * g_gate_{lid}[j];"
            )
        builder.add_line("END_FOR;")

        # Hidden state update
        builder.add_line("")
        builder.add_line(f"(* Hidden State: h_t = o_t * tanh(c_t) *)")
        builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
        with builder.indent():
            builder.add_line(f"exp_val := EXP(2.0 * c_state_{lid}[j]);")
            builder.add_line(
                f"h_state_{lid}[j] := o_gate_{lid}[j] "
                f"* (exp_val - 1.0) / (exp_val + 1.0);"
            )
        builder.add_line("END_FOR;")

        # Output based on which output is being generated
        builder.add_line("")
        if primary_output == "Y":
            # Y output: full sequence of hidden states
            # The ONNX LSTM Y output has shape (seq_len, hidden_size) after batch stripping
            # Write at offset t*hidden_size to accumulate all timesteps
            builder.add_line(
                f"(* Y output: Store h_t to output buffer at position t*{h_size} *)"
            )
            builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
            with builder.indent():
                builder.add_line(f"{output_var}[t * {h_size} + j] := h_state_{lid}[j];")
            builder.add_line("END_FOR;")
        elif primary_output == "Y_h":
            # Y_h output: only final hidden state (written on last timestep only)
            builder.add_line(
                f"(* Y_h output: Store final h_state only on last timestep *)"
            )
            builder.add_line(f"IF t = {seq_len - 1} THEN")
            with builder.indent():
                builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
                with builder.indent():
                    builder.add_line(f"{output_var}[j] := h_state_{lid}[j];")
                builder.add_line("END_FOR;")
            builder.add_line("END_IF;")
        elif primary_output == "Y_c":
            # Y_c output: only final cell state (written on last timestep only)
            builder.add_line(
                f"(* Y_c output: Store final c_state only on last timestep *)"
            )
            builder.add_line(f"IF t = {seq_len - 1} THEN")
            with builder.indent():
                builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
                with builder.indent():
                    builder.add_line(f"{output_var}[j] := c_state_{lid}[j];")
                builder.add_line("END_FOR;")
            builder.add_line("END_IF;")

    builder.add_line("END_FOR;")

    return builder.build()


def generate_gru_code(layer: GRULayer, input_var: str, output_var: str) -> STCode:
    """Generate GRU (Gated Recurrent Unit) layer code."""
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size

    builder.add_line(f"(* Layer {layer.layer_id}: GRU (hidden_size={h_size}) *)")
    builder.add_line(
        "(* Note: This is a single GRU timestep; state carried by region wrapper *)"
    )

    builder.add_line("")
    builder.add_line(f"(* Reset gate: r_t = sigmoid(sum) *)")
    builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
    with builder.indent():
        builder.add_line("sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {x_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"sum := sum + {input_var}[i] * weights_{layer.layer_id}[i * {h_size} + j];"
            )
        builder.add_line("END_FOR;")
        builder.add_line(f"sum := sum + bias_{layer.layer_id}[j];")
        builder.add_line(f"{output_var}[j] := 1.0 / (1.0 + EXP(-sum));")
    builder.add_line("END_FOR;")

    builder.add_line("")
    builder.add_line(
        "(* NOTE: This is a simplified GRU for demo. Full implementation requires state variables *)"
    )

    return builder.build()
