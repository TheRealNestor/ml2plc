"""
Recurrent layer code generators (LSTM, GRU).

Handles sequence processing with internal state management.
"""

import logging
from ...types import LSTMLayer, GRULayer
from ..st_code import STCode, STCodeBuilder
from .recurrent_utils import (
    apply_activation,
    emit_gate_computation,
    initialize_hidden_states,
    write_output,
)

logger = logging.getLogger(__name__)


def generate_lstm_code(layer: LSTMLayer, input_var: str, output_var: str) -> STCode:
    """Generate LSTM layer code with full temporal unrolling.

    LSTM equations:
    - i_t = sigmoid(W_i @ x_t + R_i @ h_{t-1} + b_i)           # input gate
    - f_t = sigmoid(W_f @ x_t + R_f @ h_{t-1} + b_f)           # forget gate
    - g_t = tanh(W_g @ x_t + R_g @ h_{t-1} + b_g)              # cell gate
    - o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o)           # output gate
    - c_t = f_t * c_{t-1} + i_t * g_t                          # cell state
    - h_t = o_t * tanh(c_t)                                     # hidden state
    """
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    seq_len = layer.sequence_length
    lid = layer.layer_id
    primary_output = layer.primary_output or "Y"

    builder.add_line(
        f"(* Layer {lid}: LSTM (hidden_size={h_size}, input_size={x_size}, seq_len={seq_len}) *)"
    )
    builder.add_line(f"(* Generating output: {primary_output} *)")

    logger.debug(
        f"LSTM code: layer_id={lid}, h={h_size}, x={x_size}, seq={seq_len}, "
        f"output={primary_output}"
    )

    # Initialize states
    initialize_hidden_states(builder, lid, h_size, ["h_state", "c_state"])

    # Timestep loop
    builder.add_line("")
    builder.add_line(f"(* Process {seq_len} timesteps *)")
    builder.add_line(f"FOR t := 0 TO {seq_len - 1} DO")

    with builder.indent():
        # Generate four gates
        for gate_name, var_name, suffix, activation in [
            ("Input Gate", "i_gate", "i", "sigmoid"),
            ("Forget Gate", "f_gate", "f", "sigmoid"),
            ("Cell Gate", "g_gate", "g", "tanh"),
            ("Output Gate", "o_gate", "o", "sigmoid"),
        ]:
            emit_gate_computation(
                builder,
                gate_name=gate_name,
                gate_var=f"{var_name}_{lid}",
                layer_id=lid,
                hidden_size=h_size,
                input_size=x_size,
                input_var=input_var,
                timestep_var="t",
                weights_suffix=suffix,
                recurrent_suffix=suffix,
                bias_suffix=suffix,
                activation=activation,
            )

        # Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
        builder.add_line("")
        builder.add_line(f"(* Cell state: c_t = f_t * c_{{t-1}} + i_t * g_t *)")
        builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"c_state_{lid}[j] := f_gate_{lid}[j] * c_state_{lid}[j] + "
                f"i_gate_{lid}[j] * g_gate_{lid}[j];"
            )
        builder.add_line("END_FOR;")

        # Hidden state update: h_t = o_t * tanh(c_t)
        builder.add_line("")
        builder.add_line(f"(* Hidden state: h_t = o_t * tanh(c_t) *)")
        builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
        with builder.indent():
            builder.add_line(f"exp_val := EXP(2.0 * c_state_{lid}[j]);")
            builder.add_line(
                f"h_state_{lid}[j] := o_gate_{lid}[j] * "
                f"((exp_val - 1.0) / (exp_val + 1.0));"
            )
        builder.add_line("END_FOR;")

        # Output
        write_output(
            builder,
            output_var=output_var,
            state_var="h_state",
            layer_id=lid,
            hidden_size=h_size,
            sequence_length=seq_len,
            primary_output=primary_output,
            timestep_var="t",
        )

    builder.add_line("END_FOR;")
    return builder.build()


def generate_gru_code(layer: GRULayer, input_var: str, output_var: str) -> STCode:
    """Generate GRU (Gated Recurrent Unit) layer code with full temporal unrolling.

    GRU equations:
    - r_t = sigmoid(W_r @ x_t + R_r @ h_{t-1} + b_r)           # reset gate
    - u_t = sigmoid(W_u @ x_t + R_u @ h_{t-1} + b_u)           # update gate
    - h'_t = tanh(W_h @ x_t + r_t * (R_h @ h_{t-1}) + b_h)     # candidate (lbr=0)
    - h'_t = tanh(W_h @ x_t + R_h @ (r_t * h_{t-1}) + b_h)     # candidate (lbr=1)
    - h_t = (1 - u_t) * h'_t + u_t * h_{t-1}                   # hidden state

    Where lbr = linear_before_reset attribute.
    """
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    seq_len = layer.sequence_length
    lid = layer.layer_id
    primary_output = layer.primary_output or "Y"
    lbr = getattr(layer, "linear_before_reset", 0)

    builder.add_line(
        f"(* Layer {lid}: GRU (hidden_size={h_size}, input_size={x_size}, seq_len={seq_len}) *)"
    )
    builder.add_line(
        f"(* Generating output: {primary_output}, linear_before_reset={lbr} *)"
    )

    logger.debug(
        f"GRU code: layer_id={lid}, h={h_size}, x={x_size}, seq={seq_len}, "
        f"output={primary_output}, lbr={lbr}"
    )

    # Initialize hidden state
    initialize_hidden_states(builder, lid, h_size, ["h_state"])

    # Timestep loop
    builder.add_line("")
    builder.add_line(f"(* Process {seq_len} timesteps *)")
    builder.add_line(f"FOR t := 0 TO {seq_len - 1} DO")

    with builder.indent():
        # Reset gate: r_t = sigmoid(W_r @ x_t + R_r @ h_{t-1} + b_r)
        emit_gate_computation(
            builder,
            gate_name="Reset Gate",
            gate_var=f"r_gate_{lid}",
            layer_id=lid,
            hidden_size=h_size,
            input_size=x_size,
            input_var=input_var,
            timestep_var="t",
            weights_suffix="r",
            recurrent_suffix="r",
            bias_suffix="r",
            activation="sigmoid",
        )

        # Update gate: u_t = sigmoid(W_u @ x_t + R_u @ h_{t-1} + b_u)
        emit_gate_computation(
            builder,
            gate_name="Update Gate",
            gate_var=f"u_gate_{lid}",
            layer_id=lid,
            hidden_size=h_size,
            input_size=x_size,
            input_var=input_var,
            timestep_var="t",
            weights_suffix="u",
            recurrent_suffix="u",
            bias_suffix="u",
            activation="sigmoid",
        )

        # Candidate hidden state (differs based on linear_before_reset)
        _emit_gru_candidate_gate(
            builder,
            layer_id=lid,
            hidden_size=h_size,
            input_size=x_size,
            input_var=input_var,
            linear_before_reset=lbr,
        )

        # Update hidden state: h_t = (1 - u_t) * h'_t + u_t * h_{t-1}
        builder.add_line("")
        builder.add_line(
            f"(* Hidden state: h_t = u_t * h_{{t-1}} + (1 - u_t) * h'_t *)"
        )
        builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"h_state_{lid}[j] := u_gate_{lid}[j] * h_state_{lid}[j] + "
                f"(1.0 - u_gate_{lid}[j]) * h_new_{lid}[j];"
            )
        builder.add_line("END_FOR;")

        # Output
        write_output(
            builder,
            output_var=output_var,
            state_var="h_state",
            layer_id=lid,
            hidden_size=h_size,
            sequence_length=seq_len,
            primary_output=primary_output,
            timestep_var="t",
        )

    builder.add_line("END_FOR;")
    return builder.build()


def _emit_gru_candidate_gate(
    builder: STCodeBuilder,
    layer_id: int,
    hidden_size: int,
    input_size: int,
    input_var: str,
    linear_before_reset: int,
) -> None:
    """Generate candidate hidden state gate for GRU.

    This helper isolates the complex linear_before_reset logic.
    """
    lid = layer_id
    h_size = hidden_size
    x_size = input_size

    builder.add_line("")
    if linear_before_reset == 1:
        builder.add_line(
            f"(* Candidate (lbr=1): h'_t = tanh(Wx + r_t * (Rh + Rb_h) + Wb_h) *)"
        )
    else:
        builder.add_line(
            f"(* Candidate (lbr=0): h'_t = tanh(Wx + R(r_t * h) + Wb_h + Rb_h) *)"
        )

    builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
    with builder.indent():
        builder.add_line("sum := 0.0;")

        # Input contribution: W_h @ x_t
        builder.add_line(f"FOR i := 0 TO {x_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"sum := sum + {input_var}[t * {x_size} + i] "
                f"* weights_{lid}_h[j * {x_size} + i];"
            )
        builder.add_line("END_FOR;")

        # Recurrent contribution depends on linear_before_reset
        if linear_before_reset == 1:
            builder.add_line("exp_val := 0.0;")
            builder.add_line(f"FOR i := 0 TO {h_size - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"exp_val := exp_val + h_state_{lid}[i] "
                    f"* recurrent_{lid}_h[j * {h_size} + i];"
                )
            builder.add_line("END_FOR;")
            builder.add_line(
                f"sum := sum + r_gate_{lid}[j] * (exp_val + bias_r_{lid}_h[j]);"
            )
            builder.add_line(f"sum := sum + bias_w_{lid}_h[j];")
        else:
            builder.add_line(f"FOR i := 0 TO {h_size - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + (h_state_{lid}[i] * r_gate_{lid}[i]) "
                    f"* recurrent_{lid}_h[j * {h_size} + i];"
                )
            builder.add_line("END_FOR;")
            builder.add_line(f"sum := sum + bias_w_{lid}_h[j] + bias_r_{lid}_h[j];")

        # Apply tanh activation
        apply_activation(builder, f"h_new_{lid}[j]", "tanh")

    builder.add_line("END_FOR;")
