"""
Shared utilities for recurrent layer code generation (LSTM, GRU).

Provides reusable components for gate computation, activation application,
and state management across different recurrent architectures.
"""

import logging
from ..st_code import STCode, STCodeBuilder

logger = logging.getLogger(__name__)


def apply_activation(builder: STCodeBuilder, var_name: str, activation: str) -> None:
    """Apply an activation function to a sum variable and store in var_name.

    Args:
        builder: Code builder to append to
        var_name: Variable to store result in (e.g., "i_gate_5[j]")
        activation: "sigmoid" or "tanh"
    """
    if activation == "sigmoid":
        builder.add_line(f"{var_name} := 1.0 / (1.0 + EXP(-sum));")
    elif activation == "tanh":
        builder.add_line(f"exp_val := EXP(2.0 * sum);")
        builder.add_line(f"{var_name} := (exp_val - 1.0) / (exp_val + 1.0);")
    else:
        raise ValueError(f"Unknown activation: {activation}")


def emit_gate_computation(
    builder: STCodeBuilder,
    gate_name: str,
    gate_var: str,
    layer_id: int,
    hidden_size: int,
    input_size: int,
    input_var: str,
    timestep_var: str,
    weights_suffix: str,
    recurrent_suffix: str,
    bias_suffix: str,
    activation: str,
    has_recurrent: bool = True,
) -> None:
    """Generate code for a single gate computation.

    Handles both input and recurrent contributions with configurable
    weight/bias naming conventions.

    Args:
        builder: Code builder
        gate_name: Human-readable name (e.g., "Input Gate")
        gate_var: Variable to store gate output
        layer_id: Layer identifier for naming
        hidden_size: Hidden state dimension
        input_size: Input feature dimension
        input_var: Name of input buffer (indexed as [t*input_size + i])
        timestep_var: Loop variable for timestep
        weights_suffix: Suffix for weights (e.g., "i" for input gate)
        recurrent_suffix: Suffix for recurrent weights (same or different)
        bias_suffix: Suffix for bias
        activation: "sigmoid" or "tanh"
        has_recurrent: Whether to include recurrent contribution
    """
    builder.add_line("")
    builder.add_line(f"(* {gate_name} *)")
    builder.add_line(f"FOR j := 0 TO {hidden_size - 1} DO")

    with builder.indent():
        builder.add_line("sum := 0.0;")

        # Input contribution
        builder.add_line(f"FOR i := 0 TO {input_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"sum := sum + {input_var}[{timestep_var} * {input_size} + i] "
                f"* weights_{layer_id}_{weights_suffix}[j * {input_size} + i];"
            )
        builder.add_line("END_FOR;")

        # Recurrent contribution
        if has_recurrent:
            builder.add_line(f"FOR i := 0 TO {hidden_size - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + h_state_{layer_id}[i] "
                    f"* recurrent_{layer_id}_{recurrent_suffix}[j * {hidden_size} + i];"
                )
            builder.add_line("END_FOR;")

        # Bias and activation
        builder.add_line(f"sum := sum + bias_{layer_id}_{bias_suffix}[j];")
        apply_activation(builder, f"{gate_var}[j]", activation)

    builder.add_line("END_FOR;")


def initialize_hidden_states(
    builder: STCodeBuilder,
    layer_id: int,
    hidden_size: int,
    states: list,
) -> None:
    """Generate code to initialize hidden/cell states to zero.

    Args:
        builder: Code builder
        layer_id: Layer identifier
        hidden_size: Hidden state dimension
        states: List of state names (e.g., ["h_state", "c_state"])
    """
    builder.add_line("")
    builder.add_line(f"(* Initialize states to zero *)")
    builder.add_line(f"FOR j := 0 TO {hidden_size - 1} DO")
    with builder.indent():
        for state in states:
            builder.add_line(f"{state}_{layer_id}[j] := 0.0;")
    builder.add_line("END_FOR;")


def write_output(
    builder: STCodeBuilder,
    output_var: str,
    state_var: str,
    layer_id: int,
    hidden_size: int,
    sequence_length: int,
    primary_output: str,
    timestep_var: str = "t",
) -> None:
    """Generate output writing code for recurrent layers.

    Handles three output modes:
    - "Y": Full sequence (write at each timestep)
    - "Y_h": Final hidden state (write only at last timestep)
    - "Y_c": Final cell state (write only at last timestep, LSTM only)

    Args:
        builder: Code builder
        output_var: Output buffer name
        state_var: State buffer name (h_state or c_state)
        layer_id: Layer identifier
        hidden_size: Hidden state dimension
        sequence_length: Length of sequence
        primary_output: "Y", "Y_h", or "Y_c"
        timestep_var: Loop variable name (default "t")
    """
    builder.add_line("")

    if primary_output == "Y":
        builder.add_line(f"(* Y output: Full sequence *)")
        builder.add_line(f"FOR j := 0 TO {hidden_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[{timestep_var} * {hidden_size} + j] := "
                f"{state_var}_{layer_id}[j];"
            )
        builder.add_line("END_FOR;")

    elif primary_output == "Y_h" or primary_output == "Y_c":
        desc = "hidden state" if primary_output == "Y_h" else "cell state"
        builder.add_line(f"(* {primary_output} output: Final {desc} only *)")
        builder.add_line(f"IF {timestep_var} = {sequence_length - 1} THEN")
        with builder.indent():
            builder.add_line(f"FOR j := 0 TO {hidden_size - 1} DO")
            with builder.indent():
                builder.add_line(f"{output_var}[j] := {state_var}_{layer_id}[j];")
            builder.add_line("END_FOR;")
        builder.add_line("END_IF;")

    else:
        raise ValueError(f"Unknown primary_output: {primary_output}")
