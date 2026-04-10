"""
Variable section generation for recurrent layers (LSTM, GRU).

Declares gate buffers, state variables, and temporary variables needed by
recurrent layer code generation.
"""

from typing import List
from ...types import LSTMLayer, GRULayer, NetworkIR
from ..st_code import STCode, STCodeBuilder


def declare_recurrent_vars(
    network: NetworkIR,
    common_vars: List[str],
) -> STCode:
    """Generate VAR declarations for all recurrent layers in network.

    This function consolidates variable declarations for LSTM and GRU layers,
    eliminating the need for ad-hoc variable management in multiple places.

    Args:
        network: Network IR containing all layers
        common_vars: List of common variable names (e.g., ["t", "exp_val", "sum"])

    Returns:
        STCode with all recurrent variable declarations
    """
    builder = STCodeBuilder()
    has_lstm = any(
        isinstance(network.layers[ln], LSTMLayer) for ln in network.execution_order
    )
    has_gru = any(
        isinstance(network.layers[ln], GRULayer) for ln in network.execution_order
    )

    if not (has_lstm or has_gru):
        return STCode.empty()

    # Common loop/temp variables
    with builder.indent():
        builder.add_line("(* Recurrent layer variables *)")
        for var in common_vars:
            builder.add_line(f"{var} : REAL;")

    # LSTM-specific buffers
    if has_lstm:
        with builder.indent():
            builder.add_line("(* LSTM gate and state buffers *)")
            for ln in network.execution_order:
                layer = network.layers[ln]
                if isinstance(layer, LSTMLayer):
                    h_size = layer.hidden_size
                    builder.add_line(f"(* Layer {layer.layer_id} *)")
                    for gate in ("i_gate", "f_gate", "g_gate", "o_gate"):
                        builder.add_line(
                            f"{gate}_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                        )
                    for state in ("h_state", "c_state"):
                        builder.add_line(
                            f"{state}_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                        )

    # GRU-specific buffers
    if has_gru:
        with builder.indent():
            builder.add_line("(* GRU gate and state buffers *)")
            for ln in network.execution_order:
                layer = network.layers[ln]
                if isinstance(layer, GRULayer):
                    h_size = layer.hidden_size
                    builder.add_line(f"(* Layer {layer.layer_id} *)")
                    for gate in ("r_gate", "u_gate"):
                        builder.add_line(
                            f"{gate}_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                        )
                    for state in ("h_state", "h_new"):
                        builder.add_line(
                            f"{state}_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                        )

    return builder.build()
