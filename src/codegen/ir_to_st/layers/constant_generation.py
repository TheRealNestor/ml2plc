"""
Shared constant generation for recurrent layers (LSTM, GRU).

Handles weight and bias constant declaration generation with a parameterized
approach to reduce duplication across different recurrent architectures.
"""

import numpy as np
import logging
from typing import Tuple, Dict
from ..st_code import STCode, STCodeBuilder
from ..utils.constant_helpers import generate_array_constant

logger = logging.getLogger(__name__)


def generate_recurrent_weight_constants(
    layer_id: int,
    hidden_size: int,
    input_size: int,
    W: np.ndarray,
    R: np.ndarray,
    B: np.ndarray,
    gate_order: Tuple[str, ...],
    gate_to_onnx_index: Dict[str, int],
    has_separate_biases: bool = False,
    separate_bias_gates: Tuple[str, ...] = (),
) -> STCode:
    """Generate weight constants for any recurrent layer with configurable gates.

    This unified function eliminates duplication between LSTM and GRU by
    accepting gate configuration as parameters.

    Args:
        layer_id: Layer identifier
        hidden_size: Hidden state dimension
        input_size: Input feature dimension
        W: Input weights (num_gates*hidden_size, input_size)
        R: Recurrent weights (num_gates*hidden_size, hidden_size)
        B: Bias vectors (varies by layer)
        gate_order: Ordered gate names (e.g., ("i", "f", "g", "o") for LSTM)
        gate_to_onnx_index: Mapping gate name to ONNX index
        has_separate_biases: Whether layer uses separate Wb/Rb biases (GRU-specific)
        separate_bias_gates: Which gates have separate Wb/Rb (e.g., ("h",) for GRU)

    Returns:
        STCode with weight constant declarations
    """
    builder = STCodeBuilder()
    weight_type = "REAL"

    for gate in gate_order:
        onnx_idx = gate_to_onnx_index[gate]
        start, end = onnx_idx, onnx_idx + 1

        # Input weight
        W_gate = W[start * hidden_size : end * hidden_size, :]
        builder.add_code(
            generate_array_constant(
                f"weights_{layer_id}_{gate}",
                W_gate.flatten(),
                weight_type,
                is_integer=False,
            )
        )

        # Recurrent weight
        R_gate = R[start * hidden_size : end * hidden_size, :]
        builder.add_code(
            generate_array_constant(
                f"recurrent_{layer_id}_{gate}",
                R_gate.flatten(),
                weight_type,
                is_integer=False,
            )
        )

        # Bias handling
        if has_separate_biases and gate in separate_bias_gates:
            # GRU candidate gate: separate Wb and Rb
            if B.shape[0] == 6 * hidden_size:
                Wb = B[: 3 * hidden_size]
                Rb = B[3 * hidden_size :]
            else:
                # Compatibility path for older single-bias exports
                Wb = B
                Rb = np.zeros_like(B)

            Wb_gate = Wb[start * hidden_size : end * hidden_size]
            Rb_gate = Rb[start * hidden_size : end * hidden_size]

            builder.add_code(
                generate_array_constant(
                    f"bias_w_{layer_id}_{gate}",
                    Wb_gate,
                    weight_type,
                    is_integer=False,
                )
            )
            builder.add_code(
                generate_array_constant(
                    f"bias_r_{layer_id}_{gate}",
                    Rb_gate,
                    weight_type,
                    is_integer=False,
                )
            )
        else:
            # Standard combined bias
            if B.shape[0] == 2 * len(gate_order) * hidden_size:
                # LSTM format: [Wb_i, Wb_f, Wb_g, Wb_o, Rb_i, Rb_f, Rb_g, Rb_o]
                Wb = B[: len(gate_order) * hidden_size]
                Rb = B[len(gate_order) * hidden_size :]
                bias = (
                    Wb[start * hidden_size : end * hidden_size]
                    + Rb[start * hidden_size : end * hidden_size]
                )
            else:
                # Standard format
                bias = B[start * hidden_size : end * hidden_size]

            builder.add_code(
                generate_array_constant(
                    f"bias_{layer_id}_{gate}",
                    bias,
                    weight_type,
                    is_integer=False,
                )
            )

    return builder.build()
