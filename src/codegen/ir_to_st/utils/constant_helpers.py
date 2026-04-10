"""
Constant declaration generation utilities.

Provides reusable patterns for generating weight, bias, quantization parameter,
and BatchNorm constant declarations in Structured Text.

These utilities are used by multiple layer generators and consolidate common
patterns for declaring arrays and scalars with initialization values.
"""

from typing import Union
import numpy as np
from ..st_code import STCode, STCodeBuilder
from ..type_conversion import numpy_to_plc_type, plc_type_from_onnx_dtype


def generate_array_constant(
    name: str,
    values: np.ndarray,
    plc_type: str,
    is_integer: bool = False,
) -> STCode:
    """
    Generate an array constant declaration with initialization.

    Args:
        name: Variable name
        values: NumPy array of values
        plc_type: PLC type for array elements (e.g., "REAL", "DINT")
        is_integer: If True, format values as integers; otherwise as floats

    Returns:
        STCode with: NAME : ARRAY[0..N-1] OF TYPE := [values];

    Example:
        >>> weights = np.array([1.0, 2.0, 3.0])
        >>> code = generate_array_constant("w1", weights, "REAL")
        >>> print(code)
        w1 : ARRAY[0..2] OF REAL := [1.000000, 2.000000, 3.000000];
    """
    flat_values = values.flatten()

    if flat_values.size == 0:
        return STCode.from_lines(
            f"(* WARNING: {name} has 0 elements — likely indicates a bug *)"
        )

    if is_integer:
        value_str = ", ".join(str(int(val)) for val in flat_values)
    else:
        value_str = ", ".join(f"{val:.6f}" for val in flat_values)

    return STCode.from_lines(
        f"{name} : ARRAY[0..{flat_values.size - 1}] OF {plc_type} := [{value_str}];"
    )


def generate_scalar_constant(
    name: str,
    value: Union[float, int],
    plc_type: str,
    is_integer: bool = False,
) -> STCode:
    """
    Generate a scalar constant declaration.

    Args:
        name: Variable name
        value: Scalar value
        plc_type: PLC type (e.g., "REAL", "DINT")
        is_integer: If True, format as integer; otherwise as float

    Returns:
        STCode with: NAME : TYPE := value;

    Example:
        >>> code = generate_scalar_constant("scale", 2.5, "REAL")
        >>> print(code)
        scale : REAL := 2.500000;
    """
    if is_integer:
        value_str = str(int(value))
    else:
        value_str = str(float(value))

    return STCode.from_lines(f"{name} : {plc_type} := {value_str};")


def is_uniform_array(arr: np.ndarray) -> bool:
    """
    Check if all elements in array are identical.

    Used to optimize storage of quantization parameters — if all values are
    the same, emit a scalar instead of an array.

    Args:
        arr: NumPy array to check

    Returns:
        True if array has size 1 or all elements are equal
    """
    return arr.size == 1 or np.all(arr == arr.flat[0])


def generate_weights_constants(layer, is_integer: bool = False) -> STCode:
    """
    Generate weight constant declarations for a layer.

    Handles optimization: if weights are quantized (int), emits as integers;
    otherwise as floats. Skips LSTM (handled by generate_lstm_weights_constants).

    Args:
        layer: Layer with .weights attribute
        is_integer: Whether to format as integers (quantized weights)

    Returns:
        STCode with weight array declaration(s)
    """
    from ..type_conversion import numpy_to_plc_type as np_to_plc

    builder = STCodeBuilder()

    weight_type = (
        np_to_plc(layer.weights.dtype)
        if is_integer
        else plc_type_from_onnx_dtype(layer.input_type)
    )

    builder.add_code(
        generate_array_constant(
            f"weights_{layer.layer_id}",
            layer.weights,
            weight_type,
            is_integer=is_integer,
        )
    )

    return builder.build()


def generate_lstm_weights_constants(layer) -> STCode:
    """
    Generate LSTM weight constant declarations.

    ONNX gate order: i (input), o (output), f (forget), g (cell/candidate)
    We reorder to: i, f, g, o for standard LSTM notation.

    LSTM weights organized as:
    - W: (4*hidden_size, input_size)  — ONNX order [i, o, f, g]
    - R: (4*hidden_size, hidden_size) — ONNX order [i, o, f, g]
    - B: (4*hidden_size,) — ONNX order [i, o, f, g]

    Args:
        layer: LSTMLayer with .W, .R, .B attributes

    Returns:
        STCode with per-gate weight declarations
    """
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    W = layer.W
    R = layer.R
    B = layer.B if layer.B is not None else np.zeros(4 * h_size)

    weight_type = "REAL"

    # ONNX gate order: i, o, f, g (cell candidate)
    # Extract gate weights using ONNX ordering
    gates_onnx = {
        "i": (0, 1),  # input gate
        "o": (1, 2),  # output gate
        "f": (2, 3),  # forget gate
        "g": (3, 4),  # cell/candidate gate
    }

    # Reorder to: i, f, g, o (standard LSTM)
    gate_order = ["i", "f", "g", "o"]

    for gate in gate_order:
        start, end = gates_onnx[gate]

        # Input weight: W[start*h_size:end*h_size, :]
        W_gate = W[start * h_size : end * h_size, :]
        builder.add_code(
            generate_array_constant(
                f"weights_{layer.layer_id}_{gate}",
                W_gate.flatten(),
                weight_type,
                is_integer=False,
            )
        )

        # Recurrent weight: R[start*h_size:end*h_size, :]
        R_gate = R[start * h_size : end * h_size, :]
        builder.add_code(
            generate_array_constant(
                f"recurrent_{layer.layer_id}_{gate}",
                R_gate.flatten(),
                weight_type,
                is_integer=False,
            )
        )

        # Bias: B[start*h_size:end*h_size]
        B_gate = B[start * h_size : end * h_size]
        builder.add_code(
            generate_array_constant(
                f"bias_{layer.layer_id}_{gate}",
                B_gate,
                weight_type,
                is_integer=False,
            )
        )

    return builder.build()


def generate_gru_weights_constants(layer) -> STCode:
    """
    Generate GRU weight constant declarations.

    GRU weights organized as:
    - W: (3*hidden_size, input_size)  — ONNX order [z, r, h] (update, reset, hidden)
    - R: (3*hidden_size, hidden_size) — ONNX order [z, r, h]
    - B: either (6*hidden_size,) raw ONNX [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
        or (3*hidden_size,) pre-combined [b_z, b_r, b_h]

    Args:
        layer: GRULayer with .W, .R, .B attributes

    Returns:
        STCode with per-gate weight declarations
    """
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    W = layer.W
    R = layer.R
    B = layer.B if layer.B is not None else np.zeros(6 * h_size)

    weight_type = "REAL"

    # ONNX gate order is [z, r, h], where z == update gate (u)
    # We map to local names [u, r, h] used by ST generator.
    gate_to_onnx_index = {"u": 0, "r": 1, "h": 2}
    gate_order = ["u", "r", "h"]

    # Prepare bias vectors
    if B.shape[0] == 6 * h_size:
        Wb = B[: 3 * h_size]
        Rb = B[3 * h_size :]
    elif B.shape[0] == 3 * h_size:
        # Legacy/combined format: treat recurrent bias as zero
        Wb = B
        Rb = np.zeros_like(B)
    else:
        raise ValueError(
            f"GRU layer {layer.layer_id}: invalid bias size {B.shape[0]}, "
            f"expected {3 * h_size} or {6 * h_size}"
        )

    for gate in gate_order:
        onnx_idx = gate_to_onnx_index[gate]
        start, end = onnx_idx, onnx_idx + 1

        # Input weight: W[start*h_size:end*h_size, :]
        W_gate = W[start * h_size : end * h_size, :]
        builder.add_code(
            generate_array_constant(
                f"weights_{layer.layer_id}_{gate}",
                W_gate.flatten(),
                weight_type,
                is_integer=False,
            )
        )

        # Recurrent weight: R[start*h_size:end*h_size, :]
        R_gate = R[start * h_size : end * h_size, :]
        builder.add_code(
            generate_array_constant(
                f"recurrent_{layer.layer_id}_{gate}",
                R_gate.flatten(),
                weight_type,
                is_integer=False,
            )
        )

        # Biases:
        # - For reset/update gates we use combined bias (Wb + Rb).
        # - For candidate gate we expose separate Wb_h / Rb_h because
        #   linear_before_reset affects where Rb_h is applied.
        Wb_gate = Wb[start * h_size : end * h_size]
        Rb_gate = Rb[start * h_size : end * h_size]

        if gate in ("u", "r"):
            builder.add_code(
                generate_array_constant(
                    f"bias_{layer.layer_id}_{gate}",
                    Wb_gate + Rb_gate,
                    weight_type,
                    is_integer=False,
                )
            )
        else:
            builder.add_code(
                generate_array_constant(
                    f"bias_w_{layer.layer_id}_{gate}",
                    Wb_gate,
                    weight_type,
                    is_integer=False,
                )
            )
            builder.add_code(
                generate_array_constant(
                    f"bias_r_{layer.layer_id}_{gate}",
                    Rb_gate,
                    weight_type,
                    is_integer=False,
                )
            )
            # Backward-compat convenience combined form
            builder.add_code(
                generate_array_constant(
                    f"bias_{layer.layer_id}_{gate}",
                    Wb_gate + Rb_gate,
                    weight_type,
                    is_integer=False,
                )
            )

    return builder.build()


def generate_bias_constant(layer) -> STCode:
    """
    Generate bias constant declaration for a layer.

    Skips LSTM and GRU layers (biases handled by their dedicated weight generators).

    Args:
        layer: Layer with .bias attribute

    Returns:
        STCode with bias array declaration
    """
    # Check layer type to avoid LSTM and GRU
    from ...types import LSTMLayer, GRULayer

    if isinstance(layer, (LSTMLayer, GRULayer)):
        return STCode.empty()

    if not hasattr(layer, "bias") or layer.bias is None:
        return STCode.empty()

    bias_type = plc_type_from_onnx_dtype(layer.output_type)
    return generate_array_constant(f"bias_{layer.layer_id}", layer.bias, bias_type)


def generate_quantization_params(layer) -> STCode:
    """
    Generate quantization parameter constants (scale, zero_point).

    Only generates arrays for per-channel quantization (per-tensor is inlined).

    Args:
        layer: QuantizeLinear or DequantizeLinear layer

    Returns:
        STCode with scale and zero_point declarations (if needed)
    """
    # Only generate if per-channel (size > 1)
    if layer.scale.size == 1:
        return STCode.empty()

    builder = STCodeBuilder()

    # Scale array (always REAL for quantization parameters)
    builder.add_code(
        generate_array_constant(f"scale_{layer.layer_id}", layer.scale, "REAL")
    )

    # Zero point array (typed based on layer)
    from ...types import QuantizeLinearLayer

    if isinstance(layer, QuantizeLinearLayer):
        dtype_str = layer.output_type
    else:  # DequantizeLinearLayer
        dtype_str = layer.input_type

    zp_type = plc_type_from_onnx_dtype(dtype_str)
    builder.add_code(
        generate_array_constant(
            f"zero_point_{layer.layer_id}",
            layer.zero_point,
            zp_type,
            is_integer=True,
        )
    )

    return builder.build()


def generate_batchnorm_constants(layer) -> STCode:
    """
    Generate BatchNorm precomputed scale and bias constants.

    Inference-mode BatchNorm uses precomputed:
    - combined_scale[c] = γ[c] / sqrt(σ²[c] + ε)
    - combined_bias[c] = β[c] - μ[c] * combined_scale[c]

    Args:
        layer: BatchNormLayer with .combined_scale and .combined_bias

    Returns:
        STCode with scale and bias array declarations
    """
    builder = STCodeBuilder()

    builder.add_code(
        generate_array_constant(
            f"bn_scale_{layer.layer_id}",
            layer.combined_scale,
            "REAL",
        )
    )

    builder.add_code(
        generate_array_constant(
            f"bn_bias_{layer.layer_id}",
            layer.combined_bias,
            "REAL",
        )
    )

    return builder.build()
