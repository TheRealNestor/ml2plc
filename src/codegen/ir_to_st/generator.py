"""
IR to Structured Text Code Generation Module

This module is responsible for generating Structured Text (ST) code from the intermediate representation (IR) of a neural network.

Architecture:
  - generate_function_block(): Main entry for single-region (legacy) models
  - generate_model_function_block(): Main entry for multi-region ModelIR
    - Uses lowerers.py for region-kind-specific ST generation
"""

from ..types import *
from .st_code import *
from .type_conversion import *
from ..ir_optimizer import OptimizationResult
from .utils.constant_helpers import (
    generate_array_constant,
    generate_scalar_constant,
    is_uniform_array,
    generate_weights_constants,
    generate_lstm_weights_constants,
    generate_bias_constant,
    generate_quantization_params,
    generate_batchnorm_constants,
)
from .utils.activation_helpers import (
    generate_activation_inline,
    generate_activation_loop,
)
from .utils.copy_helpers import (
    generate_simple_copy,
    generate_offset_copy,
    generate_strided_copy,
    generate_scalar_broadcast,
    generate_modulo_broadcast,
    generate_selective_copy,
)

import logging

logger = logging.getLogger(__name__)

# ===========================================================================
# Configuration
# ===========================================================================

# Activations that can be inlined within matrix multiplication
# (vs. requiring a separate loop pass)
INLINE_ACTIVATIONS = {
    ActivationType.NONE,
    ActivationType.RELU,
}

# ===========================================================================
# Utility Functions
# ===========================================================================


def is_uniform_array(arr: np.ndarray) -> bool:
    """
    Check if all elements in array are identical.

    Used to optimize storage of quantization parameters - if all values
    are the same, we can store a single scalar instead of an array.

    Args:
        arr: NumPy array to check

    Returns:
        True if array has size 1 or all elements are identical
    """
    return arr.size == 1 or np.all(arr == arr.flat[0])


def get_layer_type_name(layer: LinearLayer, activation: ActivationType) -> str:
    """Get descriptive name for layer type."""
    if isinstance(layer, FusedGemmLayer):
        return f"Fused Gemm + {activation.name}"
    elif isinstance(layer, FusedLinearLayer):
        return f"Fused Linear + {activation.name}"
    elif isinstance(layer, GemmLayer):
        return "Gemm"
    else:
        return "MatMul"


def get_layer_input_vars(
    layer: BaseLayer,
    network: NetworkIR,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Get all input variable names for a layer."""
    input_vars = []

    for inp_tensor in layer.inputs:
        if network.is_network_input(inp_tensor):
            input_vars.append("input_data")
        elif buffer_allocations and inp_tensor in buffer_allocations:
            # Use allocated buffer name
            input_vars.append(buffer_allocations[inp_tensor])
        elif inp_tensor in network.tensor_producers:
            producer_name = network.tensor_producers[inp_tensor]
            if producer_name in network.layers:
                producer_layer = network.layers[producer_name]
                input_vars.append(f"layer_{producer_layer.layer_id}_output")

    if len(input_vars) != len(layer.inputs):
        unresolved = [
            t
            for t in layer.inputs
            if not network.is_network_input(t)
            and t not in (buffer_allocations or {})
            and (
                t not in network.tensor_producers
                or network.tensor_producers.get(t) not in network.layers
            )
        ]
        if unresolved:
            raise ValueError(
                f"Layer {layer.layer_id} ({layer.name}) has unresolved input tensors "
                f"that cannot be mapped to ST variables: {unresolved}"
            )

    return input_vars


def get_layer_output_var(
    layer: BaseLayer,
    network: NetworkIR,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> str:
    """Get output variable name for a layer."""

    if not layer.outputs:
        raise ValueError(
            f"Layer {layer.layer_id} ({layer.name}) has no outputs; cannot generate ST output variable"
        )

    output_tensor = layer.outputs[0]  # Assuming single output for simplicity

    if network.is_network_output(output_tensor):
        return "output_data"

    if buffer_allocations and output_tensor in buffer_allocations:
        return buffer_allocations[output_tensor]

    return f"layer_{layer.layer_id}_output"


def generate_weight_access(
    layer: LinearLayer, input_var: str, layer_id: int, output_size: int
) -> str:
    """Generate the weight multiplication expression."""
    weight_expr = f"weights_{layer_id}[i * {output_size} + j]"

    if not layer.is_quantized():
        return f"{input_var}[i] * {weight_expr}"

    # Quantized weights
    cast_func = numpy_to_plc_cast_func(layer.weights.dtype, "REAL")

    # Scale expression
    if is_uniform_array(layer.weight_scale):
        scale_expr = f"weight_scale_{layer_id}"
    else:
        scale_expr = f"weight_scale_{layer_id}[j]"

    # Zero point expression
    if is_uniform_array(layer.weight_zero_point):
        zp_expr = f"weight_zero_point_{layer_id}"
    else:
        zp_expr = f"weight_zero_point_{layer_id}[j]"

    return f"{input_var}[i] * ({scale_expr} * {cast_func}({weight_expr} - {zp_expr}))"


def build_final_linear_layer_expression(layer: LinearLayer, has_bias: bool) -> str:
    """Build final expression with alpha, bias, beta."""
    alpha = getattr(layer, "alpha", 1.0)
    beta = getattr(layer, "beta", 1.0)

    expr = "sum"

    if alpha != 1.0:
        expr = f"{alpha} * {expr}"

    if has_bias:
        bias_term = f"bias_{layer.layer_id}[j]"
        if beta != 1.0:
            bias_term = f"{beta} * {bias_term}"
        expr = f"{expr} + {bias_term}"

    return expr


# ============================================================================
# Header/Footer Generation
# ============================================================================


def generate_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return st_function_block_header(fb_name)


def generate_footer() -> STCode:
    """Generate function block footer."""
    return st_function_block_footer()


def generate_input_output_vars(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT and VAR_OUTPUT sections."""
    code = STCode.empty()

    first_layer_name = network.execution_order[0]
    first_layer = network.layers[first_layer_name]

    last_layer_name = network.execution_order[-1]
    last_layer = network.layers[last_layer_name]

    # Determine actual input size.
    # For models with LSTM, the input is the full sequence (seq_len * features).
    actual_input_size = first_layer.input_size
    for layer_name in network.execution_order:
        layer = network.layers[layer_name]
        if isinstance(layer, LSTMLayer):
            lstm_total_input = layer.sequence_length * layer.input_size
            if lstm_total_input > actual_input_size:
                logger.debug(
                    f"Adjusting input_data size from {actual_input_size} to "
                    f"{lstm_total_input} (LSTM seq_len={layer.sequence_length} "
                    f"x input_size={layer.input_size})"
                )
                actual_input_size = lstm_total_input
            break

    input_type = plc_type_from_onnx_dtype(first_layer.input_type)
    code += STCode.from_lines(
        "VAR_INPUT",
        f"    input_data : ARRAY[0..{actual_input_size - 1}] OF {input_type};",
        "END_VAR",
        "",
    )

    output_type = plc_type_from_onnx_dtype(last_layer.output_type)
    code += STCode.from_lines(
        "VAR_OUTPUT",
        f"    output_data : ARRAY[0..{last_layer.output_size - 1}] OF {output_type};",
        "END_VAR",
        "",
    )

    return code


# ============================================================================
# Constants Section
# ============================================================================
# Note: generate_array_constant, generate_scalar_constant, is_uniform_array
# are now imported from utils.constant_helpers


def generate_layer_weights(layer) -> STCode:
    """
    Generate weight constants for a layer.

    Delegates to utility functions for LSTM and general layer handling.
    Handles both float and quantized weights.

    Returns all weight-related constants:
    - weights array
    - weight_scale (if quantized)
    - weight_zero_point (if quantized)
    """
    # Special handling for LSTM layers (delegated to utility)
    if isinstance(layer, LSTMLayer):
        return generate_lstm_weights_constants(layer)

    # General layer handling (delegated to utility)
    is_quantized = isinstance(layer, LinearLayer) and layer.is_quantized()
    return generate_weights_constants(layer, is_integer=is_quantized)


def generate_lstm_weights(layer: "LSTMLayer") -> STCode:
    """
    DEPRECATED: Use generate_lstm_weights_constants from utils.constant_helpers

    This wrapper is kept for backward compatibility during refactoring.
    """
    return generate_lstm_weights_constants(layer)


def generate_layer_bias(layer) -> STCode:
    """Generate bias constant for a layer.

    Note: For LSTM layers, biases are generated separately in generate_lstm_weights_constants.
    """
    # Skip bias generation for LSTM (already handled)
    if isinstance(layer, LSTMLayer):
        return STCode.empty()

    # Delegate to utility
    return generate_bias_constant(layer)


def generate_layer_quantization_params(layer) -> STCode:
    """
    Generate quantization parameters for QuantizeLinear/DequantizeLinear layers.

    Delegates to utility function. Only generates arrays for per-channel
    quantization (per-tensor is inlined).
    """
    return generate_quantization_params(layer)


def generate_constants_section(network: NetworkIR) -> STCode:
    """Generate VAR CONSTANT section."""
    code = STCode.from_lines("VAR CONSTANT")

    for layer_name in network.execution_order:
        layer = network.layers[layer_name]
        has_constants = False

        # Weights (for linear layers)
        if hasattr(layer, "weights") and layer.weights is not None:
            code += generate_layer_weights(layer).indent()
            has_constants = True

        # Bias
        if hasattr(layer, "bias") and layer.bias is not None:
            code += generate_layer_bias(layer).indent()
            has_constants = True

        # Quantization parameters (for activation quantization only)
        if isinstance(layer, (QuantizeLinearLayer, DequantizeLinearLayer)):
            if layer.input_type is not None:  # Skip weight-only dequantization
                quant_params = generate_layer_quantization_params(layer)
                if quant_params.lines:
                    code += quant_params.indent()
                    has_constants = True

        # BatchNorm precomputed parameters
        if isinstance(layer, BatchNormLayer):
            bn_code = generate_batchnorm_constants(layer)
            if bn_code.lines:
                code += bn_code.indent()
                has_constants = True

        if has_constants:
            code += STCode.blank_line()

    code += STCode.from_lines("END_VAR", "")
    return code


# ============================================================================
# VAR Section
# ============================================================================


def generate_var_section(
    network: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> STCode:
    """Generate VAR section with all internal variables."""
    builder = STCodeBuilder()
    builder.add_line("VAR")

    if buffer_allocations:
        buffer_info = {}  # buffer_name -> (size, dtype)

        for tensor_name, buffer_name in buffer_allocations.items():
            producer_name = network.tensor_producers[tensor_name]
            layer = network.layers[producer_name]
            plc_type = plc_type_from_onnx_dtype(layer.output_type)
            size = layer.output_size

            # Track max size for each buffer
            if buffer_name not in buffer_info:
                buffer_info[buffer_name] = (size, plc_type)
            else:
                existing_size, _ = buffer_info[buffer_name]
                buffer_info[buffer_name] = (max(existing_size, size), plc_type)

        builder.add_line("    (* Buffer allocation variables *)")

        with builder.indent():
            for buffer_name, (size, dtype) in buffer_info.items():
                builder.add_line(f"{buffer_name} : ARRAY[0..{size - 1}] OF {dtype};")
        builder.add_line("")

    else:
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]

            if any(network.is_network_output(out) for out in layer.outputs):
                continue

            plc_type = plc_type_from_onnx_dtype(layer.output_type)

            with builder.indent():
                builder.add_line(
                    f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size - 1}] OF {plc_type};"
                )

            builder.add_line("")

    # Temporary computation variables
    with builder.indent():
        builder.add_line("(* Temporary computation variables *)")
        builder.add_line("i : DINT;")
        builder.add_line("j : DINT;")
        builder.add_line("sum : REAL;")

    # Extra loop indices needed by Conv2D / Pool2D layers
    has_spatial_layers = any(
        isinstance(network.layers[ln], (Conv2DLayer, Pool2DLayer))
        for ln in network.execution_order
    )
    has_batchnorm = any(
        isinstance(network.layers[ln], BatchNormLayer) for ln in network.execution_order
    )
    if has_spatial_layers:
        with builder.indent():
            builder.add_line("(* Spatial loop variables for Conv / Pool layers *)")
            builder.add_line("oc : DINT;")
            builder.add_line("oh : DINT;")
            builder.add_line("ow : DINT;")
            builder.add_line("ic : DINT;")
            builder.add_line("kh : DINT;")
            builder.add_line("kw : DINT;")
            builder.add_line("ih : DINT;")
            builder.add_line("iw : DINT;")
    elif has_batchnorm:
        # BatchNorm uses 'oc' for channel loop but doesn't need full spatial vars
        with builder.indent():
            builder.add_line("(* Channel loop variable for BatchNorm *)")
            builder.add_line("oc : DINT;")

    # Transpose layers use dynamically-named loop vars (t<id>_d<dim>)
    for ln in network.execution_order:
        layer = network.layers[ln]
        if isinstance(layer, TransposeLayer) and layer.output_shape:
            ndim = len(layer.output_shape)
            if ndim > 0 and layer.input_size > 1:
                with builder.indent():
                    builder.add_line(
                        f"(* Transpose layer {layer.layer_id} loop variables *)"
                    )
                    for d in range(ndim):
                        builder.add_line(f"t{layer.layer_id}_d{d} : DINT;")

    # Check if any layer uses softmax activation
    has_softmax = any(
        getattr(network.layers[layer_name], "activation", None)
        == ActivationType.SOFTMAX
        for layer_name in network.execution_order
    )

    if has_softmax:
        with builder.indent():
            builder.add_line("max_val : REAL;")
            builder.add_line("exp_sum : REAL;")

    # Check if any LSTM layers are present and add temporary buffers for gates
    has_lstm = any(
        isinstance(network.layers[ln], LSTMLayer) for ln in network.execution_order
    )
    if has_lstm:
        with builder.indent():
            builder.add_line("(* LSTM gate buffers and temporary variables *)")
            builder.add_line("t : DINT;")
            builder.add_line("exp_val : REAL;")
            for ln in network.execution_order:
                layer = network.layers[ln]
                if isinstance(layer, LSTMLayer):
                    h_size = layer.hidden_size
                    builder.add_line(f"(* Layer {layer.layer_id} gate buffers *)")
                    builder.add_line(
                        f"i_gate_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )
                    builder.add_line(
                        f"f_gate_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )
                    builder.add_line(
                        f"g_gate_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )
                    builder.add_line(
                        f"o_gate_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )
                    builder.add_line(
                        f"c_tanh_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )
                    builder.add_line(
                        f"h_state_in_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )
                    builder.add_line(
                        f"c_state_in_{layer.layer_id} : ARRAY[0..{h_size - 1}] OF REAL;"
                    )

    builder.add_line("")
    builder.add_line("END_VAR")
    builder.add_line("")

    return builder.build()


# ============================================================================
# Layer Code Generators
# ============================================================================


def generate_activation_code(
    activation: ActivationType, input_var: str, output_var: str, size: int
) -> STCode:
    """
    Generate activation code for activations that need separate loops.

    Delegates to utility function for consistent implementation.
    """
    return generate_activation_loop(activation, input_var, output_var, size)


def generate_activation_layer_code(
    layer: ActivationLayer, input_var: str, output_var: str
) -> STCode:
    """Generate activation layer code with comment."""
    code = STCode.from_lines(
        f"(* Layer {layer.layer_id}: Activation ({layer.activation.name}) *)"
    )
    code += generate_activation_code(
        layer.activation, input_var, output_var, layer.output_size
    )
    return code


def generate_linear_layer_code(
    layer: LinearLayer, input_var: str, output_var: str
) -> STCode:
    """Generate code for all linear layer types."""
    builder = STCodeBuilder()

    activation = getattr(layer, "activation", ActivationType.NONE)
    layer_type_name = get_layer_type_name(layer, activation)

    # Header
    builder.add_line(f"(* Layer {layer.layer_id}: {layer_type_name} *)")

    # Matrix multiplication
    builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
    with builder.indent():
        builder.add_line("sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
        with builder.indent():
            weight_mult = generate_weight_access(
                layer, input_var, layer.layer_id, layer.output_size
            )
            builder.add_line(f"sum := sum + {weight_mult};")
        builder.add_line("END_FOR;")

        # Apply bias and activation inline
        final_expr = build_final_linear_layer_expression(layer, layer.bias is not None)

        # Inline activation if possible using utility
        activated_expr = generate_activation_inline(activation, final_expr)

        builder.add_line(f"{output_var}[j] := {activated_expr};")

    builder.add_line("END_FOR;")

    # Separate activation pass if needed (for activations that can't be inlined)
    if activation not in INLINE_ACTIVATIONS:
        builder.add_line("")
        builder.add_code(
            generate_activation_code(
                activation, output_var, output_var, layer.output_size
            )
        )

    return builder.build()


def generate_add_code(
    layer: AddLayer, input_vars: List[str], output_var: str
) -> STCode:
    """Generate Add layer code (supports both bias addition and element-wise)."""
    builder = STCodeBuilder()

    if layer.bias is not None:
        # Bias addition: output = input + bias
        builder.add_line(f"(* Layer {layer.layer_id}: Add (Bias) *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := {input_vars[0]}[i] + bias_{layer.layer_id}[i];"
            )
        builder.add_line("END_FOR;")
    else:
        # Element-wise addition: output = input1 + input2
        if len(input_vars) != 2:
            raise ValueError(
                f"Element-wise Add layer {layer.layer_id} expected 2 inputs, got {len(input_vars)}"
            )

        builder.add_line(f"(* Layer {layer.layer_id}: Add (Element-wise) *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := {input_vars[0]}[i] + {input_vars[1]}[i];"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_reshape_code(
    layer: ReshapeLayer, input_var: str, output_var: str
) -> STCode:
    """Generate code corresponding to a ONNX Reshape layer."""
    if layer.input_size != layer.output_size:
        raise NotImplementedError(
            "Reshape layer with different sizes is not implemented yet."
        )

    return generate_simple_copy(
        input_var,
        output_var,
        layer.output_size,
        f"Layer {layer.layer_id}: Reshape (copy input to output)",
    )


def generate_quantize_linear_code(
    layer: QuantizeLinearLayer, input_var: str, output_var: str
) -> STCode:
    """Generate QuantizeLinear code: quantized = clip(round(input/scale) + zero_point)"""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: {layer.name} - QuantizeLinear *)")

    # Get bounds and cast function from output type
    output_plc_type = plc_type_from_onnx_dtype(layer.output_type)
    min_val, max_val = get_type_limits_from_str(layer.output_type)
    cast_func = get_conversion_func("REAL", output_plc_type)

    is_per_tensor = layer.scale.size == 1

    if is_per_tensor:
        # Per-tensor quantization
        scale_val = layer.scale.flat[0]
        zero_point_val = layer.zero_point.flat[0]

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var}[i] / {scale_val}) + {zero_point_val}), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")
    else:
        # Per-channel quantization
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var}[i] / scale_{layer.layer_id}[i]) + zero_point_{layer.layer_id}[i]), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_dequantize_linear_code(
    layer: DequantizeLinearLayer, input_var: str, output_var: str
) -> STCode:
    """Generate DequantizeLinear code: float = scale * (quantized - zero_point)"""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: {layer.name} - DequantizeLinear *)")

    input_plc_type = plc_type_from_onnx_dtype(layer.input_type)
    cast_func = get_conversion_func(input_plc_type, "REAL")

    is_per_tensor = layer.scale.size == 1

    if is_per_tensor:
        # Per-tensor dequantization
        scale_val = layer.scale.flat[0]
        zero_point_val = layer.zero_point.flat[0]

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := {scale_val} * "
                f"{cast_func}({input_var}[i] - {zero_point_val});"
            )
        builder.add_line("END_FOR;")
    else:
        # Per-channel dequantization
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := scale_{layer.layer_id}[i] * "
                f"{cast_func}({input_var}[i] - zero_point_{layer.layer_id}[i]);"
            )
        builder.add_line("END_FOR;")

    return builder.build()


# NOTE: Dropout code does not actually do anything during inference. Identity operation for now (modify if we need on-device training)
def generate_dropout_code(
    layer: DropoutLayer, input_var: str, output_var: str
) -> STCode:
    """
    Generate Dropout layer code.

    At inference time, Dropout is an identity operation.
    """
    pass


def generate_conv2d_code(layer: Conv2DLayer, input_var: str, output_var: str) -> STCode:
    """
    Generate Conv2D layer code.

    All tensors are stored as flat 1-D arrays.  The index arithmetic
    follows the NCHW layout (batch dim already stripped for PLC):

        input  layout: [C_in][H_in][W_in]
        weight layout: [C_out][C_in/group][kH][kW]   (flattened row-major)
        output layout: [C_out][H_out][W_out]

    Generates 6 nested FOR loops with a boundary check for padding.
    """
    builder = STCodeBuilder()

    # Unpack spatial parameters
    in_c, in_h, in_w = (
        layer.input_shape[-3],
        layer.input_shape[-2],
        layer.input_shape[-1],
    )
    out_c, out_h, out_w = (
        layer.output_shape[-3],
        layer.output_shape[-2],
        layer.output_shape[-1],
    )
    kH, kW = layer.kernel_shape
    sH, sW = layer.strides
    pH, pW = layer.pads[0], layer.pads[1]  # top, left
    dH, dW = layer.dilations
    group = layer.group
    in_c_per_group = in_c // group

    # Weight layout sizes for index arithmetic
    w_ic_size = in_c_per_group * kH * kW  # elements per output-channel slice
    w_kh_size = kH * kW  # per input-channel slice (unused var kept for clarity)

    builder.add_line(
        f"(* Layer {layer.layer_id}: Conv2D  "
        f"in={in_c}x{in_h}x{in_w}  out={out_c}x{out_h}x{out_w}  "
        f"kernel={kH}x{kW}  stride={sH}x{sW}  pad={layer.pads}  "
        f"group={group} *)"
    )

    # -- output channels --
    builder.add_line(f"FOR oc := 0 TO {out_c - 1} DO")
    with builder.indent():
        # -- output height --
        builder.add_line(f"FOR oh := 0 TO {out_h - 1} DO")
        with builder.indent():
            # -- output width --
            builder.add_line(f"FOR ow := 0 TO {out_w - 1} DO")
            with builder.indent():
                # Initialise accumulator with bias (or 0)
                if layer.bias is not None:
                    builder.add_line(f"sum := bias_{layer.layer_id}[oc];")
                else:
                    builder.add_line("sum := 0.0;")

                # Determine the input channel range for this group
                if group == 1:
                    ic_start = "0"
                    ic_end = str(in_c_per_group - 1)
                else:
                    ic_start = f"(oc * {in_c_per_group} / {out_c // group})"
                    # For depthwise (group==in_c) each output channel uses 1 input channel
                    if group == in_c:
                        ic_start = "oc"
                        ic_end = "oc"
                    else:
                        ic_end = f"({ic_start} + {in_c_per_group - 1})"

                # -- input channels --
                builder.add_line(f"FOR ic := {ic_start} TO {ic_end} DO")
                with builder.indent():
                    # -- kernel height --
                    builder.add_line(f"FOR kh := 0 TO {kH - 1} DO")
                    with builder.indent():
                        # -- kernel width --
                        builder.add_line(f"FOR kw := 0 TO {kW - 1} DO")
                        with builder.indent():
                            # Compute input coordinates
                            if dH == 1:
                                builder.add_line(f"ih := oh * {sH} - {pH} + kh;")
                            else:
                                builder.add_line(f"ih := oh * {sH} - {pH} + kh * {dH};")
                            if dW == 1:
                                builder.add_line(f"iw := ow * {sW} - {pW} + kw;")
                            else:
                                builder.add_line(f"iw := ow * {sW} - {pW} + kw * {dW};")

                            # Boundary check (needed when padding > 0)
                            has_padding = any(p != 0 for p in layer.pads)
                            if has_padding:
                                builder.add_line(
                                    f"IF (ih >= 0) AND (ih < {in_h}) AND (iw >= 0) AND (iw < {in_w}) THEN"
                                )
                                indent_ctx = builder.indent()
                                indent_ctx.__enter__()

                            # input index:  ic * H_in * W_in  +  ih * W_in  +  iw
                            input_idx = f"ic * {in_h * in_w} + ih * {in_w} + iw"

                            # weight index: oc * (C_in_per_group * kH * kW)
                            #             + (ic - ic_group_start) * kH * kW
                            #             + kh * kW + kw
                            if group == 1:
                                weight_idx = f"oc * {w_ic_size} + ic * {kH * kW} + kh * {kW} + kw"
                            elif group == in_c:
                                # Depthwise: weight shape is (C_out, 1, kH, kW)
                                weight_idx = f"oc * {kH * kW} + kh * {kW} + kw"
                            else:
                                weight_idx = f"oc * {w_ic_size} + (ic - {ic_start}) * {kH * kW} + kh * {kW} + kw"

                            builder.add_line(
                                f"sum := sum + {input_var}[{input_idx}] "
                                f"* weights_{layer.layer_id}[{weight_idx}];"
                            )

                            if has_padding:
                                indent_ctx.__exit__(None, None, None)
                                builder.add_line("END_IF;")

                        builder.add_line("END_FOR;")  # kw
                    builder.add_line("END_FOR;")  # kh
                builder.add_line("END_FOR;")  # ic

                # Store result: flat index = oc * H_out * W_out + oh * W_out + ow
                output_idx = f"oc * {out_h * out_w} + oh * {out_w} + ow"
                builder.add_line(f"{output_var}[{output_idx}] := sum;")

            builder.add_line("END_FOR;")  # ow
        builder.add_line("END_FOR;")  # oh
    builder.add_line("END_FOR;")  # oc

    return builder.build()


def generate_pool2d_code(layer: Pool2DLayer, input_var: str, output_var: str) -> STCode:
    """
    Generate MaxPool or AveragePool layer code.

    Tensors stored as flat arrays in CHW order (batch dim stripped).
    """
    builder = STCodeBuilder()

    channels = layer.input_shape[-3]
    in_h, in_w = layer.input_shape[-2], layer.input_shape[-1]
    out_h, out_w = layer.output_shape[-2], layer.output_shape[-1]
    kH, kW = layer.kernel_shape
    sH, sW = layer.strides
    pH, pW = layer.pads[0], layer.pads[1]
    is_max = layer.pool_type == "max"

    pool_label = "MaxPool" if is_max else "AvgPool"
    builder.add_line(
        f"(* Layer {layer.layer_id}: {pool_label}  "
        f"kernel={kH}x{kW}  stride={sH}x{sW} *)"
    )

    # -- channels (preserved) --
    builder.add_line(f"FOR oc := 0 TO {channels - 1} DO")
    with builder.indent():
        builder.add_line(f"FOR oh := 0 TO {out_h - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR ow := 0 TO {out_w - 1} DO")
            with builder.indent():
                if is_max:
                    # Initialise to first valid element (large negative as fallback)
                    builder.add_line("sum := -3.402823E+38;")  # -FLT_MAX
                else:
                    builder.add_line("sum := 0.0;")

                builder.add_line(f"FOR kh := 0 TO {kH - 1} DO")
                with builder.indent():
                    builder.add_line(f"FOR kw := 0 TO {kW - 1} DO")
                    with builder.indent():
                        builder.add_line(f"ih := oh * {sH} - {pH} + kh;")
                        builder.add_line(f"iw := ow * {sW} - {pW} + kw;")

                        has_padding = any(p != 0 for p in layer.pads)
                        if has_padding:
                            builder.add_line(
                                f"IF (ih >= 0) AND (ih < {in_h}) AND (iw >= 0) AND (iw < {in_w}) THEN"
                            )
                            indent_ctx = builder.indent()
                            indent_ctx.__enter__()

                        input_idx = f"oc * {in_h * in_w} + ih * {in_w} + iw"
                        if is_max:
                            builder.add_line(f"IF {input_var}[{input_idx}] > sum THEN")
                            with builder.indent():
                                builder.add_line(f"sum := {input_var}[{input_idx}];")
                            builder.add_line("END_IF;")
                        else:
                            builder.add_line(f"sum := sum + {input_var}[{input_idx}];")

                        if has_padding:
                            indent_ctx.__exit__(None, None, None)
                            builder.add_line("END_IF;")

                    builder.add_line("END_FOR;")  # kw
                builder.add_line("END_FOR;")  # kh

                output_idx = f"oc * {out_h * out_w} + oh * {out_w} + ow"
                if is_max:
                    builder.add_line(f"{output_var}[{output_idx}] := sum;")
                else:
                    kernel_area = kH * kW
                    builder.add_line(
                        f"{output_var}[{output_idx}] := sum / {float(kernel_area)};"
                    )

            builder.add_line("END_FOR;")  # ow
        builder.add_line("END_FOR;")  # oh
    builder.add_line("END_FOR;")  # oc

    return builder.build()


def generate_flatten_code(
    layer: FlattenLayer, input_var: str, output_var: str
) -> STCode:
    """
    Generate Flatten layer code.

    Since both input and output are already stored as flat 1-D arrays
    this is simply a memcopy (same as Reshape with identical sizes).
    """
    return generate_simple_copy(
        input_var,
        output_var,
        layer.output_size,
        f"Layer {layer.layer_id}: Flatten (axis={layer.axis}) — copy",
    )


def generate_squeeze_code(
    layer: "SqueezeLayer", input_var: str, output_var: str
) -> STCode:
    """
    Generate Squeeze layer code.

    Squeeze removes dimensions of size 1.  Since both input and output
    are stored as flat 1-D arrays with the same number of elements,
    this is simply a memcopy — identical to Flatten.
    """
    return generate_simple_copy(
        input_var, output_var, layer.output_size, f"Layer {layer.layer_id}: Squeeze"
    )


def generate_lstm_code(layer: "LSTMLayer", input_var: str, output_var: str) -> STCode:
    """
    Generate LSTM layer code with full temporal unrolling.

    For each timestep t in [0, seq_len):
        i_t = sigmoid(W_i @ x_t + R_i @ h_{t-1} + b_i)
        f_t = sigmoid(W_f @ x_t + R_f @ h_{t-1} + b_f)
        g_t = tanh(W_g @ x_t + R_g @ h_{t-1} + b_g)
        o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    Input is a flat array: input_var[t * input_size + i] for timestep t, feature i.

    ONNX weight layout per gate (direction dim already stripped):
        W_gate: (hidden_size, input_size)  — flat index: [j * input_size + i]
        R_gate: (hidden_size, hidden_size) — flat index: [j * hidden_size + i]

    Output is the final hidden state h_{T-1}, copied to output_var.
    """
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    seq_len = layer.sequence_length
    lid = layer.layer_id

    builder.add_line(
        f"(* Layer {lid}: LSTM (hidden_size={h_size}, "
        f"input_size={x_size}, seq_len={seq_len}) *)"
    )
    builder.add_line(
        f"(* Unrolled over {seq_len} timesteps, output = final hidden state *)"
    )

    # Initialize hidden state and cell state to zero
    builder.add_line("")
    builder.add_line(f"(* Initialize LSTM states to zero *)")
    builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
    with builder.indent():
        builder.add_line(f"h_state_{lid}[j] := 0.0;")
        builder.add_line(f"c_state_{lid}[j] := 0.0;")
    builder.add_line("END_FOR;")

    # Outer loop over timesteps
    builder.add_line("")
    builder.add_line(f"(* Process sequence: {seq_len} timesteps *)")
    builder.add_line(f"FOR t := 0 TO {seq_len - 1} DO")
    with builder.indent():

        # Helper to emit a gate computation
        def emit_gate(gate_name, var_name, activation):
            builder.add_line("")
            builder.add_line(f"(* {gate_name} *)")
            builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
            with builder.indent():
                builder.add_line("sum := 0.0;")
                # W @ x_t: W is (hidden_size, input_size), index [j * x_size + i]
                builder.add_line(f"FOR i := 0 TO {x_size - 1} DO")
                with builder.indent():
                    gate_letter = var_name.split("_")[0]  # i, f, g, o
                    builder.add_line(
                        f"sum := sum + {input_var}[t * {x_size} + i] "
                        f"* weights_{lid}_{gate_letter}[j * {x_size} + i];"
                    )
                builder.add_line("END_FOR;")
                # R @ h_{t-1}: R is (hidden_size, hidden_size), index [j * h_size + i]
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

        emit_gate(
            "Input Gate: i_t = sigmoid(W_i @ x_t + R_i @ h_{t-1} + b_i)",
            "i_gate",
            "sigmoid",
        )
        emit_gate(
            "Forget Gate: f_t = sigmoid(W_f @ x_t + R_f @ h_{t-1} + b_f)",
            "f_gate",
            "sigmoid",
        )
        emit_gate(
            "Cell Gate: g_t = tanh(W_g @ x_t + R_g @ h_{t-1} + b_g)", "g_gate", "tanh"
        )
        emit_gate(
            "Output Gate: o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o)",
            "o_gate",
            "sigmoid",
        )

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

    builder.add_line("END_FOR;  (* end timestep loop *)")

    # Copy final hidden state to output buffer
    builder.add_line("")
    builder.add_line(f"(* Copy final hidden state to output *)")
    builder.add_line(f"FOR j := 0 TO {h_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[j] := h_state_{lid}[j];")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_gru_code(layer: "GRULayer", input_var: str, output_var: str) -> STCode:
    """
    Generate GRU (Gated Recurrent Unit) layer code.

    GRU is similar to LSTM but simpler (no cell state, 3 gates instead of 4):
    - Reset gate (r), Update gate (z), New hidden state (~h)
    - Updates hidden state (h) only

    For inference, we assume:
    - Input: x of shape (input_size,)
    - Initial h: zero or provided by recurrent region
    - Output: h of shape (hidden_size,)

    The forward pass computes (for a single timestep):
        r_t = sigmoid(W_r @ x + R_r @ h_{t-1} + b_r)
        z_t = sigmoid(W_z @ x + R_z @ h_{t-1} + b_z)
        ~h_t = tanh(W_h @ x + R_h @ (r_t * h_{t-1}) + b_h)
        h_t = (1 - z_t) * ~h_t + z_t * h_{t-1}
    """
    builder = STCodeBuilder()

    h_size = layer.hidden_size
    x_size = layer.input_size
    W = layer.W
    R = layer.R
    B = layer.B if layer.B is not None else np.zeros(3 * h_size)

    builder.add_line(f"(* Layer {layer.layer_id}: GRU (hidden_size={h_size}) *)")
    builder.add_line(
        f"(* Note: This is a single GRU timestep; state carried by region wrapper *)"
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
        builder.add_line(f"{output_var}[j] := 1.0 / (1.0 + EXP(-sum));  (* sigmoid *)")
    builder.add_line("END_FOR;")

    builder.add_line("")
    builder.add_line(
        f"(* NOTE: This is a simplified GRU for demo. Full implementation requires:"
    )
    builder.add_line("(* - State variables (h) from recurrent region wrapper *)")
    builder.add_line("(* - Proper gate interactions (update gate, candidate hidden) *)")
    builder.add_line("(* - Hidden state update computation *)")
    builder.add_line("(* This MVP demonstrates structure for extension. *)")

    return builder.build()


# ============================================================================
# Data-Movement Layer Generators
# ============================================================================


def generate_identity_copy(
    layer: BaseLayer, input_var: str, output_var: str, comment: str = ""
) -> STCode:
    """
    Generate a simple element-by-element copy loop.

    Used by layers that don't change data layout in flat-array representation
    (Unsqueeze, Reshape, Squeeze, identity Cast, identity Expand).

    Delegates to utility function for consistent implementation.
    """
    label = comment or f"Layer {layer.layer_id}: {layer.op_type}"
    return generate_simple_copy(input_var, output_var, layer.output_size, label)


def generate_cast_code(layer: "CastLayer", input_var: str, output_var: str) -> STCode:
    """Generate Cast layer — type conversion of each element."""
    input_plc = (
        plc_type_from_onnx_dtype(layer.input_type) if layer.input_type else "REAL"
    )
    output_plc = (
        plc_type_from_onnx_dtype(layer.output_type) if layer.output_type else "REAL"
    )

    # Same PLC type → identity copy
    if input_plc == output_plc:
        return generate_identity_copy(
            layer,
            input_var,
            output_var,
            f"Layer {layer.layer_id}: Cast (no-op, same PLC type {input_plc})",
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Cast {input_plc} -> {output_plc} *)")

    cast_func = f"{output_plc}_TO_{input_plc}"  # IEC 61131-3 style
    builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {cast_func}({input_var}[i]);")
    builder.add_line("END_FOR;")
    return builder.build()


def generate_slice_code(layer: "SliceLayer", input_var: str, output_var: str) -> STCode:
    """
    Generate Slice layer — extract sub-tensor.

    Handles the common cases:
    - Single-axis, step=1: contiguous offset copy
    - Single-axis, step>1: strided copy
    - General: falls back to element copy (rare on PLC targets)
    """
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Slice "
        f"starts={layer.starts} ends={layer.ends} "
        f"axes={layer.axes} steps={layer.steps} *)"
    )

    if len(layer.axes) == 1 and layer.steps[0] == 1:
        # Contiguous offset copy
        start = layer.starts[0]
        axis = layer.axes[0]

        # For axis 0 or flattened data: simple offset
        if axis == 0 or not layer.input_shape:
            offset = start
        elif layer.input_shape:
            # Stride = product of dimensions after the sliced axis
            stride = int(np.prod(layer.input_shape[axis + 1 :]))
            offset = start * stride
        else:
            offset = start

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            if offset == 0:
                builder.add_line(f"{output_var}[i] := {input_var}[i];")
            else:
                builder.add_line(f"{output_var}[i] := {input_var}[i + {offset}];")
        builder.add_line("END_FOR;")

    elif len(layer.axes) == 1 and layer.steps[0] != 1:
        # Strided copy
        start = layer.starts[0]
        step = layer.steps[0]
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {input_var}[{start} + i * {step}];")
        builder.add_line("END_FOR;")

    else:
        # Multi-axis: conservative element copy
        # (In practice, multi-axis slicing on runtime activations is very rare)
        builder.add_line(f"(* Multi-axis slice — conservative copy *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {input_var}[i];")
        builder.add_line("END_FOR;")

    return builder.build()


def generate_concat_code(
    layer: "ConcatLayer", input_vars: List[str], output_var: str
) -> STCode:
    """Generate Concat layer — sequential copy of multiple inputs."""
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Concat "
        f"axis={layer.axis} inputs={len(input_vars)} *)"
    )

    offset = 0
    for idx, (inp_var, inp_size) in enumerate(zip(input_vars, layer.input_sizes)):
        builder.add_line(f"(* Concat part {idx}: {inp_var} [{inp_size} elems] *)")
        builder.add_line(f"FOR i := 0 TO {inp_size - 1} DO")
        with builder.indent():
            if offset == 0:
                builder.add_line(f"{output_var}[i] := {inp_var}[i];")
            else:
                builder.add_line(f"{output_var}[i + {offset}] := {inp_var}[i];")
        builder.add_line("END_FOR;")
        offset += inp_size

    return builder.build()


def generate_shape_code(layer: ShapeLayer, input_var: str, output_var: str) -> STCode:
    """
    Generate Shape layer — extracts shape dimensions as int64.

    This should normally be constant-folded during compilation.
    If present, emit a comment indicating this layer exists.
    """
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Shape extraction - SHOULD BE CONSTANT-FOLDED *)"
    )
    builder.add_line(
        f"(* Output shape: {layer.output_shape}, input shape: {layer.input_shape} *)"
    )
    return builder.build()


def generate_unsqueeze_code(
    layer: "UnsqueezeLayer", input_var: str, output_var: str
) -> STCode:
    """Generate Unsqueeze layer — identity copy (only logical shape changes)."""
    return generate_identity_copy(
        layer,
        input_var,
        output_var,
        f"Layer {layer.layer_id}: Unsqueeze axes={layer.unsqueeze_axes} (identity)",
    )


def generate_expand_code(
    layer: "ExpandLayer", input_var: str, output_var: str
) -> STCode:
    """
    Generate Expand (broadcast) layer.

    Three cases:
    - No actual broadcast (same size) → identity copy
    - Scalar broadcast (input_size=1) → fill
    - General broadcast → modular indexing
    """
    if layer.input_size == layer.output_size:
        return generate_identity_copy(
            layer,
            input_var,
            output_var,
            f"Layer {layer.layer_id}: Expand (no broadcast, identity)",
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Expand to {layer.target_shape} *)")

    if layer.input_size == 1:
        # Scalar broadcast
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {input_var}[0];")
        builder.add_line("END_FOR;")
    else:
        # General: output[i] = input[i MOD input_size]
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := {input_var}[i MOD {layer.input_size}];"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_gather_code(
    layer: "GatherLayer", input_var: str, output_var: str
) -> STCode:
    """
    Generate Gather layer — index into tensor with precomputed indices.

    If indices were constant (common case), we inline them as direct accesses.
    """
    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Gather axis={layer.gather_axis} *)")

    if layer.indices is not None and layer.indices.size <= 16:
        # Small constant indices → unroll
        flat_indices = layer.indices.flatten()
        for out_idx, src_idx in enumerate(flat_indices):
            builder.add_line(f"{output_var}[{out_idx}] := {input_var}[{int(src_idx)}];")
    elif layer.indices is not None:
        # Larger constant indices → generate as constant array + loop
        idx_values = ", ".join(str(int(v)) for v in layer.indices.flatten())
        builder.add_line(f"(* indices: [{idx_values}] *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            # In ST we'd need the indices as a constant array; for now use modular
            builder.add_line(f"{output_var}[i] := {input_var}[i];")
        builder.add_line("END_FOR;")
    else:
        # Dynamic indices — shouldn't happen after constant folding
        builder.add_line(f"(* WARNING: dynamic Gather indices *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {input_var}[i];")
        builder.add_line("END_FOR;")

    return builder.build()


def generate_transpose_code(
    layer: TransposeLayer, input_var: str, output_var: str
) -> STCode:
    """
    Generate Transpose layer code.

    Permutes the dimensions of a tensor stored as a flat 1-D array.
    Uses nested loops over the output shape and computes the
    corresponding input index via the inverse permutation.

    Example: NCHW→NHWC with batch stripped means (C,H,W)→(H,W,C)
             perm = (1, 2, 0)
    """
    builder = STCodeBuilder()
    in_shape = layer.input_shape
    out_shape = layer.output_shape
    perm = layer.perm
    ndim = len(out_shape)

    builder.add_line(
        f"(* Layer {layer.layer_id}: Transpose perm={perm}  "
        f"{in_shape} -> {out_shape} *)"
    )

    if ndim == 0 or layer.input_size <= 1:
        # Scalar or single element — just copy
        builder.add_line(f"{output_var}[0] := {input_var}[0];")
        return builder.build()

    # Compute input strides (row-major) for each dimension
    in_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1]

    # Compute output strides (row-major) for each dimension
    out_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1]

    # Generate unique loop variable names based on layer id and dimension
    # e.g. t3_d0, t3_d1, t3_d2 — scales to any number of dimensions
    loop_vars = [f"t{layer.layer_id}_d{d}" for d in range(ndim)]

    # Build nested loops over the OUTPUT shape
    for d in range(ndim):
        indent = "    " * d
        builder.add_line(f"{indent}FOR {loop_vars[d]} := 0 TO {out_shape[d] - 1} DO")

    # At the innermost level, compute flat indices
    inner_indent = "    " * ndim

    # Output flat index:  sum of loop_var[d] * out_stride[d]
    out_idx_parts = []
    for d in range(ndim):
        if out_strides[d] == 1:
            out_idx_parts.append(loop_vars[d])
        else:
            out_idx_parts.append(f"{loop_vars[d]} * {out_strides[d]}")
    out_idx = " + ".join(out_idx_parts)

    # Input flat index:  output dim d corresponds to input dim perm[d]
    # So input_coord[perm[d]] = loop_vars[d]
    # We need input_coord[k] for each input dim k
    # inv_perm[perm[d]] = d  →  input_coord[k] = loop_vars[inv_perm[k]]
    inv_perm = [0] * ndim
    for d in range(ndim):
        inv_perm[perm[d]] = d

    in_idx_parts = []
    for k in range(ndim):
        var = loop_vars[inv_perm[k]]
        if in_strides[k] == 1:
            in_idx_parts.append(var)
        else:
            in_idx_parts.append(f"{var} * {in_strides[k]}")
    in_idx = " + ".join(in_idx_parts)

    builder.add_line(f"{inner_indent}{output_var}[{out_idx}] := {input_var}[{in_idx}];")

    # Close loops in reverse order
    for d in range(ndim - 1, -1, -1):
        indent = "    " * d
        builder.add_line(f"{indent}END_FOR;")

    return builder.build()


# ============================================================================
# BatchNorm Code Generation
# ============================================================================


def generate_batchnorm_code(
    layer: "BatchNormLayer", input_var: str, output_var: str
) -> STCode:
    """
    Generate Structured Text for a BatchNormalization layer (inference mode).

    The extractor has already precomputed:
        combined_scale[c] = γ[c] / sqrt(σ²[c] + ε)
        combined_bias[c]  = β[c] − μ[c] · combined_scale[c]

    So at runtime:
        output[c, h, w] = combined_scale[c] * input[c, h, w] + combined_bias[c]

    This is emitted as a per-channel loop over the spatial elements.
    """
    builder = STCodeBuilder()
    lid = layer.layer_id
    C = layer.num_channels

    # Determine spatial size per channel
    if layer.input_shape and len(layer.input_shape) >= 3:
        # Shape is (C, H, W) — spatial_size = H * W
        spatial_size = int(np.prod(layer.input_shape[1:]))
    elif layer.input_shape and len(layer.input_shape) == 1:
        # 1-D: treat each element as its own "channel" (unlikely but safe)
        spatial_size = 1
    else:
        # Fallback: total_size / num_channels
        spatial_size = layer.input_size // C if C > 0 else layer.input_size

    builder.add_line(
        f"(* Layer {lid}: BatchNorm  channels={C}  spatial={spatial_size} *)"
    )

    if spatial_size == 1:
        # Simple per-channel case (e.g. after GlobalAveragePool)
        builder.add_line(f"FOR oc := 0 TO {C - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[oc] := bn_scale_{lid}[oc] * {input_var}[oc] "
                f"+ bn_bias_{lid}[oc];"
            )
        builder.add_line("END_FOR;")
    else:
        # General spatial case: for each channel, apply to all spatial positions
        builder.add_line(f"FOR oc := 0 TO {C - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR i := 0 TO {spatial_size - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"{output_var}[oc * {spatial_size} + i] := "
                    f"bn_scale_{lid}[oc] * {input_var}[oc * {spatial_size} + i] "
                    f"+ bn_bias_{lid}[oc];"
                )
            builder.add_line("END_FOR;")
        builder.add_line("END_FOR;")

    return builder.build()


# ============================================================================
# Forward Pass Generation
# ============================================================================


def generate_forward_pass(
    network: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> STCode:
    """
    Generate the forward pass computation code for all layers.

    Uses the centralized layer code generator registry from layer_generators.py
    to maintain a single source of truth for all layer-to-ST mappings.
    """
    from .layer_generators import get_global_registry

    registry = get_global_registry()
    code = STCode.empty()

    for layer_name in network.execution_order:
        layer = network.layers[layer_name]

        input_vars = get_layer_input_vars(layer, network, buffer_allocations)
        output_var = get_layer_output_var(layer, network, buffer_allocations)
        layer_code = registry.generate(layer, input_vars, output_var)
        code += layer_code
        code += STCode.blank_line()

    return code


# ============================================================================
# Main Entry Point
# ============================================================================


def generate_function_block(
    network: NetworkIR,
    fb_name: str = "NeuralNetwork",
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> STCode:
    """Generate complete function block code."""
    logger.info(
        f"Generating function block '{fb_name}' with {len(network.layers)} layers"
    )

    code = STCode.empty()
    code += generate_header(fb_name)
    code += generate_input_output_vars(network)
    code += generate_constants_section(network)
    code += generate_var_section(network, buffer_allocations)
    code += generate_forward_pass(network, buffer_allocations)
    code += generate_footer()

    logger.info(f"Generated {len(code.lines)} lines of ST code.")
    return code


def generate_merged_constants_section(
    optimization_results: Dict[str, OptimizationResult],
) -> STCode:
    """
    Generate VAR CONSTANT section from all regions.

    Collects and merges constant declarations (weights, biases, etc.) from
    all regions' IRs to produce a global constants section.

    Args:
        optimization_results: Dict mapping region_id to OptimizationResult

    Returns:
        Complete VAR CONSTANT ... END_VAR section
    """
    code = STCode.from_lines("VAR CONSTANT")

    # Track which constants we've already added (to avoid duplicates)
    added_constants = set()

    for region_id, opt_result in optimization_results.items():
        network = opt_result.ir

        for layer_name in network.execution_order:
            layer = network.layers[layer_name]
            has_constants = False

            # Weights (for linear layers AND LSTM layers)
            # For LSTM, generate_layer_weights delegates to generate_lstm_weights
            # which handles per-gate splitting with correct ONNX gate ordering.
            if hasattr(layer, "weights") and layer.weights is not None:
                const_name = f"weights_{layer.layer_id}"
                if const_name not in added_constants:
                    code += generate_layer_weights(layer).indent()
                    added_constants.add(const_name)
                    # For LSTM, also mark per-gate constant names as added
                    # to prevent the duplicate block below from re-adding them
                    if isinstance(layer, LSTMLayer):
                        for gate in ("i", "f", "g", "o"):
                            added_constants.add(f"weights_{layer.layer_id}_{gate}")
                            added_constants.add(f"recurrent_{layer.layer_id}_{gate}")
                            added_constants.add(f"bias_{layer.layer_id}_{gate}")
                    has_constants = True
            # LSTM W/R stored directly on layer (not via .weights attribute)
            elif isinstance(layer, LSTMLayer) and layer.W is not None:
                const_name = f"weights_{layer.layer_id}"
                if const_name not in added_constants:
                    code += generate_lstm_weights(layer).indent()
                    added_constants.add(const_name)
                    for gate in ("i", "f", "g", "o"):
                        added_constants.add(f"weights_{layer.layer_id}_{gate}")
                        added_constants.add(f"recurrent_{layer.layer_id}_{gate}")
                        added_constants.add(f"bias_{layer.layer_id}_{gate}")
                    has_constants = True

            # Bias (skip for LSTM — already handled above via generate_lstm_weights)
            if (
                hasattr(layer, "bias")
                and layer.bias is not None
                and not isinstance(layer, LSTMLayer)
            ):
                const_name = f"bias_{layer.layer_id}"
                if const_name not in added_constants:
                    code += generate_layer_bias(layer).indent()
                    added_constants.add(const_name)
                    has_constants = True

            # Quantization parameters (for activation quantization only)
            if isinstance(layer, (QuantizeLinearLayer, DequantizeLinearLayer)):
                if layer.input_type is not None:  # Skip weight-only dequantization
                    quant_params = generate_layer_quantization_params(layer)
                    if quant_params.lines:
                        const_name = f"quant_{layer.layer_id}"
                        if const_name not in added_constants:
                            code += quant_params.indent()
                            added_constants.add(const_name)
                            has_constants = True

            # BatchNorm precomputed parameters
            if isinstance(layer, BatchNormLayer):
                scale_name = f"bn_scale_{layer.layer_id}"
                bias_name = f"bn_bias_{layer.layer_id}"
                if scale_name not in added_constants:
                    code += generate_array_constant(
                        scale_name,
                        layer.combined_scale,
                        "REAL",
                    ).indent()
                    added_constants.add(scale_name)
                    has_constants = True

                if bias_name not in added_constants:
                    code += generate_array_constant(
                        bias_name,
                        layer.combined_bias,
                        "REAL",
                    ).indent()
                    added_constants.add(bias_name)
                    has_constants = True

            if has_constants:
                code += STCode.blank_line()

    code += STCode.from_lines("END_VAR", "")
    return code


def collect_all_variables_from_regions(
    optimization_results: Dict[str, OptimizationResult],
) -> Dict[str, Tuple[int, str]]:
    """
    Collect all variable declarations needed across all regions.

    Merges variable information from each region's IR to produce a global
    variable list. Handles conflicts by taking the maximum size for each variable.

    Args:
        optimization_results: Dict mapping region_id to OptimizationResult

    Returns:
        Dictionary mapping variable_name -> (size, plc_type)
        where size is the array size and plc_type is the PLC data type.
    """
    all_variables = {}

    for region_id, opt_result in optimization_results.items():
        network = opt_result.ir
        buffer_allocations = opt_result.buffer_allocations or {}

        # Track buffer allocations (if available)
        for tensor_name, buffer_name in buffer_allocations.items():
            if tensor_name in network.tensor_producers:
                producer_name = network.tensor_producers[tensor_name]
                if producer_name in network.layers:
                    layer = network.layers[producer_name]
                    if layer.output_type is None:
                        raise ValueError(
                            f"Layer '{layer.name}' (id={layer.layer_id}, op_type={layer.op_type}) "
                            f"has output_type=None. This indicates incomplete type resolution during IR conversion."
                        )
                    plc_type = plc_type_from_onnx_dtype(layer.output_type)
                    size = layer.output_size

                    # Keep max size for each buffer
                    if buffer_name not in all_variables:
                        all_variables[buffer_name] = (size, plc_type)
                    else:
                        existing_size, existing_type = all_variables[buffer_name]
                        all_variables[buffer_name] = (
                            max(existing_size, size),
                            existing_type,
                        )

        # Track layer output buffers (for layers not mapped to buffer allocations)
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]

            # Skip if output is a network output (handled separately)
            if any(network.is_network_output(out) for out in layer.outputs):
                continue

            # Skip if already mapped to a buffer allocation
            if any(out in buffer_allocations for out in layer.outputs):
                continue

            if layer.output_type is None:
                raise ValueError(
                    f"Layer '{layer.name}' (id={layer.layer_id}, op_type={layer.op_type}) "
                    f"has output_type=None. This indicates incomplete type resolution during IR conversion."
                )
            plc_type = plc_type_from_onnx_dtype(layer.output_type)
            var_name = f"layer_{layer.layer_id}_output"

            if var_name not in all_variables:
                all_variables[var_name] = (layer.output_size, plc_type)

        # Collect LSTM gate and state buffers
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]
            if isinstance(layer, LSTMLayer):
                h = layer.hidden_size
                # Gate buffers (used inside the timestep loop)
                for gate in ("i_gate", "f_gate", "g_gate", "o_gate"):
                    var_name = f"{gate}_{layer.layer_id}"
                    if var_name not in all_variables:
                        all_variables[var_name] = (h, "REAL")
                # State buffers (updated across timesteps)
                for state in ("h_state", "c_state"):
                    var_name = f"{state}_{layer.layer_id}"
                    if var_name not in all_variables:
                        all_variables[var_name] = (h, "REAL")

    return all_variables


def generate_merged_var_section(all_variables: Dict[str, Tuple[int, str]]) -> STCode:
    """Generate VAR section from merged variables across all regions."""
    builder = STCodeBuilder()
    builder.add_line("VAR")

    if all_variables:
        builder.add_line("    (* Merged variables from all regions *)")

        with builder.indent():
            for var_name, (size, dtype) in sorted(all_variables.items()):
                builder.add_line(f"{var_name} : ARRAY[0..{size - 1}] OF {dtype};")

        builder.add_line("")

    # TODO: Make this smarter. We don't need all of these variables for all programs.
    # TODO: Moreover, we might have to handle certain variables differently. Do we need multiple of certain set of variables for certain neural nets?
    # Temporary computation variables
    with builder.indent():
        builder.add_line("(* Temporary computation variables *)")
        builder.add_line("i : DINT;")
        builder.add_line("j : DINT;")
        builder.add_line("t : DINT;")
        builder.add_line("sum : REAL;")

    # Extra computation helpers
    with builder.indent():
        builder.add_line("(* Computation helpers *)")
        builder.add_line("max_val : REAL;")
        builder.add_line("exp_val : REAL;")
        builder.add_line("exp_sum : REAL;")

    # Spatial loop variables
    with builder.indent():
        builder.add_line("(* Spatial loop variables *)")
        builder.add_line("oc : DINT;")
        builder.add_line("oh : DINT;")
        builder.add_line("ow : DINT;")
        builder.add_line("ic : DINT;")
        builder.add_line("kh : DINT;")
        builder.add_line("kw : DINT;")
        builder.add_line("ih : DINT;")
        builder.add_line("iw : DINT;")

    builder.add_line("END_VAR")
    builder.add_line("")

    return builder.build()


def generate_model_function_block(
    model: ModelIR,
    optimization_results: Dict[str, OptimizationResult],
    fb_name: str = "NeuralNetwork",
) -> STCode:
    """
    Generate complete function block code for a multi-region ModelIR.

    Uses region-kind-aware lowerers to generate ST for each region.
    Delegates region-specific code generation to lowerers.py to keep this
    function focused on high-level orchestration and cross-region concerns.

    Args:
        model: Regionized model with potentially multiple regions
        optimization_results: Dict mapping region_id to OptimizationResult
        fb_name: Name for the generated function block

    Returns:
        Complete ST code for the function block
    """
    from .lowerers import lower_region_to_st

    logger.info(
        f"Generating function block '{fb_name}' for model with {len(model.regions)} regions"
    )

    code = STCode.empty()
    code += generate_header(fb_name)

    # Generate input/output variables from first and last regions
    # (assuming linear pipeline of regions)
    if model.regions:
        first_region_id = model.regions[0].region_id
        last_region_id = model.regions[-1].region_id

        first_ir = optimization_results[first_region_id].ir
        last_ir = optimization_results[last_region_id].ir

        # Try to generate proper I/O declarations
        if first_ir.input_tensors and last_ir.output_tensors:
            code += STCode.from_lines("    (* Model Inputs/Outputs *)")
            code += generate_input_output_vars(first_ir)  # Reuse existing function
        else:
            code += STCode.from_lines("    (* Model Inputs/Outputs *)")
            code += STCode.from_lines("VAR_INPUT")
            code += STCode.from_lines("    input_data : ARRAY[0..1023] OF REAL;")
            code += STCode.from_lines("END_VAR")
            code += STCode.from_lines("")
            code += STCode.from_lines("VAR_OUTPUT")
            code += STCode.from_lines("    output_data : ARRAY[0..255] OF REAL;")
            code += STCode.from_lines("END_VAR")
            code += STCode.from_lines("")

    # Generate merged constants section from all regions
    merged_constants = generate_merged_constants_section(optimization_results)
    code += merged_constants

    # Collect and merge all variables from all regions
    all_variables = collect_all_variables_from_regions(optimization_results)
    code += generate_merged_var_section(all_variables)

    code += st_comment("Forward pass execution")

    for region in model.regions:
        code += STCode.blank_line()
        code += st_comment(f"Region: {region.region_id} [{region.kind.name}]")

        if region.region_id not in optimization_results:
            raise KeyError(
                f"No optimization result for region {region.region_id} ({region.kind.name})"
            )

        optimization_result = optimization_results[region.region_id]

        # Use region-kind-specific lowerer (fail-fast: exceptions propagate)
        region_code = lower_region_to_st(region, optimization_result)
        code += region_code

    code += generate_footer()
    return code


def translate_ir_to_st(
    ir: NetworkIR, fb_name: str = "NeuralNetwork", buffer_allocations=None
) -> str:
    """Translate the given NetworkIR to Structured Text code."""
    builder = STCodeBuilder()
    builder += generate_function_block(ir, fb_name, buffer_allocations)
    # TODO: might need to add openplc config / straton config generation later
    return str(builder.build())


def translate_model_to_st(
    model: ModelIR,
    optimization_results: Dict[str, OptimizationResult],
    fb_name: str = "NeuralNetwork",
) -> str:
    """Translate the given ModelIR to Structured Text code."""
    builder = STCodeBuilder()
    builder += generate_model_function_block(model, optimization_results, fb_name)
    return str(builder.build())
