"""
IR to Structured Text Code Generation Module

This module is responsible for generating Structured Text (ST) code from the intermediate representation (IR) of a neural network.
"""

from ..types import *
from .st_code import *
from .type_conversion import *

import logging

logger = logging.getLogger(__name__)

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
            producer_layer = network.layers[producer_name]
            input_vars.append(f"layer_{producer_layer.layer_id}_output")

    return input_vars


def get_layer_output_var(
    layer: BaseLayer,
    network: NetworkIR,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> str:
    """Get output variable name for a layer."""

    output_tensor = layer.outputs[0]  # Assuming single output for simplicity

    if network.is_network_output(output_tensor):
        return "output_data"

    if buffer_allocations and output_tensor in buffer_allocations:
        return buffer_allocations[output_tensor]

    return f"layer_{layer.layer_id}_output"


# Configuration: which activations to inline (vs separate loop)
INLINE_ACTIVATIONS = {
    ActivationType.NONE,
    ActivationType.RELU,
    # ActivationType.SIGMOID,
    # ActivationType.TANH,
}


def apply_activation_inline(activation: ActivationType, expr: str) -> str:
    """Apply activation inline if possible, otherwise return expression unchanged."""
    if activation == ActivationType.RELU:
        return f"MAX({expr}, 0.0)"
    elif activation == ActivationType.SIGMOID:
        return f"1.0 / (1.0 + EXP(-({expr})))"
    elif activation == ActivationType.TANH:
        return f"((EXP({expr}) - EXP(-({expr}))) / (EXP({expr}) + EXP(-({expr}))))"
    else:  # NONE, SOFTMAX, ...
        return expr


def needs_separate_activation(activation: ActivationType) -> bool:
    """Check if activation needs separate loop."""
    return activation not in INLINE_ACTIVATIONS


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
    return STCode.from_lines(f"FUNCTION_BLOCK {fb_name}", "")


def generate_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK", "")


def generate_input_output_vars(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT and VAR_OUTPUT sections."""
    code = STCode.empty()

    first_layer_name = network.execution_order[0]
    first_layer = network.layers[first_layer_name]

    last_layer_name = network.execution_order[-1]
    last_layer = network.layers[last_layer_name]

    input_type = plc_type_from_onnx_dtype(first_layer.input_type)
    code += STCode.from_lines(
        "VAR_INPUT",
        f"    input_data : ARRAY[0..{first_layer.input_size - 1}] OF {input_type};",
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


def generate_array_constant(
    name: str, values: np.ndarray, plc_type: str, is_integer: bool = False
) -> STCode:
    """
    Generate a constant array declaration.

    Args:
        name: Variable name
        values: NumPy array of values
        plc_type: PLC type string
        is_integer: If True, format as integers; otherwise as floats
    """
    flat_values = values.flatten()

    if is_integer:
        value_str = ", ".join(str(int(val)) for val in flat_values)
    else:
        value_str = ", ".join(f"{val:.6f}" for val in flat_values)

    return STCode.from_lines(
        f"{name} : ARRAY[0..{values.size - 1}] OF {plc_type} := [{value_str}];"
    )


def generate_scalar_constant(
    name: str, value: float | int, plc_type: str, is_integer: bool = False
) -> STCode:
    """Generate a scalar constant declaration."""
    if is_integer:
        value_str = str(int(value))
    else:
        value_str = str(float(value))

    return STCode.from_lines(f"{name} : {plc_type} := {value_str};")


def generate_layer_weights(layer) -> STCode:
    """
    Generate weight constants for a layer (handles both float and quantized).

    Returns all weight-related constants:
    - weights array
    - weight_scale (if quantized)
    - weight_zero_point (if quantized)
    """
    builder = STCodeBuilder()

    is_quantized = isinstance(layer, LinearLayer) and layer.is_quantized()

    if is_quantized:
        weight_type = numpy_to_plc_type(layer.weights.dtype)
        weights_code = generate_array_constant(
            f"weights_{layer.layer_id}", layer.weights, weight_type, is_integer=True
        )
    else:
        weight_type = plc_type_from_onnx_dtype(layer.input_type)
        weights_code = generate_array_constant(
            f"weights_{layer.layer_id}", layer.weights, weight_type, is_integer=False
        )

    builder.add_code(weights_code)

    # Generate quantization parameters if present
    if is_quantized:
        # Scale - use scalar if uniform
        if is_uniform_array(layer.weight_scale):
            builder.add_code(
                generate_scalar_constant(
                    f"weight_scale_{layer.layer_id}",
                    float(layer.weight_scale.flat[0]),
                    "REAL",
                )
            )
        else:
            builder.add_code(
                generate_array_constant(
                    f"weight_scale_{layer.layer_id}", layer.weight_scale, "REAL"
                )
            )

        # Zero point - use scalar if uniform
        zp_type = numpy_to_plc_type(layer.weights.dtype)
        if is_uniform_array(layer.weight_zero_point):
            builder.add_code(
                generate_scalar_constant(
                    f"weight_zero_point_{layer.layer_id}",
                    int(layer.weight_zero_point.flat[0]),
                    zp_type,
                    is_integer=True,
                )
            )
        else:
            builder.add_code(
                generate_array_constant(
                    f"weight_zero_point_{layer.layer_id}",
                    layer.weight_zero_point,
                    zp_type,
                    is_integer=True,
                )
            )

    return builder.build()


def generate_layer_bias(layer) -> STCode:
    """Generate bias constant for a layer."""
    bias_type = plc_type_from_onnx_dtype(layer.output_type)
    return generate_array_constant(f"bias_{layer.layer_id}", layer.bias, bias_type)


def generate_layer_quantization_params(layer) -> STCode:
    """
    Generate quantization parameters for QuantizeLinear/DequantizeLinear layers.
    Only generates arrays for per-channel quantization (per-tensor is inlined).
    """
    # Only generate if per-channel (size > 1)
    if layer.scale.size == 1:
        return STCode.empty()

    builder = STCodeBuilder()

    # Scale array
    builder.add_code(
        generate_array_constant(f"scale_{layer.layer_id}", layer.scale, "REAL")
    )

    # Zero point array
    if isinstance(layer, QuantizeLinearLayer):
        dtype_str = layer.output_type
    else:  # DequantizeLinearLayer
        dtype_str = layer.input_type

    zp_type = plc_type_from_onnx_dtype(dtype_str)
    builder.add_code(
        generate_array_constant(
            f"zero_point_{layer.layer_id}", layer.zero_point, zp_type, is_integer=True
        )
    )

    return builder.build()


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
            code += generate_array_constant(
                f"bn_scale_{layer.layer_id}",
                layer.combined_scale,
                "REAL",
            ).indent()
            code += generate_array_constant(
                f"bn_bias_{layer.layer_id}",
                layer.combined_bias,
                "REAL",
            ).indent()
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
    """Generate activation code for activations that need separate loops."""
    builder = STCodeBuilder()

    if activation == ActivationType.NONE:
        # Identity - should never reach here (handled inline)
        raise ValueError("NONE activation should be handled inline")

    elif activation == ActivationType.RELU:
        # ReLU - can be inline but also support separate
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := MAX({input_var}[i], 0.0);")
        builder.add_line("END_FOR;")

    elif activation == ActivationType.SIGMOID:
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := 1.0 / (1.0 + EXP(-{input_var}[i]));")
        builder.add_line("END_FOR;")

    elif activation == ActivationType.TANH:
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := (EXP({input_var}[i]) - EXP(-{input_var}[i])) / "
                f"(EXP({input_var}[i]) + EXP(-{input_var}[i]));"
            )
        builder.add_line("END_FOR;")

    elif activation == ActivationType.SOFTMAX:
        # Find maximum value
        builder.add_line(f"max_val := {input_var}[0];")
        builder.add_line(f"FOR i := 1 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"IF {input_var}[i] > max_val THEN")
            with builder.indent():
                builder.add_line(f"max_val := {input_var}[i];")
            builder.add_line("END_IF;")
        builder.add_line("END_FOR;")
        builder.add_line("")

        # Compute exp sum
        builder.add_line("exp_sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := EXP({input_var}[i] - max_val);")
            builder.add_line(f"exp_sum := exp_sum + {output_var}[i];")
        builder.add_line("END_FOR;")
        builder.add_line("")

        # Normalize
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {output_var}[i] / exp_sum;")
        builder.add_line("END_FOR;")

    return builder.build()


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
        activated_expr = apply_activation_inline(activation, final_expr)
        builder.add_line(f"{output_var}[j] := {activated_expr};")

    builder.add_line("END_FOR;")

    # Separate activation pass if needed
    if needs_separate_activation(activation):
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

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Reshape (copy input to output) *)")
    builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {input_var}[i];")
    builder.add_line("END_FOR;")

    return builder.build()


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
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Flatten (axis={layer.axis}) — copy *)"
    )
    builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {input_var}[i];")
    builder.add_line("END_FOR;")
    return builder.build()


def generate_squeeze_code(
    layer: "SqueezeLayer", input_var: str, output_var: str
) -> STCode:
    """
    Generate Squeeze layer code.

    Squeeze removes dimensions of size 1.  Since both input and output
    are stored as flat 1-D arrays with the same number of elements,
    this is simply a memcopy — identical to Flatten.
    """
    builder = STCodeBuilder()
    axes_str = ",".join(str(a) for a in layer.axes) if layer.axes else "auto"
    builder.add_line(f"(* Layer {layer.layer_id}: Squeeze *)")
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


# Mapping from layer type to code generator
def _single_input_wrapper(generator_func):
    """Wraps a generator function to handle single input layers."""
    return lambda layer, inputs, output: generator_func(layer, inputs[0], output)


LAYER_CODE_GENERATORS = {
    MatMulLayer: _single_input_wrapper(generate_linear_layer_code),
    GemmLayer: _single_input_wrapper(generate_linear_layer_code),
    FusedGemmLayer: _single_input_wrapper(generate_linear_layer_code),
    FusedLinearLayer: _single_input_wrapper(generate_linear_layer_code),
    AddLayer: generate_add_code,
    ReshapeLayer: _single_input_wrapper(generate_reshape_code),
    ActivationLayer: _single_input_wrapper(generate_activation_layer_code),
    QuantizeLinearLayer: _single_input_wrapper(generate_quantize_linear_code),
    DequantizeLinearLayer: _single_input_wrapper(generate_dequantize_linear_code),
    DropoutLayer: _single_input_wrapper(generate_dropout_code),
    Conv2DLayer: _single_input_wrapper(generate_conv2d_code),
    Pool2DLayer: _single_input_wrapper(generate_pool2d_code),
    FlattenLayer: _single_input_wrapper(generate_flatten_code),
    TransposeLayer: _single_input_wrapper(generate_transpose_code),
    BatchNormLayer: _single_input_wrapper(generate_batchnorm_code),
    SqueezeLayer: _single_input_wrapper(generate_squeeze_code),
}


def generate_forward_pass(
    network: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> STCode:
    """Generate the forward pass computation code for all layers."""
    code = STCode.empty()
    for layer_name in network.execution_order:
        layer = network.layers[layer_name]

        input_vars = get_layer_input_vars(layer, network, buffer_allocations)
        output_var = get_layer_output_var(layer, network, buffer_allocations)

        # Generate code for this layer
        layer_type = type(layer)
        if layer_type in LAYER_CODE_GENERATORS:
            layer_code = LAYER_CODE_GENERATORS[layer_type](
                layer, input_vars, output_var
            )
            code += layer_code
            code += STCode.blank_line()
        else:
            logger.warning(f"No code generator for layer type {layer_type.__name__}")

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


def translate_ir_to_st(
    ir: NetworkIR, fb_name: str = "NeuralNetwork", buffer_allocations=None
) -> str:
    """Translate the given NetworkIR to Structured Text code."""
    builder = STCodeBuilder()
    builder += generate_function_block(ir, fb_name, buffer_allocations)
    # TODO: might need to add openplc config / straton config generation later
    return str(builder.build())
