"""
Linear layer code generators (MatMul, Gemm, Fused variants).

Handles matrix multiplication with optional bias and fused activation.
"""

from ...types import LinearLayer, ActivationType
from ..st_code import STCode, STCodeBuilder
from ..type_conversion import numpy_to_plc_cast_func
from ..utils.activation_helpers import (
    generate_activation_inline,
    generate_activation_loop,
)
from ..variable import Variable
from ..utils.constant_helpers import is_uniform_array


INLINE_ACTIVATIONS = {ActivationType.NONE, ActivationType.RELU}


def get_layer_type_name(layer: LinearLayer, activation: ActivationType) -> str:
    """Get descriptive name for layer type."""
    from ...types import FusedGemmLayer, FusedLinearLayer, GemmLayer

    if isinstance(layer, FusedGemmLayer):
        return f"Fused Gemm + {activation.name}"
    elif isinstance(layer, FusedLinearLayer):
        return f"Fused Linear + {activation.name}"
    elif isinstance(layer, GemmLayer):
        return "Gemm"
    else:
        return "MatMul"


def generate_weight_access(
    layer: LinearLayer, input_var: Variable, layer_id: int, output_size: int
) -> str:
    """Generate weight multiplication expression."""
    weight_var = Variable(
        name=f"weights_{layer_id}", shape=(layer.output_size * layer.input_size,)
    )

    if not layer.is_quantized():
        return f"{input_var.at('i')} * {weight_var.at(f'i * {output_size} + j')}"

    cast_func = numpy_to_plc_cast_func(layer.weights.dtype, "REAL")

    scale_expr = (
        f"weight_scale_{layer_id}"
        if is_uniform_array(layer.weight_scale)
        else f"weight_scale_{layer_id}[j]"
    )

    zp_expr = (
        f"weight_zero_point_{layer_id}"
        if is_uniform_array(layer.weight_zero_point)
        else f"weight_zero_point_{layer_id}[j]"
    )

    return f"{input_var.at('i')} * ({scale_expr} * {cast_func}({weight_var.at(f'i * {output_size} + j')} - {zp_expr}))"


def build_final_linear_layer_expression(layer: LinearLayer, has_bias: bool) -> str:
    """Build final expression with alpha, bias, beta."""
    alpha = getattr(layer, "alpha", 1.0)
    beta = getattr(layer, "beta", 1.0)

    expr = "sum"
    if alpha != 1.0:
        expr = f"{alpha} * {expr}"

    if has_bias:
        bias_var = Variable(name=f"bias_{layer.layer_id}", shape=(layer.output_size,))
        bias_term = f"{bias_var.at('j')}"
        if beta != 1.0:
            bias_term = f"{beta} * {bias_term}"
        expr = f"{expr} + {bias_term}"

    return expr


def generate_linear_layer_code(
    layer: LinearLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate code for linear layer types (MatMul, Gemm, Fused variants)."""
    builder = STCodeBuilder()

    activation = getattr(layer, "activation", ActivationType.NONE)
    layer_type_name = get_layer_type_name(layer, activation)

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
        activated_expr = generate_activation_inline(activation, final_expr)

    builder.add_line(f"{output_var.at('j')} := {activated_expr};")

    builder.add_line("END_FOR;")

    # Separate activation pass if needed
    if activation not in INLINE_ACTIVATIONS:
        builder.add_line("")
        builder.add_code(
            generate_activation_loop(
                activation, output_var, output_var, layer.output_size
            )
        )

    return builder.build()
