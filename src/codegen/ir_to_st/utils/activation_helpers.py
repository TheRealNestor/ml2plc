"""
Activation function code generation utilities.

Provides helpers for generating ST code for various activation functions,
supporting both inline (within matmul) and separate loop implementations.
"""

from ..st_code import STCode, STCodeBuilder
from ..variable import Variable
from ...types import ActivationType


def generate_activation_inline(
    activation: ActivationType,
    expr: str,
) -> str:
    """
    Generate an inlinable expression for an activation function.

    Used when activation can be computed as part of a larger expression.

    Args:
        activation: Activation type
        expr: Expression to apply activation to

    Returns:
        ST expression string (e.g., "1.0 / (1.0 + EXP(-expr))" for sigmoid)

    Example:
        >>> generate_activation_inline(ActivationType.RELU, "sum")
        'MAX(sum, 0.0)'
    """
    if activation == ActivationType.NONE:
        return expr

    elif activation == ActivationType.RELU:
        return f"MAX({expr}, 0.0)"

    elif activation == ActivationType.SIGMOID:
        return f"1.0 / (1.0 + EXP(-({expr})))"

    elif activation == ActivationType.TANH:
        return f"(EXP({expr}) - EXP(-({expr}))) / " f"(EXP({expr}) + EXP(-({expr})))"

    else:
        # For activations that can't be inlined, return expression unchanged
        # (caller should use separate activation loop)
        return expr


def generate_activation_loop(
    activation: ActivationType,
    input_var: Variable,
    output_var: Variable,
    size: int,
) -> STCode:
    """
    Generate a separate loop for activation function computation.

    Used for activations that can't be inlined (Softmax, etc.) or require
    multi-pass computation.

    Args:
        activation: Activation type
        input_var: Input array variable name
        output_var: Output array variable name
        size: Array size

    Returns:
        STCode with for-loop implementing activation

    Example:
        >>> code = generate_activation_loop(ActivationType.RELU, "x", "y", 100)
        >>> print(code)
        FOR i := 0 TO 99 DO
            y[i] := MAX(x[i], 0.0);
        END_FOR;
    """
    builder = STCodeBuilder()

    # Accept either Variable or raw name string for backward compatibility
    from ..variable import ensure_var

    input_var = ensure_var(input_var, (size,))
    output_var = ensure_var(output_var, (size,))

    if activation == ActivationType.NONE:
        # No-op activation
        return STCode.empty()

    elif activation == ActivationType.RELU:
        builder.add_line(f"FOR i := 0 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var.at('i')} := MAX({input_var.at('i')}, 0.0);")
        builder.add_line("END_FOR;")

    elif activation == ActivationType.SIGMOID:
        builder.add_line(f"FOR i := 0 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := 1.0 / (1.0 + EXP(-{input_var.at('i')}));"
            )
        builder.add_line("END_FOR;")

    elif activation == ActivationType.TANH:
        builder.add_line(f"FOR i := 0 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := (EXP({input_var.at('i')}) - EXP(-{input_var.at('i')})) / "
                f"(EXP({input_var.at('i')}) + EXP(-{input_var.at('i')}));"
            )
        builder.add_line("END_FOR;")

    elif activation == ActivationType.SOFTMAX:
        # Softmax requires 3 passes: find max, compute exp sum, normalize
        builder.add_line("(* Softmax: find max *)")
        builder.add_line(f"max_val := {input_var.at('0')};")
        builder.add_line(f"FOR i := 1 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(f"IF {input_var.at('i')} > max_val THEN")
            with builder.indent():
                builder.add_line(f"max_val := {input_var.at('i')};")
            builder.add_line("END_IF;")
        builder.add_line("END_FOR;")
        builder.add_line("")

        builder.add_line("(* Softmax: compute exponentials and sum *)")
        builder.add_line("exp_sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := EXP({input_var.at('i')} - max_val);"
            )
            builder.add_line(f"exp_sum := exp_sum + {output_var.at('i')};")
        builder.add_line("END_FOR;")
        builder.add_line("")

        builder.add_line("(* Softmax: normalize *)")
        builder.add_line(f"FOR i := 0 TO {size - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var.at('i')} := {output_var.at('i')} / exp_sum;")
        builder.add_line("END_FOR;")

    else:
        # Unknown activation
        builder.add_line(f"(* WARNING: Unknown activation {activation.name} *)")

    return builder.build()


def supports_inline_activation(activation: ActivationType) -> bool:
    """
    Check if activation can be inlined within matrix multiplication.

    Some activations (ReLU, identity) can be computed as part of the accumulation,
    while others (Softmax, LayerNorm) require separate passes.

    Args:
        activation: Activation type

    Returns:
        True if activation can be safely inlined
    """
    # These can be computed element-wise without multi-pass logic
    inlinable = {
        ActivationType.NONE,
        ActivationType.RELU,
        ActivationType.SIGMOID,
        ActivationType.TANH,
    }
    return activation in inlinable
