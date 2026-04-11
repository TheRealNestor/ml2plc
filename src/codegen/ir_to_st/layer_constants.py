"""
Layer constant generation helpers.

Centralized functions for generating layer constants (weights, biases, quantization parameters).
Specific layer code generation is delegated to layers/* modules.
"""

from .st_code import STCode
from .utils.constant_helpers import (
    generate_weights_constants,
    generate_lstm_weights_constants,
    generate_gru_weights_constants,
    generate_bias_constant,
    generate_quantization_params,
    generate_array_constant,
)


def generate_layer_weights(layer) -> STCode:
    """Generate weight constants for a layer."""
    from ..types import LSTMLayer, GRULayer

    if isinstance(layer, LSTMLayer):
        return generate_lstm_weights_constants(layer)
    if isinstance(layer, GRULayer):
        return generate_gru_weights_constants(layer)
    from ..types import LinearLayer

    is_quantized = isinstance(layer, LinearLayer) and layer.is_quantized()
    return generate_weights_constants(layer, is_integer=is_quantized)


def generate_layer_bias(layer) -> STCode:
    """Generate bias constant for a layer."""
    from ..types import LSTMLayer, GRULayer

    if isinstance(layer, (LSTMLayer, GRULayer)):
        return STCode.empty()
    return generate_bias_constant(layer)


def generate_layer_quantization_params(layer) -> STCode:
    """Generate quantization parameters."""
    return generate_quantization_params(layer)


def generate_layer_rhs_constants(layer) -> STCode:
    """Generate RHS constant declarations for binary elementwise layers."""
    from ..types import BinaryElementwiseLayer, EinsumLayer

    if isinstance(layer, EinsumLayer):
        return generate_array_constant(
            f"einsum_rhs_{layer.layer_id}",
            layer.rhs_const,
            "REAL",
        )

    if not isinstance(layer, BinaryElementwiseLayer) or layer.rhs_const is None:
        return STCode.empty()

    return generate_array_constant(
        f"rhs_const_{layer.layer_id}",
        layer.rhs_const,
        "REAL",
    )
