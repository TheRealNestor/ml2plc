"""
Layer constant generation helpers.

Centralized functions for generating layer constants (weights, biases, quantization parameters).
Specific layer code generation is delegated to layers/* modules.
"""

from .st_code import STCode
from .utils.constant_helpers import (
    generate_weights_constants,
    generate_lstm_weights_constants,
    generate_bias_constant,
    generate_quantization_params,
)


def generate_layer_weights(layer) -> STCode:
    """Generate weight constants for a layer."""
    from ..types import LSTMLayer

    if isinstance(layer, LSTMLayer):
        return generate_lstm_weights_constants(layer)
    from ..types import LinearLayer

    is_quantized = isinstance(layer, LinearLayer) and layer.is_quantized()
    return generate_weights_constants(layer, is_integer=is_quantized)


def generate_layer_bias(layer) -> STCode:
    """Generate bias constant for a layer."""
    from ..types import LSTMLayer

    if isinstance(layer, LSTMLayer):
        return STCode.empty()
    return generate_bias_constant(layer)


def generate_layer_quantization_params(layer) -> STCode:
    """Generate quantization parameters."""
    return generate_quantization_params(layer)
