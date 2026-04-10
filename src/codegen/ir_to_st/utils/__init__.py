"""
Utilities for Structured Text code generation.

This package provides reusable components for generating ST code, organized by concern:

LOW-LEVEL (Formatting & Syntax):
- st_code.py: STCode/STCodeBuilder, comments, declarations, control flow
  Use for: Any code generation needing proper formatting

LAYER-SPECIFIC PATTERNS:
- constant_helpers: Weight, bias, and quantization constant generation
- loop_helpers: Layer-specific loop patterns, boundary checking
- array_helpers: Multidimensional array indexing and strided access
- activation_helpers: Activation function code generation
- copy_helpers: Data movement and copy patterns

ORGANIZATION PRINCIPLE:
- st_code: General-purpose, used everywhere
- utils/*: Layer-specific patterns, composed from st_code primitives
"""

from .constant_helpers import (
    generate_array_constant,
    generate_scalar_constant,
    generate_weights_constants,
    generate_lstm_weights_constants,
    generate_gru_weights_constants,
    generate_bias_constant,
    generate_quantization_params,
    generate_batchnorm_constants,
)


from .array_helpers import (
    compute_flat_index,
    compute_nd_indices,
    compute_array_stride,
    compute_conv_indices,
    compute_pool_indices,
)

from .activation_helpers import (
    generate_activation_inline,
    generate_activation_loop,
)

from .copy_helpers import (
    generate_simple_copy,
    generate_scalar_broadcast,
    generate_modulo_broadcast,
    generate_offset_copy,
    generate_strided_copy,
)

__all__ = [
    # constant_helpers
    "generate_array_constant",
    "generate_scalar_constant",
    "generate_weights_constants",
    "generate_lstm_weights_constants",
    "generate_bias_constant",
    "generate_quantization_params",
    "generate_batchnorm_constants",
    # array_helpers
    "compute_flat_index",
    "compute_nd_indices",
    "compute_array_stride",
    "compute_conv_indices",
    "compute_pool_indices",
    # activation_helpers
    "generate_activation_inline",
    "generate_activation_loop",
    # copy_helpers
    "generate_simple_copy",
    "generate_scalar_broadcast",
    "generate_modulo_broadcast",
    "generate_offset_copy",
    "generate_strided_copy",
]
