"""
Layer-specific code generation implementations organized by layer category.

Modules:
  linear: Linear layers (MatMul, Gemm, Fused variants)
  spatial: Spatial layers (Conv2D, Pool2D, BatchNorm)
  recurrent: Recurrent layers (LSTM, GRU)
  data_movement: Data movement and shape manipulation
"""

from .linear import generate_linear_layer_code
from .spatial import generate_conv2d_code, generate_pool2d_code, generate_batchnorm_code
from .recurrent import generate_lstm_code, generate_gru_code
from .data_movement import (
    generate_activation_layer_code,
    generate_add_code,
    generate_reshape_code,
    generate_quantize_linear_code,
    generate_dequantize_linear_code,
    generate_dropout_code,
    generate_flatten_code,
    generate_squeeze_code,
    generate_cast_code,
    generate_slice_code,
    generate_concat_code,
    generate_transpose_code,
    generate_unsqueeze_code,
    generate_expand_code,
    generate_shape_code,
    generate_gather_code,
    generate_reduce_mean_code,
    generate_reduce_prod_code,
    generate_binary_elementwise_code,
    generate_unary_elementwise_code,
    generate_runtime_matmul_code,
    generate_einsum_code,
)

__all__ = [
    "generate_linear_layer_code",
    "generate_conv2d_code",
    "generate_pool2d_code",
    "generate_batchnorm_code",
    "generate_lstm_code",
    "generate_gru_code",
    "generate_activation_layer_code",
    "generate_add_code",
    "generate_reshape_code",
    "generate_quantize_linear_code",
    "generate_dequantize_linear_code",
    "generate_dropout_code",
    "generate_flatten_code",
    "generate_squeeze_code",
    "generate_cast_code",
    "generate_slice_code",
    "generate_concat_code",
    "generate_transpose_code",
    "generate_unsqueeze_code",
    "generate_expand_code",
    "generate_shape_code",
    "generate_gather_code",
    "generate_reduce_mean_code",
    "generate_reduce_prod_code",
    "generate_binary_elementwise_code",
    "generate_unary_elementwise_code",
    "generate_runtime_matmul_code",
    "generate_einsum_code",
]
