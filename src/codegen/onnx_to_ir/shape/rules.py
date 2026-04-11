"""Public shape-rules API.

This is the canonical entrypoint for ONNX->IR shape rules.
"""

from .primitives import (
    infer_add_output_shape,
    infer_batchnorm_output_shape,
    infer_cast_output_shape,
    infer_conv2d_output_shape,
    infer_einsum_output_shape,
    infer_expand_output_shape,
    infer_flatten_output_shape,
    infer_gemm_output_shape,
    infer_global_avg_pool_output_shape,
    infer_matmul_output_shape,
    infer_pool2d_output_shape,
    infer_reduce_mean_output_shape,
    infer_reshape_output_shape,
    infer_slice_output_shape,
    infer_squeeze_output_shape,
    infer_transpose_output_shape,
    infer_unsqueeze_output_shape,
)
from .rules_registry import (
    OP_SHAPE_RULES,
    OpSemanticsContext,
    extract_int_tuple_from_input,
    extract_reshape_target_shape,
    first_resolved_output_shape,
    infer_output_shape_from_semantics,
    register_op_shape_rule,
    resolved_input_role,
)

__all__ = [
    "OP_SHAPE_RULES",
    "OpSemanticsContext",
    "extract_int_tuple_from_input",
    "extract_reshape_target_shape",
    "first_resolved_output_shape",
    "infer_add_output_shape",
    "infer_batchnorm_output_shape",
    "infer_cast_output_shape",
    "infer_conv2d_output_shape",
    "infer_einsum_output_shape",
    "infer_expand_output_shape",
    "infer_flatten_output_shape",
    "infer_gemm_output_shape",
    "infer_global_avg_pool_output_shape",
    "infer_matmul_output_shape",
    "infer_output_shape_from_semantics",
    "infer_pool2d_output_shape",
    "infer_reduce_mean_output_shape",
    "infer_reshape_output_shape",
    "infer_slice_output_shape",
    "infer_squeeze_output_shape",
    "infer_transpose_output_shape",
    "infer_unsqueeze_output_shape",
    "register_op_shape_rule",
    "resolved_input_role",
]
