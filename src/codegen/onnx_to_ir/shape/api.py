"""Canonical ONNX->IR shape validation and inference APIs."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .engine import infer_layer_shapes
from .validation import (
    ShapeResolutionReport,
    ShapeValidationError,
    validate_model_shapes,
)
from .rules import (
    OP_SHAPE_RULES,
    OpSemanticsContext,
    extract_int_tuple_from_input,
    extract_reshape_target_shape,
    first_resolved_output_shape,
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
    infer_output_shape_from_semantics,
    infer_pool2d_output_shape,
    infer_reduce_mean_output_shape,
    infer_reshape_output_shape,
    infer_slice_output_shape,
    infer_squeeze_output_shape,
    infer_transpose_output_shape,
    infer_unsqueeze_output_shape,
    register_op_shape_rule,
    resolved_input_role,
)

logger = logging.getLogger(__name__)


def get_feature_sizes(
    input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
) -> Tuple[int, int]:
    """Get flattened feature sizes for PLC buffer allocation/codegen."""
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0
    return input_size, output_size


def validate_inferred_shapes(
    layer_name: str,
    op_type: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    weight_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """Validate inferred shapes for key ops used by extraction/codegen."""
    if not output_shape:
        raise ValueError(f"Layer {layer_name} ({op_type}): Output shape is empty")

    if not input_shape:
        logger.warning(f"Layer {layer_name} ({op_type}): Input shape is empty")

    if op_type in ["Gemm", "FusedGemm"] and weight_shape:
        if len(weight_shape) != 2:
            raise ValueError(
                f"Layer {layer_name} ({op_type}): "
                f"Weight must be 2D, got {weight_shape}"
            )

        if input_shape and weight_shape:
            input_features = input_shape[-1]
            weight_input_features = weight_shape[0]
            if input_features != weight_input_features:
                raise ValueError(
                    f"Layer {layer_name} ({op_type}): "
                    f"Dimension mismatch - input features {input_features} "
                    f"!= weight input features {weight_input_features}"
                )

    if op_type == "MatMul" and weight_shape and input_shape:
        if len(weight_shape) < 1:
            raise ValueError(
                f"Layer {layer_name} ({op_type}): Invalid RHS shape {weight_shape}"
            )

        rhs_contract = weight_shape[-2] if len(weight_shape) >= 2 else weight_shape[0]
        lhs_contract = input_shape[-1]
        if lhs_contract != rhs_contract:
            raise ValueError(
                f"Layer {layer_name} ({op_type}): "
                f"Dimension mismatch - lhs contract dim {lhs_contract} "
                f"!= rhs contract dim {rhs_contract}"
            )

    return True


__all__ = [
    "OP_SHAPE_RULES",
    "OpSemanticsContext",
    "ShapeResolutionReport",
    "ShapeValidationError",
    "extract_int_tuple_from_input",
    "extract_reshape_target_shape",
    "first_resolved_output_shape",
    "get_feature_sizes",
    "infer_add_output_shape",
    "infer_batchnorm_output_shape",
    "infer_cast_output_shape",
    "infer_conv2d_output_shape",
    "infer_einsum_output_shape",
    "infer_expand_output_shape",
    "infer_flatten_output_shape",
    "infer_gemm_output_shape",
    "infer_global_avg_pool_output_shape",
    "infer_layer_shapes",
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
    "validate_inferred_shapes",
    "validate_model_shapes",
]
