"""Shape inference engine orchestration using rule-registry dispatch."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import logging
import numpy as np

from .rules import (
    OP_SHAPE_RULES,
    extract_reshape_target_shape,
    first_resolved_output_shape,
    infer_output_shape_from_semantics,
)

logger = logging.getLogger(__name__)


def _primary_input_shape(resolved_inputs: List[Any]) -> Tuple[int, ...]:
    data_input = next((inp for inp in resolved_inputs if not inp.is_weight), None)
    if data_input is None and resolved_inputs:
        data_input = resolved_inputs[0]
    return data_input.shape if data_input and data_input.shape else ()


def _is_onnx_output_shape_consistent(
    op_type: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    resolved_inputs: List[Any],
    resolved_outputs: List[Any],
    attrs: Dict[str, Any],
    layer_dict: Dict[str, Any],
) -> bool:
    """Validate ONNX output shape using the same registry rule path as inference."""
    if not output_shape:
        return False

    # Preserve historical behavior: if Reshape has no concrete target yet, trust ONNX.
    if op_type == "Reshape" and extract_reshape_target_shape(resolved_inputs) is None:
        return True

    if op_type not in OP_SHAPE_RULES:
        return True

    try:
        inferred = infer_output_shape_from_semantics(
            op_type,
            input_shape,
            resolved_inputs,
            resolved_outputs,
            attrs,
            layer_dict,
        )
    except Exception:
        return False

    if op_type == "Reshape":
        # Reshape equivalence is primarily element-count preserving.
        try:
            return int(np.prod(output_shape)) == int(np.prod(inferred))
        except Exception:
            return False

    return inferred == output_shape


def infer_layer_shapes(
    layer_dict: Dict[str, Any],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Infer input/output layer shapes from ONNX info + registered op rules."""
    op_type = layer_dict["op_type"]
    resolved_inputs = layer_dict["resolved_inputs"]
    resolved_outputs = layer_dict["resolved_outputs"]
    attrs = layer_dict.get("attributes", {})

    for inp in resolved_inputs:
        if inp.shape and 0 in inp.shape:
            raise RuntimeError(
                f"BUG in shape validation: infer_layer_shapes received input "
                f"'{inp.name}' with dynamic dimension {inp.shape}. "
                f"This should have been caught by validate_model_shapes()."
            )

    input_shape = _primary_input_shape(resolved_inputs)
    output_tensor_info_shape = first_resolved_output_shape(resolved_outputs)

    if output_tensor_info_shape and _is_onnx_output_shape_consistent(
        op_type,
        input_shape,
        output_tensor_info_shape,
        resolved_inputs,
        resolved_outputs,
        attrs,
        layer_dict,
    ):
        logger.debug(f"{op_type}: Using ONNX output shape {output_tensor_info_shape}")
        return input_shape, output_tensor_info_shape

    if output_tensor_info_shape:
        logger.debug(
            f"{op_type}: Ignoring inconsistent ONNX output shape "
            f"{output_tensor_info_shape}, falling back to op inference"
        )

    logger.debug(f"{op_type}: Inferring output shape (ONNX shape empty)")
    output_shape = infer_output_shape_from_semantics(
        op_type,
        input_shape,
        resolved_inputs,
        resolved_outputs,
        attrs,
        layer_dict,
    )

    logger.debug(f"{op_type}: Inferred {input_shape} -> {output_shape}")
    return input_shape, output_shape
