"""
Central shape inference for IR construction.
Infers shapes from operation semantics when ONNX shape info is incomplete.
I.e. infer output shape based on the operation (after tensors are resolved, during layer extraction).
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def infer_matmul_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for MatMul operation.

    MatMul: (M, K) @ (K, N) -> (M, N)
    For batched: (..., M, K) @ (K, N) -> (..., M, N)

    For our PLC use case, we flatten batches, so:
    (K,) @ (K, N) -> (N,)
    """
    if not weight_shape or len(weight_shape) < 2:
        logger.warning(f"Invalid weight shape for MatMul: {weight_shape}")
        return ()

    # Weight is always 2D: (input_features, output_features)
    output_features = weight_shape[1]

    if not input_shape:
        return (output_features,)

    # For PLCs, we typically work with flattened 1D inputs
    # Keep batch dimensions if present, replace last dim with output_features
    if len(input_shape) > 1:
        return (*input_shape[:-1], output_features)
    else:
        return (output_features,)


# TODO: input shape is only needed for batch dim? consider removing this
def infer_gemm_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...], transB: bool = False
) -> Tuple[int, ...]:
    """
    Infer output shape for Gemm operation.

    Gemm: Y = alpha * A @ B^T + beta * C  (if transB=True)
          Y = alpha * A @ B + beta * C     (if transB=False)

    Args:
        input_shape: Shape of input A (M, K)
        weight_shape: Shape of weight B (K, N)
        transB: Whether B is transposed
    """
    if not weight_shape or len(weight_shape) < 2:
        logger.warning(f"Invalid weight shape for Gemm: {weight_shape}")
        return ()

    # Determine output features based on transB
    if transB:
        # B is (output_features, input_features), transposed becomes (input_features, output_features)
        output_features = weight_shape[0]
    else:
        # B is (input_features, output_features)
        output_features = weight_shape[1]

    # Gemm typically produces 1D output (batch dimension removed or kept as 1)
    return (output_features,)


def infer_add_output_shape(
    input_shape: Tuple[int, ...], bias_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for Add operation.

    Add with broadcasting: typically input + bias where bias is 1D.
    """
    # Add preserves the larger shape (broadcasting rules)
    if not input_shape:
        return bias_shape
    if not bias_shape:
        return input_shape

    # In most cases, bias is 1D and broadcasts to input shape
    return input_shape


def infer_conv2d_output_shape(
    input_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    strides: Tuple[int, int] = (1, 1),
    pads: Tuple[int, ...] = (0, 0, 0, 0),
    dilations: Tuple[int, int] = (1, 1),
) -> Tuple[int, ...]:
    """
    Infer output shape for Conv2D operation.

    Input:  (C_in, H, W)  or (N, C_in, H, W)
    Weight: (C_out, C_in/groups, kH, kW)
    Output: (C_out, H_out, W_out)

    H_out = (H + pad_top + pad_bottom - dilation_h * (kH - 1) - 1) / stride_h + 1
    """
    if not input_shape or len(input_shape) < 3:
        logger.warning(f"Invalid input shape for Conv2D: {input_shape}")
        return ()

    if not weight_shape or len(weight_shape) != 4:
        logger.warning(f"Invalid weight shape for Conv2D: {weight_shape}")
        return ()

    h_in, w_in = input_shape[-2], input_shape[-1]
    out_channels = weight_shape[0]
    kH, kW = weight_shape[2], weight_shape[3]

    pad_top, pad_left, pad_bottom, pad_right = pads[0], pads[1], pads[2], pads[3]

    h_out = (h_in + pad_top + pad_bottom - dilations[0] * (kH - 1) - 1) // strides[
        0
    ] + 1
    w_out = (w_in + pad_left + pad_right - dilations[1] * (kW - 1) - 1) // strides[
        1
    ] + 1

    return (out_channels, h_out, w_out)


def infer_pool2d_output_shape(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    pads: Tuple[int, ...] = (0, 0, 0, 0),
) -> Tuple[int, ...]:
    """
    Infer output shape for 2D pooling (MaxPool / AveragePool).

    Input:  (C, H, W) or (N, C, H, W)
    Output: (C, H_out, W_out)   — channels are preserved
    """
    if not input_shape or len(input_shape) < 3:
        logger.warning(f"Invalid input shape for Pool2D: {input_shape}")
        return ()

    channels = input_shape[-3]
    h_in, w_in = input_shape[-2], input_shape[-1]
    kH, kW = kernel_shape

    pad_top, pad_left, pad_bottom, pad_right = pads[0], pads[1], pads[2], pads[3]

    h_out = (h_in + pad_top + pad_bottom - kH) // strides[0] + 1
    w_out = (w_in + pad_left + pad_right - kW) // strides[1] + 1

    return (channels, h_out, w_out)


def infer_global_avg_pool_output_shape(
    input_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Infer output shape for GlobalAveragePool.

    Input:  (C, H, W) or (N, C, H, W)
    Output: (C, 1, 1)  — spatial dims collapsed to 1
    """
    if not input_shape or len(input_shape) < 3:
        logger.warning(f"Invalid input shape for GlobalAveragePool: {input_shape}")
        return ()

    channels = input_shape[-3]
    return (channels, 1, 1)


def infer_flatten_output_shape(
    input_shape: Tuple[int, ...], axis: int = 1
) -> Tuple[int, ...]:
    """
    Infer output shape for Flatten operation.

    Flattens all dims from *axis* onward into a single dimension.
    For PLC inference the batch dim (dim 0) is typically stripped,
    so axis=1 means "flatten everything after batch" → single dim.
    """
    if not input_shape:
        return ()

    # In ONNX, axis can reference the batch dim.
    # For PLC we store without batch, so if shapes already lack batch
    # we treat axis=1 as "flatten all".
    total = int(np.prod(input_shape))
    return (total,)


def infer_transpose_output_shape(
    input_shape: Tuple[int, ...], perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for Transpose operation.

    Applies the given permutation to the input shape dimensions.
    For PLC shapes that have the batch dimension stripped, the
    permutation indices are adjusted accordingly.
    """
    if not input_shape:
        return ()

    if not perm:
        # Default: reverse all dimensions
        return tuple(reversed(input_shape))

    # The ONNX perm may include the batch dim (dim 0).
    # Our IR shapes typically have the batch dim already stripped.
    # If perm has more entries than the shape, strip the batch entry.
    if len(perm) == len(input_shape) + 1:
        # Drop dim-0 from perm and shift remaining indices down by 1
        perm = tuple(p - 1 for p in perm if p != 0)

    if len(perm) != len(input_shape):
        logger.warning(
            f"Transpose perm length {len(perm)} != input shape length "
            f"{len(input_shape)}, returning input shape unchanged"
        )
        return input_shape

    return tuple(input_shape[p] for p in perm)


def infer_batchnorm_output_shape(
    input_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Infer output shape for BatchNormalization.

    BatchNorm preserves input shape exactly (per-channel affine transform).
    Input/Output: (C, H, W)  or (C,) for 1-D
    """
    return input_shape


def infer_squeeze_output_shape(
    input_shape: Tuple[int, ...], axes: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for Squeeze operation.

    Removes dimensions of size 1 at the given axes.
    E.g. input (8, 1, 1) with axes=(1, 2) → (8,)

    If axes is empty, all dims of size 1 are removed.
    """
    if not input_shape:
        return ()

    if not axes:
        # Squeeze all dims of size 1
        return tuple(d for d in input_shape if d != 1) or (1,)

    # Remove specified axes (iterate in reverse to keep indices stable)
    result = list(input_shape)
    for ax in sorted(axes, reverse=True):
        if 0 <= ax < len(result) and result[ax] == 1:
            result.pop(ax)
        else:
            logger.warning(
                f"Squeeze axis {ax} is out of range or dim != 1 "
                f"(shape={input_shape}), skipping"
            )

    return tuple(result) if result else (1,)


def infer_reshape_output_shape(
    input_shape: Tuple[int, ...], target_shape: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Infer output shape for Reshape operation.

    Args:
        input_shape: Input tensor shape
        target_shape: Target shape (may contain -1 for inferred dimension)

    Returns:
        Resolved output shape
    """
    if not target_shape:
        # No target shape provided - flatten to 1D
        if input_shape:
            total_size = int(np.prod(input_shape))
            return (total_size,)
        return ()

    # Handle -1 in target shape (infer dimension)
    if -1 in target_shape:
        input_size = int(np.prod(input_shape)) if input_shape else 0
        known_dims = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known_dims)) if known_dims else 1

        if known_prod == 0:
            logger.warning(f"Invalid target shape {target_shape}")
            return ()

        inferred_dim = input_size // known_prod
        resolved_shape = tuple(inferred_dim if d == -1 else d for d in target_shape)
        return resolved_shape

    # All dimensions are specified
    return target_shape


def infer_layer_shapes(
    layer_dict: Dict[str, Any],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Infer input and output shapes for a layer based on operation type.

    This is the main entry point for shape inference. It tries to use ONNX
    tensor_info shapes first, and falls back to operation-specific inference.

    Args:
        layer_dict: Enriched layer dict with 'resolved_inputs' and 'resolved_outputs'

    Returns:
        (input_shape, output_shape) - tuples of integers only (no symbolic dims)
    """
    op_type = layer_dict["op_type"]
    resolved_inputs = layer_dict["resolved_inputs"]
    resolved_outputs = layer_dict["resolved_outputs"]

    # Get input shape (first data tensor, skip weights)
    data_input = None
    for inp in resolved_inputs:
        if not inp.is_weight:
            data_input = inp
            break

    if data_input is None:
        data_input = resolved_inputs[0]

    input_shape = data_input.shape if data_input.shape else ()

    # Try to get shape from ONNX tensor_info first
    output_tensor_info_shape = resolved_outputs[0].shape if resolved_outputs else ()

    # If output shape is valid (has at least one dimension), use it
    if output_tensor_info_shape:
        logger.debug(f"{op_type}: Using ONNX output shape {output_tensor_info_shape}")
        return input_shape, output_tensor_info_shape

    # Otherwise, infer from operation semantics
    logger.debug(f"{op_type}: Inferring output shape (ONNX shape empty)")

    if op_type == "MatMul":
        weight_tensor = resolved_inputs[1]
        output_shape = infer_matmul_output_shape(input_shape, weight_tensor.shape)

    elif op_type in ["Gemm", "FusedGemm"]:
        weight_tensor = resolved_inputs[1]
        attrs = layer_dict.get("attributes", {})
        transB = attrs.get("transB", 0) == 1
        output_shape = infer_gemm_output_shape(input_shape, weight_tensor.shape, transB)
    elif op_type == "Dropout":
        output_shape = input_shape

    elif op_type in ["Relu", "Sigmoid", "Tanh"]:
        output_shape = input_shape

    elif op_type == "Softmax":
        output_shape = input_shape

    elif op_type == "Add":
        # Check if second input is bias
        if len(resolved_inputs) > 1:
            bias_tensor = resolved_inputs[1]
            output_shape = infer_add_output_shape(input_shape, bias_tensor.shape)
        else:
            output_shape = input_shape

    elif op_type == "Reshape":
        # Try to get target shape from second input (shape tensor)
        target_shape = None
        if len(resolved_inputs) > 1 and resolved_inputs[1].is_weight:
            shape_array = resolved_inputs[1].value
            if shape_array is not None:
                # Filter out 0 and keep positive dimensions, convert -1
                target_shape = tuple(int(d) for d in shape_array if d != 0)
        output_shape = infer_reshape_output_shape(input_shape, target_shape)

    elif op_type == "QuantizeLinear":
        output_shape = input_shape

    elif op_type == "DequantizeLinear":
        output_shape = input_shape

    elif op_type == "Conv":
        weight_tensor = resolved_inputs[1]
        attrs = layer_dict.get("attributes", {})
        strides = tuple(attrs.get("strides", [1, 1]))
        pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(attrs.get("dilations", [1, 1]))
        output_shape = infer_conv2d_output_shape(
            input_shape, weight_tensor.shape, strides, pads, dilations
        )

    elif op_type in ["MaxPool", "AveragePool"]:
        attrs = layer_dict.get("attributes", {})
        kernel_shape = tuple(attrs.get("kernel_shape", [2, 2]))
        strides = tuple(attrs.get("strides", [1, 1]))
        pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
        output_shape = infer_pool2d_output_shape(
            input_shape, kernel_shape, strides, pads
        )

    elif op_type == "GlobalAveragePool":
        output_shape = infer_global_avg_pool_output_shape(input_shape)

    elif op_type == "Flatten":
        attrs = layer_dict.get("attributes", {})
        axis = attrs.get("axis", 1)
        output_shape = infer_flatten_output_shape(input_shape, axis)

    elif op_type == "Transpose":
        attrs = layer_dict.get("attributes", {})
        perm = tuple(attrs.get("perm", ()))
        output_shape = infer_transpose_output_shape(input_shape, perm)

    elif op_type == "BatchNormalization":
        output_shape = infer_batchnorm_output_shape(input_shape)

    elif op_type == "Squeeze":
        # Axes can come from attributes (opset < 13) or from a constant input
        attrs = layer_dict.get("attributes", {})
        axes = tuple(attrs.get("axes", ()))
        if not axes and len(resolved_inputs) > 1 and resolved_inputs[1].is_weight:
            axes_val = resolved_inputs[1].value
            if axes_val is not None:
                axes = tuple(int(a) for a in axes_val)
        # Adjust for batch-dim-stripped shapes
        if axes and any(a > 0 for a in axes):
            axes = tuple(a - 1 for a in axes if a != 0)
        output_shape = infer_squeeze_output_shape(input_shape, axes)

    if op_type == "Cast":
        # Same shape as input
        return _passthrough_shape(layer_dict)
    if op_type == "Slice":
        # Use resolved output shape if available
        return _use_resolved_output_shape(layer_dict)
    if op_type == "Concat":
        return _use_resolved_output_shape(layer_dict)

    if op_type in ("Unsqueeze", "Expand", "Gather", "Shape"):
        return _use_resolved_output_shape(layer_dict)

    else:
        logger.warning(f"No shape inference for op_type '{op_type}', using input shape")
        output_shape = input_shape

    logger.debug(f"{op_type}: Inferred {input_shape} -> {output_shape}")
    return input_shape, output_shape


def _passthrough_shape(enriched_layer: Dict):
    """Output shape = input shape."""
    resolved_in = enriched_layer.get("resolved_inputs", [])
    if resolved_in and resolved_in[0].shape:
        shape = tuple(resolved_in[0].shape)
        return shape, shape
    return None, None


def _use_resolved_output_shape(enriched_layer: Dict):
    """Use the shape from resolved_outputs (from ONNX shape inference)."""
    resolved_in = enriched_layer.get("resolved_inputs", [])
    resolved_out = enriched_layer.get("resolved_outputs", [])
    in_shape = (
        tuple(resolved_in[0].shape) if resolved_in and resolved_in[0].shape else None
    )
    out_shape = (
        tuple(resolved_out[0].shape) if resolved_out and resolved_out[0].shape else None
    )
    return in_shape, out_shape


def get_feature_sizes(
    input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
) -> Tuple[int, int]:
    """
    Get input and output feature sizes (ignoring batch dimensions).

    For PLC code generation, we typically work with flattened 1D arrays,
    so we take the last dimension as the feature size, or the total size
    if the shape is 1D.

    Args:
        input_shape: Input tensor shape
        output_shape: Output tensor shape

    Returns:
        (input_size, output_size) - number of features/elements
    """
    # For 1D shapes, use the dimension directly
    # For multi-dimensional, use the last dimension (feature dimension)
    input_size = input_shape[-1] if input_shape else 0
    output_size = output_shape[-1] if output_shape else 0

    return input_size, output_size


def validate_inferred_shapes(
    layer_name: str,
    op_type: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    weight_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Validate that inferred shapes are consistent with operation semantics.

    Args:
        layer_name: Name of the layer (for logging)
        op_type: Operation type
        input_shape: Inferred input shape
        output_shape: Inferred output shape
        weight_shape: Weight shape (if applicable)

    Returns:
        True if shapes are valid, raises ValueError otherwise
    """
    if not output_shape:
        raise ValueError(f"Layer {layer_name} ({op_type}): Output shape is empty")

    if not input_shape:
        logger.warning(f"Layer {layer_name} ({op_type}): Input shape is empty")

    # Operation-specific validation
    if op_type in ["MatMul", "Gemm", "FusedGemm"] and weight_shape:
        if len(weight_shape) != 2:
            raise ValueError(
                f"Layer {layer_name} ({op_type}): "
                f"Weight must be 2D, got {weight_shape}"
            )

        # Check dimension compatibility
        if input_shape and weight_shape:
            input_features = input_shape[-1]
            weight_input_features = weight_shape[0]

            if input_features != weight_input_features:
                raise ValueError(
                    f"Layer {layer_name} ({op_type}): "
                    f"Dimension mismatch - input features {input_features} "
                    f"!= weight input features {weight_input_features}"
                )

    return True
