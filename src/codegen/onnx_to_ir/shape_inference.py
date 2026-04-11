"""Legacy module intentionally disabled.

Use `codegen.onnx_to_ir.shape` as the only supported shape API surface.
"""

raise ModuleNotFoundError(
    "`codegen.onnx_to_ir.shape_inference` has been removed. "
    "Import from `codegen.onnx_to_ir.shape` instead."
)

"""
Central shape inference for IR construction.
Infers shapes from operation semantics when ONNX shape info is incomplete.
I.e. infer output shape based on the operation (after tensors are resolved, during layer extraction).

Three-Layer Architecture:
  Layer 1: GROUND TRUTH EXTRACTION & VALIDATION
    - Runs BEFORE layer extraction
    - Identifies all dynamic dimensions (0 in shape)
    - Attempts to resolve using heuristics

  Layer 2: DIMENSION RESOLUTION (infer dynamic batch dims as 1)
    - Dynamic batch dimension (0 in position 0) → resolve to 1
    - Other dynamic dimensions → fail with actionable error

  Layer 3: OPERATION-SPECIFIC INFERENCE
    - Infer output given ALREADY RESOLVED inputs
    - No fallbacks, no heuristics - just pure inference
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import onnx
from onnx import numpy_helper

logger = logging.getLogger(__name__)


# ============================================================================
# LAYER 1: GROUND TRUTH EXTRACTION & VALIDATION
# ============================================================================


class ShapeValidationError(ValueError):
    """Raised when shapes cannot be resolved for PLC compilation."""

    def __init__(
        self,
        tensor_name: str,
        issue: str,
        shape: Tuple[int, ...],
        suggestions: Optional[List[str]] = None,
    ):
        self.tensor_name = tensor_name
        self.issue = issue
        self.shape = shape
        self.suggestions = suggestions or []

        msg = (
            f"\n╔════════════════════════════════════════════════════════════════╗\n"
            f"║         SHAPE VALIDATION FAILED - Cannot Compile to ST         ║\n"
            f"╚════════════════════════════════════════════════════════════════╝\n\n"
            f"Tensor: '{tensor_name}' has shape {shape}\n"
            f"Issue: {issue}\n"
        )

        if self.suggestions:
            msg += "\nSolutions:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"  ✓ Option {i}: {suggestion}\n"
        else:
            msg += (
                "\nNote: Structured Text requires all tensor dimensions to be "
                "statically known at compile time.\n"
            )

        super().__init__(msg)


@dataclass(frozen=True)
class ShapeResolutionReport:
    """Summary of model-level shape resolution done during validation."""

    dynamic_tensors_found: int
    resolved_dimensions: int
    modified: bool


def validate_model_shapes(model: onnx.ModelProto) -> ShapeResolutionReport:
    """
    PRE-PASS VALIDATOR: Ensure all tensors can be resolved to concrete shapes.

    This runs BEFORE any layer extraction and:
    1. Identifies all dynamic dimensions (0 in shape)
    2. Attempts to resolve them using heuristics
    3. Fails early with actionable errors if unresolvable

    For PLC code generation, all dimensions must be concrete at compile time.
    Dynamic batch dimensions (0 in position 0) are automatically resolved to 1.

    Returns:
        ShapeResolutionReport summarizing whether the graph was mutated.

    Raises:
        ShapeValidationError if unresolvable dynamic dimensions found
    """
    # First ask ONNX to populate as much shape information as possible.
    # This gives us the strongest "provided shape info first" baseline.
    _run_onnx_shape_inference_inplace(model)

    dynamic_tensors = _find_dynamic_tensors(model)
    initial_dynamic_tensor_count = len(dynamic_tensors)

    if not dynamic_tensors:
        logger.info("✓ All tensor shapes are concrete (no dynamic dims found)")
        return ShapeResolutionReport(
            dynamic_tensors_found=0,
            resolved_dimensions=0,
            modified=False,
        )

    # Step 1: Resolve unknown dims from already-provided static shape hints
    # on other occurrences of the same tensor (input/output/value_info).
    static_hint_resolutions = _resolve_dims_from_provided_shapes(model, dynamic_tensors)
    modified_dims = _apply_resolved_dimensions(model, static_hint_resolutions)

    # Step 2: Resolve the common PLC-safe case: dynamic batch dim -> 1.
    dynamic_tensors = _find_dynamic_tensors(model)
    batch_resolutions = _resolve_dynamic_batch_dims(dynamic_tensors)
    modified_dims += _apply_resolved_dimensions(model, batch_resolutions)

    # Step 3: Resolve recurrent state dimensions from op attributes.
    # ONNX RNN-family ops (LSTM/GRU/RNN) frequently carry hidden_size as an
    # attribute while some connected tensors may still appear dynamic in
    # value_info/input metadata (e.g., (1, 0)).
    dynamic_tensors = _find_dynamic_tensors(model)
    recurrent_resolutions = _resolve_recurrent_state_dims(model, dynamic_tensors)
    modified_dims += _apply_resolved_dimensions(model, recurrent_resolutions)

    # Re-run shape inference so upstream resolutions propagate through value_info.
    if modified_dims > 0:
        _run_onnx_shape_inference_inplace(model)

    remaining_dynamic = _find_dynamic_tensors(model)

    if not remaining_dynamic:
        logger.info(
            f"✓ Resolved {modified_dims} dynamic dimension(s) "
            f"(provided shapes + inferred batch size = 1)"
        )
        return ShapeResolutionReport(
            dynamic_tensors_found=initial_dynamic_tensor_count,
            resolved_dimensions=modified_dims,
            modified=modified_dims > 0,
        )

    # If we get here, there are unresolvable dimensions
    # Find the first one for error reporting
    unresolvable = remaining_dynamic
    first_tensor = list(unresolvable.keys())[0]
    first_shape = unresolvable[first_tensor]

    # Find positions of dynamic dimensions
    dyn_pos = [i for i, d in enumerate(first_shape) if d == 0]

    suggestions = [
        "Fix model to use concrete dimension:\n"
        "    - For batch dimension (position 0): add batch_size=1 to Input()\n"
        "    - For other dimensions: model accepts variable-length sequences\n"
        "    - For PLC compilation, all dims must be static at export time",
        "Provide static input/output signatures where possible so ONNX shape inference "
        "can propagate concrete dimensions through the graph",
        "Use onnx-simplifier to resolve shapes:\n"
        "    pip install onnx-simplifier\n"
        "    onnxsim model.onnx out.onnx --input-shape 'input:1,20,1'",
    ]

    raise ShapeValidationError(
        tensor_name=first_tensor,
        issue=f"Unresolvable dynamic dimension at position {dyn_pos[0]}: {first_shape}",
        shape=first_shape,
        suggestions=suggestions,
    )


def _run_onnx_shape_inference_inplace(model: onnx.ModelProto) -> None:
    """Run ONNX shape inference and copy results back into the same model object."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        model.CopyFrom(inferred)
    except Exception as e:
        logger.warning(f"ONNX shape inference failed during validation: {e}")


def _iter_tensor_value_infos(model: onnx.ModelProto):
    """Yield all ValueInfoProto-like entries carrying tensor shapes."""
    for inp in model.graph.input:
        yield inp
    for out in model.graph.output:
        yield out
    for vi in model.graph.value_info:
        yield vi


def _resolve_dims_from_provided_shapes(
    model: onnx.ModelProto,
    dynamic_tensors: Dict[str, Tuple[int, ...]],
) -> Dict[str, Dict[int, int]]:
    """Resolve dynamic dims using static hints already present in the model."""
    known_by_tensor_and_axis: Dict[str, Dict[int, int]] = {}

    # Collect known static dims for each tensor at each axis.
    for value in _iter_tensor_value_infos(model):
        dims = value.type.tensor_type.shape.dim
        axis_map = known_by_tensor_and_axis.setdefault(value.name, {})
        for idx, dim in enumerate(dims):
            if dim.dim_value > 0:
                axis_map[idx] = dim.dim_value

    resolved: Dict[str, Dict[int, int]] = {}
    for tensor_name, shape in dynamic_tensors.items():
        axis_map = known_by_tensor_and_axis.get(tensor_name, {})
        for idx, dim in enumerate(shape):
            if dim == 0 and idx in axis_map and axis_map[idx] > 0:
                resolved.setdefault(tensor_name, {})[idx] = axis_map[idx]

    return resolved


def _resolve_dynamic_batch_dims(
    dynamic_tensors: Dict[str, Tuple[int, ...]],
) -> Dict[str, Dict[int, int]]:
    """Resolve dynamic batch dimensions (axis 0) to 1."""
    resolved: Dict[str, Dict[int, int]] = {}

    for tensor_name, shape in dynamic_tensors.items():
        if len(shape) > 0 and shape[0] == 0:
            resolved.setdefault(tensor_name, {})[0] = 1

    return resolved


def _resolve_recurrent_state_dims(
    model: onnx.ModelProto,
    dynamic_tensors: Dict[str, Tuple[int, ...]],
) -> Dict[str, Dict[int, int]]:
    """
    Resolve unresolved state-vector dims from recurrent operator hidden_size.

    Applies when a tensor with dynamic dims is consumed only by ONNX recurrent
    ops (LSTM/GRU/RNN) that agree on a positive hidden_size attribute.
    """
    if not dynamic_tensors:
        return {}

    recurrent_ops = {"LSTM", "GRU", "RNN"}
    transparent_ops = {
        "Identity",
        "Transpose",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Concat",
        "Slice",
        "Gather",
        "Cast",
        "Expand",
        "Shape",
        "ConstantOfShape",
        "Tile",
    }
    consumers_by_tensor: Dict[str, List[onnx.NodeProto]] = {}

    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers_by_tensor.setdefault(inp, []).append(node)

    resolved: Dict[str, Dict[int, int]] = {}

    for tensor_name, shape in dynamic_tensors.items():
        zero_axes = [i for i, dim in enumerate(shape) if dim == 0]
        if not zero_axes:
            continue

        hidden_sizes = _find_reachable_recurrent_hidden_sizes(
            tensor_name,
            consumers_by_tensor,
            recurrent_ops=recurrent_ops,
            transparent_ops=transparent_ops,
        )

        if len(hidden_sizes) != 1:
            continue

        hidden_size = next(iter(hidden_sizes))

        # Prefer resolving the semantic hidden axis (usually the last axis).
        hidden_axis: Optional[int] = None
        if len(shape) > 0 and shape[-1] == 0:
            hidden_axis = len(shape) - 1

        tensor_resolution = resolved.setdefault(tensor_name, {})

        if hidden_axis is not None:
            tensor_resolution[hidden_axis] = hidden_size

        # Remaining dynamic axes in recurrent-context tensors are typically
        # sequence/batch placeholders. For PLC compilation we concretize them
        # to 1 (same policy as leading dynamic batch).
        for axis in zero_axes:
            if axis == hidden_axis:
                continue
            if axis in (0, 1):
                tensor_resolution[axis] = 1

        if not tensor_resolution:
            resolved.pop(tensor_name, None)

    return resolved


def _find_reachable_recurrent_hidden_sizes(
    tensor_name: str,
    consumers_by_tensor: Dict[str, List[onnx.NodeProto]],
    *,
    recurrent_ops: set,
    transparent_ops: set,
    max_depth: int = 8,
) -> set:
    """Find hidden_size values of recurrent ops reachable through transparent ops."""
    hidden_sizes = set()
    visited_tensors = {tensor_name}
    frontier = [(tensor_name, 0)]

    while frontier:
        current_tensor, depth = frontier.pop(0)
        consumers = consumers_by_tensor.get(current_tensor, [])

        for node in consumers:
            if node.op_type in recurrent_ops:
                hidden_size = next(
                    (
                        int(attr.i)
                        for attr in node.attribute
                        if attr.name == "hidden_size" and int(attr.i) > 0
                    ),
                    0,
                )
                if hidden_size > 0:
                    hidden_sizes.add(hidden_size)
                continue

            if node.op_type not in transparent_ops or depth >= max_depth:
                continue

            for out_name in node.output:
                if not out_name or out_name in visited_tensors:
                    continue
                visited_tensors.add(out_name)
                frontier.append((out_name, depth + 1))

    return hidden_sizes


def _apply_resolved_dimensions(
    model: onnx.ModelProto,
    resolutions: Dict[str, Dict[int, int]],
) -> int:
    """Apply resolved axis values to graph input/output/value_info tensors."""
    if not resolutions:
        return 0

    modifications = 0

    for value in _iter_tensor_value_infos(model):
        tensor_resolutions = resolutions.get(value.name)
        if not tensor_resolutions:
            continue

        dims = value.type.tensor_type.shape.dim
        for axis, target in tensor_resolutions.items():
            if axis >= len(dims) or target <= 0:
                continue

            dim = dims[axis]
            if dim.dim_value != target or dim.dim_param:
                dim.ClearField("dim_param")
                dim.dim_value = target
                modifications += 1

    return modifications


def _find_dynamic_tensors(model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    """Find all tensors with dynamic (0) dimensions in the ONNX model."""
    dynamic_tensors: Dict[str, Tuple[int, ...]] = {}

    # Check graph inputs
    for inp in model.graph.input:
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 0
            for dim in inp.type.tensor_type.shape.dim
        )
        if 0 in shape:
            dynamic_tensors[inp.name] = shape

    # Check graph outputs
    for output in model.graph.output:
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 0
            for dim in output.type.tensor_type.shape.dim
        )
        if 0 in shape:
            dynamic_tensors[output.name] = shape

    # Check intermediate tensors (skip weights/initializers)
    initializer_names = {init.name for init in model.graph.initializer}
    for vi in model.graph.value_info:
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 0
            for dim in vi.type.tensor_type.shape.dim
        )
        if 0 in shape and vi.name not in initializer_names:
            dynamic_tensors[vi.name] = shape

    return dynamic_tensors


def _attempt_resolve_dynamic_shapes(
    model: onnx.ModelProto, dynamic_tensors: Dict[str, Tuple[int, ...]]
) -> Dict[str, Tuple[int, ...]]:
    """
    Attempt to resolve dynamic dimensions.

    Strategy: Dynamic batch dimensions (0 in position 0) are resolved to 1.
    Other dynamic dimensions are considered unresolvable.

    Returns:
        Dict mapping tensor_name -> resolved_shape for successfully resolved tensors
    """
    # Backward-compatible shim for older callers; this module now uses
    # dimension-level resolution + in-place graph mutation.
    resolved: Dict[str, Tuple[int, ...]] = {}
    batch_resolutions = _resolve_dynamic_batch_dims(dynamic_tensors)
    for tensor_name, shape in dynamic_tensors.items():
        axis_map = batch_resolutions.get(tensor_name, {})
        if not axis_map:
            continue
        resolved_shape = tuple(axis_map.get(i, d) for i, d in enumerate(shape))
        resolved[tensor_name] = resolved_shape

    return resolved


def infer_matmul_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for MatMul operation.

    ONNX MatMul follows NumPy matmul semantics.
    This implementation supports vector/matrix/batched variants and returns
    a static output shape whenever both input shapes are static.

    For scalar outputs (e.g., (K,) @ (K,)), return (1,) so downstream PLC
    codegen does not treat it as an "empty" shape.
    """
    if not input_shape or not weight_shape:
        logger.warning(f"Invalid shapes for MatMul: {input_shape} @ {weight_shape}")
        return ()

    a = tuple(input_shape)
    b = tuple(weight_shape)

    if len(a) == 1 and len(b) == 1:
        # Dot product -> scalar (represented as one-element tensor in this IR)
        return (1,)

    if len(a) == 1 and len(b) >= 2:
        # (K,) @ (..., K, N) -> (..., N)
        if a[0] != b[-2]:
            logger.warning(f"MatMul mismatch: {a} @ {b}")
            return ()
        batch = b[:-2]
        return (*batch, b[-1]) if batch else (b[-1],)

    if len(a) >= 2 and len(b) == 1:
        # (..., M, K) @ (K,) -> (..., M)
        if a[-1] != b[0]:
            logger.warning(f"MatMul mismatch: {a} @ {b}")
            return ()
        out = a[:-1]
        return out if out else (1,)

    # General batched matrix multiply: (..., M, K) @ (..., K, N) -> (..., M, N)
    if a[-1] != b[-2]:
        logger.warning(f"MatMul mismatch: {a} @ {b}")
        return ()

    a_batch = a[:-2]
    b_batch = b[:-2]

    try:
        batch = np.broadcast_shapes(a_batch, b_batch)
    except ValueError:
        logger.warning(f"MatMul batch broadcast mismatch: {a_batch} vs {b_batch}")
        return ()

    return (*batch, a[-2], b[-1])


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


def infer_cast_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infer output shape for Cast operation.

    Cast preserves the tensor shape exactly (only changes dtype).

    Args:
        input_shape: Input tensor shape

    Returns:
        Same as input shape
    """
    return input_shape


def infer_unsqueeze_output_shape(
    input_shape: Tuple[int, ...], axes: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for Unsqueeze operation.

    Unsqueeze inserts dimensions of size 1 at specified axes.
    For PLC flattened data, element count remains unchanged.

    Args:
        input_shape: Input tensor shape
        axes: Axes where dimensions of size 1 are inserted

    Returns:
        Output shape with size-1 dimensions inserted
    """
    if not input_shape:
        return ()

    if not axes:
        return input_shape

    # For PLC code generation with flattened data, we conservatively keep
    # the total element count the same (no actual reshape happens in code generation).
    # Return input shape unchanged as the actual reshape isn't needed for flat buffers.
    return input_shape


def infer_slice_output_shape(
    input_shape: Tuple[int, ...],
    starts: Tuple[int, ...],
    ends: Tuple[int, ...],
    axes: Tuple[int, ...] = (),
    steps: Tuple[int, ...] = (),
) -> Tuple[int, ...]:
    """
    Infer output shape for Slice operation.

    For flattened PLC data, slice reduces the total element count.

    Args:
        input_shape: Input tensor shape
        starts: Start indices for each axis
        ends: End indices for each axis
        axes: Which axes to slice (default: all axes in order)
        steps: Step sizes for each axis (default: 1)

    Returns:
        Output shape after slicing
    """
    if not input_shape:
        return ()

    if not axes:
        # If no axes specified, apply to first len(starts) dimensions
        axes = tuple(range(len(starts)))
    if not steps:
        steps = tuple([1] * len(axes))

    output_shape = list(input_shape)
    for ax, start, end, step in zip(axes, starts, ends, steps):
        if 0 <= ax < len(output_shape):
            # For each sliced dimension, compute the output size
            dim_size = output_shape[ax]
            actual_start = (
                max(0, min(start, dim_size)) if start >= 0 else max(0, dim_size + start)
            )
            actual_end = (
                max(0, min(end, dim_size)) if end >= 0 else max(0, dim_size + end)
            )
            slice_size = max(0, (actual_end - actual_start + (step - 1)) // step)
            output_shape[ax] = slice_size

    return tuple(output_shape)


def infer_expand_output_shape(
    input_shape: Tuple[int, ...], target_shape: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Infer output shape for Expand operation.

    Expand broadcasts a tensor to a larger shape.

    Args:
        input_shape: Input tensor shape
        target_shape: Target shape to broadcast to

    Returns:
        Output shape (the target shape)
    """
    if target_shape and all(s > 0 for s in target_shape):
        return target_shape
    return input_shape


def infer_reduce_mean_output_shape(
    input_shape: Tuple[int, ...],
    axes: Tuple[int, ...],
    keepdims: bool,
) -> Tuple[int, ...]:
    """Infer output shape for ReduceMean with static axes."""
    if not input_shape:
        return ()

    rank = len(input_shape)
    if not axes:
        axes = tuple(range(rank))

    norm_axes = []
    for ax in axes:
        a = int(ax)
        if a < 0:
            a += rank
        if 0 <= a < rank:
            norm_axes.append(a)

    if keepdims:
        out = list(input_shape)
        for a in norm_axes:
            out[a] = 1
        return tuple(out)

    out = [d for i, d in enumerate(input_shape) if i not in set(norm_axes)]
    return tuple(out) if out else (1,)


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


def infer_einsum_output_shape(
    equation: str,
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Infer output shape for supported Einsum equations."""
    if equation != "abcd,cde->abe":
        return lhs_shape

    if len(rhs_shape) != 3:
        return lhs_shape

    c_dim, d_dim, e_dim = rhs_shape

    if len(lhs_shape) == 4 and lhs_shape[2] == c_dim and lhs_shape[3] == d_dim:
        return (lhs_shape[0], lhs_shape[1], e_dim)

    lhs_size = int(np.prod(lhs_shape)) if lhs_shape else 0
    contract = c_dim * d_dim
    if contract <= 0:
        return (1, 1, e_dim)

    ab = max(1, int(np.ceil(lhs_size / contract))) if lhs_size > 0 else 1
    return (1, ab, e_dim)


def _extract_reshape_target_shape(
    resolved_inputs: List[Any],
) -> Optional[Tuple[int, ...]]:
    """Extract target shape tuple for Reshape from constant shape tensor."""
    if len(resolved_inputs) <= 1:
        return None

    shape_tensor = resolved_inputs[1]
    if not shape_tensor.is_weight or shape_tensor.value is None:
        return None

    return tuple(int(d) for d in shape_tensor.value if int(d) != 0)


def _is_onnx_output_shape_consistent(
    op_type: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    resolved_inputs: List[Any],
    layer_dict: Dict[str, Any],
) -> bool:
    """Return True if ONNX-provided output shape is compatible with op semantics."""
    if not output_shape:
        return False

    try:
        if op_type == "Reshape":
            target_shape = _extract_reshape_target_shape(resolved_inputs)
            if target_shape is None:
                return True

            inferred = infer_reshape_output_shape(input_shape, target_shape)
            if not inferred:
                return False

            # Reshape must preserve element count.
            return int(np.prod(output_shape)) == int(np.prod(inferred))

        if op_type == "MatMul" and len(resolved_inputs) > 1:
            inferred = infer_matmul_output_shape(input_shape, resolved_inputs[1].shape)
            return inferred == output_shape

        if op_type in ["Gemm", "FusedGemm"] and len(resolved_inputs) > 1:
            attrs = layer_dict.get("attributes", {})
            transB = attrs.get("transB", 0) == 1
            inferred = infer_gemm_output_shape(
                input_shape, resolved_inputs[1].shape, transB
            )
            return inferred == output_shape
    except Exception:
        return False

    # For all other ops we trust ONNX shape info.
    return True


def _primary_input_shape(resolved_inputs: List[Any]) -> Tuple[int, ...]:
    """Get primary data input shape (skip weights when possible)."""
    data_input = next((inp for inp in resolved_inputs if not inp.is_weight), None)
    if data_input is None and resolved_inputs:
        data_input = resolved_inputs[0]
    return data_input.shape if data_input and data_input.shape else ()


def _first_resolved_output_shape(resolved_outputs: List[Any]) -> Tuple[int, ...]:
    """Get first ONNX-resolved output shape if present."""
    if resolved_outputs and resolved_outputs[0].shape:
        return tuple(resolved_outputs[0].shape)
    return ()


def _extract_int_tuple_from_input(
    resolved_inputs: List[Any],
    input_index: int,
) -> Tuple[int, ...]:
    """Extract tuple[int] from constant tensor input if available."""
    if len(resolved_inputs) <= input_index:
        return ()
    value = resolved_inputs[input_index].value
    if value is None:
        return ()
    return tuple(int(v) for v in value.flatten().tolist())


def _infer_output_shape_from_semantics(
    op_type: str,
    input_shape: Tuple[int, ...],
    resolved_inputs: List[Any],
    resolved_outputs: List[Any],
    attrs: Dict[str, Any],
) -> Tuple[int, ...]:
    """Infer output shape from op semantics with centralized dispatch logic."""
    passthrough_ops = {
        "Dropout",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "QuantizeLinear",
        "DequantizeLinear",
        "Cast",
        "Sqrt",
        "Reciprocal",
    }
    resolved_preferred_ops = {"Sub", "Mul", "Max", "Concat", "Gather", "Shape"}

    if op_type in passthrough_ops:
        return input_shape

    if op_type in resolved_preferred_ops:
        return _first_resolved_output_shape(resolved_outputs) or input_shape

    if op_type == "MatMul":
        if len(resolved_inputs) < 2:
            return input_shape
        return infer_matmul_output_shape(input_shape, resolved_inputs[1].shape)

    if op_type in ["Gemm", "FusedGemm"]:
        if len(resolved_inputs) < 2:
            return input_shape
        transB = attrs.get("transB", 0) == 1
        return infer_gemm_output_shape(input_shape, resolved_inputs[1].shape, transB)

    if op_type == "Add":
        if len(resolved_inputs) > 1:
            return infer_add_output_shape(input_shape, resolved_inputs[1].shape)
        return input_shape

    if op_type == "Reshape":
        target_shape = _extract_reshape_target_shape(resolved_inputs)
        return infer_reshape_output_shape(input_shape, target_shape)

    if op_type == "Conv":
        if len(resolved_inputs) < 2:
            return input_shape
        strides = tuple(attrs.get("strides", [1, 1]))
        pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(attrs.get("dilations", [1, 1]))
        return infer_conv2d_output_shape(
            input_shape,
            resolved_inputs[1].shape,
            strides,
            pads,
            dilations,
        )

    if op_type in ["MaxPool", "AveragePool"]:
        kernel_shape = tuple(attrs.get("kernel_shape", [2, 2]))
        strides = tuple(attrs.get("strides", [1, 1]))
        pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
        return infer_pool2d_output_shape(input_shape, kernel_shape, strides, pads)

    if op_type == "GlobalAveragePool":
        return infer_global_avg_pool_output_shape(input_shape)

    if op_type == "Flatten":
        axis = attrs.get("axis", 1)
        return infer_flatten_output_shape(input_shape, axis)

    if op_type == "Transpose":
        perm = tuple(attrs.get("perm", ()))
        return infer_transpose_output_shape(input_shape, perm)

    if op_type == "BatchNormalization":
        return infer_batchnorm_output_shape(input_shape)

    if op_type == "Squeeze":
        axes = tuple(attrs.get("axes", ()))
        if not axes and len(resolved_inputs) > 1 and resolved_inputs[1].is_weight:
            axes_val = resolved_inputs[1].value
            if axes_val is not None:
                axes = tuple(int(a) for a in axes_val)
        if axes and any(a > 0 for a in axes):
            axes = tuple(a - 1 for a in axes if a != 0)
        return infer_squeeze_output_shape(input_shape, axes)

    if op_type == "Unsqueeze":
        axes = _extract_int_tuple_from_input(resolved_inputs, 1)
        if not axes:
            axes = tuple(attrs.get("axes", ()))
        return infer_unsqueeze_output_shape(input_shape, axes)

    if op_type == "Slice":
        starts = _extract_int_tuple_from_input(resolved_inputs, 1)
        ends = _extract_int_tuple_from_input(resolved_inputs, 2)
        axes = _extract_int_tuple_from_input(resolved_inputs, 3)
        steps = _extract_int_tuple_from_input(resolved_inputs, 4)
        return infer_slice_output_shape(input_shape, starts, ends, axes, steps)

    if op_type == "Expand":
        target_shape = _extract_int_tuple_from_input(resolved_inputs, 1)
        return infer_expand_output_shape(input_shape, target_shape or None)

    if op_type in {"ReduceMean", "ReduceProd"}:
        axes = tuple(int(a) for a in attrs.get("axes", ()))
        keepdims = bool(attrs.get("keepdims", 1))
        return infer_reduce_mean_output_shape(input_shape, axes, keepdims)

    if op_type == "Einsum":
        equation = str(attrs.get("equation", ""))
        rhs_shape = (
            tuple(resolved_inputs[1].shape)
            if len(resolved_inputs) > 1 and resolved_inputs[1].shape
            else ()
        )
        return infer_einsum_output_shape(equation, input_shape, rhs_shape)

    logger.warning(f"No shape inference for op_type '{op_type}', using input shape")
    return input_shape


def infer_layer_shapes(
    layer_dict: Dict[str, Any],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Infer input and output shapes for a layer based on operation type.

    PRECONDITION: All input tensors must have concrete shapes (no 0 dimensions).
    If this precondition is violated, that's a bug in the shape validation pass.

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
    attrs = layer_dict.get("attributes", {})

    # ✅ SAFETY CHECK: Ensure no dynamic dimensions slip through
    for inp in resolved_inputs:
        if inp.shape and 0 in inp.shape:
            raise RuntimeError(
                f"BUG in shape validation: infer_layer_shapes received input "
                f"'{inp.name}' with dynamic dimension {inp.shape}. "
                f"This should have been caught by validate_model_shapes()."
            )

    input_shape = _primary_input_shape(resolved_inputs)

    # Try to get shape from ONNX tensor_info first
    output_tensor_info_shape = _first_resolved_output_shape(resolved_outputs)

    # If output shape is valid and semantically consistent, use it.
    if output_tensor_info_shape and _is_onnx_output_shape_consistent(
        op_type,
        input_shape,
        output_tensor_info_shape,
        resolved_inputs,
        layer_dict,
    ):
        logger.debug(f"{op_type}: Using ONNX output shape {output_tensor_info_shape}")
        return input_shape, output_tensor_info_shape

    if output_tensor_info_shape:
        logger.debug(
            f"{op_type}: Ignoring inconsistent ONNX output shape "
            f"{output_tensor_info_shape}, falling back to op inference"
        )

    # Otherwise, infer from operation semantics
    logger.debug(f"{op_type}: Inferring output shape (ONNX shape empty)")
    output_shape = _infer_output_shape_from_semantics(
        op_type,
        input_shape,
        resolved_inputs,
        resolved_outputs,
        attrs,
    )

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
    Get input and output feature sizes as total flattened element counts.

    For PLC code generation, we work with flattened 1D buffers, so we need
    the total number of elements in each tensor.

    Args:
        input_shape: Input tensor shape
        output_shape: Output tensor shape

    Returns:
        (input_size, output_size) - total flattened element counts
    """
    # Compute total flattened size as product of all dimensions
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
    if op_type in ["Gemm", "FusedGemm"] and weight_shape:
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
