"""Model-level shape validation and dynamic-dimension resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
import onnx

logger = logging.getLogger(__name__)


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
    """Validate and resolve dynamic dimensions before ONNX->IR extraction."""
    _run_onnx_shape_inference_inplace(model)

    dynamic_tensors = _find_dynamic_tensors(model)
    initial_dynamic_tensor_count = len(dynamic_tensors)

    if not dynamic_tensors:
        logger.info("✓ All tensor shapes are concrete (no dynamic dims found)")
        return ShapeResolutionReport(0, 0, False)

    static_hint_resolutions = _resolve_dims_from_provided_shapes(model, dynamic_tensors)
    modified_dims = _apply_resolved_dimensions(model, static_hint_resolutions)

    dynamic_tensors = _find_dynamic_tensors(model)
    batch_resolutions = _resolve_dynamic_batch_dims(dynamic_tensors)
    modified_dims += _apply_resolved_dimensions(model, batch_resolutions)

    dynamic_tensors = _find_dynamic_tensors(model)
    recurrent_resolutions = _resolve_recurrent_state_dims(model, dynamic_tensors)
    modified_dims += _apply_resolved_dimensions(model, recurrent_resolutions)

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

    first_tensor = list(remaining_dynamic.keys())[0]
    first_shape = remaining_dynamic[first_tensor]
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
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        model.CopyFrom(inferred)
    except Exception as e:
        logger.warning(f"ONNX shape inference failed during validation: {e}")


def _iter_tensor_value_infos(model: onnx.ModelProto):
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
    known_by_tensor_and_axis: Dict[str, Dict[int, int]] = {}
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
    resolved: Dict[str, Dict[int, int]] = {}
    for tensor_name, shape in dynamic_tensors.items():
        if len(shape) > 0 and shape[0] == 0:
            resolved.setdefault(tensor_name, {})[0] = 1
    return resolved


def _resolve_recurrent_state_dims(
    model: onnx.ModelProto,
    dynamic_tensors: Dict[str, Tuple[int, ...]],
) -> Dict[str, Dict[int, int]]:
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
        hidden_axis: Optional[int] = None
        if len(shape) > 0 and shape[-1] == 0:
            hidden_axis = len(shape) - 1

        tensor_resolution = resolved.setdefault(tensor_name, {})
        if hidden_axis is not None:
            tensor_resolution[hidden_axis] = hidden_size

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
    dynamic_tensors: Dict[str, Tuple[int, ...]] = {}

    for inp in model.graph.input:
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 0
            for dim in inp.type.tensor_type.shape.dim
        )
        if 0 in shape:
            dynamic_tensors[inp.name] = shape

    for output in model.graph.output:
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else 0
            for dim in output.type.tensor_type.shape.dim
        )
        if 0 in shape:
            dynamic_tensors[output.name] = shape

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
    """Resolve dynamic shape placeholders using configured fallback rules."""
    resolved: Dict[str, Tuple[int, ...]] = {}
    batch_resolutions = _resolve_dynamic_batch_dims(dynamic_tensors)
    for tensor_name, shape in dynamic_tensors.items():
        axis_map = batch_resolutions.get(tensor_name, {})
        if not axis_map:
            continue
        resolved_shape = tuple(axis_map.get(i, d) for i, d in enumerate(shape))
        resolved[tensor_name] = resolved_shape

    return resolved
