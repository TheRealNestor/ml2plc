"""Constant-graph folding utilities for ONNX->IR conversion.

This module handles compile-time evaluation of ONNX nodes when all required
inputs are constant (or when Shape can be resolved from static tensor shapes).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
import onnx

logger = logging.getLogger(__name__)


# Operators that can be constant-folded when all inputs are known at compile time
FOLDABLE_OPS = {
    "Shape",
    "Cast",
    "Slice",
    "Concat",
    "Expand",
    "Unsqueeze",
    "Gather",
    "Reshape",
    "Squeeze",
    "Transpose",
    "ReduceMean",
    "ReduceProd",
    "Max",
    "Mul",
    "Sub",
    "Neg",
    "MatMul",
}


def _evaluate_constant_op(
    op: str, node, inputs: List[Optional[np.ndarray]]
) -> np.ndarray:
    """Evaluate a single ONNX op on constant numpy inputs."""

    def _get(i):
        return inputs[i] if i < len(inputs) else None

    if op == "Shape":
        return np.array(inputs[0].shape, dtype=np.int64)

    if op == "Cast":
        to_type = next(a.i for a in node.attribute if a.name == "to")
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(to_type, np.float32)
        return inputs[0].astype(np_dtype)

    if op == "Slice":
        data = inputs[0]
        starts = _get(1).flatten().tolist() if _get(1) is not None else [0]
        ends = _get(2).flatten().tolist() if _get(2) is not None else [data.shape[0]]
        axes = (
            _get(3).flatten().tolist()
            if _get(3) is not None
            else list(range(len(starts)))
        )
        steps = _get(4).flatten().tolist() if _get(4) is not None else [1] * len(starts)
        slices = [slice(None)] * data.ndim
        for a, s, e, st in zip(axes, starts, ends, steps):
            dim = data.shape[a]
            s = min(max(s + dim if s < 0 else s, 0), dim)
            e = min(max(e + dim if e < 0 else e, 0), dim)
            slices[a] = slice(int(s), int(e), int(st))
        return data[tuple(slices)]

    if op == "Concat":
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        real_inputs = [x for x in inputs if x is not None]
        return np.concatenate(real_inputs, axis=axis)

    if op == "Unsqueeze":
        data = inputs[0]
        axes_input = _get(1)
        if axes_input is not None:
            axes = sorted(axes_input.flatten().tolist())
        else:
            axes = sorted(
                next((list(a.ints) for a in node.attribute if a.name == "axes"), [])
            )
        result = data
        for ax in axes:
            result = np.expand_dims(result, axis=int(ax))
        return result

    if op == "Squeeze":
        data = inputs[0]
        axes_input = _get(1)
        if axes_input is not None:
            axes = tuple(sorted(axes_input.flatten().tolist(), reverse=True))
        else:
            axes = tuple(
                sorted(
                    next(
                        (list(a.ints) for a in node.attribute if a.name == "axes"), []
                    ),
                    reverse=True,
                )
            )
        if axes:
            result = data
            for ax in axes:
                result = np.squeeze(result, axis=int(ax))
            return result
        return np.squeeze(data)

    if op == "Expand":
        target_shape = inputs[1].flatten().tolist()
        return np.broadcast_to(inputs[0], [int(s) for s in target_shape]).copy()

    if op == "Gather":
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        return np.take(inputs[0], inputs[1].astype(np.intp), axis=axis)

    if op == "Reshape":
        shape = inputs[1].flatten().tolist()
        return inputs[0].reshape([int(s) for s in shape])

    if op == "Transpose":
        data = inputs[0]
        perm = tuple(
            next((list(a.ints) for a in node.attribute if a.name == "perm"), [])
        )
        if not perm:
            return np.transpose(data)
        return np.transpose(data, axes=perm)

    if op in {"ReduceMean", "ReduceProd"}:
        data = inputs[0]
        axes_input = _get(1)
        if axes_input is not None:
            axes = tuple(int(a) for a in np.array(axes_input).flatten().tolist())
        else:
            axes = tuple(
                int(a)
                for a in next(
                    (list(attr.ints) for attr in node.attribute if attr.name == "axes"),
                    [],
                )
            )
        keepdims = bool(next((a.i for a in node.attribute if a.name == "keepdims"), 1))
        if op == "ReduceMean":
            return np.mean(data, axis=axes if axes else None, keepdims=keepdims)
        return np.prod(data, axis=axes if axes else None, keepdims=keepdims)

    if op == "Max":
        real_inputs = [x for x in inputs if x is not None]
        result = real_inputs[0]
        for nxt in real_inputs[1:]:
            result = np.maximum(result, nxt)
        return result

    if op == "Mul":
        return np.multiply(inputs[0], inputs[1])

    if op == "Sub":
        return np.subtract(inputs[0], inputs[1])

    if op == "Neg":
        return np.negative(inputs[0])

    if op == "MatMul":
        return np.matmul(inputs[0], inputs[1])

    raise ValueError(f"Unsupported constant-fold op: {op}")


def try_constant_fold_node(
    node,
    constant_values: Dict[str, np.ndarray],
    static_input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> bool:
    """Try to constant-fold a node whose required inputs are known."""
    op = node.op_type
    if op not in FOLDABLE_OPS:
        return False

    if op == "Shape" and static_input_shapes is not None:
        input_name = node.input[0] if node.input else None
        if input_name:
            shape = None
            if input_name in constant_values:
                shape = constant_values[input_name].shape
            elif input_name in static_input_shapes:
                shape = static_input_shapes[input_name]

            if shape is not None:
                result = np.array(shape, dtype=np.int64)
                for out in node.output:
                    if out:
                        constant_values[out] = result
                logger.debug(
                    f"Constant-folded Shape node '{node.name}' via static shape -> {result}"
                )
                return True

    inputs = [constant_values.get(inp) if inp else None for inp in node.input]
    if any(v is None and inp != "" for inp, v in zip(node.input, inputs)):
        return False

    try:
        result = _evaluate_constant_op(op, node, inputs)
    except Exception as e:
        logger.debug(f"Could not constant-fold {op} '{node.name}': {e}")
        return False

    for out in node.output:
        if out:
            constant_values[out] = np.array(result)

    logger.debug(
        f"Constant-folded {op} node '{node.name}' -> "
        f"shape {np.array(result).shape}, dtype {np.array(result).dtype}"
    )
    return True
