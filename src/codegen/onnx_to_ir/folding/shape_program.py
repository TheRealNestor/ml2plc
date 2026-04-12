"""Shape-program folding helpers used during ONNX->IR extraction.

This module intentionally stays functional and small:
- one public entrypoint (`try_fold_enriched_shape_layer`)
- tiny per-op handlers
- explicit op->handler registry
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import onnx
from onnx import TensorProto


def _resolve_input_value(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
    index: int,
) -> Optional[np.ndarray]:
    inputs = enriched_layer.get("resolved_inputs", [])
    if index >= len(inputs):
        return None

    value = inputs[index].value
    if value is not None:
        return value

    return constant_values.get(inputs[index].name)


def _fold_shape(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    del constant_values
    inputs = enriched_layer.get("resolved_inputs", [])
    if not inputs or not inputs[0].shape:
        return None
    return np.array(inputs[0].shape, dtype=np.int64)


def _fold_gather(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    attrs = enriched_layer.get("attributes", {})
    data = _resolve_input_value(enriched_layer, constant_values, 0)
    indices = _resolve_input_value(enriched_layer, constant_values, 1)
    if data is None or indices is None:
        return None
    axis = int(attrs.get("axis", 0))
    return np.take(data, indices.astype(np.intp), axis=axis)


def _fold_reduce_prod(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    attrs = enriched_layer.get("attributes", {})
    data = _resolve_input_value(enriched_layer, constant_values, 0)
    if data is None:
        return None

    axes_input = _resolve_input_value(enriched_layer, constant_values, 1)
    if axes_input is not None:
        axes = tuple(int(a) for a in np.array(axes_input).flatten().tolist())
    else:
        axes = tuple(int(a) for a in attrs.get("axes", ()))

    keepdims = bool(attrs.get("keepdims", 1))
    return np.prod(data, axis=axes if axes else None, keepdims=keepdims)


def _fold_concat(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    attrs = enriched_layer.get("attributes", {})
    inputs = enriched_layer.get("resolved_inputs", [])
    vals = [
        _resolve_input_value(enriched_layer, constant_values, i)
        for i in range(len(inputs))
    ]
    if any(v is None for v in vals):
        return None
    axis = int(attrs.get("axis", 0))
    return np.concatenate([v for v in vals if v is not None], axis=axis)


def _fold_max(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    inputs = enriched_layer.get("resolved_inputs", [])
    vals = [
        _resolve_input_value(enriched_layer, constant_values, i)
        for i in range(len(inputs))
    ]
    if not vals or any(v is None for v in vals):
        return None

    out = vals[0]
    for v in vals[1:]:
        out = np.maximum(out, v)
    return out


def _fold_sub(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    a = _resolve_input_value(enriched_layer, constant_values, 0)
    b = _resolve_input_value(enriched_layer, constant_values, 1)
    if a is None or b is None:
        return None
    return np.subtract(a, b)


def _fold_mul(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    a = _resolve_input_value(enriched_layer, constant_values, 0)
    b = _resolve_input_value(enriched_layer, constant_values, 1)
    if a is None or b is None:
        return None
    return np.multiply(a, b)


def _fold_cast(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    attrs = enriched_layer.get("attributes", {})
    a = _resolve_input_value(enriched_layer, constant_values, 0)
    if a is None:
        return None
    to_type = int(attrs.get("to", TensorProto.FLOAT))
    np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(to_type, np.float32)
    return np.asarray(a).astype(np_dtype)


ShapeFoldHandler = Callable[[Dict, Dict[str, np.ndarray]], Optional[np.ndarray]]

_SHAPE_FOLD_HANDLERS: Dict[str, ShapeFoldHandler] = {
    "Shape": _fold_shape,
    "Gather": _fold_gather,
    "ReduceProd": _fold_reduce_prod,
    "Concat": _fold_concat,
    "Max": _fold_max,
    "Sub": _fold_sub,
    "Mul": _fold_mul,
    "Cast": _fold_cast,
}


def try_fold_enriched_shape_layer(
    enriched_layer: Dict,
    constant_values: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Try to fold a shape-subgraph layer using resolved/enriched inputs.

    Returns folded numpy value when successful; otherwise None.
    """
    op = enriched_layer.get("op_type")
    handler = _SHAPE_FOLD_HANDLERS.get(op)
    if handler is None:
        return None

    try:
        return handler(enriched_layer, constant_values)
    except Exception:
        return None
