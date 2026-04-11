"""Einsum lowering utilities.

This module rewrites supported ONNX Einsum nodes into existing core operators
(Reshape + MatMul) so the downstream extractor/codegen stack can remain simple.

Design goals:
- Keep lowering explicit and pattern-driven (easy to read/extend).
- Reuse existing IR ops instead of introducing ad-hoc runtime kernels.
- Preserve graph output tensor names to avoid downstream rewiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..onnx_model import ONNXModel


@dataclass(frozen=True)
class EinsumLoweringReport:
    """Summary of lowering work done for logging/diagnostics."""

    lowered_count: int = 0
    skipped_count: int = 0


def _shape_from_analyzer(
    analyzer: ONNXModel, tensor_name: str
) -> Optional[Tuple[int, ...]]:
    """Return fully-static shape from analyzer tensor info, or None."""
    info = analyzer.tensor_info.get(tensor_name, {})
    raw_shape = info.get("shape", ())
    if not raw_shape:
        return None
    if not all(isinstance(d, int) and d > 0 for d in raw_shape):
        return None
    return tuple(raw_shape)


def _shape_from_constants(
    tensor_name: str,
    constants: Dict[str, np.ndarray],
    analyzer: ONNXModel,
) -> Optional[Tuple[int, ...]]:
    """Resolve tensor shape from constants/weights/tensor-info."""
    if tensor_name in constants:
        return tuple(constants[tensor_name].shape)
    if tensor_name in analyzer.weights:
        return tuple(analyzer.weights[tensor_name].shape)
    return _shape_from_analyzer(analyzer, tensor_name)


def _reconstruct_abcd_shape_from_output_and_rhs(
    output_shape: Optional[Tuple[int, ...]],
    rhs_shape: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Reconstruct ``abcd`` for equation ``abcd,cde->abe`` when input shape is missing."""
    if output_shape is None or rhs_shape is None:
        return None
    if len(output_shape) != 3 or len(rhs_shape) != 3:
        return None

    a_dim, b_dim, e_dim = output_shape
    c_dim, d_dim, e_rhs = rhs_shape
    if e_dim != e_rhs:
        return None
    if not all(isinstance(v, int) and v > 0 for v in (a_dim, b_dim, c_dim, d_dim)):
        return None
    return (a_dim, b_dim, c_dim, d_dim)


def _make_const_name(base: str, suffix: str, constants: Dict[str, np.ndarray]) -> str:
    """Create a unique synthetic constant tensor name."""
    candidate = f"{base}__{suffix}"
    if candidate not in constants:
        return candidate
    i = 1
    while f"{candidate}_{i}" in constants:
        i += 1
    return f"{candidate}_{i}"


def _lower_abcd_cde_to_abe(
    layer: Dict,
    analyzer: ONNXModel,
    constants: Dict[str, np.ndarray],
    layer_id: int,
) -> Optional[List[Dict]]:
    """Lower equation: ``abcd,cde->abe`` into Reshape+MatMul+Reshape.

    Math:
        A[a,b,c,d], B[c,d,e] -> Y[a,b,e]
        reshape A -> [a*b, c*d]
        reshape B -> [c*d, e]
        matmul      [a*b, e]
        reshape     [a, b, e]
    """
    inputs = list(layer.get("inputs", []))
    outputs = list(layer.get("outputs", []))
    if len(inputs) != 2 or len(outputs) != 1:
        return None

    a_name, b_name = inputs
    y_name = outputs[0]

    a_shape = _shape_from_constants(a_name, constants, analyzer)
    b_shape = _shape_from_constants(b_name, constants, analyzer)
    y_shape = _shape_from_constants(y_name, constants, analyzer)

    if a_shape is None:
        a_shape = _reconstruct_abcd_shape_from_output_and_rhs(y_shape, b_shape)

    if a_shape is None or b_shape is None:
        return None
    if len(a_shape) != 4 or len(b_shape) != 3:
        return None

    a_dim, b_dim, c_dim, d_dim = a_shape
    c_rhs, d_rhs, e_dim = b_shape
    if c_dim != c_rhs or d_dim != d_rhs:
        return None

    base = layer.get("name") or f"einsum_{layer_id}"

    shape_a2d_name = _make_const_name(base, "shape_a2d", constants)
    b2d_const_name = _make_const_name(base, "rhs_cd_e", constants)
    shape_out_name = _make_const_name(base, "shape_out", constants)

    constants[shape_a2d_name] = np.array([a_dim * b_dim, c_dim * d_dim], dtype=np.int64)
    rhs_value = constants.get(b_name, analyzer.weights.get(b_name))
    if rhs_value is None:
        return None
    constants[b2d_const_name] = np.asarray(rhs_value).reshape(c_dim * d_dim, e_dim)
    constants[shape_out_name] = np.array([a_dim, b_dim, e_dim], dtype=np.int64)

    a2d = f"{base}__a2d"
    mm = f"{base}__mm"

    attrs = layer.get("attributes", {})

    def _node(
        name: str,
        op_type: str,
        node_inputs: Sequence[str],
        node_outputs: Sequence[str],
        node_attrs: Dict,
    ):
        return {
            "name": name,
            "op_type": op_type,
            "inputs": list(node_inputs),
            "outputs": list(node_outputs),
            "attributes": dict(node_attrs),
        }

    return [
        _node(f"{base}__reshape_a", "Reshape", [a_name, shape_a2d_name], [a2d], {}),
        _node(f"{base}__matmul", "MatMul", [a2d, b2d_const_name], [mm], {}),
        _node(f"{base}__reshape_out", "Reshape", [mm, shape_out_name], [y_name], attrs),
    ]


def lower_supported_einsum_layers(
    layers: List[Dict],
    analyzer: ONNXModel,
    constants: Dict[str, np.ndarray],
) -> Tuple[List[Dict], EinsumLoweringReport]:
    """Lower supported Einsum layers to core operators.

    Currently supported equations:
    - ``abcd,cde->abe`` (Transformer attention-output projection style)
    """
    lowered_layers: List[Dict] = []
    lowered_count = 0
    skipped_count = 0

    for layer_id, layer in enumerate(layers):
        if layer.get("op_type") != "Einsum":
            lowered_layers.append(layer)
            continue

        equation = layer.get("attributes", {}).get("equation", "")
        replacement: Optional[List[Dict]] = None

        if equation == "abcd,cde->abe":
            replacement = _lower_abcd_cde_to_abe(layer, analyzer, constants, layer_id)

        if replacement is None:
            skipped_count += 1
            lowered_layers.append(layer)
            continue

        lowered_count += 1
        lowered_layers.extend(replacement)

    return lowered_layers, EinsumLoweringReport(
        lowered_count=lowered_count,
        skipped_count=skipped_count,
    )
