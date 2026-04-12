"""Constant-folding pipeline helpers.

Keeps converter orchestration thin by encapsulating:
- constant collection
- static-shape discovery
- fixed-point graph folding
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import logging
import numpy as np
from onnx import numpy_helper

from ...onnx_model import ONNXModel
from ...graph_algorithms import topo_sort_onnx_nodes
from ..tensor_resolution import TensorResolver
from ..shape import infer_layer_shapes
from .constant_graph import try_constant_fold_node

logger = logging.getLogger(__name__)


def _collect_constant_values(analyzer: ONNXModel) -> Dict[str, np.ndarray]:
    constants: Dict[str, np.ndarray] = {}

    for init in analyzer.model.graph.initializer:
        constants[init.name] = numpy_helper.to_array(init)

    for node in analyzer.model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            val = numpy_helper.to_array(attr.t)
            for out in node.output:
                constants[out] = val

    return constants


def _collect_static_input_shapes(analyzer: ONNXModel) -> Dict[str, Tuple[int, ...]]:
    initializer_names = {init.name for init in analyzer.model.graph.initializer}
    static_shapes: Dict[str, Tuple[int, ...]] = {}

    for inp in analyzer.model.graph.input:
        if inp.name in initializer_names:
            continue
        type_proto = inp.type.tensor_type
        if not type_proto.HasField("shape"):
            continue
        dims = []
        is_static = True
        for dim in type_proto.shape.dim:
            if dim.dim_value > 0:
                dims.append(dim.dim_value)
            else:
                is_static = False
                break
        if is_static and dims:
            static_shapes[inp.name] = tuple(dims)

    return static_shapes


def _collect_static_tensor_shapes_from_layers(
    analyzer: ONNXModel,
    constant_values: Dict[str, np.ndarray],
) -> Dict[str, Tuple[int, ...]]:
    resolver = TensorResolver(analyzer, compile_time_constants=constant_values)
    inferred: Dict[str, Tuple[int, ...]] = {}

    for layer_dict in analyzer.layers:
        try:
            enriched = resolver.resolve_layer_tensors(layer_dict)
            _, output_shape = infer_layer_shapes(enriched)

            outputs = enriched.get("outputs", [])
            if outputs and output_shape:
                resolver.store_inferred_shape(outputs[0], output_shape)
                inferred[outputs[0]] = output_shape

            for resolved_out in enriched.get("resolved_outputs", []):
                if resolved_out.shape:
                    resolver.store_inferred_shape(resolved_out.name, resolved_out.shape)
                    inferred[resolved_out.name] = tuple(resolved_out.shape)
        except Exception:
            continue

    return inferred


def run_constant_folding(
    analyzer: ONNXModel,
    *,
    max_fold_rounds: int = 3,
) -> Tuple[Dict[str, np.ndarray], Set[str]]:
    """Run fixed-point constant folding and return constants + folded outputs."""
    constant_values = _collect_constant_values(analyzer)
    folded_outputs: Set[str] = set()

    for _ in range(max_fold_rounds):
        static_input_shapes = _collect_static_input_shapes(analyzer)
        static_input_shapes.update(
            _collect_static_tensor_shapes_from_layers(analyzer, constant_values)
        )

        newly_folded = 0
        for node in topo_sort_onnx_nodes(analyzer.model.graph):
            if node.op_type == "Constant":
                continue

            if try_constant_fold_node(node, constant_values, static_input_shapes):
                for out in node.output:
                    if out and out not in folded_outputs:
                        folded_outputs.add(out)
                        newly_folded += 1

        if newly_folded == 0:
            break

    logger.debug(
        "Constant folding summary: %d initial constants, %d folded outputs",
        len(constant_values),
        len(folded_outputs),
    )
    return constant_values, folded_outputs
