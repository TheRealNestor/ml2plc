"""
Main ONNX to IR conversion orchestration.
"""

import numpy as np
import logging
from typing import Dict, List
from collections import defaultdict

from ..types import NetworkIR, BaseLayer
from ..onnx_model import ONNXModel
from ..graph.core import LayerGraph
from ..shape_semantics import ShapeSemanticsTracker
from .tensor_resolution import TensorResolver
from .shape import (
    infer_layer_shapes,
    validate_model_shapes,
    ShapeValidationError,
)
from .layer_extractors import LAYER_EXTRACTORS
from .state_detection import detect_state_tensors
from .einsum_lowering import lower_supported_einsum_layers
from .folding import run_constant_folding, try_fold_enriched_shape_layer

logger = logging.getLogger(__name__)


# ============================================================================
# Main Conversion Entry Point
# ============================================================================


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """
    Convert ONNX model to intermediate representation (IR).

    This creates a complete IR without optimization.
    Use IROptimizer for post-processing.

    The conversion includes a constant-folding pre-pass that resolves
    shape-manipulation operators (Shape, Cast, Slice, Concat, Expand,
    Unsqueeze, Gather, Reshape, Squeeze) when all their inputs are
    compile-time constants. This eliminates the auxiliary graph nodes
    that commonly surround LSTM and other RNN operators.
    """
    logger.info("Converting ONNX model to IR...")

    # ✅ LAYER 1: GROUND TRUTH EXTRACTION & VALIDATION
    # Run shape validation BEFORE any layer extraction to fail fast if model
    # has unresolvable dynamic dimensions (e.g., dynamic batch size)
    try:
        resolution_report = validate_model_shapes(analyzer.model)
        if resolution_report.modified:
            analyzer.refresh_after_model_mutation()
    except ShapeValidationError as e:
        logger.error(str(e))
        raise

    # --- Constant-folding pre-pass ---
    constant_values, folded_outputs = run_constant_folding(analyzer)

    folded_count = len(folded_outputs)
    if folded_count > 0:
        logger.info(
            f"Constant folding: resolved {folded_count} tensor(s) at compile time"
        )

    # --- Einsum lowering pass (pattern-driven canonicalization) ---
    working_layers, einsum_report = lower_supported_einsum_layers(
        analyzer.layers,
        analyzer,
        constant_values,
    )
    if einsum_report.lowered_count:
        logger.info(
            "Einsum lowering: lowered %d node(s) to core ops",
            einsum_report.lowered_count,
        )

    # --- Main layer extraction ---
    resolver = TensorResolver(analyzer, compile_time_constants=constant_values)
    shape_semantics = ShapeSemanticsTracker(constant_values)
    input_info, output_info = analyzer.get_input_output_info()
    input_tensors = tuple(input_info["names"])
    output_tensors = tuple(output_info["names"])

    layers: Dict[str, BaseLayer] = {}
    tensor_producers: Dict[str, str] = {}
    tensor_consumers: Dict[str, List[str]] = defaultdict(list)
    unsupported_ops: Dict[str, List[int]] = defaultdict(list)  # op_type -> [layer_ids]

    # Process each layer
    for layer_id, layer_dict in enumerate(working_layers):

        node_outputs = layer_dict.get("outputs", [])
        op_type = layer_dict.get("op_type", "")

        if node_outputs and all(out in folded_outputs for out in node_outputs):
            logger.debug(
                f"Skipping constant-folded node {layer_id}: "
                f"{layer_dict.get('name', '?')} ({op_type})"
            )
            continue

        enriched_layer = resolver.resolve_layer_tensors(layer_dict)
        enriched_layer["_shape_semantics"] = shape_semantics
        _, output_shape = infer_layer_shapes(enriched_layer)
        enriched_layer["_inferred_output_shape"] = output_shape

        # Propagate inferred shape conservatively.
        # IMPORTANT: A single inferred output shape cannot be blindly applied to
        # all outputs of multi-output operators (or to outputs with already-known
        # distinct ONNX shapes), as that can corrupt downstream rank semantics.
        if enriched_layer["outputs"]:
            if output_shape:
                # Update primary output shape (index 0) from semantic inference.
                resolver.store_inferred_shape(
                    enriched_layer["outputs"][0], output_shape
                )

            # Preserve any per-output shapes that were already resolved from ONNX
            # tensor info or compile-time constants.
            for resolved_out in enriched_layer["resolved_outputs"]:
                if (
                    resolved_out.shape
                    and resolved_out.name not in resolver.inferred_shapes
                ):
                    resolver.store_inferred_shape(resolved_out.name, resolved_out.shape)

        output_names = tuple(name for name in enriched_layer.get("outputs", []) if name)

        folded_value = try_fold_enriched_shape_layer(enriched_layer, constant_values)
        if folded_value is not None:
            for out_name in output_names:
                constant_values[out_name] = np.array(folded_value)
                folded_outputs.add(out_name)
                resolver.store_inferred_shape(
                    out_name, tuple(np.array(folded_value).shape)
                )
            continue

        shape_semantics.record_layer(enriched_layer)

        if output_names and not output_shape:
            if any(
                shape_semantics.role_of(name).value == "VALUE" for name in output_names
            ):
                raise ValueError(
                    f"Layer {layer_id} ({enriched_layer.get('name', '?')} / "
                    f"{enriched_layer.get('op_type', '?')}) has unresolved VALUE output shape. "
                    "Refuse to lower without fully static runtime shape contracts."
                )

        if output_names and all(
            shape_semantics.role_of(name).value == "SHAPE" for name in output_names
        ):
            logger.debug(
                "Skipping SHAPE-only node %d: %s (%s)",
                layer_id,
                enriched_layer.get("name", "?"),
                enriched_layer.get("op_type", "?"),
            )
            continue

        op_type = enriched_layer["op_type"]
        if op_type in LAYER_EXTRACTORS:
            try:
                ir_layer = LAYER_EXTRACTORS[op_type](enriched_layer, layer_id, analyzer)
                layers[ir_layer.name] = ir_layer
                logger.debug(f"Extracted layer {layer_id}: {ir_layer.name} ({op_type})")

                for inp in ir_layer.inputs:
                    tensor_consumers[inp].append(ir_layer.name)
                for out in ir_layer.outputs:
                    tensor_producers[out] = ir_layer.name

            except NotImplementedError as e:
                unsupported_ops[op_type].append(layer_id)
                logger.warning(
                    f"Skipping unsupported op '{op_type}' at layer {layer_id} "
                    f"({layer_dict.get('name', '?')}): {e}"
                )

            except Exception as e:
                logger.error(f"Failed to extract layer {layer_id} ({op_type}): {e}")
                raise
        else:
            unsupported_ops[op_type].append(layer_id)
            logger.warning(
                f"Skipping unsupported op '{op_type}' at layer {layer_id} "
                f"({layer_dict.get('name', '?')})"
            )

    if unsupported_ops:
        summary = ", ".join(
            f"'{op}' (layer(s) {ids})" for op, ids in sorted(unsupported_ops.items())
        )
        raise NotImplementedError(
            f"Unsupported ONNX operators encountered: {summary}. "
            f"Add extractors to LAYER_EXTRACTORS before converting this model."
        )

    # Execution ordering through canonical LayerGraph abstraction
    temp_ir = NetworkIR(
        layers=layers,
        execution_order=[],
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors={},
    )
    execution_order = LayerGraph(temp_ir).get_execution_order()

    # Detect state tensors from RNN-family operators (LSTM, GRU, RNN, etc.)
    state_tensors = detect_state_tensors(analyzer, layers)
    if state_tensors:
        logger.info(
            f"Detected {len(state_tensors)} state tensors: {list(state_tensors.keys())}"
        )

    logger.info(f"Created IR with {len(layers)} layers in execution order")

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors=state_tensors,
    )
