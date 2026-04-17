"""
Main ONNX to IR conversion orchestration.
"""

import numpy as np
import logging
from typing import Dict, List, Set
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

NormalizedIRInputs = tuple[List[Dict], Dict[str, np.ndarray], Set[str]]


def _normalize_working_layers_and_constants(
    analyzer: ONNXModel,
) -> tuple[List[Dict], Dict[str, np.ndarray], set[str]]:
    """Prepare normalized layer stream and compile-time constants.

    Phase responsibilities:
      1) Validate/resolve model shapes (fail-fast on unresolved runtime shapes)
      2) Constant-fold compile-time subgraphs
    3) Normalize supported Einsum patterns

    Returns:
        (working_layers, constant_values, folded_outputs)
    """
    # Run shape validation BEFORE any layer extraction to fail fast if model
    # has unresolvable dynamic dimensions (e.g., dynamic batch size).
    ok, model_copy, changes, diagnostics = validate_model_shapes(analyzer.model)
    if model_copy is not None:
        analyzer.model = model_copy
        analyzer.refresh_after_model_mutation()
    if not ok:
        # Surface diagnostics as a ShapeValidationError to preserve existing
        # control flow for callers.
        raise ShapeValidationError(
            tensor_name="<unknown>",
            issue="; ".join(diagnostics or ["validation failed"]),
            shape=(),
            suggestions=diagnostics,
        )

    # Constant-folding pre-pass
    constant_values, folded_outputs = run_constant_folding(analyzer)
    if folded_outputs:
        logger.info(
            "Constant folding: resolved %d tensor(s) at compile time",
            len(folded_outputs),
        )

    # Einsum normalization pass
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

    return working_layers, constant_values, folded_outputs


def normalize_model_for_ir(analyzer: ONNXModel) -> NormalizedIRInputs:
    """Pass 1: Normalize ONNX model for IR extraction.

    This pass performs shape validation, constant folding, and supported Einsum
    normalization.

    Returns:
        (working_layers, constant_values, folded_outputs)
    """
    try:
        working_layers, constant_values, folded_outputs = (
            _normalize_working_layers_and_constants(analyzer)
        )
    except ShapeValidationError as e:
        logger.error(str(e))
        raise

    return (working_layers, constant_values, folded_outputs)


def canonicalize_model_for_ir(analyzer: ONNXModel) -> NormalizedIRInputs:
    """Backward-compatible alias for ``normalize_model_for_ir``."""
    return normalize_model_for_ir(analyzer)


def prepare_model_for_ir(analyzer: ONNXModel) -> NormalizedIRInputs:
    """Backward-compatible alias for ``normalize_model_for_ir``."""
    return normalize_model_for_ir(analyzer)


def _build_network_ir_unordered(
    *,
    layers: Dict[str, BaseLayer],
    tensor_producers: Dict[str, str],
    tensor_consumers: Dict[str, List[str]],
    input_tensors: tuple,
    output_tensors: tuple,
    state_tensors: Dict[str, str],
) -> NetworkIR:
    """Assemble a structurally complete ``NetworkIR`` before execution ordering.

    This makes phase boundaries explicit:
      1) graph assembly (layers/tensor maps/state)
      2) execution-order finalization (topological/SCC-aware ordering)
    """
    return NetworkIR(
        layers=layers,
        execution_order=[],
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors=state_tensors,
    )


def _finalize_network_ir_execution_order(ir_unordered: NetworkIR) -> NetworkIR:
    """Finalize ``NetworkIR`` by computing and attaching execution order."""
    execution_order = LayerGraph(ir_unordered).get_execution_order()
    return NetworkIR(
        layers=ir_unordered.layers,
        execution_order=execution_order,
        tensor_producers=ir_unordered.tensor_producers,
        tensor_consumers=ir_unordered.tensor_consumers,
        input_tensors=ir_unordered.input_tensors,
        output_tensors=ir_unordered.output_tensors,
        state_tensors=ir_unordered.state_tensors,
    )


def extract_typed_ir_graph(
    analyzer: ONNXModel,
    working_layers: List[Dict],
    constant_values: Dict[str, np.ndarray],
    folded_outputs: Set[str],
) -> NetworkIR:
    """Pass 2: Extract typed IR graph without execution order.

    Produces an unordered ``NetworkIR`` that already contains typed layers,
    tensor producer/consumer maps, and detected state tensors.
    """
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

    # Detect state tensors from RNN-family operators (LSTM, GRU, RNN, etc.)
    state_tensors = detect_state_tensors(analyzer, layers)
    if state_tensors:
        logger.info(
            f"Detected {len(state_tensors)} state tensors: {list(state_tensors.keys())}"
        )

    return _build_network_ir_unordered(
        layers=layers,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors=state_tensors,
    )


def schedule_network_ir(ir_unordered: NetworkIR) -> NetworkIR:
    """Pass 3: Compute dependency-respecting execution order on typed IR graph."""
    return _finalize_network_ir_execution_order(ir_unordered)


# ============================================================================
# Main Conversion Entry Point
# ============================================================================


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """
    Convert ONNX model to intermediate representation (IR).

    This creates a complete, ordered ``NetworkIR`` without optimization.
    Use ``IROptimizer`` for post-processing.

    High-level phases:
    1) Prepare/normalize ONNX layers (shape validation, constant folding,
         supported Einsum lowering)
      2) Extract typed IR layers + tensor maps
      3) Assemble unordered ``NetworkIR``
      4) Finalize execution order (topological/SCC-aware)
    """
    logger.info("Converting ONNX model to IR...")
    working_layers, constant_values, folded_outputs = normalize_model_for_ir(analyzer)
    ir_unordered = extract_typed_ir_graph(
        analyzer,
        working_layers,
        constant_values,
        folded_outputs,
    )
    ir_ordered = schedule_network_ir(ir_unordered)

    logger.info(f"Created IR with {len(ir_ordered.layers)} layers in execution order")

    return ir_ordered
