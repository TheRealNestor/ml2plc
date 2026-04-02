"""
Main ONNX to IR conversion orchestration.
"""

import numpy as np
import logging
from typing import Dict, List
from collections import defaultdict

from ..types import NetworkIR, BaseLayer
from ..onnx_model import ONNXModel
from ..graph_algorithms import (
    condensation_execution_order,
)
from .tensor_resolution import TensorResolver, ResolvedTensor
from .shape_inference import infer_layer_shapes
from .layer_extractors import LAYER_EXTRACTORS

logger = logging.getLogger(__name__)


# ONNX RNN operator input index specifications
# See: https://github.com/onnx/onnx/blob/main/docs/Operators.md
_STATE_INPUT_INDICES = {
    # LSTM: inputs[5]=initial_h, inputs[6]=initial_c
    "LSTM": (5, 6),
    # GRU: inputs[5]=initial_h
    "GRU": (5,),
    # RNN: inputs[5]=initial_h
    "RNN": (5,),
}


def _extract_lstm_state_tensors(
    layer_name: str,
    ir_layer: BaseLayer,
    state_tensors: Dict[str, str],
) -> None:
    """
    Extract state tensors from an LSTM layer.

    Per ONNX spec, LSTM initial_h and initial_c are at inputs[5] and inputs[6].
    """
    if len(ir_layer.inputs) >= 7:
        state_tensors[ir_layer.inputs[5]] = "state"  # initial_h
        state_tensors[ir_layer.inputs[6]] = "state"  # initial_c
        logger.debug(f"LSTM '{layer_name}': marked state tensors at inputs[5:7]")
    else:
        logger.debug(
            f"LSTM '{layer_name}': insufficient inputs for state detection "
            f"(got {len(ir_layer.inputs)}, need >= 7)"
        )


def _extract_gru_state_tensors(
    layer_name: str,
    ir_layer: BaseLayer,
    state_tensors: Dict[str, str],
) -> None:
    """
    Extract state tensors from a GRU layer.

    Per ONNX spec, GRU initial_h is at inputs[5].
    """
    if len(ir_layer.inputs) >= 6:
        state_tensors[ir_layer.inputs[5]] = "state"  # initial_h
        logger.debug(f"GRU '{layer_name}': marked state tensor at inputs[5]")
    else:
        logger.debug(
            f"GRU '{layer_name}': insufficient inputs for state detection "
            f"(got {len(ir_layer.inputs)}, need >= 6)"
        )


def _extract_rnn_state_tensors(
    layer_name: str,
    ir_layer: BaseLayer,
    state_tensors: Dict[str, str],
) -> None:
    """
    Extract state tensors from an RNN layer.

    Per ONNX spec, RNN initial_h is at inputs[5].
    """
    if len(ir_layer.inputs) >= 6:
        state_tensors[ir_layer.inputs[5]] = "state"  # initial_h
        logger.debug(f"RNN '{layer_name}': marked state tensor at inputs[5]")
    else:
        logger.debug(
            f"RNN '{layer_name}': insufficient inputs for state detection "
            f"(got {len(ir_layer.inputs)}, need >= 6)"
        )


def _extract_scan_state_tensors(
    layer_name: str,
    layer_dict: Dict,
    state_tensors: Dict[str, str],
) -> None:
    """
    Extract state tensors from a Scan layer.

    Scan operators declare state variables via the 'loop_state_variables'
    attribute, which specifies which inputs participate in the loop state.
    """
    attributes = layer_dict.get("attributes", {})
    if "loop_state_variables" in attributes:
        for state_var in attributes["loop_state_variables"]:
            state_tensors[state_var] = "state"
        logger.debug(
            f"Scan '{layer_name}': marked {len(attributes['loop_state_variables'])} "
            f"state variables from loop_state_variables attribute"
        )


def _extract_loop_state_tensors(
    layer_name: str,
    layer_dict: Dict,
    state_tensors: Dict[str, str],
) -> None:
    """
    Extract state tensors from a Loop layer.

    Note: Loop state detection requires analyzing the subgraph body, which is
    more complex. This is marked for future enhancement.
    TODO: Implement subgraph analysis for Loop state detection
    """
    logger.debug(
        f"Loop '{layer_name}': state detection requires subgraph analysis (TODO)"
    )


# Dispatcher for state tensor extraction by operator type
_STATE_EXTRACTORS = {
    "LSTM": _extract_lstm_state_tensors,
    "GRU": _extract_gru_state_tensors,
    "RNN": _extract_rnn_state_tensors,
    "Scan": _extract_scan_state_tensors,
    "Loop": _extract_loop_state_tensors,
}


def _detect_state_tensors(
    analyzer: ONNXModel,
    layers: Dict[str, BaseLayer],
) -> Dict[str, str]:
    """
    Detect state tensors from RNN-family operators.

    Scans the ONNX model for RNN-family operators (LSTM, GRU, RNN, Scan, Loop)
    and extracts their state tensor information. This allows regionization to
    correctly identify recurrent control flow without heuristics.

    The detection leverages ONNX operator specifications, which define explicit
    positions or attributes for state tensors. Each operator type has a dedicated
    extraction function for clarity and maintainability.

    Args:
        analyzer: The loaded ONNX model
        layers: Dictionary of extracted IR layers

    Returns:
        Dict mapping tensor_name -> "state" for each detected state tensor
    """
    state_tensors: Dict[str, str] = {}

    for layer_dict in analyzer.layers:
        op_type = layer_dict.get("op_type", "")

        # Skip if this operator type doesn't have state semantics
        if op_type not in _STATE_EXTRACTORS:
            continue

        # Find the corresponding IR layer
        layer_name = layer_dict.get("name", "")
        if layer_name not in layers:
            logger.debug(
                f"Operator '{op_type}' layer '{layer_name}' not found in IR layers"
            )
            continue

        ir_layer = layers[layer_name]

        # Delegate to operator-specific extractor
        # Scan and Loop need the full layer_dict for attributes
        if op_type in {"Scan", "Loop"}:
            _STATE_EXTRACTORS[op_type](layer_name, layer_dict, state_tensors)
        else:
            _STATE_EXTRACTORS[op_type](layer_name, ir_layer, state_tensors)

    return state_tensors


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """
    Convert ONNX model to intermediate representation (IR).

    This creates a complete IR without optimization.
    Use IROptimizer for post-processing.
    """
    logger.info("Converting ONNX model to IR...")

    resolver = TensorResolver(analyzer)
    input_info, output_info = analyzer.get_input_output_info()
    input_tensors = tuple(input_info["names"])
    output_tensors = tuple(output_info["names"])

    layers: Dict[str, BaseLayer] = {}
    tensor_producers: Dict[str, str] = {}
    tensor_consumers: Dict[str, List[str]] = defaultdict(list)

    # Process each layer
    for layer_id, layer_dict in enumerate(analyzer.layers):

        enriched_layer = resolver.resolve_layer_tensors(layer_dict)
        _, output_shape = infer_layer_shapes(enriched_layer)

        for out_name in enriched_layer["outputs"]:
            resolver.store_inferred_shape(out_name, output_shape)

        if output_shape and enriched_layer["resolved_outputs"]:
            enriched_layer["resolved_outputs"] = [
                ResolvedTensor(
                    name=out.name,
                    shape=output_shape,
                    dtype=out.dtype,
                    size=int(np.prod(output_shape)) if output_shape else 0,
                    value=out.value,
                    is_weight=out.is_weight,
                )
                for out in enriched_layer["resolved_outputs"]
            ]

        op_type = enriched_layer["op_type"]
        if op_type in LAYER_EXTRACTORS:
            try:
                ir_layer = LAYER_EXTRACTORS[op_type](enriched_layer, layer_id, analyzer)
                layers[ir_layer.name] = ir_layer
                logger.debug(f"Extracted layer {layer_id}: {ir_layer.name} ({op_type})")

                # Track graph structure
                for inp in ir_layer.inputs:
                    tensor_consumers[inp].append(ir_layer.name)
                for out in ir_layer.outputs:
                    tensor_producers[out] = ir_layer.name

            except Exception as e:
                logger.error(f"Failed to extract layer {layer_id} ({op_type}): {e}")
                raise
        else:
            logger.warning(f"Unsupported layer type: {op_type}")

    # Execution ordering using SCC-condensation
    # This automatically handles both cyclic and acyclic graphs gracefully
    execution_order = condensation_execution_order(
        layers, tensor_producers, input_tensors
    )

    # Detect state tensors from RNN-family operators
    state_tensors = _detect_state_tensors(analyzer, layers)
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
