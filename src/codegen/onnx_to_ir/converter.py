"""
Main ONNX to IR conversion orchestration.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import onnx
from onnx import numpy_helper, TensorProto

from ..types import NetworkIR, BaseLayer
from ..onnx_model import ONNXModel
from ..graph_algorithms import condensation_execution_order, topo_sort_onnx_nodes
from .tensor_resolution import TensorResolver, ResolvedTensor
from .shape_inference import infer_layer_shapes
from .layer_extractors import LAYER_EXTRACTORS
from .state_detection import detect_state_tensors

logger = logging.getLogger(__name__)


# ============================================================================
# Constant Folding
# ============================================================================

# Operators that can be constant-folded when all inputs are known at compile time
_FOLDABLE_OPS = {
    "Shape",
    "Cast",
    "Slice",
    "Concat",
    "Expand",
    "Unsqueeze",
    "Gather",
    "Reshape",
    "Squeeze",
}


def _collect_constant_values(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """
    Pre-collect all constant/initializer tensors for constant folding.

    Gathers values from:
    - Graph initializers (model weights/parameters)
    - Constant nodes (embedded constant tensors)
    - Graph inputs with fully static shapes (treated as shape-only constants
      for ops like Shape that only inspect the tensor's dimensions)

    Returns:
        Dict mapping tensor name -> numpy array value
    """
    constants: Dict[str, np.ndarray] = {}

    # From initializers
    for init in model.graph.initializer:
        constants[init.name] = numpy_helper.to_array(init)

    # From Constant nodes
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    val = numpy_helper.to_array(attr.t)
                    for out in node.output:
                        constants[out] = val

    return constants


def _collect_static_input_shapes(model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    """
    Collect fully-static shapes for graph inputs (excluding initializers).

    Returns:
        Dict mapping input tensor name -> static shape tuple.
        Only includes inputs where every dimension is a concrete integer.
    """
    initializer_names = {init.name for init in model.graph.initializer}
    static_shapes: Dict[str, Tuple[int, ...]] = {}

    for inp in model.graph.input:
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


def _try_constant_fold(
    node,
    constant_values: Dict[str, np.ndarray],
    static_input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> bool:
    """
    Try to constant-fold a node whose inputs are all known constants.

    If all inputs to the node are available in constant_values, we evaluate the
    operation using NumPy and store the result back into constant_values for
    downstream nodes.

    Special case: Shape nodes can be folded if the input has a known static
    shape, even if the input data is not constant.

    Args:
        node: ONNX graph node
        constant_values: Mutable dict of tensor_name -> numpy value
        static_input_shapes: Optional dict of tensor_name -> static shape tuple

    Returns:
        True if the node was successfully folded (outputs stored in constant_values),
        False if it cannot be folded (has runtime inputs or unsupported op).
    """
    op = node.op_type

    if op not in _FOLDABLE_OPS:
        return False

    # Special case: Shape op only needs the input's shape, not its data
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

    # Preserve positional alignment: use None for empty/optional inputs.
    # This is critical for ops like Slice where inputs[3] means "axes" —
    # if we compact out empty strings, the index mapping breaks.
    inputs = [constant_values.get(inp) if inp else None for inp in node.input]

    # All non-optional inputs must be known constants.
    # An input slot is "required" if its name is a non-empty string.
    if any(v is None and inp != "" for inp, v in zip(node.input, inputs)):
        return False

    try:
        result = _evaluate_constant_op(op, node, inputs)
    except Exception as e:
        logger.debug(f"Could not constant-fold {op} '{node.name}': {e}")
        return False

    # Store folded result for downstream nodes
    for out in node.output:
        if out:
            constant_values[out] = np.array(result)

    logger.debug(
        f"Constant-folded {op} node '{node.name}' -> "
        f"shape {np.array(result).shape}, dtype {np.array(result).dtype}"
    )
    return True


def _evaluate_constant_op(
    op: str, node, inputs: List[Optional[np.ndarray]]
) -> np.ndarray:
    """
    Evaluate a single ONNX operator on constant numpy inputs.

    inputs is positionally aligned with node.input — optional slots that were
    empty strings in the ONNX node are represented as None here.

    Args:
        op: ONNX operator type string
        node: ONNX node (for reading attributes)
        inputs: Positionally-aligned list of numpy arrays; None means the slot
                was an empty/optional input in the ONNX graph.

    Returns:
        Result numpy array

    Raises:
        ValueError: If the op is not supported for constant folding
        Exception: If evaluation fails for any reason
    """

    def _get(i):
        """Return inputs[i] if it exists and is not None."""
        return inputs[i] if i < len(inputs) else None

    if op == "Shape":
        return np.array(inputs[0].shape, dtype=np.int64)

    elif op == "Cast":
        to_type = next(a.i for a in node.attribute if a.name == "to")
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(to_type, np.float32)
        return inputs[0].astype(np_dtype)

    elif op == "Slice":
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

    elif op == "Concat":
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        # Filter out None slots (empty optional inputs are invalid for Concat,
        # but be defensive)
        real_inputs = [x for x in inputs if x is not None]
        return np.concatenate(real_inputs, axis=axis)

    elif op == "Unsqueeze":
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

    elif op == "Squeeze":
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
        else:
            return np.squeeze(data)

    elif op == "Expand":
        target_shape = inputs[1].flatten().tolist()
        return np.broadcast_to(inputs[0], [int(s) for s in target_shape]).copy()

    elif op == "Gather":
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        return np.take(inputs[0], inputs[1].astype(np.intp), axis=axis)

    elif op == "Reshape":
        shape = inputs[1].flatten().tolist()
        return inputs[0].reshape([int(s) for s in shape])

    else:
        raise ValueError(f"Unsupported constant-fold op: {op}")


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

    # --- Constant-folding pre-pass ---
    # Collect all known constant tensors (initializers + Constant nodes)
    constant_values = _collect_constant_values(analyzer.model)
    logger.debug(
        f"Constant folding: {len(constant_values)} initial constants collected"
    )

    # Collect static input shapes so Shape ops on inputs can be folded
    static_input_shapes = _collect_static_input_shapes(analyzer.model)
    if static_input_shapes:
        logger.debug(
            f"Constant folding: {len(static_input_shapes)} static input shape(s): "
            + ", ".join(f"{k} -> {v}" for k, v in static_input_shapes.items())
        )

    # Walk the graph in topological order and fold what we can.
    # Build a set of node output names that were fully folded so we can skip
    # them during the main layer-extraction loop.
    folded_outputs: set = set()
    for node in topo_sort_onnx_nodes(analyzer.model.graph):
        if node.op_type == "Constant":
            # Constant nodes are already in constant_values and will be naturally skipped by the extractor
            # because they have no layer extractor registered.
            continue

        if _try_constant_fold(node, constant_values, static_input_shapes):
            for out in node.output:
                if out:
                    folded_outputs.add(out)

    folded_count = len(folded_outputs)
    if folded_count > 0:
        logger.info(
            f"Constant folding: resolved {folded_count} tensor(s) at compile time"
        )

    # --- Main layer extraction ---
    resolver = TensorResolver(analyzer)
    input_info, output_info = analyzer.get_input_output_info()
    input_tensors = tuple(input_info["names"])
    output_tensors = tuple(output_info["names"])

    layers: Dict[str, BaseLayer] = {}
    tensor_producers: Dict[str, str] = {}
    tensor_consumers: Dict[str, List[str]] = defaultdict(list)
    unsupported_ops: Dict[str, List[int]] = defaultdict(list)  # op_type -> [layer_ids]

    # Process each layer
    for layer_id, layer_dict in enumerate(analyzer.layers):

        node_outputs = layer_dict.get("outputs", [])
        op_type = layer_dict.get("op_type", "")

        if node_outputs and all(out in folded_outputs for out in node_outputs):
            logger.debug(
                f"Skipping constant-folded node {layer_id}: "
                f"{layer_dict.get('name', '?')} ({op_type})"
            )
            continue

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

                for inp in ir_layer.inputs:
                    tensor_consumers[inp].append(ir_layer.name)
                for out in ir_layer.outputs:
                    tensor_producers[out] = ir_layer.name

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

    # Execution ordering using SCC-condensation
    # This automatically handles both cyclic and acyclic graphs gracefully
    execution_order = condensation_execution_order(
        layers, tensor_producers, input_tensors
    )

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
