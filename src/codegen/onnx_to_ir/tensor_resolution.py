"""
Tensor resolution utilities.
Resolves tensor shapes, types, and values from ONNX model.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from ..onnx_model import ONNXModel

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTensor:
    """Fully resolved tensor information (all static)"""

    name: str
    shape: Tuple[int, ...]
    dtype: Optional[str]
    size: int
    value: Optional[np.ndarray]  # None if not a constant/weight
    is_weight: bool


class TensorResolver:
    """
    Resolves tensor information from ONNX model.
    Tracks inferred shapes as the graph is processed.

    This class handles:
    1. Extracting static shapes from ONNX tensor_info
    2. Tracking inferred shapes from previous layers (shape propagation)
    3. Resolving weight tensors (including dequantized weights)
    4. Determining dtypes for computation

    It does NOT perform operation-specific shape inference - that's in `onnx_to_ir.shape`.
    """

    def __init__(
        self,
        analyzer: ONNXModel,
        compile_time_constants: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.analyzer = analyzer
        self.inferred_shapes: Dict[str, Tuple[int, ...]] = {}
        self.compile_time_constants = compile_time_constants or {}

        # Initialize with network input shapes
        input_info, _ = analyzer.get_input_output_info()
        for i, inp_name in enumerate(input_info["names"]):
            shape = input_info["shapes"][i]
            # Strict contract: graph inputs must be fully static after validation.
            if any(not (isinstance(d, int) and d > 0) for d in shape):
                raise ValueError(
                    f"Input tensor '{inp_name}' has unresolved shape {shape}. "
                    "All input dimensions must be concrete before ONNX->IR conversion."
                )
            static_shape = tuple(int(d) for d in shape)
            if static_shape:
                self.inferred_shapes[inp_name] = static_shape
                logger.debug(f"Initialized input shape: {inp_name} -> {static_shape}")

    def infer_computation_dtype(self, layer_dict: Dict) -> str:
        """
        Infer the dtype for a layer's computation.

        Priority:
        1. Use first data input's dtype (not weights)
        2. Fall back to network input dtype
        """
        # Try to get dtype from first non-weight input
        for inp_name in layer_dict["inputs"]:
            if inp_name not in self.analyzer.weights:  # Skip weights
                tensor_info = self.analyzer.tensor_info.get(inp_name, {})
                if dtype := tensor_info.get("onnx_type"):
                    return dtype

        # Fall back to network input dtype
        input_info, _ = self.analyzer.get_input_output_info()
        if input_info["dtypes"]:
            return input_info["dtypes"][0]

        raise ValueError(
            f"Cannot infer dtype for layer '{layer_dict['name']}'. "
            f"Try running onnx.shape_inference.infer_shapes(model)"
        )

    def _get_shape_from_onnx(self, tensor_name: str) -> Tuple[int, ...]:
        """Extract static shape from ONNX tensor_info."""
        tensor_info = self.analyzer.tensor_info.get(tensor_name, {})
        shape_from_onnx = tensor_info.get("shape", ())

        if not shape_from_onnx:
            return ()

        # Strict contract: never substitute unknown/symbolic dims with defaults.
        # If ONNX shape metadata is not fully static, let shape inference resolve
        # it explicitly downstream rather than inventing dimensions.
        if any(not (isinstance(d, int) and d > 0) for d in shape_from_onnx):
            return ()

        return tuple(int(d) for d in shape_from_onnx)

    def resolve_tensor(
        self, tensor_name: str, computation_dtype: str, is_output: bool = False
    ) -> ResolvedTensor:
        """
            Resolve a single tensor (input or output).

            Shape resolution priority:
            1. Actual numpy array shape (for weights)
            2. Previously inferred shape (from earlier layers)
            3. ONNX tensor_info static shape
        4. Empty tuple (will be inferred by the shape engine later)

            Args:
                tensor_name: Name of the tensor
                computation_dtype: Fallback dtype for computation
                is_output: Whether this is an output tensor
        """
        tensor_info = self.analyzer.tensor_info.get(tensor_name, {})

        # Check if this is a weight
        is_weight = tensor_name in self.analyzer.weights
        weight_value = self.analyzer.weights.get(tensor_name)

        # Compile-time constants from constant-folding should behave like weights
        # for extraction purposes (e.g., Reshape target shapes).
        if not is_weight and weight_value is None:
            folded_value = self.compile_time_constants.get(tensor_name)
            if folded_value is not None:
                weight_value = folded_value
                is_weight = True

        # Try to get dequantized weight if not found directly
        if not is_weight and weight_value is None:
            from .weight_utils import try_get_dequantized_weight

            weight_value = try_get_dequantized_weight(tensor_name, self.analyzer)
            if weight_value is not None:
                is_weight = True

        # Determine shape (priority order as documented above)
        if is_weight and weight_value is not None:
            # Priority 1: Actual numpy array shape
            static_shape = tuple(weight_value.shape)
            logger.debug(f"Resolved weight {tensor_name}: shape={static_shape}")

            # Validate against ONNX info if available
            onnx_shape = self._get_shape_from_onnx(tensor_name)
            if onnx_shape and onnx_shape != static_shape:
                logger.warning(
                    f"Weight {tensor_name}: ONNX shape {onnx_shape} != "
                    f"actual shape {static_shape}"
                )
        else:
            # Priority 2: Previously inferred shape
            if tensor_name in self.inferred_shapes:
                static_shape = self.inferred_shapes[tensor_name]
                logger.debug(
                    f"Resolved tensor {tensor_name} from inference: shape={static_shape}"
                )
            else:
                # Priority 3: ONNX tensor_info
                static_shape = self._get_shape_from_onnx(tensor_name)
                if static_shape:
                    logger.debug(
                        f"Resolved tensor {tensor_name} from ONNX: shape={static_shape}"
                    )
                else:
                    # Priority 4: Empty (will be inferred later)
                    logger.debug(
                        f"Tensor {tensor_name} has no shape info, will be inferred"
                    )

        # Dtype resolution:
        # - For weights: can be None (weights don't need type info)
        # - For non-weights (activations/outputs): MUST have a dtype
        dtype = tensor_info.get("onnx_type")

        if is_weight:
            # Weights can have None dtype
            pass
        else:
            # Non-weights MUST have a dtype
            if dtype is None:
                dtype = computation_dtype
                logger.debug(
                    f"Tensor '{tensor_name}': dtype was None, using computation_dtype={computation_dtype}"
                )

        size = int(np.prod(static_shape)) if static_shape else 0

        return ResolvedTensor(
            name=tensor_name,
            shape=static_shape,
            dtype=dtype,
            size=size,
            value=weight_value,
            is_weight=is_weight,
        )

    def resolve_layer_tensors(self, layer_dict: Dict) -> Dict:
        """
        Resolve all tensors for a layer.

        Note: Empty output names (representing optional ONNX outputs like LSTM's
        Y_h, Y_c) are still resolved to ResolvedTensors but may have limited info.

        Returns:
            Layer dict enriched with 'resolved_inputs' and 'resolved_outputs'
        """
        computation_dtype = self.infer_computation_dtype(layer_dict)

        resolved_inputs = [
            self.resolve_tensor(inp_name, computation_dtype, is_output=False)
            for inp_name in layer_dict["inputs"]
        ]

        resolved_outputs = [
            self.resolve_tensor(out_name, computation_dtype, is_output=True)
            for out_name in layer_dict["outputs"]
        ]

        return {
            **layer_dict,
            "resolved_inputs": resolved_inputs,
            "resolved_outputs": resolved_outputs,
        }

    def store_inferred_shape(self, tensor_name: str, shape: Tuple[int, ...]):
        """Store an inferred shape for a tensor (from `onnx_to_ir.shape`)."""
        if shape:
            self.inferred_shapes[tensor_name] = shape
            logger.debug(f"Stored inferred shape: {tensor_name} -> {shape}")
