"""
Unified state tensor detection.

Consolidates ONNX operator analysis in one place.
Identifies which tensors represent state (vs regular intermediate tensors).
"""

import logging
from typing import Dict, Set
from ..types import BaseLayer
from ..onnx_model import ONNXModel

logger = logging.getLogger(__name__)

# ONNX RNN operator state input indices
_STATE_INPUT_SPECS = {
    "LSTM": {"initial_h": 5, "initial_c": 6},
    "GRU": {"initial_h": 5},
    "RNN": {"initial_h": 5},
}

_CONTROL_FLOW_OPS = {"Loop", "Scan", "If"}


class StateDetector:
    """
    Detects state tensors from ONNX operators.

    Single responsibility: identify which tensors represent state
    (vs regular intermediate tensors).

    Strategy:
    1. Scan all ONNX operators for RNN-family operators
    2. Extract state input tensors based on ONNX specification
    3. Mark state tensors in a dictionary for later use

    Example:
        analyzer = ONNXModel(onnx_model_proto)
        layers = convert_to_ir(onnx_model_proto, analyzer)
        detector = StateDetector(analyzer, layers)
        state_tensors = detector.detect_all()
        # state_tensors = {"h_0": "state", "c_0": "state", ...}
    """

    def __init__(self, analyzer: ONNXModel, layers: Dict[str, BaseLayer]):
        """
        Args:
            analyzer: ONNXModel instance with model metadata
            layers: Dictionary of converted IR layers keyed by layer name
        """
        self.analyzer = analyzer
        self.layers = layers
        self.state_tensors: Dict[str, str] = {}

    def detect_all(self) -> Dict[str, str]:
        """
        Scan model for all state tensor definitions.

        Returns:
            Dictionary mapping tensor_name -> "state" for state tensors
        """
        for layer_dict in self.analyzer.layers:
            op_type = layer_dict.get("op_type", "")
            if op_type in _STATE_INPUT_SPECS:
                self._detect_rnn_state(op_type, layer_dict)
            elif op_type in _CONTROL_FLOW_OPS:
                self._detect_control_flow_state(op_type, layer_dict)
        return self.state_tensors

    def _detect_rnn_state(self, op_type: str, layer_dict: Dict) -> None:
        """
        Extract state tensors from RNN-family operators (LSTM, GRU, RNN).

        Uses ONNX specification to identify which inputs are state:
        - LSTM: initial_h at index 5, initial_c at index 6
        - GRU: initial_h at index 5
        - RNN: initial_h at index 5

        Args:
            op_type: Operator type ("LSTM", "GRU", or "RNN")
            layer_dict: Original ONNX layer dictionary
        """
        ir_layer = self.layers.get(layer_dict["name"])
        if not ir_layer:
            return

        specs = _STATE_INPUT_SPECS[op_type]
        for state_name, idx in specs.items():
            if len(ir_layer.inputs) > idx:
                tensor_name = ir_layer.inputs[idx]
                if tensor_name:
                    self.state_tensors[tensor_name] = "state"
                    logger.debug(
                        f"{op_type} '{layer_dict['name']}': "
                        f"marked {state_name} = '{tensor_name}' as state"
                    )

    def _detect_control_flow_state(self, op_type: str, layer_dict: Dict) -> None:
        """
        Extract state from Loop/Scan operators.

        Loop and Scan operators have subgraphs that define body computation.
        State is carried across iterations via loop variables.

        TODO: Requires subgraph analysis to extract state tensors from body.
        For now, this is logged for future implementation.

        Args:
            op_type: Operator type ("Loop", "Scan", or "If")
            layer_dict: Original ONNX layer dictionary
        """
        logger.debug(f"{op_type} state detection requires subgraph analysis (TODO)")


def detect_state_tensors(
    analyzer: ONNXModel,
    layers: Dict[str, BaseLayer],
) -> Dict[str, str]:
    """
    Convenience function for detecting state tensors.

    High-level API that matches existing usage patterns.

    Args:
        analyzer: ONNXModel instance
        layers: Dictionary of IR layers

    Returns:
        Dictionary mapping tensor_name -> "state" for state tensors

    Example:
        state_tensors = detect_state_tensors(analyzer, layers)
    """
    detector = StateDetector(analyzer, layers)
    return detector.detect_all()
