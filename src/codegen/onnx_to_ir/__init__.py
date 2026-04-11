"""
ONNX to Intermediate Representation (IR) conversion module.

This module converts ONNX models into the ml2plc intermediate representation (IR),
which is optimized for PLC code generation.

Key Design Principles:
-----------------------

1. **State Detection at Conversion Time**
   Rather than trying to infer state information later during regionization,
   we detect it early in the converter. ONNX operators like LSTM, GRU, and Scan
   have well-defined semantics about which inputs/outputs represent state.
   This information is extracted and stored in NetworkIR.state_tensors.

2. **Ground Truth from ONNX Specifications**
   Each RNN operator (LSTM, GRU, RNN, Scan) has a dedicated state detector that
   follows ONNX spec. This eliminates heuristic guessing and makes the code
   self-documenting and maintainable.

3. **Graceful Fallback for Topology Analysis**
   If annotated state information is available, regionization uses it. Otherwise,
   it falls back to topology-based back-edge analysis. This provides robustness
   for older or partially-supported ONNX operators.

Pipeline:
----------
1. ONNXModel.load_model() → Loads and validates ONNX file
2. onnx_to_ir() → Converts to NetworkIR with detected state tensors
3. regionize_network_ir() → Partitions into typed regions (acyclic/recurrent/loop)
4. Each region can then be lowered to PLC code

Example:
--------
    analyzer = ONNXModel("model.onnx")
    analyzer.load_model()
    network_ir = onnx_to_ir(analyzer)
    # network_ir.state_tensors now contains {"h_0": "state", "c_0": "state", ...}
    model_ir = regionize_network_ir(network_ir)
"""

from .converter import onnx_to_ir
from .regionizer import regionize_network_ir
from .tensor_resolution import TensorResolver, ResolvedTensor
from .shape import (
    infer_layer_shapes,
    get_feature_sizes,
    validate_model_shapes,
    ShapeValidationError,
)
from .layer_extractors import LAYER_EXTRACTORS

__all__ = [
    "onnx_to_ir",
    "regionize_network_ir",
    "TensorResolver",
    "ResolvedTensor",
    "infer_layer_shapes",
    "get_feature_sizes",
    "validate_model_shapes",
    "ShapeValidationError",
    "LAYER_EXTRACTORS",
]
