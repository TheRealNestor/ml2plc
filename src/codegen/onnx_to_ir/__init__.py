"""
ONNX to Intermediate Representation (IR) conversion module.
"""

from .converter import onnx_to_ir
from .regionizer import regionize_network_ir
from .tensor_resolution import TensorResolver, ResolvedTensor
from .shape_inference import infer_layer_shapes, get_feature_sizes
from .layer_extractors import LAYER_EXTRACTORS

__all__ = [
    "onnx_to_ir",
    "regionize_network_ir",
    "TensorResolver",
    "ResolvedTensor",
    "infer_layer_shapes",
    "get_feature_sizes",
    "LAYER_EXTRACTORS",
]
