"""
Type descriptions related to the neural network.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class ActivationType(Enum):
    """Types of activation functions supported"""

    NONE = "none"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


@dataclass(frozen=True, kw_only=True)
class BaseLayer:
    """Base class for all layers"""

    layer_id: int
    name: str
    op_type: str
    input_size: int
    output_size: int
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()

    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class ActivationLayer(BaseLayer):
    """Represents an activation function layer"""

    activation: ActivationType


@dataclass(frozen=True, kw_only=True)
class DropoutLayer(BaseLayer):
    """Represents a Dropout layer"""

    ratio: float = 0.5


@dataclass(frozen=True, kw_only=True)
class LinearLayer(BaseLayer):
    """Base class for layers with weights and biases"""

    weights: np.ndarray
    bias: Optional[np.ndarray] = None  # Some linear layers may have bias

    # Quantization metadata (None if weights are not quantized)
    weight_scale: Optional[np.ndarray] = None
    weight_zero_point: Optional[np.ndarray] = None

    def is_quantized(self) -> bool:
        """Check if this layer has quantized weights."""
        return self.weight_scale is not None

    def is_per_tensor_quantized(self) -> bool:
        """Check if quantization is per-tensor (vs per-channel)."""
        return self.is_quantized() and self.weight_scale.size == 1


@dataclass(frozen=True, kw_only=True)
class MatMulLayer(LinearLayer):
    """Y = X * W"""

    pass


@dataclass(frozen=True, kw_only=True)
class GemmLayer(LinearLayer):
    """Y = alpha * X * W + beta * B"""

    alpha: float = 1.0
    beta: float = 1.0
    transA: bool = False
    transB: bool = False


@dataclass(frozen=True, kw_only=True)
class FusedLinearLayer(LinearLayer):
    """
    This represents a fully-fused dense layer operation combining:
    - Matrix multiplication (weights @ input)
    - Bias addition
    - Optional activation function
    """

    activation: ActivationType


@dataclass(frozen=True, kw_only=True)
class FusedGemmLayer(FusedLinearLayer):
    """Base class for Fused Gemm + Activation layers"""

    alpha: float = 1.0
    beta: float = 1.0
    transA: bool = False
    transB: bool = False


@dataclass(frozen=True, kw_only=True)
class AddLayer(BaseLayer):
    """Represents an ONNX Add layer"""

    bias: np.ndarray


@dataclass(frozen=True, kw_only=True)
class ReshapeLayer(BaseLayer):
    """Represents an ONNX Reshape layer"""

    pass


@dataclass(frozen=True, kw_only=True)
class QuantizeLinearLayer(BaseLayer):
    """Represents an ONNX QuantizeLinear layer"""

    scale: np.ndarray
    zero_point: np.ndarray
    axis: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class DequantizeLinearLayer(BaseLayer):
    """Represents an ONNX DequantizeLinear layer"""

    scale: np.ndarray
    zero_point: np.ndarray
    axis: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class Conv2DLayer(LinearLayer):
    """Represents a 2D Convolution layer: Y = Conv(X, W) + B"""

    kernel_shape: Tuple[int, int]  # (kH, kW)
    strides: Tuple[int, int] = (1, 1)
    pads: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (top, left, bottom, right)
    dilations: Tuple[int, int] = (1, 1)
    group: int = 1
    # input_shape/output_shape from BaseLayer carry (C,H,W) or (N,C,H,W)


@dataclass(frozen=True, kw_only=True)
class Pool2DLayer(BaseLayer):
    """Represents a 2D Pooling layer (Max or Average)"""

    pool_type: str  # "max" or "avg"
    kernel_shape: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    pads: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True, kw_only=True)
class FlattenLayer(BaseLayer):
    """Reshape from N-D to 1-D (bridges Conv→Dense)"""

    axis: int = 1  # ONNX default: flatten from axis=1


@dataclass(frozen=True, kw_only=True)
class TransposeLayer(BaseLayer):
    """Represents an ONNX Transpose layer that permutes tensor dimensions."""

    perm: Tuple[int, ...]  # Permutation of dimensions, e.g. (0, 2, 3, 1) for NCHW -> NHWC


@dataclass(frozen=True)
class NetworkIR:
    """Intermediate graph-based representation of a neural network"""

    # layer_name -> layer
    layers: Dict[str, BaseLayer]

    # List of layer names in execution order (topological sort)
    execution_order: List[str]

    # tensor_name -> layer_name
    tensor_producers: Dict[str, str] = field(default_factory=dict)

    # tensor_name -> [layer_names]
    tensor_consumers: Dict[str, List[str]] = field(default_factory=dict)

    input_tensors: Tuple[str, ...] = ()
    output_tensors: Tuple[str, ...] = ()

    def get_layer(self, name: str) -> BaseLayer:
        """Get layer by name"""
        return self.layers[name]

    def get_input_layers(self, layer_name: str) -> List[str]:
        """Get names of layers that produce inputs for the given layer"""
        return [
            self.tensor_producers[tensor_name]
            for tensor_name in self.get_layer(layer_name).inputs
            if tensor_name in self.tensor_producers
        ]

    def get_output_layers(self, layer_name: str) -> List[str]:
        """Get names of layers that consume outputs from the given layer"""
        return [
            consumer
            for tensor_name in self.get_layer(layer_name).outputs
            if tensor_name in self.tensor_consumers
            for consumer in self.tensor_consumers[tensor_name]
        ]

    def is_network_input(self, tensor_name: str) -> bool:
        """Check if a tensor is a network input"""
        return tensor_name in self.input_tensors

    def is_network_output(self, tensor_name: str) -> bool:
        """Check if a tensor is a network output"""
        return tensor_name in self.output_tensors

    def __str__(self) -> str:
        layer_types = [type(layer).__name__ for layer in self.layers.values()]
        layers_str = "\n  ".join(layer_types)
        return (
            f"NetworkIR(layers={len(self.layers)})\n"
            f"Layer types (in order):\n  {layers_str}"
        )
