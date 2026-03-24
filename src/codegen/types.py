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


class RegionKind(Enum):
    """Execution region kinds for structured model planning."""

    ACYCLIC = "acyclic"
    RECURRENT = "recurrent"
    LOOP = "loop"
    UNSUPPORTED = "unsupported"


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
class SqueezeLayer(BaseLayer):
    """
    Represents an ONNX Squeeze layer that removes dimensions of size 1.

    E.g. (C, 1, 1) → (C,) after GlobalAveragePool before a Dense layer.
    In flat-array PLC code this is a no-op (same data, different logical shape).
    """

    axes: Tuple[int, ...]  # Which dimensions to squeeze (already batch-adjusted)


@dataclass(frozen=True, kw_only=True)
class TransposeLayer(BaseLayer):
    """Represents an ONNX Transpose layer that permutes tensor dimensions."""

    perm: Tuple[
        int, ...
    ]  # Permutation of dimensions, e.g. (0, 2, 3, 1) for NCHW -> NHWC


@dataclass(frozen=True, kw_only=True)
class BatchNormLayer(BaseLayer):
    """
    Represents a BatchNormalization layer (inference mode).

    During inference BN is a per-channel affine transform:
        Y[c] = scale[c] * (X[c] - mean[c]) / sqrt(var[c] + eps) + bias[c]

    We precompute combined_scale and combined_bias at compile time so the
    PLC only needs:  Y[c] = combined_scale[c] * X[c] + combined_bias[c]
    """

    num_channels: int
    combined_scale: np.ndarray  # shape (C,)  — precomputed γ / sqrt(σ² + ε)
    combined_bias: np.ndarray  # shape (C,)  — precomputed β − μ·combined_scale


@dataclass(frozen=True)
class NetworkIR:
    """
    Intermediate representation of a neural network graph.

    Canonical representation used throughout the pipeline:
    - After ONNX parsing (unoptimized full graph)
    - After optimization (refined graph)
    - As subgraphs within regions (region-local graphs)

    Attributes:
        layers: Dictionary mapping layer_name -> layer object
        execution_order: List of layer names in topological order
        tensor_producers: Mapping of tensor_name -> producing_layer_name
        tensor_consumers: Mapping of tensor_name -> [consuming_layer_names]
        input_tensors: Tuple of network input tensor names
        output_tensors: Tuple of network output tensor names
    """

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


@dataclass(frozen=True)
class RegionIR:
    """Base region contract for planning and optimization."""

    region_id: str
    kind: RegionKind
    graph: NetworkIR


@dataclass(frozen=True)
class AcyclicRegionIR(RegionIR):
    """Region representing an acyclic, topologically sorted subgraph."""

    def __post_init__(self):
        if self.kind != RegionKind.ACYCLIC:
            raise ValueError("AcyclicRegionIR.kind must be RegionKind.ACYCLIC")


@dataclass(frozen=True)
class RecurrentRegionIR(RegionIR):
    """Region representing recurrent step execution with explicit state tensors."""

    state_inputs: Tuple[str, ...] = ()
    state_outputs: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind != RegionKind.RECURRENT:
            raise ValueError("RecurrentRegionIR.kind must be RegionKind.RECURRENT")


@dataclass(frozen=True)
class LoopRegionIR(RegionIR):
    """Region representing control-flow loop semantics (e.g. ONNX Loop/Scan)."""

    loop_inputs: Tuple[str, ...] = ()
    loop_outputs: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind != RegionKind.LOOP:
            raise ValueError("LoopRegionIR.kind must be RegionKind.LOOP")


@dataclass(frozen=True)
class ModelIR:
    """Structured model representation as ordered execution regions."""

    regions: Tuple[RegionIR, ...]
    input_tensors: Tuple[str, ...] = ()
    output_tensors: Tuple[str, ...] = ()
    metadata: Dict[str, str] = field(default_factory=dict)

    def first_region(self) -> RegionIR:
        if not self.regions:
            raise ValueError("ModelIR has no regions")
        return self.regions[0]
