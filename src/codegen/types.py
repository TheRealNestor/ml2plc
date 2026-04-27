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

    # State role for RNN/LSTM/GRU layers (e.g., "state_input", "state_output", None)
    state_role: Optional[str] = None

    # For multi-output layers: mapping of output tensor name to ONNX output index
    # E.g., for LSTM: {"Y": 0, "Y_h": 1, "Y_c": 2}
    output_indices: Dict[str, int] = field(default_factory=dict)

    # Which output this layer generates code for (primary/used output)
    # E.g., for LSTM used in sequence processing: "Y"
    primary_output: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class ActivationLayer(BaseLayer):
    """Represents an activation function layer"""

    activation: ActivationType


@dataclass(frozen=True, kw_only=True)
class DropoutLayer(BaseLayer):
    """Represents a Dropout layer"""

    ratio: float = 0.5


@dataclass(frozen=True, kw_only=True)
class CastLayer(BaseLayer):
    """Cast layer — converts tensor element types at runtime."""

    target_type: str = "float32"


@dataclass(frozen=True, kw_only=True)
class SliceLayer(BaseLayer):
    """Slice layer — extracts a sub-tensor along given axes at runtime."""

    starts: List[int] = field(default_factory=list)
    ends: List[int] = field(default_factory=list)
    axes: List[int] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ConcatLayer(BaseLayer):
    """Concat layer — concatenates multiple tensors along an axis at runtime."""

    axis: int = 0
    input_sizes: List[int] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class UnsqueezeLayer(BaseLayer):
    """Unsqueeze layer — inserts size-1 dimensions (identity on flat data)."""

    unsqueeze_axes: List[int] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ShapeLayer(BaseLayer):
    """Shape layer — extracts shape information from a tensor (int64 output)."""

    pass  # No special attributes needed


@dataclass(frozen=True, kw_only=True)
class ExpandLayer(BaseLayer):
    """Expand layer — broadcasts tensor to a larger shape at runtime."""

    target_shape: Tuple[int, ...] = ()


@dataclass(frozen=True, kw_only=True)
class GatherLayer(BaseLayer):
    """Gather layer — indexes into a tensor along an axis."""

    gather_axis: int = 0
    indices: Optional[np.ndarray] = None


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
class RuntimeMatMulLayer(BaseLayer):
    """MatMul where RHS is provided at runtime (non-constant tensor)."""

    rhs_shape: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True, kw_only=True)
class EinsumLayer(BaseLayer):
    """First-class Einsum support for selected static equations."""

    equation: str
    rhs_const: np.ndarray
    rhs_shape: Tuple[int, ...]


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

    bias: Optional[np.ndarray]


@dataclass(frozen=True, kw_only=True)
class ReduceMeanLayer(BaseLayer):
    """Represents ONNX ReduceMean with static axes/keepdims."""

    axes: Tuple[int, ...] = ()
    keepdims: bool = True


@dataclass(frozen=True, kw_only=True)
class ReduceProdLayer(BaseLayer):
    """Represents ONNX ReduceProd with static axes/keepdims."""

    axes: Tuple[int, ...] = ()
    keepdims: bool = True


@dataclass(frozen=True, kw_only=True)
class BinaryElementwiseLayer(BaseLayer):
    """Represents binary elementwise ops (Sub, Mul, Max, Add variants)."""

    operation: str
    rhs_const: Optional[np.ndarray] = None
    rhs_shape: Optional[Tuple[int, ...]] = None
    rhs_runtime_size: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class UnaryElementwiseLayer(BaseLayer):
    """Represents unary elementwise ops (Sqrt, Reciprocal, etc.)."""

    operation: str


@dataclass(frozen=True, kw_only=True)
class ArgMaxLayer(BaseLayer):
    """Represents an ArgMax reduction that returns the index of the maximum value.

    This compiler emits a single integer index (INT32/DINT) per ArgMax layer.
    """

    axis: int = -1


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


@dataclass(frozen=True, kw_only=True)
class LSTMLayer(BaseLayer):
    """
    Represents an ONNX LSTM layer (Long Short-Term Memory).

    LSTM performs recurrent computation with gates and cell state:
    - Input gate (i), Forget gate (f), Cell gate (g), Output gate (o)
    - Hidden state (h) and Cell state (c) carry-over between timesteps

    Per ONNX spec (opset 7+), the LSTM operator has:
    - Inputs: [X, W, R, B, sequence_lens, initial_h, initial_c, P]
    - Outputs: [Y, Y_h, Y_c]
    - Attributes: activation_alpha, activation_beta, activations, clip, direction, hidden_size

    The IR layer stores which ONNX outputs are used (via output_indices) and which is
    the primary output being generated (via primary_output).

    Attributes:
        hidden_size: Number of hidden units (h dimension)
        sequence_length: Length of input sequence
        W: Input weight matrix (num_directions, 4*hidden_size, input_size)
        R: Recurrent weight matrix (num_directions, 4*hidden_size, hidden_size)
        B: Bias vectors (num_directions, 8*hidden_size) [W_bias + R_bias]
        P: Peephole weights (num_directions, 3*hidden_size) [optional]
        activations: List of activation function types (default: ["Sigmoid", "Tanh", "Tanh"])
        direction: "forward", "reverse", or "bidirectional"
        clip: Optional clipping threshold for cell state
        output_indices: Dict mapping output tensor names to ONNX output indices
                        E.g., {"Y": 0, "Y_h": 1, "Y_c": 2} (inherited from BaseLayer)
        primary_output: Which output is generated (e.g., "Y" for full sequence)
                        (inherited from BaseLayer)
    """

    hidden_size: int
    sequence_length: int
    W: np.ndarray  # Input weight matrix
    R: np.ndarray  # Recurrent weight matrix
    B: Optional[np.ndarray] = None  # Bias (optional)
    P: Optional[np.ndarray] = None  # Peephole weights (optional)
    activations: Tuple[str, ...] = ("Sigmoid", "Tanh", "Tanh")  # (i, f, c) gates
    direction: str = "forward"
    clip: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class GRULayer(BaseLayer):
    """
    Represents an ONNX GRU layer (Gated Recurrent Unit).

    GRU is a simplified RNN variant with reset and update gates.
    Similar to LSTM but with fewer parameters (3 gates instead of 4).

    Per ONNX spec (opset 7+), the GRU operator has:
    - Inputs: [X, W, R, B, sequence_lens, initial_h]
    - Outputs: [Y, Y_h]
    - Attributes: activation_alpha, activation_beta, activations, clip, direction, hidden_size

    The IR layer stores which ONNX outputs are used (via output_indices) and which is
    the primary output being generated (via primary_output).

    Attributes:
        hidden_size: Number of hidden units (h dimension)
        sequence_length: Length of input sequence
        W: Input weight matrix (num_directions, 3*hidden_size, input_size)
        R: Recurrent weight matrix (num_directions, 3*hidden_size, hidden_size)
        B: Bias vectors (num_directions, 6*hidden_size) [optional]
        activations: List of activation function types (default: ["Sigmoid", "Tanh"])
        direction: "forward", "reverse", or "bidirectional"
        clip: Optional clipping threshold
        output_indices: Dict mapping output tensor names to ONNX output indices
                        E.g., {"Y": 0, "Y_h": 1} (inherited from BaseLayer)
        primary_output: Which output is generated (e.g., "Y" for full sequence)
                        (inherited from BaseLayer)
    """

    hidden_size: int
    sequence_length: int
    W: np.ndarray
    R: np.ndarray
    B: Optional[np.ndarray] = None
    activations: Tuple[str, ...] = ("Sigmoid", "Tanh")
    direction: str = "forward"
    clip: Optional[float] = None
    linear_before_reset: int = 0


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
        state_tensors: Mapping of state_tensor_name -> semantic role (e.g., "state")
                       Detected from RNN operators (LSTM, GRU, RNN, Scan, Loop)
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

    # Semantic state tensor information: tensor_name -> "state"
    # Populated by the ONNX converter when detecting RNN-family operators
    state_tensors: Dict[str, str] = field(default_factory=dict)

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
