"""
Layer extraction functions.
Convert enriched layer dicts to IR layer objects.
"""

import numpy as np
import logging
from typing import Dict, List
from ..types import *
from .shape_inference import (
    infer_layer_shapes,
    get_feature_sizes,
    validate_inferred_shapes,
)
from .weight_utils import extract_quantized_weight, validate_weight_quantization
from ..onnx_model import ONNXModel
import onnx
from onnx import TensorProto

logger = logging.getLogger(__name__)


def extract_activation_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> ActivationLayer:
    """Extract activation layer."""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]
    activation_type = (
        layer["op_type"].upper()
        if layer["op_type"].upper() in ActivationType.__members__
        else "NONE"
    )

    return ActivationLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        activation=ActivationType[activation_type],
        input_size=inputs[0].size,
        output_size=outputs[0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=inputs[0].shape,
        output_shape=outputs[0].shape,
        input_type=inputs[0].dtype,
        output_type=outputs[0].dtype,
    )


def extract_add_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> AddLayer:
    """Extract Add layer (element-wise addition with optional constant)."""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]

    # Check if second input is a constant (bias/weight)
    bias = inputs[1].value if inputs[1].is_weight else None

    # For bias addition, only pass the tensor input
    # For element-wise, pass both tensor inputs
    layer_inputs = (
        (inputs[0].name,) if bias is not None else tuple(t.name for t in inputs)
    )

    return AddLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=outputs[0].size,
        inputs=layer_inputs,
        outputs=tuple(t.name for t in outputs),
        input_shape=inputs[0].shape,
        output_shape=outputs[0].shape,
        input_type=inputs[0].dtype,
        output_type=outputs[0].dtype,
        bias=bias,  # None for element-wise, array for bias addition
    )


def extract_matmul_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> MatMulLayer:
    """Extract MatMul layer."""
    inputs = layer["resolved_inputs"]

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    validate_inferred_shapes(
        layer["name"], "MatMul", input_shape, output_shape, inputs[1].shape
    )

    weights, scale, zero_point = extract_quantized_weight(
        inputs[1].name, analyzer.layers, analyzer.weights
    )

    validate_weight_quantization(weights, scale, zero_point, input_size, output_size)

    return MatMulLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        weight_scale=scale,
        weight_zero_point=zero_point,
    )


def extract_gemm_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> GemmLayer:
    """Extract Gemm layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    validate_inferred_shapes(
        layer["name"], "Gemm", input_shape, output_shape, inputs[1].shape
    )

    weights, scale, zero_point = extract_quantized_weight(
        inputs[1].name, analyzer.layers, analyzer.weights
    )

    validate_weight_quantization(weights, scale, zero_point, input_size, output_size)

    return GemmLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        bias=inputs[2].value if len(inputs) > 2 and inputs[2].is_weight else None,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", 0) == 1,
        transB=attrs.get("transB", 0) == 1,
        weight_scale=scale,
        weight_zero_point=zero_point,
    )


def extract_fused_gemm_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> FusedGemmLayer:
    """Extract FusedGemm layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    validate_inferred_shapes(
        layer["name"], "FusedGemm", input_shape, output_shape, inputs[1].shape
    )

    weights, scale, zero_point = extract_quantized_weight(
        inputs[1].name, analyzer.layers, analyzer.weights
    )

    validate_weight_quantization(weights, scale, zero_point, input_size, output_size)

    return FusedGemmLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        activation=ActivationType[attrs.get("activation", "RELU").upper()],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        bias=inputs[2].value if len(inputs) > 2 and inputs[2].is_weight else None,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", 0) == 1,
        transB=attrs.get("transB", 0) == 1,
        weight_scale=scale,
        weight_zero_point=zero_point,
    )


def extract_reshape_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> ReshapeLayer:
    """Extract Reshape layer."""
    inputs = layer["resolved_inputs"]

    if not inputs[1].is_weight or inputs[1].value is None:
        raise ValueError(f"Reshape layer {layer_id}: Target shape must be constant")

    target_shape = tuple(int(dim) for dim in inputs[1].value)
    input_size = inputs[0].size

    # Handle -1 in target shape
    if -1 in target_shape:
        known_dims = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known_dims)) if known_dims else 1
        inferred = input_size // known_prod
        target_shape = tuple(inferred if d == -1 else d for d in target_shape)

    output_size = int(np.prod(target_shape))

    if output_size != input_size:
        raise ValueError(
            f"Reshape {layer_id}: Size mismatch - "
            f"input {input_size} != output {output_size}"
        )

    return ReshapeLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=inputs[0].shape,
        output_shape=target_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
    )


def extract_quantize_linear_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> QuantizeLinearLayer:
    """Extract QuantizeLinear layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    if not inputs[1].is_weight or inputs[1].value is None:
        raise ValueError(f"QuantizeLinear {layer_id} missing scale")

    scale = inputs[1].value
    zero_point = (
        inputs[2].value
        if len(inputs) > 2 and inputs[2].value is not None
        else np.array([0])
    )

    return QuantizeLinearLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=layer["resolved_outputs"][0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=inputs[0].shape,
        output_shape=layer["resolved_outputs"][0].shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        scale=scale,
        zero_point=zero_point,
        axis=attrs.get("axis"),
    )


def extract_dequantize_linear_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> DequantizeLinearLayer:
    """Extract DequantizeLinear layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    if not inputs[1].is_weight or inputs[1].value is None:
        raise ValueError(f"DequantizeLinear {layer_id} missing scale")

    scale = inputs[1].value
    zero_point = (
        inputs[2].value
        if len(inputs) > 2 and inputs[2].value is not None
        else np.array([0])
    )

    return DequantizeLinearLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=layer["resolved_outputs"][0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=inputs[0].shape,
        output_shape=layer["resolved_outputs"][0].shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        scale=scale,
        zero_point=zero_point,
        axis=attrs.get("axis"),
    )


def extract_dropout_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> DropoutLayer:
    """
    Extract Dropout layer.

    Note: Dropout is only active during training. At inference time,
    it acts as an identity/pass-through operation.
    """
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]
    attrs = layer.get("attributes", {})

    ratio = attrs.get("ratio", 0.5)

    return DropoutLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=outputs[0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=inputs[0].shape,
        output_shape=outputs[0].shape,
        input_type=inputs[0].dtype,
        output_type=outputs[0].dtype,
        ratio=ratio,
    )


def extract_conv2d_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> Conv2DLayer:
    """Extract Conv2D layer from ONNX Conv node."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    # Weight tensor: (out_channels, in_channels/groups, kH, kW)
    weight_tensor = inputs[1]
    weights = weight_tensor.value
    if weights is None:
        raise ValueError(f"Conv layer {layer_id}: weight tensor must be constant")

    bias = None
    if len(inputs) > 2 and inputs[2].is_weight and inputs[2].value is not None:
        bias = inputs[2].value

    kernel_shape = tuple(attrs.get("kernel_shape", list(weights.shape[2:])))
    strides = tuple(attrs.get("strides", [1, 1]))
    pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
    dilations = tuple(attrs.get("dilations", [1, 1]))
    group = attrs.get("group", 1)

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    return Conv2DLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        bias=bias,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group,
    )


def extract_pool2d_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel, pool_type: str
) -> Pool2DLayer:
    """Extract MaxPool or AveragePool layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    kernel_shape = tuple(attrs.get("kernel_shape", [2, 2]))
    strides = tuple(
        attrs.get("strides", kernel_shape)
    )  # ONNX default: strides = kernel_shape
    pads = tuple(attrs.get("pads", [0, 0, 0, 0]))

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    return Pool2DLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        pool_type=pool_type,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
    )


def extract_maxpool_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> Pool2DLayer:
    """Extract MaxPool layer."""
    return extract_pool2d_layer(layer, layer_id, analyzer, pool_type="max")


def extract_avgpool_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> Pool2DLayer:
    """Extract AveragePool layer."""
    return extract_pool2d_layer(layer, layer_id, analyzer, pool_type="avg")


def extract_global_avgpool_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> Pool2DLayer:
    """
    Extract GlobalAveragePool layer.

    GlobalAveragePool pools each channel over the entire spatial extent,
    so kernel_shape = (H, W) of the input.
    """
    inputs = layer["resolved_inputs"]
    input_shape, output_shape = infer_layer_shapes(layer)

    h_in = input_shape[-2] if len(input_shape) >= 2 else 1
    w_in = input_shape[-1] if len(input_shape) >= 1 else 1

    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    return Pool2DLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        pool_type="avg",
        kernel_shape=(h_in, w_in),
        strides=(h_in, w_in),
        pads=(0, 0, 0, 0),
    )


def extract_flatten_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> FlattenLayer:
    """Extract Flatten layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})
    axis = attrs.get("axis", 1)

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    return FlattenLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        axis=axis,
    )


def extract_transpose_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> TransposeLayer:
    """Extract Transpose layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})
    perm = tuple(attrs.get("perm", ()))

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    # Adjust perm for batch-stripped shapes (same logic as shape inference)
    if perm and len(perm) == len(input_shape) + 1:
        perm = tuple(p - 1 for p in perm if p != 0)

    return TransposeLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        perm=perm,
    )


def extract_batchnorm_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> "BatchNormLayer":
    """
    Extract BatchNormalization layer (inference mode).

    ONNX inputs: X, scale (γ), B (β), input_mean (μ), input_var (σ²)
    Attribute:   epsilon (default 1e-5)

    At inference:
        Y = γ * (X − μ) / sqrt(σ² + ε) + β

    We precompute per-channel parameters so the PLC only needs:
        Y[c] = combined_scale[c] * X[c] + combined_bias[c]
    """
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})
    epsilon = attrs.get("epsilon", 1e-5)

    # Validate constant inputs
    scale = inputs[1].value  # γ  — shape (C,)
    bias = inputs[2].value  # β  — shape (C,)
    mean = inputs[3].value  # μ  — shape (C,)
    var = inputs[4].value  # σ² — shape (C,)

    for idx, (name, val) in enumerate(
        [("scale", scale), ("bias", bias), ("mean", mean), ("var", var)]
    ):
        if val is None:
            raise ValueError(
                f"BatchNormalization layer {layer_id}: "
                f"input '{name}' (index {idx + 1}) must be a constant tensor"
            )

    num_channels = scale.shape[0]

    # Precompute combined parameters
    combined_scale = scale / np.sqrt(var + epsilon)
    combined_bias = bias - mean * combined_scale

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    return BatchNormLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=(inputs[0].name,),  # Only the data tensor is an edge input
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        num_channels=num_channels,
        combined_scale=combined_scale.astype(np.float32),
        combined_bias=combined_bias.astype(np.float32),
    )


def extract_squeeze_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> "SqueezeLayer":
    """
    Extract Squeeze layer.

    Squeeze removes dimensions of size 1.  In ONNX opset < 13 the axes
    come from an attribute; in opset >= 13 they come from a second constant
    input tensor.

    For the flat-array PLC representation this is essentially a no-op
    (same data, different logical shape).
    """
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    # Get axes — attribute (opset < 13) or constant input (opset >= 13)
    axes = tuple(attrs.get("axes", ()))
    if (
        not axes
        and len(inputs) > 1
        and inputs[1].is_weight
        and inputs[1].value is not None
    ):
        axes = tuple(int(a) for a in inputs[1].value)

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size = int(np.prod(input_shape)) if input_shape else 0
    output_size = int(np.prod(output_shape)) if output_shape else 0

    # Squeeze is a reshape — total element count must be preserved.
    # If they differ, it's a shape inference issue; use the consistent value.
    # TODO: If this is always an inference issue, it may be better to simply throw an exception! Fail fast!
    if input_size != output_size:
        raise ValueError(
            f"Squeeze '{layer['name']}': input_size ({input_size}) != output_size ({output_size}), "
            f"input_shape={input_shape}, output_shape={output_shape}. "
            f"Squeeze cannot change element count — this indicates a shape inference bug."
        )

    if input_size == 0:
        raise ValueError(
            f"Squeeze '{layer['name']}': computed input_size is 0, "
            f"input_shape={input_shape}, output_shape={output_shape}. "
            f"This indicates a shape inference failure."
        )

    # Adjust axes for batch-dim-stripped shapes
    if axes and any(a > 0 for a in axes):
        axes = tuple(a - 1 for a in axes if a != 0)

    return SqueezeLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=(inputs[0].name,),  # Only the data tensor
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        axes=axes,
    )


def extract_lstm_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> "LSTMLayer":
    """
    Extract LSTM (Long Short-Term Memory) layer.

    Per ONNX LSTM spec (opset 7+):
    - Inputs:  [X, W, R, B, sequence_lens, initial_h, initial_c, P]
    - Outputs: [Y, Y_h, Y_c]
    - Where: W (input weights), R (recurrent weights), B (bias), P (peephole)

    ONNX weight shapes (for forward-only direction):
    - W: (num_directions, 4*hidden_size, input_size)
    - R: (num_directions, 4*hidden_size, hidden_size)
    - B: (num_directions, 8*hidden_size)

    We strip the num_directions dimension and combine Wb + Rb biases.
    """
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    # Extract hidden_size from attributes
    hidden_size = attrs.get("hidden_size")
    if hidden_size is None:
        raise ValueError(
            f"LSTM layer '{layer['name']}': hidden_size attribute required"
        )

    # Input tensor properties
    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    # Extract weights from inputs
    # Input 0: X (sequence data)
    # Input 1: W (input weights)
    # Input 2: R (recurrent weights)
    # Input 3: B (bias, optional)
    # Input 4: sequence_lens (optional)
    # Input 5: initial_h (optional)
    # Input 6: initial_c (optional)
    # Input 7: P (peephole, optional)

    if len(inputs) < 3:
        raise ValueError(
            f"LSTM layer '{layer['name']}': requires at least 3 inputs (X, W, R), "
            f"got {len(inputs)}"
        )

    # Extract W (input weights)
    W_tensor = inputs[1]
    if not W_tensor.is_weight:
        raise ValueError(
            f"LSTM layer '{layer['name']}': input 1 (W) must be a weight tensor"
        )
    W = W_tensor.value
    if W is None:
        W = analyzer.weights.get(W_tensor.name)
    if W is None:
        raise ValueError(f"LSTM layer '{layer['name']}': weight W not found")

    # Extract R (recurrent weights)
    R_tensor = inputs[2]
    if not R_tensor.is_weight:
        raise ValueError(
            f"LSTM layer '{layer['name']}': input 2 (R) must be a weight tensor"
        )
    R = R_tensor.value
    if R is None:
        R = analyzer.weights.get(R_tensor.name)
    if R is None:
        raise ValueError(f"LSTM layer '{layer['name']}': weight R not found")

    # Extract B (bias, optional)
    B = None
    if len(inputs) > 3 and inputs[3].is_weight and inputs[3].value is not None:
        B = inputs[3].value

    # Extract P (peephole weights, optional)
    P = None
    if len(inputs) > 7 and inputs[7].is_weight and inputs[7].value is not None:
        P = inputs[7].value

    # ── Strip num_directions dimension ──────────────────────────────────────
    # ONNX W: (num_directions, 4*hidden_size, input_size) → (4*hidden_size, input_size)
    if W.ndim == 3:
        logger.debug(
            f"LSTM '{layer['name']}': stripping num_directions from W "
            f"{W.shape} -> {W[0].shape}"
        )
        W = W[0]

    # ONNX R: (num_directions, 4*hidden_size, hidden_size) → (4*hidden_size, hidden_size)
    if R.ndim == 3:
        logger.debug(
            f"LSTM '{layer['name']}': stripping num_directions from R "
            f"{R.shape} -> {R[0].shape}"
        )
        R = R[0]

    # ONNX B: (num_directions, 8*hidden_size) → (8*hidden_size,)
    if B is not None and B.ndim == 2:
        logger.debug(
            f"LSTM '{layer['name']}': stripping num_directions from B "
            f"{B.shape} -> {B[0].shape}"
        )
        B = B[0]

    # ── Combine Wb and Rb biases ────────────────────────────────────────────
    # ONNX B layout: [Wb_i, Wb_o, Wb_f, Wb_g, Rb_i, Rb_o, Rb_f, Rb_g]
    # Each sub-vector has length hidden_size.
    # For ST code generation we want a single bias per gate: b_gate = Wb_gate + Rb_gate
    # Result: (4*hidden_size,) with layout [b_i, b_o, b_f, b_g]
    if B is not None:
        if B.shape[0] == 8 * hidden_size:
            Wb = B[: 4 * hidden_size]
            Rb = B[4 * hidden_size :]
            B = Wb + Rb  # Element-wise addition
            logger.debug(
                f"LSTM '{layer['name']}': combined Wb+Rb biases -> shape {B.shape}"
            )
        elif B.shape[0] == 4 * hidden_size:
            logger.debug(
                f"LSTM '{layer['name']}': B already has 4*hidden_size elements"
            )
        else:
            logger.warning(
                f"LSTM '{layer['name']}': unexpected B shape {B.shape}, "
                f"expected ({8 * hidden_size},) or ({4 * hidden_size},)"
            )

    # ── Validate shapes ─────────────────────────────────────────────────────
    # Derive actual input_size from W (more reliable than shape inference for features)
    actual_input_size = W.shape[1]

    if W.shape != (4 * hidden_size, actual_input_size):
        raise ValueError(
            f"LSTM '{layer['name']}': W shape mismatch — "
            f"expected ({4 * hidden_size}, {actual_input_size}), got {W.shape}"
        )
    if R.shape != (4 * hidden_size, hidden_size):
        raise ValueError(
            f"LSTM '{layer['name']}': R shape mismatch — "
            f"expected ({4 * hidden_size}, {hidden_size}), got {R.shape}"
        )

    # ── Extract sequence_length from input shape ────────────────────────────
    # After batch-dim stripping, input_shape is typically (seq_length, input_size)
    # or just (seq_length,) if input_size == 1.
    if input_shape and len(input_shape) >= 2:
        sequence_length = input_shape[0]
    elif input_shape and len(input_shape) == 1:
        # Flat input: seq_length = total_size / input_size
        sequence_length = (
            input_shape[0] // actual_input_size if actual_input_size > 0 else 1
        )
    else:
        logger.warning(
            f"LSTM '{layer['name']}': could not determine sequence_length "
            f"from input_shape {input_shape}, defaulting to 1"
        )
        sequence_length = 1

    logger.info(
        f"LSTM '{layer['name']}': seq_len={sequence_length}, "
        f"input_size={actual_input_size}, hidden_size={hidden_size}, "
        f"W={W.shape}, R={R.shape}, B={'None' if B is None else B.shape}"
    )

    # Extract activations
    activations = tuple(attrs.get("activations", ["Sigmoid", "Tanh", "Tanh"]))
    direction = attrs.get("direction", "forward")
    clip = attrs.get("clip")

    return LSTMLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=actual_input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        W=W,
        R=R,
        B=B,
        P=P,
        activations=activations,
        direction=direction,
        clip=clip,
    )


def extract_gru_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> "GRULayer":
    """
    Extract GRU (Gated Recurrent Unit) layer.

    Per ONNX GRU spec (opset 7+):
    - Inputs:  [X, W, R, B, sequence_lens, initial_h]
    - Outputs: [Y, Y_h]
    - Where: W (input weights), R (recurrent weights), B (bias)

    Similar to LSTM but simpler (no cell state, 3 gates instead of 4).
    """
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    # Extract hidden_size from attributes
    hidden_size = attrs.get("hidden_size")
    if hidden_size is None:
        raise ValueError(f"GRU layer '{layer['name']}': hidden_size attribute required")

    # Input tensor properties
    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    if len(inputs) < 3:
        raise ValueError(
            f"GRU layer '{layer['name']}': requires at least 3 inputs (X, W, R), "
            f"got {len(inputs)}"
        )

    # Extract W (input weights)
    W_tensor = inputs[1]
    if not W_tensor.is_weight:
        raise ValueError(
            f"GRU layer '{layer['name']}': input 1 (W) must be a weight tensor"
        )
    W = W_tensor.value
    if W is None:
        W = analyzer.weights.get(W_tensor.name)
    if W is None:
        raise ValueError(f"GRU layer '{layer['name']}': weight W not found")

    # Extract R (recurrent weights)
    R_tensor = inputs[2]
    if not R_tensor.is_weight:
        raise ValueError(
            f"GRU layer '{layer['name']}': input 2 (R) must be a weight tensor"
        )
    R = R_tensor.value
    if R is None:
        R = analyzer.weights.get(R_tensor.name)
    if R is None:
        raise ValueError(f"GRU layer '{layer['name']}': weight R not found")

    # Extract B (bias, optional)
    B = None
    if len(inputs) > 3 and inputs[3].is_weight and inputs[3].value is not None:
        B = inputs[3].value

    # Extract activations
    activations = tuple(attrs.get("activations", ["Sigmoid", "Tanh"]))
    direction = attrs.get("direction", "forward")
    clip = attrs.get("clip")

    return GRULayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        hidden_size=hidden_size,
        W=W,
        R=R,
        B=B,
        activations=activations,
        direction=direction,
        clip=clip,
    )


# ============================================================================
# Data-movement / shape-manipulation extractors
# ============================================================================


def _extract_cast_layer(enriched_layer: Dict, layer_id: int, analyzer) -> CastLayer:
    """Extract Cast layer — element-wise type conversion."""
    attrs = enriched_layer.get("attributes", {})
    to_type = attrs.get("to", TensorProto.FLOAT)
    np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(to_type, np.float32)
    target_type_str = str(np_dtype)

    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    input_size = resolved_in[0].size if resolved_in else 1
    output_size = resolved_out[0].size if resolved_out else input_size

    input_dtype = resolved_in[0].dtype if resolved_in else None
    output_dtype = target_type_str

    return CastLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"cast_{layer_id}",
        op_type="Cast",
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(enriched_layer["inputs"]),
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=(
            tuple(resolved_in[0].shape)
            if resolved_in and resolved_in[0].shape
            else None
        ),
        output_shape=(
            tuple(resolved_out[0].shape)
            if resolved_out and resolved_out[0].shape
            else None
        ),
        input_type=input_dtype,
        output_type=output_dtype,
        target_type=target_type_str,
    )


def _extract_slice_layer(enriched_layer: Dict, layer_id: int, analyzer) -> SliceLayer:
    """Extract Slice layer — sub-tensor extraction along axes."""
    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    # Slice has up to 5 inputs: data, starts, ends, axes, steps
    # starts/ends/axes/steps are typically constants already resolved
    def _get_const_list(idx: int, default):
        if idx < len(resolved_in) and resolved_in[idx].value is not None:
            return resolved_in[idx].value.flatten().tolist()
        return default

    starts = _get_const_list(1, [0])
    ends = _get_const_list(2, [2**31])
    axes = _get_const_list(3, list(range(len(starts))))
    steps = _get_const_list(4, [1] * len(starts))

    data_input = resolved_in[0] if resolved_in else None
    input_size = data_input.size if data_input else 1
    output_size = resolved_out[0].size if resolved_out else input_size

    # Only the data tensor is a runtime input; the rest are parameters
    runtime_inputs = (enriched_layer["inputs"][0],) if enriched_layer["inputs"] else ()

    return SliceLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"slice_{layer_id}",
        op_type="Slice",
        input_size=input_size,
        output_size=output_size,
        inputs=runtime_inputs,
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=(
            tuple(data_input.shape) if data_input and data_input.shape else None
        ),
        output_shape=(
            tuple(resolved_out[0].shape)
            if resolved_out and resolved_out[0].shape
            else None
        ),
        input_type=data_input.dtype if data_input else None,
        output_type=resolved_out[0].dtype if resolved_out else None,
        starts=[int(s) for s in starts],
        ends=[int(e) for e in ends],
        axes=[int(a) for a in axes],
        steps=[int(s) for s in steps],
    )


def _extract_concat_layer(enriched_layer: Dict, layer_id: int, analyzer) -> ConcatLayer:
    """Extract Concat layer — concatenation along an axis."""
    attrs = enriched_layer.get("attributes", {})
    axis = attrs.get("axis", 0)

    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    input_sizes = [ri.size for ri in resolved_in]
    total_size = sum(input_sizes)
    output_size = resolved_out[0].size if resolved_out else total_size

    return ConcatLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"concat_{layer_id}",
        op_type="Concat",
        input_size=input_sizes[0] if input_sizes else 0,
        output_size=output_size,
        inputs=tuple(enriched_layer["inputs"]),
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=(
            tuple(resolved_in[0].shape)
            if resolved_in and resolved_in[0].shape
            else None
        ),
        output_shape=(
            tuple(resolved_out[0].shape)
            if resolved_out and resolved_out[0].shape
            else None
        ),
        input_type=resolved_in[0].dtype if resolved_in else None,
        output_type=resolved_out[0].dtype if resolved_out else None,
        axis=axis,
        input_sizes=input_sizes,
    )


def _extract_unsqueeze_layer(
    enriched_layer: Dict, layer_id: int, analyzer
) -> UnsqueezeLayer:
    """Extract Unsqueeze layer — insert size-1 dimensions (identity on flat data)."""
    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    # Axes from second input (opset 13+) or from attribute (opset < 13)
    if len(resolved_in) > 1 and resolved_in[1].value is not None:
        axes = resolved_in[1].value.flatten().tolist()
    else:
        attrs = enriched_layer.get("attributes", {})
        axes = attrs.get("axes", [])

    data_input = resolved_in[0] if resolved_in else None
    input_size = data_input.size if data_input else 1

    # Only the data tensor is a runtime input
    runtime_inputs = (enriched_layer["inputs"][0],) if enriched_layer["inputs"] else ()

    return UnsqueezeLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"unsqueeze_{layer_id}",
        op_type="Unsqueeze",
        input_size=input_size,
        output_size=input_size,  # same element count
        inputs=runtime_inputs,
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=(
            tuple(data_input.shape) if data_input and data_input.shape else None
        ),
        output_shape=(
            tuple(resolved_out[0].shape)
            if resolved_out and resolved_out[0].shape
            else None
        ),
        input_type=data_input.dtype if data_input else None,
        output_type=data_input.dtype if data_input else None,
        unsqueeze_axes=[int(a) for a in axes],
    )


def _extract_expand_layer(enriched_layer: Dict, layer_id: int, analyzer) -> ExpandLayer:
    """Extract Expand layer — broadcast tensor to a larger shape."""
    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    # Target shape from second input (always a constant)
    if len(resolved_in) > 1 and resolved_in[1].value is not None:
        target_shape = tuple(int(s) for s in resolved_in[1].value.flatten().tolist())
        output_size = int(np.prod(target_shape)) if target_shape else 1
    else:
        target_shape = ()
        output_size = resolved_out[0].size if resolved_out else 1

    data_input = resolved_in[0] if resolved_in else None
    input_size = data_input.size if data_input else 1

    # Only the data tensor is a runtime input
    runtime_inputs = (enriched_layer["inputs"][0],) if enriched_layer["inputs"] else ()

    return ExpandLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"expand_{layer_id}",
        op_type="Expand",
        input_size=input_size,
        output_size=output_size,
        inputs=runtime_inputs,
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=(
            tuple(data_input.shape) if data_input and data_input.shape else None
        ),
        output_shape=(
            tuple(resolved_out[0].shape)
            if resolved_out and resolved_out[0].shape
            else None
        ),
        input_type=data_input.dtype if data_input else None,
        output_type=data_input.dtype if data_input else None,
        target_shape=target_shape,
    )


def _extract_gather_layer(enriched_layer: Dict, layer_id: int, analyzer) -> GatherLayer:
    """Extract Gather layer — index into a tensor along an axis."""
    attrs = enriched_layer.get("attributes", {})
    axis = attrs.get("axis", 0)

    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    # Indices from second input (often a constant)
    indices = None
    if len(resolved_in) > 1 and resolved_in[1].value is not None:
        indices = resolved_in[1].value

    data_input = resolved_in[0] if resolved_in else None
    input_size = data_input.size if data_input else 1
    output_size = resolved_out[0].size if resolved_out else 1

    # Only the data tensor is a runtime input if indices are constant
    if indices is not None:
        runtime_inputs = (enriched_layer["inputs"][0],)
    else:
        runtime_inputs = tuple(enriched_layer["inputs"][:2])

    return GatherLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"gather_{layer_id}",
        op_type="Gather",
        input_size=input_size,
        output_size=output_size,
        inputs=runtime_inputs,
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=(
            tuple(data_input.shape) if data_input and data_input.shape else None
        ),
        output_shape=(
            tuple(resolved_out[0].shape)
            if resolved_out and resolved_out[0].shape
            else None
        ),
        input_type=data_input.dtype if data_input else None,
        output_type=data_input.dtype if data_input else None,
        gather_axis=axis,
        indices=indices,
    )


def _extract_shape_layer(enriched_layer: Dict, layer_id: int, analyzer) -> BaseLayer:
    """
    Extract Shape layer.

    Shape should almost always be constant-folded. If we reach here, the input
    has a known static shape from shape inference, so we emit a no-op layer.
    """
    resolved_in = enriched_layer["resolved_inputs"]
    resolved_out = enriched_layer["resolved_outputs"]

    input_shape = (
        tuple(resolved_in[0].shape) if resolved_in and resolved_in[0].shape else ()
    )
    # The output is a 1-D tensor of the shape values
    output_size = len(input_shape)

    return BaseLayer(
        layer_id=layer_id,
        name=enriched_layer.get("name") or f"shape_{layer_id}",
        op_type="Shape",
        input_size=resolved_in[0].size if resolved_in else 0,
        output_size=output_size,
        inputs=tuple(enriched_layer["inputs"]),
        outputs=tuple(enriched_layer["outputs"]),
        input_shape=input_shape,
        output_shape=(output_size,),
        input_type=resolved_in[0].dtype if resolved_in else None,
        output_type="int64",
    )


# ============================================================================
# Layer Extractor Registry
# ============================================================================

LAYER_EXTRACTORS = {
    # Core compute layers
    "MatMul": extract_matmul_layer,
    "Add": extract_add_layer,
    "Gemm": extract_gemm_layer,
    "FusedGemm": extract_fused_gemm_layer,
    # Activation layers
    "Relu": extract_activation_layer,
    "Sigmoid": extract_activation_layer,
    "Tanh": extract_activation_layer,
    "Softmax": extract_activation_layer,
    # Shape/layout layers
    "Reshape": extract_reshape_layer,
    "Flatten": extract_flatten_layer,
    "Transpose": extract_transpose_layer,
    "Squeeze": extract_squeeze_layer,
    # Quantization layers
    "QuantizeLinear": extract_quantize_linear_layer,
    "DequantizeLinear": extract_dequantize_linear_layer,
    # Regularization layers
    "Dropout": extract_dropout_layer,
    # Convolution / Pooling layers
    "Conv": extract_conv2d_layer,
    "MaxPool": extract_maxpool_layer,
    "AveragePool": extract_avgpool_layer,
    "GlobalAveragePool": extract_global_avgpool_layer,
    # Normalization layers
    "BatchNormalization": extract_batchnorm_layer,
    # RNN layers
    "LSTM": extract_lstm_layer,
    "GRU": extract_gru_layer,
    # Shape-manipulation layers (runtime fallbacks for ops not constant-folded)
    "Shape": _extract_shape_layer,
    "Cast": _extract_cast_layer,
    "Slice": _extract_slice_layer,
    "Concat": _extract_concat_layer,
    "Unsqueeze": _extract_unsqueeze_layer,
    "Expand": _extract_expand_layer,
    "Gather": _extract_gather_layer,
}
