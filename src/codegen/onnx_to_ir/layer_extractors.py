"""
Layer extraction functions.
Convert enriched layer dicts to IR layer objects.
"""

import numpy as np
import logging
from typing import Dict
from ..types import *
from .shape_inference import (
    infer_layer_shapes,
    get_feature_sizes,
    validate_inferred_shapes,
)
from .weight_utils import extract_quantized_weight, validate_weight_quantization
from ..onnx_model import ONNXModel

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


# Registry of layer extractors
LAYER_EXTRACTORS = {
    "MatMul": extract_matmul_layer,
    "Add": extract_add_layer,
    "Gemm": extract_gemm_layer,
    "FusedGemm": extract_fused_gemm_layer,
    "Relu": extract_activation_layer,
    "Sigmoid": extract_activation_layer,
    "Tanh": extract_activation_layer,
    "Softmax": extract_activation_layer,
    "Reshape": extract_reshape_layer,
    "QuantizeLinear": extract_quantize_linear_layer,
    "DequantizeLinear": extract_dequantize_linear_layer,
    "Dropout": extract_dropout_layer,
    "Conv": extract_conv2d_layer,
    "MaxPool": extract_maxpool_layer,
    "AveragePool": extract_avgpool_layer,
    "GlobalAveragePool": extract_global_avgpool_layer,
    "Flatten": extract_flatten_layer,
    "Transpose": extract_transpose_layer,
    "BatchNormalization": extract_batchnorm_layer,
    "Squeeze": extract_squeeze_layer,
}
