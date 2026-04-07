"""
Shared test fixtures and helper functions.

Consolidates common test utilities used across multiple test modules.
This module provides factory functions for creating test layers of various types.

Each layer factory follows the pattern:
  - Sensible defaults for common cases
  - Customizable parameters for specific tests
  - Consistent naming and sizing conventions
"""

import numpy as np
from typing import Tuple, Optional, List

from codegen.types import (
    MatMulLayer,
    GemmLayer,
    AddLayer,
    ActivationLayer,
    Conv2DLayer,
    Pool2DLayer,
    ReshapeLayer,
    ActivationType,
    NetworkIR,
)


# ============================================================================
# Linear Layer Fixtures
# ============================================================================


def create_simple_matmul_layer(
    name: str,
    layer_id: int,
    input_size: int = 10,
    output_size: int = 5,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> MatMulLayer:
    """
    Create a simple MatMul layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        input_size: Input vector size
        output_size: Output vector size
        inputs: Tuple of input tensor names
        outputs: Tuple of output tensor names

    Returns:
        MatMulLayer instance with random weights
    """
    weights = np.random.randn(output_size, input_size).astype(np.float32)
    bias = np.random.randn(output_size).astype(np.float32)

    return MatMulLayer(
        layer_id=layer_id,
        name=name,
        op_type="MatMul",
        input_size=input_size,
        output_size=output_size,
        inputs=inputs or (f"{name}_input",),
        outputs=outputs or (f"{name}_output",),
        input_shape=(input_size,),
        output_shape=(output_size,),
        input_type="tensor(float)",
        output_type="tensor(float)",
        weights=weights,
        bias=bias,
    )


def create_gemm_layer(
    name: str,
    layer_id: int,
    input_size: int = 10,
    output_size: int = 5,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
    alpha: float = 1.0,
    beta: float = 1.0,
    transA: int = 0,
    transB: int = 0,
) -> GemmLayer:
    """
    Create a GEMM (General Matrix Multiply) layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        input_size: Input vector size
        output_size: Output vector size
        inputs: Tuple of input tensor names
        outputs: Tuple of output tensor names
        alpha: GEMM alpha parameter
        beta: GEMM beta parameter
        transA: Whether to transpose A
        transB: Whether to transpose B

    Returns:
        GemmLayer instance with random weights
    """
    weights = np.random.randn(output_size, input_size).astype(np.float32)
    bias = np.random.randn(output_size).astype(np.float32)

    return GemmLayer(
        layer_id=layer_id,
        name=name,
        op_type="Gemm",
        input_size=input_size,
        output_size=output_size,
        inputs=inputs or (f"{name}_input",),
        outputs=outputs or (f"{name}_output",),
        input_shape=(input_size,),
        output_shape=(output_size,),
        input_type="tensor(float)",
        output_type="tensor(float)",
        weights=weights,
        bias=bias,
        alpha=alpha,
        beta=beta,
        transA=transA,
        transB=transB,
    )


# ============================================================================
# Activation Layer Fixtures
# ============================================================================


def create_activation_layer(
    name: str,
    layer_id: int,
    activation_type: ActivationType = ActivationType.RELU,
    input_size: int = 10,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> ActivationLayer:
    """
    Create an activation layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        activation_type: Type of activation (RELU, SIGMOID, TANH, etc.)
        input_size: Size of input tensor
        inputs: Tuple of input tensor names
        outputs: Tuple of output tensor names

    Returns:
        ActivationLayer instance
    """
    return ActivationLayer(
        layer_id=layer_id,
        name=name,
        op_type=activation_type.name,
        activation_type=activation_type,
        inputs=inputs or (f"{name}_input",),
        outputs=outputs or (f"{name}_output",),
        input_shape=(input_size,),
        output_shape=(input_size,),
        input_type="tensor(float)",
        output_type="tensor(float)",
    )


def create_relu_layer(
    name: str,
    layer_id: int,
    input_size: int = 10,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> ActivationLayer:
    """Create a ReLU activation layer (convenience wrapper)."""
    return create_activation_layer(
        name, layer_id, ActivationType.RELU, input_size, inputs, outputs
    )


def create_sigmoid_layer(
    name: str,
    layer_id: int,
    input_size: int = 10,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> ActivationLayer:
    """Create a Sigmoid activation layer (convenience wrapper)."""
    return create_activation_layer(
        name, layer_id, ActivationType.SIGMOID, input_size, inputs, outputs
    )


def create_tanh_layer(
    name: str,
    layer_id: int,
    input_size: int = 10,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> ActivationLayer:
    """Create a Tanh activation layer (convenience wrapper)."""
    return create_activation_layer(
        name, layer_id, ActivationType.TANH, input_size, inputs, outputs
    )


# ============================================================================
# Convolutional and Pooling Layer Fixtures
# ============================================================================


def create_conv2d_layer(
    name: str,
    layer_id: int,
    input_channels: int = 3,
    output_channels: int = 16,
    kernel_h: int = 3,
    kernel_w: int = 3,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    input_h: int = 32,
    input_w: int = 32,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> Conv2DLayer:
    """
    Create a Conv2D layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        input_channels: Number of input channels
        output_channels: Number of output channels
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride_h: Stride in height dimension
        stride_w: Stride in width dimension
        pad_h: Padding in height dimension
        pad_w: Padding in width dimension
        input_h: Input height
        input_w: Input width
        inputs: Tuple of input tensor names
        outputs: Tuple of output tensor names

    Returns:
        Conv2DLayer instance with random weights
    """
    # Calculate output dimensions
    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    # Create random weights and bias
    weights = np.random.randn(
        output_channels, input_channels, kernel_h, kernel_w
    ).astype(np.float32)
    bias = np.random.randn(output_channels).astype(np.float32)

    return Conv2DLayer(
        layer_id=layer_id,
        name=name,
        op_type="Conv",
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        inputs=inputs or (f"{name}_input",),
        outputs=outputs or (f"{name}_output",),
        input_shape=(input_channels, input_h, input_w),
        output_shape=(output_channels, output_h, output_w),
        input_type="tensor(float)",
        output_type="tensor(float)",
        weights=weights,
        bias=bias,
    )


def create_pool2d_layer(
    name: str,
    layer_id: int,
    pool_type: str = "MaxPool",
    input_channels: int = 16,
    kernel_h: int = 2,
    kernel_w: int = 2,
    stride_h: int = 2,
    stride_w: int = 2,
    pad_h: int = 0,
    pad_w: int = 0,
    input_h: int = 32,
    input_w: int = 32,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> Pool2DLayer:
    """
    Create a Pool2D layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        pool_type: Type of pooling ("MaxPool", "AveragePool")
        input_channels: Number of input channels
        kernel_h: Kernel (pool size) height
        kernel_w: Kernel (pool size) width
        stride_h: Stride in height dimension
        stride_w: Stride in width dimension
        pad_h: Padding in height dimension
        pad_w: Padding in width dimension
        input_h: Input height
        input_w: Input width
        inputs: Tuple of input tensor names
        outputs: Tuple of output tensor names

    Returns:
        Pool2DLayer instance
    """
    # Calculate output dimensions
    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    return Pool2DLayer(
        layer_id=layer_id,
        name=name,
        op_type=pool_type,
        pool_type=pool_type,
        input_channels=input_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        inputs=inputs or (f"{name}_input",),
        outputs=outputs or (f"{name}_output",),
        input_shape=(input_channels, input_h, input_w),
        output_shape=(input_channels, output_h, output_w),
        input_type="tensor(float)",
        output_type="tensor(float)",
    )


# ============================================================================
# Element-wise Operation Layer Fixtures
# ============================================================================


def create_add_layer(
    name: str,
    layer_id: int,
    tensor_size: int = 10,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> AddLayer:
    """
    Create an Add (element-wise addition) layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        tensor_size: Size of tensors being added
        inputs: Tuple of input tensor names (should be 2 for binary addition)
        outputs: Tuple of output tensor names

    Returns:
        AddLayer instance
    """
    return AddLayer(
        layer_id=layer_id,
        name=name,
        op_type="Add",
        inputs=inputs or (f"{name}_a", f"{name}_b"),
        outputs=outputs or (f"{name}_output",),
        input_shape=(tensor_size,),
        output_shape=(tensor_size,),
        input_type="tensor(float)",
        output_type="tensor(float)",
    )


# ============================================================================
# Reshape and View Operation Fixtures
# ============================================================================


def create_reshape_layer(
    name: str,
    layer_id: int,
    input_shape: Tuple[int, ...] = (10,),
    output_shape: Tuple[int, ...] = (5, 2),
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
) -> "ReshapeLayer":
    """
    Create a Reshape layer for testing.

    Args:
        name: Layer name
        layer_id: Layer ID
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        inputs: Tuple of input tensor names
        outputs: Tuple of output tensor names

    Returns:
        ReshapeLayer instance
    """
    return ReshapeLayer(
        layer_id=layer_id,
        name=name,
        op_type="Reshape",
        inputs=inputs or (f"{name}_input",),
        outputs=outputs or (f"{name}_output",),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type="tensor(float)",
        output_type="tensor(float)",
        shape=np.array(output_shape, dtype=np.int64),
    )


# ============================================================================
# Network Fixtures
# ============================================================================


def create_simple_network(
    layers: Optional[dict] = None,
    execution_order: Optional[List[str]] = None,
    input_tensors: Optional[Tuple[str, ...]] = None,
    output_tensors: Optional[Tuple[str, ...]] = None,
) -> NetworkIR:
    """
    Create a simple NetworkIR for testing.

    Args:
        layers: Dictionary of layer_name -> layer objects
        execution_order: List of layer names in execution order
        input_tensors: Tuple of input tensor names
        output_tensors: Tuple of output tensor names

    Returns:
        NetworkIR instance
    """
    if layers is None:
        layers = {
            "layer_0": create_simple_matmul_layer(
                "layer_0", 0, inputs=("input",), outputs=("output",)
            )
        }

    if execution_order is None:
        execution_order = list(layers.keys())

    if input_tensors is None:
        input_tensors = ("input",)

    if output_tensors is None:
        output_tensors = ("output",)

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers={},
        tensor_consumers={},
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors={},
    )
