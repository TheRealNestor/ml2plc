"""
Shared test fixtures and helper functions.

Consolidates common test utilities used across multiple test modules.
"""

import numpy as np
from typing import Tuple

from src.codegen.types import MatMulLayer, NetworkIR


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
