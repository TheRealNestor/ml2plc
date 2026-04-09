"""
Layer-specific loop generation utilities.

Provides helpers for loop patterns that appear across multiple layer generators,
such as Conv2D/Pool2D loop structures with boundary checking, LSTM timestep loops,
and common strided/nested iteration patterns.

For general-purpose loop generation (st_for_loop, st_if_block), use st_code.py.
For nested FOR loops, use st_nested_for_loops from st_code.py.
"""

from typing import Optional, Union, Generator, Any
from contextlib import contextmanager
from ..st_code import STCode, STCodeBuilder


# Re-export general utilities for convenience
from ..st_code import st_for_loop


def generate_nested_spatial_loops(
    output_dims: tuple,
    input_dims: tuple,
    kernel_size: tuple,
    strides: tuple,
    pads: tuple,
    body_fn=None,
) -> STCode:
    """
    Generate nested loops for spatial operations (Conv2D, Pool2D, etc.).

    Handles the common pattern: outer loops over output spatial dims,
    inner loops over kernel spatial dims, with input coordinate computation.

    Args:
        output_dims: (C_out, H_out, W_out) or similar
        input_dims: (C_in, H_in, W_in) or similar
        kernel_size: (kH, kW)
        strides: (sH, sW)
        pads: (pH_top, pW_left, pH_bottom, pW_right) or (pH, pW)
        body_fn: Optional function(builder, oh, ow, ih, iw) that adds loop body

    Returns:
        STCode with nested loop structure
    """
    # This is a helper for complex nested structures
    # Specific Conv2D/Pool2D code generation should remain in generator.py
    # This is just a partial pattern, not a complete solution
    return STCode.empty()


@contextmanager
def with_boundary_check(
    builder: STCodeBuilder,
    condition: str,
) -> Generator[None, None, None]:
    """
    Context manager for conditional boundary checking (used in Conv2D/Pool2D).

    Emits an IF statement that wraps code generation within the context.
    Useful for padding handling in spatial operations.

    Args:
        builder: STCodeBuilder to emit to
        condition: IF condition expression (e.g., "(ih >= 0) AND (ih < H_max)")

    Yields:
        None (modifies builder in-place)

    Example:
        >>> builder = STCodeBuilder()
        >>> with with_boundary_check(builder, f"(ih >= 0) AND (ih < {H})"):
        ...     builder.add_line("sum := sum + input[idx];")
        >>> print(builder.build())
        IF (ih >= 0) AND (ih < H) THEN
            sum := sum + input[idx];
        END_IF;
    """
    builder.add_line(f"IF {condition} THEN")

    try:
        yield
    finally:
        builder.add_line("END_IF;")
