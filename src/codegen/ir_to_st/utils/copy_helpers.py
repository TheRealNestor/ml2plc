"""
Data movement and copy pattern utilities.

Provides helpers for generating common data transfer patterns:
- Simple element-wise copy
- Scalar broadcast
- Modular broadcast (for expand operations)
- Offset copy (for slicing)
- Strided copy (for stepping through data)

These patterns are used by many layers (Reshape, Squeeze, Unsqueeze,
Cast identity, Expand, Slice, etc.).
"""

from typing import Optional, Union
from ..st_code import STCode, STCodeBuilder


def generate_simple_copy(
    input_var: str,
    output_var: str,
    size: int,
    comment: str = "",
) -> STCode:
    """
    Generate a simple element-by-element copy loop.

    Used for operations that don't change data semantically (Reshape with same size,
    Squeeze, Unsqueeze, identity Cast, identity Expand).

    Args:
        input_var: Source array variable name
        output_var: Destination array variable name
        size: Number of elements to copy
        comment: Optional descriptive comment

    Returns:
        STCode with FOR loop performing copy

    Example:
        >>> code = generate_simple_copy("input", "output", 100, "Copy operation")
        >>> print(code)
        (* Copy operation *)
        FOR i := 0 TO 99 DO
            output[i] := input[i];
        END_FOR;
    """
    builder = STCodeBuilder()

    if comment:
        builder.add_line(f"(* {comment} *)")

    builder.add_line(f"FOR i := 0 TO {size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {input_var}[i];")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_offset_copy(
    input_var: str,
    output_var: str,
    size: int,
    offset: int,
    comment: str = "",
) -> STCode:
    """
    Generate a copy with offset (for slice operations).

    Reads from input[offset + i] and writes to output[i].

    Args:
        input_var: Source array variable name
        output_var: Destination array variable name
        size: Number of elements to copy
        offset: Base offset in source array (can be negative, but will be converted to positive)
        comment: Optional descriptive comment

    Returns:
        STCode with FOR loop performing offset copy

    Example:
        >>> code = generate_offset_copy("input", "output", 50, 100, "Extract slice")
        >>> print(code)
        (* Extract slice *)
        FOR i := 0 TO 49 DO
            output[i] := input[i + 100];
        END_FOR;
    """
    builder = STCodeBuilder()

    if comment:
        builder.add_line(f"(* {comment} *)")

    builder.add_line(f"FOR i := 0 TO {size - 1} DO")
    with builder.indent():
        if offset == 0:
            builder.add_line(f"{output_var}[i] := {input_var}[i];")
        elif offset > 0:
            builder.add_line(f"{output_var}[i] := {input_var}[i + {offset}];")
        else:
            # Negative offset: convert to addition. E.g., i + (-5) becomes i - 5
            builder.add_line(f"{output_var}[i] := {input_var}[i - {-offset}];")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_strided_copy(
    input_var: str,
    output_var: str,
    output_size: int,
    stride: int,
    start_index: int = 0,
    comment: str = "",
) -> STCode:
    """
    Generate a strided copy (for slice with step > 1).

    Reads from input[start_index + i * stride] and writes to output[i].

    Args:
        input_var: Source array variable name
        output_var: Destination array variable name
        output_size: Number of elements to write
        stride: Step size in source array
        start_index: Starting index in source array
        comment: Optional descriptive comment

    Returns:
        STCode with FOR loop performing strided copy

    Example:
        >>> code = generate_strided_copy("input", "output", 50, 2, 10, "Extract stride-2")
        >>> print(code)
        (* Extract stride-2 *)
        FOR i := 0 TO 49 DO
            output[i] := input[10 + i * 2];
        END_FOR;
    """
    builder = STCodeBuilder()

    if comment:
        builder.add_line(f"(* {comment} *)")

    builder.add_line(f"FOR i := 0 TO {output_size - 1} DO")
    with builder.indent():
        if start_index == 0:
            builder.add_line(f"{output_var}[i] := {input_var}[i * {stride}];")
        else:
            builder.add_line(
                f"{output_var}[i] := {input_var}[{start_index} + i * {stride}];"
            )
    builder.add_line("END_FOR;")

    return builder.build()


def generate_scalar_broadcast(
    input_var: str,
    output_var: str,
    output_size: int,
    comment: str = "",
) -> STCode:
    """
    Generate scalar broadcast (broadcast single value to array).

    Used by Expand when input_size=1.

    Args:
        input_var: Source scalar variable (or input[0])
        output_var: Destination array variable name
        output_size: Output size
        comment: Optional descriptive comment

    Returns:
        STCode with FOR loop broadcasting scalar

    Example:
        >>> code = generate_scalar_broadcast("input", "output", 100, "Broadcast scalar")
        >>> print(code)
        (* Broadcast scalar *)
        FOR i := 0 TO 99 DO
            output[i] := input[0];
        END_FOR;
    """
    builder = STCodeBuilder()

    if comment:
        builder.add_line(f"(* {comment} *)")

    builder.add_line(f"FOR i := 0 TO {output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {input_var}[0];")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_modulo_broadcast(
    input_var: str,
    output_var: str,
    input_size: int,
    output_size: int,
    comment: str = "",
) -> STCode:
    """
    Generate broadcast using modulo indexing (for Expand with arbitrary broadcasts).

    Maps output[i] = input[i MOD input_size].

    Args:
        input_var: Source array variable name
        output_var: Destination array variable name
        input_size: Input size
        output_size: Output size
        comment: Optional descriptive comment

    Returns:
        STCode with FOR loop performing modulo broadcast

    Example:
        >>> code = generate_modulo_broadcast("input", "output", 10, 50, "Expand 10→50")
        >>> print(code)
        (* Expand 10→50 *)
        FOR i := 0 TO 49 DO
            output[i] := input[i MOD 10];
        END_FOR;
    """
    builder = STCodeBuilder()

    if comment:
        builder.add_line(f"(* {comment} *)")

    builder.add_line(f"FOR i := 0 TO {output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {input_var}[i MOD {input_size}];")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_selective_copy(
    input_var: str,
    output_var: str,
    indices: list,
    comment: str = "",
) -> STCode:
    """
    Generate copy of specific elements by constant indices (for Gather with constant indices).

    Args:
        input_var: Source array variable name
        output_var: Destination array variable name
        indices: List of input indices to copy (e.g., [0, 2, 5, 7])
        comment: Optional descriptive comment

    Returns:
        STCode with unrolled copy statements (for small index lists)

    Example:
        >>> code = generate_selective_copy("input", "output", [0, 2, 5, 7], "Gather")
        >>> print(code)
        (* Gather *)
        output[0] := input[0];
        output[1] := input[2];
        output[2] := input[5];
        output[3] := input[7];
    """
    builder = STCodeBuilder()

    if comment:
        builder.add_line(f"(* {comment} *)")

    # For small constant index lists, unroll directly
    if len(indices) <= 16:
        for out_idx, in_idx in enumerate(indices):
            builder.add_line(f"{output_var}[{out_idx}] := {input_var}[{int(in_idx)}];")
    else:
        # For large lists, use array of indices (would need to generate constant)
        builder.add_line(
            f"(* WARNING: large Gather index list, using conservative copy *)"
        )
        builder.add_line(f"FOR i := 0 TO {len(indices) - 1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {input_var}[i];")
        builder.add_line("END_FOR;")

    return builder.build()
