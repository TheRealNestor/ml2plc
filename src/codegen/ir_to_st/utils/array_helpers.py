"""
Array indexing and multidimensional access utilities.

Provides helpers for computing flat indices in 1-D PLC arrays that represent
multidimensional data. Handles:
- Row-major (C-style) layout
- Strided access
- Conv2D/Pool2D spatial indexing
- Transpose permutation
- Boundary calculations
"""

from typing import List, Tuple, Optional, Union
import numpy as np


def compute_array_stride(shape: Tuple[int, ...], axis: int = -1) -> int:
    """
    Compute the stride (# of elements to skip) to move one unit along an axis.

    Args:
        shape: Array shape
        axis: Axis to compute stride for (negative indices supported)

    Returns:
        Stride value (product of dimensions after the axis)

    Example:
        >>> compute_array_stride((3, 4, 5), axis=0)  # 4*5 = 20
        20
        >>> compute_array_stride((3, 4, 5), axis=1)  # 5
        5
        >>> compute_array_stride((3, 4, 5), axis=-1)  # 1
        1
    """
    shape_len = len(shape)
    if axis < 0:
        axis = shape_len + axis

    if axis < 0 or axis >= shape_len:
        raise IndexError(f"Axis {axis} out of range for shape {shape}")

    stride = 1
    for dim in shape[axis + 1 :]:
        stride *= dim
    return stride


def compute_nd_indices(flat_index: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Convert flat index to multidimensional indices (row-major / C-style).

    Args:
        flat_index: Index in flattened array
        shape: Original array shape

    Returns:
        Tuple of indices for each dimension

    Example:
        >>> compute_nd_indices(14, (3, 4, 5))
        (0, 2, 4)
    """
    indices = []
    remaining = flat_index
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.insert(0, strides[0] * dim)

    for stride in strides[:-1]:
        indices.append(remaining // stride)
        remaining %= stride

    return tuple(indices)


def compute_flat_index(
    indices: Union[Tuple[int, ...], List[int]],
    shape: Tuple[int, ...],
) -> int:
    """
    Convert multidimensional indices to flat index (row-major / C-style).

    Args:
        indices: Tuple of indices for each dimension
        shape: Array shape

    Returns:
        Flat index in flattened array

    Example:
        >>> compute_flat_index((0, 2, 4), (3, 4, 5))
        14
    """
    if len(indices) != len(shape):
        raise ValueError(f"Index count {len(indices)} doesn't match shape {len(shape)}")

    flat = 0
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        flat += indices[i] * stride
        stride *= shape[i]

    return flat


def compute_strided_index(
    flat_index: int,
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    offset: int = 0,
) -> int:
    """
    Compute index with strides and offset (for slicing operations).

    Args:
        flat_index: Index into the strided view
        shape: Shape of the strided view
        strides: Step sizes for each dimension
        offset: Base offset in the original array

    Returns:
        Index in the original (non-strided) array

    Example:
        >>> # View with stride 2 on first axis
        >>> compute_strided_index(3, (5, 4), (2, 1), offset=10)
        22  # offset + 3*2
    """
    indices = compute_nd_indices(flat_index, shape)
    strided_indices = tuple(idx * stride for idx, stride in zip(indices, strides))
    base_index = compute_flat_index(strided_indices, shape)
    return offset + base_index


def compute_conv_indices(
    out_idx: int,
    out_shape: Tuple[int, int, int],  # (C_out, H_out, W_out)
    in_shape: Tuple[int, int, int],  # (C_in, H_in, W_in)
    kernel_shape: Tuple[int, int],  # (kH, kW)
    strides: Tuple[int, int],  # (sH, sW)
    pads: Tuple[int, int, int, int],  # (pH_top, pW_left, pH_bottom, pW_right)
) -> dict:
    """
    Compute indices for Conv2D output element.

    Returns loop bounds and formulas for the corresponding input receptive field.

    Args:
        out_idx: Flat output index
        out_shape: Output shape (C_out, H_out, W_out)
        in_shape: Input shape (C_in, H_in, W_in)
        kernel_shape: Kernel size (kH, kW)
        strides: Strides (sH, sW)
        pads: Padding (top, left, bottom, right)

    Returns:
        Dict with:
        - output_oc, output_oh, output_ow: Output indices
        - input_h_range: (h_start, h_end) for input height
        - input_w_range: (w_start, w_end) for input width
        - receptive_field_size: kH * kW

    Example:
        >>> info = compute_conv_indices(
        ...     out_idx=5,
        ...     out_shape=(1, 3, 3),
        ...     in_shape=(3, 5, 5),
        ...     kernel_shape=(3, 3),
        ...     strides=(1, 1),
        ...     pads=(1, 1, 1, 1)
        ... )
        >>> info["output_oc"], info["output_oh"], info["output_ow"]
        (0, 1, 2)
    """
    c_out, h_out, w_out = out_shape
    c_in, h_in, w_in = in_shape
    kh, kw = kernel_shape
    sh, sw = strides
    ph_top, pw_left, ph_bottom, pw_right = pads

    # Convert flat output index to 3-D
    oc = out_idx // (h_out * w_out)
    oh = (out_idx % (h_out * w_out)) // w_out
    ow = out_idx % w_out

    # Compute input receptive field bounds
    # Output (oh, ow) corresponds to input region starting at:
    h_start = oh * sh - ph_top
    w_start = ow * sw - pw_left
    h_end = h_start + kh
    w_end = w_start + kw

    # Clamp to input bounds
    h_start_valid = max(0, h_start)
    w_start_valid = max(0, w_start)
    h_end_valid = min(h_in, h_end)
    w_end_valid = min(w_in, w_end)

    return {
        "output_oc": oc,
        "output_oh": oh,
        "output_ow": ow,
        "input_h_range": (h_start_valid, h_end_valid),
        "input_w_range": (w_start_valid, w_end_valid),
        "receptive_field_size": kh * kw,
        "has_padding": h_start < 0 or w_start < 0,
    }


def compute_pool_indices(
    out_idx: int,
    out_shape: Tuple[int, int, int],  # (C, H_out, W_out)
    in_shape: Tuple[int, int, int],  # (C, H_in, W_in)
    kernel_shape: Tuple[int, int],  # (kH, kW)
    strides: Tuple[int, int],  # (sH, sW)
    pads: Tuple[int, int, int, int],  # (pH_top, pW_left, pH_bottom, pW_right)
) -> dict:
    """
    Compute indices for Pool2D output element.

    Similar to compute_conv_indices but for pooling operations.

    Args:
        out_idx: Flat output index
        out_shape: Output shape (C, H_out, W_out)
        in_shape: Input shape (C, H_in, W_in)
        kernel_shape: Kernel size (kH, kW)
        strides: Strides (sH, sW)
        pads: Padding (top, left, bottom, right)

    Returns:
        Dict with output indices and input receptive field information
    """
    # For pool, channels are preserved, so logic is very similar to Conv
    c, h_out, w_out = out_shape
    c_in, h_in, w_in = in_shape
    kh, kw = kernel_shape
    sh, sw = strides
    ph_top, pw_left, _, _ = pads

    # Convert flat output index to 3-D
    c_idx = out_idx // (h_out * w_out)
    oh = (out_idx % (h_out * w_out)) // w_out
    ow = out_idx % w_out

    # Compute input receptive field
    h_start = oh * sh - ph_top
    w_start = ow * sw - pw_left

    h_start_valid = max(0, h_start)
    w_start_valid = max(0, w_start)
    h_end_valid = min(h_in, h_start + kh)
    w_end_valid = min(w_in, w_start + kw)

    return {
        "channel": c_idx,
        "output_h": oh,
        "output_w": ow,
        "input_h_range": (h_start_valid, h_end_valid),
        "input_w_range": (w_start_valid, w_end_valid),
        "pool_size": kh * kw,
    }


def compute_transpose_strides(shape: Tuple[int, ...], perm: Tuple[int, ...]) -> dict:
    """
    Compute strides for transposed array access.

    Args:
        shape: Original array shape
        perm: Permutation of axes

    Returns:
        Dict with:
        - original_shape: Input shape
        - transposed_shape: Output shape
        - original_strides: Strides in original layout
        - transposed_strides: Strides in transposed layout
        - inverse_perm: Inverse permutation (for converting indices)

    Example:
        >>> info = compute_transpose_strides((2, 3, 4), (2, 0, 1))
        >>> info["transposed_shape"]
        (4, 2, 3)
    """
    ndim = len(shape)

    # Compute original strides (row-major)
    orig_strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        orig_strides[i] = orig_strides[i + 1] * shape[i + 1]

    # Compute transposed shape
    transposed_shape = tuple(shape[p] for p in perm)

    # Compute transposed strides
    trans_strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        trans_strides[i] = trans_strides[i + 1] * transposed_shape[i + 1]

    # Compute inverse permutation
    inv_perm = [0] * ndim
    for i in range(ndim):
        inv_perm[perm[i]] = i

    return {
        "original_shape": shape,
        "transposed_shape": transposed_shape,
        "original_strides": orig_strides,
        "transposed_strides": trans_strides,
        "inverse_perm": inv_perm,
        "perm": perm,
    }
