"""Pure shape math helpers for ONNX operators.

No registry, no semantics tracking, no side effects beyond logging warnings.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import logging
import numpy as np

logger = logging.getLogger(__name__)


def infer_matmul_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    if not input_shape or not weight_shape:
        logger.warning(f"Invalid shapes for MatMul: {input_shape} @ {weight_shape}")
        return ()

    a = tuple(input_shape)
    b = tuple(weight_shape)

    if len(a) == 1 and len(b) == 1:
        return (1,)
    if len(a) == 1 and len(b) >= 2:
        if a[0] != b[-2]:
            logger.warning(f"MatMul mismatch: {a} @ {b}")
            return ()
        batch = b[:-2]
        return (*batch, b[-1]) if batch else (b[-1],)
    if len(a) >= 2 and len(b) == 1:
        if a[-1] != b[0]:
            logger.warning(f"MatMul mismatch: {a} @ {b}")
            return ()
        out = a[:-1]
        return out if out else (1,)

    if a[-1] != b[-2]:
        logger.warning(f"MatMul mismatch: {a} @ {b}")
        return ()

    a_batch = a[:-2]
    b_batch = b[:-2]
    try:
        batch = np.broadcast_shapes(a_batch, b_batch)
    except ValueError:
        logger.warning(f"MatMul batch broadcast mismatch: {a_batch} vs {b_batch}")
        return ()

    return (*batch, a[-2], b[-1])


def infer_gemm_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...], transB: bool = False
) -> Tuple[int, ...]:
    if not weight_shape or len(weight_shape) < 2:
        logger.warning(f"Invalid weight shape for Gemm: {weight_shape}")
        return ()

    output_features = weight_shape[0] if transB else weight_shape[1]
    return (output_features,)


def infer_add_output_shape(
    input_shape: Tuple[int, ...], bias_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    if not input_shape:
        return bias_shape
    if not bias_shape:
        return input_shape
    return input_shape


def infer_conv2d_output_shape(
    input_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    strides: Tuple[int, int] = (1, 1),
    pads: Tuple[int, ...] = (0, 0, 0, 0),
    dilations: Tuple[int, int] = (1, 1),
) -> Tuple[int, ...]:
    if not input_shape or len(input_shape) < 3:
        logger.warning(f"Invalid input shape for Conv2D: {input_shape}")
        return ()
    if not weight_shape or len(weight_shape) != 4:
        logger.warning(f"Invalid weight shape for Conv2D: {weight_shape}")
        return ()

    h_in, w_in = input_shape[-2], input_shape[-1]
    out_channels = weight_shape[0]
    kH, kW = weight_shape[2], weight_shape[3]
    pad_top, pad_left, pad_bottom, pad_right = pads[0], pads[1], pads[2], pads[3]

    h_out = (h_in + pad_top + pad_bottom - dilations[0] * (kH - 1) - 1) // strides[
        0
    ] + 1
    w_out = (w_in + pad_left + pad_right - dilations[1] * (kW - 1) - 1) // strides[
        1
    ] + 1
    return (out_channels, h_out, w_out)


def infer_pool2d_output_shape(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    pads: Tuple[int, ...] = (0, 0, 0, 0),
) -> Tuple[int, ...]:
    if not input_shape or len(input_shape) < 3:
        logger.warning(f"Invalid input shape for Pool2D: {input_shape}")
        return ()

    channels = input_shape[-3]
    h_in, w_in = input_shape[-2], input_shape[-1]
    kH, kW = kernel_shape
    pad_top, pad_left, pad_bottom, pad_right = pads[0], pads[1], pads[2], pads[3]

    h_out = (h_in + pad_top + pad_bottom - kH) // strides[0] + 1
    w_out = (w_in + pad_left + pad_right - kW) // strides[1] + 1
    return (channels, h_out, w_out)


def infer_global_avg_pool_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if not input_shape or len(input_shape) < 3:
        logger.warning(f"Invalid input shape for GlobalAveragePool: {input_shape}")
        return ()
    return (input_shape[-3], 1, 1)


def infer_flatten_output_shape(
    input_shape: Tuple[int, ...], axis: int = 1
) -> Tuple[int, ...]:
    if not input_shape:
        return ()
    return (int(np.prod(input_shape)),)


def infer_transpose_output_shape(
    input_shape: Tuple[int, ...],
    perm: Tuple[int, ...],
    *,
    strict: bool = False,
    context: str = "Transpose",
) -> Tuple[int, ...]:
    if not input_shape:
        return ()

    if not perm:
        return tuple(reversed(input_shape))

    if len(perm) == len(input_shape) + 1:
        perm = tuple(p - 1 for p in perm if p != 0)

    if len(perm) != len(input_shape):
        msg = (
            f"{context}: perm length {len(perm)} != input shape rank "
            f"{len(input_shape)} (shape={input_shape}, perm={perm})"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    if sorted(perm) != list(range(len(input_shape))):
        msg = f"{context}: invalid permutation {perm} for shape {input_shape}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    return tuple(input_shape[p] for p in perm)


def infer_batchnorm_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return input_shape


def infer_squeeze_output_shape(
    input_shape: Tuple[int, ...], axes: Tuple[int, ...]
) -> Tuple[int, ...]:
    if not input_shape:
        return ()
    if not axes:
        return tuple(d for d in input_shape if d != 1) or (1,)

    result = list(input_shape)
    for ax in sorted(axes, reverse=True):
        if 0 <= ax < len(result) and result[ax] == 1:
            result.pop(ax)
        else:
            logger.warning(
                f"Squeeze axis {ax} is out of range or dim != 1 "
                f"(shape={input_shape}), skipping"
            )
    return tuple(result) if result else (1,)


def infer_cast_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return input_shape


def infer_unsqueeze_output_shape(
    input_shape: Tuple[int, ...],
    axes: Tuple[int, ...],
    *,
    strict: bool = False,
    context: str = "Unsqueeze",
) -> Tuple[int, ...]:
    if not input_shape:
        return ()

    if not axes:
        if strict:
            raise ValueError(f"{context}: missing axes for Unsqueeze")
        return input_shape

    out_rank = len(input_shape) + len(axes)
    norm_axes: List[int] = []
    for ax in axes:
        a = int(ax)
        if a < 0:
            a += out_rank
        if a < 0 or a >= out_rank:
            msg = f"{context}: axis {ax} is out of range for output rank {out_rank}"
            if strict:
                raise ValueError(msg)
            logger.warning(f"{msg}, returning input shape unchanged")
            return input_shape
        norm_axes.append(a)

    if len(set(norm_axes)) != len(norm_axes):
        msg = f"{context}: duplicate axes {axes}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    axes_set = set(norm_axes)
    out_shape: List[int] = []
    src_idx = 0
    for out_idx in range(out_rank):
        if out_idx in axes_set:
            out_shape.append(1)
        else:
            out_shape.append(input_shape[src_idx])
            src_idx += 1

    return tuple(out_shape)


def infer_slice_output_shape(
    input_shape: Tuple[int, ...],
    starts: Tuple[int, ...],
    ends: Tuple[int, ...],
    axes: Tuple[int, ...] = (),
    steps: Tuple[int, ...] = (),
    *,
    strict: bool = False,
    context: str = "Slice",
) -> Tuple[int, ...]:
    if not input_shape:
        return ()

    if not starts or not ends:
        if strict:
            raise ValueError(f"{context}: starts/ends are required for Slice")
        return input_shape

    if not axes:
        axes = tuple(range(len(starts)))
    if not steps:
        steps = tuple([1] * len(axes))

    if len(starts) != len(ends):
        msg = f"{context}: starts len {len(starts)} != ends len {len(ends)}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    if len(axes) != len(starts) or len(steps) != len(starts):
        msg = (
            f"{context}: mismatched Slice parameter lengths "
            f"starts={len(starts)}, axes={len(axes)}, steps={len(steps)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    output_shape = list(input_shape)
    for ax, start, end, step in zip(axes, starts, ends, steps):
        rank = len(output_shape)
        a = int(ax)
        if a < 0:
            a += rank

        if a < 0 or a >= rank:
            msg = f"{context}: axis {ax} is out of range for rank {rank}"
            if strict:
                raise ValueError(msg)
            logger.warning(f"{msg}, returning input shape unchanged")
            return input_shape

        if int(step) == 0:
            msg = f"{context}: step cannot be 0"
            if strict:
                raise ValueError(msg)
            logger.warning(f"{msg}, returning input shape unchanged")
            return input_shape

        dim_size = output_shape[a]
        s = int(start)
        e = int(end)
        k = int(step)

        if k > 0:
            actual_start = max(0, min(s if s >= 0 else dim_size + s, dim_size))
            actual_end = max(0, min(e if e >= 0 else dim_size + e, dim_size))
            slice_size = max(0, (actual_end - actual_start + (k - 1)) // k)
        else:
            actual_start = max(-1, min(s if s >= 0 else dim_size + s, dim_size - 1))
            actual_end = max(-1, min(e if e >= 0 else dim_size + e, dim_size - 1))
            step_abs = -k
            slice_size = max(
                0, (actual_start - actual_end + (step_abs - 1)) // step_abs
            )

        output_shape[a] = slice_size

    return tuple(output_shape)


def infer_expand_output_shape(
    input_shape: Tuple[int, ...],
    target_shape: Optional[Tuple[int, ...]],
    *,
    strict: bool = False,
    context: str = "Expand",
) -> Tuple[int, ...]:
    if not target_shape:
        if strict:
            raise ValueError(f"{context}: missing target shape for Expand")
        return input_shape

    if not all(int(s) > 0 for s in target_shape):
        msg = f"{context}: invalid non-positive target shape {target_shape}"
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    in_rev = list(reversed(input_shape))
    tgt_rev = list(reversed(tuple(int(s) for s in target_shape)))
    for i, tgt_dim in enumerate(tgt_rev):
        in_dim = in_rev[i] if i < len(in_rev) else 1
        if in_dim not in (1, tgt_dim):
            msg = (
                f"{context}: cannot broadcast input shape {input_shape} "
                f"to target shape {target_shape}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(f"{msg}, returning input shape unchanged")
            return input_shape

    if len(target_shape) < len(input_shape):
        msg = (
            f"{context}: target rank {len(target_shape)} < input rank "
            f"{len(input_shape)}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, returning input shape unchanged")
        return input_shape

    return tuple(int(s) for s in target_shape)


def infer_reduce_mean_output_shape(
    input_shape: Tuple[int, ...],
    axes: Tuple[int, ...],
    keepdims: bool,
) -> Tuple[int, ...]:
    if not input_shape:
        return ()

    rank = len(input_shape)
    if not axes:
        axes = tuple(range(rank))

    norm_axes = []
    for ax in axes:
        a = int(ax)
        if a < 0:
            a += rank
        if 0 <= a < rank:
            norm_axes.append(a)

    if keepdims:
        out = list(input_shape)
        for a in norm_axes:
            out[a] = 1
        return tuple(out)

    out = [d for i, d in enumerate(input_shape) if i not in set(norm_axes)]
    return tuple(out) if out else (1,)


def infer_reshape_output_shape(
    input_shape: Tuple[int, ...], target_shape: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    if not target_shape:
        return input_shape

    if -1 in target_shape:
        input_size = int(np.prod(input_shape)) if input_shape else 0
        known_dims = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known_dims)) if known_dims else 1

        if known_prod == 0:
            logger.warning(f"Invalid target shape {target_shape}")
            return ()

        inferred_dim = input_size // known_prod
        return tuple(inferred_dim if d == -1 else d for d in target_shape)

    return target_shape


def infer_einsum_output_shape(
    equation: str,
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    if equation != "abcd,cde->abe":
        return lhs_shape
    if len(rhs_shape) != 3:
        return lhs_shape

    c_dim, d_dim, e_dim = rhs_shape
    if len(lhs_shape) == 4 and lhs_shape[2] == c_dim and lhs_shape[3] == d_dim:
        return (lhs_shape[0], lhs_shape[1], e_dim)

    lhs_size = int(np.prod(lhs_shape)) if lhs_shape else 0
    contract = c_dim * d_dim
    if contract <= 0:
        return (1, 1, e_dim)

    ab = max(1, int(np.ceil(lhs_size / contract))) if lhs_size > 0 else 1
    return (1, ab, e_dim)
