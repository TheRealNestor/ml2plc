"""
Data movement and utility layer code generators.

Handles reshaping, slicing, concatenation, transposition, and type conversions.
"""

import numpy as np
import logging
from typing import List

from ...types import (
    ReshapeLayer,
    ActivationLayer,
    AddLayer,
    ActivationType,
    QuantizeLinearLayer,
    DequantizeLinearLayer,
    DropoutLayer,
    FlattenLayer,
    SqueezeLayer,
    TransposeLayer,
    UnsqueezeLayer,
    ExpandLayer,
    ShapeLayer,
    GatherLayer,
    CastLayer,
    SliceLayer,
    ConcatLayer,
    ReduceMeanLayer,
    ReduceProdLayer,
    BinaryElementwiseLayer,
    UnaryElementwiseLayer,
    RuntimeMatMulLayer,
    EinsumLayer,
)
from ...matmul_contract import validate_runtime_matmul_contract
from ..st_code import STCode, STCodeBuilder
from ..type_conversion import (
    plc_type_from_onnx_dtype,
    get_conversion_func,
    get_type_limits_from_str,
)
from ..utils.copy_helpers import (
    generate_simple_copy,
    generate_offset_copy,
    generate_strided_copy,
    generate_scalar_broadcast,
    generate_modulo_broadcast,
    generate_selective_copy,
)
from ..variable import Variable
from ..utils.activation_helpers import generate_activation_loop

logger = logging.getLogger(__name__)


def generate_activation_layer_code(
    layer: ActivationLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate activation layer code."""
    code = STCode.from_lines(
        f"(* Layer {layer.layer_id}: Activation ({layer.activation.name}) *)"
    )
    # Accept either Variable or raw names
    from ..variable import ensure_var

    input_var = ensure_var(input_var, layer.input_shape)
    output_var = ensure_var(output_var, layer.output_shape)

    code += generate_activation_loop(
        layer.activation, input_var, output_var, layer.output_size
    )
    return code


def generate_add_code(
    layer: AddLayer, input_vars: List[Variable], output_var: Variable
) -> STCode:
    """Generate Add layer code (bias addition or element-wise)."""
    builder = STCodeBuilder()

    if layer.bias is not None:
        # Bias addition: output = input + bias
        bias_var = Variable(name=f"bias_{layer.layer_id}", shape=(layer.output_size,))
        builder.add_line(f"(* Layer {layer.layer_id}: Add (Bias) *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := {input_vars[0].at('i')} + {bias_var.at('i')};"
            )
        builder.add_line("END_FOR;")
    else:
        # Element-wise addition: output = input1 + input2
        if len(input_vars) != 2:
            raise ValueError(
                f"Element-wise Add layer {layer.layer_id} expected 2 inputs, got {len(input_vars)}"
            )

        builder.add_line(f"(* Layer {layer.layer_id}: Add (Element-wise) *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := {input_vars[0].at('i')} + {input_vars[1].at('i')};"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_reshape_code(
    layer: ReshapeLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Reshape layer code."""
    if layer.input_size != layer.output_size:
        raise NotImplementedError("Reshape with different sizes not implemented.")
    return generate_simple_copy(
        input_var,
        output_var,
        f"Layer {layer.layer_id}: Reshape (copy input to output)",
    )


def generate_quantize_linear_code(
    layer: QuantizeLinearLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate QuantizeLinear code: quantized = clip(round(input/scale) + zero_point)."""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: QuantizeLinear *)")

    output_plc_type = plc_type_from_onnx_dtype(layer.output_type)
    min_val, max_val = get_type_limits_from_str(layer.output_type)
    cast_func = get_conversion_func("REAL", output_plc_type)

    is_per_tensor = layer.scale.size == 1

    if is_per_tensor:
        scale_val = layer.scale.flat[0]
        zero_point_val = layer.zero_point.flat[0]

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var.at('i')} / {scale_val}) + {zero_point_val}), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")
    else:
        scale_var = Variable(name=f"scale_{layer.layer_id}", shape=(layer.output_size,))
        zero_point_var = Variable(
            name=f"zero_point_{layer.layer_id}", shape=(layer.output_size,)
        )

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var.at('i')} / {scale_var.at('i')}) + {zero_point_var.at('i')}), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_dequantize_linear_code(
    layer: DequantizeLinearLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate DequantizeLinear code: float = scale * (quantized - zero_point)."""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: DequantizeLinear *)")

    input_plc_type = plc_type_from_onnx_dtype(layer.input_type)
    cast_func = get_conversion_func(input_plc_type, "REAL")

    is_per_tensor = layer.scale.size == 1

    if is_per_tensor:
        scale_val = layer.scale.flat[0]
        zero_point_val = layer.zero_point.flat[0]

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := {scale_val} * "
                f"{cast_func}({input_var.at('i')} - {zero_point_val});"
            )
        builder.add_line("END_FOR;")
    else:
        scale_var = Variable(name=f"scale_{layer.layer_id}", shape=(layer.output_size,))
        zero_point_var = Variable(
            name=f"zero_point_{layer.layer_id}", shape=(layer.output_size,)
        )

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var.at('i')} := {scale_var.at('i')} * "
                f"{cast_func}({input_var.at('i')} - {zero_point_var.at('i')});"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_dropout_code(
    layer: DropoutLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Dropout layer code (identity at inference time)."""
    # Dropout is a no-op at inference time (identity copy)
    return generate_simple_copy(
        input_var,
        output_var,
        f"Layer {layer.layer_id}: Dropout (identity at inference time)",
    )


def generate_flatten_code(
    layer: FlattenLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Flatten layer code (identity copy)."""
    return generate_simple_copy(
        input_var,
        output_var,
        f"Layer {layer.layer_id}: Flatten (axis={layer.axis})",
    )


def generate_squeeze_code(
    layer: SqueezeLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Squeeze layer code (identity copy)."""
    return generate_simple_copy(
        input_var, output_var, f"Layer {layer.layer_id}: Squeeze"
    )


def generate_cast_code(
    layer: CastLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Cast layer code."""
    input_plc = (
        plc_type_from_onnx_dtype(layer.input_type) if layer.input_type else "REAL"
    )
    output_plc = (
        plc_type_from_onnx_dtype(layer.output_type) if layer.output_type else "REAL"
    )

    if input_plc == output_plc:
        return generate_simple_copy(
            input_var,
            output_var,
            f"Layer {layer.layer_id}: Cast (no-op, same type {input_plc})",
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Cast {input_plc} -> {output_plc} *)")

    cast_func = f"{output_plc}_TO_{input_plc}"
    builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var.at('i')} := {cast_func}({input_var.at('i')});")
    builder.add_line("END_FOR;")
    return builder.build()


def generate_slice_code(
    layer: SliceLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Slice layer code."""
    comment = (
        f"Layer {layer.layer_id}: Slice starts={layer.starts} "
        f"ends={layer.ends} axes={layer.axes} steps={layer.steps}"
    )

    logger.debug(
        f"Generating Slice code: input_shape={layer.input_shape}, "
        f"output_shape={layer.output_shape}, starts={layer.starts}, "
        f"ends={layer.ends}, axes={layer.axes}, steps={layer.steps}, "
        f"input_size={layer.input_size}, output_size={layer.output_size}"
    )

    if len(layer.axes) == 1 and layer.steps[0] == 1:
        # Simple offset copy
        start = layer.starts[0]
        axis = layer.axes[0]

        if layer.input_shape and axis < len(layer.input_shape):
            # Calculate stride: product of all dimensions after this axis
            stride = int(np.prod(layer.input_shape[axis + 1 :]))
            offset = start * stride
        else:
            # Fallback: assume flat or axis 0
            offset = start

        logger.debug(
            f"  Slice: simple offset copy with start={start}, axis={axis}, stride={stride if layer.input_shape else 'N/A'}, offset={offset}"
        )
        return generate_offset_copy(input_var, output_var, offset, comment)

    elif len(layer.axes) == 1 and layer.steps[0] != 1:
        # Strided copy
        start = layer.starts[0]
        step = layer.steps[0]

        logger.debug(f"  Slice: strided copy with start={start}, step={step}")
        return generate_strided_copy(input_var, output_var, step, start, comment)

    else:
        # Multi-axis slice — conservative copy
        logger.debug(f"  Slice: multi-axis slice, using conservative copy")
        return generate_simple_copy(
            input_var,
            output_var,
            f"{comment} (multi-axis, conservative)",
        )


def generate_concat_code(
    layer: ConcatLayer, input_vars: List[Variable], output_var: Variable
) -> STCode:
    """Generate Concat layer code."""
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Concat axis={layer.axis} inputs={len(input_vars)} *)"
    )

    offset = 0
    for idx, (inp_var, inp_size) in enumerate(zip(input_vars, layer.input_sizes)):
        builder.add_line(f"(* Concat part {idx}: {inp_var.name} [{inp_size} elems] *)")
        builder.add_line(f"FOR i := 0 TO {inp_size - 1} DO")
        with builder.indent():
            if offset == 0:
                builder.add_line(f"{output_var.at('i')} := {inp_var.at('i')};")
            else:
                builder.add_line(
                    f"{output_var.at(f'i + {offset}')} := {inp_var.at('i')};"
                )
        builder.add_line("END_FOR;")
        offset += inp_size

    return builder.build()


def generate_transpose_code(
    layer: TransposeLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Transpose layer code with nested loops over output shape."""
    builder = STCodeBuilder()
    in_shape = layer.input_shape
    out_shape = layer.output_shape
    perm = layer.perm
    ndim = len(out_shape)

    builder.add_line(
        f"(* Layer {layer.layer_id}: Transpose perm={perm}  "
        f"{in_shape} -> {out_shape} *)"
    )

    if ndim == 0 or layer.input_size <= 1:
        builder.add_line(f"{output_var.at('0')} := {input_var.at('0')};")
        return builder.build()

    # Compute strides
    in_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1]

    out_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1]

    loop_vars = [f"t{layer.layer_id}_d{d}" for d in range(ndim)]

    # Build nested loops
    for d in range(ndim):
        indent = "    " * d
        builder.add_line(f"{indent}FOR {loop_vars[d]} := 0 TO {out_shape[d] - 1} DO")

    # Compute indices
    inner_indent = "    " * ndim

    out_idx_parts = []
    for d in range(ndim):
        if out_strides[d] == 1:
            out_idx_parts.append(loop_vars[d])
        else:
            out_idx_parts.append(f"{loop_vars[d]} * {out_strides[d]}")
    out_idx = " + ".join(out_idx_parts)

    inv_perm = [0] * ndim
    for d in range(ndim):
        inv_perm[perm[d]] = d

    in_idx_parts = []
    for k in range(ndim):
        var = loop_vars[inv_perm[k]]
        if in_strides[k] == 1:
            in_idx_parts.append(var)
        else:
            in_idx_parts.append(f"{var} * {in_strides[k]}")
    in_idx = " + ".join(in_idx_parts)

    builder.add_line(
        f"{inner_indent}{output_var.at(out_idx)} := {input_var.at(in_idx)};"
    )

    # Close loops
    for d in range(ndim - 1, -1, -1):
        indent = "    " * d
        builder.add_line(f"{indent}END_FOR;")

    return builder.build()


def generate_unsqueeze_code(
    layer: UnsqueezeLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Unsqueeze layer code (identity copy)."""
    return generate_simple_copy(
        input_var,
        output_var,
        layer.output_size,
        f"Layer {layer.layer_id}: Unsqueeze axes={layer.unsqueeze_axes}",
    )


def generate_expand_code(
    layer: ExpandLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Expand (broadcast) layer code."""
    if layer.input_size == layer.output_size:
        return generate_simple_copy(
            input_var,
            output_var,
            layer.output_size,
            f"Layer {layer.layer_id}: Expand (no broadcast)",
        )

    if layer.input_size == 1:
        # Scalar broadcast
        return generate_scalar_broadcast(
            input_var,
            output_var,
            layer.output_size,
            f"Layer {layer.layer_id}: Expand (scalar broadcast) to {layer.target_shape}",
        )
    else:
        # General broadcast
        return generate_modulo_broadcast(
            input_var,
            output_var,
            layer.input_size,
            layer.output_size,
            f"Layer {layer.layer_id}: Expand (modulo broadcast) to {layer.target_shape}",
        )


def generate_shape_code(
    layer: ShapeLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Shape layer code (should be constant-folded)."""
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Shape extraction - SHOULD BE CONSTANT-FOLDED *)"
    )
    builder.add_line(
        f"(* Output shape: {layer.output_shape}, input shape: {layer.input_shape} *)"
    )
    return builder.build()


def generate_gather_code(
    layer: GatherLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Gather layer code."""
    if layer.indices is not None and layer.indices.size <= 16:
        flat_indices = layer.indices.flatten().tolist()
        return generate_selective_copy(
            input_var,
            output_var,
            flat_indices,
            f"Layer {layer.layer_id}: Gather axis={layer.gather_axis}",
        )
    else:
        # Dynamic or large indices — conservative copy
        return generate_simple_copy(
            input_var,
            output_var,
            layer.output_size,
            f"Layer {layer.layer_id}: Gather axis={layer.gather_axis} (dynamic/large indices)",
        )


def generate_reduce_mean_code(
    layer: ReduceMeanLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate ReduceMean code for arbitrary static-axis reductions."""
    if not layer.input_shape or not layer.output_shape:
        raise ValueError(
            f"ReduceMean layer {layer.layer_id}: static input/output shapes are required"
        )

    ndim = len(layer.input_shape)
    if ndim == 0:
        return generate_simple_copy(
            input_var,
            output_var,
            layer.output_size,
            f"Layer {layer.layer_id}: ReduceMean scalar passthrough",
        )

    axes = layer.axes if layer.axes else tuple(range(ndim))
    normalized_axes = tuple(sorted(a if a >= 0 else a + ndim for a in axes))
    reduced_set = set(normalized_axes)
    kept_axes = tuple(a for a in range(ndim) if a not in reduced_set)

    if layer.keepdims:
        output_axis_for_input_axis = {a: a for a in kept_axes}
    else:
        output_axis_for_input_axis = {a: idx for idx, a in enumerate(kept_axes)}

    in_strides = []
    run = 1
    for d in reversed(layer.input_shape):
        in_strides.append(run)
        run *= int(d)
    in_strides = tuple(reversed(in_strides))

    out_shape = layer.output_shape if layer.output_shape else (1,)
    out_strides = []
    run = 1
    for d in reversed(out_shape):
        out_strides.append(run)
        run *= int(d)
    out_strides = tuple(reversed(out_strides))

    reduction_factor = int(np.prod([layer.input_shape[a] for a in normalized_axes]))
    if reduction_factor <= 0 or layer.input_size % reduction_factor != 0:
        raise ValueError(
            f"ReduceMean layer {layer.layer_id}: invalid reduction_factor={reduction_factor} "
            f"for input_size={layer.input_size}"
        )

    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: ReduceMean axes={normalized_axes} keepdims={layer.keepdims} *)"
    )

    builder.add_line(f"FOR j := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var.at('j')} := 0.0;")
    builder.add_line("END_FOR;")

    builder.add_line(f"FOR i := 0 TO {layer.input_size - 1} DO")
    with builder.indent():
        builder.add_line("j := 0;")
        for axis in kept_axes:
            in_stride = int(in_strides[axis])
            in_dim = int(layer.input_shape[axis])
            out_axis = output_axis_for_input_axis[axis]
            out_stride = int(out_strides[out_axis])
            builder.add_line(f"k := (i / {in_stride}) MOD {in_dim};")
            builder.add_line(f"j := j + k * {out_stride};")
        builder.add_line(
            f"{output_var.at('j')} := {output_var.at('j')} + {input_var.at('i')};"
        )
    builder.add_line("END_FOR;")

    builder.add_line(f"FOR j := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(
            f"{output_var.at('j')} := {output_var.at('j')} / {float(reduction_factor)};"
        )
    builder.add_line("END_FOR;")

    return builder.build()


def generate_reduce_prod_code(
    layer: ReduceProdLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate ReduceProd code for arbitrary static-axis reductions."""
    if not layer.input_shape or not layer.output_shape:
        raise ValueError(
            f"ReduceProd layer {layer.layer_id}: static input/output shapes are required"
        )

    ndim = len(layer.input_shape)
    if ndim == 0:
        return generate_simple_copy(
            input_var,
            output_var,
            layer.output_size,
            f"Layer {layer.layer_id}: ReduceProd scalar passthrough",
        )

    axes = layer.axes if layer.axes else tuple(range(ndim))
    normalized_axes = tuple(sorted(a if a >= 0 else a + ndim for a in axes))
    reduced_set = set(normalized_axes)
    kept_axes = tuple(a for a in range(ndim) if a not in reduced_set)

    if layer.keepdims:
        output_axis_for_input_axis = {a: a for a in kept_axes}
    else:
        output_axis_for_input_axis = {a: idx for idx, a in enumerate(kept_axes)}

    in_strides = []
    run = 1
    for d in reversed(layer.input_shape):
        in_strides.append(run)
        run *= int(d)
    in_strides = tuple(reversed(in_strides))

    out_shape = layer.output_shape if layer.output_shape else (1,)
    out_strides = []
    run = 1
    for d in reversed(out_shape):
        out_strides.append(run)
        run *= int(d)
    out_strides = tuple(reversed(out_strides))

    reduction_factor = int(np.prod([layer.input_shape[a] for a in normalized_axes]))
    if reduction_factor <= 0 or layer.input_size % reduction_factor != 0:
        raise ValueError(
            f"ReduceProd layer {layer.layer_id}: invalid reduction_factor={reduction_factor} "
            f"for input_size={layer.input_size}"
        )

    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: ReduceProd axes={normalized_axes} keepdims={layer.keepdims} *)"
    )

    builder.add_line(f"FOR j := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var.at('j')} := 1.0;")
    builder.add_line("END_FOR;")

    builder.add_line(f"FOR i := 0 TO {layer.input_size - 1} DO")
    with builder.indent():
        builder.add_line("j := 0;")
        for axis in kept_axes:
            in_stride = int(in_strides[axis])
            in_dim = int(layer.input_shape[axis])
            out_axis = output_axis_for_input_axis[axis]
            out_stride = int(out_strides[out_axis])
            builder.add_line(f"k := (i / {in_stride}) MOD {in_dim};")
            builder.add_line(f"j := j + k * {out_stride};")
        builder.add_line(
            f"{output_var.at('j')} := {output_var.at('j')} * {input_var.at('i')};"
        )
    builder.add_line("END_FOR;")

    return builder.build()


def generate_runtime_matmul_code(
    layer: RuntimeMatMulLayer, input_vars: List[Variable], output_var: Variable
) -> STCode:
    """Generate runtime MatMul code for non-constant RHS tensors."""
    if len(input_vars) != 2:
        raise ValueError(
            f"Runtime MatMul layer {layer.layer_id}: expected 2 inputs, got {len(input_vars)}"
        )

    lhs_var, rhs_var = input_vars
    contract = validate_runtime_matmul_contract(
        tuple(layer.input_shape or ()),
        tuple(layer.rhs_shape or ()),
        context=f"Runtime MatMul layer {layer.layer_id}",
    )
    lhs_shape = contract.lhs_shape
    rhs_shape = contract.rhs_shape

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Runtime MatMul *)")

    # Vector dot: (K,) @ (K,) -> (1,)
    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        builder.add_line("sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {lhs_shape[0] - 1} DO")
        with builder.indent():
            builder.add_line(f"sum := sum + {lhs_var.at('i')} * {rhs_var.at('i')};")
        builder.add_line("END_FOR;")
        builder.add_line(f"{output_var.at('0')} := sum;")
        return builder.build()

    # Matrix-matrix: (M,K) @ (K,N) -> (M,N)
    if len(lhs_shape) == 2 and len(rhs_shape) == 2:
        m, k = lhs_shape
        _, n = rhs_shape

        builder.add_line(f"FOR j := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {k - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + {lhs_var.at(f'(j / {n}) * {k} + i')} * {rhs_var.at(f'i * {n} + (j MOD {n})')} ;"
                )
            builder.add_line("END_FOR;")
            builder.add_line(f"{output_var.at('j')} := sum;")
        builder.add_line("END_FOR;")
        return builder.build()

    # Vector-matrix: (K,) @ (K,N) -> (N,)
    if len(lhs_shape) == 1 and len(rhs_shape) == 2:
        k, n = rhs_shape

        builder.add_line(f"FOR j := 0 TO {n - 1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {k - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + {lhs_var.at('i')} * {rhs_var.at(f'i * {n} + j')} ;"
                )
            builder.add_line("END_FOR;")
            builder.add_line(f"{output_var.at('j')} := sum;")
        builder.add_line("END_FOR;")
        return builder.build()

    # Matrix-vector: (M,K) @ (K,) -> (M,)
    if len(lhs_shape) == 2 and len(rhs_shape) == 1:
        m, k = lhs_shape

        builder.add_line(f"FOR j := 0 TO {m - 1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {k - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + {lhs_var.at(f'j * {k} + i')} * {rhs_var.at('i')} ;"
                )
            builder.add_line("END_FOR;")
            builder.add_line(f"{output_var.at('j')} := sum;")
        builder.add_line("END_FOR;")
        return builder.build()

    # Batched matrix-matrix: (..., M, K) @ (..., K, N) -> (..., M, N)
    if len(lhs_shape) >= 3 and len(rhs_shape) >= 3:
        lhs_batch = lhs_shape[:-2]
        rhs_batch = rhs_shape[:-2]
        if lhs_batch != rhs_batch:
            raise NotImplementedError(
                f"Runtime MatMul layer {layer.layer_id}: unsupported batch broadcast "
                f"lhs_batch={lhs_batch}, rhs_batch={rhs_batch}"
            )

        m, k = lhs_shape[-2], lhs_shape[-1]
        _, n = rhs_shape[-2], rhs_shape[-1]
        batch_count = int(np.prod(lhs_batch)) if lhs_batch else 1

        lhs_batch_stride = m * k
        rhs_batch_stride = k * n
        out_batch_stride = m * n

        builder.add_line(f"FOR b := 0 TO {batch_count - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR j := 0 TO {out_batch_stride - 1} DO")
            with builder.indent():
                builder.add_line("sum := 0.0;")
                builder.add_line(f"FOR i := 0 TO {k - 1} DO")
                with builder.indent():
                    builder.add_line(
                        f"sum := sum + {lhs_var.at(f'b * {lhs_batch_stride} + (j / {n}) * {k} + i')} * "
                        f"{rhs_var.at(f'b * {rhs_batch_stride} + i * {n} + (j MOD {n})')} ;"
                    )
                builder.add_line("END_FOR;")
                builder.add_line(
                    f"{output_var.at(f'b * {out_batch_stride} + j')} := sum;"
                )
            builder.add_line("END_FOR;")
        builder.add_line("END_FOR;")
        return builder.build()

    raise NotImplementedError(
        f"Runtime MatMul layer {layer.layer_id}: unsupported rank combination "
        f"lhs={lhs_shape}, rhs={rhs_shape}"
    )


def generate_einsum_code(
    layer: EinsumLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate Einsum code for equation ``abcd,cde->abe`` with constant RHS."""
    if layer.equation != "abcd,cde->abe":
        raise NotImplementedError(
            f"Einsum layer {layer.layer_id}: unsupported equation {layer.equation}"
        )
    if not layer.output_shape or len(layer.output_shape) != 3:
        raise ValueError(
            f"Einsum layer {layer.layer_id}: expected 3D output shape, got {layer.output_shape}"
        )

    a_dim, b_dim, e_dim = layer.output_shape
    c_dim, d_dim, e_rhs = layer.rhs_shape
    if e_dim != e_rhs:
        raise ValueError(
            f"Einsum layer {layer.layer_id}: output/rhs mismatch e={e_dim} vs {e_rhs}"
        )

    lhs_size = int(layer.input_size)

    builder = STCodeBuilder()
    # Accept either Variable or raw names
    from ..variable import ensure_var

    input_var = ensure_var(input_var, layer.input_shape)
    output_var = ensure_var(output_var, layer.output_shape)
    builder.add_line(
        f"(* Layer {layer.layer_id}: Einsum {layer.equation} {layer.input_shape} -> {layer.output_shape} *)"
    )
    builder.add_line(f"FOR i := 0 TO {a_dim - 1} DO")
    with builder.indent():
        builder.add_line(f"FOR j := 0 TO {b_dim - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR k := 0 TO {e_dim - 1} DO")
            with builder.indent():
                builder.add_line("sum := 0.0;")
                builder.add_line(f"FOR l := 0 TO {c_dim - 1} DO")
                with builder.indent():
                    builder.add_line(f"FOR m := 0 TO {d_dim - 1} DO")
                    with builder.indent():
                        builder.add_line(
                            f"tmp_idx := (((i * {b_dim} + j) * {c_dim} + l) * {d_dim} + m);"
                        )
                        builder.add_line(f"IF tmp_idx < {lhs_size} THEN")
                        with builder.indent():
                            einsum_rhs_var = Variable(
                                name=f"einsum_rhs_{layer.layer_id}",
                                shape=(c_dim * d_dim * e_dim,),
                            )
                            builder.add_line(
                                f"sum := sum + {input_var.at('tmp_idx')} * {einsum_rhs_var.at(f'((l * {d_dim} + m) * {e_dim} + k)')};"
                            )
                        builder.add_line("END_IF;")
                    builder.add_line("END_FOR;")
                builder.add_line("END_FOR;")
                builder.add_line(
                    f"{output_var.at('((i * {b_dim} + j) * {e_dim} + k)')} := sum;"
                )
            builder.add_line("END_FOR;")
        builder.add_line("END_FOR;")
    builder.add_line("END_FOR;")
    return builder.build()


def generate_binary_elementwise_code(
    layer: BinaryElementwiseLayer, input_vars: List[Variable], output_var: Variable
) -> STCode:
    """Generate binary elementwise code for Sub/Mul/Max with optional const RHS."""
    op = layer.operation
    if op not in {"Sub", "Mul", "Max"}:
        raise NotImplementedError(
            f"Binary elementwise op '{op}' is not supported in ST generator"
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: {op} (Element-wise) *)")

    if layer.rhs_const is not None:
        rhs_size = int(layer.rhs_const.size)
        if rhs_size <= 0:
            raise ValueError(f"{op} layer {layer.layer_id}: empty RHS constant")

        rhs_var = Variable(name=f"rhs_const_{layer.layer_id}", shape=(rhs_size,))
        if rhs_size == layer.output_size:
            rhs_expr = rhs_var.at("i")
        else:
            rhs_expr = rhs_var.at(f"i MOD {rhs_size}")

        lhs_expr = input_vars[0].at("i")
    else:
        if len(input_vars) != 2:
            raise ValueError(
                f"{op} layer {layer.layer_id}: expected 2 runtime inputs, got {len(input_vars)}"
            )
        lhs_expr = input_vars[0].at("i")
        rhs_runtime_size = layer.rhs_runtime_size or layer.output_size
        if rhs_runtime_size == layer.output_size:
            rhs_expr = input_vars[1].at("i")
        else:
            rhs_expr = input_vars[1].at(f"i MOD {rhs_runtime_size}")

    if op == "Sub":
        expr = f"{lhs_expr} - {rhs_expr}"
    elif op == "Mul":
        expr = f"{lhs_expr} * {rhs_expr}"
    else:  # Max
        expr = f"MAX({lhs_expr}, {rhs_expr})"

    builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var.at('i')} := {expr};")
    builder.add_line("END_FOR;")
    return builder.build()


def generate_unary_elementwise_code(
    layer: UnaryElementwiseLayer, input_var: Variable, output_var: Variable
) -> STCode:
    """Generate unary elementwise code for Sqrt/Reciprocal/Neg."""
    op = layer.operation
    if op not in {"Sqrt", "Reciprocal", "Neg"}:
        raise NotImplementedError(
            f"Unary elementwise op '{op}' is not supported in ST generator"
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: {op} (Element-wise) *)")
    builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        if op == "Sqrt":
            builder.add_line(f"{output_var.at('i')} := SQRT({input_var.at('i')});")
        elif op == "Reciprocal":
            builder.add_line(f"{output_var.at('i')} := 1.0 / {input_var.at('i')};")
        else:
            builder.add_line(f"{output_var.at('i')} := -{input_var.at('i')};")
    builder.add_line("END_FOR;")
    return builder.build()
