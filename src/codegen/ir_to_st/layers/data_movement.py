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
)
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
from ..utils.activation_helpers import generate_activation_loop

logger = logging.getLogger(__name__)


def generate_activation_layer_code(
    layer: ActivationLayer, input_var: str, output_var: str
) -> STCode:
    """Generate activation layer code."""
    code = STCode.from_lines(
        f"(* Layer {layer.layer_id}: Activation ({layer.activation.name}) *)"
    )
    code += generate_activation_loop(
        layer.activation, input_var, output_var, layer.output_size
    )
    return code


def generate_add_code(
    layer: AddLayer, input_vars: List[str], output_var: str
) -> STCode:
    """Generate Add layer code (bias addition or element-wise)."""
    builder = STCodeBuilder()

    if layer.bias is not None:
        # Bias addition: output = input + bias
        builder.add_line(f"(* Layer {layer.layer_id}: Add (Bias) *)")
        builder.add_line(f"FOR i := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := {input_vars[0]}[i] + bias_{layer.layer_id}[i];"
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
                f"{output_var}[i] := {input_vars[0]}[i] + {input_vars[1]}[i];"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_reshape_code(
    layer: ReshapeLayer, input_var: str, output_var: str
) -> STCode:
    """Generate Reshape layer code."""
    if layer.input_size != layer.output_size:
        raise NotImplementedError("Reshape with different sizes not implemented.")

    return generate_simple_copy(
        input_var,
        output_var,
        layer.output_size,
        f"Layer {layer.layer_id}: Reshape (copy input to output)",
    )


def generate_quantize_linear_code(
    layer: QuantizeLinearLayer, input_var: str, output_var: str
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
                f"{output_var}[i] := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var}[i] / {scale_val}) + {zero_point_val}), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")
    else:
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var}[i] / scale_{layer.layer_id}[i]) + zero_point_{layer.layer_id}[i]), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_dequantize_linear_code(
    layer: DequantizeLinearLayer, input_var: str, output_var: str
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
                f"{output_var}[i] := {scale_val} * "
                f"{cast_func}({input_var}[i] - {zero_point_val});"
            )
        builder.add_line("END_FOR;")
    else:
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := scale_{layer.layer_id}[i] * "
                f"{cast_func}({input_var}[i] - zero_point_{layer.layer_id}[i]);"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_dropout_code(
    layer: DropoutLayer, input_var: str, output_var: str
) -> STCode:
    """Generate Dropout layer code (identity at inference time)."""
    pass


def generate_flatten_code(
    layer: FlattenLayer, input_var: str, output_var: str
) -> STCode:
    """Generate Flatten layer code (identity copy)."""
    return generate_simple_copy(
        input_var,
        output_var,
        layer.output_size,
        f"Layer {layer.layer_id}: Flatten (axis={layer.axis})",
    )


def generate_squeeze_code(
    layer: SqueezeLayer, input_var: str, output_var: str
) -> STCode:
    """Generate Squeeze layer code (identity copy)."""
    return generate_simple_copy(
        input_var, output_var, layer.output_size, f"Layer {layer.layer_id}: Squeeze"
    )


def generate_cast_code(layer: CastLayer, input_var: str, output_var: str) -> STCode:
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
            layer.output_size,
            f"Layer {layer.layer_id}: Cast (no-op, same type {input_plc})",
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Cast {input_plc} -> {output_plc} *)")

    cast_func = f"{output_plc}_TO_{input_plc}"
    builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {cast_func}({input_var}[i]);")
    builder.add_line("END_FOR;")
    return builder.build()


def generate_slice_code(layer: SliceLayer, input_var: str, output_var: str) -> STCode:
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
        return generate_offset_copy(
            input_var, output_var, layer.output_size, offset, comment
        )

    elif len(layer.axes) == 1 and layer.steps[0] != 1:
        # Strided copy
        start = layer.starts[0]
        step = layer.steps[0]

        logger.debug(f"  Slice: strided copy with start={start}, step={step}")
        return generate_strided_copy(
            input_var, output_var, layer.output_size, step, start, comment
        )

    else:
        # Multi-axis slice — conservative copy
        logger.debug(f"  Slice: multi-axis slice, using conservative copy")
        return generate_simple_copy(
            input_var,
            output_var,
            layer.output_size,
            f"{comment} (multi-axis, conservative)",
        )


def generate_concat_code(
    layer: ConcatLayer, input_vars: List[str], output_var: str
) -> STCode:
    """Generate Concat layer code."""
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Concat axis={layer.axis} inputs={len(input_vars)} *)"
    )

    offset = 0
    for idx, (inp_var, inp_size) in enumerate(zip(input_vars, layer.input_sizes)):
        builder.add_line(f"(* Concat part {idx}: {inp_var} [{inp_size} elems] *)")
        builder.add_line(f"FOR i := 0 TO {inp_size - 1} DO")
        with builder.indent():
            if offset == 0:
                builder.add_line(f"{output_var}[i] := {inp_var}[i];")
            else:
                builder.add_line(f"{output_var}[i + {offset}] := {inp_var}[i];")
        builder.add_line("END_FOR;")
        offset += inp_size

    return builder.build()


def generate_transpose_code(
    layer: TransposeLayer, input_var: str, output_var: str
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
        builder.add_line(f"{output_var}[0] := {input_var}[0];")
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

    builder.add_line(f"{inner_indent}{output_var}[{out_idx}] := {input_var}[{in_idx}];")

    # Close loops
    for d in range(ndim - 1, -1, -1):
        indent = "    " * d
        builder.add_line(f"{indent}END_FOR;")

    return builder.build()


def generate_unsqueeze_code(
    layer: UnsqueezeLayer, input_var: str, output_var: str
) -> STCode:
    """Generate Unsqueeze layer code (identity copy)."""
    return generate_simple_copy(
        input_var,
        output_var,
        layer.output_size,
        f"Layer {layer.layer_id}: Unsqueeze axes={layer.unsqueeze_axes}",
    )


def generate_expand_code(layer: ExpandLayer, input_var: str, output_var: str) -> STCode:
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


def generate_shape_code(layer: ShapeLayer, input_var: str, output_var: str) -> STCode:
    """Generate Shape layer code (should be constant-folded)."""
    builder = STCodeBuilder()
    builder.add_line(
        f"(* Layer {layer.layer_id}: Shape extraction - SHOULD BE CONSTANT-FOLDED *)"
    )
    builder.add_line(
        f"(* Output shape: {layer.output_shape}, input shape: {layer.input_shape} *)"
    )
    return builder.build()


def generate_gather_code(layer: GatherLayer, input_var: str, output_var: str) -> STCode:
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
