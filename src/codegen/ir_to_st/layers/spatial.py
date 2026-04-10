"""
Spatial layer code generators (Conv2D, Pool2D, BatchNorm).

Handles convolution, pooling, and batch normalization operations.
"""

import numpy as np
from typing import Optional

from ...types import Conv2DLayer, Pool2DLayer, BatchNormLayer
from ..st_code import STCode, STCodeBuilder


def generate_conv2d_code(layer: Conv2DLayer, input_var: str, output_var: str) -> STCode:
    """Generate Conv2D layer code with 6 nested loops and boundary checking."""
    builder = STCodeBuilder()

    # Unpack spatial parameters
    in_c, in_h, in_w = (
        layer.input_shape[-3],
        layer.input_shape[-2],
        layer.input_shape[-1],
    )
    out_c, out_h, out_w = (
        layer.output_shape[-3],
        layer.output_shape[-2],
        layer.output_shape[-1],
    )
    kH, kW = layer.kernel_shape
    sH, sW = layer.strides
    pH, pW = layer.pads[0], layer.pads[1]
    dH, dW = layer.dilations
    group = layer.group
    in_c_per_group = in_c // group
    w_ic_size = in_c_per_group * kH * kW

    builder.add_line(
        f"(* Layer {layer.layer_id}: Conv2D  in={in_c}x{in_h}x{in_w}  "
        f"out={out_c}x{out_h}x{out_w}  kernel={kH}x{kW}  stride={sH}x{sW}  "
        f"pad={layer.pads}  group={group} *)"
    )

    builder.add_line(f"FOR oc := 0 TO {out_c - 1} DO")
    with builder.indent():
        builder.add_line(f"FOR oh := 0 TO {out_h - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR ow := 0 TO {out_w - 1} DO")
            with builder.indent():
                if layer.bias is not None:
                    builder.add_line(f"sum := bias_{layer.layer_id}[oc];")
                else:
                    builder.add_line("sum := 0.0;")

                # Determine input channel range for this group
                if group == 1:
                    ic_start = "0"
                    ic_end = str(in_c_per_group - 1)
                elif group == in_c:
                    ic_start = "oc"
                    ic_end = "oc"
                else:
                    ic_start = f"(oc * {in_c_per_group} / {out_c // group})"
                    ic_end = f"({ic_start} + {in_c_per_group - 1})"

                builder.add_line(f"FOR ic := {ic_start} TO {ic_end} DO")
                with builder.indent():
                    builder.add_line(f"FOR kh := 0 TO {kH - 1} DO")
                    with builder.indent():
                        builder.add_line(f"FOR kw := 0 TO {kW - 1} DO")
                        with builder.indent():
                            if dH == 1:
                                builder.add_line(f"ih := oh * {sH} - {pH} + kh;")
                            else:
                                builder.add_line(f"ih := oh * {sH} - {pH} + kh * {dH};")
                            if dW == 1:
                                builder.add_line(f"iw := ow * {sW} - {pW} + kw;")
                            else:
                                builder.add_line(f"iw := ow * {sW} - {pW} + kw * {dW};")

                            has_padding = any(p != 0 for p in layer.pads)
                            if has_padding:
                                builder.add_line(
                                    f"IF (ih >= 0) AND (ih < {in_h}) AND (iw >= 0) AND (iw < {in_w}) THEN"
                                )
                                indent_ctx = builder.indent()
                                indent_ctx.__enter__()

                            input_idx = f"ic * {in_h * in_w} + ih * {in_w} + iw"

                            if group == 1:
                                weight_idx = f"oc * {w_ic_size} + ic * {kH * kW} + kh * {kW} + kw"
                            elif group == in_c:
                                weight_idx = f"oc * {kH * kW} + kh * {kW} + kw"
                            else:
                                weight_idx = f"oc * {w_ic_size} + (ic - {ic_start}) * {kH * kW} + kh * {kW} + kw"

                            builder.add_line(
                                f"sum := sum + {input_var}[{input_idx}] "
                                f"* weights_{layer.layer_id}[{weight_idx}];"
                            )

                            if has_padding:
                                indent_ctx.__exit__(None, None, None)
                                builder.add_line("END_IF;")

                        builder.add_line("END_FOR;")
                    builder.add_line("END_FOR;")
                builder.add_line("END_FOR;")

                output_idx = f"oc * {out_h * out_w} + oh * {out_w} + ow"
                builder.add_line(f"{output_var}[{output_idx}] := sum;")

            builder.add_line("END_FOR;")
        builder.add_line("END_FOR;")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_pool2d_code(layer: Pool2DLayer, input_var: str, output_var: str) -> STCode:
    """Generate MaxPool or AveragePool layer code."""
    builder = STCodeBuilder()

    channels = layer.input_shape[-3]
    in_h, in_w = layer.input_shape[-2], layer.input_shape[-1]
    out_h, out_w = layer.output_shape[-2], layer.output_shape[-1]
    kH, kW = layer.kernel_shape
    sH, sW = layer.strides
    pH, pW = layer.pads[0], layer.pads[1]
    is_max = layer.pool_type == "max"

    pool_label = "MaxPool" if is_max else "AvgPool"
    builder.add_line(
        f"(* Layer {layer.layer_id}: {pool_label}  kernel={kH}x{kW}  stride={sH}x{sW} *)"
    )

    builder.add_line(f"FOR oc := 0 TO {channels - 1} DO")
    with builder.indent():
        builder.add_line(f"FOR oh := 0 TO {out_h - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR ow := 0 TO {out_w - 1} DO")
            with builder.indent():
                builder.add_line("sum := -3.402823E+38;" if is_max else "sum := 0.0;")

                builder.add_line(f"FOR kh := 0 TO {kH - 1} DO")
                with builder.indent():
                    builder.add_line(f"FOR kw := 0 TO {kW - 1} DO")
                    with builder.indent():
                        builder.add_line(f"ih := oh * {sH} - {pH} + kh;")
                        builder.add_line(f"iw := ow * {sW} - {pW} + kw;")

                        has_padding = any(p != 0 for p in layer.pads)
                        if has_padding:
                            builder.add_line(
                                f"IF (ih >= 0) AND (ih < {in_h}) AND (iw >= 0) AND (iw < {in_w}) THEN"
                            )
                            indent_ctx = builder.indent()
                            indent_ctx.__enter__()

                        input_idx = f"oc * {in_h * in_w} + ih * {in_w} + iw"
                        if is_max:
                            builder.add_line(f"IF {input_var}[{input_idx}] > sum THEN")
                            with builder.indent():
                                builder.add_line(f"sum := {input_var}[{input_idx}];")
                            builder.add_line("END_IF;")
                        else:
                            builder.add_line(f"sum := sum + {input_var}[{input_idx}];")

                        if has_padding:
                            indent_ctx.__exit__(None, None, None)
                            builder.add_line("END_IF;")

                    builder.add_line("END_FOR;")
                builder.add_line("END_FOR;")

                output_idx = f"oc * {out_h * out_w} + oh * {out_w} + ow"
                if is_max:
                    builder.add_line(f"{output_var}[{output_idx}] := sum;")
                else:
                    kernel_area = kH * kW
                    builder.add_line(
                        f"{output_var}[{output_idx}] := sum / {float(kernel_area)};"
                    )

            builder.add_line("END_FOR;")
        builder.add_line("END_FOR;")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_batchnorm_code(
    layer: BatchNormLayer, input_var: str, output_var: str
) -> STCode:
    """Generate BatchNorm layer code (inference mode)."""
    builder = STCodeBuilder()
    lid = layer.layer_id
    C = layer.num_channels

    # Determine spatial size per channel
    if layer.input_shape and len(layer.input_shape) >= 3:
        spatial_size = int(np.prod(layer.input_shape[1:]))
    elif layer.input_shape and len(layer.input_shape) == 1:
        spatial_size = 1
    else:
        spatial_size = layer.input_size // C if C > 0 else layer.input_size

    builder.add_line(
        f"(* Layer {lid}: BatchNorm  channels={C}  spatial={spatial_size} *)"
    )

    if spatial_size == 1:
        builder.add_line(f"FOR oc := 0 TO {C - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[oc] := bn_scale_{lid}[oc] * {input_var}[oc] "
                f"+ bn_bias_{lid}[oc];"
            )
        builder.add_line("END_FOR;")
    else:
        builder.add_line(f"FOR oc := 0 TO {C - 1} DO")
        with builder.indent():
            builder.add_line(f"FOR i := 0 TO {spatial_size - 1} DO")
            with builder.indent():
                builder.add_line(
                    f"{output_var}[oc * {spatial_size} + i] := "
                    f"bn_scale_{lid}[oc] * {input_var}[oc * {spatial_size} + i] "
                    f"+ bn_bias_{lid}[oc];"
                )
            builder.add_line("END_FOR;")
        builder.add_line("END_FOR;")

    return builder.build()
