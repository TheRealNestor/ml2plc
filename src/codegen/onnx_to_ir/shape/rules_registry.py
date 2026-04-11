"""Shape semantics context, rule registry, and op-rule dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ...shape_semantics import TensorRole
from .primitives import (
    infer_add_output_shape,
    infer_batchnorm_output_shape,
    infer_conv2d_output_shape,
    infer_einsum_output_shape,
    infer_expand_output_shape,
    infer_flatten_output_shape,
    infer_gemm_output_shape,
    infer_global_avg_pool_output_shape,
    infer_matmul_output_shape,
    infer_pool2d_output_shape,
    infer_reduce_mean_output_shape,
    infer_reshape_output_shape,
    infer_slice_output_shape,
    infer_squeeze_output_shape,
    infer_transpose_output_shape,
    infer_unsqueeze_output_shape,
)

logger = logging.getLogger(__name__)


def extract_reshape_target_shape(
    resolved_inputs: List[Any],
) -> Optional[Tuple[int, ...]]:
    if len(resolved_inputs) <= 1:
        return None

    shape_tensor = resolved_inputs[1]
    if not shape_tensor.is_weight or shape_tensor.value is None:
        return None

    return tuple(int(d) for d in shape_tensor.value if int(d) != 0)


def first_resolved_output_shape(resolved_outputs: List[Any]) -> Tuple[int, ...]:
    if resolved_outputs and resolved_outputs[0].shape:
        return tuple(resolved_outputs[0].shape)
    return ()


def extract_int_tuple_from_input(
    resolved_inputs: List[Any],
    input_index: int,
) -> Tuple[int, ...]:
    if len(resolved_inputs) <= input_index:
        return ()
    value = resolved_inputs[input_index].value
    if value is None:
        return ()
    return tuple(int(v) for v in value.flatten().tolist())


def resolved_input_role(
    layer_dict: Optional[Dict[str, Any]],
    input_index: int,
) -> TensorRole:
    if not layer_dict:
        return TensorRole.VALUE

    semantics = layer_dict.get("_shape_semantics")
    input_names = layer_dict.get("inputs", [])
    if semantics is None or input_index >= len(input_names):
        return TensorRole.VALUE

    tensor_name = input_names[input_index]
    if not tensor_name:
        return TensorRole.VALUE

    return semantics.role_of(tensor_name)


@dataclass(frozen=True)
class OpSemanticsContext:
    op_type: str
    input_shape: Tuple[int, ...]
    resolved_inputs: List[Any]
    resolved_outputs: List[Any]
    attrs: Dict[str, Any]
    layer_dict: Optional[Dict[str, Any]]


OpShapeRule = Callable[[OpSemanticsContext], Tuple[int, ...]]
OP_SHAPE_RULES: Dict[str, OpShapeRule] = {}


def register_op_shape_rule(*op_types: str) -> Callable[[OpShapeRule], OpShapeRule]:
    def _decorator(func: OpShapeRule) -> OpShapeRule:
        for op in op_types:
            OP_SHAPE_RULES[op] = func
        return func

    return _decorator


def _layer_name(ctx: OpSemanticsContext) -> str:
    if ctx.layer_dict:
        return str(ctx.layer_dict.get("name", "?"))
    return "?"


def _require_shape_role_inputs(
    ctx: OpSemanticsContext, indices: Tuple[int, ...]
) -> None:
    for idx in indices:
        if idx >= len(ctx.resolved_inputs):
            continue
        role = resolved_input_role(ctx.layer_dict, idx)
        if role != TensorRole.SHAPE:
            raise ValueError(
                f"{ctx.op_type} '{_layer_name(ctx)}' expects SHAPE tensor "
                f"at input[{idx}], got {role.value}"
            )


@register_op_shape_rule(
    "Dropout",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "QuantizeLinear",
    "DequantizeLinear",
    "Cast",
    "Sqrt",
    "Reciprocal",
)
def _rule_passthrough(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    return ctx.input_shape


@register_op_shape_rule("Sub", "Mul", "Max", "Concat", "Gather", "Shape")
def _rule_prefer_resolved_output(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    return first_resolved_output_shape(ctx.resolved_outputs) or ctx.input_shape


@register_op_shape_rule("MatMul")
def _rule_matmul(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    if len(ctx.resolved_inputs) < 2:
        return ctx.input_shape
    return infer_matmul_output_shape(ctx.input_shape, ctx.resolved_inputs[1].shape)


@register_op_shape_rule("Gemm", "FusedGemm")
def _rule_gemm(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    if len(ctx.resolved_inputs) < 2:
        return ctx.input_shape
    transB = ctx.attrs.get("transB", 0) == 1
    return infer_gemm_output_shape(
        ctx.input_shape, ctx.resolved_inputs[1].shape, transB
    )


@register_op_shape_rule("Add")
def _rule_add(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    if len(ctx.resolved_inputs) > 1:
        return infer_add_output_shape(ctx.input_shape, ctx.resolved_inputs[1].shape)
    return ctx.input_shape


@register_op_shape_rule("Reshape")
def _rule_reshape(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    strict = resolved_input_role(ctx.layer_dict, 0) == TensorRole.SHAPE
    if (
        strict
        and len(ctx.resolved_inputs) > 1
        and resolved_input_role(ctx.layer_dict, 1) != TensorRole.SHAPE
    ):
        raise ValueError(
            f"Reshape '{_layer_name(ctx)}' expects SHAPE tensor at input[1], "
            f"got {resolved_input_role(ctx.layer_dict, 1).value}"
        )
    target_shape = extract_reshape_target_shape(ctx.resolved_inputs)
    return infer_reshape_output_shape(ctx.input_shape, target_shape)


@register_op_shape_rule("Conv")
def _rule_conv(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    if len(ctx.resolved_inputs) < 2:
        return ctx.input_shape
    strides = tuple(ctx.attrs.get("strides", [1, 1]))
    pads = tuple(ctx.attrs.get("pads", [0, 0, 0, 0]))
    dilations = tuple(ctx.attrs.get("dilations", [1, 1]))
    return infer_conv2d_output_shape(
        ctx.input_shape,
        ctx.resolved_inputs[1].shape,
        strides,
        pads,
        dilations,
    )


@register_op_shape_rule("MaxPool", "AveragePool")
def _rule_pool(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    kernel_shape = tuple(ctx.attrs.get("kernel_shape", [2, 2]))
    strides = tuple(ctx.attrs.get("strides", [1, 1]))
    pads = tuple(ctx.attrs.get("pads", [0, 0, 0, 0]))
    return infer_pool2d_output_shape(ctx.input_shape, kernel_shape, strides, pads)


@register_op_shape_rule("GlobalAveragePool")
def _rule_global_avg_pool(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    return infer_global_avg_pool_output_shape(ctx.input_shape)


@register_op_shape_rule("Flatten")
def _rule_flatten(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    axis = ctx.attrs.get("axis", 1)
    return infer_flatten_output_shape(ctx.input_shape, axis)


@register_op_shape_rule("Transpose")
def _rule_transpose(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    perm = tuple(ctx.attrs.get("perm", ()))
    strict = resolved_input_role(ctx.layer_dict, 0) == TensorRole.SHAPE
    return infer_transpose_output_shape(
        ctx.input_shape,
        perm,
        strict=strict,
        context=f"Transpose '{_layer_name(ctx)}'",
    )


@register_op_shape_rule("BatchNormalization")
def _rule_batchnorm(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    return infer_batchnorm_output_shape(ctx.input_shape)


@register_op_shape_rule("Squeeze")
def _rule_squeeze(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    axes = tuple(ctx.attrs.get("axes", ()))
    if not axes and len(ctx.resolved_inputs) > 1 and ctx.resolved_inputs[1].is_weight:
        axes_val = ctx.resolved_inputs[1].value
        if axes_val is not None:
            axes = tuple(int(a) for a in axes_val)
    if axes and any(a > 0 for a in axes):
        axes = tuple(a - 1 for a in axes if a != 0)
    return infer_squeeze_output_shape(ctx.input_shape, axes)


@register_op_shape_rule("Unsqueeze")
def _rule_unsqueeze(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    strict = resolved_input_role(ctx.layer_dict, 0) == TensorRole.SHAPE
    if strict:
        _require_shape_role_inputs(ctx, (1,))
    axes = extract_int_tuple_from_input(ctx.resolved_inputs, 1)
    if not axes:
        axes = tuple(ctx.attrs.get("axes", ()))
    return infer_unsqueeze_output_shape(
        ctx.input_shape,
        axes,
        strict=strict,
        context=f"Unsqueeze '{_layer_name(ctx)}'",
    )


@register_op_shape_rule("Slice")
def _rule_slice(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    strict = resolved_input_role(ctx.layer_dict, 0) == TensorRole.SHAPE
    if strict:
        _require_shape_role_inputs(ctx, (1, 2, 3, 4))
    starts = extract_int_tuple_from_input(ctx.resolved_inputs, 1)
    ends = extract_int_tuple_from_input(ctx.resolved_inputs, 2)
    axes = extract_int_tuple_from_input(ctx.resolved_inputs, 3)
    steps = extract_int_tuple_from_input(ctx.resolved_inputs, 4)
    return infer_slice_output_shape(
        ctx.input_shape,
        starts,
        ends,
        axes,
        steps,
        strict=strict,
        context=f"Slice '{_layer_name(ctx)}'",
    )


@register_op_shape_rule("Expand")
def _rule_expand(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    strict = resolved_input_role(ctx.layer_dict, 0) == TensorRole.SHAPE
    if strict:
        _require_shape_role_inputs(ctx, (1,))
    target_shape = extract_int_tuple_from_input(ctx.resolved_inputs, 1)
    return infer_expand_output_shape(
        ctx.input_shape,
        target_shape or None,
        strict=strict,
        context=f"Expand '{_layer_name(ctx)}'",
    )


@register_op_shape_rule("ReduceMean", "ReduceProd")
def _rule_reduce(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    axes = tuple(int(a) for a in ctx.attrs.get("axes", ()))
    keepdims = bool(ctx.attrs.get("keepdims", 1))
    return infer_reduce_mean_output_shape(ctx.input_shape, axes, keepdims)


@register_op_shape_rule("Einsum")
def _rule_einsum(ctx: OpSemanticsContext) -> Tuple[int, ...]:
    equation = str(ctx.attrs.get("equation", ""))
    rhs_shape = (
        tuple(ctx.resolved_inputs[1].shape)
        if len(ctx.resolved_inputs) > 1 and ctx.resolved_inputs[1].shape
        else ()
    )
    return infer_einsum_output_shape(equation, ctx.input_shape, rhs_shape)


def infer_output_shape_from_semantics(
    op_type: str,
    input_shape: Tuple[int, ...],
    resolved_inputs: List[Any],
    resolved_outputs: List[Any],
    attrs: Dict[str, Any],
    layer_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[int, ...]:
    ctx = OpSemanticsContext(
        op_type=op_type,
        input_shape=input_shape,
        resolved_inputs=resolved_inputs,
        resolved_outputs=resolved_outputs,
        attrs=attrs,
        layer_dict=layer_dict,
    )

    rule = OP_SHAPE_RULES.get(op_type)
    if rule is not None:
        return rule(ctx)

    logger.warning(f"No shape inference for op_type '{op_type}', using input shape")
    return input_shape


__all__ = [
    "OP_SHAPE_RULES",
    "OpSemanticsContext",
    "extract_int_tuple_from_input",
    "extract_reshape_target_shape",
    "first_resolved_output_shape",
    "infer_output_shape_from_semantics",
    "register_op_shape_rule",
    "resolved_input_role",
]
