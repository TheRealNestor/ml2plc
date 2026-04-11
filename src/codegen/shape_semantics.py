"""Role-aware tensor shape semantics and provenance tracking.

This module is intentionally independent from backend code generation. It provides
small, composable contracts for tensor-role propagation (VALUE vs SHAPE) and a
provenance tree to improve diagnostics during extraction/validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class TensorRole(str, Enum):
    VALUE = "VALUE"
    SHAPE = "SHAPE"


@dataclass(frozen=True)
class TensorSemantics:
    role: TensorRole
    produced_by_node: Optional[str]
    produced_by_op: Optional[str]
    parents: Tuple[str, ...]


ShapeRule = Callable[
    [str, Tuple[TensorRole, ...], Dict],
    Tuple[TensorRole, ...],
]

_SHAPE_RULES: Dict[str, ShapeRule] = {}


def register_shape_rule(op_type: str) -> Callable[[ShapeRule], ShapeRule]:
    """Register a role-propagation rule for an ONNX op type."""

    def _decorator(func: ShapeRule) -> ShapeRule:
        _SHAPE_RULES[op_type] = func
        return func

    return _decorator


def _first_or_value(input_roles: Tuple[TensorRole, ...]) -> TensorRole:
    return input_roles[0] if input_roles else TensorRole.VALUE


@register_shape_rule("Shape")
def _shape_rule(
    op_type: str,
    input_roles: Tuple[TensorRole, ...],
    layer_dict: Dict,
) -> Tuple[TensorRole, ...]:
    return (TensorRole.SHAPE,)


@register_shape_rule("Cast")
@register_shape_rule("Gather")
@register_shape_rule("Unsqueeze")
@register_shape_rule("Transpose")
@register_shape_rule("Slice")
def _passthrough_role_rule(
    op_type: str,
    input_roles: Tuple[TensorRole, ...],
    layer_dict: Dict,
) -> Tuple[TensorRole, ...]:
    return (_first_or_value(input_roles),)


@register_shape_rule("Concat")
def _concat_role_rule(
    op_type: str,
    input_roles: Tuple[TensorRole, ...],
    layer_dict: Dict,
) -> Tuple[TensorRole, ...]:
    if input_roles and all(role == TensorRole.SHAPE for role in input_roles):
        return (TensorRole.SHAPE,)
    return (TensorRole.VALUE,)


@register_shape_rule("Reshape")
def _reshape_role_rule(
    op_type: str,
    input_roles: Tuple[TensorRole, ...],
    layer_dict: Dict,
) -> Tuple[TensorRole, ...]:
    # Reshape consumes a SHAPE tensor but emits a VALUE tensor.
    if len(input_roles) > 1 and input_roles[1] != TensorRole.SHAPE:
        raise ValueError(
            f"Reshape '{layer_dict.get('name', '?')}' expects input[1] to be SHAPE, "
            f"got {input_roles[1].value}"
        )
    return (TensorRole.VALUE,)


@register_shape_rule("MatMul")
def _matmul_role_rule(
    op_type: str,
    input_roles: Tuple[TensorRole, ...],
    layer_dict: Dict,
) -> Tuple[TensorRole, ...]:
    # MatMul should always operate on VALUE tensors.
    if any(role == TensorRole.SHAPE for role in input_roles[:2]):
        raise ValueError(
            f"MatMul '{layer_dict.get('name', '?')}' received SHAPE tensor as data input"
        )
    return (TensorRole.VALUE,)


class ShapeSemanticsTracker:
    """Tracks role/provenance facts for tensors across a topological ONNX walk."""

    def __init__(self, constant_tensors: Optional[Dict[str, np.ndarray]] = None):
        self._facts: Dict[str, TensorSemantics] = {}
        for name, value in (constant_tensors or {}).items():
            self._seed_constant_tensor(name, value)

    def role_of(self, tensor_name: str) -> TensorRole:
        fact = self._facts.get(tensor_name)
        return fact.role if fact else TensorRole.VALUE

    def record_layer(self, layer_dict: Dict) -> None:
        """Apply role rule for the layer and record output provenance."""
        self._seed_shape_like_inputs(layer_dict)

        op_type = str(layer_dict.get("op_type", ""))
        node_name = str(layer_dict.get("name") or op_type or "<unnamed>")
        input_names = tuple(name for name in layer_dict.get("inputs", []) if name)
        output_names = tuple(name for name in layer_dict.get("outputs", []) if name)

        input_roles = tuple(self.role_of(name) for name in input_names)
        rule = _SHAPE_RULES.get(op_type)
        out_roles = (
            rule(op_type, input_roles, layer_dict)
            if rule is not None
            else (_first_or_value(input_roles),)
        )

        if not out_roles:
            out_roles = (TensorRole.VALUE,)
        if len(out_roles) == 1 and len(output_names) > 1:
            out_roles = tuple(out_roles[0] for _ in output_names)

        for i, out_name in enumerate(output_names):
            role = out_roles[i] if i < len(out_roles) else out_roles[-1]
            self._facts[out_name] = TensorSemantics(
                role=role,
                produced_by_node=node_name,
                produced_by_op=op_type,
                parents=input_names,
            )

    def format_lineage(self, tensor_name: str, max_depth: int = 8) -> str:
        lines: List[str] = []
        self._append_lineage_lines(
            tensor_name=tensor_name,
            lines=lines,
            depth=0,
            max_depth=max_depth,
            visited=set(),
        )
        return "\n".join(lines)

    def _append_lineage_lines(
        self,
        *,
        tensor_name: str,
        lines: List[str],
        depth: int,
        max_depth: int,
        visited: set,
    ) -> None:
        indent = "  " * depth
        role = self.role_of(tensor_name).value
        fact = self._facts.get(tensor_name)

        if fact and fact.produced_by_node:
            lines.append(
                f"{indent}- {tensor_name} [{role}] <= "
                f"{fact.produced_by_node} ({fact.produced_by_op})"
            )
        else:
            lines.append(f"{indent}- {tensor_name} [{role}] <= <graph input/constant>")

        if depth >= max_depth or not fact or not fact.parents:
            return

        visit_key = (tensor_name, depth)
        if visit_key in visited:
            return
        visited.add(visit_key)

        for parent in fact.parents:
            self._append_lineage_lines(
                tensor_name=parent,
                lines=lines,
                depth=depth + 1,
                max_depth=max_depth,
                visited=visited,
            )

    def _seed_constant_tensor(self, name: str, value: np.ndarray) -> None:
        role = self._role_for_constant(value)
        self._facts.setdefault(
            name,
            TensorSemantics(
                role=role,
                produced_by_node=None,
                produced_by_op="Constant",
                parents=(),
            ),
        )

    def _seed_shape_like_inputs(self, layer_dict: Dict) -> None:
        resolved_inputs: Sequence = layer_dict.get("resolved_inputs", [])
        input_names: Sequence[str] = layer_dict.get("inputs", [])

        for idx, resolved in enumerate(resolved_inputs):
            if idx >= len(input_names):
                continue

            name = input_names[idx]
            if not name or getattr(resolved, "value", None) is None:
                continue
            if not getattr(resolved, "is_weight", False):
                continue

            value = resolved.value
            role = self._role_for_constant(value)
            # Never downgrade SHAPE -> VALUE if it was already elevated.
            existing = self._facts.get(name)
            if (
                existing
                and existing.role == TensorRole.SHAPE
                and role != TensorRole.SHAPE
            ):
                continue

            self._facts[name] = TensorSemantics(
                role=role,
                produced_by_node=existing.produced_by_node if existing else None,
                produced_by_op=existing.produced_by_op if existing else "Constant",
                parents=existing.parents if existing else (),
            )

    @staticmethod
    def _role_for_constant(value: np.ndarray) -> TensorRole:
        arr = np.asarray(value)
        if arr.ndim <= 1 and np.issubdtype(arr.dtype, np.integer):
            return TensorRole.SHAPE
        return TensorRole.VALUE
