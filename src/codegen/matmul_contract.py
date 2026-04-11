"""Shared shape contract for runtime MatMul lowering paths."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RuntimeMatMulContract:
    """Validated runtime matmul shape contract."""

    lhs_shape: Tuple[int, ...]
    rhs_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]


def validate_runtime_matmul_contract(
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
    *,
    context: str,
) -> RuntimeMatMulContract:
    """Validate runtime MatMul shape compatibility and infer output shape.

    Supported rank combinations:
      - (K,)   @ (K,)   -> (1,)
      - (M,K)  @ (K,N)  -> (M,N)
      - (K,)   @ (K,N)  -> (N,)
      - (M,K)  @ (K,)   -> (M,)
    """
    lhs = tuple(int(d) for d in lhs_shape)
    rhs = tuple(int(d) for d in rhs_shape)

    if not lhs or not rhs or any(d <= 0 for d in (*lhs, *rhs)):
        raise ValueError(f"{context}: invalid shapes {lhs} @ {rhs}")

    if len(lhs) == 1 and len(rhs) == 1:
        if lhs[0] != rhs[0]:
            raise ValueError(f"{context}: incompatible shapes {lhs} @ {rhs}")
        return RuntimeMatMulContract(lhs, rhs, (1,))

    if len(lhs) == 2 and len(rhs) == 2:
        if lhs[1] != rhs[0]:
            raise ValueError(f"{context}: incompatible shapes {lhs} @ {rhs}")
        return RuntimeMatMulContract(lhs, rhs, (lhs[0], rhs[1]))

    if len(lhs) == 1 and len(rhs) == 2:
        if lhs[0] != rhs[0]:
            raise ValueError(f"{context}: incompatible shapes {lhs} @ {rhs}")
        return RuntimeMatMulContract(lhs, rhs, (rhs[1],))

    if len(lhs) == 2 and len(rhs) == 1:
        if lhs[1] != rhs[0]:
            raise ValueError(f"{context}: incompatible shapes {lhs} @ {rhs}")
        return RuntimeMatMulContract(lhs, rhs, (lhs[0],))

    raise NotImplementedError(
        f"{context}: unsupported rank combination lhs={lhs}, rhs={rhs}"
    )
