"""Experimental ONNX model heuristics for symbolic-dimension resolution.

This module contains narrowly-scoped, opt-in heuristics that can attempt to
resolve trivial symbolic input dimensions (for example treating a single
symbolic axis with all other axes == 1 as a batch axis of size 1). These
heuristics are NOT used by default; `ONNXModel.load_model()` intentionally
fails fast on unresolved symbolic shapes. Import and call these helpers only
when you explicitly accept the semantic risk and want a convenience fallback
for exploratory runs.

Functions return a boolean indicating whether the ModelProto was modified.
The helpers operate in-place on the provided model object.
"""

from __future__ import annotations

import onnx
import logging

logger = logging.getLogger(__name__)


def heuristically_resolve_symbolic_inputs(model: onnx.ModelProto) -> list:
    """Apply a conservative heuristic to resolve trivial symbolic input dims.

    Heuristic:
      - For each graph input, if exactly one dimension is symbolic (has
        dim_param and dim_value == 0) and all other dimensions are the
        concrete integer 1, set the symbolic dimension to 1 and clear the
        dim_param where possible.

        Returns:
            A list of change records (empty if no changes). Each record is a dict
            containing: {"input": <name>, "axis": <idx>, "old": <old_param>, "new": 1}.

    Notes:
      - This mutates the provided ModelProto in-place. Callers who need to
        preserve the original should make a deep copy before invoking.
      - This heuristic is intentionally narrow to reduce the chance of
        silently changing model semantics, but it is still lossy compared to
        running full shape inference with domain knowledge.
    """
    changes: list[dict] = []

    if model is None or model.graph is None:
        return False

    for inp in model.graph.input:
        try:
            dims = inp.type.tensor_type.shape.dim
        except Exception:
            continue

        # Identify symbolic dims (dim_value == 0 and dim_param present)
        symbolic_axes = [
            i
            for i, d in enumerate(dims)
            if (getattr(d, "dim_value", 0) == 0 and getattr(d, "dim_param", None))
        ]
        if len(symbolic_axes) != 1:
            continue

        # Check other axes are concrete positive integers. We relax the
        # previous requirement of "== 1" to allow common exporter patterns
        # where non-batch dimensions are concrete (e.g. ('unk__N', 5, 1)).
        other_ok = True
        for i, d in enumerate(dims):
            if i in symbolic_axes:
                continue
            if not (getattr(d, "dim_value", 0) and int(getattr(d, "dim_value", 0)) > 0):
                other_ok = False
                break

        if not other_ok:
            continue

        # Apply the fix and record the change
        sym_idx = symbolic_axes[0]
        try:
            old_param = getattr(dims[sym_idx], "dim_param", None) or None
            # Set concrete dim and clear symbolic param
            dims[sym_idx].dim_value = 1
            if hasattr(dims[sym_idx], "ClearField"):
                try:
                    dims[sym_idx].ClearField("dim_param")
                except Exception:
                    # Some onnx library versions may not allow ClearField here
                    dims[sym_idx].dim_param = ""
            changes.append(
                {
                    "input": inp.name,
                    "axis": int(sym_idx),
                    "old": str(old_param) if old_param is not None else None,
                    "new": 1,
                }
            )
            logger.debug(
                f"Heuristically set symbolic dim at index {sym_idx} to 1 for input '{inp.name}'."
            )
        except Exception as exc:
            logger.debug(f"Failed to apply heuristic to input '{inp.name}': {exc}")

    if changes:
        logger.info("Heuristic resolution changed one or more input dims to 1.")

    return changes
