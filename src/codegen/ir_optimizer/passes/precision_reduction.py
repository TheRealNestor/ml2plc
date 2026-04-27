"""Precision reduction pass.

Rounds floating-point constants (weights, biases, precomputed arrays) to a
specified number of decimal places to reduce generated ST code size while
keeping data types unchanged.

This is a lossy but type-preserving transformation (it does not change dtype).
"""

import logging
import numpy as np
from typing import Any

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class PrecisionReductionPass(OptimizationPass):
    def __init__(self, decimals: int = 3):
        super().__init__()
        self.decimals = int(decimals)

    def get_name(self) -> str:
        return f"precision_reduction_{self.decimals}"

    def _round_value(self, val: Any) -> Any:
        """Round floats or numpy arrays of floats while preserving dtype."""
        if val is None:
            return val

        # Python float
        if isinstance(val, float):
            return round(val, self.decimals)

        # numpy arrays / sequences
        try:
            arr = np.asarray(val)
        except Exception:
            return val

        if not np.issubdtype(arr.dtype, np.floating):
            return val

        # Round while preserving dtype
        rounded = np.round(arr, decimals=self.decimals)
        try:
            return rounded.astype(arr.dtype, copy=False)
        except Exception:
            return rounded

    def optimize(self, network: NetworkIR) -> None:
        updated = 0

        # Attributes commonly holding float constants
        float_attrs = (
            "weights",
            "bias",
            "combined_scale",
            "combined_bias",
            "weight_scale",
            "weight_zero_point",
        )

        for layer in list(network.layers.values()):
            for attr in float_attrs:
                if not hasattr(layer, attr):
                    continue

                try:
                    old = getattr(layer, attr)
                except Exception:
                    continue

                if old is None:
                    continue

                new = self._round_value(old)
                # If value changed, write back preserving dataclass frozen nature
                if not self._equal_floats(old, new):
                    try:
                        object.__setattr__(layer, attr, new)
                        updated += 1
                    except Exception:
                        logger.debug(f"Could not set attribute {attr} on {layer.name}")

        if updated:
            logger.info(f"Precision reduction: updated {updated} attributes (decimals={self.decimals})")

    def _equal_floats(self, a: Any, b: Any) -> bool:
        """Compare numeric or array-like floats for equality after rounding."""
        try:
            arr_a = np.asarray(a)
            arr_b = np.asarray(b)
            if arr_a.shape != arr_b.shape:
                return False
            if np.issubdtype(arr_a.dtype, np.floating):
                return np.allclose(arr_a, arr_b, atol=0.0, rtol=0.0)
            else:
                return np.array_equal(arr_a, arr_b)
        except Exception:
            return a == b
