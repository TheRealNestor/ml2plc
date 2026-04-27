"""Pass that transposes weight matrices to preferred layout.

This pass transposes `weights` arrays on Linear-like layers where beneficial.
It mutates the layer's `weights` attribute (uses object.__setattr__ for frozen dataclasses)
and records a small descriptive metadata flag `weight_transposed`.
"""

import logging
import numpy as np

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class TransposeWeightsPass(OptimizationPass):
    def get_name(self) -> str:
        return "transpose_weights"

    def optimize(self, network: NetworkIR) -> None:
        changed = 0

        for layer in list(network.layers.values()):
            if not hasattr(layer, "weights"):
                continue

            w = getattr(layer, "weights")
            if w is None:
                continue

            # Only transpose 2-D arrays
            try:
                arr = np.asarray(w)
                if arr.ndim != 2:
                    continue
            except Exception:
                continue

            # Transpose and write back
            new_w = arr.T.copy()
            try:
                object.__setattr__(layer, "weights", new_w)
                object.__setattr__(layer, "weight_transposed", True)
                changed += 1
            except Exception:
                logger.debug(f"Failed to transpose weights for {layer.name}")

        if changed:
            logger.info(f"Transposed weights for {changed} layer(s)")
