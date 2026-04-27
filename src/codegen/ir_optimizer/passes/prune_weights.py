"""Prune small weights by zeroing values below a threshold.

This pass zeros-out small weight elements to introduce sparsity.
It preserves dtype and records `pruned_fraction` on the layer for diagnostics.
"""

import logging
import numpy as np

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class PruneWeightsPass(OptimizationPass):
    def __init__(self, threshold: float = 1e-3):
        super().__init__()
        self.threshold = float(threshold)

    def get_name(self) -> str:
        return f"prune_weights_{self.threshold}"

    def optimize(self, network: NetworkIR) -> None:
        updated = 0

        for layer in list(network.layers.values()):
            if not hasattr(layer, "weights"):
                continue

            w = getattr(layer, "weights")
            if w is None:
                continue

            arr = np.asarray(w)
            if not np.issubdtype(arr.dtype, np.floating):
                continue

            mask = np.abs(arr) < self.threshold
            if not np.any(mask):
                continue

            pruned = arr.copy()
            pruned[mask] = 0
            try:
                object.__setattr__(layer, "weights", pruned.astype(arr.dtype, copy=False))
                pruned_fraction = float(np.sum(mask) / mask.size)
                object.__setattr__(layer, "pruned_fraction", pruned_fraction)
                updated += 1
            except Exception:
                logger.debug(f"Failed to write pruned weights for {layer.name}")

        if updated:
            logger.info(f"Pruned weights on {updated} layer(s) (threshold={self.threshold})")
