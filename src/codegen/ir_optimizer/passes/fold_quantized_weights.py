"""Fold quantized weights into integer constants.

Converts weight arrays with `weight_scale` and `weight_zero_point` metadata
into integer representations (e.g., uint8). Leaves the metadata in place so
codegen may emit dequantization where needed.
"""

import logging
import numpy as np

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class FoldQuantizedWeightsPass(OptimizationPass):
    def get_name(self) -> str:
        return "fold_quantized_weights"

    def optimize(self, network: NetworkIR) -> None:
        folded = 0

        for layer in list(network.layers.values()):
            if not hasattr(layer, "weights"):
                continue

            w = getattr(layer, "weights")
            if w is None:
                continue

            if not (hasattr(layer, "weight_scale") and hasattr(layer, "weight_zero_point")):
                continue

            try:
                scale = float(np.asarray(layer.weight_scale).item())
                zp = int(np.asarray(layer.weight_zero_point).item())
            except Exception:
                continue

            arr = np.asarray(w)
            # Only fold floating weights
            if not np.issubdtype(arr.dtype, np.floating):
                continue

            q = np.round(arr / scale).astype(np.int32) + zp
            # Clip to uint8 range
            q = np.clip(q, 0, 255).astype(np.uint8)

            try:
                object.__setattr__(layer, "weights", q)
                object.__setattr__(layer, "weights_quantized", True)
                folded += 1
            except Exception:
                logger.debug(f"Failed to fold quantized weights for {layer.name}")

        if folded:
            logger.info(f"Folded quantized weights for {folded} layer(s)")
