"""Precompute static index patterns (Gather/Slice) when indices are constant.

Annotates the layer with `precomputed_indices` attribute for use by codegen
so runtime indexing can be simplified or replaced by direct accesses.
"""

import logging
import numpy as np

from ..base_pass import OptimizationPass
from ...types import NetworkIR, GatherLayer, SliceLayer

logger = logging.getLogger(__name__)


class IndexPrecomputationPass(OptimizationPass):
    def get_name(self) -> str:
        return "index_precompute"

    def optimize(self, network: NetworkIR) -> None:
        updated = 0

        for layer in list(network.layers.values()):
            # Gather with constant indices
            if isinstance(layer, GatherLayer) and layer.indices is not None:
                try:
                    inds = np.asarray(layer.indices)
                    object.__setattr__(layer, "precomputed_indices", inds)
                    updated += 1
                except Exception:
                    logger.debug(f"Failed to set precomputed indices for {layer.name}")

            # Slice with constant starts/ends/axes
            if isinstance(layer, SliceLayer):
                try:
                    starts = getattr(layer, "starts", None)
                    ends = getattr(layer, "ends", None)
                    axes = getattr(layer, "axes", None)
                    steps = getattr(layer, "steps", None)
                    if starts is not None and ends is not None:
                        # store as numpy arrays for codegen
                        object.__setattr__(layer, "precomputed_slice", {
                            "starts": np.asarray(starts),
                            "ends": np.asarray(ends),
                            "axes": np.asarray(axes) if axes is not None else None,
                            "steps": np.asarray(steps) if steps is not None else None,
                        })
                        updated += 1
                except Exception:
                    logger.debug(f"Failed to precompute slice for {layer.name}")

        if updated:
            logger.info(f"Index precomputation: annotated {updated} layer(s)")
