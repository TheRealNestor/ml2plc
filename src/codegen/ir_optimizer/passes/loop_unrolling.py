"""Loop unrolling annotation pass.

Marks small `Loop` or `Scan` layers for unrolling by setting `unrolled=True`.
Actual IR transformation is left to codegen; this pass only annotates loops
that are safe to unroll.
"""

import logging

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class LoopUnrollingPass(OptimizationPass):
    def __init__(self, max_trip_count: int = 8):
        super().__init__()
        self.max_trip_count = int(max_trip_count)

    def get_name(self) -> str:
        return f"loop_unrolling_{self.max_trip_count}"

    def optimize(self, network: NetworkIR) -> None:
        marked = 0

        for layer in list(network.layers.values()):
            # Detect by op_type
            op = getattr(layer, "op_type", "")
            if op not in ("Loop", "Scan"):
                continue

            # Try to find a trip count attribute
            trip = getattr(layer, "trip", None) or getattr(layer, "sequence_length", None)
            try:
                if trip is not None and int(trip) <= self.max_trip_count:
                    object.__setattr__(layer, "unrolled", True)
                    marked += 1
            except Exception:
                continue

        if marked:
            logger.info(f"Marked {marked} loop(s) for unrolling (max_trip={self.max_trip_count})")
