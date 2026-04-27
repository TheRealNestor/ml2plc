"""Buffer minimization pass.

Produces a compact buffer assignment mapping by grouping tensors of equal
size into shared buffer names. This is a conservative heuristic to reduce
the number of buffers without analyzing lifetimes.
"""

import logging
from collections import defaultdict
import numpy as np

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class BufferMinimizationPass(OptimizationPass):
    def get_name(self) -> str:
        return "buffer_minimization"

    def optimize(self, ir: NetworkIR) -> None:
        # Compute tensor sizes from output_shape
        sizes = {}
        for layer_name in ir.execution_order:
            layer = ir.get_layer(layer_name)
            if hasattr(layer, "output_shape") and layer.output_shape:
                for t in layer.outputs:
                    sizes[t] = int(np.prod(layer.output_shape))

        # Group tensors by size
        groups = defaultdict(list)
        for t, s in sizes.items():
            groups[s].append(t)

        # Assign buffers per group
        self.buffer_assignments = {}
        for i, (s, tensors) in enumerate(groups.items()):
            buf_name = f"buf_{i}"
            for t in tensors:
                self.buffer_assignments[t] = type("Alloc", (), {"buffer_name": buf_name, "size": s})()

        logger.info(f"Buffer minimization grouped {len(sizes)} tensors into {len(groups)} buffers")
