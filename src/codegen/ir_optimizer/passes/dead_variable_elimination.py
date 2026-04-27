"""Remove layers whose outputs are never consumed and are not network outputs.

This is a conservative dead-variable elimination that removes unreachable
or unused computation from the IR. It is safe for acyclic regions and should
be run before code generation.
"""

import logging

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class DeadVariableEliminationPass(OptimizationPass):
    def get_name(self) -> str:
        return "dead_variable_elim"

    def optimize(self, network: NetworkIR) -> None:
        removed = 0

        # For each layer, if none of its outputs are consumed and none are network outputs, mark for removal
        for layer in list(network.layers.values()):
            outputs = getattr(layer, "outputs", ())
            if not outputs:
                continue

            keep = False
            for out in outputs:
                # If tensor is a network output, keep
                if out in network.output_tensors:
                    keep = True
                    break

                # If any consumer exists for this tensor, keep
                consumers = network.tensor_consumers.get(out, [])
                if consumers:
                    keep = True
                    break

            if not keep:
                self.mark_for_removal(layer)
                removed += 1

        if removed:
            logger.info(f"Marked {removed} unused layer(s) for removal")
