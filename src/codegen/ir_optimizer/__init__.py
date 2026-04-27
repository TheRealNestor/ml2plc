"""
IR optimization module.
"""

from .optimizer import (
    IROptimizer,
    DEFAULT_PASSES,
    optimize_model_regions,
    PROFILE_PASSES,
)

from .result import OptimizationResult
from .base_pass import OptimizationPass
from .passes import (
    RemoveIdentityPass,
    RemoveNoOpReshapePass,
    RemoveRedundantQuantPairPass,
    RemoveWeightDequantPass,
    BufferAllocationPass,
    RemoveDropoutPass,
    InsertQuantizePass,
)

__all__ = [
    "IROptimizer",
    "DEFAULT_PASSES",
    "PROFILE_PASSES",
    "OptimizationPass",
    "RemoveIdentityPass",
    "RemoveNoOpReshapePass",
    "RemoveRedundantQuantPairPass",
    "RemoveWeightDequantPass",
    "BufferAllocationPass",
    "RemoveDropoutPass",
    "OptimizationResult",
    "OptimizeModelRegions",
]
