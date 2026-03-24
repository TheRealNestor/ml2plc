"""
Region-kind-aware optimization strategy dispatch.

Routes optimization passes to appropriate regions based on kind and pass applicability.
"""

import logging
from typing import Dict, List
from .result import OptimizationResult
from ..types import ModelIR
from .base_pass import OptimizationPass


logger = logging.getLogger(__name__)


def optimize_region_with_passes(
    region_ir,
    passes: List[OptimizationPass],
) -> OptimizationResult:
    """
    Apply applicable optimization passes to a single region.

    Only runs passes that declare support for the region's kind.

    Args:
        region_ir: The region to optimize (RegionIR)
        passes: List of OptimizationPass instances to consider

    Returns:
        OptimizationResult with optimized IR
    """
    # Import here to avoid circular dependency
    from .optimizer import IROptimizer

    # Region graph is already a NetworkIR (no conversion needed)
    network_ir = region_ir.graph

    # Filter passes by applicability to this region kind
    applicable_passes = [
        p for p in passes if region_ir.kind in p.supports_region_kinds()
    ]

    if not applicable_passes:
        logger.info(
            f"  No applicable passes for {region_ir.kind.value} region {region_ir.region_id}"
        )
        return OptimizationResult(ir=network_ir)

    logger.info(
        f"  Applying {len(applicable_passes)} pass(es) to {region_ir.kind.value} region {region_ir.region_id}"
    )

    # Use optimizer with filtered passes
    optimizer = IROptimizer(network_ir, passes=applicable_passes)
    return optimizer.optimize()


def validate_pass_applicability(
    model_ir: ModelIR, passes: List[OptimizationPass]
) -> Dict[str, List[str]]:
    """
    Check applicability of all passes to all regions.

    Useful for diagnostics and planning.

    Args:
        model_ir: The model to check
        passes: List of passes to validate

    Returns:
        Dictionary mapping region_id to list of applicable pass names
    """
    applicability = {}

    for region in model_ir.regions:
        applicable = [
            p.get_name() for p in passes if region.kind in p.supports_region_kinds()
        ]
        applicability[region.region_id] = applicable

        status = "applicable" if applicable else "no applicable passes"
        logger.debug(f"  Region {region.region_id} ({region.kind.value}): {status}")

    return applicability
