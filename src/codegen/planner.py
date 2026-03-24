"""Execution planning for regioned models."""

from dataclasses import dataclass
from typing import Tuple

from .types import ModelIR, RegionKind
from .backends import BackendCapabilities, validate_model_against_capabilities


@dataclass(frozen=True)
class PlannedRegion:
    """Backend-targeted region execution unit."""

    region_id: str
    kind: RegionKind


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated execution plan produced from a ModelIR."""

    model_ir: ModelIR
    regions: Tuple[PlannedRegion, ...]


def create_execution_plan(
    model_ir: ModelIR,
    capabilities: BackendCapabilities,
) -> ExecutionPlan:
    """Validate capabilities and produce a simple ordered execution plan."""
    validate_model_against_capabilities(model_ir, capabilities)

    planned_regions = tuple(
        PlannedRegion(region_id=region.region_id, kind=region.kind)
        for region in model_ir.regions
    )

    return ExecutionPlan(model_ir=model_ir, regions=planned_regions)
