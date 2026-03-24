"""Backend capability contracts and planning-time validation."""

from dataclasses import dataclass
from typing import FrozenSet, Tuple

from ..types import ModelIR, RegionKind


@dataclass(frozen=True)
class BackendCapabilities:
    """Describes which region kinds a backend can lower."""

    supports_regions: FrozenSet[RegionKind]
    supports_dynamic_shapes: bool = False


@dataclass(frozen=True)
class CapabilityViolation:
    region_id: str
    region_kind: RegionKind
    message: str


class CapabilityError(ValueError):
    """Raised when model regions cannot be handled by selected backend."""

    def __init__(self, violations: Tuple[CapabilityViolation, ...]):
        self.violations = violations
        details = "\n".join(
            f"- {v.region_id} ({v.region_kind.value}): {v.message}" for v in violations
        )
        super().__init__(
            "Backend capability validation failed:\n"
            f"{details}\n"
            "Tip: choose a backend with matching region support or lower the model to supported regions."
        )


def validate_model_against_capabilities(
    model_ir: ModelIR, capabilities: BackendCapabilities
) -> None:
    """Validate model regions before optimization/codegen.

    This is intentionally early in the pipeline so unsupported regions fail with
    actionable diagnostics instead of deep codegen errors.
    """
    violations = []

    for region in model_ir.regions:
        if region.kind not in capabilities.supports_regions:
            violations.append(
                CapabilityViolation(
                    region_id=region.region_id,
                    region_kind=region.kind,
                    message=f"Region kind '{region.kind.value}' is not supported by backend",
                )
            )

    if violations:
        raise CapabilityError(tuple(violations))


def default_st_backend_capabilities() -> BackendCapabilities:
    """Current ST backend (milestone-1): acyclic only."""
    return BackendCapabilities(supports_regions=frozenset({RegionKind.ACYCLIC}))
