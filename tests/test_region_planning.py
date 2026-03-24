import pytest

from codegen.types import (
    NetworkIR,
    BaseLayer,
    GraphIR,
    RegionIR,
    RegionKind,
    ModelIR,
)
from codegen.onnx_to_ir.regionizer import regionize_network_ir
from codegen.backends.capabilities import (
    BackendCapabilities,
    CapabilityError,
    default_st_backend_capabilities,
)
from codegen.planner import create_execution_plan


def _make_empty_network_ir() -> NetworkIR:
    return NetworkIR(layers={}, execution_order=[])


def test_regionize_network_ir_creates_single_acyclic_region():
    ir = _make_empty_network_ir()

    model_ir = regionize_network_ir(ir)

    assert len(model_ir.regions) == 1
    assert model_ir.regions[0].kind == RegionKind.ACYCLIC
    assert model_ir.metadata.get("regionizer") == "single_acyclic_region"


def test_capability_validation_fails_for_unsupported_region_kind():
    graph = GraphIR(layers={}, execution_order=[])
    unsupported_region = RegionIR(
        region_id="r0",
        kind=RegionKind.RECURRENT,
        graph=graph,
    )
    model_ir = ModelIR(regions=(unsupported_region,))

    with pytest.raises(CapabilityError):
        create_execution_plan(model_ir, default_st_backend_capabilities())


def test_create_execution_plan_passes_for_acyclic_region():
    graph = GraphIR(layers={}, execution_order=[])
    acyclic_region = RegionIR(region_id="r0", kind=RegionKind.ACYCLIC, graph=graph)
    model_ir = ModelIR(regions=(acyclic_region,))

    plan = create_execution_plan(
        model_ir,
        BackendCapabilities(supports_regions=frozenset({RegionKind.ACYCLIC})),
    )

    assert len(plan.regions) == 1
    assert plan.regions[0].region_id == "r0"
    assert plan.regions[0].kind == RegionKind.ACYCLIC
