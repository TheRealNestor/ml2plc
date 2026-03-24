import pytest

from codegen.types import (
    NetworkIR,
    BaseLayer,
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


def _make_layer(name: str, op_type: str, inputs=(), outputs=()):
    return BaseLayer(
        layer_id=0,
        name=name,
        op_type=op_type,
        input_size=1,
        output_size=1,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def test_regionize_network_ir_creates_single_acyclic_region():
    ir = _make_empty_network_ir()

    model_ir = regionize_network_ir(ir)

    assert len(model_ir.regions) == 1
    assert model_ir.regions[0].kind == RegionKind.ACYCLIC
    assert model_ir.metadata.get("regionizer") == "scc_partitioner"


def test_regionizer_detects_recurrent_cycle_scc():
    layers = {
        "A": _make_layer("A", "Add", inputs=("x", "c_out"), outputs=("a_out",)),
        "B": _make_layer("B", "Relu", inputs=("a_out",), outputs=("b_out",)),
        "C": _make_layer("C", "Add", inputs=("b_out",), outputs=("c_out",)),
    }
    ir = NetworkIR(
        layers=layers,
        execution_order=["A", "B", "C"],
        tensor_producers={
            "a_out": "A",
            "b_out": "B",
            "c_out": "C",
        },
        tensor_consumers={
            "x": ["A"],
            "a_out": ["B"],
            "b_out": ["C"],
            "c_out": ["A"],
        },
        input_tensors=("x",),
        output_tensors=("c_out",),
    )

    model_ir = regionize_network_ir(ir)

    assert len(model_ir.regions) == 1
    assert model_ir.regions[0].kind == RegionKind.RECURRENT
    assert set(model_ir.regions[0].graph.layers.keys()) == {"A", "B", "C"}


def test_regionizer_marks_loop_ops_as_loop_region():
    layers = {
        "L": _make_layer("L", "Loop", inputs=("x",), outputs=("y",)),
    }
    ir = NetworkIR(
        layers=layers,
        execution_order=["L"],
        tensor_producers={"y": "L"},
        tensor_consumers={"x": ["L"]},
        input_tensors=("x",),
        output_tensors=("y",),
    )

    model_ir = regionize_network_ir(ir)

    assert len(model_ir.regions) == 1
    assert model_ir.regions[0].kind == RegionKind.LOOP


def test_capability_validation_fails_for_unsupported_region_kind():
    graph = NetworkIR(layers={}, execution_order=[])
    unsupported_region = RegionIR(
        region_id="r0",
        kind=RegionKind.RECURRENT,
        graph=graph,
    )
    model_ir = ModelIR(regions=(unsupported_region,))

    with pytest.raises(CapabilityError):
        create_execution_plan(model_ir, default_st_backend_capabilities())


def test_create_execution_plan_passes_for_acyclic_region():
    graph = NetworkIR(layers={}, execution_order=[])
    acyclic_region = RegionIR(region_id="r0", kind=RegionKind.ACYCLIC, graph=graph)
    model_ir = ModelIR(regions=(acyclic_region,))

    plan = create_execution_plan(
        model_ir,
        BackendCapabilities(supports_regions=frozenset({RegionKind.ACYCLIC})),
    )

    assert len(plan.regions) == 1
    assert plan.regions[0].region_id == "r0"
    assert plan.regions[0].kind == RegionKind.ACYCLIC
