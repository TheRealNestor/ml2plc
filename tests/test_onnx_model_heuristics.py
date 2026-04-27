import json
import copy
from pathlib import Path

import onnx

from src.codegen.onnx_model_heuristics import heuristically_resolve_symbolic_inputs
from src.codegen.onnx_model import ONNXModel


def make_simple_model_with_symbolic_input(dim_name: str = "unk__N") -> onnx.ModelProto:
    """Create a tiny ONNX model with one input that has a symbolic dim.

    The input shape will be [<symbolic dim>, 1, 1]. The model contains a
    single Identity node connecting input -> output so it's a valid proto.
    """
    from onnx import helper, TensorProto

    # Create ValueInfo with symbolic dim using a dim name string so this test
    # works with a range of onnx versions (some lack make_tensor_dimension_param).
    vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [dim_name, 1, 1])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [dim_name, 1, 1])

    node = helper.make_node("Identity", ["input"], ["output"], name="id")

    graph = helper.make_graph([node], "g", [vi], [out], [])
    model = helper.make_model(graph)
    return model


def test_heuristic_modifies_copy_not_original(tmp_path: Path):
    model = make_simple_model_with_symbolic_input("unk__1")
    original = copy.deepcopy(model)

    # Run heuristic on a copy
    model_copy = copy.deepcopy(model)
    changes = heuristically_resolve_symbolic_inputs(model_copy)

    assert isinstance(changes, list)

    # Heuristic should have made a change for our pattern
    assert len(changes) == 1

    # Original model should remain symbolic
    inp = original.graph.input[0]
    dim = inp.type.tensor_type.shape.dim[0]
    assert getattr(dim, "dim_param", None)


def test_load_model_with_allow_heuristics(tmp_path: Path, monkeypatch):
    model = make_simple_model_with_symbolic_input("unk__2")
    model_path = tmp_path / "m.onnx"
    onnx.save(model, str(model_path))

    # Force validate_model_shapes to report failure so code enters heuristic path
    import src.codegen.onnx_model as onnx_model_mod

    def fake_validate(_m):
        return False, None, [], ["forced"]

    monkeypatch.setattr(onnx_model_mod, "validate_model_shapes", fake_validate)

    om = ONNXModel(str(model_path))
    ok = om.load_model(allow_heuristics=True)
    assert ok is True

    # Sidecar provenance file should exist
    sidecar = model_path.with_name(
        f"{model_path.stem}{model_path.suffix}.heuristics.json"
    )
    assert sidecar.exists()

    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    assert prov.get("heuristic") == "single_symbolic_axis_to_1"
    assert isinstance(prov.get("changes"), list) and len(prov.get("changes")) >= 1

    # The on-disk original file should remain unchanged (still symbolic)
    reloaded = onnx.load(str(model_path))
    dim_param = reloaded.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    assert dim_param
