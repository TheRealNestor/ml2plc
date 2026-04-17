import pytest
from pathlib import Path
from src.codegen.onnx_model import ONNXModel
from src.codegen.onnx_to_ir.tensor_resolution import TensorResolver

MODEL = Path(__file__).resolve().parents[1] / "benchmarks" / "onnx" / "MLP1.onnx"


def test_load_model_fails_on_symbolic_inputs():
    an = ONNXModel(MODEL)
    ok = an.load_model()
    assert (
        ok is False
    ), "load_model should fail (return False) when inputs have symbolic dims"


def test_tensorresolver_raises_for_ambiguous_inputs():
    an = ONNXModel(MODEL)
    # Call load_model() to populate tensor_info (it will return False due to ambiguous shapes)
    an.load_model()
    with pytest.raises(ValueError) as excinfo:
        TensorResolver(analyzer=an)
    assert "unresolved shape" in str(excinfo.value) or "unresolved" in str(
        excinfo.value
    )
