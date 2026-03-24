import pytest
from codegen.main import compile_onnx_to_st
from codegen.types import RegionKind
from unittest.mock import patch, MagicMock


class MockRegion:
    def __init__(self, kind, region_id="r0", layer_count=1):
        self.kind = kind
        self.region_id = region_id
        self.graph = MagicMock()
        self.graph.layers = {
            f"l{i}": MagicMock(op_type="MockOp") for i in range(layer_count)
        }


def test_compile_succeeds_on_multi_region_model():
    """Test that main compiler supports multi-region models."""

    with patch("codegen.main.regionize_network_ir") as mock_regionizer:
        with patch("codegen.main.ONNXModel"), patch("codegen.main.onnx_to_ir"):
            # Mock translate_model_to_st to verifying it gets called
            with patch("codegen.main.translate_model_to_st") as mock_translate:
                with patch("codegen.main.check_memory"):
                    mock_translate.return_value = "FUNCTION_BLOCK NeuralNetwork..."

                    mock_model_ir = MagicMock()
                    # Create two regions
                    r0 = MockRegion(RegionKind.ACYCLIC, "r0")
                    r1 = MockRegion(RegionKind.ACYCLIC, "r1")
                    mock_model_ir.regions = [r0, r1]
                    mock_regionizer.return_value = mock_model_ir

                    output = compile_onnx_to_st(
                        "dummy_path.onnx", output_path="dummy.st"
                    )

                    assert "FUNCTION_BLOCK NeuralNetwork" in output
                    mock_translate.assert_called_once()
                    args, _ = mock_translate.call_args
                    assert args[0] == mock_model_ir


def test_compile_handles_recurrent_region_gracefully():
    """Test that compiler passes recurrent regions to generator."""

    with patch("codegen.main.regionize_network_ir") as mock_regionizer:
        with patch("codegen.main.ONNXModel"), patch("codegen.main.onnx_to_ir"):
            with patch("codegen.main.translate_model_to_st") as mock_translate:
                mock_translate.return_value = "(* Region: r0 [RECURRENT] *)"

                mock_model_ir = MagicMock()
                mock_model_ir.regions = [
                    MockRegion(RegionKind.RECURRENT, "r0", layer_count=5)
                ]
                mock_regionizer.return_value = mock_model_ir

                output = compile_onnx_to_st("dummy_path.onnx", output_path="dummy.st")

                assert "RECURRENT" in output
                mock_translate.assert_called_once()
