import pytest
from unittest.mock import MagicMock
from codegen.ir_to_st.codegen_core import (
    translate_model_to_st,
    generate_model_function_block,
)
from codegen.types import (
    ModelIR,
    RegionIR,
    AcyclicRegionIR,
    RecurrentRegionIR,
    RegionKind,
    NetworkIR,
)
from codegen.ir_optimizer import OptimizationResult
from codegen.ir_to_st.st_code import STCode


def test_translate_model_to_st_single_acyclic_region():
    """Test that a single acyclic region is correctly lowered to ST."""
    # Setup - create a minimal NetworkIR mock with required I/O tensors
    mock_ir = MagicMock(spec=NetworkIR)
    mock_ir.input_tensors = ("input_data",)  # Must be non-empty
    mock_ir.output_tensors = ("output_data",)  # Must be non-empty
    mock_layer = MagicMock()
    mock_layer.input_type = 1  # ONNX type constant for FLOAT
    mock_layer.output_type = 1  # ONNX type constant for FLOAT
    mock_layer.input_size = 10
    mock_layer.output_size = 5
    mock_ir.layers = {"layer0": mock_layer}
    mock_ir.execution_order = ["layer0"]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "codegen.ir_to_st.forward_pass.generate_forward_pass",
            lambda ir, allocs: STCode.from_lines("    (* Mock Forward Pass *)"),
        )

        # Use proper typed region
        region = AcyclicRegionIR(
            region_id="r0", kind=RegionKind.ACYCLIC, graph=MagicMock()
        )
        model = ModelIR(regions=[region])

        opt_results = {"r0": OptimizationResult(ir=mock_ir, buffer_allocations={})}

        # Execute
        code = translate_model_to_st(model, opt_results, fb_name="TestFB")

        # Verify - check for key structural elements instead of exact strings
        assert "FUNCTION_BLOCK TestFB" in code
        # Should reference the region (exact format may vary)
        assert "r0" in code and "ACYCLIC" in code
        # Should include the mock forward pass
        assert "Mock Forward Pass" in code
        assert "END_FUNCTION_BLOCK" in code


def test_translate_model_to_st_multi_region_mixed():
    """Test mixed acyclic and recurrent regions are lowered correctly."""
    with pytest.MonkeyPatch.context() as m:
        # Mock the functions that need proper setup
        m.setattr(
            "codegen.ir_to_st.forward_pass.generate_forward_pass",
            lambda ir, allocs: STCode.from_lines("    (* Mock Forward Pass *)"),
        )

        # Create proper region types with required I/O tensors
        mock_ir1 = MagicMock(spec=NetworkIR)
        mock_ir1.input_tensors = ("input_data",)  # Must be non-empty
        mock_ir1.output_tensors = ("output_data",)  # Must be non-empty
        mock_layer1 = MagicMock()
        mock_layer1.input_type = 1  # ONNX type constant for FLOAT
        mock_layer1.output_type = 1  # ONNX type constant for FLOAT
        mock_layer1.input_size = 10
        mock_layer1.output_size = 5
        mock_ir1.layers = {"layer0": mock_layer1}
        mock_ir1.execution_order = ["layer0"]

        mock_ir2 = MagicMock(spec=NetworkIR)
        mock_ir2.input_tensors = ("h_prev",)  # Must be non-empty
        mock_ir2.output_tensors = ("h",)  # Must be non-empty
        mock_layer2 = MagicMock()
        mock_layer2.input_type = 1  # ONNX type constant for FLOAT
        mock_layer2.output_type = 1  # ONNX type constant for FLOAT
        mock_layer2.input_size = 10
        mock_layer2.output_size = 5
        mock_ir2.layers = {"layer0": mock_layer2}
        mock_ir2.execution_order = ["layer0"]

        r1 = AcyclicRegionIR(region_id="r1", kind=RegionKind.ACYCLIC, graph=MagicMock())
        r2 = RecurrentRegionIR(
            region_id="r2",
            kind=RegionKind.RECURRENT,
            graph=MagicMock(),
            state_inputs=("h_prev",),
            state_outputs=("h",),
        )
        model = ModelIR(regions=[r1, r2])

        opt_results = {
            "r1": OptimizationResult(ir=mock_ir1, buffer_allocations={}),
            "r2": OptimizationResult(ir=mock_ir2, buffer_allocations={}),
        }

        # Execute
        code = translate_model_to_st(model, opt_results, fb_name="MixedFB")

        # Verify - be flexible about exact format, check for key elements
        assert "FUNCTION_BLOCK MixedFB" in code
        # Should reference both regions
        assert "r1" in code and "ACYCLIC" in code
        assert "r2" in code and "RECURRENT" in code
        # Should include the mock forward pass
        assert "Mock Forward Pass" in code
        # Should have loop structure for recurrent
        assert "FOR" in code and "END_FOR" in code


def test_translate_model_to_st_missing_optimization_result():
    """Test that missing optimization results are handled gracefully."""
    # Setup - use proper typed region
    region = AcyclicRegionIR(
        region_id="r_missing", kind=RegionKind.ACYCLIC, graph=MagicMock()
    )
    model = ModelIR(regions=[region])
    opt_results = {}  # Empty - no optimization result for this region

    # Execute - should either raise or handle gracefully
    # We test that it doesn't silently succeed with wrong data
    try:
        code = translate_model_to_st(model, opt_results, fb_name="MissingFB")
        # If it succeeds, should have error indication or comment
        assert "Error" in code or "missing" in code.lower()
    except KeyError:
        # Also acceptable: explicit error for missing optimization
        pass
