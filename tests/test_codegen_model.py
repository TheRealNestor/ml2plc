import pytest
from unittest.mock import MagicMock
from codegen.ir_to_st.generator import (
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
    # Setup
    mock_ir = MagicMock(spec=NetworkIR)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "codegen.ir_to_st.generator.generate_forward_pass",
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

        # Verify
        assert "FUNCTION_BLOCK TestFB" in code
        assert "(* Region: r0 [ACYCLIC] *)" in code
        assert "(* Mock Forward Pass *)" in code
        assert "END_FUNCTION_BLOCK" in code


def test_translate_model_to_st_multi_region_mixed():
    """Test mixed acyclic and recurrent regions are lowered correctly."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "codegen.ir_to_st.generator.generate_forward_pass",
            lambda ir, allocs: STCode.from_lines("    (* Mock Forward Pass *)"),
        )

        # Create proper region types
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
            "r1": OptimizationResult(ir=MagicMock(), buffer_allocations={}),
            "r2": OptimizationResult(ir=MagicMock(), buffer_allocations={}),
        }

        # Execute
        code = translate_model_to_st(model, opt_results, fb_name="MixedFB")

        # Verify
        assert "FUNCTION_BLOCK MixedFB" in code
        assert "(* Region: r1 [ACYCLIC] *)" in code
        assert "(* Region: r2 [RECURRENT] *)" in code
        assert "(* Mock Forward Pass *)" in code  # From r1 lowering
        assert "(* Recurrent Region r2 *)" in code  # From r2 lowering
        assert "FOR step" in code  # Recurrent loop structure


def test_translate_model_to_st_missing_optimization_result():
    """Test that missing optimization results produce error comments."""
    # Setup - use proper typed region
    region = AcyclicRegionIR(
        region_id="r_missing", kind=RegionKind.ACYCLIC, graph=MagicMock()
    )
    model = ModelIR(regions=[region])
    opt_results = {}  # Empty - no optimization result for this region

    # Execute
    code = translate_model_to_st(model, opt_results, fb_name="MissingFB")

    # Verify - should have error comment for missing optimization result
    assert "(* Error: No optimization result for r_missing *)" in code
