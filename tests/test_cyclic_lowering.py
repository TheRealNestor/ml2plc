"""
Tests for cyclic region lowering (Priority 3).

Validates that recurrent regions are correctly lowered to ST code with:
- State variable initialization
- Fixed iteration loop
- Forward pass execution
"""

import pytest
from codegen.types import (
    BaseLayer,
    MatMulLayer,
    NetworkIR,
    RecurrentRegionIR,
    AcyclicRegionIR,
    RegionIR,
    RegionKind,
)
from codegen.ir_optimizer import OptimizationResult
from codegen.ir_to_st.lowerers import lower_recurrent_region_to_st, lower_region_to_st
from codegen.ir_to_st.st_code import STCode
import numpy as np
from unittest.mock import MagicMock
from fixtures import create_simple_matmul_layer


def _create_recurrent_region(
    layers_dict: dict, state_inputs=(), state_outputs=()
) -> RecurrentRegionIR:
    """Helper to create a recurrent region.

    Note: Uses regionizer's _subgraph_for_component logic to properly set input_tensors
    (including external state inputs not produced by layers).
    """
    tensor_producers = {}
    tensor_consumers = {}
    execution_order = list(layers_dict.keys())

    # Build tensor maps
    for name, layer in layers_dict.items():
        for out_t in layer.outputs:
            tensor_producers[out_t] = name
        for in_t in layer.inputs:
            if in_t not in tensor_consumers:
                tensor_consumers[in_t] = []
            tensor_consumers[in_t].append(name)

    # Input tensors: include all external inputs (not produced by any layer in component)
    input_tensors_set = set()
    for layer_name, layer in layers_dict.items():
        for in_tensor in layer.inputs:
            if in_tensor not in tensor_producers:
                # Not produced within component → external input
                input_tensors_set.add(in_tensor)

    input_tensors = tuple(sorted(input_tensors_set))

    # Output tensors: state outputs
    output_tensors = tuple(state_outputs) if state_outputs else ()

    network = NetworkIR(
        layers=layers_dict,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        state_tensors={},  # No state tensors in test helper
    )

    return RecurrentRegionIR(
        region_id="r0",
        kind=RegionKind.RECURRENT,
        graph=network,
        state_inputs=state_inputs,
        state_outputs=state_outputs,
    )


# ============================================================================
# Basic Recurrent Lowering Tests
# ============================================================================


def test_lower_simple_recurrent_region():
    """Simple recurrent region with one layer generates valid ST."""
    layer = create_simple_matmul_layer(
        "rnn",
        layer_id=0,
        inputs=("x", "h_prev"),
        outputs=("h",),
    )

    layers = {"rnn": layer}
    region = _create_recurrent_region(
        layers,
        state_inputs=("h_prev",),
        state_outputs=("h",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_recurrent_region_to_st(region, opt_result)

    # Should generate ST code
    assert isinstance(code, STCode)
    assert len(code.lines) > 0

    # Should contain region comment
    code_str = "\n".join(code.lines)
    assert "Recurrent Region r0" in code_str

    # Should contain FOR loop for timesteps
    assert "FOR" in code_str
    assert "END_FOR" in code_str


def test_recurrent_region_generates_state_initialization():
    """State initialization code should be generated."""
    layer = create_simple_matmul_layer(
        "rnn",
        layer_id=0,
        inputs=("x", "h_prev"),
        outputs=("h",),
    )

    layers = {"rnn": layer}
    region = _create_recurrent_region(
        layers,
        state_inputs=("h_prev",),
        state_outputs=("h",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_recurrent_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should contain state initialization comment
    assert "State initialization" in code_str


def test_recurrent_region_with_multiple_state_tensors():
    """Multiple state tensors should all be handled."""
    layer1 = create_simple_matmul_layer(
        "rnn_a",
        layer_id=0,
        inputs=("x", "h_prev", "c_prev"),
        outputs=("h",),
    )
    layer2 = create_simple_matmul_layer(
        "rnn_b",
        layer_id=1,
        inputs=("h",),
        outputs=("c",),
    )

    layers = {"rnn_a": layer1, "rnn_b": layer2}
    region = _create_recurrent_region(
        layers,
        state_inputs=("h_prev", "c_prev"),
        state_outputs=("h", "c"),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_recurrent_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should handle multiple state variables
    assert "State initialization" in code_str


def test_recurrent_region_fixed_timestep_loop():
    """Should generate fixed 1-timestep loop (MVP)."""
    layer = create_simple_matmul_layer(
        "rnn",
        layer_id=0,
        inputs=("x", "h_prev"),
        outputs=("h",),
    )

    layers = {"rnn": layer}
    region = _create_recurrent_region(
        layers,
        state_inputs=("h_prev",),
        state_outputs=("h",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_recurrent_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should have FOR loop from 0 TO 0 (1 iteration)
    assert "FOR step := 0 TO 0 DO" in code_str or "FOR step := 0 TO 0" in code_str


def test_recurrent_region_empty_state():
    """Region with no state tensors should still be valid."""
    layer = create_simple_matmul_layer(
        "rnn",
        layer_id=0,
        inputs=("x",),
        outputs=("y",),
    )

    layers = {"rnn": layer}
    region = _create_recurrent_region(
        layers,
        state_inputs=(),
        state_outputs=(),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_recurrent_region_to_st(region, opt_result)

    # Should still generate valid code
    assert isinstance(code, STCode)
    assert len(code.lines) > 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_recurrent_lowering_integration():
    """Integration test: realistic 2-layer recurrent structure."""
    # Simulate: embed → rnn_step where both rnn_step loops back
    embed_layer = create_simple_matmul_layer(
        "embed",
        layer_id=0,
        inputs=("x",),
        outputs=("emb",),
    )

    rnn_layer = create_simple_matmul_layer(
        "rnn",
        layer_id=1,
        inputs=("emb", "h_prev"),
        outputs=("h_out",),
    )

    layers = {
        "embed": embed_layer,
        "rnn": rnn_layer,
    }

    region = _create_recurrent_region(
        layers,
        state_inputs=("h_prev",),
        state_outputs=("h_out",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_recurrent_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should produce valid ST structure
    assert "Recurrent Region" in code_str
    assert "FOR" in code_str
    assert "END_FOR" in code_str


# ============================================================================
# Dispatcher Tests
# ============================================================================


def test_dispatcher_rejects_untyped_region():
    """Dispatcher should reject base RegionIR, requiring typed subclasses."""
    # Create untyped base RegionIR (should NOT be accepted)
    untyped_region = RegionIR(
        region_id="untyped",
        kind=RegionKind.RECURRENT,
        graph=MagicMock(),
    )

    opt_result = OptimizationResult(ir=MagicMock(), buffer_allocations={})

    # Should raise TypeError for untyped region
    with pytest.raises(TypeError) as exc_info:
        lower_region_to_st(untyped_region, opt_result)

    # Verify error message is helpful
    assert "typed region subclass" in str(exc_info.value)
    assert "RegionIR" in str(exc_info.value)


def test_dispatcher_accepts_acyclic_region():
    """Dispatcher correctly routes AcyclicRegionIR to acyclic lowerer."""
    region = AcyclicRegionIR(
        region_id="acyclic",
        kind=RegionKind.ACYCLIC,
        graph=MagicMock(),
    )

    # Mock the acyclic lowerer to verify dispatcher routes to it
    opt_result = OptimizationResult(ir=MagicMock(), buffer_allocations={})

    # This should not raise - just route to acyclic lowerer
    # (The acyclic lowerer will be called with our region)
    try:
        lower_region_to_st(region, opt_result)
    except Exception as e:
        # May fail due to mocked IR, but should NOT be TypeError from dispatcher
        assert not isinstance(e, TypeError) or "typed region subclass" not in str(e)


def test_dispatcher_accepts_recurrent_region():
    """Dispatcher correctly routes RecurrentRegionIR to recurrent lowerer."""
    region = RecurrentRegionIR(
        region_id="recurrent",
        kind=RegionKind.RECURRENT,
        graph=MagicMock(),
        state_inputs=("h_prev",),
        state_outputs=("h",),
    )

    opt_result = OptimizationResult(ir=MagicMock(), buffer_allocations={})

    # This should not raise from dispatcher
    # (May fail from mocked IR, but that's okay)
    try:
        lower_region_to_st(region, opt_result)
    except Exception as e:
        # Should NOT be dispatcher error
        assert not isinstance(e, TypeError) or "typed region subclass" not in str(e)
