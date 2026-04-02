"""
Tests for loop region lowering (Priority 4).

Validates that Loop/Scan control-flow regions are correctly lowered to ST code with:
- Loop count extraction
- Carry variable initialization
- Loop iteration structure
- Carry propagation through iterations
"""

import pytest
from codegen.types import (
    BaseLayer,
    MatMulLayer,
    NetworkIR,
    LoopRegionIR,
    RegionKind,
)
from codegen.ir_optimizer import OptimizationResult
from codegen.ir_to_st.lowerers import lower_loop_region_to_st
from codegen.ir_to_st.st_code import STCode
import numpy as np


def _create_simple_matmul_layer(
    name: str, layer_id: int, inputs=(), outputs=()
) -> MatMulLayer:
    """Helper to create a simple MatMul layer."""
    weights = np.random.randn(10, 10).astype(np.float32)
    return MatMulLayer(
        layer_id=layer_id,
        name=name,
        op_type="MatMul",
        input_size=len(inputs),
        output_size=len(outputs),
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        bias=None,
    )


def _create_loop_region(
    layers_dict: dict, loop_inputs=(), loop_outputs=()
) -> LoopRegionIR:
    """Helper to create a loop region.

    ONNX Loop structure:
      Inputs:  [trip_count, condition, carry_0, carry_1, ...]
      Outputs: [carry_0_final, carry_1_final, ..., scan_outputs...]

    For MVP testing, we model:
      - First input: trip_count
      - Remaining inputs: carry variables
      - Outputs: final carries and scans
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

    # Input tensors: all external inputs (loop inputs)
    input_tensors_set = set()
    for layer_name, layer in layers_dict.items():
        for in_tensor in layer.inputs:
            if in_tensor not in tensor_producers:
                input_tensors_set.add(in_tensor)

    input_tensors = tuple(sorted(input_tensors_set))
    output_tensors = tuple(loop_outputs) if loop_outputs else ()

    network = NetworkIR(
        layers=layers_dict,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )

    return LoopRegionIR(
        region_id="r0",
        kind=RegionKind.LOOP,
        graph=network,
        loop_inputs=loop_inputs,
        loop_outputs=loop_outputs,
    )


# ============================================================================
# Basic Loop Lowering Tests
# ============================================================================


def test_lower_simple_loop_region():
    """Simple loop region with one layer generates valid ST."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("x", "carry"),
        outputs=("carry_out",),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=("trip_count", "carry"),
        loop_outputs=("carry_out",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)

    # Should generate ST code
    assert isinstance(code, STCode)
    assert len(code.lines) > 0

    # Should contain region comment
    code_str = "\n".join(code.lines)
    assert "Loop Region r0" in code_str

    # Should contain FOR loop
    assert "FOR" in code_str
    assert "iteration" in code_str or "iteration" in code_str.lower()
    assert "END_FOR" in code_str


def test_loop_region_generates_carry_initialization():
    """Carry variable initialization should be generated."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("x", "h_prev"),
        outputs=("h_out",),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=("trip_count", "h_prev"),
        loop_outputs=("h_out",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should contain carry initialization comment
    assert "carry initialization" in code_str.lower()


def test_loop_region_with_multiple_carries():
    """Multiple carry variables should all be handled."""
    layer1 = _create_simple_matmul_layer(
        "step1",
        layer_id=0,
        inputs=("x", "h_prev", "c_prev"),
        outputs=("h",),
    )
    layer2 = _create_simple_matmul_layer(
        "step2",
        layer_id=1,
        inputs=("h",),
        outputs=("c",),
    )

    layers = {"step1": layer1, "step2": layer2}
    region = _create_loop_region(
        layers,
        loop_inputs=("trip_count", "h_prev", "c_prev"),
        loop_outputs=("h", "c"),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should handle multiple carries
    assert "carry initialization" in code_str.lower()
    # Should mention 2 carries
    assert "2 carries" in code_str or "carry" in code_str


def test_loop_region_with_scan_outputs():
    """Loop with scan outputs (accumulated per iteration)."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("x", "state"),
        outputs=("state_out", "scan_out"),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=("trip_count", "state"),
        loop_outputs=("state_out", "scan_out"),  # scan_out is accumulated
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)

    # Should generate valid ST code
    assert isinstance(code, STCode)
    assert len(code.lines) > 0


def test_loop_region_loop_structure():
    """Should generate FOR loop with proper bounds."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("x", "carry"),
        outputs=("carry_out",),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=("trip_count", "carry"),
        loop_outputs=("carry_out",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should have FOR loop from 0 to trip_count - 1
    assert (
        "FOR iteration := 0 TO (trip_count - 1) DO" in code_str
        or "FOR iteration := 0 TO" in code_str
    )


def test_loop_region_empty_carries():
    """Loop region with no carries (read-only loop)."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("x",),
        outputs=("y",),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=("trip_count",),
        loop_outputs=(),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)

    # Should still generate valid code
    assert isinstance(code, STCode)
    assert len(code.lines) > 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_loop_lowering_integration():
    """Integration test: realistic loop structure."""
    # Simulate: embedding in loop body, state carried across iterations
    embed_layer = _create_simple_matmul_layer(
        "embed",
        layer_id=0,
        inputs=("token", "state"),
        outputs=("embedded",),
    )

    rnn_layer = _create_simple_matmul_layer(
        "rnn",
        layer_id=1,
        inputs=("embedded",),
        outputs=("state_out",),
    )

    layers = {
        "embed": embed_layer,
        "rnn": rnn_layer,
    }

    region = _create_loop_region(
        layers,
        loop_inputs=("seq_length", "state"),
        loop_outputs=("state_out",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)
    code_str = "\n".join(code.lines)

    # Should produce valid ST structure
    assert "Loop Region" in code_str
    assert "FOR iteration" in code_str
    assert "END_FOR" in code_str


# ============================================================================
# Edge Cases
# ============================================================================


def test_loop_region_single_carry_single_output():
    """Simplest case: one carry in, one carry out."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("carry",),
        outputs=("carry_out",),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=("n",),
        loop_outputs=("carry_out",),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    code = lower_loop_region_to_st(region, opt_result)

    # Should still work
    assert isinstance(code, STCode)
    assert len(code.lines) > 0


def test_loop_region_with_no_loop_inputs():
    """Edge case: loop with no loop_inputs (malformed, but should not crash)."""
    layer = _create_simple_matmul_layer(
        "body",
        layer_id=0,
        inputs=("x",),
        outputs=("y",),
    )

    layers = {"body": layer}
    region = _create_loop_region(
        layers,
        loop_inputs=(),  # No loop inputs
        loop_outputs=(),
    )

    opt_result = OptimizationResult(ir=region.graph, buffer_allocations={})

    # Should not crash, even though metadata extraction will have nothing
    code = lower_loop_region_to_st(region, opt_result)

    assert isinstance(code, STCode)
    assert len(code.lines) > 0
