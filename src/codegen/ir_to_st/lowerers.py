"""
Backend lowering: Convert regions to Structured Text by kind.

Separates region-kind-specific ST generation into dedicated lowerers.
Each lowerer handles a specific region type (acyclic, recurrent, loop).

This layer decouples region-specific codegen complexity from the main
generator, allowing independent evolution of each region kind's lowering.
"""

import logging
from typing import Optional, Dict

from ..types import (
    AcyclicRegionIR,
    RecurrentRegionIR,
    LoopRegionIR,
    RegionKind,
    NetworkIR,
)
from ..ir_optimizer import OptimizationResult
from .st_code import STCode

logger = logging.getLogger(__name__)


def lower_acyclic_region_to_st(
    region: AcyclicRegionIR,
    optimization_result: OptimizationResult,
) -> STCode:
    """
    Lower an acyclic (DAG) region to Structured Text.

    Acyclic regions contain only feedforward computation with no cycles.
    Uses standard layer-by-layer topological execution.

    Args:
        region: Acyclic region to lower
        optimization_result: Optimized IR and buffer allocations for this region

    Returns:
        Generated ST code for this region
    """
    from .generator import generate_forward_pass

    logger.debug(f"Lowering acyclic region {region.region_id}")

    ir = optimization_result.ir
    buffer_allocations = optimization_result.buffer_allocations

    # Use existing forward pass generation (DAG-oriented)
    st_code = generate_forward_pass(ir, buffer_allocations)

    return st_code


def lower_recurrent_region_to_st(
    region: RecurrentRegionIR,
    optimization_result: OptimizationResult,
    num_timesteps: int = 1,
) -> STCode:
    """
    Lower a recurrent (cyclic) region to Structured Text.

    Recurrent regions contain cycles and require iterative execution with
    state variables. Typically used for RNNs, LSTMs, GRUs, etc.

    State inputs: fed from previous iteration
    State outputs: passed to next iteration

    Args:
        region: Recurrent region to lower
        optimization_result: Optimized IR for this region
        num_timesteps: Number of unrolled timesteps (default: 1 MVP)

    Returns:
        Generated ST code for this region (with state handling)
    """
    from .generator import generate_forward_pass

    logger.debug(
        f"Lowering recurrent region {region.region_id} "
        f"with state_inputs={region.state_inputs}, state_outputs={region.state_outputs}, "
        f"num_timesteps={num_timesteps}"
    )

    ir = optimization_result.ir
    buffer_allocations = optimization_result.buffer_allocations

    code = STCode.empty()

    # Comment header for region
    code += STCode.from_lines(f"(* Recurrent Region {region.region_id} *)")
    code += STCode.blank_line()

    # Generate state initialization section
    if region.state_inputs and region.state_outputs:
        code += _generate_state_initialization(region, ir, buffer_allocations)
        code += STCode.blank_line()

    # Generate timestep loop
    code += _generate_recurrent_loop(region, ir, buffer_allocations, num_timesteps)
    code += STCode.blank_line()

    return code


def _generate_state_initialization(
    region: RecurrentRegionIR,
    ir: NetworkIR,
    buffer_allocations: Dict[str, str],
) -> STCode:
    """Generate code to initialize state variables.

    For each (state_input, state_output) pair, generates:
        state_var := state_input_var;

    This seeds the recurrent computation with initial state values.
    """
    code = STCode.from_lines("(* State initialization *)")

    for state_in, state_out in zip(region.state_inputs, region.state_outputs):
        input_var = _resolve_variable_name(state_in, buffer_allocations)
        output_var = _resolve_variable_name(state_out, ir, buffer_allocations)

        code += STCode.from_lines(f"{output_var} := {input_var};")

    return code


def _generate_recurrent_loop(
    region: RecurrentRegionIR,
    ir: NetworkIR,
    buffer_allocations: Dict[str, str],
    num_timesteps: int,
) -> STCode:
    """Generate the main recurrent timestep loop.

    Generates:
        FOR step := 0 TO (num_timesteps - 1) DO
            <forward pass>
        END_FOR;

    The forward pass reuses the same acyclic code generation, repeating
    it for each timestep. Future work may unroll or specialize per timestep.
    """
    from .generator import generate_forward_pass

    code = STCode.from_lines(f"(* Recurrent loop: {num_timesteps} timestep(s) *)")

    # Loop bounds: 0 to (num_timesteps - 1)
    end_step = num_timesteps - 1
    code += STCode.from_lines(f"FOR step := 0 TO {end_step} DO")

    # Forward pass body (reuse DAG generation, indent it)
    forward_code = generate_forward_pass(ir, buffer_allocations)
    for line in forward_code.lines:
        code += STCode.from_lines("\t" + line)

    code += STCode.from_lines("END_FOR;")

    return code


def _resolve_variable_name(
    tensor_name: str,
    ir_or_buffer_alloc,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve a tensor name to its variable name.

    Tries in order:
    1. Buffer allocation mapping
    2. Layer output variable (layer_<id>_output)
    3. Fallback to tensor name itself

    This abstracts away the variable naming scheme, making lowering
    more robust to future naming changes.
    """
    # Handle overloaded signature: _resolve_variable_name(tensor, ir, buffer_alloc)
    if buffer_allocations is not None:
        ir = ir_or_buffer_alloc
    else:
        # Called with 2 args: tensor, buffer_alloc_dict (no IR)
        buffer_allocations = ir_or_buffer_alloc
        ir = None

    # Try buffer allocations first
    if tensor_name in buffer_allocations:
        return buffer_allocations[tensor_name]

    # Try to resolve through IR tensor producer
    if ir is not None and tensor_name in ir.tensor_producers:
        producer_name = ir.tensor_producers[tensor_name]
        if producer_name in ir.layers:
            producer = ir.layers[producer_name]
            return f"layer_{producer.layer_id}_output"

    # Fallback: use tensor name as variable
    return tensor_name


def lower_loop_region_to_st(
    region: LoopRegionIR,
    optimization_result: OptimizationResult,
) -> STCode:
    """
    Lower a control-flow loop region to Structured Text.

    Loop regions represent explicit control flow constructs from the ONNX model
    (Loop, Scan operators). They have explicit loop count and carry variables.

    ONNX Loop Operator:
      Inputs:  [trip_count, condition, carry_0, carry_1, ...]
      Body:    Iterative computation with state carry-over
      Outputs: [carry_0_final, carry_1_final, ..., scan_outputs...]

    Strategy:
      1. Initialize carry variables from loop_inputs
      2. FOR loop: 0 to trip_count - 1
      3. Execute body (forward pass)
      4. Collect final carries and scan outputs

    Args:
        region: Loop region to lower (LoopRegionIR)
        optimization_result: Optimized IR for this region

    Returns:
        Generated ST code for this region (with loop control)

    Note:
        Current implementation assumes:
        - Single trip_count input (first loop input)
        - Remaining inputs are carry variables
        - Forward pass handles internal computation

        Future work:
        - Condition-based loops (WHILE instead of FOR)
        - Dynamic loop count from tensor
        - Scan output specialization
        - Loop body optimization/unrolling
    """
    logger.debug(
        f"Lowering loop region {region.region_id} "
        f"with loop_inputs={region.loop_inputs}, loop_outputs={region.loop_outputs}"
    )

    ir = optimization_result.ir
    buffer_allocations = optimization_result.buffer_allocations

    code = STCode.empty()

    # Comment header for region
    code += STCode.from_lines(f"(* Loop Region {region.region_id} *)")
    code += STCode.blank_line()

    # Extract loop metadata
    loop_metadata = _extract_loop_metadata(region, ir, buffer_allocations)

    # Generate carry variable initialization if needed
    if loop_metadata["carry_vars"]:
        code += _generate_loop_initialization(loop_metadata)
        code += STCode.blank_line()

    # Generate main loop structure
    code += _generate_loop_body(region, ir, buffer_allocations, loop_metadata)
    code += STCode.blank_line()

    return code


def _extract_loop_metadata(
    region: LoopRegionIR,
    ir: NetworkIR,
    buffer_allocations: Dict[str, str],
) -> Dict:
    """
    Extract and structure loop metadata for code generation.

    Parses loop_inputs and loop_outputs to identify:
    - trip_count: loop iteration count
    - carry_vars: state variables that persist across iterations
    - scan_outputs: accumulated outputs from each iteration

    Returns:
        Dictionary with loop execution parameters:
        {
            "trip_count_var": str,      # Variable name for loop count
            "carry_vars": [(in, out),], # (carry_input, carry_output) pairs
            "scan_outputs": [str, ...], # Accumulated output names
        }
    """
    metadata = {
        "trip_count_var": "trip_count",  # Default; could be extracted from first input
        "carry_vars": [],
        "scan_outputs": [],
    }

    # For MVP: assume first loop_input is trip_count
    if region.loop_inputs:
        trip_count_tensor = region.loop_inputs[0]
        metadata["trip_count_var"] = _resolve_variable_name(
            trip_count_tensor, buffer_allocations
        )

        # Remaining inputs are carry variables
        # Assume they pair with first N outputs
        for idx, carry_in in enumerate(region.loop_inputs[1:]):
            if idx < len(region.loop_outputs):
                carry_out = region.loop_outputs[idx]
                metadata["carry_vars"].append((carry_in, carry_out))

        # Remaining outputs are scan outputs
        num_carries = len(metadata["carry_vars"])
        if len(region.loop_outputs) > num_carries:
            metadata["scan_outputs"] = list(region.loop_outputs[num_carries:])

    return metadata


def _generate_loop_initialization(loop_metadata: Dict) -> STCode:
    """
    Generate carry variable initialization code.

    For ONNX Loop, carry variables must be initialized from inputs
    before the loop begins.

    Generates:
        carry_var_0 := input_carry_var_0;
        carry_var_1 := input_carry_var_1;
        ...
    """
    code = STCode.from_lines("(* Loop carry initialization *)")

    for carry_in, carry_out in loop_metadata["carry_vars"]:
        # Simple assignment: output := input
        # In real usage, might map through buffer allocations
        code += STCode.from_lines(f"{carry_out} := {carry_in};")

    return code


def _generate_loop_body(
    region: LoopRegionIR,
    ir: NetworkIR,
    buffer_allocations: Dict[str, str],
    loop_metadata: Dict,
) -> STCode:
    """
    Generate the main loop structure with body execution.

    Generates:
        FOR iteration := 0 TO (trip_count - 1) DO
            <loop body execution>
        END_FOR;

    The loop body is the forward pass of the loop region's computation graph.
    """
    from .generator import generate_forward_pass

    trip_count_var = loop_metadata["trip_count_var"]

    code = STCode.from_lines(f"(* Loop: {len(loop_metadata['carry_vars'])} carries *)")

    # FOR loop from 0 to trip_count - 1
    code += STCode.from_lines(f"FOR iteration := 0 TO ({trip_count_var} - 1) DO")

    # Generate loop body (forward pass over the loop region graph)
    body_code = generate_forward_pass(ir, buffer_allocations)
    for line in body_code.lines:
        code += STCode.from_lines("\t" + line)

    code += STCode.from_lines("END_FOR;")

    return code


def lower_region_to_st(
    region,
    optimization_result: OptimizationResult,
) -> STCode:
    """
    Dispatch region lowering based on region type.

    Routes to appropriate lowerer for the region type. Requires properly typed
    region objects (AcyclicRegionIR, RecurrentRegionIR, LoopRegionIR) to ensure
    all required attributes are present.

    Args:
        region: Region to lower (must be a typed subclass, not base RegionIR)
        optimization_result: Optimization result for this region

    Returns:
        Generated ST code for this region

    Raises:
        TypeError: If region is not a properly typed subclass
        ValueError: If region kind is unsupported
    """
    if isinstance(region, AcyclicRegionIR):
        return lower_acyclic_region_to_st(region, optimization_result)

    elif isinstance(region, RecurrentRegionIR):
        return lower_recurrent_region_to_st(region, optimization_result)

    elif isinstance(region, LoopRegionIR):
        return lower_loop_region_to_st(region, optimization_result)

    else:
        # Provide helpful error message
        region_type = type(region).__name__
        raise TypeError(
            f"Expected a typed region subclass (AcyclicRegionIR, RecurrentRegionIR, "
            f"or LoopRegionIR), but got {region_type}. Base RegionIR is not supported "
            f"by the lowerer. Ensure regions are created through regionizer, which always "
            f"produces properly typed regions."
        )
