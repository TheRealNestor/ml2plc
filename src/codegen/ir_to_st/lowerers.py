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

    Loop inputs: loop count, carry variables
    Loop outputs: final carry values, scan outputs

    Args:
        region: Loop region to lower
        optimization_result: Optimized IR for this region

    Returns:
        Generated ST code for this region (with loop control)

    Note:
        Current implementation is a placeholder. Full support requires:
        - Loop count extraction
        - Carry variable management
        - Loop body generation
        - Scan output collection
    """
    logger.debug(f"Lowering loop region {region.region_id}")
    logger.warning(f"Loop region {region.region_id} lowering not fully implemented")

    code = STCode.from_lines(f"(* Loop region {region.region_id} - placeholder *)")
    code += STCode.from_lines("(* TODO: Implement loop control flow *)")

    # TODO: Implement:
    # 1. Loop count extraction
    # 2. Carry variable initialization
    # 3. FOR/WHILE loop generation
    # 4. Body execution
    # 5. Carry update and output collection

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
