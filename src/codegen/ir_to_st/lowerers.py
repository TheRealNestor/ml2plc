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

    Returns:
        Generated ST code for this region (with state handling)

    Note:
        Current implementation is a placeholder. Full support requires:
        - State variable declaration and initialization
        - Iteration loop generation
        - State update logic between iterations
    """
    logger.debug(f"Lowering recurrent region {region.region_id}")
    logger.warning(
        f"Recurrent region {region.region_id} lowering not fully implemented"
    )

    code = STCode.from_lines(f"(* Recurrent region {region.region_id} - placeholder *)")
    code += STCode.from_lines("(* TODO: Implement recurrent state handling *)")

    # TODO: Implement:
    # 1. State variable initialization
    # 2. Iteration loop (unroll or while loop)
    # 3. Layer execution with state flow
    # 4. State update/persistence

    return code


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
    Dispatch region lowering based on region kind.

    Routes to appropriate lowerer for the region type.

    Args:
        region: Region to lower (AcyclicRegionIR, RecurrentRegionIR, LoopRegionIR, etc.)
        optimization_result: Optimization result for this region

    Returns:
        Generated ST code for this region

    Raises:
        ValueError: If region kind is unsupported or unknown
    """
    if isinstance(region, AcyclicRegionIR) or region.kind == RegionKind.ACYCLIC:
        return lower_acyclic_region_to_st(region, optimization_result)

    elif isinstance(region, RecurrentRegionIR) or region.kind == RegionKind.RECURRENT:
        return lower_recurrent_region_to_st(region, optimization_result)

    elif isinstance(region, LoopRegionIR) or region.kind == RegionKind.LOOP:
        return lower_loop_region_to_st(region, optimization_result)

    else:
        raise ValueError(
            f"Unsupported region kind: {region.kind}. "
            f"Supported kinds: {[k.value for k in RegionKind]}"
        )
