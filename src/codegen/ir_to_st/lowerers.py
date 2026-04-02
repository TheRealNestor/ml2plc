"""
Backend lowering: Convert regions to Structured Text by kind.

Separates region-kind-specific ST generation into dedicated lowerers using
a template method pattern, which provides a common structure while allowing
region-kind-specific customization points.

Architecture:
  - RegionLowerer: Abstract base class with template method `lower()`
  - Region-specific subclasses: AcyclicLowerer, RecurrentLowerer, LoopLowerer
    Each implements hooks for pre-loop, loop bounds, and loop body generation.

This pattern consolidates common logic (loop structure, indentation, code building)
while making it easy to extend for new region types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

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


class RegionLowerer(ABC):
    """
    Abstract base class for region lowering using template method pattern.

    Subclasses implement three hooks to customize loop generation:
      1. pre_loop_code(): Code before loop (state init, etc.)
      2. loop_bounds(): Returns (init_value, end_value) for loop variable
      3. loop_body_code(): Code inside loop
    """

    def __init__(
        self,
        region,
        optimization_result: OptimizationResult,
    ):
        self.region = region
        self.ir = optimization_result.ir
        self.buffer_allocations = optimization_result.buffer_allocations or {}

    @abstractmethod
    def pre_loop_code(self) -> STCode:
        """Code before loop (state initialization, etc.)."""
        pass

    @abstractmethod
    def loop_bounds(self) -> Tuple[str, str]:
        """Returns (init_value, end_value) for FOR loop."""
        pass

    @abstractmethod
    def loop_body_code(self) -> STCode:
        """Code inside loop body."""
        pass

    def lower(self) -> STCode:
        """Template method: orchestrates lowering by combining hooks."""
        code = STCode.empty()

        # Region header
        code += STCode.from_lines(
            f"(* {self.region_type()} Region {self.region.region_id} *)"
        )
        code += STCode.blank_line()

        # Pre-loop section
        pre_code = self.pre_loop_code()
        if pre_code.lines:
            code += pre_code
            code += STCode.blank_line()

        # Loop structure
        init_val, end_val = self.loop_bounds()
        code += STCode.from_lines(f"FOR step := {init_val} TO {end_val} DO")

        body_code = self.loop_body_code()
        for line in body_code.lines:
            code += STCode.from_lines("\t" + line)

        code += STCode.from_lines("END_FOR;")
        code += STCode.blank_line()

        return code

    def region_type(self) -> str:
        """Return human-readable region type."""
        if isinstance(self.region, AcyclicRegionIR):
            return "Acyclic"
        elif isinstance(self.region, RecurrentRegionIR):
            return "Recurrent"
        elif isinstance(self.region, LoopRegionIR):
            return "Loop"
        else:
            return "Unknown"


class AcyclicLowerer(RegionLowerer):
    """Lowerer for acyclic (DAG) regions."""

    def pre_loop_code(self) -> STCode:
        """No pre-loop code for acyclic regions."""
        return STCode.empty()

    def loop_bounds(self) -> Tuple[str, str]:
        """Acyclic: single execution (0 to 0)."""
        return ("0", "0")

    def loop_body_code(self) -> STCode:
        """Use standard forward pass for acyclic body."""
        from .generator import generate_forward_pass

        return generate_forward_pass(self.ir, self.buffer_allocations)


class RecurrentLowerer(RegionLowerer):
    """Lowerer for recurrent (cyclic) regions."""

    def __init__(
        self, region: RecurrentRegionIR, optimization_result: OptimizationResult
    ):
        super().__init__(region, optimization_result)
        self.num_timesteps = 1  # MVP: fixed 1 timestep

    def pre_loop_code(self) -> STCode:
        """Generate state variable initialization."""
        if not self.region.state_inputs or not self.region.state_outputs:
            return STCode.empty()

        code = STCode.from_lines("(* State initialization *)")
        for state_in, state_out in zip(
            self.region.state_inputs, self.region.state_outputs
        ):
            input_var = _resolve_variable_name(state_in, self.buffer_allocations)
            output_var = _resolve_variable_name(
                state_out, self.ir, self.buffer_allocations
            )
            code += STCode.from_lines(f"{output_var} := {input_var};")

        return code

    def loop_bounds(self) -> Tuple[str, str]:
        """Loop from 0 to (num_timesteps - 1)."""
        return ("0", str(self.num_timesteps - 1))

    def loop_body_code(self) -> STCode:
        """Use standard forward pass for body."""
        from .generator import generate_forward_pass

        return generate_forward_pass(self.ir, self.buffer_allocations)


class LoopLowerer(RegionLowerer):
    """Lowerer for control-flow loop regions (ONNX Loop/Scan)."""

    def __init__(self, region: LoopRegionIR, optimization_result: OptimizationResult):
        super().__init__(region, optimization_result)
        self.loop_metadata = self._extract_loop_metadata()

    def _extract_loop_metadata(self) -> Dict:
        """
        Extract loop metadata for code generation.

        Parses loop_inputs and loop_outputs to identify:
        - trip_count: loop iteration count variable
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
            "trip_count_var": "trip_count",
            "carry_vars": [],
            "scan_outputs": [],
        }

        # First loop_input is trip_count
        if self.region.loop_inputs:
            trip_count_tensor = self.region.loop_inputs[0]
            metadata["trip_count_var"] = _resolve_variable_name(
                trip_count_tensor, self.buffer_allocations
            )

            # Remaining inputs are carry variables (pair with first N outputs)
            for idx, carry_in in enumerate(self.region.loop_inputs[1:]):
                if idx < len(self.region.loop_outputs):
                    carry_out = self.region.loop_outputs[idx]
                    metadata["carry_vars"].append((carry_in, carry_out))

            # Remaining outputs are scan outputs
            num_carries = len(metadata["carry_vars"])
            if len(self.region.loop_outputs) > num_carries:
                metadata["scan_outputs"] = list(self.region.loop_outputs[num_carries:])

        return metadata

    def pre_loop_code(self) -> STCode:
        """Generate carry variable initialization."""
        if not self.loop_metadata["carry_vars"]:
            return STCode.empty()

        code = STCode.from_lines("(* Loop carry initialization *)")
        for carry_in, carry_out in self.loop_metadata["carry_vars"]:
            code += STCode.from_lines(f"{carry_out} := {carry_in};")

        return code

    def loop_bounds(self) -> Tuple[str, str]:
        """Loop from 0 to (trip_count - 1)."""
        trip_count_var = self.loop_metadata["trip_count_var"]
        return ("0", f"({trip_count_var} - 1)")

    def loop_body_code(self) -> STCode:
        """Use standard forward pass for body."""
        from .generator import generate_forward_pass

        body_code = generate_forward_pass(self.ir, self.buffer_allocations)
        # Optionally add carry update logic here in future
        return body_code


# ============================================================================
# Public API Functions
# ============================================================================


def lower_acyclic_region_to_st(
    region: AcyclicRegionIR,
    optimization_result: OptimizationResult,
) -> STCode:
    """Lower an acyclic (DAG) region to Structured Text."""
    logger.debug(f"Lowering acyclic region {region.region_id}")
    lowerer = AcyclicLowerer(region, optimization_result)
    return lowerer.lower()


def lower_recurrent_region_to_st(
    region: RecurrentRegionIR,
    optimization_result: OptimizationResult,
    num_timesteps: int = 1,
) -> STCode:
    """Lower a recurrent (cyclic) region to Structured Text."""
    logger.debug(
        f"Lowering recurrent region {region.region_id} "
        f"with state_inputs={region.state_inputs}, state_outputs={region.state_outputs}, "
        f"num_timesteps={num_timesteps}"
    )
    lowerer = RecurrentLowerer(region, optimization_result)
    lowerer.num_timesteps = num_timesteps
    return lowerer.lower()


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
    """
    logger.debug(
        f"Lowering loop region {region.region_id} "
        f"with loop_inputs={region.loop_inputs}, loop_outputs={region.loop_outputs}"
    )
    lowerer = LoopLowerer(region, optimization_result)
    return lowerer.lower()


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
