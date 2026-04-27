"""
Unified region lowering using strategy pattern.
Eliminates duplication across AcyclicLowerer, RecurrentLowerer, LoopLowerer.
"""

from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from ..types import (
    NetworkIR,
    RegionIR,
    AcyclicRegionIR,
    RecurrentRegionIR,
    LoopRegionIR,
)
from ..ir_optimizer import OptimizationResult
from .st_code import STCode, st_comment
from .st_templates import st_for_loop


@dataclass
class LoweringContext:
    """Shared context across all lowering strategies."""

    region: RegionIR
    optimization_result: OptimizationResult
    ir: NetworkIR = None  # Set from optimization_result
    buffer_allocations: Dict[str, str] = None

    def __post_init__(self):
        if self.ir is None:
            self.ir = self.optimization_result.ir
        if self.buffer_allocations is None:
            self.buffer_allocations = self.optimization_result.buffer_allocations or {}


class RegionLoweringStrategy(ABC):
    """Base strategy for lowering any region type to ST code."""

    def __init__(self, context: LoweringContext):
        self.ctx = context

    @abstractmethod
    def pre_loop_code(self) -> STCode:
        """Code before main computation loop (initialization, state setup)."""
        pass

    @abstractmethod
    def loop_bounds(self) -> tuple[str, str]:
        """Return (loop_var, upper_bound) for main loop."""
        pass

    @abstractmethod
    def loop_body_code(self) -> STCode:
        """Main computation inside loop."""
        pass

    @abstractmethod
    def post_loop_code(self) -> STCode:
        """Code after main computation loop (finalization)."""
        pass

    def lower(self) -> STCode:
        """Template method: orchestrates lowering."""
        code = STCode.from_lines(
            f"(* Region: {self.ctx.region.region_id} [{self.ctx.region.kind.name}] *)"
        )
        code += self.pre_loop_code()
        code += self._generate_main_loop()
        code += self.post_loop_code()
        return code

    def _generate_main_loop(self) -> STCode:
        """Common loop generation logic for all strategies."""
        loop_var, upper_bound = self.loop_bounds()
        body = self.loop_body_code()
        return st_for_loop(loop_var, 0, upper_bound, body)


class AcyclicLoweringStrategy(RegionLoweringStrategy):
    """Strategy for acyclic regions: no loop, single forward pass."""

    def pre_loop_code(self) -> STCode:
        return STCode.empty()

    def loop_bounds(self) -> tuple[str, str]:
        # Dummy: no loop for acyclic
        return ("_unused", "-1")

    def loop_body_code(self) -> STCode:
        """Forward pass for all layers in execution order."""
        return self._generate_forward_pass()

    def post_loop_code(self) -> STCode:
        return STCode.empty()

    def lower(self) -> STCode:
        """Override: acyclic doesn't use loop template."""
        code = STCode.from_lines(f"(* Acyclic Region {self.ctx.region.region_id} *)")
        code += self._generate_forward_pass()
        return code

    def _generate_forward_pass(self) -> STCode:
        """Reuse existing generator logic."""
        from .forward_pass import generate_forward_pass

        return generate_forward_pass(self.ctx.ir, self.ctx.buffer_allocations)


class RecurrentLoweringStrategy(RegionLoweringStrategy):
    """Strategy for recurrent regions: state init + timestep loop."""

    def __init__(self, context: LoweringContext, num_timesteps: int = 1):
        super().__init__(context)
        self.num_timesteps = num_timesteps
        self.region: RecurrentRegionIR = context.region

    def pre_loop_code(self) -> STCode:
        """Initialize state variables to zero."""
        if not self.region.state_inputs or not self.region.state_outputs:
            return STCode.empty()

        code = st_comment("State initialization")
        for state_in, state_out in zip(
            self.region.state_inputs, self.region.state_outputs
        ):
            input_var = self._resolve_variable(state_in, is_input=True)
            output_var = self._resolve_variable(state_out, is_input=False)
            code += STCode.from_lines(f"{output_var} := {input_var};")

        return code

    def loop_bounds(self) -> tuple[str, str]:
        return ("step", str(self.num_timesteps - 1))

    def loop_body_code(self) -> STCode:
        """Forward pass for timestep."""
        from .forward_pass import generate_forward_pass

        return generate_forward_pass(self.ctx.ir, self.ctx.buffer_allocations)

    def post_loop_code(self) -> STCode:
        return STCode.empty()

    def _resolve_variable(self, tensor_name: str, is_input: bool) -> str:
        """Resolve tensor to variable name."""
        return self.ctx.buffer_allocations.get(tensor_name, tensor_name)


class LoopLoweringStrategy(RegionLoweringStrategy):
    """Strategy for explicit Loop/Scan regions (ONNX control flow)."""

    def __init__(self, context: LoweringContext):
        super().__init__(context)
        self.region: LoopRegionIR = context.region

    def pre_loop_code(self) -> STCode:
        """Initialize loop carry variables."""
        code = st_comment("Loop carry initialization")
        for carry_in, carry_out in zip(
            self.region.loop_inputs, self.region.loop_outputs
        ):
            out_var = self.ctx.buffer_allocations.get(carry_out, carry_out)
            code += STCode.from_lines(f"{out_var} := 0.0;  (* TODO: from loop carry *)")
        return code

    def loop_bounds(self) -> tuple[str, str]:
        # Extract from ONNX Loop spec (TODO: implement properly)
        return ("iter", "0")

    def loop_body_code(self) -> STCode:
        from .forward_pass import generate_forward_pass

        return generate_forward_pass(self.ctx.ir, self.ctx.buffer_allocations)

    def post_loop_code(self) -> STCode:
        return STCode.empty()


# ============================================================================
# Dispatcher (Single Entry Point)
# ============================================================================


def lower_region_to_st(
    region: RegionIR,
    optimization_result: OptimizationResult,
    num_timesteps: int = 1,
) -> STCode:
    """
    Unified dispatcher for all region types.

    Single entry point eliminates the need for separate lowering functions.
    """
    ctx = LoweringContext(
        region=region,
        optimization_result=optimization_result,
    )

    if isinstance(region, AcyclicRegionIR):
        strategy = AcyclicLoweringStrategy(ctx)
    elif isinstance(region, RecurrentRegionIR):
        strategy = RecurrentLoweringStrategy(ctx, num_timesteps)
    elif isinstance(region, LoopRegionIR):
        strategy = LoopLoweringStrategy(ctx)
    else:
        raise TypeError(f"Unknown region type: {type(region)}")

    return strategy.lower()
