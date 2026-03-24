"""
Main entry point for ONNX to Structured Text compiler.

The compilation pipeline is split into explicit, composable stages:
  analyze_model()      → ONNXModel
  build_model_ir()     → ModelIR
  optimize_regions()   → Dict[region_id -> OptimizationResult]
  plan_execution()     → ExecutionPlan
  generate_st()        → ST code string

Each stage is pure (input→output with minimal side effects) and can be tested independently.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir import onnx_to_ir, regionize_network_ir
from codegen.ir_optimizer import optimize_model_regions, OptimizationResult
from codegen.memory_check.memory_analyzer import check_memory
from codegen.ir_to_st import translate_ir_to_st, translate_model_to_st
from codegen.types import ModelIR, RegionKind
from codegen.backends import default_st_backend_capabilities
from codegen.planner import create_execution_plan

logger = logging.getLogger(__name__)


# ============================================================================
# Stage 1: Analyze Model
# ============================================================================


def analyze_model(model_path: str) -> ONNXModel:
    """
    Load and validate ONNX model.

    Stage 1 of the pipeline. Pure function: loads from disk, validates structure.

    Args:
        model_path: Path to ONNX model file

    Returns:
        Loaded ONNXModel analyzer

    Raises:
        FileNotFoundError: If model file does not exist
        ValueError: If model cannot be parsed
    """
    logger.info(f"Stage 1: Analyzing ONNX model from {model_path}")
    analyzer = ONNXModel(model_path)
    analyzer.load_model()
    logger.info(f"  Loaded model with {len(analyzer.layers)} layers")
    return analyzer


# ============================================================================
# Stage 2: Build Model IR
# ============================================================================


def build_model_ir(analyzer: ONNXModel) -> ModelIR:
    """
    Convert ONNX model to regionized ModelIR.

    Stage 2 of the pipeline. Produces the only internal representation used
    downstream: ModelIR (not NetworkIR).

    Flow:
      ONNX model → NetworkIR (unoptimized) → regionize → ModelIR

    Args:
        analyzer: Loaded ONNX model analyzer

    Returns:
        Regionized ModelIR with identified regions by kind (acyclic/recurrent/loop)
    """
    logger.info("Stage 2: Building ModelIR")

    # Convert to intermediate NetworkIR
    logger.info("  Converting ONNX to NetworkIR...")
    network_ir = onnx_to_ir(analyzer)
    logger.info(f"    Created IR with {len(network_ir.layers)} layers")

    # Regionize into ModelIR
    logger.info("  Regionizing into typed regions...")
    model_ir = regionize_network_ir(network_ir)
    logger.info(f"    Created {len(model_ir.regions)} region(s)")

    if len(model_ir.regions) > 1:
        region_types = [r.kind.value for r in model_ir.regions]
        logger.info(f"    Region types: {region_types}")

    return model_ir


# ============================================================================
# Stage 3: Optimize Regions
# ============================================================================


def optimize_regions(model_ir: ModelIR) -> Dict[str, OptimizationResult]:
    """
    Apply optimization passes to each region independently.

    Stage 3 of the pipeline. Region-aware: only optimizes supported region kinds.

    Args:
        model_ir: Regionized model

    Returns:
        Dictionary mapping region_id to OptimizationResult (containing optimized IR
        and optional buffer allocation hints)
    """
    logger.info("Stage 3: Optimizing regions")
    optimization_results = optimize_model_regions(model_ir)
    logger.info(f"  Optimized {len(optimization_results)} region(s)")
    return optimization_results


# ============================================================================
# Stage 4: Plan Execution
# ============================================================================


def plan_execution(model_ir: ModelIR, validate: bool = True):
    """
    Validate backend capabilities and plan region execution.

    Stage 4 of the pipeline. Creates an execution plan or validates feasibility.

    Args:
        model_ir: Regionized model
        validate: If True, validate that all regions are supported

    Returns:
        ExecutionPlan (when fully implemented) or None for now

    Raises:
        RuntimeError: If validate=True and unsupported regions are detected
    """
    logger.info("Stage 4: Planning execution")
    capabilities = default_st_backend_capabilities()

    if validate:
        for region in model_ir.regions:
            if region.kind == RegionKind.UNSUPPORTED:
                raise RuntimeError(f"Unsupported region type: {region.kind}")

    logger.info(
        f"  Backend capabilities validated for {len(model_ir.regions)} region(s)"
    )
    # TODO: Return structured ExecutionPlan once planning is formalized
    return None


# ============================================================================
# Stage 5: Generate Structured Text
# ============================================================================


def generate_st(
    model_ir: ModelIR,
    optimization_results: Dict[str, OptimizationResult],
    output_path: Optional[str] = None,
    fb_name: str = "NeuralNetworkFB",
) -> str:
    """
    Generate Structured Text code from optimized ModelIR.

    Stage 5 of the pipeline. Uses region-aware lowering to produce ST.

    Args:
        model_ir: Regionized model
        optimization_results: Results from Stage 3
        output_path: Optional path to write ST code to disk
        fb_name: Name for the generated function block

    Returns:
        Generated Structured Text code as string
    """
    logger.info("Stage 5: Generating Structured Text")

    # Choose code generation path based on model structure
    use_legacy_path = (
        len(model_ir.regions) == 1 and model_ir.regions[0].kind == RegionKind.ACYCLIC
    )

    if use_legacy_path:
        # Simple single-region acyclic model: use legacy single-IR path
        logger.info("  Using legacy single-region ST generation")
        target_region_id = model_ir.regions[0].region_id
        result = optimization_results[target_region_id]

        st_code = translate_ir_to_st(
            result.ir,
            fb_name=fb_name,
            buffer_allocations=result.buffer_allocations,
        )
    else:
        # Multi-region or non-acyclic: use region-aware path
        logger.info(
            f"  Using multi-region ST generation for {len(model_ir.regions)} region(s)"
        )
        st_code = translate_model_to_st(
            model_ir, optimization_results=optimization_results, fb_name=fb_name
        )

    if output_path:
        logger.info(f"  Writing ST code to {output_path}")
        with open(output_path, "w") as f:
            f.write(st_code)

    return st_code


# ============================================================================
# Unified Pipeline
# ============================================================================


def compile_onnx_to_st(
    model_path: str, optimize: bool = True, output_path: str = None
) -> str:
    """
    Complete compilation pipeline: ONNX → IR → Optimized IR → ST Code

    Orchestrates all five stages:
      1. analyze_model()      → load ONNX
      2. build_model_ir()     → convert to regionized ModelIR
      3. optimize_regions()   → apply optimization passes
      4. plan_execution()     → validate backend capabilities
      5. generate_st()        → produce ST code

    Args:
        model_path: Path to ONNX model file
        optimize: Whether to apply optimization passes in Stage 3
        output_path: Optional path to save generated ST code

    Returns:
        Generated Structured Text code as string
    """
    logger.info(f"Starting compilation pipeline: {model_path}")
    input_path = Path(model_path)

    # Stage 1: Analyze
    analyzer = analyze_model(model_path)

    # Stage 2: Build IR
    model_ir = build_model_ir(analyzer)

    # Stage 3: Optimize
    if optimize:
        optimization_results = optimize_regions(model_ir)
    else:
        logger.info("Stage 3: Skipping optimization (optimize=False)")
        # Wrap unoptimized graphs
        optimization_results = {}
        for region in model_ir.regions:
            optimization_results[region.region_id] = OptimizationResult(ir=region.graph)

    # Stage 4: Plan
    plan_execution(model_ir, validate=True)

    # Step 5.5: Check memory consumption (TODO: integrate into planning)
    if any(r.kind == RegionKind.ACYCLIC for r in model_ir.regions):
        first_acyclic = next(
            r.region_id for r in model_ir.regions if r.kind == RegionKind.ACYCLIC
        )
        if first_acyclic in optimization_results:
            logger.debug("Checking memory for first acyclic region")
            check_memory(optimization_results[first_acyclic].ir)

    # Stage 5: Generate
    fb_name = input_path.stem
    st_code = generate_st(
        model_ir, optimization_results, output_path=output_path, fb_name=fb_name
    )

    logger.info(f"Compilation complete. ST code generated ({len(st_code)} chars)")
    return st_code


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile ONNX neural network models to IEC 61131-3 Structured Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=str, help="Path to input ONNX model file")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output Structured Text file (default: <input_name>.st)",
    )

    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable IR optimization passes"
    )

    parser.add_argument(
        "--fb-name",
        type=str,
        default="NeuralNetworkFB",
        help="Name for the generated function block (default: NeuralNetworkFB)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose/debug output"
    )

    parser.add_argument(
        "--version", action="version", version="ONNX to ST Compiler v0.1.0"
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not input_path.suffix.lower() == ".onnx":
        logger.warning(f"Input file does not have .onnx extension: {args.input}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".st")
        if input_path.parent.name.lower() == "onnx":
            output_path = (
                input_path.parent.parent / "structured_text" / output_path.name
            )

        logger.info(f"Auto-generated output path: {output_path}")

    try:
        compile_onnx_to_st(
            model_path=str(input_path),
            optimize=not args.no_optimize,
            output_path=str(output_path),
        )

        logger.info(f"Successfully compiled {input_path.name}")
        logger.info(f"Output written to {output_path}")

    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
