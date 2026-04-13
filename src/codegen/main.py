"""
Main entry point for ONNX to Structured Text compiler.

The compilation pipeline is split into explicit, composable stages:
  analyze_model()      → ONNXModel
    normalize_model_for_ir()   → (working_layers, constant_values, folded_outputs)
    extract_typed_network_ir() → NetworkIR (unordered)
    schedule_network_ir()    → NetworkIR (ordered)
    regionize_model_ir()     → ModelIR
    optimize_regions()       → Dict[region_id -> OptimizationResult]
    generate_st()            → ST code string

Each stage is pure (input→output with minimal side effects) and can be tested independently.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir import (
    onnx_to_ir,
    regionize_network_ir,
    normalize_model_for_ir,
    extract_typed_ir_graph,
    schedule_network_ir as schedule_ir_graph,
    NormalizedIRInputs,
)
from codegen.ir_optimizer import optimize_model_regions, OptimizationResult
from codegen.memory_check.memory_analyzer import check_memory
from codegen.ir_to_st import translate_model_to_st
from codegen.types import ModelIR, RegionKind, NetworkIR

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
# Stage 2: Prepare ONNX for IR Extraction
# ============================================================================


def normalize_ir_inputs(analyzer: ONNXModel) -> NormalizedIRInputs:
    """
    Normalize ONNX graph for typed IR extraction.

    Stage 2 of the pipeline.

    Args:
        analyzer: Loaded ONNX model analyzer

    Returns:
        Prepared artifacts for typed extraction pass
    """
    logger.info("Stage 2: Normalizing model for IR extraction")
    working_layers, constant_values, folded_outputs = normalize_model_for_ir(analyzer)
    logger.info(
        "  Prepared %d working layer(s), %d compile-time constant(s)",
        len(working_layers),
        len(constant_values),
    )
    return (working_layers, constant_values, folded_outputs)


# ============================================================================
# Stage 3: Extract Typed NetworkIR (unordered)
# ============================================================================


def extract_typed_network_ir(
    analyzer: ONNXModel, prepared: NormalizedIRInputs
) -> NetworkIR:
    """Extract typed layer/tensor graph without execution order."""
    logger.info("Stage 3: Extracting typed NetworkIR (unordered)")
    working_layers, constant_values, folded_outputs = prepared
    ir_unordered = extract_typed_ir_graph(
        analyzer,
        working_layers,
        constant_values,
        folded_outputs,
    )
    logger.info("  Extracted %d typed layer(s)", len(ir_unordered.layers))
    return ir_unordered


# ============================================================================
# Stage 4: Schedule Execution Order
# ============================================================================


def schedule_network_ir(ir_unordered: NetworkIR) -> NetworkIR:
    """Attach execution order to typed graph (topological/SCC-aware)."""
    logger.info("Stage 4: Scheduling execution order")
    ir_ordered = schedule_ir_graph(ir_unordered)
    logger.info(
        "  Computed execution order with %d entries", len(ir_ordered.execution_order)
    )
    return ir_ordered


# ============================================================================
# Stage 5: Regionize NetworkIR into ModelIR
# ============================================================================


def regionize_model_ir(network_ir: NetworkIR) -> ModelIR:
    """Partition ordered NetworkIR into typed regions."""
    logger.info("Stage 5: Regionizing ordered graph")
    model_ir = regionize_network_ir(network_ir)
    logger.info(f"  Created {len(model_ir.regions)} region(s)")

    if len(model_ir.regions) > 1:
        region_types = [r.kind.value for r in model_ir.regions]
        logger.info(f"  Region types: {region_types}")

    return model_ir


# ============================================================================
# Compatibility wrapper: Build Model IR (legacy stage API)
# ============================================================================


def build_model_ir(analyzer: ONNXModel) -> ModelIR:
    """Backward-compatible wrapper for the explicit Stage 2→5 pipeline."""
    prepared = normalize_ir_inputs(analyzer)
    ir_unordered = extract_typed_network_ir(analyzer, prepared)
    ir_ordered = schedule_network_ir(ir_unordered)
    return regionize_model_ir(ir_ordered)


# ============================================================================
# Stage 6: Optimize Regions
# ============================================================================


def optimize_regions(model_ir: ModelIR) -> Dict[str, OptimizationResult]:
    """
    Apply optimization passes to each region independently.

    Stage 6 of the pipeline. Region-aware: only optimizes supported region kinds.

    Args:
        model_ir: Regionized model

    Returns:
        Dictionary mapping region_id to OptimizationResult (containing optimized IR
        and optional buffer allocation hints)
    """
    logger.info("Stage 6: Optimizing regions")
    optimization_results = optimize_model_regions(model_ir)
    logger.info(f"  Optimized {len(optimization_results)} region(s)")
    return optimization_results


# ============================================================================
# Stage 7: Generate Structured Text
# ============================================================================


def generate_st(
    model_ir: ModelIR,
    optimization_results: Dict[str, OptimizationResult],
    output_path: Optional[str] = None,
    fb_name: str = "NeuralNetworkFB",
) -> str:
    """
    Generate Structured Text code from optimized ModelIR.

    Stage 7 of the pipeline. Uses region-aware lowering to produce ST.

    Args:
        model_ir: Regionized model
        optimization_results: Results from Stage 3
        output_path: Optional path to write ST code to disk
        fb_name: Name for the generated function block

    Returns:
        Generated Structured Text code as string
    """
    logger.info("Stage 7: Generating Structured Text")

    logger.info(f"ST generation for {len(model_ir.regions)} region(s)")
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
    model_path: str,
    optimize: bool = True,
    output_path: str = None,
    fb_name: Optional[str] = None,
) -> str:
    """
                Complete compilation pipeline: ONNX → ModelIR → Optimized regions → ST code.

                Orchestrates four top-level stages:
            1. analyze_model()      → load ONNX
            2. build_model_ir()     → normalize + extract + schedule + regionize
            3. optimize_regions()   → apply optimization passes
            4. generate_st()        → produce ST code

                Also performs an advisory memory check for the first acyclic region
                between stages 3 and 4.

    Args:
        model_path: Path to ONNX model file
        optimize: Whether to apply optimization passes in Stage 3
        output_path: Optional path to save generated ST code
        fb_name: Optional function block name. If None, defaults to input filename stem.

    Returns:
        Generated Structured Text code as string
    """
    logger.info(f"Starting compilation pipeline: {model_path}")
    input_path = Path(model_path)

    # Stage 1: Analyze
    analyzer = analyze_model(model_path)

    # Stage 2: Build IR (internally normalize → extract → schedule → regionize)
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

    # Advisory memory check for first acyclic region (warning-only)
    if any(r.kind == RegionKind.ACYCLIC for r in model_ir.regions):
        first_acyclic = next(
            r.region_id for r in model_ir.regions if r.kind == RegionKind.ACYCLIC
        )
        if first_acyclic in optimization_results:
            logger.debug("Checking memory for first acyclic region")
            memory_result = check_memory(
                optimization_results[first_acyclic].ir,
                fail_on_exceed=False,
            )
            if memory_result.errors:
                logger.warning(
                    "Continuing compilation despite memory-limit exceedance "
                    "(warning-only mode)."
                )

    # Stage 4: Generate
    fb_name = fb_name or input_path.stem
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
        default=None,
        help="Name for the generated function block (default: input model filename)",
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
            fb_name=args.fb_name,
        )

        logger.info(f"Successfully compiled {input_path.name}")
        logger.info(f"Output written to {output_path}")

    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
