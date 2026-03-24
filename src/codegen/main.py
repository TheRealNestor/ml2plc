"""
Main entry point for ONNX to Structured Text compiler.
"""

import logging
import argparse
import sys
from pathlib import Path

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir import onnx_to_ir, regionize_network_ir
from codegen.ir_optimizer import IROptimizer, OptimizationResult
from codegen.memory_check.memory_analyzer import check_memory
from codegen.ir_to_st import translate_ir_to_st
from codegen.types import graph_ir_to_network_ir
from codegen.backends import default_st_backend_capabilities
from codegen.planner import create_execution_plan

logger = logging.getLogger(__name__)


def compile_onnx_to_st(
    model_path: str, optimize: bool = True, output_path: str = None
) -> str:
    """
    Complete compilation pipeline: ONNX → IR → Optimized IR → ST Code

    Args:
        model_path: Path to ONNX model file
        optimize: Whether to apply optimization passes
        output_path: Optional path to save generated ST code

    Returns:
        Generated Structured Text code as string
    """
    logger.info(f"Compiling ONNX model: {model_path}")

    # Step 1: Load and analyze ONNX model
    logger.info("Step 1: Loading ONNX model...")
    analyzer = ONNXModel(model_path)
    analyzer.load_model()

    # Step 2: Convert to IR (complete, unoptimized)
    logger.info("Step 2: Converting to IR...")
    ir = onnx_to_ir(analyzer)
    logger.info(f"  Created IR with {len(ir.layers)} layers")

    # Step 3: Regionize model (milestone-1: single acyclic region)
    logger.info("Step 3: Regionizing model...")
    model_ir = regionize_network_ir(ir)
    logger.info(f"  Regionized model into {len(model_ir.regions)} region(s)")

    # Step 4: Plan execution against backend capabilities
    logger.info("Step 4: Planning execution...")
    capabilities = default_st_backend_capabilities()
    execution_plan = create_execution_plan(model_ir, capabilities)

    # Bridge back to legacy NetworkIR for existing optimizer/codegen stack
    ir = graph_ir_to_network_ir(execution_plan.model_ir.first_region().graph)

    # Step 5: Optimize IR (optional)
    buffer_allocations = None
    if optimize:
        logger.info("Step 5: Optimizing IR...")
        optimizer = IROptimizer(ir)
        result: OptimizationResult = optimizer.optimize()
        ir = result.ir
        buffer_allocations = result.buffer_allocations

        logger.info(f"  Optimized IR has {len(ir.layers)} layers")

    else:
        logger.info("Step 5: Skipping optimization (optimize=False)")

    # Step 6: Check memory consumption
    logger.info("Step 6: Checking memory consumption...")
    memory_report = check_memory(ir, memory_limit_kb=96, fail_on_exceed=False)
    logger.info(f"  Memory utilization: {memory_report.utilization_percent:.1f}%")

    # Step 7: Generate Structured Text code
    logger.info("Step 7: Generating Structured Text code...")
    st_code = translate_ir_to_st(
        ir, fb_name="NeuralNetworkFB", buffer_allocations=buffer_allocations
    )

    # Step 8: Save to file (optional)
    if output_path:
        logger.info(f"Step 8: Writing to {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(st_code)

    logger.info("Compilation complete!")
    logger.info(f"Generated ST code lines: {len(st_code.splitlines())}")
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
