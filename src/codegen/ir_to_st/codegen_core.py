"""
Core IR-to-ST code generation orchestration.

High-level entry points and function block structure generation:
- translate_ir_to_st: Single NetworkIR → ST code
- translate_model_to_st: Multi-region ModelIR → ST code
- generate_function_block: Core FB generation logic

Delegates layer-specific code generation to codegen_layers.py
and region-specific lowering to lowerers.py.
"""

from typing import Dict, Optional, Tuple
import logging

from .variable import Variable
from ..types import *
from ..ir_optimizer import OptimizationResult
from .st_code import STCode, STCodeBuilder, st_comment
from .type_conversion import plc_type_from_onnx_dtype, get_type_limits_from_str
from .layer_constants import (
    generate_layer_weights,
    generate_layer_bias,
    generate_layer_quantization_params,
    generate_layer_rhs_constants,
)
from .utils.constant_helpers import (
    generate_array_constant,
    generate_batchnorm_constants,
)
from .st_templates import st_function_block_header, st_function_block_footer
from .forward_pass import generate_forward_pass

logger = logging.getLogger(__name__)


# =============================================================================
# Header / Footer Generation
# =============================================================================


def generate_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return st_function_block_header(fb_name)


def generate_footer() -> STCode:
    """Generate function block footer."""
    return st_function_block_footer()


# =============================================================================
# I/O Section Generation
# =============================================================================


def generate_input_output_vars(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT and VAR_OUTPUT sections from network."""
    code = STCode.empty()
    first_layer = network.layers[network.execution_order[0]]
    last_layer = network.layers[network.execution_order[-1]]

    # Adjust input size for LSTM/GRU models (which use full sequences)
    actual_input_size = first_layer.input_size
    for layer_name in network.execution_order:
        layer = network.layers[layer_name]
        if isinstance(layer, (LSTMLayer, GRULayer)):
            recurrent_total_input = layer.sequence_length * layer.input_size
            actual_input_size = max(actual_input_size, recurrent_total_input)
            logger.debug(
                f"Adjusted input_data for {layer.__class__.__name__}: "
                f"seq_len={layer.sequence_length} x input_size={layer.input_size}"
            )
            break

    input_type = plc_type_from_onnx_dtype(first_layer.input_type)
    # Decide whether to declare input_data as a scalar or an ARRAY.
    # Default: prefer scalar when actual_input_size == 1 to avoid emitting
    # ARRAY[0..0]. However, if the model's first layer or input shape
    # indicates multi-dimensional access (e.g., Conv2D, Flatten, Reshape,
    # or an input_shape with multiple axes), the generated forward-pass will
    # index into input_data even when the product of dims == 1. In that
    # case we must declare an ARRAY (possibly of length 1) to preserve
    # correct index semantics and avoid runtime "index into scalar" errors.
    need_array_for_indexing = False
    # If input_shape has more than one axis, prefer array even if total size == 1
    if first_layer.input_shape is not None and len(first_layer.input_shape) > 1:
        need_array_for_indexing = True
    # If the first layer type is a conv/reshape/flatten/pool that will index into
    # the input tensor, prefer array semantics
    if isinstance(
        first_layer,
        (
            Conv2DLayer,
            FlattenLayer,
            ReshapeLayer,
            Pool2DLayer,
            UnsqueezeLayer,
            SqueezeLayer,
        ),
    ):
        need_array_for_indexing = True

    if actual_input_size == 1 and not need_array_for_indexing:
        code += STCode.from_lines(
            "VAR_INPUT",
            f"    input_data : {input_type};",
            "END_VAR",
            "",
        )
    else:
        code += STCode.from_lines(
            "VAR_INPUT",
            f"    input_data : ARRAY[0..{actual_input_size - 1}] OF {input_type};",
            "END_VAR",
            "",
        )

    output_type = plc_type_from_onnx_dtype(last_layer.output_type)
    if last_layer.output_size == 1:
        code += STCode.from_lines(
            "VAR_OUTPUT",
            f"    output_data : {output_type};",
            "END_VAR",
            "",
        )
    else:
        code += STCode.from_lines(
            "VAR_OUTPUT",
            f"    output_data : ARRAY[0..{last_layer.output_size - 1}] OF {output_type};",
            "END_VAR",
            "",
        )

    return code


# =============================================================================
# Constants Section Generation
# =============================================================================


def generate_constants_section(network: NetworkIR) -> STCode:
    """Generate VAR CONSTANT section from network layers."""
    code = STCode.from_lines("VAR CONSTANT")

    for layer_name in network.execution_order:
        layer = network.layers[layer_name]

        # Weights
        if hasattr(layer, "weights") and layer.weights is not None:
            code += generate_layer_weights(layer).indent()

        # Bias (skip LSTM and GRU—handled in generate_layer_weights)
        if (
            hasattr(layer, "bias")
            and layer.bias is not None
            and not isinstance(layer, (LSTMLayer, GRULayer))
        ):
            code += generate_layer_bias(layer).indent()

        # Quantization parameters
        if isinstance(layer, (QuantizeLinearLayer, DequantizeLinearLayer)):
            if layer.input_type is not None:
                quant_params = generate_layer_quantization_params(layer)
                if quant_params.lines:
                    code += quant_params.indent()

        # Binary elementwise RHS constants
        rhs_constants = generate_layer_rhs_constants(layer)
        if rhs_constants.lines:
            code += rhs_constants.indent()

        # BatchNorm parameters
        if isinstance(layer, BatchNormLayer):
            bn_code = generate_batchnorm_constants(layer)
            if bn_code.lines:
                code += bn_code.indent()

        code += STCode.blank_line()

    code += STCode.from_lines("END_VAR", "")
    return code


def generate_merged_constants_section(
    optimization_results: Dict[str, OptimizationResult],
) -> STCode:
    """Merge constants from all regions into a single section."""
    code = STCode.from_lines("VAR CONSTANT")
    added_constants = set()

    for region_id, opt_result in optimization_results.items():
        network = opt_result.ir

        for layer_name in network.execution_order:
            layer = network.layers[layer_name]

            # Weights
            if hasattr(layer, "weights") and layer.weights is not None:
                const_name = f"weights_{layer.layer_id}"
                if const_name not in added_constants:
                    code += generate_layer_weights(layer).indent()
                    added_constants.add(const_name)
                    # Mark per-gate constants for LSTM
                    if isinstance(layer, LSTMLayer):
                        for gate in ("i", "f", "g", "o"):
                            added_constants.add(f"weights_{layer.layer_id}_{gate}")
                            added_constants.add(f"recurrent_{layer.layer_id}_{gate}")
                            added_constants.add(f"bias_{layer.layer_id}_{gate}")
                    # Mark per-gate constants for GRU
                    elif isinstance(layer, GRULayer):
                        for gate in ("r", "u", "h"):
                            added_constants.add(f"weights_{layer.layer_id}_{gate}")
                            added_constants.add(f"recurrent_{layer.layer_id}_{gate}")
                            added_constants.add(f"bias_{layer.layer_id}_{gate}")

            # Bias (skip LSTM and GRU)
            elif isinstance(layer, (LSTMLayer, GRULayer)) and layer.W is not None:
                const_name = f"weights_{layer.layer_id}"
                if const_name not in added_constants:
                    code += generate_layer_weights(layer).indent()
                    added_constants.add(const_name)
                    if isinstance(layer, LSTMLayer):
                        for gate in ("i", "f", "g", "o"):
                            added_constants.add(f"weights_{layer.layer_id}_{gate}")
                            added_constants.add(f"recurrent_{layer.layer_id}_{gate}")
                            added_constants.add(f"bias_{layer.layer_id}_{gate}")
                    elif isinstance(layer, GRULayer):
                        for gate in ("r", "u", "h"):
                            added_constants.add(f"weights_{layer.layer_id}_{gate}")
                            added_constants.add(f"recurrent_{layer.layer_id}_{gate}")
                            added_constants.add(f"bias_{layer.layer_id}_{gate}")

            if (
                hasattr(layer, "bias")
                and layer.bias is not None
                and not isinstance(layer, (LSTMLayer, GRULayer))
            ):
                const_name = f"bias_{layer.layer_id}"
                if const_name not in added_constants:
                    code += generate_layer_bias(layer).indent()
                    added_constants.add(const_name)

            # Quantization and BatchNorm
            if isinstance(layer, (QuantizeLinearLayer, DequantizeLinearLayer)):
                if layer.input_type is not None:
                    quant_params = generate_layer_quantization_params(layer)
                    if quant_params.lines:
                        const_name = f"quant_{layer.layer_id}"
                        if const_name not in added_constants:
                            code += quant_params.indent()
                            added_constants.add(const_name)

            rhs_constants = generate_layer_rhs_constants(layer)
            if rhs_constants.lines:
                const_name = f"rhs_const_{layer.layer_id}"
                if const_name not in added_constants:
                    code += rhs_constants.indent()
                    added_constants.add(const_name)

            if isinstance(layer, BatchNormLayer):
                scale_name = f"bn_scale_{layer.layer_id}"
                bias_name = f"bn_bias_{layer.layer_id}"
                if scale_name not in added_constants:
                    code += generate_array_constant(
                        scale_name, layer.combined_scale, "REAL"
                    ).indent()
                    added_constants.add(scale_name)
                if bias_name not in added_constants:
                    code += generate_array_constant(
                        bias_name, layer.combined_bias, "REAL"
                    ).indent()
                    added_constants.add(bias_name)

            code += STCode.blank_line()

    code += STCode.from_lines("END_VAR", "")
    return code


# =============================================================================
# Variable Section Generation
# =============================================================================


def generate_var_section(
    network: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> STCode:
    """Generate VAR section with all internal variables."""
    builder = STCodeBuilder()
    builder.add_line("VAR")

    # Buffer allocations or layer outputs
    if buffer_allocations:
        buffer_info = {}
        logger.debug(f"Buffer allocations mapping: {buffer_allocations}")
        for tensor_name, buffer_name in buffer_allocations.items():
            producer_name = network.tensor_producers[tensor_name]
            layer = network.layers[producer_name]
            plc_type = plc_type_from_onnx_dtype(layer.output_type)
            size = layer.output_size

            if buffer_name not in buffer_info:
                buffer_info[buffer_name] = (size, plc_type)
            else:
                existing_size, _ = buffer_info[buffer_name]
                buffer_info[buffer_name] = (max(existing_size, size), plc_type)

        builder.add_line("    (* Buffer allocation variables *)")
        with builder.indent():
            logger.debug(f"Computed buffer_info (name -> (size, type)): {buffer_info}")
            for buffer_name, (size, dtype) in buffer_info.items():
                var = Variable(buffer_name, (size,), dtype)
                builder.add_line(var.declare_st())
        builder.add_line("")

    else:
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]
            if any(network.is_network_output(out) for out in layer.outputs):
                continue
            plc_type = plc_type_from_onnx_dtype(layer.output_type)
            var = Variable(
                f"layer_{layer.layer_id}_output", (layer.output_size,), plc_type
            )
            with builder.indent():
                builder.add_line(var.declare_st())
            builder.add_line("")

    # Temporary computation variables
    with builder.indent():
        builder.add_line("(* Temporary computation variables *)")
        builder.add_line("i : DINT;")
        builder.add_line("j : DINT;")
        builder.add_line("sum : REAL;")

    # Spatial layer variables
    has_spatial = any(
        isinstance(network.layers[ln], (Conv2DLayer, Pool2DLayer))
        for ln in network.execution_order
    )
    if has_spatial:
        with builder.indent():
            builder.add_line("(* Spatial loop variables for Conv / Pool layers *)")
            builder.add_line("oc : DINT;")
            builder.add_line("oh : DINT;")
            builder.add_line("ow : DINT;")
            builder.add_line("ic : DINT;")
            builder.add_line("kh : DINT;")
            builder.add_line("kw : DINT;")
            builder.add_line("ih : DINT;")
            builder.add_line("iw : DINT;")

    # Transpose layer variables
    for ln in network.execution_order:
        layer = network.layers[ln]
        if isinstance(layer, TransposeLayer) and layer.output_shape:
            ndim = len(layer.output_shape)
            if ndim > 0 and layer.input_size > 1:
                with builder.indent():
                    builder.add_line(
                        f"(* Transpose loop variables for layer {layer.layer_id} *)"
                    )
                    for d in range(ndim):
                        builder.add_line(f"t{layer.layer_id}_d{d} : DINT;")
    # LSTM variables
    has_lstm = any(
        isinstance(network.layers[ln], LSTMLayer) for ln in network.execution_order
    )
    if has_lstm:
        with builder.indent():
            builder.add_line("(* LSTM gate buffers and temporary variables *)")
            builder.add_line("t : DINT;")
            builder.add_line("exp_val : REAL;")
            for ln in network.execution_order:
                layer = network.layers[ln]
                if isinstance(layer, LSTMLayer):
                    h_size = layer.hidden_size
                    builder.add_line(f"(* Layer {layer.layer_id} gate buffers *)")
                    for gate in ("i_gate", "f_gate", "g_gate", "o_gate"):
                        var = Variable(f"{gate}_{layer.layer_id}", (h_size,), "REAL")
                        builder.add_line(var.declare_st())
                    for state in ("h_state", "c_state"):
                        var = Variable(f"{state}_{layer.layer_id}", (h_size,), "REAL")
                        builder.add_line(var.declare_st())

    # GRU variables
    has_gru = any(
        isinstance(network.layers[ln], GRULayer) for ln in network.execution_order
    )
    if has_gru:
        with builder.indent():
            builder.add_line("(* GRU gate buffers and temporary variables *)")
            builder.add_line("t : DINT;")
            builder.add_line("exp_val : REAL;")
            for ln in network.execution_order:
                layer = network.layers[ln]
                if isinstance(layer, GRULayer):
                    h_size = layer.hidden_size
                    builder.add_line(f"(* Layer {layer.layer_id} gate buffers *)")
                    for gate in ("r_gate", "u_gate"):
                        var = Variable(f"{gate}_{layer.layer_id}", (h_size,), "REAL")
                        builder.add_line(var.declare_st())
                    for state in ("h_state", "h_new"):
                        var = Variable(f"{state}_{layer.layer_id}", (h_size,), "REAL")
                        builder.add_line(var.declare_st())

    builder.add_line("")
    builder.add_line("END_VAR")
    builder.add_line("")

    return builder.build()


def collect_all_variables_from_regions(
    optimization_results: Dict[str, OptimizationResult],
) -> Dict[str, Tuple[int, str]]:
    """Collect and merge variables from all regions."""
    all_variables = {}

    for region_id, opt_result in optimization_results.items():
        network = opt_result.ir
        buffer_allocations = opt_result.buffer_allocations or {}
        logger.debug(f"Region {region_id} buffer_allocations: {buffer_allocations}")

        # Buffer allocations
        for tensor_name, buffer_name in buffer_allocations.items():
            if tensor_name in network.tensor_producers:
                producer_name = network.tensor_producers[tensor_name]
                if producer_name in network.layers:
                    layer = network.layers[producer_name]
                    if layer.output_type is None:
                        raise ValueError(f"Layer '{layer.name}' has output_type=None")
                    plc_type = plc_type_from_onnx_dtype(layer.output_type)
                    size = layer.output_size

                    if buffer_name not in all_variables:
                        all_variables[buffer_name] = (size, plc_type)
                    else:
                        existing_size, existing_type = all_variables[buffer_name]
                        all_variables[buffer_name] = (
                            max(existing_size, size),
                            existing_type,
                        )

        # Layer outputs
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]

            if any(network.is_network_output(out) for out in layer.outputs):
                continue

            if any(out in buffer_allocations for out in layer.outputs):
                continue

            if layer.output_type is None:
                raise ValueError(f"Layer '{layer.name}' has output_type=None")

            plc_type = plc_type_from_onnx_dtype(layer.output_type)
            var_name = f"layer_{layer.layer_id}_output"

            if var_name not in all_variables:
                all_variables[var_name] = (layer.output_size, plc_type)

        # LSTM buffers
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]
            if isinstance(layer, LSTMLayer):
                h = layer.hidden_size
                for gate in ("i_gate", "f_gate", "g_gate", "o_gate"):
                    var_name = f"{gate}_{layer.layer_id}"
                    if var_name not in all_variables:
                        all_variables[var_name] = (h, "REAL")
                for state in ("h_state", "c_state"):
                    var_name = f"{state}_{layer.layer_id}"
                    if var_name not in all_variables:
                        all_variables[var_name] = (h, "REAL")

        # GRU buffers
        for layer_name in network.execution_order:
            layer = network.layers[layer_name]
            if isinstance(layer, GRULayer):
                h = layer.hidden_size
                for gate in ("r_gate", "u_gate"):
                    var_name = f"{gate}_{layer.layer_id}"
                    if var_name not in all_variables:
                        all_variables[var_name] = (h, "REAL")
                for state in ("h_state", "h_new"):
                    var_name = f"{state}_{layer.layer_id}"
                    if var_name not in all_variables:
                        all_variables[var_name] = (h, "REAL")

    return all_variables


def generate_merged_var_section(
    all_variables: Dict[str, Tuple[int, str]],
    optimization_results: Optional[Dict[str, "OptimizationResult"]] = None,
) -> STCode:
    """Generate merged VAR section from all variables."""
    builder = STCodeBuilder()
    builder.add_line("VAR")

    if all_variables:
        builder.add_line("    (* Merged variables from all regions *)")
        with builder.indent():
            for var_name, (size, dtype) in sorted(all_variables.items()):
                var = Variable(var_name, (size,), dtype)
                builder.add_line(var.declare_st())
        builder.add_line("")

    # Temporary variables
    with builder.indent():
        builder.add_line("(* Temporary computation variables *)")
        builder.add_line("i : DINT;")
        builder.add_line("j : DINT;")
        builder.add_line("t : DINT;")
        builder.add_line("sum : REAL;")

    with builder.indent():
        builder.add_line("(* Computation helpers *)")
        builder.add_line("max_val : REAL;")
        builder.add_line("exp_val : REAL;")
        builder.add_line("exp_sum : REAL;")

    with builder.indent():
        builder.add_line("(* Spatial loop variables *)")
        builder.add_line("oc : DINT;")
        builder.add_line("oh : DINT;")
        builder.add_line("ow : DINT;")
        builder.add_line("ic : DINT;")
        builder.add_line("kh : DINT;")
        builder.add_line("kw : DINT;")
        builder.add_line("ih : DINT;")
        builder.add_line("iw : DINT;")

    transpose_vars_added = set()
    for region_id, opt_result in (optimization_results or {}).items():
        network = opt_result.ir
        for ln in network.execution_order:
            layer = network.layers[ln]
            if isinstance(layer, TransposeLayer) and layer.output_shape:
                ndim = len(layer.output_shape)
                if ndim > 0 and layer.input_size > 1:
                    for d in range(ndim):
                        var_name = f"t{layer.layer_id}_d{d}"
                        if var_name not in transpose_vars_added:
                            transpose_vars_added.add(var_name)
    if transpose_vars_added:
        with builder.indent():
            builder.add_line("(* Transpose loop variables *)")
            for var_name in sorted(transpose_vars_added):
                builder.add_line(f"{var_name} : DINT;")

    builder.add_line("END_VAR")
    builder.add_line("")

    return builder.build()


# =============================================================================
# Main Entry Points
# =============================================================================


def generate_function_block(
    network: NetworkIR,
    fb_name: str = "NeuralNetwork",
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> STCode:
    """Generate complete function block code for a single NetworkIR."""
    logger.info(
        f"Generating function block '{fb_name}' with {len(network.layers)} layers"
    )

    code = STCode.empty()
    code += generate_header(fb_name)
    code += generate_input_output_vars(network)
    code += generate_constants_section(network)
    code += generate_var_section(network, buffer_allocations)
    code += generate_forward_pass(network, buffer_allocations)
    code += generate_footer()

    logger.info(f"Generated {len(code.lines)} lines of ST code.")
    return code


def generate_model_function_block(
    model: ModelIR,
    optimization_results: Dict[str, OptimizationResult],
    fb_name: str = "NeuralNetwork",
) -> STCode:
    """Generate function block for multi-region ModelIR."""
    from .lowerers import lower_region_to_st

    logger.info(
        f"Generating function block '{fb_name}' for model with {len(model.regions)} regions"
    )

    code = STCode.empty()
    code += generate_header(fb_name)

    # I/O variables
    if model.regions:
        first_region_id = model.regions[0].region_id
        last_region_id = model.regions[-1].region_id
        first_ir = optimization_results[first_region_id].ir
        last_ir = optimization_results[last_region_id].ir

        # Fail fast: require explicit I/O tensor information
        if not first_ir.input_tensors:
            raise ValueError(
                f"First region '{first_region_id}' has no input_tensors. "
                "All regions must have complete I/O tensor information for code generation."
            )
        if not last_ir.output_tensors:
            raise ValueError(
                f"Last region '{last_region_id}' has no output_tensors. "
                "All regions must have complete I/O tensor information for code generation."
            )

        code += STCode.from_lines("    (* Model Inputs/Outputs *)")
        code += generate_input_output_vars(first_ir)

    # Merged constants and variables
    code += generate_merged_constants_section(optimization_results)
    all_variables = collect_all_variables_from_regions(optimization_results)
    code += generate_merged_var_section(all_variables, optimization_results)

    code += st_comment("Forward pass execution")

    # Lower each region
    for region in model.regions:
        code += STCode.blank_line()
        code += st_comment(f"Region: {region.region_id} [{region.kind.name}]")

        if region.region_id not in optimization_results:
            raise KeyError(f"No optimization result for region {region.region_id}")

        optimization_result = optimization_results[region.region_id]
        region_code = lower_region_to_st(region, optimization_result)
        code += region_code

    code += generate_footer()
    return code


def translate_ir_to_st(
    ir: NetworkIR, fb_name: str = "NeuralNetwork", buffer_allocations=None
) -> str:
    """Translate NetworkIR to Structured Text code."""
    builder = STCodeBuilder()
    builder += generate_function_block(ir, fb_name, buffer_allocations)
    code_str = str(builder.build())
    # Post-process: collapse ARRAY[0..0] declarations to scalar when the
    # declared variable is never indexed anywhere in the code. This keeps
    # generated ST tidy (avoids ARRAY[0..0]) while preserving correct
    # indexing semantics when generators do use element access.
    import re

    def _collapse_unused_arrays(s: str) -> str:
        # Find all ARRAY[0..0] declarations like 'name : ARRAY[0..0] OF TYPE;'
        pattern = re.compile(r"^(\s*)(\w+)\s*:\s*ARRAY\[0\.\.0\]\s+OF\s+(\w+);", re.M)

        def repl(match):
            indent, name, typ = match.groups()
            # If the variable is indexed elsewhere (name[) keep array form
            if re.search(rf"\b{re.escape(name)}\s*\[", s):
                return match.group(0)
            # Otherwise collapse to scalar declaration
            return f"{indent}{name} : {typ};"

        return pattern.sub(repl, s)

    code_str = _collapse_unused_arrays(code_str)
    return code_str


def translate_model_to_st(
    model: ModelIR,
    optimization_results: Dict[str, OptimizationResult],
    fb_name: str = "NeuralNetwork",
) -> str:
    """Translate multi-region ModelIR to Structured Text code."""
    builder = STCodeBuilder()
    builder += generate_model_function_block(model, optimization_results, fb_name)
    code_str = str(builder.build())

    # Same post-processing as single-network translation: collapse unused
    # ARRAY[0..0] declarations to scalars when safe.
    import re

    def _collapse_unused_arrays(s: str) -> str:
        pattern = re.compile(r"^(\s*)(\w+)\s*:\s*ARRAY\[0\.\.0\]\s+OF\s+(\w+);", re.M)

        def repl(match):
            indent, name, typ = match.groups()
            if re.search(rf"\b{re.escape(name)}\s*\[", s):
                return match.group(0)
            return f"{indent}{name} : {typ};"

        return pattern.sub(repl, s)

    code_str = _collapse_unused_arrays(code_str)
    return code_str
