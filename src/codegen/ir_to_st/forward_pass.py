"""
Forward pass code generation orchestration.

Generates complete forward pass code for a network by:
1. Resolving layer input/output variables (handling network I/O and intermediate buffers)
2. Using the layer generator registry to produce code for each layer
3. Combining all layer code in execution order

This module acts as the bridge between high-level network IR and layer-specific code generation.
"""

from typing import Dict, Optional, List
from ..types import BaseLayer, NetworkIR
from .st_code import STCode
from .variable import Variable


def get_layer_input_vars(
    layer: BaseLayer,
    network: NetworkIR,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Get all input variable names for a layer."""
    input_vars = []

    for inp_tensor in layer.inputs:
        if network.is_network_input(inp_tensor):
            input_vars.append("input_data")
        elif buffer_allocations and inp_tensor in buffer_allocations:
            input_vars.append(buffer_allocations[inp_tensor])
        elif inp_tensor in network.tensor_producers:
            producer_name = network.tensor_producers[inp_tensor]
            if producer_name in network.layers:
                producer_layer = network.layers[producer_name]
                input_vars.append(f"layer_{producer_layer.layer_id}_output")

    if len(input_vars) != len(layer.inputs):
        unresolved = [
            t
            for t in layer.inputs
            if not network.is_network_input(t)
            and t not in (buffer_allocations or {})
            and (
                t not in network.tensor_producers
                or network.tensor_producers.get(t) not in network.layers
            )
        ]
        if unresolved:
            raise ValueError(
                f"Layer {layer.layer_id} has unresolved input tensors: {unresolved}"
            )

    return input_vars


def get_layer_output_var(
    layer: BaseLayer,
    network: NetworkIR,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> str:
    """Get output variable name for a layer."""
    if not layer.outputs:
        raise ValueError(f"Layer {layer.layer_id} has no outputs")

    output_tensor = layer.outputs[0]

    if network.is_network_output(output_tensor):
        return "output_data"

    if buffer_allocations and output_tensor in buffer_allocations:
        return buffer_allocations[output_tensor]

    return f"layer_{layer.layer_id}_output"


def generate_forward_pass(
    network: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> STCode:
    """Generate forward pass code using registry from layer_generators.py."""
    from .layer_generators import get_global_registry

    registry = get_global_registry()
    code = STCode.empty()

    for layer_name in network.execution_order:
        layer = network.layers[layer_name]

        # Resolve input and output variable names (strings) and convert them
        # into Variable objects so generators may call .at() / .declare_st().
        input_var_names = get_layer_input_vars(layer, network, buffer_allocations)
        output_var_name = get_layer_output_var(layer, network, buffer_allocations)

        # Determine shapes from layer metadata if available, otherwise default to scalar.
        # For layers with multiple inputs the layer.input_shape may refer to the primary
        # input; fall back to scalar for unknown shapes.
        def mk_var(name: str, shape_hint: Optional[tuple[int, ...]] = None) -> Variable:
            shape = tuple(shape_hint) if shape_hint else (1,)
            return Variable(name=name, shape=shape)

        # Map input names -> Variable objects. If layer.input_shape is present use it.
        input_vars = [mk_var(n, layer.input_shape) for n in input_var_names]

        output_var = mk_var(output_var_name, layer.output_shape)

        layer_code = registry.generate(layer, input_vars, output_var)
        code += layer_code
        code += STCode.blank_line()

    return code
