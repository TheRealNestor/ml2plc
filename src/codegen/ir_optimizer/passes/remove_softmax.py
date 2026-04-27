"""
Replace Softmax activations with ArgMax when opt-in.

This pass is opt-in because replacing softmax with argmax is lossy
and only valid when probabilities are not required (classification index only).
"""

import logging

from ..base_pass import OptimizationPass
from ...types import (
    NetworkIR,
    ActivationLayer,
    ActivationType,
    ArgMaxLayer,
    FusedLinearLayer,
)

logger = logging.getLogger(__name__)


class RemoveSoftmaxPass(OptimizationPass):
    def get_name(self) -> str:
        return "remove_softmax"

    def optimize(self, network: NetworkIR) -> None:
        """Find ActivationLayer(SOFTMAX) and replace it with ArgMaxLayer.

        This pass only handles standalone ActivationLayer nodes (not fused activations).
        """
        replaced = 0

        for layer_name in list(network.execution_order):
            layer = network.get_layer(layer_name)

            # Case 1: standalone ActivationLayer(softmax)
            if isinstance(layer, ActivationLayer) and layer.activation == ActivationType.SOFTMAX:
                argmax_layer = ArgMaxLayer(
                    name=f"{layer.name}/ArgMax",
                    layer_id=layer.layer_id,
                    op_type="ArgMax",
                    inputs=layer.inputs,
                    outputs=layer.outputs,
                    input_size=layer.input_size,
                    output_size=1,
                    input_shape=layer.input_shape,
                    output_shape=(),
                    input_type=layer.input_type,
                    output_type="INT32",
                    axis=-1,
                )

                # Replace softmax activation with argmax
                self.replace_layer(layer, argmax_layer, network)
                replaced += 1

            # Case 2: Fused linear layer with softmax activation
            elif isinstance(layer, FusedLinearLayer) and layer.activation == ActivationType.SOFTMAX:
                # Create a new fused-linear layer without activation that writes to an intermediate tensor
                linear_out_tensor = f"{layer.name}/linear_out"

                new_linear = FusedLinearLayer(
                    layer_id=layer.layer_id,
                    name=f"{layer.name}/Linear",
                    op_type=layer.op_type,
                    input_size=layer.input_size,
                    output_size=layer.output_size,
                    inputs=layer.inputs,
                    outputs=(linear_out_tensor,),
                    input_shape=layer.input_shape,
                    output_shape=layer.output_shape,
                    input_type=layer.input_type,
                    output_type=layer.output_type,
                    weights=layer.weights,
                    bias=layer.bias,
                    weight_scale=getattr(layer, "weight_scale", None),
                    weight_zero_point=getattr(layer, "weight_zero_point", None),
                    activation=ActivationType.NONE,
                )

                # Create ArgMax layer that consumes the linear output and emits the original outputs
                argmax_layer = ArgMaxLayer(
                    name=f"{layer.name}/ArgMax",
                    layer_id=layer.layer_id,
                    op_type="ArgMax",
                    inputs=(linear_out_tensor,),
                    outputs=layer.outputs,
                    input_size=new_linear.output_size,
                    output_size=1,
                    input_shape=new_linear.output_shape,
                    output_shape=(),
                    input_type=new_linear.output_type,
                    output_type="INT32",
                    axis=-1,
                )

                # Replace the fused layer with argmax (this removes the old layer and adds argmax)
                self.replace_layer(layer, argmax_layer, network)

                # Insert the linear layer that produces the intermediate tensor
                network.layers[new_linear.name] = new_linear

                # Remap argmax input from the original input tensor to the linear output
                # (the optimizer will rebuild graph based on tensor_mapping)
                if layer.inputs:
                    self.remap_tensor(layer.inputs[0], linear_out_tensor)

                replaced += 1

        if replaced:
            logger.info(f"Replaced {replaced} Softmax activations with ArgMax (opt-in)")
