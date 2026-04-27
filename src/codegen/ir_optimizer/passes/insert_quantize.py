"""Insert QuantizeLinear/DequantizeLinear pairs before linear operators.

This pass inserts activation quantization nodes before supported linear ops
(MatMul, Gemm, Linear-like) using available weight metadata if present.
It's a conservative, opt-in transformation used by the quantization scaffold.
"""

import logging
from typing import Tuple
import numpy as np

from ..base_pass import OptimizationPass
from ...types import (
    NetworkIR,
    QuantizeLinearLayer,
    DequantizeLinearLayer,
    MatMulLayer,
    GemmLayer,
    LinearLayer,
    FusedLinearLayer,
)

logger = logging.getLogger(__name__)


class InsertQuantizePass(OptimizationPass):
    def get_name(self) -> str:
        return "insert_quantize"

    def supports_region_kinds(self):
        return [ ]  # We'll rely on optimizer validation; default empty means ACYCLIC by default

    def _get_scale_zp(self, layer) -> Tuple[float, int]:
        # Prefer layer metadata if present
        try:
            if hasattr(layer, "weight_scale") and layer.weight_scale is not None:
                scale = float(np.asarray(layer.weight_scale).item())
                zp = int(np.asarray(layer.weight_zero_point).item()) if hasattr(layer, "weight_zero_point") else 0
                return scale, zp
        except Exception:
            pass

        # Fallback defaults
        return 1.0, 0

    def optimize(self, network: NetworkIR) -> None:
        inserted = 0

        for layer_name in list(network.execution_order):
            layer = network.get_layer(layer_name)

            if not isinstance(layer, (MatMulLayer, GemmLayer, LinearLayer, FusedLinearLayer)):
                continue

            # Create unique intermediate tensor names
            q_tensor = f"{layer.name}/quant_in"
            dq_tensor = f"{layer.name}/dequant_in"

            scale, zp = self._get_scale_zp(layer)

            # Create quantize + dequantize layers
            q_layer = QuantizeLinearLayer(
                layer_id=layer.layer_id,
                name=f"{layer.name}/Quantize",
                op_type="QuantizeLinear",
                input_size=1,
                output_size=1,
                inputs=layer.inputs,
                outputs=(q_tensor,),
                scale=np.array(scale),
                zero_point=np.array(zp),
                axis=None,
            )

            dq_layer = DequantizeLinearLayer(
                layer_id=layer.layer_id,
                name=f"{layer.name}/Dequantize",
                op_type="DequantizeLinear",
                input_size=1,
                output_size=1,
                inputs=(q_tensor,),
                outputs=(dq_tensor,),
                scale=np.array(scale),
                zero_point=np.array(zp),
                axis=None,
            )

            # Insert into layers dict and execution order before current layer
            network.layers[q_layer.name] = q_layer
            network.layers[dq_layer.name] = dq_layer

            idx = network.execution_order.index(layer_name)
            network.execution_order.insert(idx, q_layer.name)
            network.execution_order.insert(idx + 1, dq_layer.name)

            # Create a replacement for the original layer with input rewired to the dequant output
            # We preserve most attributes depending on layer type
            if isinstance(layer, FusedLinearLayer):
                new_layer = FusedLinearLayer(
                    layer_id=layer.layer_id,
                    name=layer.name,
                    op_type=layer.op_type,
                    input_size=layer.input_size,
                    output_size=layer.output_size,
                    inputs=(dq_tensor,),
                    outputs=layer.outputs,
                    input_shape=layer.input_shape,
                    output_shape=layer.output_shape,
                    input_type=layer.input_type,
                    output_type=layer.output_type,
                    weights=layer.weights,
                    bias=layer.bias,
                    weight_scale=getattr(layer, "weight_scale", None),
                    weight_zero_point=getattr(layer, "weight_zero_point", None),
                    activation=layer.activation,
                )
            elif isinstance(layer, GemmLayer):
                new_layer = GemmLayer(
                    layer_id=layer.layer_id,
                    name=layer.name,
                    op_type=layer.op_type,
                    input_size=layer.input_size,
                    output_size=layer.output_size,
                    inputs=(dq_tensor,),
                    outputs=layer.outputs,
                    input_shape=layer.input_shape,
                    output_shape=layer.output_shape,
                    input_type=layer.input_type,
                    output_type=layer.output_type,
                    weights=layer.weights,
                    bias=layer.bias,
                    transA=getattr(layer, "transA", False),
                    transB=getattr(layer, "transB", False),
                    alpha=getattr(layer, "alpha", 1.0),
                    beta=getattr(layer, "beta", 1.0),
                )
            else:
                # Generic Linear-like
                new_layer = LinearLayer(
                    layer_id=layer.layer_id,
                    name=layer.name,
                    op_type=layer.op_type,
                    input_size=layer.input_size,
                    output_size=layer.output_size,
                    inputs=(dq_tensor,),
                    outputs=layer.outputs,
                    input_shape=layer.input_shape,
                    output_shape=layer.output_shape,
                    input_type=layer.input_type,
                    output_type=layer.output_type,
                    weights=getattr(layer, "weights", None),
                    bias=getattr(layer, "bias", None),
                    weight_scale=getattr(layer, "weight_scale", None),
                    weight_zero_point=getattr(layer, "weight_zero_point", None),
                )

            # Replace layer in-place
            self.replace_layer(layer, new_layer, network)

            inserted += 1

        if inserted:
            logger.info(f"Inserted quantize/dequantize pairs for {inserted} linear layer(s)")
