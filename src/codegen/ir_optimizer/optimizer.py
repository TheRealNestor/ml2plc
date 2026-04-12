"""
Main IR optimizer that orchestrates optimization passes.

Uses region strategies to dispatch passes based on region kind.
Only applies passes that declare support for each region type.
"""

import logging
from typing import List, Optional, Dict

from ..graph import LayerGraph
from ..ir_utils import TensorMapBuilder
from ..types import NetworkIR, ModelIR, RegionKind
from .base_pass import OptimizationPass
from .result import OptimizationResult
from .passes import (
    RemoveIdentityPass,
    RemoveNoOpReshapePass,
    RemoveRedundantQuantPairPass,
    RemoveWeightDequantPass,
    FuseLinearActivationPass,
    BufferAllocationPass,
    RemoveDropoutPass,
)
from .region_strategies import optimize_region_with_passes, validate_pass_applicability

logger = logging.getLogger(__name__)

DEFAULT_PASSES: List[OptimizationPass] = [
    RemoveDropoutPass(),
    RemoveIdentityPass(),
    RemoveWeightDequantPass(),
    RemoveNoOpReshapePass(),
    RemoveRedundantQuantPairPass(),
    FuseLinearActivationPass(),
    BufferAllocationPass(),  # Produces code generation hints, doesn't modify IR
]


def optimize_model_regions(
    model_ir: ModelIR, passes: Optional[List[OptimizationPass]] = None
) -> Dict[str, OptimizationResult]:
    """
    Run region-aware optimizations on all regions in the model.

    Each region is optimized with only the passes it declares support for.

    Args:
        model_ir: The regionized model.
        passes: Optional list of custom passes. Uses DEFAULT_PASSES if not provided.

    Returns:
        Dictionary mapping region_id to OptimizationResult.
    """
    if passes is None:
        passes = DEFAULT_PASSES

    logger.info(f"Optimizing {len(model_ir.regions)} region(s)")

    # Validate pass applicability for diagnostics
    applicability = validate_pass_applicability(model_ir, passes)
    for region_id, applicable_pass_names in applicability.items():
        logger.debug(f"  {region_id}: {applicable_pass_names}")

    results = {}

    for region in model_ir.regions:
        logger.info(f"Optimizing region {region.region_id} ({region.kind.value})")

        result = optimize_region_with_passes(region, passes)
        results[region.region_id] = result

    return results


class IROptimizer:
    """Applies optimization passes to NetworkIR."""

    def __init__(self, ir: NetworkIR, passes: Optional[List[OptimizationPass]] = None):
        self.ir = ir
        self.passes = passes if passes is not None else DEFAULT_PASSES

    def optimize(self) -> OptimizationResult:
        """
        Apply optimization passes to the IR.

        Returns:
            OptimizationResult containing optimized IR and optional code generation hints
        """
        initial_layer_count = len(self.ir.layers)
        logger.info(f"Starting optimization with {initial_layer_count} layers")

        buffer_allocations = None

        for pass_instance in self.passes:
            logger.info(f"Running pass: {pass_instance.get_name()}")
            pass_instance.optimize(self.ir)

            # Rebuild IR if pass modified the graph structure
            if pass_instance.removed_layers or pass_instance.tensor_mapping:
                self.ir = self._rebuild_ir(
                    pass_instance.removed_layers, pass_instance.tensor_mapping
                )

            # Extract code generation hints (doesn't modify IR)
            if (
                hasattr(pass_instance, "buffer_assignments")
                and pass_instance.buffer_assignments
            ):
                buffer_allocations = {
                    tensor: alloc.buffer_name
                    for tensor, alloc in pass_instance.buffer_assignments.items()
                }
                logger.info(f"Extracted {len(buffer_allocations)} buffer allocations")

        final_layer_count = len(self.ir.layers)
        logger.info(
            f"Optimization complete: {initial_layer_count} -> {final_layer_count} layers "
            f"({initial_layer_count - final_layer_count} removed)"
        )

        return OptimizationResult(ir=self.ir, buffer_allocations=buffer_allocations)

    def _filter_removed_layers(self, removed_layers: set) -> dict:
        """Remove layers marked for deletion."""
        return {
            name: layer
            for name, layer in self.ir.layers.items()
            if name not in removed_layers
        }

    def _follow_tensor_mapping(self, tensor: str, tensor_mapping: dict) -> str:
        """
        Follow tensor mapping chain to find final tensor.

        Handles transitive mappings: A->B, B->C results in A->C
        """
        source = tensor
        while source in tensor_mapping:
            source = tensor_mapping[source]
        return source

    def _rewire_layers(self, layers: dict, tensor_mapping: dict) -> None:
        """Update layer inputs and outputs of individual layers, following tensor remapping."""
        for layer in layers.values():
            new_inputs = [
                self._follow_tensor_mapping(inp, tensor_mapping) for inp in layer.inputs
            ]
            if new_inputs != list(layer.inputs):
                object.__setattr__(layer, "inputs", tuple(new_inputs))

            new_outputs = [
                self._follow_tensor_mapping(out, tensor_mapping)
                for out in layer.outputs
            ]
            if new_outputs != list(layer.outputs):
                object.__setattr__(layer, "outputs", tuple(new_outputs))

    def _remap_network_outputs(self, tensor_mapping: dict) -> list:
        """Remap network output tensors, i.e. updates the final network output tensor list."""
        new_outputs = []

        for out in self.ir.output_tensors:
            remapped = self._follow_tensor_mapping(out, tensor_mapping)
            new_outputs.append(remapped)

            if remapped != out:
                logger.info(f"Remapped network output: {out} -> {remapped}")

        return new_outputs

    def _rebuild_graph_structure(self, layers: dict) -> tuple[dict, dict]:
        """
        Rebuild tensor producer/consumer maps.

        Delegates to centralized graph structure builder.

        Returns:
            (tensor_producers, tensor_consumers) tuple
        """
        return TensorMapBuilder.build(layers).as_tuple()

    def _renumber_layer_ids(self, layers: dict, execution_order: list) -> dict:
        """Renumber layer IDs to be sequential based on execution order."""
        new_layers = {}
        for new_id, layer_name in enumerate(execution_order):
            layer = layers[layer_name]
            # Update layer_id using frozen dataclass workaround
            object.__setattr__(layer, "layer_id", new_id)
            new_layers[layer_name] = layer
        return new_layers

    def _rebuild_ir(self, removed_layers: set, tensor_mapping: dict) -> NetworkIR:
        """Rebuild IR with removed layers and rewired tensors.

        Args:
            removed_layers: Set of layer names to remove.
            tensor_mapping: Dict mapping old tensor names to new tensor names.
        Returns:
            Rebuilt NetworkIR with layers removed and tensors rewired.
        """

        # 1. Remove layers
        new_layers = self._filter_removed_layers(removed_layers)

        # 2. Rewire tensor references in remaining layers
        self._rewire_layers(new_layers, tensor_mapping)

        # 3. Remap network outputs
        new_output_tensors = self._remap_network_outputs(tensor_mapping)

        # 4. Rebuild graph structure
        new_tensor_producers, new_tensor_consumers = self._rebuild_graph_structure(
            new_layers
        )

        # 5. Rebuild execution order using LayerGraph (single source of truth)
        temp_ir = NetworkIR(
            layers=new_layers,
            execution_order=[],  # Temporary
            tensor_producers=new_tensor_producers,
            tensor_consumers=new_tensor_consumers,
            input_tensors=self.ir.input_tensors,
            output_tensors=tuple(new_output_tensors),
            state_tensors=self.ir.state_tensors,
        )
        layer_graph = LayerGraph(temp_ir)
        new_execution_order = layer_graph.get_execution_order()

        # 6. Renumber layer IDs sequentially
        new_layers = self._renumber_layer_ids(new_layers, new_execution_order)

        return NetworkIR(
            layers=new_layers,
            execution_order=new_execution_order,
            tensor_producers=new_tensor_producers,
            tensor_consumers=new_tensor_consumers,
            input_tensors=self.ir.input_tensors,
            output_tensors=tuple(new_output_tensors),
            state_tensors=self.ir.state_tensors,  # Preserve state tensor information
        )
