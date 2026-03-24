"""
Base class for IR optimization passes.

Passes can declare which region kinds they support via supports_region_kinds().
This prevents accidental execution on incompatible region types (e.g., DAG-specific
passes should not run on recurrent regions).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Set, Dict
from ..types import NetworkIR, RegionKind, BaseLayer

logger = logging.getLogger(__name__)


class OptimizationPass(ABC):
    """Base class for all optimization passes."""

    def __init__(self):
        self.removed_layers: Set[str] = set()
        self.tensor_mapping: Dict[str, str] = {}  # Maps removed tensor -> replacement

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this optimization pass."""
        pass

    def supports_region_kinds(self) -> List[RegionKind]:
        """
        Declare which region kinds this pass can safely optimize.

        By default, passes only support ACYCLIC regions (traditional DAG optimization).
        Subclasses should override to declare support for additional kinds.

        Returns:
            List of RegionKind values that this pass supports.
            Default: [RegionKind.ACYCLIC]
        """
        return [RegionKind.ACYCLIC]

    @abstractmethod
    def optimize(self, ir: NetworkIR) -> None:
        """
        Analyze IR and mark layers for removal.

        This should populate self.removed_layers and self.tensor_mapping.
        Don't modify the IR directly - just mark what should be removed.

        Note: This is only called after checking supports_region_kinds().
        """
        pass

    def mark_for_removal(self, layer: BaseLayer):
        """
        Mark a layer for removal.

        Note: This does NOT create tensor mappings. Use bypass_layer() if you
        want automatic remapping, or manually call remap_tensor() for custom logic.
        """
        self.removed_layers.add(layer.name)

    def should_remove(self, layer_name: str) -> bool:
        """Check if a layer is marked for removal."""
        return layer_name in self.removed_layers

    def get_mapped_tensor(self, tensor_name: str) -> str:
        """Get the mapped tensor name, following the chain."""
        source = tensor_name
        while source in self.tensor_mapping:
            source = self.tensor_mapping[source]
        return source

    def remap_tensor(self, old_tensor: str, new_tensor: str):
        """
        Remap a tensor to point to a different source.

        Args:
            old_tensor: Original tensor name
            new_tensor: New tensor to use instead
        """
        self.tensor_mapping[old_tensor] = new_tensor

    def bypass_layer(self, layer: BaseLayer):
        """
        Bypass a layer by remapping all its outputs to its first input.

        Common pattern for removing pass-through layers like
        Identity, no-op Reshape, etc.

        Args:
            layer: Layer to bypass
        """
        if layer.outputs and layer.inputs:
            source_tensor = layer.inputs[0]
            for output_tensor in layer.outputs:
                self.remap_tensor(output_tensor, source_tensor)

        self.mark_for_removal(layer)

    def bypass_layer_chain(self, layers: List[BaseLayer]):
        """
        Bypass a chain of layers by remapping the last layer's outputs
        to the first layer's inputs.

        Useful for removing patterns like Quant → Dequant pairs.

        Args:
            layers: List of consecutive layers to bypass (in order)
        """
        if not layers:
            return

        first_layer = layers[0]
        last_layer = layers[-1]

        if last_layer.outputs and first_layer.inputs:
            source_tensor = first_layer.inputs[0]
            for output_tensor in last_layer.outputs:
                self.remap_tensor(output_tensor, source_tensor)

        # Mark all layers in chain for removal
        for layer in layers:
            self.mark_for_removal(layer)

    def replace_layer(
        self, old_layer: BaseLayer, new_layer: BaseLayer, ir: "NetworkIR"
    ):
        """
        Replace a layer with a new layer.

        Removes the old layer from ir.layers and adds the new layer.
        The new layer can have a different name than the old layer.

        Args:
            old_layer: Layer being replaced (will be removed)
            new_layer: New layer to add
            ir: NetworkIR to update
        """
        # Explicitly remove old layer
        if old_layer.name in ir.layers:
            del ir.layers[old_layer.name]

        # Add new layer
        ir.layers[new_layer.name] = new_layer

        # Remap old outputs to new outputs
        for old_out, new_out in zip(old_layer.outputs, new_layer.outputs):
            if old_out != new_out:
                self.remap_tensor(old_out, new_out)
