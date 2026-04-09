"""
Layer Code Generator Registry

Centralizes and organizes all layer-to-ST code generation logic.
Provides clean registration, lookup, and extension mechanisms.

Architecture:
  - GeneratorMetadata: Metadata describing a registered generator
  - LayerCodeGeneratorRegistry: Central registry mapping layer types to generators
  - Generator functions: Implementation for each layer type
  - Registry instance: Singleton registry used throughout codegen
"""

from typing import Dict, Callable, Optional, Set
from dataclasses import dataclass
import logging

from ..types import *
from .st_code import STCode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorMetadata:
    """
    Metadata describing a registered code generator.

    Attributes:
        layer_type: The layer type this generator handles (e.g., MatMulLayer)
        generator: The generator function implementation
        wrap_single_input: Whether to wrap generator to handle single-input convention
        supported_regions: Set of region types this generator can be used in
                          (e.g., {'acyclic', 'recurrent', 'loop'})
        requires_state: Whether this generator requires state information
        fused_activation: Whether this layer can fuse activation functions
    """

    layer_type: type
    generator: Callable
    wrap_single_input: bool = False
    supported_regions: Set[str] = None
    requires_state: bool = False
    fused_activation: bool = False

    def __post_init__(self):
        if self.supported_regions is None:
            object.__setattr__(
                self, "supported_regions", {"acyclic", "recurrent", "loop"}
            )

    def supports_region(self, region_type: str) -> bool:
        """Check if this generator supports a specific region type."""
        return region_type in self.supported_regions


class LayerCodeGeneratorRegistry:
    """
    Central registry for layer code generation.

    Maps layer types to their respective code generation functions and metadata.
    Provides a clean interface for:
    - Registering new layer generators with metadata
    - Looking up generators and their capabilities
    - Handling single-input vs multi-input layers transparently
    - Querying capabilities (supported regions, state requirements, fused activations)
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._generators: Dict[type, GeneratorMetadata] = {}

    def register(
        self,
        layer_type: type,
        generator: Callable,
        wrap_single_input: bool = False,
        supported_regions: Optional[Set[str]] = None,
        requires_state: bool = False,
        fused_activation: bool = False,
    ) -> None:
        """
        Register a code generator for a layer type with metadata.

        Args:
            layer_type: The layer class to register for (e.g., MatMulLayer)
            generator: Function implementing code generation for this layer
            wrap_single_input: If True, wrap generator to handle single-input convention
                              (layer, inputs, output) instead of (layer, input_var, output)
            supported_regions: Set of region types this generator supports
                              (default: {'acyclic', 'recurrent', 'loop'})
            requires_state: Whether this generator requires state information
            fused_activation: Whether this layer can fuse activation functions
        """
        if wrap_single_input:
            # Wrap generator to unpack single input from list
            generator = self._single_input_wrapper(generator)

        if supported_regions is None:
            supported_regions = {"acyclic", "recurrent", "loop"}

        metadata = GeneratorMetadata(
            layer_type=layer_type,
            generator=generator,
            wrap_single_input=wrap_single_input,
            supported_regions=supported_regions,
            requires_state=requires_state,
            fused_activation=fused_activation,
        )

        self._generators[layer_type] = metadata
        logger.debug(
            f"Registered generator for {layer_type.__name__} "
            f"(regions: {', '.join(supported_regions)}, state: {requires_state})"
        )

    def get(self, layer_type: type) -> Optional[Callable]:
        """
        Look up a generator for a layer type.

        Args:
            layer_type: The layer class to look up

        Returns:
            Generator function, or None if not registered
        """
        metadata = self._generators.get(layer_type)
        return metadata.generator if metadata else None

    def get_metadata(self, layer_type: type) -> Optional[GeneratorMetadata]:
        """
        Look up metadata for a generator.

        Args:
            layer_type: The layer class to look up

        Returns:
            GeneratorMetadata object, or None if not registered
        """
        return self._generators.get(layer_type)

    def has_generator(self, layer_type: type) -> bool:
        """Check if a generator is registered for this layer type."""
        return layer_type in self._generators

    def generate(
        self,
        layer: BaseLayer,
        input_vars: list,
        output_var: str,
    ) -> STCode:
        """
        Generate code for a layer.

        Args:
            layer: The layer to generate code for
            input_vars: List of input variable names
            output_var: Output variable name

        Returns:
            Generated STCode for this layer

        Raises:
            ValueError: If no generator is registered for this layer type
        """
        layer_type = type(layer)
        generator = self.get(layer_type)

        if generator is None:
            raise ValueError(
                f"No code generator registered for layer type {layer_type.__name__}. "
                f"Register one using registry.register({layer_type.__name__}, generator_func)."
            )

        return generator(layer, input_vars, output_var)

    @staticmethod
    def _single_input_wrapper(generator_func: Callable) -> Callable:
        """
        Wrap a generator function to handle single-input layers.

        Converts:
          generator_func(layer, input_var, output_var)
        To:
          wrapper(layer, input_vars_list, output_var)

        This provides a uniform interface for all generators.
        """

        def wrapper(layer, input_vars, output_var):
            if not input_vars:
                layer_name = getattr(layer, "name", type(layer).__name__)
                layer_id = getattr(layer, "layer_id", "?")
                declared_inputs = getattr(layer, "inputs", [])
                raise ValueError(
                    f"Layer {layer_id} ({layer_name}) has no resolved input variables "
                    f"for ST generation. Declared inputs: {declared_inputs}"
                )

            return generator_func(layer, input_vars[0], output_var)

        return wrapper

    def __repr__(self) -> str:
        """Return a string representation of registered generators."""
        registered_types = sorted([t.__name__ for t in self._generators.keys()])
        state_required = sum(1 for m in self._generators.values() if m.requires_state)
        fused_act = sum(1 for m in self._generators.values() if m.fused_activation)
        return (
            f"LayerCodeGeneratorRegistry("
            f"{len(registered_types)} registered, "
            f"{state_required} require state, "
            f"{fused_act} support fused activation)"
        )


# Global registry instance (singleton)
_GLOBAL_REGISTRY: Optional[LayerCodeGeneratorRegistry] = None


def get_global_registry() -> LayerCodeGeneratorRegistry:
    """
    Get or initialize the global layer code generator registry.

    This ensures all modules use a single consistent registry.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = LayerCodeGeneratorRegistry()
        _initialize_default_generators(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _initialize_default_generators(registry: LayerCodeGeneratorRegistry) -> None:
    """
    Initialize the registry with default generators for all supported layers.

    Called once when the global registry is created.
    """
    # Import here to avoid circular imports
    from . import generator

    registry.register(
        MatMulLayer,
        generator.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        GemmLayer,
        generator.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        FusedGemmLayer,
        generator.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        FusedLinearLayer,
        generator.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(AddLayer, generator.generate_add_code, wrap_single_input=False)
    registry.register(
        ReshapeLayer, generator.generate_reshape_code, wrap_single_input=True
    )
    registry.register(
        ActivationLayer,
        generator.generate_activation_layer_code,
        wrap_single_input=True,
    )
    registry.register(
        QuantizeLinearLayer,
        generator.generate_quantize_linear_code,
        wrap_single_input=True,
    )
    registry.register(
        DequantizeLinearLayer,
        generator.generate_dequantize_linear_code,
        wrap_single_input=True,
    )
    registry.register(
        DropoutLayer, generator.generate_dropout_code, wrap_single_input=True
    )
    registry.register(
        Conv2DLayer,
        generator.generate_conv2d_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        Pool2DLayer, generator.generate_pool2d_code, wrap_single_input=True
    )
    registry.register(
        FlattenLayer, generator.generate_flatten_code, wrap_single_input=True
    )
    registry.register(
        TransposeLayer, generator.generate_transpose_code, wrap_single_input=True
    )
    registry.register(
        BatchNormLayer, generator.generate_batchnorm_code, wrap_single_input=True
    )
    registry.register(
        SqueezeLayer, generator.generate_squeeze_code, wrap_single_input=True
    )
    registry.register(
        LSTMLayer,
        generator.generate_lstm_code,
        wrap_single_input=True,
        requires_state=True,
        supported_regions={"recurrent", "loop"},
    )
    registry.register(
        GRULayer,
        generator.generate_gru_code,
        wrap_single_input=True,
        requires_state=True,
        supported_regions={"recurrent", "loop"},
    )

    registry.register(
        CastLayer,
        lambda layer, inputs, output: generator.generate_cast_code(
            layer, inputs[0], output
        ),
    )
    registry.register(
        SliceLayer,
        lambda layer, inputs, output: generator.generate_slice_code(
            layer, inputs[0], output
        ),
    )
    registry.register(
        ConcatLayer,
        lambda layer, inputs, output: generator.generate_concat_code(
            layer, inputs, output
        ),
    )
    registry.register(
        UnsqueezeLayer,
        lambda layer, inputs, output: generator.generate_unsqueeze_code(
            layer, inputs[0], output
        ),
    )
    registry.register(
        ShapeLayer,
        lambda layer, inputs, output: generator.generate_shape_code(
            layer, inputs[0], output
        ),
    )
    registry.register(
        ExpandLayer,
        lambda layer, inputs, output: generator.generate_expand_code(
            layer, inputs[0], output
        ),
    )
    registry.register(
        GatherLayer,
        lambda layer, inputs, output: generator.generate_gather_code(
            layer, inputs[0], output
        ),
    )

    logger.info(f"Initialized default generators: {registry}")
