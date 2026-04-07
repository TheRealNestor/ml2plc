"""
Layer Code Generator Registry

Centralizes and organizes all layer-to-ST code generation logic.
Provides clean registration, lookup, and extension mechanisms.

Architecture:
  - LayerCodeGeneratorRegistry: Central registry mapping layer types to generators
  - Generator functions: Implementation for each layer type
  - Registry instance: Singleton registry used throughout codegen
"""

from typing import Dict, Callable, Optional
import logging

from ..types import (
    BaseLayer,
    MatMulLayer,
    GemmLayer,
    FusedGemmLayer,
    FusedLinearLayer,
    AddLayer,
    ReshapeLayer,
    ActivationLayer,
    QuantizeLinearLayer,
    DequantizeLinearLayer,
    DropoutLayer,
    Conv2DLayer,
    Pool2DLayer,
    FlattenLayer,
    TransposeLayer,
    BatchNormLayer,
    SqueezeLayer,
    LSTMLayer,
    GRULayer,
)
from .st_code import STCode

logger = logging.getLogger(__name__)


class LayerCodeGeneratorRegistry:
    """
    Central registry for layer code generation.

    Maps layer types to their respective code generation functions.
    Provides a clean interface for:
    - Registering new layer generators
    - Looking up generators by layer type
    - Handling single-input vs multi-input layers transparently
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._generators: Dict[type, Callable] = {}

    def register(
        self,
        layer_type: type,
        generator: Callable,
        wrap_single_input: bool = False,
    ) -> None:
        """
        Register a code generator for a layer type.

        Args:
            layer_type: The layer class to register for (e.g., MatMulLayer)
            generator: Function implementing code generation for this layer
            wrap_single_input: If True, wrap generator to handle single-input convention
                              (layer, inputs, output) instead of (layer, input_var, output)
        """
        if wrap_single_input:
            # Wrap generator to unpack single input from list
            generator = self._single_input_wrapper(generator)

        self._generators[layer_type] = generator
        logger.debug(f"Registered generator for {layer_type.__name__}")

    def get(self, layer_type: type) -> Optional[Callable]:
        """
        Look up a generator for a layer type.

        Args:
            layer_type: The layer class to look up

        Returns:
            Generator function, or None if not registered
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
        return f"LayerCodeGeneratorRegistry({len(registered_types)} registered: {', '.join(registered_types)})"


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

    # Register all layer generators
    # Note: Most use wrap_single_input=True because they follow the
    # (layer, input_var, output_var) convention

    registry.register(
        MatMulLayer, generator.generate_linear_layer_code, wrap_single_input=True
    )
    registry.register(
        GemmLayer, generator.generate_linear_layer_code, wrap_single_input=True
    )
    registry.register(
        FusedGemmLayer, generator.generate_linear_layer_code, wrap_single_input=True
    )
    registry.register(
        FusedLinearLayer, generator.generate_linear_layer_code, wrap_single_input=True
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
        Conv2DLayer, generator.generate_conv2d_code, wrap_single_input=True
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
    registry.register(LSTMLayer, generator.generate_lstm_code, wrap_single_input=True)
    registry.register(GRULayer, generator.generate_gru_code, wrap_single_input=True)

    logger.info(f"Initialized default generators: {registry}")
