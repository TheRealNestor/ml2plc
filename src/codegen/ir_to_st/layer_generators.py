"""
Layer code generator registry.

Central registry mapping layer types to their code generation implementations.
Maintains metadata about each generator (region support, state requirements, etc).

Provides:
  - GeneratorMetadata: Describes a registered generator
  - LayerCodeGeneratorRegistry: Central lookup and code generation
  - get_global_registry(): Singleton instance
"""

from typing import Dict, Callable, Optional, Set
from dataclasses import dataclass
import logging

from ..types import *
from .st_code import STCode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorMetadata:
    """Metadata describing a registered code generator."""

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
    """Central registry for layer-to-ST code generation."""

    def __init__(self):
        """Initialize empty registry."""
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
        """Register a code generator for a layer type."""
        if wrap_single_input:
            generator = self._single_input_wrapper(generator)

        metadata = GeneratorMetadata(
            layer_type=layer_type,
            generator=generator,
            wrap_single_input=wrap_single_input,
            supported_regions=supported_regions or {"acyclic", "recurrent", "loop"},
            requires_state=requires_state,
            fused_activation=fused_activation,
        )

        self._generators[layer_type] = metadata
        logger.debug(f"Registered {layer_type.__name__}")

    def get(self, layer_type: type) -> Optional[Callable]:
        """Look up a generator function for a layer type."""
        metadata = self._generators.get(layer_type)
        return metadata.generator if metadata else None

    def get_metadata(self, layer_type: type) -> Optional[GeneratorMetadata]:
        """Look up metadata for a generator."""
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
        """Generate code for a layer using its registered generator."""
        layer_type = type(layer)
        generator = self.get(layer_type)

        if generator is None:
            raise ValueError(
                f"No generator for {layer_type.__name__}. "
                f"Register with: registry.register({layer_type.__name__}, func)"
            )

        # Generate code
        code = generator(layer, input_vars, output_var)

        logger.debug(
            f"Generated code for {layer_type.__name__} '{layer.name}' "
            f"(output_size={layer.output_size}, primary_output={layer.primary_output})"
        )

        return code

    @staticmethod
    def _single_input_wrapper(generator_func: Callable) -> Callable:
        """Wrap generator to convert (layer, input_var, output) from (layer, [inputs], output)."""

        def wrapper(layer, input_vars, output_var):
            if not input_vars:
                raise ValueError(
                    f"Layer {layer.layer_id} ({layer.name}) has no resolved input variables"
                )
            return generator_func(layer, input_vars[0], output_var)

        return wrapper

    def __repr__(self) -> str:
        """String representation of registry."""
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
    """Get or initialize the global registry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = LayerCodeGeneratorRegistry()
        _initialize_default_generators(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _initialize_default_generators(registry: LayerCodeGeneratorRegistry) -> None:
    """Initialize registry with default layer generators."""
    from . import layers as gen

    # Linear/matmul layers
    registry.register(
        MatMulLayer,
        gen.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        GemmLayer,
        gen.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        FusedGemmLayer,
        gen.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(
        FusedLinearLayer,
        gen.generate_linear_layer_code,
        wrap_single_input=True,
        fused_activation=True,
    )

    # Elementwise and data movement
    registry.register(AddLayer, gen.generate_add_code, wrap_single_input=False)
    registry.register(
        ReduceMeanLayer, gen.generate_reduce_mean_code, wrap_single_input=True
    )
    registry.register(
        ReduceProdLayer, gen.generate_reduce_prod_code, wrap_single_input=True
    )
    registry.register(
        RuntimeMatMulLayer,
        gen.generate_runtime_matmul_code,
        wrap_single_input=False,
    )
    registry.register(EinsumLayer, gen.generate_einsum_code, wrap_single_input=True)
    registry.register(
        BinaryElementwiseLayer,
        gen.generate_binary_elementwise_code,
        wrap_single_input=False,
    )
    registry.register(
        UnaryElementwiseLayer,
        gen.generate_unary_elementwise_code,
        wrap_single_input=True,
    )
    registry.register(ReshapeLayer, gen.generate_reshape_code, wrap_single_input=True)
    registry.register(FlattenLayer, gen.generate_flatten_code, wrap_single_input=True)
    registry.register(SqueezeLayer, gen.generate_squeeze_code, wrap_single_input=True)
    registry.register(
        TransposeLayer, gen.generate_transpose_code, wrap_single_input=True
    )
    registry.register(
        UnsqueezeLayer, gen.generate_unsqueeze_code, wrap_single_input=True
    )
    registry.register(ExpandLayer, gen.generate_expand_code, wrap_single_input=True)
    registry.register(SliceLayer, lambda l, i, o: gen.generate_slice_code(l, i[0], o))
    registry.register(CastLayer, lambda l, i, o: gen.generate_cast_code(l, i[0], o))
    registry.register(ShapeLayer, lambda l, i, o: gen.generate_shape_code(l, i[0], o))
    registry.register(GatherLayer, lambda l, i, o: gen.generate_gather_code(l, i[0], o))

    # Concatenation (multiple inputs)
    registry.register(
        ConcatLayer,
        lambda layer, inputs, output: gen.generate_concat_code(layer, inputs, output),
    )

    # Activation layers
    registry.register(
        ActivationLayer,
        gen.generate_activation_layer_code,
        wrap_single_input=True,
    )

    # Quantization
    registry.register(
        QuantizeLinearLayer,
        gen.generate_quantize_linear_code,
        wrap_single_input=True,
    )
    registry.register(
        DequantizeLinearLayer,
        gen.generate_dequantize_linear_code,
        wrap_single_input=True,
    )

    # Dropout (no-op at inference)
    registry.register(DropoutLayer, gen.generate_dropout_code, wrap_single_input=True)

    # Spatial layers (Conv, Pool, BatchNorm)
    registry.register(
        Conv2DLayer,
        gen.generate_conv2d_code,
        wrap_single_input=True,
        fused_activation=True,
    )
    registry.register(Pool2DLayer, gen.generate_pool2d_code, wrap_single_input=True)
    registry.register(
        BatchNormLayer, gen.generate_batchnorm_code, wrap_single_input=True
    )

    # Recurrent layers
    registry.register(
        LSTMLayer,
        gen.generate_lstm_code,
        wrap_single_input=True,
        requires_state=True,
        supported_regions={"recurrent", "loop"},
    )
    registry.register(
        GRULayer,
        gen.generate_gru_code,
        wrap_single_input=True,
        requires_state=True,
        supported_regions={"recurrent", "loop"},
    )

    logger.info(f"Initialized registry: {registry}")
