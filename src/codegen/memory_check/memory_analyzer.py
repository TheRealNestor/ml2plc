"""
Memory analyzer for PLC deployment.
Computes memory requirements and validates against device limits.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ..ir_to_st.type_conversion import get_type_size_bytes

from ..types import (
    NetworkIR,
    BaseLayer,
    MatMulLayer,
    GemmLayer,
    FusedGemmLayer,
    FusedLinearLayer,
    AddLayer,
    QuantizeLinearLayer,
    DequantizeLinearLayer,
)

logger = logging.getLogger(__name__)

DEFAULT_ELEMENT_SIZE = 4  # bytes (float32)

DTYPE_BYTE_ALIASES: Dict[str, int] = {
    "tensor(float)": 4,
    "float": 4,
    "float32": 4,
    "fp32": 4,
    "tensor(double)": 8,
    "double": 8,
    "float64": 8,
    "fp64": 8,
    "tensor(float16)": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
}

LAYER_TAG_RULES: Dict[str, Tuple[str, ...]] = {
    "lstm": ("recurrent", "lstm"),
    "gru": ("recurrent", "gru"),
    "rnn": ("recurrent",),
    "attention": ("attention",),
    "attn": ("attention",),
    "einsum": ("attention_like",),
}

LINEAR_LAYER_TYPES: Tuple[type, ...] = (
    MatMulLayer,
    GemmLayer,
    FusedGemmLayer,
    FusedLinearLayer,
)


@dataclass
class MemoryBreakdown:
    """Detailed memory usage breakdown."""

    weights_bytes: int = 0
    biases_bytes: int = 0
    activations_bytes: int = 0
    constants_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.biases_bytes
            + self.activations_bytes
            + self.constants_bytes
        )

    @property
    def total_kb(self) -> float:
        return self.total_bytes / 1024

    def __str__(self) -> str:
        return (
            f"Memory Breakdown:\n"
            f"  Weights:     {self.weights_bytes:>8} bytes ({self.weights_bytes/1024:.2f} KB)\n"
            f"  Biases:      {self.biases_bytes:>8} bytes ({self.biases_bytes/1024:.2f} KB)\n"
            f"  Activations: {self.activations_bytes:>8} bytes ({self.activations_bytes/1024:.2f} KB)\n"
            f"  Constants:   {self.constants_bytes:>8} bytes ({self.constants_bytes/1024:.2f} KB)\n"
            f"  ─────────────────────────────\n"
            f"  TOTAL:       {self.total_bytes:>8} bytes ({self.total_kb:.2f} KB)"
        )


@dataclass
class MemoryCheckResult:
    """Result of memory validation check."""

    fits_in_memory: bool
    breakdown: MemoryBreakdown
    limit_bytes: int
    utilization_percent: float
    warnings: List[str]
    errors: List[str]

    def __str__(self) -> str:
        status = "✓ PASS" if self.fits_in_memory else "✗ FAIL"
        limit_kb = self.limit_bytes / 1024
        return (
            f"\nMemory Check: {status}\n"
            f"{'=' * 40}\n"
            f"{self.breakdown}\n"
            f"{'=' * 40}\n"
            f"Device Limit: {limit_kb:.2f} KB\n"
            f"Utilization:  {self.utilization_percent:.1f}%\n"
            f"Remaining:    {(self.limit_bytes - self.breakdown.total_bytes)/1024:.2f} KB"
        )


def _get_element_size(dtype: Optional[str]) -> int:
    """Get element size in bytes from dtype string."""
    if not dtype:
        return DEFAULT_ELEMENT_SIZE

    try:
        return get_type_size_bytes(dtype)
    except Exception:
        normalized = str(dtype).strip().lower()
        alias_size = DTYPE_BYTE_ALIASES.get(normalized)
        if alias_size is not None:
            return alias_size

        logger.debug(
            "Unknown dtype '%s' in memory analyzer; falling back to %d-byte elements",
            dtype,
            DEFAULT_ELEMENT_SIZE,
        )
        return DEFAULT_ELEMENT_SIZE


def _tensor_bytes(dtype: Optional[str], elements: int) -> int:
    """Compute tensor byte size from element type and element count."""
    return _get_element_size(dtype) * max(int(elements), 0)


def _safe_nbytes(value: object) -> int:
    """Best-effort extraction of numpy-like nbytes."""
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, (int, float)):
        return int(nbytes)
    return 0


def _layer_tags(layer: BaseLayer) -> Set[str]:
    """
    Infer architecture tags for a layer from stable metadata.

    Tags are intentionally coarse and conservative. They are used only for
    memory heuristics, not functional behavior.
    """
    tags: Set[str] = set()
    searchable = " ".join(
        [
            layer.__class__.__name__.lower(),
            getattr(layer, "op_type", "").lower(),
            getattr(layer, "name", "").lower(),
        ]
    )

    for token, implied_tags in LAYER_TAG_RULES.items():
        if token in searchable:
            tags.update(implied_tags)

    if isinstance(layer, LINEAR_LAYER_TYPES):
        tags.add("linear")

    return tags


def _layer_has_bias(layer: BaseLayer) -> bool:
    """Robust bias detection across layer variants."""
    if hasattr(layer, "has_bias"):
        return bool(getattr(layer, "has_bias"))

    if getattr(layer, "bias", None) is not None:
        return True

    if getattr(layer, "B", None) is not None:
        return True

    if getattr(layer, "bias_name", None):
        return True

    if isinstance(layer, (GemmLayer, FusedGemmLayer)):
        return True

    if isinstance(layer, FusedLinearLayer):
        # Typical linear fusion inputs: [X, W, B]
        return len(layer.inputs) >= 3

    return False


def _estimate_input_tensor_bytes(ir: NetworkIR, tensor_name: str) -> int:
    """
    Estimate byte size of a named network input tensor.

    Uses first consumer metadata as a best-effort proxy. If unavailable, returns 0.
    """
    consumers = ir.tensor_consumers.get(tensor_name)
    if not consumers:
        return 0

    estimated = 0
    for consumer_name in consumers:
        layer = ir.get_layer(consumer_name)
        estimated = max(
            estimated,
            _tensor_bytes(
                getattr(layer, "input_type", None), getattr(layer, "input_size", 0)
            ),
        )
    return estimated


def _compute_layer_weights(layer: BaseLayer) -> Tuple[int, int]:
    """
    Compute weight and bias memory for a layer in bytes.

    Strategy:
    1) Prefer explicit in-memory tensors (weights/W/R/P/rhs_const, bias/B).
    2) Fall back to shape/type estimates for compact linear layers.
    """
    weights_bytes = 0
    biases_bytes = 0

    # Prefer explicit tensors where available (robust across new architectures).
    for weight_attr in ("weights", "W", "R", "P", "rhs_const"):
        weights_bytes += _safe_nbytes(getattr(layer, weight_attr, None))

    for bias_attr in ("bias", "B"):
        biases_bytes += _safe_nbytes(getattr(layer, bias_attr, None))

    # Fallback for compact linear layers that don't carry arrays directly.
    tags = _layer_tags(layer)
    if weights_bytes == 0 and "linear" in tags:
        weights_bytes = _tensor_bytes(
            getattr(layer, "weight_type", None) or layer.input_type,
            layer.input_size * layer.output_size,
        )

    if biases_bytes == 0 and "linear" in tags and _layer_has_bias(layer):
        biases_bytes = _tensor_bytes(
            getattr(layer, "bias_type", None) or layer.output_type,
            layer.output_size,
        )

    # Standalone Add layer used as bias (backward compatibility).
    if isinstance(layer, AddLayer) and getattr(layer, "is_bias", False):
        biases_bytes = max(
            biases_bytes,
            _tensor_bytes(layer.output_type, layer.output_size),
        )

    logger.debug(
        f"Layer {layer.name}: weights={weights_bytes}B, bias={biases_bytes}B, tags={sorted(tags)}"
    )

    return weights_bytes, biases_bytes


def _compute_constants_size(layer: BaseLayer) -> int:
    """Compute size of constants (quantization params, etc.)."""
    if isinstance(layer, QuantizeLinearLayer):
        scale_type = getattr(layer, "scale_type", "float32")
        scale_size = _get_element_size(scale_type)
        zp_size = _get_element_size(layer.output_type)
        return scale_size + zp_size

    elif isinstance(layer, DequantizeLinearLayer):
        scale_type = getattr(layer, "scale_type", "float32")
        scale_size = _get_element_size(scale_type)
        zp_size = _get_element_size(layer.input_type)
        return scale_size + zp_size

    return 0


def _estimate_activation_memory(
    ir: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> int:
    """
    Estimate activation memory needed.

    If buffer_allocations provided, use actual buffer reuse.
    Otherwise, assume separate output variable for each layer.

    Args:
        ir: Network IR
        buffer_allocations: Optional dict mapping tensor names to buffer names

    Returns:
        Activation memory in bytes
    """
    peak_runtime_bytes = _estimate_peak_runtime_memory(ir)

    if buffer_allocations:
        # Calculate actual buffer sizes from allocations
        buffer_sizes: Dict[str, int] = {}

        for layer_name in ir.execution_order:
            layer = ir.get_layer(layer_name)

            for output_tensor in layer.outputs:
                # Skip network outputs (they need their own storage)
                if output_tensor in ir.output_tensors:
                    continue

                buffer_name = buffer_allocations.get(output_tensor)
                if buffer_name:
                    tensor_bytes = _tensor_bytes(layer.output_type, layer.output_size)
                    buffer_sizes[buffer_name] = max(
                        buffer_sizes.get(buffer_name, 0), tensor_bytes
                    )

        # Also account for network inputs/outputs (not in buffer pool)
        io_bytes = 0
        for tensor_name in ir.input_tensors:
            io_bytes = max(io_bytes, _estimate_input_tensor_bytes(ir, tensor_name))

        for layer in ir.layers.values():
            for out in layer.outputs:
                if out in ir.output_tensors:
                    io_bytes = max(
                        io_bytes, _tensor_bytes(layer.output_type, layer.output_size)
                    )

        total_buffer_bytes = sum(buffer_sizes.values())
        allocated_bytes = total_buffer_bytes + io_bytes
        conservative_bytes = max(allocated_bytes, peak_runtime_bytes)

        logger.info(
            f"Activation memory (buffer allocation): {len(buffer_sizes)} buffers = {total_buffer_bytes} bytes, "
            f"I/O = {io_bytes} bytes, allocated_total = {allocated_bytes} bytes, "
            f"peak_runtime = {peak_runtime_bytes} bytes, conservative = {conservative_bytes} bytes"
        )
        return conservative_bytes
    else:
        logger.info(
            f"Activation memory (peak runtime estimate, no buffer allocation): {peak_runtime_bytes} bytes"
        )
        return peak_runtime_bytes


def _estimate_layer_workspace_bytes(layer: BaseLayer) -> int:
    """
    Estimate temporary workspace for a single layer execution.

    This is deliberately conservative and architecture-aware:
    - Baseline scratch = 1x output tensor
    - Recurrent layers reserve hidden/cell/gate scratch
    - Attention-like layers reserve Q/K/V + score/probability-style temporaries
    """
    output_bytes = _tensor_bytes(layer.output_type, layer.output_size)
    workspace = output_bytes

    tags = _layer_tags(layer)
    hidden_size = int(getattr(layer, "hidden_size", 0) or 0)
    sequence_length = int(getattr(layer, "sequence_length", 1) or 1)
    elem_size = _get_element_size(layer.output_type)
    direction = str(getattr(layer, "direction", "forward") or "forward").lower()
    num_directions = 2 if direction == "bidirectional" else 1

    if "recurrent" in tags and hidden_size > 0:
        # Hidden state, plus cell state for LSTM.
        state_buffers = hidden_size * elem_size * (2 if "lstm" in tags else 1)

        # Gate workspace: 4x hidden (LSTM) or 3x hidden (GRU) at minimum.
        gate_factor = 4 if "lstm" in tags else 3
        gate_workspace = hidden_size * gate_factor * elem_size

        # Sequence bookkeeping scratch (conservative upper-bound proxy).
        seq_scratch = (hidden_size * elem_size) * max(1, min(sequence_length, 4))

        recurrent_workspace = (
            state_buffers + gate_workspace + seq_scratch
        ) * num_directions
        workspace += recurrent_workspace

    if "attention" in tags or "attention_like" in tags:
        # Coarse conservative proxy for Q/K/V projections and attention maps.
        workspace += 3 * output_bytes

    return workspace


def _estimate_peak_runtime_memory(ir: NetworkIR) -> int:
    """
    Estimate peak live activation memory during network execution.

    Combines:
    - tensor liveness (materialize outputs, free dead intermediates), and
    - per-layer temporary workspace.
    """
    consumers_remaining: Dict[str, int] = {}
    for tensor_name, consumers in ir.tensor_consumers.items():
        consumers_remaining[tensor_name] = len(consumers)

    # Fallback if tensor_consumers is not populated.
    if not consumers_remaining:
        for layer in ir.layers.values():
            for inp in layer.inputs:
                consumers_remaining[inp] = consumers_remaining.get(inp, 0) + 1

    live_tensors: Dict[str, int] = {}

    # Keep network inputs live conservatively.
    for tensor_name in ir.input_tensors:
        bytes_est = _estimate_input_tensor_bytes(ir, tensor_name)
        if bytes_est > 0:
            live_tensors[tensor_name] = bytes_est

    live_total = sum(live_tensors.values())
    peak = live_total

    for layer_name in ir.execution_order:
        layer = ir.get_layer(layer_name)
        output_bytes = _tensor_bytes(layer.output_type, layer.output_size)

        # Materialize outputs (multi-output layers are conservatively counted per output tensor).
        for output_tensor in layer.outputs:
            existing = live_tensors.get(output_tensor, 0)
            if output_bytes > existing:
                live_tensors[output_tensor] = output_bytes
                live_total += output_bytes - existing

        # Peak includes layer workspace while outputs are live.
        peak = max(peak, live_total + _estimate_layer_workspace_bytes(layer))

        # Consume layer inputs and free dead intermediates.
        for input_tensor in layer.inputs:
            if input_tensor not in consumers_remaining:
                continue

            consumers_remaining[input_tensor] -= 1
            can_free = (
                consumers_remaining[input_tensor] <= 0
                and input_tensor not in ir.input_tensors
                and input_tensor not in ir.output_tensors
            )

            if can_free and input_tensor in live_tensors:
                live_total -= live_tensors.pop(input_tensor)

    return peak


def analyze_memory(
    ir: NetworkIR,
    memory_limit_bytes: int = 96 * 1024,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> MemoryCheckResult:
    """
    Analyze memory requirements for the network IR.

    Args:
        ir: Network IR (should be after optimization passes)
        memory_limit_bytes: Device memory limit in bytes (default: 96 KB)
        buffer_allocations: Optional buffer allocations from optimizer

    Returns:
        MemoryCheckResult with detailed breakdown
    """
    breakdown = MemoryBreakdown()
    warnings: List[str] = []
    errors: List[str] = []

    for layer in ir.layers.values():
        layer_weights, layer_biases = _compute_layer_weights(layer)
        breakdown.weights_bytes += layer_weights
        breakdown.biases_bytes += layer_biases
        breakdown.constants_bytes += _compute_constants_size(layer)

    breakdown.activations_bytes = _estimate_activation_memory(ir, buffer_allocations)

    # Validation
    total = breakdown.total_bytes
    utilization = (total / memory_limit_bytes) * 100
    fits = total <= memory_limit_bytes

    # Generate warnings
    if utilization > 80:
        warnings.append(
            f"Memory utilization is high ({utilization:.1f}%). "
            f"Consider quantization or model pruning."
        )

    if breakdown.weights_bytes > memory_limit_bytes * 0.7:
        warnings.append(
            f"Weights use {breakdown.weights_bytes/1024:.1f} KB "
            f"({breakdown.weights_bytes/memory_limit_bytes*100:.1f}% of limit). "
            f"Consider reducing model size."
        )

    # Generate errors
    if not fits:
        overflow = total - memory_limit_bytes
        errors.append(
            f"Model exceeds memory limit by {overflow/1024:.2f} KB. "
            f"Required: {total/1024:.2f} KB, Limit: {memory_limit_bytes/1024:.2f} KB"
        )

    return MemoryCheckResult(
        fits_in_memory=fits,
        breakdown=breakdown,
        limit_bytes=memory_limit_bytes,
        utilization_percent=utilization,
        warnings=warnings,
        errors=errors,
    )


def check_memory(
    ir: NetworkIR,
    memory_limit_kb: float = 96,
    fail_on_exceed: bool = True,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> MemoryCheckResult:
    """
    Check memory requirements and optionally fail if exceeded.

    Args:
        ir: Network IR (post-optimization)
        memory_limit_kb: Memory limit in KB (default: 96)
        fail_on_exceed: Raise exception if memory exceeded
        buffer_allocations: Optional buffer allocations from optimizer

    Returns:
        MemoryCheckResult

    Raises:
        MemoryError: If fail_on_exceed=True and model exceeds limit
    """
    result = analyze_memory(
        ir,
        memory_limit_bytes=int(memory_limit_kb * 1024),
        buffer_allocations=buffer_allocations,
    )

    logger.info(str(result))

    for warning in result.warnings:
        logger.warning(warning)

    if result.errors:
        for error in result.errors:
            logger.error(error)

        if fail_on_exceed:
            raise MemoryError(
                f"Model exceeds PLC memory limit. "
                f"Required: {result.breakdown.total_kb:.2f} KB, "
                f"Limit: {memory_limit_kb:.2f} KB"
            )

    return result
