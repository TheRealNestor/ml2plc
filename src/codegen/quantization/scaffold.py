"""Minimal quantization scaffolding (post-training quantization helpers).

This module provides a lightweight, opt-in PTQ flow used by the compiler when
`--auto-quant` is requested. It's intentionally small: it computes simple
per-tensor symmetric scales for weight arrays and annotates IR layers with
`weight_scale` and `weight_zero_point` metadata so downstream passes and
codegen can reason about quantized weights.

The goal is scaffolding: provide deterministic, testable metadata and not a
full production PTQ pipeline. Users can replace this with more advanced
calibration later.
"""

from typing import Optional
import numpy as np
import logging

from ..types import ModelIR, RegionKind, NetworkIR

logger = logging.getLogger(__name__)


def _compute_scale_and_zp(weights: np.ndarray) -> tuple[float, int]:
    """Compute a per-tensor scale and zero point for uint8 quantization.

    Uses a simple unsigned symmetric mapping to [0,255]. If weights are constant
    (max==min) returns a scale of 1.0 and zero point 0 to avoid division by zero.
    """
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))

    if w_max == w_min:
        return 1.0, 0

    scale = (w_max - w_min) / 255.0
    if scale == 0.0:
        scale = 1.0

    # Choose zero point so that w_min maps near 0
    zp = int(round(-w_min / scale))
    zp = max(0, min(255, zp))
    return scale, zp


def apply_post_training_quantization(model_ir: ModelIR, calib_dataset: Optional[object] = None) -> ModelIR:
    """Annotate ModelIR layers with quantization metadata (per-tensor weights).

    This function mutates the provided `model_ir` in-place and returns it for convenience.
    For each region, it inspects layer objects and, when `weights` are present,
    computes `weight_scale` and `weight_zero_point` and writes them using
    `object.__setattr__` (dataclasses are frozen).
    """
    logger.info("Applying lightweight post-training quantization scaffold")

    for region in model_ir.regions:
        # Only operate on acyclic regions (safe by default)
        if region.kind != RegionKind.ACYCLIC:
            continue

        graph: NetworkIR = region.graph
        for layer in list(graph.layers.values()):
            # Only annotate layers that have a `weights` attribute
            if hasattr(layer, "weights") and getattr(layer, "weights") is not None:
                w = getattr(layer, "weights")
                try:
                    scale, zp = _compute_scale_and_zp(np.asarray(w))
                except Exception:
                    # Fallback to safe defaults
                    scale, zp = 1.0, 0

                # Write metadata onto frozen dataclass
                try:
                    object.__setattr__(layer, "weight_scale", np.array(scale))
                    object.__setattr__(layer, "weight_zero_point", np.array(zp))
                    logger.debug(f"Annotated {layer.name} with scale={scale}, zp={zp}")
                except Exception:
                    logger.debug(f"Failed to annotate quant metadata for {layer.name}")

    return model_ir

