"""
Optimization result data structure.

Separated into its own module to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from ..types import NetworkIR


@dataclass
class OptimizationResult:
    """Result of IR optimization, including optional code generation hints."""

    ir: NetworkIR
    buffer_allocations: Optional[Dict[str, str]] = None  # tensor_name -> buffer_name

    def has_buffer_allocations(self) -> bool:
        """Check if buffer allocations are available."""
        return self.buffer_allocations is not None
