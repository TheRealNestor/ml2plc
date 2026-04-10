"""
IR-to-Structured Text code generation.

High-level entry points for translating network IR to PLC Structured Text.

Modules:
  codegen_core: Main orchestration and FB structure generation
  layer_generators: Centralized generator registry
  layers/: Layer-specific code generation implementations
  lowerers: Region-aware code generation strategies
  st_code: ST code primitives and builder
  st_templates: ST code templates (FB headers, program wrappers, configurations)
"""

from .codegen_core import translate_ir_to_st, translate_model_to_st

__all__ = ["translate_ir_to_st", "translate_model_to_st"]
