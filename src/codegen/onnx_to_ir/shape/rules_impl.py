"""Removed shim module.

Use `codegen.onnx_to_ir.shape.rules` (or `codegen.onnx_to_ir.shape`) directly.
"""

raise ModuleNotFoundError(
    "`codegen.onnx_to_ir.shape.rules_impl` has been removed. "
    "Import from `codegen.onnx_to_ir.shape.rules` instead."
)
