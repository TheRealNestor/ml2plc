"""Removed shim module.

Use `codegen.onnx_to_ir.shape` as the only supported shape API surface.
"""

raise ModuleNotFoundError(
    "`codegen.onnx_to_ir.shape_validation` has been removed. "
    "Import from `codegen.onnx_to_ir.shape` instead."
)
