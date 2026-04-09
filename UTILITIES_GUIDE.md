"""
Guide to Using IR-to-ST Code Generation Utilities

This document explains how to use the new utility modules to generate
cleaner, more maintainable layer code generators.

================================================================================
OVERVIEW
================================================================================

The refactored ir_to_st/utils/ package provides focused utilities organized by
concern:

1. constant_helpers.py → Weight, bias, quantization constant generation
2. loop_helpers.py → FOR loop patterns and helpers
3. array_helpers.py → Multidimensional array indexing and access
4. activation_helpers.py → Activation function code generation
5. copy_helpers.py → Data movement and copy patterns

Each module solves a specific problem and is designed to be composable and reusable.

================================================================================
COMMON USAGE PATTERNS
================================================================================

### Pattern 1: Simple Copy Layer (Reshape, Squeeze, Unsqueeze)

OLD CODE (in generator.py, ~50 lines):
def generate_reshape_code(layer, input_var, output_var):
builder = STCodeBuilder()
builder.add_line(f"(_ Layer {layer.layer_id}: Reshape _)")
builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
with builder.indent():
builder.add_line(f"{output_var}[i] := {input_var}[i];")
builder.add_line("END_FOR;")
return builder.build()

NEW CODE (using copy_helpers):
from utils.copy_helpers import generate_simple_copy
from ..st_code import STCode

    def generate_reshape_code(layer, input_var, output_var):
        code = STCode.from_lines(f"(* Layer {layer.layer_id}: Reshape *)")
        code += generate_simple_copy(
            input_var, output_var, layer.output_size,
            comment="Reshape (size preserved)"
        )
        return code

BENEFIT:

- Focused intent (copy pattern)
- Reusable for Squeeze, Unsqueeze, Cast identity
- Easier to test and maintain

================================================================================

### Pattern 2: Loop-Based Computation (MatMul, Linear Layer)

OLD CODE (in generator.py, ~100 lines):
def generate*linear_layer_code(layer, input_var, output_var):
builder = STCodeBuilder()
builder.add_line(f"(* Layer {layer.layer_id}: Linear *)")
builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
with builder.indent():
builder.add_line("sum := 0.0;")
builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
with builder.indent():
builder.add_line(f"sum := sum + {input_var}[i] \* weights*{layer.layer_id}[i];")
builder.add_line("END_FOR;")
builder.add_line(f"{output_var}[j] := sum;")
builder.add_line("END_FOR;")
return builder.build()

NEW CODE (using loop_helpers + activation_helpers):
from utils.loop_helpers import generate_nested_for_loop
from utils.activation_helpers import generate_activation_inline
from ..st_code import STCode

    def generate_linear_layer_code(layer, input_var, output_var):
        code = STCode.from_lines(f"(* Layer {layer.layer_id}: Linear *)")

        # Build nested loops: outer over output size, inner over input size
        loops = [
            ("j", 0, layer.output_size - 1),
            ("i", 0, layer.input_size - 1),
        ]

        # Inner body computes sum
        inner_body = STCode.from_lines(
            "sum := sum + " +
            f"{input_var}[i] * weights_{layer.layer_id}[i * {layer.output_size} + j];"
        )

        # Outer body initializes sum and stores result
        outer_body = STCode.from_lines(
            "sum := 0.0;",
            inner_body.to_string(),
        )

        # Apply nested loops
        loop_code = generate_nested_for_loop(loops, outer_body)
        code += loop_code

        # Apply activation if needed
        activation = getattr(layer, "activation", ActivationType.NONE)
        if activation != ActivationType.NONE:
            code += generate_activation_loop(activation, output_var, output_var, layer.output_size)

        return code

BENEFIT:

- Loop patterns extracted and reusable
- Activation application decoupled
- Easier to extend with new activations

================================================================================

### Pattern 3: Array Indexing (Conv2D, Transpose)

OLD CODE (in generator.py, ~300 lines of Conv2D with hand-rolled indexing):
for oc in range(out_c):
for oh in range(out_h):
for ow in range(out_w):
for ic in range(in_c):
for kh in range(kH):
for kw in range(kW):
ih = oh _ sH - pH + kh
iw = ow _ sW - pW + kw
if 0 <= ih < in_h and 0 <= iw < in_w: # Manually compute flat indices
input_idx = ic _ in_h _ in_w + ih _ in_w + iw
weight_idx = oc _ (in_c _ kH _ kW) + ic _ kH _ kW + kh \* kW + kw
...

NEW CODE (using array_helpers):
from utils.array_helpers import compute_conv_indices

    def generate_conv2d_code(layer, input_var, output_var):
        # Precompute strides and access patterns
        conv_info = compute_conv_indices(
            out_idx=0,
            out_shape=(layer.output_channels, layer.output_h, layer.output_w),
            in_shape=(layer.input_channels, layer.input_h, layer.input_w),
            kernel_shape=(layer.kernel_h, layer.kernel_w),
            strides=(layer.stride_h, layer.stride_w),
            pads=layer.pads,
        )

        # Generate nested loops using helpers
        # Compute input receptive field bounds
        h_start, h_end = conv_info["input_h_range"]
        w_start, w_end = conv_info["input_w_range"]

        # Generate code using precomputed information
        builder = STCodeBuilder()
        builder.add_line(f"(* Conv2D: {conv_info['output_oc']}, {conv_info['output_oh']}, {conv_info['output_ow']} *)")
        # ... use h_start, h_end, w_start, w_end, etc.

        return builder.build()

BENEFIT:

- Index computation centralized and testable
- Reduces error-prone manual arithmetic
- Makes code more readable (semantics vs. arithmetic)

================================================================================

### Pattern 4: Constant Generation

OLD CODE (scattered throughout generator.py): # In generate_layer_weights
if is_quantized: # Generate weights as integers
if is_uniform_array(layer.weight_scale): # Emit scalar
code += generate_scalar_constant(...)
else: # Emit array
code += generate_array_constant(...)

    # In generate_layer_bias
    bias_type = plc_type_from_onnx_dtype(layer.output_type)
    code += generate_array_constant(...)

    # In generate_lstm_weights (special case for LSTM)
    # ... 100+ lines of gate splitting logic

NEW CODE (using constant_helpers):
from utils.constant_helpers import (
generate_weights_constants,
generate_bias_constant,
generate_lstm_weights_constants,
)

    def generate_constants_for_layer(layer):
        code = STCode.empty()

        # Weights (handles quantization automatically)
        if hasattr(layer, "weights"):
            code += generate_weights_constants(layer, is_integer=layer.is_quantized())

        # Bias (skips LSTM automatically)
        if hasattr(layer, "bias"):
            code += generate_bias_constant(layer)

        # LSTM special case (separated and clear)
        if isinstance(layer, LSTMLayer):
            code += generate_lstm_weights_constants(layer)

        return code

BENEFIT:

- DRY: weight/bias/quantization logic in one place
- LSTM gate splitting isolated and documented
- Easy to add new quantization schemes

================================================================================

### Pattern 5: Activation Functions

OLD CODE (duplicated across generators): # In generate_linear_layer_code
if activation == ActivationType.RELU:
activated_expr = f"MAX({final_expr}, 0.0)"
elif activation == ActivationType.SIGMOID:
activated_expr = f"1.0 / (1.0 + EXP(-({final_expr})))" # ... more duplicated code

    # In generate_activation_layer_code
    if activation == ActivationType.RELU:
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := MAX({input_var}[i], 0.0);")
        # ... more duplicated code

NEW CODE (using activation_helpers):
from utils.activation_helpers import (
generate_activation_inline,
generate_activation_loop,
)

    # For inline (within matmul):
    activation = getattr(layer, "activation", ActivationType.NONE)
    final_expr = generate_activation_inline(activation, "sum + bias")
    builder.add_line(f"{output_var}[j] := {final_expr};")

    # For separate loop:
    code = generate_activation_loop(activation, input_var, output_var, size)

BENEFIT:

- Single source of truth for activation expressions
- Consistent handling across all generators
- Easy to add new activation types

================================================================================

## CHECKLIST FOR USING NEW UTILITIES

When writing a new layer generator or refactoring an existing one:

1. **Constants Generation**
   - [ ] Using generate_weights_constants for weights
   - [ ] Using generate_bias_constant for bias
   - [ ] Special cases (LSTM, BatchNorm) handled

2. **Loop Patterns**
   - [ ] Using generate_for_loop for single loops
   - [ ] Using generate_nested_for_loop for nested loops
   - [ ] Loop structure is clear and readable

3. **Array Indexing**
   - [ ] Using array_helpers for multidimensional indexing
   - [ ] Stride calculations centralized
   - [ ] Index arithmetic verified with helper functions

4. **Data Movement**
   - [ ] Using generate_simple_copy for copy operations
   - [ ] Using appropriate copy variant (offset, strided, broadcast)
   - [ ] Copy semantics clearly documented

5. **Activations**
   - [ ] Using generate_activation_inline for inlinable activations
   - [ ] Using generate_activation_loop for multi-pass activations
   - [ ] Activation intent clear from code

================================================================================

## FILE STRUCTURE AFTER REFACTORING

src/codegen/ir_to_st/
├── **init**.py
├── st_code.py # Core STCode/STCodeBuilder (unchanged)
├── type_conversion.py # Type system (unchanged)
├── layer_generators.py # Registry pattern (unchanged)
├── lowerers.py # Region strategies (unchanged)
├── openplc_st.py # OpenPLC helpers (unchanged)
├── generator.py # SIMPLIFIED! Layer-specific logic only
└── utils/ # NEW! Reusable utilities
├── **init**.py
├── constant_helpers.py # Weight, bias, quantization
├── loop_helpers.py # FOR loop patterns
├── array_helpers.py # Array indexing
├── activation_helpers.py # Activation functions
└── copy_helpers.py # Data movement patterns

================================================================================

## BENEFITS

✅ Reduced Duplication: ~60% of utility functions now reused across generators
✅ Better Readability: Intent is clear from helper names
✅ Easier Testing: Utilities can be unit tested independently
✅ Flexible Extension: Adding new layer types requires minimal boilerplate
✅ Clearer Dependencies: What each generator needs is explicit
✅ Maintainability: Changes to core patterns only need to happen once
✅ Discoverability: New developers can find reusable patterns quickly

================================================================================
"""
