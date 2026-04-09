# IR-to-ST Code Generation Refactoring Plan

## Current State Analysis

### File Breakdown & Issues

| File | Lines | Issues |
|------|-------|--------|
| `generator.py` | 1608 | **MASSIVE** - Contains 40+ layer generators + 10+ helper functions. Monolithic and hard to navigate. Duplicate code. |
| `layer_generators.py` | 317 | Registry pattern exists but not fully utilized. Many generators still in generator.py. |
| `st_code.py` | 241 | Good structure but could be more concise. Has both STCode class AND helper functions. |
| `lowerers.py` | 159 | Good use of strategy pattern. Clean. |
| `type_conversion.py` | 88 | Functional but basic. Could use more abstractions. |
| **Helpers** | 634 | Well-organized by concern. Could consolidate further. |
| **Total** | 4132 lines | Sprawling, hard to maintain. |

### Key Problems

1. **Generator Monolith** (`generator.py` @ 1608 lines)
   - 40+ layer-specific functions mixed with orchestration logic
   - Utility functions (`is_uniform_array`, etc.) duplicated across files
   - Layer registration incomplete - many generators not in registry
   - Main orchestration buried among implementation details

2. **Incomplete Registry Usage**
   - `layer_generators.py` has infrastructure but only partial registration
   - Generators scattered across multiple files instead of unified
   - Makes adding new layers and extensions hard

3. **Repeated Patterns**
   - Loop generation logic repeated (matrix mult, spatial loops, etc.)
   - Builder pattern usage inconsistent
   - Indentation/nesting logic scattered

4. **Poor Separation of Concerns**
   - Section generation mixed with layer logic
   - No clear boundaries between orchestration and implementation
   - Helpers scattered across multiple utils files

5. **Maintainability Issues**
   - New developers can't easily find where layer X is handled
   - Adding support for new layer types requires understanding multiple files
   - Testing individual layer generators is difficult without refactoring
   - Code duplication makes fixes/updates dangerous

## Refactoring Goals

✅ **Reduce total lines** from 4132 → ~2500 (40% reduction)
✅ **Improve readability** with clear module boundaries
✅ **Better maintainability** through consistent patterns
✅ **Extensibility** - easy to add new layer types
✅ **Testability** - isolated, pure functions where possible

## Refactoring Strategy

### Phase 1: Consolidate Layer Generators
**Target:** Move ALL layer generators to `layer_generators.py` via registry
**Benefits:** Single source of truth, easier to navigate

**Changes:**
- Extract 40+ layer generation functions from `generator.py`
- Create proper generator stubs in `layer_generators.py`
- Complete the registry with ALL layer types
- Use lazy imports to avoid circular dependencies

### Phase 2: Simplify Orchestration
**Target:** Streamline `generator.py` to focus on high-level flow
**Benefits:** Clear separation of "what to generate" vs "how to generate"

**Changes:**
- Keep only: entry points, section assembly, variable collection, orchestration
- Remove layer-specific logic (moved to registry)
- Consolidate variable/constant section generation
- Streamline helper functions

### Phase 3: Unify Utilities
**Target:** Consolidate `utils/` folder while maintaining clarity
**Benefits:** Fewer files, clearer organization, easier to find helpers

**Changes:**
- Keep separate files by *concern* (activation, copy, constant, loop, array)
- Remove redundant utilities
- Create base patterns/templates to reduce duplication
- Document each utility's purpose and usage

### Phase 4: Clean ST Code Builders
**Target:** Streamline `st_code.py` and builder usage
**Benefits:** More concise code generation, less boilerplate

**Changes:**
- Keep STCode and STCodeBuilder (they're good)
- Move repetitive ST constructs to helper factories
- Create builder extensions for common patterns (nested loops, gates, etc.)
- Simplify indentation logic

## Concrete Improvements

### 1. Layer Generator Registry (Completeness)

Before:
```python
# generator.py - scattered across 1600 lines
def generate_linear_layer_code(layer, input_var, output_var): ...
def generate_conv2d_code(layer, input_var, output_var): ...
def generate_lstm_code(layer, input_var, output_var): ...
# ... 37 more scattered functions
```

After:
```python
# layer_generators.py - unified, complete registry
registry = LayerCodeGeneratorRegistry()
registry.register(LinearLayer, _generate_linear)
registry.register(Conv2DLayer, _generate_conv2d)
registry.register(LSTMLayer, _generate_lstm)
# ... all 40+ types registered, discoverable in one place
```

### 2. Reduce Generator.py Size

**Before:** 1608 lines (40+ functions)
**After:** 300-400 lines (only orchestration)

Keep in `generator.py`:
- Entry points: `translate_ir_to_st`, `translate_model_to_st`
- Section builders: constants, variables, I/O
- Forward pass orchestration
- Variable/buffer collection

Remove (move to registry):
- All 40+ layer-specific generators
- Duplicated utilities

### 3. Better Builder Patterns

**Before:**
```python
def generate_linear_layer_code(layer, input_var, output_var):
    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: ... *)")
    builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
    with builder.indent():
        builder.add_line("sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
        with builder.indent():
            # ... 20 more lines
```

**After:**
```python
def _generate_linear(layer, input_var, output_var):
    builder = ForLoopBuilder(input_var, layer.input_size)
    builder.add_dot_product(input_var, layer)
    builder.apply_bias_and_activation(layer)
    return builder.build()
```

### 4. Extract Common Patterns

Create reusable templates:
- `ForLoopBuilder` - nested loops with common patterns
- `MatmulPattern` - weight access + accumulation
- `GatePattern` - LSTM/GRU gate computation
- `SpatialLoopPattern` - Conv2D/Pool2D nested loops

## Implementation Order

1. **Create new `layer_generators.py` structure** (move registry to top)
2. **Extract layer generators to registry** (group by pattern type)
3. **Prune `generator.py`** (remove layer logic, keep orchestration)
4. **Consolidate utils** (keep organized, remove duplication)
5. **Create builder extensions** (for common patterns)
6. **Update imports** (ensure registry is used)
7. **Test & validate** (all layer types still generate correctly)

## Expected Outcomes

- **Better organization:** Clear "what generates what"
- **Easier to extend:** Add new layer → register generator
- **Faster to understand:** Specific layers in isolated functions
- **Less duplication:** Patterns extracted to templates
- **Smaller codebase:** ~40% reduction without losing functionality
