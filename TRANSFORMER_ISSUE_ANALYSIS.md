# Architecture-First Compiler Refactor Status

**Date**: April 12, 2026  
**Priority**: Foundational compiler architecture > model-specific Transformer fixes  
**Current Direction**: Keep the codebase slim, readable, decoupled, and scalable with explicit compiler passes and minimal technical debt.

---

## Executive Summary

The project direction is now explicitly **architectural** rather than patch-oriented.

The most important recent work (shape-inference overhaul + workspace-wide consistency sweeps) is a strong example of the target design style:

- clear module boundaries,
- low coupling,
- small focused files,
- predictable extension points,
- fail-fast diagnostics.

The immediate goal is to apply that same approach across the compiler workspace so we can prevent future regressions and reduce design debt early.

---

## Design Intent (What We’re Optimizing For)

We are intentionally optimizing for:

1. **Slim code paths**
   - Keep each unit small and understandable.
   - Avoid large god-files and tangled condition trees.

2. **Readability first**
   - Make control flow obvious.
   - Prefer explicit semantics over implicit fallback behavior.

3. **Extensibility / scalability**
   - New operators and passes should be additive.
   - Minimal changes required in existing stable modules.

4. **Separation of concerns (compiler pass discipline)**
   - Distinct decoupled passes with explicit responsibilities.
   - No hidden cross-pass side effects.

5. **Early failure + high-quality diagnostics**
   - Invalid states fail where they originate.
   - Errors include context and lineage where possible.

6. **Low technical debt by design**
   - Remove duplication as we see it.
   - Avoid legacy compatibility layers unless strictly required.

---

## Current State (Recent Changes + Validation)

### What has been improved

- Shape inference has been reworked into a dedicated module/folder architecture.
- Shape logic has been split into focused components (public API, engine orchestration, primitive math, registry/dispatch, validation).
- Runtime matmul contract handling is centralized and validated earlier.
- Better regression coverage exists around shape strictness and runtime matmul behavior.
- Graph analysis now follows a cleaner single-source approach (`LayerGraph`) with reduced duplicate edge-construction logic.
- Transitional/deprecation-oriented noise has been removed from core graph APIs to keep pre-release design intent explicit.
- Removed-shim modules and shape API messaging are now more consistent across the workspace.
- The code now reflects a cleaner foundation for future compiler passes.

### Validation status

Latest full-suite verification result:

- **113 passed, 2 skipped**
- **0 warnings**

This confirms the current refactor state is stable and test-backed.

---

## Updated Position on Transformer-Specific Failures

Transformer compilation issues are still useful as stress tests, but they are now treated as **symptoms of architecture pressure**, not just one-off model bugs.

So the strategy is:

- continue using Transformer paths as regression fixtures,
- but prioritize structural compiler improvements first,
- then address remaining model-path issues within the improved architecture.

---

## Why Architecture Refactor Is More Pressing Right Now

Model-specific fixes without foundational cleanup tend to create:

- duplicated logic,
- fragile branching,
- hidden coupling between extraction/inference/lowering,
- recurring regressions in adjacent operators.

A pass-oriented, modular design reduces all of these and keeps future feature work fast.

---

## Compiler Structure Target (Proposed Canonical Layout)

We should converge toward explicit pass folders (names can be adapted):

- `frontend/` (model loading, normalization, graph preparation)
- `analysis/` (shape/role/provenance, state detection, graph metadata)
- `lowering/` (pattern rewrites like einsum lowering)
- `ir_build/` (extractors + IR construction)
- `optimization/` (IR transformations)
- `backend/` (ST generation + backend-specific validations)
- `validation/` (cross-pass invariants + diagnostics)

With this, each pass has:

- defined input/output contract,
- no overlapping responsibility,
- explicit hand-off artifacts.

---

## Immediate Refactor Priorities (Workspace-Wide)

1. **Codify pass contracts**
   - Document input/output invariants per pass.
   - Enforce with lightweight assertions and targeted tests.

2. **Remove duplication around shape and operator semantics**
   - Single source of truth for shape role/shape math rules.
   - No duplicate compatibility shims.
   - No duplicate graph-construction paths for the same analysis concept.

3. **Modularize extractor responsibilities**
   - Keep extraction pure and context-driven.
   - Push shared validation into centralized contract modules.

4. **Standardize diagnostics**
   - Consistent error format with node/layer context and lineage when available.
   - Keep error/warning policy intentional (do not rely on migration-era warning noise during pre-release).

5. **Keep files intentionally small**
   - Split on responsibility boundaries, not arbitrary size.
   - Avoid over-abstraction and unnecessary OOP layers.

6. **Strengthen regression coverage by pass**
   - A few focused tests per pass contract.
   - Keep full-suite gate as final quality check.

---

## What “Done Right” Looks Like

A healthy compiler codebase here should let us:

- add support for a new operator with a small, local change,
- understand failures from one error message,
- run clear pass-by-pass debugging,
- avoid introducing architecture debt during feature work.

If a change increases complexity without improving pass boundaries, readability, or contract clarity, it is not aligned with this direction.

---

## Next Action Plan

### Short-term (next cycle)

- Continue workspace-level modular cleanup (not only Transformer path).
- Remove remaining duplicated or shim-style paths where safe.
- Convert remaining wildcard imports to explicit imports in core modules for readability and API clarity.
- Add pass-level invariant tests where gaps still exist.

### Mid-term

- Introduce/finish canonical pass directory organization.
- Keep model-specific stress fixtures (Transformer/RNN/etc.) to verify architecture robustness.

### Current baseline to preserve

- Full suite remains green after each non-trivial structural sweep.
- New refactors are accepted only when they reduce coupling/duplication or improve pass-boundary clarity.
- Regression tests should remain stable and not require churn for internal refactors.

### Ongoing quality gate

- Preserve full-suite green baseline after each non-trivial structural change.

---

## Final Note

The recent shape refactor demonstrates the right trajectory.

Going forward, the standard is:

- **fundamental design first**,
- **slim and readable implementations**,
- **clear pass separation**,
- **fix issues early to prevent debt accumulation**.
