---
applyTo: "src/riskquant/valuation/**/*.py,src/riskquant/simlab/**/*.py,src/riskquant/credit/**/*.py,src/riskquant/sensitivity_pnl/**/*.py,tests/valuation/**/*.py,tests/simlab/**/*.py,tests/credit/**/*.py,tests/sensitivity_pnl/**/*.py"
---
# Numerical Core Standards

## Representation

Use compact primitive arrays and Structure-of-Arrays representations. Strings, dataframes, Pydantic models, and object graphs stop at the boundary.

Default core assumptions are explicit shape, dtype, order, and ownership contracts. Prefer C-contiguous arrays where the kernel contract expects them.

## Scenario architecture

Resolve scenario-independent work before the scenario loop:

- mapping and identity resolution;
- product/model dispatch;
- interpolation state;
- static flags;
- gathers/permutations;
- dimensions and domain validation.

Scenario execution must contain scenario-dependent numerical work only.

## Kernel rules

Kernels are numerical and policy-free. Do not perform catalogue lookups, product selection, logging, string handling, or configuration interpretation in kernels.

Avoid Python containers and hidden allocations. Preallocate when complete overwrite is guaranteed.

## Numba

Use the repository canonical Numba decorators. `parallel=True` requires both algorithmic justification and benchmark evidence. Fastmath must remain constrained by the global constitution.

## Performance

Material changes require benchmark evidence and numerical equivalence evidence. Do not replace a clear implementation with a lower-level variant merely because it appears faster in theory.
