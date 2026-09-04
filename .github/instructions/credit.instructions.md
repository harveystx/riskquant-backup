---
applyTo:
	"src/riskquant/credit/**/*.py,tests/credit/**/*.py"
---
# Credit Risk Standards

Credit implementations must make the economic and statistical contract explicit before optimisation.

## Required decisions

Where applicable, state:

- exposure definition;
- EAD semantics;
- default event/timing;
- recovery model;
- discounting/timing;
- dependence/correlation model;
- calibration method;
- aggregation semantics;
- horizon;
- default/recovery edge cases.

## Numerical core

The JIT boundary contains canonical primitive state only. Model selection, calibration policy, data validation, and dispatch remain outside kernels.

## Validation

Use independent analytical or reference checks, limiting cases, monotonicity/positivity invariants where mathematically applicable, and reproducibility checks.
