---
applyTo: "src/riskquant/observations/**/*.py,tests/observations/**/*.py"
---
# Observations Methodology and Boundary Standards

Observations are a mathematical input contract, not a generic time-series container.

## Required semantics

Every observation pipeline must make explicit:

- time ordering;
- sample membership;
- frequency;
- missingness semantics;
- return/change convention;
- horizon;
- shape and dtype;
- ownership/copy semantics.

Return conventions must be exact where applicable:

- log return: `ln(P_t / P_{t-1})`;
- simple return: `(P_t / P_{t-1}) - 1`;
- difference: `P_t - P_{t-1}`.

## Integrity

Inputs must not be silently dropped, re-sorted, filled, interpolated, or replaced unless explicitly authorised by the governing methodology.

Chronological ordering is a contract. Duplicate timestamps and irregular frequency must have explicit semantics.

## Boundary

Convert rich external representations into canonical arrays once. Keep strings and dataframe objects out of numerical execution.
