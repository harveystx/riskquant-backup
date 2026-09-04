---
applyTo: "src/riskquant/sensitivity_pnl/**/*.py,tests/sensitivity_pnl/**/*.py"
---
# Sensitivity-PnL Standards

Sensitivity PnL must implement the approved Taylor-expansion methodology exactly as specified by the methodology artefact.

## Contract

Document and test:

- Taylor order;
- included cross terms;
- derivative definitions/signs;
- units/scaling;
- base-state and shocked-state semantics;
- treatment of FX and other translation factors;
- numerical approximation status.

## Architecture

Canonicalise inputs and construct scenario-independent sensitivity state before the scenario loop. Numerical scenario evaluation belongs in compact CORE kernels.

## Dependence integration

Dependence stress acts on scenario alignment/indexing. It must not silently alter the row marginals or the underlying sensitivity model.

## Tests

Use independent analytical checks for first- and second-order terms where feasible. Label approximations explicitly and retain regression masters for approved scenarios.
