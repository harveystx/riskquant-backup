---
applyTo: "src/riskquant/**/*var*.py,src/riskquant/**/*es*.py,src/riskquant/**/*quantile*.py,src/riskquant/**/*risk_measure*.py,tests/**/*var*.py,tests/**/*es*.py,tests/**/*quantile*.py,tests/**/*risk_measure*.py"
---
# Tail-Risk Methodology Standards

Tail-risk implementations must never inherit statistical semantics accidentally from library defaults.

## Required contract

Every material quantile/VaR/ES implementation must explicitly define:

- P&L versus loss convention;
- upper versus lower tail;
- quantile probability and endpoint semantics;
- finite-sample convention;
- estimator/interpolation method;
- treatment of duplicate observations;
- NaN/inf policy;
- weighting, if applicable;
- horizon;
- units and sign of the returned measure.

## Quantiles

NumPy/SciPy defaults are not methodology. The selected quantile estimator must be an explicit contract.

## Expected Shortfall

ES must be mathematically defined. “Average beyond VaR” is not a sufficient specification because finite-sample and endpoint semantics matter.

## Numerical integrity

Handle small samples, boundary probabilities, repeated values, constant samples, and floating-point edge cases explicitly.

## Attribution

Any VaR/ES attribution method must define the decomposition identity, residual/interaction terms, and sign convention. Do not imply additivity when the methodology does not guarantee it.
