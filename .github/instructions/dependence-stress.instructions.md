---
applyTo:
  "src/riskquant/dependence_stress/**/*.py,tests/dependence_stress/**/*.py,scripts/**/dependence/**/*.py"
---
# Dependence Stress — Normative Methodology Contract

This file is normative for the `dependence_stress` package and its tests/demos. It must agree with any approved methodology artefact under `.github/methodology/`.

## 1. Objective

Stress cross-sectional dependence of an empirical observation matrix while preserving each row's empirical marginal distribution exactly.

For `X ∈ R^(K×S)`:

`X_stress[k, s] = X[k, π[k, s]]`

where every row `π[k, :]` is a permutation of `0, …, S-1`.

## 2. Result contract

The authoritative result contains exactly:

- `permutations`: `K×S` integer permutation matrix;
- `portfolio_pnl`: `1×S` or equivalent vector of stressed portfolio aggregate PnL.

Do not add a `stressed` matrix field unless the public contract is explicitly changed.

## 3. Exact invariants

For every row `k`:

`sort(X_stress[k, :]) == sort(X[k, :])`.

Also required:

- every permutation row contains each index `0…S-1` exactly once;
- reconstruction through the returned permutations is exact;
- the portfolio vector equals aggregation applied after scenario realignment;
- input data ownership is preserved unless copying is explicitly part of the contract.

These are `EXACT` properties.

## 4. Correlation semantics

Keep distinct:

- Pearson correlation of observations;
- Spearman correlation of ranks;
- latent copula correlation;
- empirical dependence/tail metrics.

Never use one label to mean another.

## 5. Ranks and ties

Tie handling must be explicit. Rank transforms must state whether average/mid-rank or another convention is used.

Copula transformations must document the relationship between empirical ranks/uniforms and latent scores.

## 6. Tail dependence

Lower-tail dependence diagnostics must state:

- uniform construction;
- tie convention;
- threshold definition;
- finite-sample estimator;
- behaviour when threshold counts are zero.

Do not silently equate empirical tail co-occurrence with an asymptotic copula coefficient.

## 7. Copula stress

`GaussianCopula` and `TCopula` operate through latent dependence and rank-induced permutation construction. Copula generation must document correlation semantics, degrees of freedom (for t-copula), finite-sample behaviour, and RNG ownership.

## 8. RNG contract

RNG ownership must be explicit. Malformed RNG inputs must fail with a deterministic type/contract error. No hidden global RNGs.

Reproducibility includes seed/state semantics and, where relevant, random-number consumption order.

## 9. Statistical claims

Claims about achieving a target dependence structure through finite permutations are generally `STATISTICAL`, not exact. Do not describe finite-sample empirical alignment as exact copula calibration unless the methodology proves that property.

## 10. Failure and validation

Validate observation matrix shape, finiteness/domain requirements, permutation shape and dtype, probability bounds, and correlation matrices before the numerical path.

Strict positive-definiteness is required where the model mathematically requires a non-degenerate latent covariance/correlation matrix. Canonicalise symmetric correlation inputs deterministically.
