# `riskquant.credit` — Stable Public API

This is the frozen public surface of the credit modelling core. Everything listed
here is supported and versioned: **changing a public name or signature is a
breaking change.** Everything not listed is internal — it may move, rename, or
change shape at any time without notice.

## Policy

- **Public** = listed below = enforced by that module's `__all__`. Stable
  signatures.
- **Internal** = everything else (no `__all__` membership). No stability promise.
- **Import convention.** `__init__.py` are empty by constitution; there is no
  re-export hub. Import public names from their owning module by full path
  (`from riskquant.credit.jtd.engine import run_jtd`). `__all__` on each public
  module is the machine-checkable record of that module's surface. This keeps the
  Numba-free ingest modules free of the kernel import tax.

## Public surface

### Masters — ingest (edge; pandas in)
- `masters.obligors.ingest.compile_obligor_master`
- `masters.netting_set.ingest.compile_netting_set_master`
- `masters.factor_universe.ingest.compile_credit_factor_universe`

### Masters — compiled values (SoA; inspectable)
- `masters.obligors.compiled.CompiledObligorCredit`
- `masters.netting_set.compiled.CompiledNettingSetMaster`
- `masters.factor_universe.compiled.CompiledCreditFactorUniverse`
- `masters.cross_master.compiled.CompiledCreditMasters`

### Factor simulation
- `factors.simulate.base.build_credit_factor_matrix`
- `factors.simulate.mixed.MixedCreditFactorGenerator` (`.from_universe`)
- `factors.state.CreditFactorMatrix` (the latent-factor value `run_jtd` consumes)

### JTD ingest + execution
- `jtd.ingest.bundle.JtdIngestBundle` (incl. `from_instrument_frame`)
- `jtd.ingest.compile.compile_jtd_portfolio`
- `jtd.compiled.CompiledJtdPortfolio`
- `jtd.engine.run_jtd`
- `jtd.results.JtdLossResult`

### Aggregation / risk metrics
- `aggregate.summary`: `build_portfolio_loss_distribution`, `summarize_jtd_loss`,
  `JtdRiskSummary`, `model_tail_var_es`
- `aggregate.loss_metrics`: `expected_loss`, `loss_var_es`, `value_at_risk`,
  `expected_shortfall`, `tail_var_es_by_group`, `loss_quantiles`,
  `PortfolioLossDistribution`

### Taxonomy (stable integer enums)
- `jtd.models.JtdModel`
- `masters.eums.credit_factor_kind.CreditFactorKind`

## Model parameter contracts

### Beta-binomial (D-C18)

`BETA_BINOMIAL` marginal default probability is `alpha / (alpha + beta)`. The
obligor master's PD column is **not** a model parameter: the ingest contract for
this model requires `uid`, `uniform_factor` and `beta_factor` only, and the
builder joins the master for LGD alone. A master PD that disagrees with the Beta
mean is ignored by design, not rejected.

A pool is the BETA_PD factor an obligor names, nothing more. The distinction
that matters is between the shared LATENT FACTOR and the DEFAULT INDICATORS it
drives; they are not the same object and do not have the same dependence.

Obligors naming the SAME BETA_PD factor share the latent per-scenario PD
exactly: `P_i,s == P_j,s`. Their default indicators are NOT perfectly dependent.
Conditional on that factor they are independent draws,
`D_i,s | P_s ~ Bernoulli(P_s)` and `D_j,s | P_s ~ Bernoulli(P_s)`, and
unconditionally

    Corr(D_i, D_j) = 1 / (alpha + beta + 1)

which is strictly below 1 for every legal `(alpha, beta)`. Perfect dependence of
the latent factor is not perfect dependence of the defaults.

Obligors naming DISTINCT BETA_PD factors have independent latent factors, hence
independent defaults: the generator derives each factor column from a disjoint
column of an i.i.d. standard-normal core, and the factor-universe validator
forbids two special columns from aliasing.

`riskquant.credit` does NOT enforce one factor per pool — `beta_factor` is a
free per-obligor column, so two cohorts MAY share a latent factor. Assigning a
distinct BETA_PD factor per group is the `beta_binomial_grouped` recipe's
guarantee (`riskquant.solutions.credit_var`), not this package's.



### Multi-factor copulas (C-07)

The obligor latent variable is

    A_i = sum_k beta_ik * Z_k  +  w_i * eps_i,     w_i = sqrt(1 - sum_k beta_ik^2)

with `Z_k` the systemic factors, `eps_i` the obligor's own idiosyncratic factor,
and every one of those a distinct column of an i.i.d. standard-normal core.
Hence `Var(A_i) = 1` and

    Corr(A_i, A_j) = sum_k beta_ik * beta_jk

That identity holds **only because `eps_i` is private to obligor i.** Every
copula model therefore declares `idiosyncratic_factor` a distinct column, and
ingest rejects two obligors naming the same one: sharing it would add `w_i*w_j`
to the pair's correlation without appearing anywhere in the loadings.

Independent systemic factors do NOT mean independent obligors. `Z_k` are
mutually independent by construction; obligors are coupled through shared
LOADINGS. Two obligors loading 0.6 on the same factor have latent correlation
0.36; two obligors loading 0.6 on different factors have latent correlation 0.

**Student-t: zero correlation is NOT independence.** The t models scale the same
composite by a shared chi-square mixing variable:

    T_i = sqrt(nu_i) * W_c(i)^(-1/2) * A_i,     W ~ chi^2(nu)

Conditional on `W`, the `T_i` have exactly the Gaussian dependence above, so
disjoint loadings give conditional independence. Unconditionally they do NOT:
both obligors default more readily when `W` is small, so for disjoint loadings

    P(default_i and default_j)
        = E_W[ Phi(t_i sqrt(W/nu)) * Phi(t_j sqrt(W/nu)) ]
        >  P(default_i) * P(default_j)

strictly. This is the t-copula's tail dependence and it is intended. Genuine
independence between two cohorts under Student-t requires BOTH disjoint systemic
loadings AND distinct `chi_squared_factor` columns.

## Deprecated (removed next release)

Importable but warn on call; not in `__all__`. Migrate to the new name.

None.

## Explicitly internal (NOT part of the contract)

`jtd.registry` (model wiring; the supported way to *extend*, not a stability
promise), `jtd.ingest.builders.*`, `jtd.ingest._shared`, `jtd.ingest.contracts`
(`JtdModelContract`, `FACTOR_COLUMN_KIND`), `jtd.ingest.validate` predicates, all
`jtd.kernels.*` and `jtd.evaluators.*`, block classes / constructors
(`jtd.blocks.*`), the derived constants in `jtd.models` (`NUM_JTD_MODELS`,
`*_ID`, `MF_COPULA_MODELS`), and `scenario_metrics`
(a separate shared tail-stats package with its own surface).

## Resolved decisions
