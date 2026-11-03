Status: internal contract of `riskquant.credit`. It binds riskquant code only.

# `riskquant.credit`: internal contract

## Status

- `riskquant.credit` is a core package. D001 makes core packages private: code
  outside riskquant imports from `riskquant.solutions.*` only. The credit-VaR
  entry point for such code is `riskquant.solutions.credit_var.api`.
- This file gives no promise to code outside riskquant. A name, a signature
  or a module path in this file can change in any commit.
- The list below records the names that each module of `riskquant.credit`
  exports to other riskquant packages. The list equals the union of the
  `__all__` lists of the `riskquant.credit` modules, each name at the module
  that owns it. Test S5.T4 of WS_CREDIT_VAR_API checks this equality. A change
  of an `__all__` list needs a change of this list in the same commit.
- Import convention. Each `__init__.py` is empty (D002); no re-export hub
  exists. Import a listed name from its owning module by full path, for
  example `from riskquant.credit.jtd.engine import run_jtd`. This keeps the
  Numba-free ingest modules free of the kernel import cost.

## Listed names

### _scenario_coverage

- `riskquant.credit._scenario_coverage.assert_scenario_coverage_complete`
- `riskquant.credit._scenario_coverage.new_scenario_coverage`
- `riskquant.credit._scenario_coverage.record_scenario_span`

### aggregate

- `riskquant.credit.aggregate.loss_metrics.PortfolioLossDistribution`
- `riskquant.credit.aggregate.loss_metrics.expected_loss`
- `riskquant.credit.aggregate.loss_metrics.expected_shortfall`
- `riskquant.credit.aggregate.loss_metrics.loss_quantiles`
- `riskquant.credit.aggregate.loss_metrics.loss_var_es`
- `riskquant.credit.aggregate.loss_metrics.loss_var_es_contribution`
- `riskquant.credit.aggregate.loss_metrics.tail_var_es_by_group`
- `riskquant.credit.aggregate.loss_metrics.value_at_risk`
- `riskquant.credit.aggregate.summary.JtdRiskSummary`
- `riskquant.credit.aggregate.summary.build_portfolio_loss_distribution`
- `riskquant.credit.aggregate.summary.model_tail_var_es`
- `riskquant.credit.aggregate.summary.streaming_model_tail_var_es`
- `riskquant.credit.aggregate.summary.summarize_jtd_loss`
- `riskquant.credit.aggregate.summary.summarize_streaming_jtd_loss`

### exposure

- `riskquant.credit.exposure.add_on.compiled.CompiledAddOnFactors`
- `riskquant.credit.exposure.add_on.ingest.compile_add_on_factors`
- `riskquant.credit.exposure.add_on.resolve.SACCR_MF_ONE_YEAR_CAP`
- `riskquant.credit.exposure.add_on.resolve.SACCR_MF_TEN_BUSINESS_DAY_FLOOR`
- `riskquant.credit.exposure.add_on.resolve.resolve_quantity_price_effective_notional`
- `riskquant.credit.exposure.add_on.resolve.resolve_saccr_effective_notional`
- `riskquant.credit.exposure.add_on.resolve.resolve_supervisory_factors`
- `riskquant.credit.exposure.obligor_ead.align.align_obligor_ead_to_portfolio`
- `riskquant.credit.exposure.obligor_ead.compiled.CompiledObligorEad`
- `riskquant.credit.exposure.obligor_ead.ingest.compile_obligor_ead`
- `riskquant.credit.exposure.saccr.aggregate.compute_saccr_add_on`
- `riskquant.credit.exposure.saccr.config.CompiledSaccrConfig`
- `riskquant.credit.exposure.saccr.ingest.compile_saccr_config`

### factors

- `riskquant.credit.factors.simulate.base.build_credit_factor_matrix`
- `riskquant.credit.factors.simulate.mixed.MixedCreditFactorGenerator`
- `riskquant.credit.factors.simulate.parametric_beta.BetaGenerator`
- `riskquant.credit.factors.simulate.parametric_copula_gaussian.GaussianCopulaGenerator`
- `riskquant.credit.factors.simulate.parametric_copula_t.TCopulaGenerator`
- `riskquant.credit.factors.simulate.parametric_uniform.IndependentUniformGenerator`
- `riskquant.credit.factors.state.CreditFactorMatrix`

### jtd

- `riskquant.credit.jtd.bridge.aggregate_ead_to_obligor`
- `riskquant.credit.jtd.compiled.CompiledJtdPortfolio`
- `riskquant.credit.jtd.engine.run_jtd`
- `riskquant.credit.jtd.engine.run_jtd_for_obligor_ead`
- `riskquant.credit.jtd.ingest.bundle.JtdIngestBundle`
- `riskquant.credit.jtd.ingest.compile.compile_jtd_portfolio`
- `riskquant.credit.jtd.models.JtdModel`
- `riskquant.credit.jtd.results.JtdLossResult`
- `riskquant.credit.jtd.results.StreamingJtdResult`
- `riskquant.credit.jtd.streaming.constant_ead_provider`
- `riskquant.credit.jtd.streaming.run_jtd_streaming`
- `riskquant.credit.jtd.streaming.run_jtd_streaming_from_generator`

### masters

- `riskquant.credit.masters.cross_master.compiled.CompiledCreditMasters`
- `riskquant.credit.masters.enums.credit_factor_kind.CreditFactorKind`
- `riskquant.credit.masters.factor_universe.compiled.CompiledCreditFactorUniverse`
- `riskquant.credit.masters.factor_universe.ingest.compile_credit_factor_universe`
- `riskquant.credit.masters.netting_set.compiled.CompiledNettingSetMaster`
- `riskquant.credit.masters.netting_set.ingest.compile_netting_set_master`
- `riskquant.credit.masters.obligors.compiled.CompiledObligorCredit`
- `riskquant.credit.masters.obligors.ingest.compile_obligor_master`

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

## Not listed

Each name that is in no `__all__` list is outside this contract. Examples:
`jtd.registry` (model wiring; the supported way to extend the model set),
`jtd.ingest.builders.*`, `jtd.ingest._shared`, `jtd.ingest.contracts`
(`JtdModelContract`, `FACTOR_COLUMN_KIND`), the `jtd.ingest.validate`
predicates, all `jtd.kernels.*` and `jtd.evaluators.*`, block classes and
constructors (`jtd.blocks.*`), and the derived constants in `jtd.models`
(`NUM_JTD_MODELS`, `*_ID`, `MF_COPULA_MODELS`). `scenario_metrics` is a
separate shared tail-statistics package with its own contract.
