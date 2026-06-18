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
