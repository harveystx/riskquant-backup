# CONTINUATION.md — Current working state

Ephemeral. Allowed to go stale. See `CLAUDE.md` (durable rules) and
`DECISIONS.md` (rationale) for anything permanent.

Working checkout: `/tmp/repo/riskquant` (reconstructed from
`context_riskquant_all.txt`). Run tests with `PYTHONPATH=src python3 -m pytest`.

---

# CURRENT MISSION

Build, stage by stage, a production-owned end-to-end **sensitivity-based VaR**
pipeline, demonstrated by `scripts/demos/e2e/demo_e2e_sensi_based_var.py`. Each
increment reuses canonical production components, adds the smallest missing
production abstraction only when genuinely absent, validates against an
independent oracle, and ends with `ALL CHECKS PASSED`.

---

# CURRENT WORKSTREAM

The E2E demo pipeline. Last completed increment: **historical scenario path**
(replaced the parametric generator with `HistoricalGenerator` fed by a hand-
authored `RiskFactorMatrix`), which surfaced and fixed two production dtype/layout
defects. Prior increments (all landed): pricing, closed-form sensitivities,
deterministic scenario shocks + proxy, sensitivity P&L, reporting-currency
conversion + aggregation, unweighted VaR/ES.

---

# CURRENT STATUS

## DONE
- Positions → `CompiledPortfolio` (SPOT-only) + `UixMapper`.
- `RiskFactorLevelSnapshot` → `MarketDataMatrix` → `ValuationState` → `run_valuation` (MTM).
- Production `riskquant.sensitivities` package (sibling of valuation): result,
  spot kernel, spot sensitizer, registry, engine, book adapter. Closed-form,
  rf_id-keyed, no Polars in core.
- Deterministic scenario shocks → proxy resolution → `apply_shocks_to_base_levels`
  → shocked levels → `FactorAxis.levels` → `SensitivityPnlCalculator.run`.
- `riskquant.reporting.fx_conversion` (convention (a), scenario FX, aggregate-
  after-convert) + independent oracle.
- Unweighted VaR/ES via `scenario_metrics.compute_var_es` + independent oracle.
- Historical path via `HistoricalGenerator` + hand-authored history.
- Two production defects fixed: matrix-builder/history float64 default (D017);
  apply-shocks kernel dual-layout overload for single-row batches (D008 extended).
- Demo ends `ALL CHECKS PASSED` (verified this session).

## IN PROGRESS
- None (last increment complete and validated).

## BLOCKED
- None. `CURRENT BLOCKER: NONE`.

## NOT STARTED (next candidates)
- EWMA / filtered-historical-simulation (FHS) weighting. `historical_fhs.py` and
  `core_math/ewma_volatility.py` already exist → reuse-first.
- Diversification decomposition / component-VaR (`scenario_metrics` has
  `compute_var_es_contribution` / `_kernel_lower_tail_contribution` — RQ-AUD-009).
- Forward/option/cashflow block compilers (the gate for non-spot portfolios;
  only SPOT is registered in both pricer and sensitizer registries).

## KNOWN / ACCEPTED
- 2 pre-existing failures + several excluded test modules — see KNOWN ISSUES.

---

# CURRENT OBJECTIVE

The historical increment is finished and green. The immediate objective is to
confirm state and pick the next increment (most likely EWMA/FHS weighting, or
diversification decomposition), per the "reuse-first" rule.

---

# IMMEDIATE NEXT ACTION

Before any new increment: **restore the missing handoff and confirm green.**

1. Run: `cd /tmp/repo/riskquant && PYTHONPATH=src python3 scripts/demos/e2e/demo_e2e_sensi_based_var.py`
   — confirm it exits `rc=0` and prints `ALL CHECKS PASSED`.
2. Run the focused increment scope:
   `PYTHONPATH=src python3 -m pytest tests/sensitivities/ tests/reporting/ tests/simlab/ tests/scenario_metrics/ tests/sensitivity_pnl/ tests/solutions/ tests/e2e/ tests/config/ tests/market_data/ --ignore=tests/market_data/calibrate -q`
   — expect `1193 passed, 2 skipped`.
3. Then, if starting EWMA/FHS: inspect `src/riskquant/simlab/scenarios/historical_fhs.py`
   and `src/riskquant/simlab/scenarios/core_math/ewma_volatility.py` FIRST; confirm
   `FilteredHistoricalGenerator` (name to verify) satisfies the same
   `ScenarioGenerator` protocol and emits the same `(S, K)` float32 F-contig
   `ScenarioSetBatch`. Wire it exactly as `HistoricalGenerator` is wired in
   `_run_simulated_scenarios`; do NOT add a new abstraction. Add an independent
   EWMA-weight oracle. Run `tests/simlab/` before touching the demo.

Do NOT begin EWMA before steps 1–2 are green.

---

# RELEVANT FILES

### Demo (orchestration only)
- `scripts/demos/e2e/demo_e2e_sensi_based_var.py`
  - Purpose: composes the whole pipeline; ends with `ALL CHECKS PASSED`.
  - `main()` call order: `_run_risk_static_config` → `_run_proxy_topology` →
    `_run_portfolio` → `_run_risk_factor_snapshot` → `_run_market_data` →
    `_run_pricing` → `_run_sensitivities` → `_run_scenario_shocks` →
    `_run_proxy_resolved_levels` → `_run_scenario_pnl` → `_run_golden_checks` →
    `_run_reporting_currency`(deterministic) → `_run_simulated_scenarios`
    (HISTORICAL) → `_run_reporting_currency`(simulated) → `_run_risk_measures` →
    `_run_scenario_golden_checks` → `_run_reporting_currency_golden_checks` ×2 →
    `_run_risk_measure_golden_checks` → combined verdict.
  - Note: the HISTORICAL section is still titled `# HISTORICAL SCENARIOS` and the
    function is named `_run_simulated_scenarios` (historically named; it now runs
    `HistoricalGenerator`). Constants: `_HISTORY_DATES` (10), `_HISTORY_LEVELS`
    (10×3), `_HISTORY_HORIZON=1`, `_HISTORY_BATCH_SIZE=4`, `_REPORTING_CCY="USD"`,
    `_VAR_CONFIDENCE_LEVEL=2/3`.
  - Tests: `tests/e2e/test_demo_e2e_sensi_based_var.py` (14).

### Production `riskquant.sensitivities` (NEW package this workstream)
- `src/riskquant/sensitivities/result.py` — `BlockSensitivities` SoA (int64 uids,
  int32 rf_ids, float64 coeffs; dtype/contiguity/finiteness guarded).
- `src/riskquant/sensitivities/sensitizer_kernels/spot.py` — `@parallel_njit`
  closed-form partials only.
- `src/riskquant/sensitivities/sensitizers/spot.py` —
  `sensitize_spot_block(*, block, uix_mapper, market_data, scenario=0)`; unpacks
  identity graph, calls kernel, resolves uix→rf_id; raises `NotImplementedError`
  on inverted FX legs.
- `src/riskquant/sensitivities/sensitizer_registry.py` — `Sensitizer` Protocol,
  `BLOCK_SENSITIZERS` (SPOT only), `resolve_sensitizer`,
  `UNREGISTERED_BLOCK_TYPE_NAMES`.
- `src/riskquant/sensitivities/engine.py` —
  `run_sensitivities(*, portfolio, uix_mapper, market_data, scenario=0) -> BlockSensitivities`.
- `src/riskquant/sensitivities/book_adapter.py` — ONLY Polars importer;
  `build_sensitivity_book_spec(*, sensitivities, displacement_convention=ABSOLUTE, quote_base=1.0) -> SensitivityBookSpec`.
  - Tests: `tests/sensitivities/test_spot_sensitizer.py` (12).

### Production `riskquant.reporting` (NEW package this workstream)
- `src/riskquant/reporting/fx_conversion.py` —
  `build_fx_conversion_plan(*, settlement_ccys, reporting_ccy, fx_factor_names) -> FxConversionPlan`;
  `convert_scenario_pnl_to_reporting_currency(*, settlement_pnl, fx_levels, plan) -> ReportingCurrencyPnl`.
  Convention (a); FX from `portfolio.fx`; fail-fast on missing/invalid/non-finite/
  non-positive FX, bad ccy, shape/dtype mismatch, float32 pnl.
  - Tests: `tests/reporting/test_fx_conversion.py` (19).

### Modified production files (defects fixed this workstream)
- `src/riskquant/market_data/risk_factors/builder_matrix.py` — default
  `float_dtype=np.float64` on both builders (D017).
- `src/riskquant/ingest_fixtures/constants.py` — `HISTORY_DTYPE=np.float64`.
- `src/riskquant/ingest_fixtures/market_data/canonical_market_data_history.py` —
  docstring updated to float64.
- `src/riskquant/simlab/shocks/apply.py` — float32→float64 `shock_scale`
  promotion (prior increment) + comment on single-row overload.
- `src/riskquant/simlab/shocks/shock_kernel.py` — `_kernel_apply_shocks` now
  carries BOTH `_F32_2D_F` and `_F32_2D_C` overloads (D008 extended).

### Reused production (unchanged this increment)
- `src/riskquant/simlab/scenarios/historical.py` — `HistoricalGenerator(matrix, horizon)`.
- `src/riskquant/simlab/scenarios/parametric.py` — `ParametricGenerator.standard_normal(...)`.
- `src/riskquant/simlab/builder/shock_universe.py` — `build_shock_universe`.
- `src/riskquant/proxy_resolver/proxy.py` —
  `resolve_required_factor_ids_to_market_data_factor_ids`, `ProxyResolution`.
- `src/riskquant/market_data/assemble.py` — `build_universe_mapping`,
  `build_market_data_matrix_from_snapshot`, `extract_levels_from_snapshot`.
- `src/riskquant/scenario_metrics/api/var_es.py` — `compute_var_es`.
- `src/riskquant/solutions/sensitivity_pnl/api.py` — `SensitivityBookSpec`,
  `FactorAxis`, `SensitivityPnlCalculator`, `SensitivityPnlResult`,
  `DisplacementConvention`.

### Other new tests this workstream
- `tests/sensitivities/test_scenario_pnl.py` (18)
- `tests/sensitivities/test_scenario_risk_measures.py` (12)
- `tests/sensitivities/test_simulated_scenarios_and_revaluation.py` (14)
- `tests/sensitivities/test_historical_scenarios.py` (18)

---

# IMPLEMENTATION STATE

| Component | Status | Current reality | Remaining work |
|---|---|---|---|
| Positions → portfolio (SPOT) | DONE | Only SPOT registered in pricer & sensitizer registries | Forward/option/cashflow compilers (separate workstream) |
| Pricing | DONE | `run_valuation`; MTM per settlement ccy | — |
| Sensitivities | DONE | `riskquant.sensitivities`, closed-form, rf_id-keyed | forward/option sensitizers when compilers land |
| Deterministic scenarios | DONE | 3 hand-authored log-return shocks | — |
| Proxy resolution | DONE | reused; WTI→BRENT×0.95, `ID_PROXY` | — |
| Sensitivity P&L | DONE | `SensitivityPnlCalculator.run` via `FactorAxis.levels` | — |
| Reporting currency | DONE | `riskquant.reporting`, convention (a) | triangulation via 3rd ccy (fails fast today) |
| VaR / ES | DONE | `compute_var_es`, C=2/3 | EWMA/weighted (next) |
| Historical scenarios | DONE | `HistoricalGenerator`, 10 obs → 9 scen, 3 batches | — |
| EWMA / FHS | NOT STARTED | `historical_fhs.py` + `ewma_volatility.py` exist | wire like historical; add oracle |
| Diversification / component VaR | NOT STARTED | `compute_var_es_contribution` exists (RQ-AUD-009) | wire + oracle |

---

# API STATE

- `run_sensitivities(*, portfolio: CompiledPortfolio, uix_mapper: UixMapper, market_data: MarketDataMatrix, scenario: int = 0) -> BlockSensitivities`
- `build_sensitivity_book_spec(*, sensitivities: BlockSensitivities, displacement_convention: DisplacementConvention = ABSOLUTE, quote_base: float = 1.0) -> SensitivityBookSpec`
- `resolve_sensitizer(block: Block) -> Sensitizer` — raises `NotImplementedError`
  if unregistered; `BLOCK_SENSITIZERS: tuple[Sensitizer|None, ...]` indexed by
  `Block.BLOCK_TYPE_ID`.
- `build_fx_conversion_plan(*, settlement_ccys: Sequence[str], reporting_ccy: str, fx_factor_names: Sequence[str]) -> FxConversionPlan`
  — `FxConversionPlan.fx_row_index (M,) int32` (`INVALID_UIX` = identity),
  `.invert (M,) int8`, `.settlement_ccys`, `.reporting_ccy`.
- `convert_scenario_pnl_to_reporting_currency(*, settlement_pnl: float64 (M,S) C-contig, fx_levels: float64 (K,S) C-contig, plan) -> ReportingCurrencyPnl`
  — `.converted_pnl (M,S) f64`, `.portfolio_pnl (S,) f64`, `.reporting_ccy`.
- `compute_var_es(group_value_matrix: float64 (n_groups, n_scenarios) C-contig, confidence_level: float) -> VarEsResult` (`.var`, `.es` per group; VaR negative for loss).
- `HistoricalGenerator(matrix: RiskFactorMatrix, horizon: int = 1)` →
  `.iter_batches(*, required_shift_type_id (Nreq,) int8, source_factor_ids (Nreq,) int32, shock_scale (Nreq,) float32, batch_size) -> Iterator[ScenarioSetBatch]`.
  `total_scenarios = T - horizon`.
- `build_risk_factor_matrix_from_df(*, df, name_to_id_map=None, use_int32_dates=True, float_dtype=np.float64) -> RiskFactorMatrix`  ← default now float64 (D017).
- `SensitivityPnlResult.row_for(position_uid: int) -> NDArray[float32]`; `.pnl (M,S) float32`, `.position_uids`.

---

# NUMERICAL STATE

Golden fixture (rf_id order 0=USDEUR, 1=AAPL_EQ, 2=BRENT_CRUDE, 3=WTI_CRUDE):
- Positions: `1010 +100 AAPL_EQ USD`, `1020 -50 BRENT_CRUDE EUR`, `1030 +75 WTI_CRUDE USD`.
- Proxy: `WTI_CRUDE → BRENT_CRUDE`, `vol_scaling_factor 0.95` (`ID_PROXY`).
- Base levels (snapshot as-of 2027-01-04): USDEUR 1.0800, AAPL_EQ 200.00,
  BRENT_CRUDE 82.30, WTI_CRUDE 70.00.
- MTM: AAPL +20,000.00 USD; BRENT −4,444.20 EUR (`= -50·82.30·1.08`); WTI +5,250.00 USD.
- Sensitivities: `dV/dAAPL=+100`, `dV/dBRENT=-54`, `dV/dUSDEUR=-4115`,
  `dV/dWTI=+75`; cross `BRENT×USDEUR = -50`. All 2nd-order own terms 0.
- Deterministic shocks `(S=3, K=3)` log returns, source order (USDEUR, AAPL_EQ, BRENT):
  `(0.000,0.010,-0.020)`, `(-0.010,0.000,0.015)`, `(0.020,-0.015,-0.010)`.
- Deterministic shocked levels (rows USDEUR/AAPL/BRENT/WTI, cols s0/s1/s2):
  USDEUR `1.080000/1.069254/1.101817`, AAPL `202.010033/200.0/197.022388`,
  BRENT `80.670354/83.543808/81.481104`, WTI `68.682555/71.004641/69.338149`.
- Deterministic scenario P&L (settlement ccy):
  AAPL `201.0033/0.0000/-297.7612`, BRENT(EUR) `88.0011/-22.2766/-44.6650`,
  WTI `-98.8083/75.3481/-49.6388`.
- Historical: 10 obs `2026-12-01..2026-12-14`, horizon 1 → **9 scenarios**,
  batch 4 → 3 batches (trailing single-row batch is the case that needed the
  kernel overload). WTI level = `70·exp(0.95·ln(Bₜ₊₁/Bₜ))`.
- Risk measures at **C=2/3** (α=1/3, S=9 → **m=3.0** via ε-snap):
  VaR/ES — AAPL `-160.1602 / -256.9353`, BRENT(EUR) `-103.7921 / -133.1128`,
  WTI `-71.1908 / -73.6665`, PORTFOLIO(USD) `-159.4309 / -257.1254`. All VaR < 0.
- Cross-term convention: `c·x_i·x_j` (no ½); diagonal 2nd-order `0.5·d2·x²`.
- FX: `USDEUR` = USD→EUR; Brent EUR→USD via `1/USDEUR` at each scenario's FX.

---

# TEST STATE (actual evidence, this session)

- Demo: `PYTHONPATH=src python3 scripts/demos/e2e/demo_e2e_sensi_based_var.py`
  → `rc=0`, `ALL CHECKS PASSED`, 0 FAIL. Check counts printed: 8 pricing/
  sensitivity, 25 scenario, 5+5 reporting-currency, 12 risk-measure.
- Focused increment scope (sensitivities, reporting, simlab, scenario_metrics,
  sensitivity_pnl, solutions, e2e, config, market_data − calibrate):
  **1193 passed, 2 skipped**.
- Per new/changed file: spot_sensitizer 12, scenario_pnl 18,
  scenario_risk_measures 12, simulated_scenarios_and_revaluation 14,
  historical_scenarios 18, reporting/fx_conversion 19, e2e demo 14.
- Full sweep (excluding known-broken modules below):
  **1539 passed, 8 skipped, 2 failed** — the 2 failures are the pre-existing
  `synthetic_level_generators.py` path-comment issue (see KNOWN ISSUES), proved
  by removing only that file → 44/44 architecture tests pass.
- Full-revaluation reconciliation (oracle): linear spot 1e-12, quanto+cross 1e-9,
  simulated quanto 1e-8, proxied WTI 1e-9.

---

# KNOWN ISSUES

## MUST FIX
- None currently blocking the workstream.

## SHOULD FIX (pre-existing, environmental — not caused by this workstream)
- `src/riskquant/synthetic_level_generators.py` — missing/incorrect first-line
  path comment → fails `tests/test_architecture.py` and
  `tests/test_no_stale_path_comments.py` (2 failures). Dated before this session.
- `tests/observations/test_compile.py` vs `tests/sensitivity_pnl/unit/test_compile.py`
  — duplicate basename with no `__init__.py` → pytest collection collision when
  both are collected in one run. Pre-existing.
- `tests/valuation/*` — collection `ImportError`s in this reconstruction:
  `NO_DATE_UIX` absent from `riskquant.constants`;
  `VanillaEuropeanOption{Node,Curve}Block` absent from
  `valuation.blocks.vanilla_european_option_block`. Excluded from sweeps here.
- `tests/credit/*` — `ImportError`: `_join_netting_set_ids` absent from
  `credit.exposure.compile`; `CANONICAL_REPLACEMENT_COST_COLLATERAL` absent from
  the credit-exposure fixture. Excluded.
- `tests/market_data/calibrate/*` — 5 failures (Black-76 calibrate `ValueError`
  in `option_quotes/master.py:242`). Related to RQ-AUD-003 tolerance area; not in
  this workstream's scope.

## ACCEPTED / KNOWN
- `tests/dependence_stress/*` cannot collect here — `hypothesis` not installed.
- `SimlabRunner.run()` has ZERO test coverage (which is why the float32
  `shock_scale` / single-row layout defects went unnoticed until this workstream;
  both now fixed at the boundary).
- RQ-AUD-003 (A&S CDF), RQ-AUD-010 (EWMA vol clamp telemetry), RQ-AUD-011
  (`searchsorted_or_propagate_sentinel_int32` OOB return — clip explicitly
  prohibited) — all ACCEPTED-RISK/DOCUMENTED.
- RQ-AUD-004 RNG-ownership redesign — OPEN (D021).
- RQ-AUD-008 pandas-at-EDGE boundary tests — OPEN (D022).

---

# RECENT CHANGES (BEFORE → AFTER → WHY)

- `build_risk_factor_matrix_from_df` default `float32` → `float64` → the
  historical-shock kernel is float64-pinned; the float32 default produced
  matrices that raised `TypeError` and could not drive `HistoricalGenerator`
  (D017).
- `HISTORY_DTYPE` / canonical-history fixture `float32` → `float64` → same
  contract; the fixture explicitly mirrored the old (wrong) default.
- `_kernel_apply_shocks` single `_F32_2D_F` signature → dual `[_F32_2D_F,
  _F32_2D_C]` overloads → a trailing single-row `(1,K)` historical batch is both
  C/F-contiguous and Numba resolves to 'C'; `asfortranarray` can't disambiguate
  (D008 extended consumer-side).
- Demo `ParametricGenerator.standard_normal(...)` → `HistoricalGenerator(matrix,
  horizon)` fed by `_build_historical_matrix` → the increment's goal: canonical
  historical path. (`_run_simulated_scenarios` kept its name.)
- Demo `_run_risk_measures` USD-subset workaround → full-portfolio row from
  `ReportingCurrencyPnl.portfolio_pnl` → reporting-currency conversion makes a
  true whole-portfolio total possible (prior increment).
- ~75 files touched by a blanket `ruff --fix` → restored from the dump → scope
  discipline (D023).

---

# DO NOT REGRESS

- VaR sign: negative for a loss. Never a positive loss magnitude.
- Cross-term: `c·x_i·x_j` (no ½). Diagonal 2nd-order: `0.5·d2·x²`.
- Reporting convention (a): scenario-FX × settlement P&L; aggregate after convert;
  never sum mixed-currency P&L.
- FX direction from `portfolio/fx.py`; `USDEUR` = USD→EUR; EUR→USD = `1/USDEUR`.
- Proxy: WTI keeps base 70.00 and `dV/dWTI=+75`; shock only = `0.95×ΔBRENT`
  inside the exp. Never substitute Brent's level for WTI.
- Historical: `float64` matrix; `T-horizon` scenarios; dual-overload kernel.
- Scenario/axis orientation and dtypes exactly as in NUMERICAL STATE.
- `sensitivities` core is Polars-free; `book_adapter` is the only Polars importer;
  rf_id resolution happens once; no bump-and-revalue.
- Only SPOT registered in the sensitizer registry — do NOT add stubs (D010).
- All golden numbers above and the demo `ALL CHECKS PASSED` verdict.
- Empty `__init__.py`; per-file path comment (except the pre-existing
  `synthetic_level_generators.py`, tracked above).

---

# OPEN QUESTIONS

- EWMA/FHS: does `historical_fhs.py` expose a generator with the same
  `ScenarioGenerator` protocol and `(S,K)` f32 F-contig batch contract? (Verify by
  inspection before wiring.) — OPEN.
- Reporting-currency triangulation (settle→3rd→report) is unsupported and fails
  fast; is triangulation in scope for any near-term increment? — OPEN.
- RQ-AUD-004 generator RNG-ownership redesign timing. — OPEN.

---

# CURRENT BLOCKER

NONE.

---

# NEXT STEPS

1. Run the demo and the focused increment scope; confirm `ALL CHECKS PASSED` and
   `1193 passed, 2 skipped`.
2. Choose the next increment (default: EWMA/FHS weighting).
3. For EWMA/FHS: inspect `simlab/scenarios/historical_fhs.py` +
   `core_math/ewma_volatility.py`; confirm the generator protocol; wire it exactly
   like `HistoricalGenerator` in `_run_simulated_scenarios`; add an independent
   EWMA-weight oracle; run `tests/simlab/` before touching the demo.
4. Stage only changed files to `/mnt/user-data/outputs/riskquant/<path>`;
   `present_files`; verify byte-identical. Do not blanket-lint.
