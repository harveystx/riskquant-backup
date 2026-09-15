# CLAUDE.md — RiskQuant Operating Manual

Permanent operating manual for Claude instances working in this repository.
Durable project knowledge only. For architectural rationale see `DECISIONS.md`;
for current task state see `CONTINUATION.md`.

> **Provenance note.** This file was authored from a working reconstruction of
> the `riskquant` repo checked out at `/tmp/repo/riskquant`, plus the estate
> constitution referenced by prior audits (`context_github.txt`) and the audit
> ledger `RISKQUANT_AUDIT_LEDGER.md`. Where a fact could not be verified against
> the checked-out code it is marked `UNKNOWN`.

---

## 1. Project purpose

RiskQuant is a high-performance Python/Numba **credit and market risk engine**.
It prices portfolios, computes sensitivities, generates risk-factor scenarios
(parametric Monte-Carlo, historical, bootstrap/FHS), applies proxy topologies,
computes scenario P&L, converts to a reporting currency, and computes tail risk
measures (VaR/ES). The design philosophy is Data-Oriented Design (DOD): human-
readable at the edge, contiguous typed arrays at the core.

The estate is three repositories (only `riskquant` is checked out here):
- **`riskquant`** — core pricing/simulation/sensitivity/credit engine (this repo).
- **`stx-data-god`** — Anti-Corruption Layer over the trading data warehouse
  (Polars LazyFrames + Pandera). Not present in this checkout.
- **`riskquant-excel`** — xlwings Excel terminal over `solutions.*`. Not present.

The live workstream in this repo is the end-to-end sensitivity-based VaR demo
`scripts/demos/e2e/demo_e2e_sensi_based_var.py`, which composes the production
pipeline stage by stage. See `CONTINUATION.md`.

---

## 2. Architecture

### 2.1 Four-tier model (Constitution)

Dependencies flow one direction only: **TERMINAL → SOLUTION → CORE → (nothing)**.

- **EDGE / ingest**: Polars, pandas (tolerated legacy), Pydantic, `fastexcel`.
  Strings, dates, business identifiers live here and die here.
- **CORE** (`valuation`, `simlab`, `sensitivities`, `sensitivity_pnl`,
  `scenario_metrics`, `credit`, `dependence`, `observations`): NumPy/Numba only.
  Flat SoA arrays, `int32` execution indices, sentinels, zero allocation in hot
  paths, no frames/strings/objects.
- **SOLUTION** (`riskquant.solutions.*`): the seam. Pydantic contracts, model
  catalogs. Depends on core; core never depends on it. **The only public import
  surface for end users.**
- **TERMINAL** (`riskquant-excel`, VBA): thin shell. Not in this repo.

### 2.2 Compiler / orchestrator / kernel responsibilities

- **Compilers** (edge→core boundary): resolve strings→ids, validate, lay out
  SoA. Own all policy. Fail fast. e.g. `compile_risk_static_config_from_dict_of_dfs`,
  `compile_portfolio_from_ingest_batch`, `SensitivityPnlCalculator.compile`.
- **Orchestrators** (`*/engine.py`, demo `_run_*` helpers): dispatch on integer
  block/type ids, gather already-resolved arrays, call kernels, concatenate.
  No mathematics of policy significance.
- **Kernels** (`*/pricer_kernels/*`, `*/sensitizer_kernels/*`,
  `simlab/shocks/shock_kernel.py`, `scenario_metrics/_kernels/*`): dumb math on
  primitive typed arrays. `@njit`/`@parallel_njit`. No identity resolution, no
  sentinel interpretation, no allocation, no strings.

### 2.3 The E2E pipeline (current, verified)

RiskStaticConfig (compile)

ProxyTopology (compile, independent of RiskStaticConfig)
positions (pandas edge) -> ingest -> CompiledPortfolio (SoA, SPOT block) + UixMapper
RiskFactorLevelSnapshot (name->rf_id resolved once at edge)
│
├── build_universe_mapping -> build_market_data_matrix_from_snapshot -> MarketDataMatrix
│ │
│ ├── assemble_valuation_state -> run_valuation -> ValuationResult (MTM)
│ └── run_sensitivities -> BlockSensitivities -> build_sensitivity_book_spec
│ -> SensitivityBookSpec -> SensitivityPnlCalculator.compile()
│
├── scenario shocks (deterministic tuple OR HistoricalGenerator/ParametricGenerator)
│ -> resolve_required_factor_ids_to_market_data_factor_ids (proxy)
│ -> build_shock_universe -> ScenarioSetBatch
│ -> apply_shocks_to_base_levels -> shocked_levels (N_req, S)
│
├── FactorAxis.levels(base, shocked) -> SensitivityPnlCalculator.run() -> SensitivityPnlResult (M,S)
├── reporting.fx_conversion -> ReportingCurrencyPnl (per-position + portfolio, one ccy)
└── scenario_metrics.compute_var_es -> VarEsResult (per group VaR/ES)


### 2.4 Key architectural invariants

- **`valuation` and `sensitivities` are siblings, not consumer/producer.** Both
  read the same `CompiledPortfolio` and market state at different points of the
  fold. `sensitivities` never calls `run_valuation` (no bump-and-revalue).
- **`ValuationState` is FOLDED** (`effective_spot_prices = spot × fx`, its
  `spot_prices`/`fx_spots_effective` fields are commented out). Sensitivities
  need the UNFOLDED components, so they take `MarketDataMatrix` (which retains
  `prices (N_mkt,S)` and `fx_spots_effective (K_fx,S)`), never `ValuationState`.
- **`solutions.sensitivity_pnl` is currency-agnostic.** It has no reporting-
  currency concept. Currency conversion is a layer placed *around* scenario P&L
  aggregation (`riskquant.reporting`), never threaded into the engine.
- **Proxying changes shock provenance, never a base level.** A proxied factor
  keeps its own base level and its own sensitivity identity.
- **`__init__.py` files are always empty.** Public surfaces are explicitly named
  modules (`solutions/<name>/api.py`). Enforced by `tests/test_architecture.py`.
- **First line of every `.py` module** is a comment with its repo-relative path
  (e.g. `# src/riskquant/sensitivities/engine.py`). Enforced by
  `tests/test_no_stale_path_comments.py` + `tests/test_architecture.py`.

---

## 3. Core engineering principles (concrete rules)

- **FAIL FAST.** Reject malformed input at the human-readable boundary before any
  kernel. No silent repair: no assumed defaults, no `1.0` for a missing FX rate,
  no dropped factors, no broadcast/transpose to make a shape "fit", no NaN as an
  error signal. Every validator raises with table/column/value/rule context.
- **Kernels do dumb math.** No policy, no identity resolution, no sentinel
  interpretation, no allocation inside a kernel. Everything is resolved to
  primitive typed arrays by the caller.
- **Validation/orchestration lives outside kernels**, at the compiler/edge.
- **Human-readable at the edge, contiguous at the core.** Strings die at ingest
  (`name -> rf_id` resolution happens exactly once). Names only re-appear at the
  presentation boundary (demo print layer).
- **Golden-oracle validation.** Every numeric result is checked against an
  independent oracle derived from the mathematical definition — never by calling
  the production function under test.
- **Benchmark-first / prove empirically.** Performance and layout claims require
  a probe recorded in code; never assert a speedup or a dtype behaviour from
  memory. (See `engineering-principles`: adversarial array sweeps, mutation
  tests, run-to-run variance are standard gates.)
- **Explicit dtypes, explicit indexing, no implicit sentinel semantics.** Use the
  canonical sentinel constants from `riskquant.constants` (e.g. `INVALID_UIX`);
  never invent a new one.
- **Smallest correct change.** Do not replace a narrow architectural solution
  with a broader abstraction merely because it looks more elegant.
- **No unrelated changes in one patch** (Constitution §21): never bundle an
  architecture migration, a math fix and cleanup because they touch one file.

---

## 4. Type / dtype conventions

Three representations, kept distinct:

| Concept | Python/container (edge) | Boundary (DataFrame/ingest) | Computational (core) |
|---|---|---|---|
| identifiers | `int` | Polars `Int64`/`Int32` | see below |
| `*_uid` (forged global trade id) | `int` | `Int64` | `np.int64` |
| `rf_id` / `factor_id` (forged risk-factor id) | `int` | `Int32` | `np.int32` |
| `*_uix` (compact block-local index) | — | — | `np.int32` |
| `ccy_id` | `int` (0..127) | `Int8` | `np.int8` (bounds-checked, RQ-AUD-014) |
| dates | `datetime.date` / pandas ts | `pl.Date` / `datetime64[D]` | `np.int32` ordinal (days since 1970-01-01) |
| flags / shift types / conventions | `int`/enum | `Int8` | `np.int8` |
| prices / PVs / quantities / strikes / levels | `float` | `Float64` | `np.float64` |
| **historical observation matrix values** | — | — | **`np.float64`** (kernel-pinned; see §5, RQ-defect-2026-09-14) |
| **scenario shocks** | — | — | **`np.float32`**, **F-contiguous** |
| **shocked levels buffer** | — | — | **`np.float64`, C-contiguous, shape `(N_req, S)`** |
| **P&L (SensitivityPnlResult.pnl)** | — | — | `np.float32`, shape `(M, S)` |
| **reporting-currency P&L** | — | — | `np.float64` |
| **VaR/ES group matrix** | — | — | `np.float64`, C-contiguous, `(n_groups, n_scenarios)` |
| scenario FX levels (reporting) | — | — | `np.float64`, C-contiguous, `(K, S)` |

Rules:
- **Never** `datetime`/`pd.Timestamp` in core; use int32 ordinals.
- **Never** `np.nan`/`None`/`pd.NA` in core; use `riskquant.constants` sentinels
  (`INVALID_UIX = np.int32(-999)`, `INVALID_UNIT_SCALE = np.float64(-999_999.0)`).
- `int8` narrowing must be bounds-checked at the edge (RQ-AUD-014 fixed the
  unguarded `ccy_id` case).
- Roll-metadata numeric columns are read from Excel as `Float64`, integrality-
  checked, then cast to `Int32` — a non-integral value is rejected, never
  truncated.

---

## 5. Numba rules

`riskquant.jit` is the canonical module for all Numba related definitions and
alias.

- **May enter a kernel:** primitive typed NumPy arrays and NumPy scalars with the
  exact dtype and contiguity the signature pins.
- **May NOT enter a kernel:** strings, Python objects/`object` dtype, DataFrames,
  Pydantic models, `datetime`, dicts. No NaN/None as an error signal.
- **Validation location:** entirely outside the kernel (compiler/edge). Kernels
  assume validated input.
- **Typing:** kernels carry pinned `_NB_VOID(...)` signatures. Use the canonical
  type aliases from `riskquant.jit` (`_F64_2D_F`, `_F32_2D_F`, `_F32_2D_C`,
  `_I32_1D_C`, `_I8_1D_C`, `_F64_1D_C`, …). **Never** redefine a layout alias
  locally (RQ-AUD-005 removed 12 files' worth; `tests/test_no_local_numba_aliases.py`
  guards this). **Never** import a layout-suffix-less alias that doesn't exist in
  `jit.py` (RQ-AUD-001; `tests/test_no_undefined_jit_aliases.py` guards this).
- **Allocation:** none in hot paths. Pre-allocate `out` at the edge; pass explicit
  `out=` buffers.
- **Memory layout:** `order="F"` down columns (scenario/time-series generation);
  `order="C"` across rows (pricing, shocked-levels output). The apply-shocks
  kernel reads shock columns with unit stride → shocks are F-contiguous.
- **Read-only:** where the signature declares `_..._RO`, honour it.
- **Decorators:** dense loops use `@parallel_njit(...)` (which the repo configures
  with the canonical `fastmath`/`cache`/`boundscheck` policy — do not hand-set
  these per-call; use the `jit.py` partials).
- **Dual-layout overloads:** a kernel that can receive an ambiguous single-row
  `(1,K)` batch (both C- and F-contiguous; Numba resolves to 'C') must carry BOTH
  `_F32_2D_F` and `_F32_2D_C` overloads (RQ-AUD-017 generator-side; the same
  pattern was extended consumer-side to `simlab/shocks/shock_kernel.py`,
  2026-09-14). `asfortranarray` **cannot** disambiguate a single row.
- **Scalar vs array kernels:** both exist; keep orchestration (loops over blocks,
  concatenation, dispatch) in Python outside the kernel.

---

## 6. Numerical conventions (permanent)

- **Scenario orientation.**
  - `ScenarioSetBatch.shocks`: `(S, K_source)` float32 **F-contiguous** — one
    ROW per scenario, one COLUMN per source factor.
  - Shocked levels / sensitivity inputs: `(N_required_factor, S)` float64 C-contig.
  - Sensitivity-P&L output `(M_position, S)` float32.
  - VaR/ES input `(n_groups, n_scenarios)` float64 C-contig.
- **Shocks are displacements already in each factor's declared shift convention.**
  Nothing derives them; no base levels needed to interpret a `ScenarioShocks`.
- **`shift_type` (level reconstruction).** For `LOG_RETURN` (the demo universe):
  `shocked_level = base_level * exp(shock * shock_scale)`. This is the
  shock→level stage inside `_kernel_apply_shocks`.
- **Proxy scaling multiplies the shock INSIDE the exponent.**
  `WTI_level(s) = 70.00 * exp(0.95 * BRENT_shock(s))`. The proxied factor keeps
  its own base level (70.00) and its own sensitivity (`dV/dWTI = +75`).
- **`DisplacementConvention` (sensitivity space).** Describes the space the
  sensitivity was differentiated in. `ABSOLUTE`: `x = shocked_level - base_level`.
  On the levels path (`FactorAxis.levels`) the kernel derives `x`; on the shocks
  path the caller must pre-convert. The two stages (shift→level, level→displacement)
  are joined by the shocked-levels buffer; there is **no** LOG_RETURN↔ABSOLUTE
  conversion to hand-write.
- **Sensitivity-P&L Taylor form (empirically verified):**
  `PnL = Σ_i [ d1_i·x_i + 0.5·d2_i·x_i² ] + Σ_{i<j} c_ij·x_i·x_j`.
  The **cross term has NO ½** (½ is absorbed by summing only `i<j`). The DIAGONAL
  second-order term DOES carry ½.
- **FX convention (authority: `portfolio/fx.py`).** Pair name is
  `{underlying_ccy}{settlement_ccy}`. So `USDEUR` multiplies **USD → EUR**. The
  inverse (EUR → USD) is `1 / USDEUR` (recorded as `invert=1`). Never hard-code
  the direction; derive from the FX authority.
- **Reporting-currency P&L (LOCKED convention (a); see `DECISIONS.md` D014).**
  `dV_report[i,s] = dV_settlement[i,s] * F_settlement→report[i,s]` at the
  **scenario** FX rate. It is NOT `V_shocked·F_shocked − V_base·F_base` (that
  cancels the FX term and deletes the FX exposure the book carries). The two
  disagree in sign once FX moves. Aggregate to portfolio ONLY after every
  position is in the reporting currency.
- **VaR/ES (authority: `scenario_metrics`).**
  - `alpha = 1 - confidence_level`; `m = alpha * n_scenarios`.
  - **VaR is the lower P&L quantile ITSELF, negative for a loss.** Never
    sign-flipped to a positive loss magnitude. More negative = greater downside.
  - VaR = the `ceil(m)`-th smallest observation (order statistic).
  - ES integrates exactly `m` probability mass: `full_tail_count` observations at
    unit weight + a fractional weight on the boundary observation when `m` is
    non-integral, divided by `m`.
  - A documented float-noise snap (`≈16·eps`) means e.g. `(1-2/3)*9` resolves to
    `m = 3.0`, not `3.0000000000000004`. Any independent oracle MUST replicate
    this snap.
- **Historical ordering.** `RiskFactorMatrix.dates` sorted ascending (oldest
  first), unique. Historical simulation yields `T - horizon` scenarios from
  consecutive `h`-day changes. Batch size does **not** change historical
  scenarios (deterministic function of observations).
- **Parametric reproducibility.** Seeded, but reproducibility is per
  `(seed, batch_size)` — the generator draws a `(batch_size, k)` block per batch,
  so batch geometry partitions the stream. Both must be pinned.

---

## 7. Testing philosophy

- **Golden-oracle**: independent economic/mathematical oracle, never the
  production function under test. Demos end with an explicit
  `ALL CHECKS PASSED` / `VALIDATION FAILED` verdict computed from oracle checks.
- **Anti-fragile two-layer golden masters**: Layer 1 = input-agnostic structural
  invariants; Layer 2 = a tiny test-owned frozen book with hand-computed values.
- **Do not duplicate existing production coverage.** Add tests for genuinely new
  seams only. (E.g. the 148-test `scenario_metrics` suite was NOT re-implemented;
  only the integration seam was tested.)
- **Fail-fast tests** are first-class: assert that malformed axis/shape/dtype/
  currency/missing-factor/non-finite/non-positive-FX inputs RAISE.
- **Tolerances driven by dtype**, never loosened to hide an error. float32 shocks
  carry ~1.2e-7 relative precision → a double-precision `rtol=1e-12` is
  mathematically unsatisfiable (RQ-AUD-017 relaxed such assertions to `rtol=1e-6`).
- **Run focused tests before broad ones.** Never claim a suite passed without
  running it.
- Demo scripts are loaded in tests by file path via
  `importlib.util.spec_from_file_location` (there is no `scripts/__init__.py`).

---

## 8. Performance philosophy

- Zero allocation in hot paths; pre-allocate and pass `out=`.
- SoA + C/F-contiguity contracts are load-bearing, not cosmetic.
- Streaming generators pre-allocate a single buffer and yield row-slices.
- Warm/cold JIT distinction matters; kernels are `cache=True` via the `jit.py`
  partials.
- Performance is measured and the measurement recorded; never asserted.
- No specific latency SLA is recorded in this checkout → `UNKNOWN`.

---

## 9. Dependency policy

- **Python:** 3.12 (the environment). Exact minimum `UNKNOWN`.
- **Required:** `numpy`, `numba`, `polars`, `pandas` (edge/legacy only),
  `pydantic` v2 (solution tier), `pyarrow` (polars↔pandas), `fastexcel`
  (`pl.read_excel` calamine engine — an OPTIONAL polars dep that MUST be pinned;
  RQ-AUD-021), `xlsxwriter`/`openpyxl` (xlsx tests).
- **Test-only:** `pytest`, `hypothesis` (property-based; NOT installed in this
  environment → `tests/dependence_stress` cannot collect here).
- **Deliberately avoided:** pandas/DataFrames/Pydantic/strings inside core/Numba;
  pandas past the ingest edge into any hot path.
- **Import rule:** end users import only from `riskquant.solutions.*`. Core
  packages are not public API.
- Verified versions in the reconstruction environment: numba 0.67.0,
  pydantic 2.13.5, pyarrow 25.0.1, fastexcel 0.21.0, polars 1.44.2, numpy 2.4.4,
  pandas 3.0.2. These are the *environment's*, not necessarily the repo's pins.

---

## 10. Code style

- First line of every `.py` = `# <repo-relative-path>` comment.
- `from __future__ import annotations`; strict concrete type hints on every
  signature; keyword-only (`*,`) public functions.
- `@dataclass(frozen=True, slots=True, kw_only=True)` for config/state/result
  containers. `eq=False` on any array-bearing dataclass (autogenerated `__eq__`
  would return an elementwise array — a recurring P0).
- Names encode the stable output contract, not asset-class dispatch
  (`resolve_quantity_price_effective_notional`, not `resolve_commodity_*`).
- NumPy-style docstrings; comments explain *why* (quant/layout decisions),
  not what. Every optimisation/layout decision cites its probe.
- Error messages name table/column/value/rule. No bare exceptions.
- No `# type: ignore` in `src/riskquant/config` (grep-enforced); prefer
  `typing.cast` at genuine boundaries.
- `ruff check` + `ruff format` clean. `mypy --strict` for non-kernel files.

---

## 11. Agent operating rules

1. **Read the repo before changing architecture.** The repo is the source of
   truth for what exists; `DECISIONS.md` for why. Inspect before proposing.
2. **Do not restart or re-derive the architecture.** Do not casually revisit
   LOCKED decisions. Do not replace an established narrow design with a "cleaner"
   broad one.
3. **Reuse existing production machinery** before adding an abstraction. Add a
   new module only for a genuinely missing reusable capability, not a demo
   convenience. (No scenario/proxy/P&L/VaR reimplementation.)
4. **Prefer the smallest correct change.** One concern per patch; never bundle
   unrelated architecture/math/cleanup edits.
5. **Preserve exactly:** dtype semantics, ID/UID/UIX semantics, shape/axis
   orientation, sign conventions, VaR sign, cross-term convention, FX direction,
   reporting-currency convention (a), historical ordering, proxy separation,
   golden values, known regression values.
6. **FAIL FAST.** If repo state contradicts the handoff or a required file is
   missing, surface the discrepancy and stop — do not silently invent
   architecture or substitute one workstream for another.
7. **Scope lint/format to the files you changed.** Do NOT run `ruff --fix` across
   the whole tree (a prior session did this and touched ~75 unrelated files that
   then had to be restored from the dump).
8. **Run focused tests, then broad. Never claim a test passed without running it.**
   Record actual pass/fail counts and the exact command.
9. **Stage deliverables** to `/mnt/user-data/outputs/riskquant/<repo-path>` and
   `present_files`; verify byte-identical to the working tree.
10. No conversational summaries in code or handoff artifacts.
