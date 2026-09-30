# DECISIONS.md — RiskQuant Architectural Decision Record

Durable decisions and rationale. New decisions append sequentially. Do not turn
an unresolved discussion into LOCKED. Mark superseded decisions clearly.

> Scope note: D001–D0xx below cover both the long-running audit workstream
> (`RISKQUANT_AUDIT_LEDGER.md`, RQ-AUD-###) and the E2E sensitivity-VaR demo
> build-out. Audit remediations are recorded compactly; the demo/pipeline
> decisions carry full rationale because they most constrain ongoing work.



## D001 — Four-tier model, one-way dependencies
**Status:** LOCKED
**Decision:** EDGE → SOLUTION → CORE with strict one-way deps; `solutions.*` is
the sole public import surface; core packages are private.
**Reason:** DOD + separation of policy (edge/compilers) from math (kernels).
**Implications:** Never import a core module from user code; never let core
import solution. New public capability is exposed via `solutions/<name>/api.py`.

## D002 — `__init__.py` always empty; per-file path comment
**Status:** LOCKED
**Decision:** every `__init__.py` empty; first line of every `.py` is its
repo-relative path comment.
**Reason:** explicit named public surfaces; architecture greppability.
**Implications:** enforced by `tests/test_architecture.py` +
`tests/test_no_stale_path_comments.py`. RQ-AUD-006 remediated 28 files.

## D003 — Canonical Numba type aliases live only in `jit.py`
**Status:** LOCKED
**Decision:** all Numba array-type aliases imported from `riskquant.jit`; no
local redefinition; imported names must exist in `jit.py`.
**Reason:** one source of truth for layout/dtype signatures; prevents
ImportError-at-load and signature drift.
**Implications:** guarded by `tests/test_no_local_numba_aliases.py` and
`tests/test_no_undefined_jit_aliases.py`. RQ-AUD-001, RQ-AUD-005.

## D004 — Sentinels over NaN/None in core
**Status:** LOCKED
**Decision:** invalid financial data in core uses explicit sentinel constants
from `riskquant.constants` (`INVALID_UIX = np.int32(-999)`,
`INVALID_UNIT_SCALE = np.float64(-999_999.0)`), never NaN/None/pd.NA. Consumers
guard sentinels explicitly.
**Reason:** silent NaN propagation is a P0 failure class.
**Implications:** RQ-AUD-012 fixed a `np.nan` unit-scale slot. Do not introduce
new sentinels; reuse the constants.

## D005 — Explicit int8 narrowing with bounds check
**Status:** LOCKED
**Decision:** narrowing an unconstrained `int` to `np.int8` (e.g. `ccy_id`)
requires an edge bounds check (`0 ≤ x ≤ 127`) raising `ConfigurationError`.
**Reason:** silent overflow → cryptic `OverflowError`/unbounded matrix.
**Implications:** RQ-AUD-014. `_validate_ccy_id_bounds` in config validate.

## D006 — Config ingest is Polars, not pandas
**Status:** LOCKED
**Decision:** both config ingest pipelines (static + proxy) are Polars end to
end. pandas tolerated only at explicit legacy consumer contracts.
**Reason:** Constitution default; pandas must not cross into core.
**Implications:** RQ-AUD-018 (B-class migration; numerical equivalence proven by
byte-for-byte JSON diff on the canonical fixture). `fastexcel` must be pinned
(RQ-AUD-021) — `pl.read_excel`'s calamine engine needs it.

## D007 — `Block.__init_subclass__` uses `object.__init_subclass__()`
**Status:** LOCKED
**Decision:** explicit `object.__init_subclass__()` instead of zero-arg
`super()` in `valuation/blocks/_base_block.py`.
**Reason:** Python 3.12 `@dataclass(slots=True)` rebuilds the class, poisoning
the `__class__` cell used by zero-arg `super()` → `TypeError` at subclass
definition. RQ-AUD-016.
**Implications:** `Block`'s only base is `object`, so this is exactly equivalent;
the two real validation paths (missing/invalid `BLOCK_TYPE_ID`) still fire.

## D008 — Streaming generators: fresh F-buffer + dual-overload kernels
**Status:** LOCKED
**Decision:** partial/single-row batch slices are copied into a fresh F-order
buffer before a kernel call; kernels that can see an ambiguous `(1,K)` batch
carry BOTH `_F32_2D_F` and `_F32_2D_C` overloads.
**Reason:** a `(1,K)` array is both C- and F-contiguous; Numba resolves to 'C';
`asfortranarray` cannot disambiguate. RQ-AUD-017 (generator side).
**Implications:** extended CONSUMER-side to `simlab/shocks/shock_kernel.py`
(2026-09-14) so `apply_shocks_to_base_levels` accepts a single-row historical
batch. Do not remove either overload.

## D009 — `valuation` and `sensitivities` are siblings; no bump-and-revalue
**Status:** LOCKED
**Decision:** sensitivities are computed in closed form from the payoff, in a
`sensitivities` package mirroring `valuation`'s block-dispatch structure. The
engine never calls `run_valuation`.
**Reason:** `ValuationState` is folded (`effective_spot_prices = S×FX`, with
`spot_prices`/`fx_spots_effective` commented out) and cannot supply the unfolded
components sensitivities need. Bump-and-revalue would be wrong-tier and slow.
**Implications:** `sensitize_spot_block` takes `MarketDataMatrix` (unfolded
`prices`/`fx_spots_effective`), `UixMapper`, NOT `ValuationState`, NOT
`RiskStaticConfig`. Spot partials are exact (affine/bilinear payoff), machine-
precision vs full revaluation.
**Rejected alternatives:** bump-and-revalue (wrong tier, imprecise, couples to
pricing); passing `ValuationState` (insufficient — folded).

## D010 — Sensitizer registry fails fast at CALL time, not import time
**Status:** LOCKED
**Decision:** `BLOCK_SENSITIZERS` registers only authored block types (SPOT
today); `resolve_sensitizer` raises `NotImplementedError` for an unregistered
block; `UNREGISTERED_BLOCK_TYPE_NAMES` names the gap at import.
**Reason:** `valuation.pricer_registry` fails at IMPORT if any block type is
unregistered — copying that would force stub sensitizers, and a stub returning
zero sensitivities silently reports a risk-free position for a risky trade.
**Implications:** do NOT add stub sensitizers for forward/option/cashflow to
"complete" the registry. Add a real one when authored.
**Rejected alternatives:** import-time completeness guard (forces dangerous
stubs).

## D011 — `SensitivityBookSpec` assembled at the edge (option A)
**Status:** LOCKED
**Decision:** the numerical core emits contiguous NumPy `BlockSensitivities`
(`sensitivities/result.py`); a separate edge adapter (`book_adapter.py`, the
ONLY Polars importer in the package) converts to the validated
`SensitivityBookSpec`.
**Reason:** `SensitivityBookSpec` is a Pydantic/Polars-validated edge
representation, not the natural output of numerical work. Polars must not reach
a kernel.
**Implications:** kernels/sensitizers/engine/result stay pure NumPy. `FactorAxis`
is built from an explicit Int32 series matching the `factor_id` dtype (validator
rejects Int32/Int64 mismatch).

## D012 — Sensitivities keyed by GLOBAL rf_id, resolved once
**Status:** LOCKED
**Decision:** the sensitizer resolves uix→global `rf_id` exactly once at the end
of the numerical stage; the book, factor axis, and result are keyed by `rf_id`.
**Reason:** downstream consumers must not need the portfolio's mapper to
interpret a sensitivity row; strings never enter the numerical core.
**Implications:** identity chain is
`effective_spot_price_uix → UixMapper.spot_effective_pairs → (price_uix, fx_eff_uix)`,
`fx_eff_uix == INVALID_UIX` ⇒ no FX leg. Sorted-unique rf_ids ⇒ zero-copy DIRECT
`FactorAxis` encoding.

## D013 — Proxy is a shock transformation; identity ≠ provenance
**Status:** LOCKED
**Decision:** proxying re-routes which column a required factor's shock reads and
scales it (`shock_scale` inside the exponent). The proxied factor keeps its own
base level and its own sensitivity identity.
**Reason:** `ProxyResolution.shock_scale` is documented "multiplicative scale
applied to the source shock". WTI must keep `dV/dWTI=+75` while its shock is
`0.95 × BRENT`.
**Implications:** `WTI_level(s) = 70·exp(0.95·ΔBRENT(s))`; base 70.00 unchanged.
Reuse `resolve_required_factor_ids_to_market_data_factor_ids` +
`build_shock_universe` + `apply_shocks_to_base_levels`; never reimplement proxy
in the demo. Proxy occurs AFTER simulation, BEFORE P&L.

**Status: SUPERSEDED by D088 (2026-09-25).** The definition of proxying above — re-routing which column a required factor's shock reads and scaling it, with `shock_scale` inside the exponent, selected by `resolve_required_factor_ids_to_market_data_factor_ids` — is no longer authoritative. The RiskQuant Proxy V1 contract (D088) defines a proxy as a model for the target's shock: `X_A = Σ_j β_Aj · Y_(d_j, τ_j, h)` over world columns, with an explicit driver chart `τ_j` per term and a signed finite float64 loading applied to the already-formed h-horizon column, followed by the target's own reconstruction `A_s = T⁻¹_κA(A_0, X_A)`. Mapping is compile-time and node-level; no runtime resolver selects a source. The example `WTI_level(s) = 70·exp(0.95·ΔBRENT(s))` remains valid only as the special case `κ_A = LOG`, `J_A = {(BRENT, LOG)}`, `β = 0.95`. Retained and still locked: identity ≠ provenance; the proxied factor keeps its own base level and its own sensitivity identity; proxying acts after simulation and before P&L; the D019 golden values as that special case.

## D014 — Reporting-currency P&L uses scenario FX on the settlement P&L (convention a)
**Status:** LOCKED
**Decision:** `dV_report[i,s] = dV_settlement[i,s] * F_settlement→report[i,s]` at
the SCENARIO FX rate. Aggregate to portfolio ONLY after conversion.
**Reason:** the alternative `V_shocked·F_shocked − V_base·F_base` cancels the FX
factor (`V_usd = Q·B`) and DELETES the USDEUR exposure the book carries
(`dV/dUSDEUR=-4115`). The two disagree in sign once FX moves (deterministic case:
−40.536 vs +40.945). A conversion that erases a reported risk is self-
contradictory.
**Implications:** new `riskquant.reporting` package; the sensitivity engine stays
currency-agnostic. FX pair/direction from `portfolio/fx.py`
(`{under}{settle}`; `USDEUR` = USD→EUR; EUR→USD is `1/USDEUR`, `invert=1`).
Never sum mixed-currency P&L before conversion.
**Rejected alternatives:** convention (b) value-convert-then-difference (deletes
FX risk); a reporting-ccy concept threaded through `solutions.sensitivity_pnl`
(the engine is deliberately currency-agnostic).

**Status: SUPERSEDED by D081 (2026-09-20).** The equation `dV_report = dV_settlement × F_scenario` is retained here as history. It is the `V_0 = 0` special case of ordinary MTM and omits base-value translation `V_0(q_s − q_0)`. Its justification was inverted: for a USD-priced, EUR-settled position under USD reporting, `W = V_EUR / F = Q·B` has no USDEUR exposure, so ordinary MTM does not "delete" a USD risk — the EUR-book exposure and the translation exposure net exactly. On D014's own example the two measures give −40.5 versus +40.9 USD (a short into a falling price); the sign flip is the omitted translation. Production now uses D081 §2; the former form is unreachable because `V_0` and `q_0` are required inputs.

## D015 — VaR/ES via `scenario_metrics`; VaR is the negative lower quantile
**Status:** LOCKED
**Decision:** reuse `scenario_metrics.api.var_es.compute_var_es`. VaR is the
lower P&L quantile itself (negative for a loss), NOT a positive loss magnitude.
ES integrates exactly `m = (1-C)·S` mass with fractional boundary weight.
**Reason:** the canonical package (148 tests) already defines this; a second
implementation would drift.
**Implications:** no new VaR/ES code. Oracles must replicate the `≈16·eps` snap
(`(1-2/3)·9 → m=3.0`). Per-trade rows stay in settlement ccy; the PORTFOLIO row
is the reporting-ccy aggregate.

## D016 — Historical path reuses `HistoricalGenerator`; no scenario adapter

**Status:** LOCKED

**Decision:** the demo composes existing public production functions
(`build_risk_factor_matrix_from_df` → `HistoricalGenerator` → proxy →
`apply_shocks_to_base_levels`). No `scenario_adapter` module created.

**Reason:** every capability already exists as a public function; `SimlabRunner`
is a full-revaluation runner requiring a `ScenarioGenerator` and is not reusable
for the sensitivity path, but its constituents are.

**Implications:** demo is orchestration-only. Historical simulation is
deterministic in the historical observations. For stochastic generators,
scenario-stream reproducibility is governed by D029: for fixed seed, model,
data, parameters, and scenario count, the concatenated stream is invariant to
`batch_size`.

**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.

## D017 — `RiskFactorMatrix` / history default dtype is float64 (kernel contract)
**Status:** LOCKED (2026-09-14)
**Decision:** `build_risk_factor_matrix_from_df` /
`build_risk_factor_matrix_from_xlsx` default `float_dtype=np.float64`;
`HISTORY_DTYPE = np.float64`.
**Reason:** the historical-shock kernel `_kernel_fused_h_day_shock` is dtype-
pinned to `_F64_2D_F` values (an h-day shock differences nearby levels — the
precision is load-bearing). The prior float32 default silently produced matrices
that raised `TypeError: No matching definition` and could NOT drive the canonical
historical generator (the shipped canonical fixture was float32 too).
**Implications:** a float32 matrix still (correctly) fails the kernel — that is
the contract, pinned by `test_float32_matrix_is_rejected_by_the_kernel_contract`.
**Rejected alternatives:** adding a float32 kernel overload (precision is the
point; a level-difference in float32 loses significance).

## D018 — Full revaluation is an ORACLE, not the risk path
**Status:** LOCKED
**Decision:** the main risk path is sensitivity P&L; full revaluation of a small
subset is used only as an independent validation oracle.
**Reason:** sensitivity P&L is the production approximation; for affine/bilinear
spot payoffs it reconciles to full revaluation at machine precision (with the
cross term), which validates the approximation without switching the main path.
**Implications:** oracle tests reconcile linear spot (1e-12), quanto-with-cross
(1e-9), simulated quanto (1e-8). Do not route the main pipeline through
`run_valuation` per scenario.

**Status: SUPERSEDED by D081 (2026-09-20).** "Full revaluation is an ORACLE, not the risk path" predates batch valuation and encoded a performance judgement. Under D081 the production methodologies are FULL_REVAL, TAYLOR/SUPPLIED and TAYLOR/DERIVED; FULL_REVAL is a production methodology and the exact valuation route, and additionally serves as the consistency reference for Taylor positions that have a raw representation. D018's reconciliation evidence (Taylor ≡ full revaluation for affine/bilinear spot payoffs) is retained as the certification standard for DERIVED on spot.


## D019 — Demo constants are a golden fixture; do not drift
**Status:** LOCKED
**Decision:** the demo's universe (rf_ids 0–3, positions, proxy 0.95, base
levels, deterministic shocks, historical levels, `C=2/3`, reporting USD) is a
hand-verifiable golden fixture; its numbers are pinned by golden checks and the
e2e test.
**Reason:** these values back the independent oracles; changing them silently
would invalidate every reconciliation.
**Implications:** change a constant only together with its golden checks and
`tests/e2e/test_demo_e2e_sensi_based_var.py`.

## D020 — ACCEPTED-RISK: A&S 7.1.26 normal-CDF approximation
**Status:** LOCKED (ACCEPTED-RISK, RQ-AUD-003)
**Decision:** retain the A&S 7.1.26 CDF approximation (max CDF error 6.99e-8 ≤
published 7.5e-8; max pricing error ~0.0005%); calibrate tests tolerance 5e-4
with inline attribution.
**Reason:** negligible for a Monte-Carlo VaR engine.
**Implications:** documented in `math.py`; do not "fix" by swapping for
`scipy.special.ndtr` without a benchmark decision.

## D021 — RNG ownership / independence remains partially open

**Status:** OPEN (RQ-AUD-004 remainder)

**Decision:** the batch-size reproducibility defect is CLOSED by D029. The
remaining open question is **independence between separately constructed
generators given the same base seed**, together with the longer-term ownership
model for RNG state.

The current generators rebuild `default_rng(self.seed)` inside `iter_batches`.
This provides deterministic restart semantics for a fixed seed but does NOT
provide independent streams for co-seeded generator instances.

**Reason:** changing RNG ownership to explicit generator state at construction
would be a public API/design change across multiple stochastic generators.
That decision should be made deliberately rather than bundled into the
batch-size fix.

**Implications:**

* D029's batch-size-invariant scenario-stream contract is already LOCKED.
* Do not rely on same-seed generator instances being statistically independent.
* Do not introduce cross-generator `SeedSequence` policy until the remaining
  RNG ownership question is separately resolved.
* The deferred redesign must preserve D029's batch-size invariance.


## D022 — pandas-at-EDGE audit: no defects, boundary tests not added
**Status:** OPEN (RQ-AUD-008)
**Decision:** pandas at EDGE where Polars is default was audited; no defects
found in sensitivity_pnl/proxy_resolver/market_data assembly; boundary tests not
yet added.
**Implications:** tracked; not blocking.

## D023 — Lint/format scoped to changed files only
**Status:** LOCKED
**Decision:** never run `ruff --fix`/`ruff format` across the whole tree; scope
to the files changed in the increment.
**Reason:** a session ran a blanket `ruff --fix` and cosmetically modified ~75
unrelated files (credit, ingest_fixtures, …), which then had to be restored from
the dump — violating one-concern-per-patch and inflating the diff.
**Implications:** lint/format only your changed paths; verify diff scope before
staging.

## D024 — Single-column `(T,1)` C/F ambiguity: `_F64_2D_C` overloads on historical-family kernels
**Status:** LOCKED (2026-09-14)
**Decision:** the plain and filtered historical shock kernels accept a single-
column `(T,1)` float64 array via an added `_F64_2D_C` overload alongside the
existing `_F64_2D_F`. Fixed kernels:
`ar1_filter.kernel_ar1_residuals` (+ F-in/C-out and C-in/C-out),
`ewma_volatility.kernel_ewma_vol_ratios_from_residuals` (+ C-in/C-out),
`historical._kernel_fused_h_day_shock` (+ values-C/out-C),
`historical_fhs._kernel_fused_filtered_h_day_shock` (+ residuals-C/vol-C/out-C).
**Reason:** a single-required-factor universe (`K_req==1`) or single-factor
matrix (`K_total==1`) presents a `(T,1)` array. NumPy reports it as BOTH C- and
F-contiguous; Numba resolves the ambiguity to 'C'; the `_F64_2D_F`-only
signatures raised `TypeError: No matching definition`. Same ambiguity class as
D008 / RQ-AUD-017 on the shocks OUTPUT — here on the float64 INPUT/scratch,
along the K (column) axis. Both `HistoricalGenerator` and
`FilteredHistoricalGenerator` were affected; the demo never hit it (3-factor
book), and both generators were untested for the single-factor shape.
**Implications:** per-column reads (`arr[:, k]`) are stride-correct under either
layout, so the C overload is value-identical to F — proven: a single-factor
result equals the corresponding column of the multi-factor run at
`max_abs_diff = 0.0`, for both generators. Covered by
`tests/simlab/scenarios/test_filtered_historical_scenarios.py` (single-factor
matrix + single-required-factor-from-multi, both generators). Do NOT remove
either overload (cf. D008).
**Known twins (NOT fixed here):**
`bootstrap_fhs._kernel_fused_filtered_bootstrap_h_day_shock` and
`bootstrap_sbb._kernel_fused_sbb_h_day_shock` carry the identical
`_F64_2D_F`-pinned-input defect. Apply the same `_F64_2D_C` overload when the
bootstrap generators are next touched (they also carry the OPEN RNG-ownership
caveat, D021). The shared `ar1`/`ewma` fixes above already benefit
`bootstrap_fhs`.
**Rejected alternatives:** (a) caller-side `asfortranarray` — cannot
disambiguate `(T,1)` (both contiguity flags stay True), exactly as D008 found
for `(1,K)`; (b) a single `'A'`-layout signature — discards the load-bearing
F-contiguity contract (D003) and its vectorization guarantee for the common
multi-factor path.

## D025 — Shift taxonomy completion: NO_SHIFT + ABSOLUTE_BASE_RETURN (R1)

Context: SimLab supported only LOG_RETURN(1)/SIMPLE_RETURN(2)/DIFFERENCE(3).
NO_SHIFT(-1) existed in the enum but was hard-rejected at the shock-universe
boundary; ABSOLUTE_BASE_RETURN did not exist. Findings F1 (silent-NaN domain)
and F2 (unwritten `np.empty` on unhandled shift) required a complete taxonomy.

Decisions:
- `ShiftType.ABSOLUTE_BASE_RETURN = 4`. `0` stays intentionally unused/invalid
  (avoids confusion with zero-initialised arrays); positives remain 1..4.
- Canonical valid set `VALID_SHIFT_TYPE_IDS = {-1,1,2,3,4}`, enum-derived
  (single source of truth). `build_shock_universe` now validates explicit
  membership, replacing the `min()<0` range check: `-1` is valid (round-trips
  through the low-8-bit dedup key: `-1 & 0xFF = 255 -> int8 -1`), `0`/unknown
  fail fast.
- `ABSOLUTE_BASE_RETURN` forward: `r = (v_t - v_{t-1}) / abs(v_{t-1})`.
  Inverse (apply reconstruction): `out = base + shock*scale*abs(base)`. Equals
  SIMPLE for `base>0`; differs only for `base<0`, where the shock magnitude
  scales by `|base|` so a negative rate/spread moves with the shock sign.
- NO_SHIFT gets BOTH protections: the filtered/bootstrap fused kernels
  short-circuit to shock=0 BEFORE reading residual/vol inputs, AND `ar1_filter`
  writes zero residuals for NO_SHIFT (no residual column is ever left unwritten).
  Apply reconstructs `out = base`.
- Shift-specific level-domain validation at the generator edge
  (`simlab/builder/shift_domain.py`, called from each historical-family
  `iter_batches`): LOG requires levels>0; SIMPLE/ABSOLUTE require levels!=0;
  DIFFERENCE requires finite; NO_SHIFT skipped. NOT a global positive-levels
  rule — DIFFERENCE and ABSOLUTE admit negative levels.

MULTI-DAY AGGREGATION (RESOLVED — canonical). ABSOLUTE_BASE_RETURN is the
sign-robust SIMPLE_RETURN: `|V_prev|` replaces `V_prev` in the daily/endpoint
return, `|base|` handles sign in reconstruction, and the aggregation is
IDENTICAL to SIMPLE at every stage — never additive.
  - Historical (endpoint 2-point read): `(V_t - V_{t-h}) / |V_{t-h}|`.
  - SBB / FHS / FilteredBootstrap (daily composition): `prod_j (1 + r_j) - 1`,
    `r_j = (V_j - V_{j-1}) / |V_{j-1}|` — i.e. the SIMPLE branch fed absolute
    residuals/ratios.
Rationale: on strictly positive levels `|V|=V`, so ABSOLUTE is definitionally
identical to SIMPLE; this must hold for every generator and every horizon.
SIMPLE aggregates multiplicatively (endpoint-ratio in Historical, product in the
daily kernels); therefore ABSOLUTE must too. The numerator being a difference
does NOT imply additive — the `|denominator|` makes it a return, and returns
compose like SIMPLE, not like DIFFERENCE (only LOG returns and raw DIFFERENCES
add). Counterexample (h=2, path 100->110->121): SIMPLE=endpoint=multiplicative
=0.21; additive=0.20 (drops compounding, rejected). At h=1 all conventions
collapse to the single-period definition. On negative levels the generators
differ (endpoint vs multiplicative) exactly as they already do for SIMPLE via
filtering/resampling — inherent, not a defect; Historical endpoint recovers the
realised level. Superseded the initial additive implementation; SBB/FHS/
FilteredBootstrap ABSOLUTE branches corrected to multiplicative. Historical and
apply were already correct and unchanged.

Test correction (not a weakening): `test_parametric_scenarios.py` used
`MOCK_REQ_SHIFT_TYPES = [0,0]` with a comment mislabelling `0` as LOG_RETURN
(LOG_RETURN is 1). The old `min()<0` guard admitted `0`; the valid-set guard
correctly rejects it. Corrected to the real `SHIFT_LOG_RETURN_ID`; the
covariance assertion (shift-id-independent) is unchanged.

Out of R1 scope (unchanged): RNG batch-size (F6/D021), bootstrap single-column
layout (F4), ScenarioSetBatch contract (F7), config/Excel, P3 hygiene.

## D026 — R2 strict boundary contracts (F3/F5/F7)

Scope: strict discrete typing, historical matrix dtype boundary, and the
ScenarioSetBatch contract. RNG/D021, F4 bootstrap layout, configuration, and P3
were explicitly out of scope and untouched.

F3 — discrete quantities. `horizon` (Historical, FilteredHistorical, SBB,
FilteredBootstrap) and `total_scenarios` (SBB, FilteredBootstrap, Parametric)
are strict: a Python int or NumPy integer is accepted; bool and float raise
TypeError; non-positive raises ValueError; the canonical Python int is persisted
via `object.__setattr__` so kernels receive integral types without downstream
repair. One shared gate `base.require_positive_int(value, name) -> int` backs
all seven call sites — justified by genuine duplication, and it does not obscure
the model-specific checks (s_total>0, history length) that stay in each
`__post_init__`. Parametric's silent `int(self.total_scenarios)` coercion was
replaced by the strict gate. Historical/FilteredHistorical derive
`total_scenarios = T - horizon` (init=False); validating `horizon` makes that
derived count integral, so they need no separate `total_scenarios` gate.

DECISION (flag for override): `expected_block_size` (SBB) is CONTINUOUS, not
discrete. It is the mean block length of the stationary bootstrap's geometric
block-length distribution and is genuinely real-valued (its default is `1.0`, a
float; an int-only policy would reject its own default). It is validated as a
finite real (int or float) >= 1.0, rejecting bool and non-finite, and persisted
as float. The discrete int-only policy applies to counts (horizon,
total_scenarios), not to this rate parameter.

F5 — historical matrix dtype boundary (D017). `RiskFactorMatrix` accepts any
float dtype and silently coerces layout to F, so a float32 matrix previously
reached a deep, opaque Numba "No matching definition" error. The four
historical-family generators now call `base.require_float64_values` in
`__post_init__`: a non-float64 matrix fails at the generator boundary with a
clear message. No float32 kernel overloads were added; no input data is cast at
the boundary — the caller fixes dtype at the data source. Parametric is exempt
(it consumes a Cholesky factor and volatilities, not a historical matrix).

F7 — ScenarioSetBatch contract, now formal. Canonical contract:
`shocks` 2-D float32 F-contiguous; `required_to_shock_uix` 1-D int32;
`shock_scale` 1-D floating; `required_shift_type_id` 1-D int8; the three (Nreq,)
arrays share length; `scenario_offset` int >= 0; `scenario_count_total` int > 0
or None; bool rejected for both scalars.

DECISION: `shock_scale` is validated as 1-D floating (float32 OR float64), not
strict float32. The generators emit float32, but `apply_shocks_to_base_levels`
deliberately casts `shock_scale` to float64 at the kernel seam and a standing
regression (`test_scenario_pnl.py::...:535`) constructs a batch with float64
scale to pin that seam. Enforcing strict float32 would break a legitimate,
tested tolerance. `shocks` remains strict float32 (kernel-pinned, no cast).

DECISION: batch validation is an O(1) structural `__post_init__` — dtype, ndim,
contiguity flags, scalar type/sign, and shape-length equality only. No O(N) data
scans on the hot path: the sole O(N) invariant (every `required_to_shock_uix`
entry indexes a valid shock column) is guaranteed once upstream by
`build_shock_universe`, mirroring the existing one-shot
`validate_apply_shocks_buffers`. All five generators already emit a
contract-conformant batch, verified empirically before the contract was pinned.

**F7 status: SUPERSEDED IN PART by D088 (2026-09-25).** The proxy-specific batch fields `required_to_shock_uix`, `shock_scale` and `required_shift_type_id`, and the `(Nreq,)` length-sharing rule, are no longer part of the generator batch contract. Under D088 a generator is requested for world columns keyed `(d, τ, h)` and a batch carries the values of those columns as a `(B, K)` matrix; the `shocks` layout rule (2-D float32, F-contiguous), `scenario_offset`, `scenario_count_total`, the bool rejections and the O(1) structural-validation principle remain locked. Term structure and loadings belong to the proxy model (D088). The target reconstruction chart `κ_A` remains owned by the target factor definition and bound on the run axis (D082 clause 11); it does not become proxy-model-owned because the proxy operator reads it during reconstruction. F3 (discrete quantities), F5 (float64 matrix boundary) and the `expected_block_size` decision are unaffected.

Test alignment (not a weakening):
`test_float32_matrix_is_rejected_by_the_kernel_contract` previously asserted the
deep-Numba failure at `iter_batches`; it now asserts the boundary TypeError at
construction (`match="must be float64"`). Same intent — float32 fails loudly —
now earlier and clearer, which is the F5 improvement.

Observation (pre-existing, out of scope, not an R2 defect): the E2E demo's
"Holiday calendars" diagnostic prints its US/NL rows in a non-deterministic
order (set/dict iteration) — it flips run-to-run independent of R2. Values are
identical; only the print order varies. The 55 internal golden checks pass and
D019 LOG_RETURN goldens are unchanged. Noted for a future P3 hygiene pass.

## D027 — R3 F4 bootstrap single-column layout completeness

Scope: the D024 twin defect only. The fused SBB and FilteredBootstrap h-day
kernels were `_F64_2D_F` input-only. A single-column `(T, 1)` float64 matrix is
reported C-contiguous by NumPy (the `(T,1)` C/F ambiguity), so Numba resolved
the call to a C layout and raised "No matching definition" — the same defect
D024 fixed for the historical-family kernels.

Fix (minimal, D024 pattern): added one `_F64_2D_C` input overload to each kernel,
mirroring the existing `_kernel_fused_filtered_h_day_shock` three-signature
shape:
  - `_kernel_fused_sbb_h_day_shock`: added `(values C, out C)`.
  - `_kernel_fused_filtered_bootstrap_h_day_shock`: added
    `(residuals C, vol_ratios C, out C)`.
The kernel bodies are layout-agnostic (column-slice indexing), so Numba
specialises per signature with no body change. No `'A'` layout, no caller
copy/reshape to force layout, no change to the F-contiguous common path, and no
change to numerical formulas, batching, RNG, or shift semantics.

Dispatch proof (compiled Numba signatures captured, both kernels now carry three
specialisations): all five shapes run — K_total=1/K_req=1/B>1, K_total>1/K_req=1/
B>1, K_total=1/K_req=1/B=1, K_total>1/K_req=1/B=1, and a partial final batch.
The single-column matrix now dispatches to the C-input specialisation.

Numerical equivalence: for the same seed, the single-column `(T,1)` result is
BITWISE equal (max|diff| = 0.0) to the corresponding column of a multi-factor
`(T,2)` matrix, for both generators and every horizon — the column's output does
not depend on how many sibling columns share the matrix.

Note (left untouched, unrelated cleanup out of R3 scope): the R1 SBB oracle in
`test_shift_types.py` still uses a `(T,2)` matrix with a comment about avoiding
this defect. That test passes as-is; simplifying it to `(T,1)` is cleanup for a
later pass, not part of R3.

## D028 — R4 independent numerical oracles (test-only)

Scope: closed the SimLab audit's oracle-coverage gap with independent tests
only. No production code changed (scope diff: `src/` byte-identical to the
pristine tree); no defect surfaced. RNG/D021, configuration, P3, kernel layouts,
and shift semantics untouched.

Oracles are structurally independent of production (they compute from the
mathematical definitions, not the kernel algorithm):
  - Apply-shocks (`tests/simlab/shocks/test_apply_shocks_oracle.py`): vectorised
    NumPy reconstruction of all five shift ids (NO_SHIFT, LOG, SIMPLE,
    DIFFERENCE, ABSOLUTE) for positive and negative base levels with a non-unit
    proxy `shock_scale`; the production per-exposure Numba loop is the SUT.
  - FHS pipeline (`tests/simlab/scenarios/core_math/test_fhs_pipeline_oracle.py`):
    AR(1) residuals via closed-form OLS + a 95% t-test de-mean gate (production
    uses a two-pass running-sum kernel); EWMA vol ratios via the explicit
    geometric-weighted variance sum `var_t = lam^(t-1) e_1^2 +
    (1-lam) sum_k lam^(t-k) e_k^2` (production uses the forward recursion);
    end-to-end filtered shocks composed as `f_t = e_t * v_t` aggregated over the
    deterministic HistoricalFHS window `[s+1 .. s+h]`, h=1 and h>1.
  - Shock-universe (`tests/simlab/builder/test_shock_universe_dedup.py`):
    `(source_factor_id, shift_type_id)` dedup via plain Python-tuple dict
    (production uses int64 bit-packing + `unique_first_seen`). Verified exact
    mapping, order-independence under permutation, no collision across distinct
    shift types, NO_SHIFT=-1 round-trips as int8 -1 (not 255), and the canonical
    valid-id set {-1,1,2,3,4} is accepted while others are rejected.

Evidence: 34 R4 tests pass; tests/simlab 236; focused regression 1396 (+2
skipped); E2E `ALL CHECKS PASSED`, D019 LOG goldens unchanged; changed files
lint-clean.

## D029 — R5 batch-size reproducibility (F6; batch-size half of D021)

**Status:** LOCKED (F6 / RQ-AUD-004 batch-size half)

**Contract:** for fixed model + input data + model parameters + seed +
total_scenarios, the concatenated scenario stream is IDENTICAL regardless of
`batch_size`. `batch_size` is an orchestration/memory parameter only.

**Archaeology:** SBB and FilteredBootstrap already draw all path indices up
front and slice them per batch, so they were batch-size invariant. Parametric
drew base normals per batch and then, for distributions requiring auxiliary
randomness, drew a second variate per batch. This caused random-number
consumption to depend on batch partitioning.

For `standard_normal` no auxiliary draw exists, so it was already invariant.
For `student_t`, `laplace`, `variance_gamma`, and `regime_switching_gmm`, the
auxiliary draw was interleaved with the normal draw at each batch boundary,
causing different streams for different batch sizes. This was a genuine
production defect.

**Fix:** localized to `ParametricGenerator.iter_batches`. All randomness is
drawn once up front in the canonical full-run order, then sliced per batch.
`_apply_distribution_scaling` was split into a stochastic upfront draw phase
and a deterministic per-batch application phase. No public API change and no
change to the underlying RNG algorithm or distribution formulas.

The full-batch pre-fix stream is preserved exactly; previously divergent
smaller-batch streams now reproduce the canonical full-run stream.

**Established semantics:**

* Fixed integer seed → deterministic scenario stream.
* Same seed + same model/data/parameters/count → same stream.
* Different `batch_size` → same concatenated stream.
* Repeated `iter_batches()` calls restart the same deterministic stream.
* Same-seed generator instances are intentionally identical under the current
  API, not independent.
* `seed=None` provides entropy-backed stochastic output; no reproducibility
  claim is made beyond normal execution semantics.

**Tests:** `tests/simlab/scenarios/test_batch_size_reproducibility.py` covers
exact stream equality across multiple batch sizes for SBB, FilteredBootstrap,
and every Parametric distribution, plus repeat-call determinism and seed
semantics.

**Evidence:** 42 R5 tests; `tests/simlab` 278 passed; focused regression 1438
passed, 2 skipped; E2E `ALL CHECKS PASSED`; D019 LOG_RETURN goldens unchanged;
only `parametric.py` changed in production.

**Remaining D021 question:** co-seeded generator independence / explicit RNG
ownership remains OPEN and is not part of D029.

## D030 — R6 final SimLab hygiene review (F8–F11): no production change

Evidence-driven audit of the four P3 hygiene findings. Conclusion: none is a
genuine defect warranting a production change now. `src/` is byte-identical to
the pristine tree; one documenting test added for F11.

F8 — shock_scale f32→f64 promotion. ACCEPTED (not a defect). Benchmarked the
per-batch `np.ascontiguousarray(shock_scale, float64)` against the apply kernel
across representative shapes: the promotion is 0.16% (Nreq=50,B=1000), 0.009%
(500,2000), 0.0017% (2000,5000) of apply cost — ~0.5–1.1 µs against 0.3–68 ms —
and apply(f32) vs apply(f64 pre-promoted) are equal within noise. Hoisting to
runner-build saves nothing measurable and would move a documented conversion off
its correct execution boundary. No change.

F9 — raw `kernel_ewma_vol_ratios`. ACCEPTED as an intentional public primitive.
It is exported in the module `__all__` (a public core-math surface), a
returns-inline alternative to the `_from_residuals` twin that the generators
use; no current caller, but exported, so NOT dead private code — do not delete.
The latent single-column `(T,1)` defect (F-only signature, D024 class) has no
active trigger (no caller); per the D024 "known twins" pattern the `_F64_2D_C`
overload is added WHEN a caller appears, not speculatively. Left open by design.

F10 — SBB stitch scalar `int32`. ACCEPTED as justified. `max_valid_length`
(= T−1) is not a loop counter (h/B are `_NB_INT64` because they bound loops); it
is an index-space bound used in modular index arithmetic
`wrapped = 1 + ((raw - 1) % max_valid_length)` on `int32` draw-pool indices.
T−1 fits int32 comfortably; unifying to int64 would force an int32→int64
promotion inside the index `%` with no correctness or performance benefit and
less consistency with the int32 index contract. No change.

F11 — empty required set / duplicate required ids. ACCEPTED as intentional and
correct.
- Empty required set is a LEGITIMATE no-shock operation: `_smoke_test_inputs`
  admits it explicitly (`size == 0: return`); `build_shock_universe` returns an
  empty universe and apply runs a `(0, B)` no-op end-to-end (verified). A
  no-simulatable-risk portfolio is a valid degenerate case; fail-fast would
  break it, so none is imposed (aesthetic-only consistency is not a reason).
- Duplicate required factors are VALID input: dedup collapses them to one shock
  column; each required maps to its shared unique shock. Guaranteed mapping:
  first-seen order of unique `(source_factor_id, shift_type_id)` pairs, with
  `required_to_shock_uix[i]` resolving required i to its pair. Verified:
  `[(10,1),(10,1),(10,2),(20,1)]` → K=3, uix `[0,0,1,2]`.
- Validation/normalization site: `build_shock_universe` (dedup + empty guard).
  No change; empty semantics pinned by `test_empty_required_set.py`, duplicate
  semantics already pinned by `test_shock_universe_dedup.py` (R4/D028).

Final coherence checks (all pass): no valid shift leaves output unwritten (all
five branches present in every fused/apply kernel); no silent dtype conversion
(the only conversions are explicit layout normalisations and the documented,
benchmark-negligible shock_scale promotion); discrete quantities remain strictly
typed (D026); batch-size invariance intact (D029, 4/4 distributions); the
`ScenarioGenerator` protocol (`iter_batches` only) remains sufficient.

Evidence: 3 R6 tests; tests/simlab 281; focused regression 1441 (+2 skipped);
changed file lint-clean; scope diff — src byte-identical (no production change).

## D031 — Scenario-model configuration seam (Historical, smallest increment)

Archaeology. Only the E2E demo constructs a generator, and it does so directly
(`HistoricalGenerator(matrix=..., horizon=...)`); the runner consumes the
`ScenarioGenerator` protocol and never selects a model. Constructor parameters
are genuinely heterogeneous — Historical(horizon); FHS(+decay,max_vol);
SBB(+total_scenarios,expected_block_size,seed); FilteredBootstrap(+total_scenarios,
decay,max_vol,seed); Parametric(cholesky,vols,factor_ids,total_scenarios,
distribution_kind,+dist params,seed) — and Parametric does not even take a
matrix. Data (matrix/cholesky) is market-data input, not user config. The credit
tier has a string-keyed `*/catalog.py` model catalog and Pydantic `schemas.py`
(`ConfigDict(extra="forbid")`), but nothing selects/constructs scenario
generators — so no existing abstraction already provides this seam.

Decision. Introduce the smallest typed seam in the SOLUTION tier (Pydantic lives
here, never in the NumPy/Numba core), mirroring the existing schema pattern:
`solutions/scenario_models/schemas.py` + `api.py`.
- Per-model config carries ONLY that model's parameters (no nullable blob).
  `HistoricalGeneratorConfig`: `model: Literal["historical"]` discriminator +
  `horizon: StrictInt = Field(gt=0)`. `extra="forbid"` rejects any parameter
  belonging to another model (total_scenarios, expected_block_size, seed, …);
  the `Literal` rejects an unknown model; `StrictInt` rejects float/bool.
- `ScenarioModelConfig` is a one-member alias today; it becomes
  `Annotated[HistoricalGeneratorConfig | <Next>, Field(discriminator="model")]`
  when a second model is wired — callers import the alias, so the promotion is
  source-compatible. No `GeneratorFactory`/`ModelRegistry` (a single isinstance
  dispatch in `build_scenario_generator` suffices for one model and extends by
  one branch per model).
- `build_scenario_generator(config, *, matrix) -> ScenarioGenerator` constructs
  the concrete generator; the generator's own `__post_init__` validation runs
  unchanged (no duplicated model logic, e.g. the `T - horizon` count check stays
  in the generator). Data is passed separately from config.

Why justified over existing abstractions: the parameter heterogeneity makes a
single shared config a nullable blob (a stated anti-goal); a per-model typed
config with a discriminator is the minimal typed selector, and it reuses the
established Pydantic `extra="forbid"` idiom rather than inventing a new one. A
discriminated union is deferred until the second model exists (single-member
unions are not expressible in Pydantic), but the shape is prepared.

Scope / migration. `src/` gains ONLY the new `solutions/scenario_models` package
— no existing production file changed. The demo is wired through the seam
(`build_scenario_generator(HistoricalGeneratorConfig(model="historical",
horizon=_HISTORY_HORIZON), matrix=matrix)`); it constructs the identical
generator, so demo output is unchanged (only the pre-existing non-deterministic
holiday-calendar diagnostic row order differs, per D026). Downstream still
consumes `ScenarioGenerator`; no parallel interface, no kernel/RNG/layout/shift
change. D021 (co-seeded independence) remains open and is unaffected.

Tests: `tests/solutions/scenario_models/test_scenario_model_seam.py` — valid
selection; unknown-model rejection; required-parameter and foreign-parameter
rejection; strict-int and positivity; seed-is-not-a-historical-parameter; build
returns HistoricalGenerator with the propagated horizon; the seam still runs the
generator's own validation; and direct-vs-seam equivalence in value, ordering,
dtype, layout, and count. Evidence: 23 seam tests; tests/simlab 281; focused
regression + solutions 1464 (+2 skipped); E2E `ALL CHECKS PASSED`, D019 goldens
unchanged; changed files lint-clean; scope diff — only the new package in src.

Deferred (next increments, not this one): wiring the stochastic generators
(SBB / FilteredBootstrap / Parametric) — at which point `ScenarioModelConfig`
becomes a real discriminated union and seed propagation gets a live test on a
seeded model (Historical is deterministic and has no seed to propagate).

## D032 — Seam extended to SBB (second real model; discriminated union now justified)

Second proof point for the D031 seam: the Stationary Block Bootstrap, wired with
no new abstraction. D031 boundaries re-verified first — `solutions.scenario_models`
imports only downward (pydantic, `market_data`, `simlab` core); no demo/edge/
sibling-solution leak; `HistoricalGeneratorConfig` stays model-specific.

New schema type. `SbbGeneratorConfig` (model-specific only, no nullable blob):
`model: Literal["sbb"]`; `horizon` and `total_scenarios` `StrictInt(gt=0)`;
`expected_block_size: float = Field(default=1.0, ge=1.0, allow_inf_nan=False)`
with a `before` validator rejecting bool (continuous mean block length per D026 —
int or float accepted, bool and non-finite rejected); `seed: StrictInt | None =
None` (strict optional; None = OS entropy, no reproducibility claim, D029).
Configuration-domain validation only — the generator's `__post_init__` stays
authoritative for runtime/model invariants (float64 matrix, T>=2, canonical
int/float persistence).

Discriminated union — now justified. With two genuinely implemented configs,
`ScenarioModelConfig` is promoted from a one-member alias to
`Annotated[HistoricalGeneratorConfig | SbbGeneratorConfig, Field(discriminator=
"model")]`. This is not speculative: the second real tag requires a real
discriminator. No registry/factory — `build_scenario_generator` dispatches with
one `isinstance` branch per model (two now), which is the minimal extension
point.

Construction API — matrix-as-input still viable (for SBB). SBB consumes a
`RiskFactorMatrix`, exactly like Historical (and FHS/FilteredBootstrap), so
`build_scenario_generator(config, *, matrix) -> ScenarioGenerator` extends
cleanly with an SBB branch and no signature change. matrix-as-input is viable for
four of five models. The exception is Parametric, whose data contract is
different (`cholesky_matrix`, `volatilities`, `available_factor_ids`, no matrix);
it is NOT solved here. When Parametric is wired, the build's data-passing — not
the config union — will need generalisation (per-model data arguments or a small
data protocol); baking `matrix=` as a universal construction input is the one
place the current signature would strain. Flagged, deferred.

Equivalence & invariance. Direct construction vs seam are identical in scenario
count, ordering, values, dtype (float32), and layout (F-contiguous). A seeded SBB
built through the seam reproduces the direct stream exactly AND stays batch-size
invariant across bs∈{1,7,full} (D029 preserved through the seam). Cross-model
leakage is rejected both directions: SBB config rejects `decay_factor`,
`max_vol_ratio`, `distribution_kind`, `student_t_df`, `vg_nu`, `cholesky_matrix`;
Historical config rejects `total_scenarios`/`expected_block_size`/`seed`. No
optional `df=None`/`decay=None`/… fields anywhere.

Scope. Only `scenario_models/schemas.py` and `api.py` changed in `src/`; new test
`tests/solutions/scenario_models/test_sbb_seam.py`. No change to the
`ScenarioGenerator` protocol, SBB kernels/RNG, D029, shift semantics, layout
contracts, P3, Parametric, FilteredHistorical, or FilteredBootstrap. The demo is
untouched this increment (still Historical through the seam). Evidence: 45 SBB +
23 Historical seam tests (68); tests/simlab 281; focused regression + solutions
1509 (+2 skipped); E2E `ALL CHECKS PASSED`, D019 goldens unchanged; changed files
lint-clean.

Remaining architectural concern: the Parametric data-contract divergence above —
to be resolved when (and only when) Parametric is wired. Third model deferred.

## D033 — Seam extended to FilteredHistorical (third model; union stays clean)

Third real model through the D031/D032 seam, proving the discriminated union
absorbs materially different model-specific parameters while keeping the shared
`RiskFactorMatrix` construction-data contract.

New schema type. `FilteredHistoricalGeneratorConfig` (model-specific only):
`model: Literal["filtered_historical"]`; `horizon: StrictInt(gt=0)`;
`decay_factor: float = Field(default=EWMA_DECAY_FACTOR, gt=0.0, lt=1.0,
allow_inf_nan=False)` — the EWMA memory lambda's valid open unit interval;
`max_vol_ratio: float = Field(default=MAX_EWMA_VOL_RATIO, ge=1.0,
allow_inf_nan=False)` — the vol-amplification cap. Both reject bool (before
validator) and non-finite. Defaults import the repository constants so seam
construction with defaults reproduces the generator's own defaults exactly.
No `seed` / `total_scenarios` / `expected_block_size` / `distribution_kind`
fields; `extra="forbid"` rejects them.

Config-domain vs generator validation. FHS's `__post_init__` validates the
matrix (float64), horizon (strict positive int) and history length, but does
NOT bound `decay_factor` or `max_vol_ratio` at all. The seam therefore adds the
missing config-domain fail-fast (lambda in (0,1); cap >= 1.0; finite; non-bool)
without touching numerical production code and without duplicating the EWMA/AR(1)
math — the generator stays authoritative for matrix/history invariants.

Architecture — matrix-as-input confirmed for this model. FHS consumes a
`RiskFactorMatrix` exactly like Historical and SBB, so `build_scenario_generator
(config, *, matrix)` extends with a third `isinstance` branch and no signature
change. The union grows by one tag:
`Annotated[HistoricalGeneratorConfig | SbbGeneratorConfig |
FilteredHistoricalGeneratorConfig, Field(discriminator="model")]`. No registry,
no factory, no nullable data blob, no data protocol. Per-model cost stayed
linear: one config class + one union tag + one build branch + one test file.

Equivalence & determinism. Direct vs seam identical in scenario count, ordering,
values, dtype (float32) and layout (F-contiguous), for explicit and default
parameters. FHS is deterministic (no seed), so the seam-built stream is
repeatable run to run. Cross-model leakage rejected both directions.

Scope. Only `scenario_models/schemas.py` and `api.py` changed in `src/`; new test
`tests/solutions/scenario_models/test_filtered_historical_seam.py`. No change to
the `ScenarioGenerator` protocol, any numerical kernel, RNG, D029, shift
semantics, layout contracts, P3, Parametric, or the construction-data API. Demo
untouched. Evidence: 40 FHS + 45 SBB + 23 Historical seam tests (108);
tests/simlab 281; focused regression + solutions 1549 (+2 skipped); E2E
`ALL CHECKS PASSED`, D019 goldens unchanged; changed files lint-clean.

Seam verdict for three heterogeneous configs: CLEAN. Genuinely different
parameter sets coexist with no blob, no leakage, no factory. The single open
architectural question is unchanged and deferred: Parametric's construction data
(`cholesky`, vols, factor ids — no matrix) will force a decision on the
construction-data API, NOT on the config union. Parametric remains unwired by
deliberate choice.



## D034 — Parametric through the seam: heterogeneous construction data (typed data union + nested distribution union)

The proof point for heterogeneous construction data. Parametric consumes
`(available_factor_ids, cholesky_matrix, volatilities)` plus distribution
parameters — not a `RiskFactorMatrix`. Resolved by separating two concerns: a
typed construction-data object and a nested distribution config union, wired
through one build entry point. No production change to `ParametricGenerator`.

**Archaeology.** Parametric has no production/demo caller — only two test
fixtures construct it, and there is no in-repo producer of Cholesky/vols. The
three data inputs are dimensionally coupled by K and form one coherent bundle.
No existing typed structure represents them (`RiskFactorMatrix` is a T×N level
time-series — unrelated). The generator resolves Cholesky columns / vols / GMM
multipliers by `searchsorted` on `available_factor_ids`, so sorted-ascending +
unique ids are an implicit, previously-unenforced requirement.

**Construction data — `ParametricRiskData` (SimLab core, new).**
`simlab/scenarios/parametric_data.py`: `@dataclass(frozen=True, slots=True)` over
NumPy (no Pydantic in core). Owns DATA invariants only, rejected never coerced:
`factor_ids` int32 1-D C-contig non-empty sorted-ascending unique;
`cholesky_matrix` float64 2-D (K,K) square C-contig finite; `volatilities`
float64 1-D length-K C-contig finite strictly-positive; optional
`crisis_vol_multipliers` float64 1-D length-K C-contig finite strictly-positive,
aligned with `factor_ids` (added by D092). Placement proven from
architecture, not origin: SimLab already owns typed-array data contracts
(`ScenarioSetBatch`), the contract's shape is dictated by the SimLab generator's
`searchsorted` need, and keeping it in SimLab avoids forcing a market_data
dependency or a cycle. Rejected placing it in `market_data` merely because the
values originate there.

**Configuration — nested distribution discriminated union.** `schemas.py`:
`ParametricDistributionConfig = Annotated[StandardNormalDistribution |
LaplaceDistribution | StudentTDistribution{df} | VarianceGammaDistribution{nu} |
RegimeSwitchingDistribution{prob_crisis, crisis_vol_multipliers}, Field(
discriminator="kind")]`. Each variant carries only its own parameters, so
invalid combinations are unrepresentable (standard_normal cannot carry `df`;
student_t cannot omit it; a foreign parameter is rejected by `extra="forbid"`).
`ParametricGeneratorConfig(model="parametric", total_scenarios StrictInt gt=0,
seed StrictInt|None, distribution)`. Outer union grows to four tags. Q8 answer:
one generator class, but the distribution kinds are best modelled as separate
config variants.

**Strict real typing.** Lax Pydantic `float` admits strings ("2.0"->2.0). A
`_require_real` before-validator (reject bool and non-numeric incl. str) plus
`allow_inf_nan=False` gives: real int|float accepted, bool/str/NaN/±inf rejected,
each parameter's domain preserved (df>2, nu>0, prob_crisis in [0,1];
`crisis_vol_multipliers` is a strictly positive scalar or the `"per_factor"`
declaration, with per-factor values supplied by `ParametricRiskData` under D092
(amended by D092)). NOTE: D032/D033 use lax `float` +
reject-bool only, so they still admit strings; those decisions are LOCKED and
were not retrofitted. Inconsistency flagged for a future harmonization.

**Build API — B (typed data union), matrix->data.**
`build_scenario_generator(config, *, data: RiskFactorMatrix | ParametricRiskData)`.
Confirmed as the smallest long-term seam: one entry point, one typed union, no
nullable blob. `_require_matrix` / `_require_parametric_data` enforce fail-fast
pairing (Historical/SBB/FHS require a matrix; Parametric requires the data
object; a mismatch raises TypeError at the solution boundary). `_build_parametric`
maps the nested distribution variant onto the generator's flat kwargs; the
generator re-validates unchanged. Rejected: nullable data args (A — forbidden,
invalid combos representable), a `ScenarioData` protocol (C — disproven: the two
data types share no behavioural interface, so a protocol abstracts nothing; a
union + isinstance is strictly smaller), and separate per-data build functions
(fragments the entry point).

**Validation split (no duplicate validation).** `ParametricRiskData` owns data
invariants; `ParametricGenerator` stays authoritative for model semantics
(total_scenarios, seed, distribution selection and distribution parameters).
Per-factor multiplier values are a `ParametricRiskData` invariant of shape `(K,)`,
aligned with `factor_ids`, finite and strictly positive; the declaration/presence
pairing is a composition check under D092 (amended by D092). The generator's own
array checks remain because
it is independently callable (two test fixtures construct it directly); the
overlap is intentional tiered defense, as with StrictInt vs require_positive_int
in D031–D033, not accidental duplication.

**Migration.** Additive 4th union member; the three existing configs untouched.
The build data keyword renamed `matrix`->`data`; the demo and the three existing
seam test files took a mechanical kwarg update at the build call only
(assertions and coverage unchanged; direct-generator `matrix=` constructions kept).
D019 goldens unchanged (only the known US/NL calendar-order non-determinism).
NOTE: because the rename couples `api.py` to its callers, a later dump sync
landed the src rename without the caller update and broke the three seam test
files + the E2E demo (`unexpected keyword argument 'matrix'`); re-applying the
same mechanical `matrix=`->`data=` migration to the four callers restored green.
Src and caller changes must land together.

**Equivalence & determinism.** Direct vs seam identical in scenario count,
ordering, values, dtype (float32) and layout (F-contiguous), across
standard_normal / student_t / variance_gamma / regime_switching. Seeded seam
construction preserves the D029 batch-size invariant (bs1 == bs7 == bs10).

**Evidence.** New `tests/solutions/scenario_models/test_parametric_seam.py`.
Seam tests 170 passed; tests/simlab 281; focused regression + solutions 1611
passed / 2 skipped; E2E `ALL CHECKS PASSED`; changed-file lint clean; scope diff
against the current dump shows only the intended files.

**Out-of-scope drift flagged.** The project dump was updated mid-phase again;
one drift was isolated to `dependence_stress/{benchmarks,engine,types}.py`,
unrelated to this work. On that updated dump, three `dependence_stress` test
modules (`test_copulas`, `test_invariants`, `test_tails`) fail COLLECTION with a
bare `from canonical_measures import ...` ImportError — pre-existing on the
pristine dump, not caused by this change. FilteredBootstrap deliberately NOT
added (Parametric was the requested proof point).

**Status: SUPERSEDED IN PART by D088 (2026-09-25).** `ParametricRiskData = (factor_ids, cholesky_matrix, volatilities)` is no longer the complete semantic definition of a parametric world, and `hash_parametric_risk_data` over those three arrays is no longer the complete data identity. Under D088 every parametric world column is keyed `(d, τ, h)` — resolved node `rf_id`, concrete chart, and the world-declared horizon in the world's declared step unit — and a marginal scale carries meaning only for that key; an untagged `σ_d` is not a world variable. Where several charts of one node are supplied, the data declares Case 1 (jointly declared shock-space random variables with an explicit joint law) or Case 2 (alternative representations of one underlying level process with a common scenario-specific start/end state); no implicit conversion exists. The data digest distinguishes column keys and the representation declaration. Unaffected: the typed construction-data object in SimLab core, the nested distribution union (except for the per-factor multiplier alternative, which is replaced by D092), the single build entry point, the data-invariant ownership (sorted-unique keys, finite square Cholesky factor, finite strictly positive scales) and the model/config/data pairing.

**Status: AMENDED IN PART by D090, D091 and D092 (2026-09-29).** D090 adds the effective-period fields and D091 adds `chart_ids` and the `R_ii = 1` invariant to the `ParametricRiskData` data invariants (additive). D092 moves per-factor crisis multipliers from `RegimeSwitchingDistribution` to `ParametricRiskData`; the three clauses above marked "D092" record that change. Invariant ownership is unchanged.

## D035 — Seam extended to FilteredBootstrap (fifth and final model; seam proven complete)

The fifth and final scenario model through the D031–D034 configuration seam. It
required no new abstraction: one config class, one union tag, one builder
branch, one test file — the same linear per-model cost as D032/D033. The seam is
now complete for every SimLab scenario generator.

**Exact config (verified against the constructor, not inferred).**
`FilteredBootstrapGenerator(matrix: RiskFactorMatrix, total_scenarios: int,
horizon: int = 1, decay_factor: float = EWMA_DECAY_FACTOR, max_vol_ratio: float =
MAX_EWMA_VOL_RATIO, seed: int | None = None)`. `FilteredBootstrapGeneratorConfig`
mirrors this exactly: `model: Literal["filtered_bootstrap"]`; `horizon`,
`total_scenarios` `StrictInt(gt=0)`; `decay_factor float default EWMA_DECAY_FACTOR,
gt=0 lt=1 allow_inf_nan=False`; `max_vol_ratio float default MAX_EWMA_VOL_RATIO,
ge=1 allow_inf_nan=False`; both reject bool via the before-validator;
`seed StrictInt | None = None`. It is the SBB discrete/seed shape (D032) plus the
FHS continuous shape (D033) — no `expected_block_size`, no distribution fields.

**Typing follows D032/D033, not D034.** `decay_factor`/`max_vol_ratio` are the
identical FHS parameters, so they use the established D033 lax-float +
reject-bool + `allow_inf_nan=False` pattern (rejects bool/NaN/±inf; the known
string-coercion inconsistency vs D034's strict-real is NOT harmonized here — that
remains a separate future decision). Discrete fields use `StrictInt` (reject
bool/float/str). The generator does not bound `decay_factor`/`max_vol_ratio`
(confirmed in `__post_init__`), so the seam adds that config-domain fail-fast,
exactly as D033 did.

**Matrix-as-input remains valid; the typed `data` union is sufficient.**
FilteredBootstrap consumes a `RiskFactorMatrix` like Historical/SBB/FHS. The
existing `build_scenario_generator(config, *, data: RiskFactorMatrix |
ParametricRiskData)` absorbed it with one `isinstance` branch and
`_require_matrix` fail-fast pairing (FilteredBootstrap + `ParametricRiskData` ->
TypeError). Parametric stays the only heterogeneous-data exception; no
`ScenarioData`, registry, factory, or new abstraction was needed — the default
assumption held. Five explicit builder branches remain a plain dispatch.

**Validation split preserved (D031–D034).** Config owns strict types,
positivity, finiteness, discriminator, foreign-field rejection; builder owns
data pairing; generator's `__post_init__` stays authoritative for matrix float64
(D017), positive-int (D026), and history-length (>=3) invariants. No numerical
logic moved into the schema or builder; no generator validation removed.

**Equivalence & D029.** Direct vs seam identical in scenario count, ordering,
values, dtype (float32) and layout (F-contiguous). Seeded seam construction is
deterministic and preserves the D029 batch-size invariant (bs1 == bs7 == full);
different seeds differ. D021 untouched (still OPEN); no `SeedSequence`, no
`default_rng` ownership change.

**Evidence.** FilteredBootstrap seam tests 54 passed; all scenario-model seam
tests 224; tests/simlab 281; focused regression + solutions 1384 passed / 2
skipped; E2E `ALL CHECKS PASSED`; D019 LOG_RETURN goldens preserved (only the
known US/NL calendar-order non-determinism); changed-file lint clean; src diff
scope exactly `scenario_models/schemas.py` + `api.py` + new
`test_filtered_bootstrap_seam.py`.

**Pre-existing unrelated failure (not this increment).** On the current pristine
dump, `tests/dependence_stress/test_invariants.py` fails COLLECTION (bare
`from canonical_measures import ...` ImportError family, unchanged from the D034
note, now one module rather than three). Untouched here.

**Seam complete.** ScenarioModelConfig = {Historical, Sbb, FilteredHistorical,
Parametric, FilteredBootstrap}; matrix-backed models -> RiskFactorMatrix,
Parametric -> ParametricRiskData. No further scenario models outstanding.


## D036 — Final scenario-model seam audit: SEAM FROZEN

Adversarial whole-system audit of the config -> data -> builder -> generator
seam after D031–D035. No production defect found; one additive cross-model
completeness test added. The seam is declared FROZEN.

**Correction (2026-09-22, SimLab P0).** "No production defect found" did not hold: shock columns were assigned to the wrong factors for every non-involutive source order (the four kernel-based generators) and for every non-identity order (`parametric`). No audited fixture used such an order. The defect is fixed at the consumers of the shock plan; the contract is recorded in the D030 amendment. The seam API is unchanged and remains FROZEN.

**Architecture (final).**
```
ScenarioModelConfig (discriminated union on `model`, 5 tags)
  Historical | Sbb | FilteredHistorical | FilteredBootstrap | Parametric
        |                                           (Parametric.distribution:
        |                                            nested union on `kind`)
        v
build_scenario_generator(config, *, data: RiskFactorMatrix | ParametricRiskData)
  5 explicit isinstance branches; _require_matrix / _require_parametric_data pair
        v
ScenarioGenerator (Protocol) -> iter_batches(...) -> Iterator[ScenarioSetBatch]
```

**Model / config / data / generator matrix.**
| model | config | data | generator | callers |
|---|---|---|---|---|
| historical | HistoricalGeneratorConfig | RiskFactorMatrix | HistoricalGenerator | demo + tests |
| sbb | SbbGeneratorConfig | RiskFactorMatrix | StationaryBlockBootstrapGenerator | tests |
| filtered_historical | FilteredHistoricalGeneratorConfig | RiskFactorMatrix | FilteredHistoricalGenerator | tests |
| filtered_bootstrap | FilteredBootstrapGeneratorConfig | RiskFactorMatrix | FilteredBootstrapGenerator | tests |
| parametric | ParametricGeneratorConfig | ParametricRiskData | ParametricGenerator | tests |

**Validation-ownership matrix (no ambiguous duplication).**
- Pydantic config: discriminator, strict discrete ints (StrictInt), continuous
  domains (gt/lt/ge), finiteness (allow_inf_nan=False), bool rejection,
  foreign-field rejection (extra="forbid"), nested distribution variant selection.
- ParametricRiskData: intrinsic data invariants only — dtype, ndim, layout
  (C-contig), square, K-coupling, sorted+unique factor_ids, finiteness, vol
  positivity. No model semantics.
- Builder: config/data pairing only (matrix-backed -> RiskFactorMatrix;
  parametric -> ParametricRiskData); TypeError on mismatch.
- Generator __post_init__: matrix float64 (D017), positive-int (D026), history
  length, model/runtime invariants. Overlap with config on positivity/finiteness
  is intentional independent tier defense: the generators are independently
  constructed by direct callers (tests), so their guards are not redundant with
  the config's boundary checks. No rule was removed.
- Kernels: none — dumb math only.

**Adversarial findings.**
1. Config/data separation — SOUND. Full pairing matrix fails fast: every
   matrix-backed config rejects ParametricRiskData; parametric rejects a
   RiskFactorMatrix. No invalid (config, data) pair is representable.
2. Validation ownership — SOUND. Each rule has a clear owner; the config/generator
   overlap is deliberate tier defense, not accidental duplication.
3. Strict typing — GENUINE, KNOWN, NON-BLOCKING INCONSISTENCY. D032/D033/D035
   continuous params (`expected_block_size`, `decay_factor`, `max_vol_ratio`)
   use lax Pydantic `float`, which coerces a numeric STRING ("0.9" -> 0.9); D034
   uses strict-real, which rejects it. Confirmed empirically. It is not a
   fail-fast hole — no invalid VALUE passes (domain, bool, NaN, ±inf are all
   rejected on both paths); only a string REPRESENTATION of a valid value is
   coerced on the lax path. It is a uniformity gap against "typed to the nth
   degree." Per D035 and the audit scope it is NOT harmonized here; it is
   recorded as a deferred, separate future decision (harmonize all continuous
   scenario-config params to strict-real). Does not block FREEZE.
4. Parametric distribution union — SOUND. Five variants, no nullable
   cross-variant fields; each rejects foreign params (student_t+vg_nu,
   standard_normal+df) and missing/invalid values.
5. ParametricRiskData — SOUND. Holds only the three arrays + their invariants;
   docstring and fields carry no model semantics. Location
   `simlab/scenarios/parametric_data.py` is D001-correct (SimLab core owns the
   contract; solution/tests import it downward).
6. Builder complexity — SOUND. Five explicit isinstance branches remain the
   smallest, clearest design; no registry/factory warranted.
7. Protocol sufficiency — SOUND. All five nominally subclass ScenarioGenerator
   and share the identical iter_batches signature and Iterator[ScenarioSetBatch]
   return; no model-specific type leaks downstream. (The Protocol is not
   @runtime_checkable — an intentional detail; dispatch is on config type, not
   generator type, so no runtime isinstance on the protocol is needed. Left
   unchanged.)
8. Boundary purity — SOUND. The builder passes only primitives and typed NumPy
   arrays into generators (config.<attr>, pdata.<array>); no Pydantic object,
   dict, or model-name string is handed to a generator constructor, and no
   string reaches a kernel (the distribution_kind string dies at the generator's
   dispatch, above the kernel).
9. Bypass audit — SOUND. Generators are constructed only inside
   `build_scenario_generator` (api.py) in production; the E2E demo uses the seam;
   all other direct constructions are in tests exercising the generators
   themselves (legitimate).

**Genuine defects vs accepted design choices.** Zero genuine defects. One
accepted-and-deferred inconsistency (finding 3, strict-vs-lax typing), already
flagged in D034/D035, tracked for a future dedicated decision. The
non-runtime_checkable Protocol (finding 7) is an accepted design choice.

**Change made.** Test-only, additive: new
`tests/solutions/scenario_models/test_seam_completeness.py` asserting, across all
five models as one matrix: tag discrimination, the full correct/incorrect
config/data pairing matrix (closing the previously-implicit SBB+ParametricRiskData
and FHS+ParametricRiskData cases), and uniform ScenarioSetBatch yield. No
production change; `src/` byte-identical to the pristine dump.

**Evidence.** completeness audit 21 passed; all scenario-model seam tests 245;
tests/simlab 281; focused regression + solutions 1405 passed / 2 skipped; E2E
`ALL CHECKS PASSED`; D019 goldens preserved; changed-file lint clean; scope diff
= the one new test file only. Pre-existing unrelated failure unchanged:
`tests/dependence_stress/test_invariants.py` fails COLLECTION (bare
`from canonical_measures import ...` ImportError), not touched by this work.

**Conclusion — SCENARIO MODEL SEAM = FROZEN.** No invalid combinations, no
hidden ownership ambiguity, no accidental abstraction, no bypassed boundary, no
model-specific leakage, no unexplained duplication. The only open item is the
deferred strict-typing harmonization (finding 3), which is a separate future
decision and does not affect the freeze. D021 (RNG ownership / co-seeded
independence) remains OPEN and is the next increment — not started here.

**Status: SUPERSEDED IN PART by D088 (2026-09-25).** The frozen seam signature `iter_batches(required_shift_type_id, source_factor_ids, shock_scale, batch_size)` is no longer authoritative for the proxy architecture. Under D088 a generator request is a set of world-column keys `(d, τ, h)` and a generator output is batches of those world columns; the generator is responsible for columns that satisfy the world-column contract, the proxy operator for `X_A = Σ_j β_Aj Y_j`, and no loadings or per-required-row arrays cross the seam. Unaffected and still FROZEN: the `ScenarioModelConfig` discriminated union (5 tags), `build_scenario_generator` with its explicit model/data pairing, the validation-ownership matrix, the D030 permutation contract and the SimLab P0 correction recorded above.



## D037 — RNG ownership / co-seeded independence (methodology RESOLVED; implementation DEFERRED)

Final canonicalization of D021. Methodology only; no production edits, no tests.
Full memo: `D037_RNG_OWNERSHIP_DECISION.md`. This entry supersedes the earlier
D037 draft: the derived-stream formula is now the direct
`SeedSequence(seed, spawn_key=(stream_id,))`, not `spawn(stream_id)[stream_id-1]`.

**Current architecture (re-verified).** SBB, FilteredBootstrap, Parametric each
rebuild `np.random.default_rng(self.seed)` at the top of every `iter_batches`,
draw all randomness upfront at full run size, then slice into batches (D029
mechanism). No `SeedSequence` anywhere in `src/`. Historical/FilteredHistorical
are deterministic. `seed=int` -> deterministic restart + batch-invariant +
cross-instance IDENTICAL; `seed=None` -> entropy, no reproducibility.

**Gap (D021).** `seed` conflates reproducibility with stream identity, so streams
cannot be both reproducible and independent; co-seeded engines/replicas/workers
are perfectly correlated. Genuine but LATENT (no co-seeded consumer today).

**Canonical semantics.** Two orthogonal value ints on the stochastic generators:
`seed` = reproducibility root (public name UNCHANGED; `None` = OS entropy) and
`stream_id: int >= 0` = explicit stream identity (default 0). Derivation:

```
seed is None    -> default_rng()                                            # entropy, unchanged
stream_id == 0  -> default_rng(seed)                                        # LEGACY STREAM (exact back-compat anchor)
stream_id  > 0  -> default_rng(SeedSequence(seed, spawn_key=(stream_id,)))  # DERIVED STREAM (directly addressable)
```

Contract: for fixed model, construction data, model parameters, seed, stream_id,
scenario_count, the concatenated stream is deterministic and batch-invariant;
same seed + same stream_id -> identical; same seed + different stream_id ->
practically independent; stream_id=0 -> exact legacy stream; seed=None -> no
reproducibility (stream_id irrelevant).

**Empirically confirmed.** `spawn_key=(stream_id,)` is deterministic and directly
addressable (stream 7 needs no enumeration of 1..6); `stream_id=0` ==
`default_rng(seed)` EXACTLY while `spawn_key=(0,)` DIFFERS from it (so the stream-0
special case is required, not incidental); different stream_ids differ;
construction order does not matter (7-then-1 == 1-then-7 per stream); same
`(seed, stream_id)` yields the SAME stream across DIFFERENT models (so identity is
NOT auto-independent across consumers — callers pick distinct stream_ids).

**Independence claim.** Reproducible + practically independent under NumPy's
SeedSequence/PRNG design. NOT provably-disjoint state spaces; independence is
never asserted merely because two arrays differ.

**Rejected.** `seed`->independent (destroys restart/goldens); enumeration
`spawn(k)[k-1]` (couples identity to spawn count; not addressable); construction-
order `spawn()` (identity depends on build order/scheduling — parallelism bug);
live `Generator` field (mutable/consumed -> breaks restart+D029, non-value,
non-serializable); model tag in the derivation by default (would move the
stream_id=0 legacy anchor — complexity unjustified; cross-consumer independence
comes from distinct stream_ids); renaming `seed`->`master_seed` (needless compat
surface).

**Implementation (deferred; describe only).** Add `stream_id: int = 0` to the
three stochastic generators; replace `default_rng(self.seed)` with a shared
`build_rng(seed, stream_id)` helper implementing the derivation above; downstream
draw-all-upfront/slice unchanged. Helper lives in the SimLab scenarios layer
(e.g. `simlab/scenarios/rng.py` or `core_math/`), NOT `riskquant.jit` (RNG
orchestration is not a Numba/kernel concern, D003). `stream_id` boundary is
D026-consistent StrictInt (int/np.integer accepted; bool/negative/float/string
rejected; canonical int persisted).

**D029 preservation.** Structural: the derivation changes only RNG
initialization; draw order/count, distribution algorithm, and slice order are
untouched, and `batch_size` never enters the draw path. `stream_id=0` is exactly
`default_rng(seed)`, so existing exact-equality tests pass with NO golden
re-baseline. A future increment adds batch-invariance coverage for `stream_id>0`
without pinning absolute values.

**Verdict — METHODOLOGY RESOLVED / IMPLEMENTATION DEFERRED.** The current `seed`
semantics are correct and preserved; the single-seed API simply cannot express
co-seeded independence. Contract fixed here; implement additively when the first
co-seeded/parallel/ensemble consumer appears. D036 seam remains FROZEN and
untouched. Next: D038 — continuous-config strictness harmonization.

## D038 — Continuous configuration strictness harmonization (methodology RESOLVED; implementation DEFERRED)

Archaeology + decision memo only; no production edits, no test edits. Defines the
canonical continuous-numerical-input contract for the scenario-model config seam.
Full memo: `D038_CONTINUOUS_CONFIG_STRICTNESS.md`.

**Problem.** Two policies coexist. Lax (D032/D033/D035:
`expected_block_size`, `decay_factor`, `max_vol_ratio`) is plain `float` +
reject-bool, so it silently coerces `"0.9"`, `Decimal`, and `Fraction` to float.
Strict (D034: Parametric `df`/`nu`/`prob_crisis`/`crisis_vol_multipliers`) uses
`isinstance(x, (int, float))`, which is itself incoherent across NumPy scalars —
it accidentally accepts `np.float64` (a `float` subclass) but rejects `np.int64`
and `np.float32`. `numbers.Real` is not a usable criterion (admits `bool` and
`Fraction`, rejects `Decimal`).

**Canonical policy.** One coherent strict-real rule for every continuous
scenario-config parameter: a valid input is a finite real-valued scalar of the
float family, in domain, stored as Python `float`.

```
accept : int, float, np.integer, np.floating   (NOT bool)
reject : bool, str, Decimal, Fraction, complex, None, NaN, +inf, -inf
store  : Python float ; domain via existing Field bounds, unchanged
```

* **Decimal / Fraction — REJECT** (explicit): not float-family; `Decimal` is
  outside `numbers.Real`; they never arise from JSON/Polars/Excel edges; a caller
  converts at the edge (`float(x)`).
* **Integer-valued reals — ACCEPT** for all params on their mathematics
  (`expected_block_size=1`, `max_vol_ratio=8`, `df=3`, `nu=1`, `prob_crisis=0|1`),
  stored as float. `bool` still rejected (accepting `True` as `1.0` is type
  confusion / fail-fast violation).
* **NaN/±inf — REJECT** via `allow_inf_nan=False` (verified sufficient alongside
  the before-validator).
* **Boundary (D001/D006).** EDGE owns representation parsing (a text `"0.9"` is
  parsed to a number at ingest — precedent: credit tier's explicit `float(v)`);
  SOLUTION validates an already-typed real and does not coerce representations.
  JSON numbers arrive typed, so strict solution typing costs the normal path
  nothing.

**Mechanism.** Smallest correct option: correct the existing `_require_real`
before-validator to the float-family predicate `isinstance(value, bool) or not
isinstance(value, (int, float, np.integer, np.floating))`, and apply that single
validator to all seven continuous fields, replacing the two `_reject_bool_real`
validators; keep every `float`/`Field(..., allow_inf_nan=False)`/domain bound
as-is. Change confined to `scenario_models/schemas.py`; no `api.py`/generator/
kernel/union/`ParametricRiskData` change. D036 validation-ownership split
preserved (config owns typing + finite + domain; generator keeps model/runtime
invariants; no math moves into config).

**Backward compatibility (measured).** No existing test passes a
`str`/`Decimal`/`Fraction` to any continuous field expecting success (lax fields
receive only numeric literals; Parametric already asserts string rejection), so
rejecting them breaks zero tests — a safe narrowing. Widening the strict path to
accept `np.int64`/`np.float32` is a safe loosening on genuine numeric types.
Input typing only; accepted values still store as the same Python `float`, so
D019 goldens and all D029/equivalence tests are unaffected — no golden
re-baseline.

**Supersession.** D038 supersedes the continuous-input typing portions of D032,
D033, and D035, and corrects the NumPy-scalar asymmetry of D034's strict-real.
Those decisions are NOT rewritten; this entry records the supersession. Scope is
the scenario-model seam only; extending the same contract to the credit-tier
configs (currently bare lax `float`) is a separate future decision, not touched
here.

**Verdict — NO PRODUCTION CHANGE THIS INCREMENT; METHODOLOGY RESOLVED /
IMPLEMENTATION DEFERRED.** The architecture is not unsafe today (both policies
reject bool and non-finite; lax coercions yield in-domain values, not corrupt
ones), so no immediate change is required. The contract is fixed here;
implementation is a later mechanical increment. D036 seam remains FROZEN; D037
remains methodology RESOLVED / implementation DEFERRED.

**Implementation status — CLOSED (production-closure pass).** D038 is implemented
in `scenario_models/schemas.py`: one canonical `_require_real` (float-family
predicate: accepts int/float/np.integer/np.floating; rejects bool/str/Decimal/
Fraction/complex/None) defined once at module top and applied to all nine
continuous parameters, replacing the two ad-hoc bool-only validators and the
duplicate helper; `allow_inf_nan=False` and every `Field` domain bound unchanged;
values stored as Python `float`. `crisis_vol_multipliers` is a scalar or the
`"per_factor"` declaration; the sequence form is removed by D092 (amended by D092).
Evidence: new
`tests/solutions/scenario_models/test_continuous_typing.py` 160 passed / 2
skipped; seam suite 405 passed / 2 skipped; tests/simlab 281; focused regression
+ solutions 1565 passed / 4 skipped; E2E `ALL CHECKS PASSED`; D019 goldens
preserved (no re-baseline); changed-file lint clean; scope diff = `schemas.py`
plus the new test. D037 remains deferred: archaeology confirms no production or
demo caller constructs co-seeded independent engines. See
`SIMLAB_PRODUCTION_CLOSURE.md`.



**Status: AMENDED IN PART by D092 (2026-09-29).** The `crisis_vol_multipliers` sequence form is removed; the typing rule applies to the scalar form. The evidence counts above are historical.

## D039 — seed domain bounded at the solution boundary (amends D032/D035 seed contract); D021 closed as superseded

Raised by the adversarial production audit (`SIMLAB_ADVERSARIAL_PRODUCTION_AUDIT.md`).
A new decision is recorded because the audit proved the `seed` *contract* itself
had to change, not merely its implementation.

**Defect.** `seed = -1` parsed successfully, built a generator successfully, then
crashed during `iter_batches` inside `numpy/random/bit_generator.pyx`
(`ValueError: expected non-negative integer`). `StrictInt` carried no lower bound,
so a malformed run failed late instead of at the solution boundary — a fail-fast
violation.

**Proof no behaviour is lost.** Verified on the pristine tree that a negative seed
never produced a working run (parse OK -> build OK -> iterate CRASH), and negative
seeds appear in no caller in the repository. The fix changes only WHERE the error
is raised.

**Fix.** `Field(ge=0)` on `seed` in `SbbGeneratorConfig`,
`FilteredBootstrapGeneratorConfig` and `ParametricGeneratorConfig`
(`scenario_models/schemas.py`). `seed=0` and `seed=None` remain valid; every valid
seed's stream is bit-identical; D019 goldens unchanged. This also aligns the
boundary with `SeedSequence`'s own domain, so a future D037 `stream_id` needs no
further widening.

**Amendment to locked decisions (surfaced, not smuggled).** This tightens the
`seed` domain recorded in D032 and D035. One existing test,
`test_sbb_seam.py::test_seed_accepts_int_or_none[-3]`, asserted parse-acceptance of
a negative seed — it encoded the latent defect and was narrowed to `[7, 0, None]`
with an explanatory comment; rejection is asserted in the new
`tests/solutions/scenario_models/test_rng_semantics.py`.

**RNG semantics pinned (new regression suite, 72 tests).** Measured and now locked:
restart-identical; co-seeded instances identical (NOT independent); `seed=None`
re-randomizes per `iter_batches` call; batch-invariant across
{1,2,3,5,7,11,S-1,S,S+50} for every Parametric distribution branch including the
vector regime form; construction-order independent; no dependence on NumPy global
state; **bitwise identical across fresh OS processes** (sha256 of the stream) and
across batch sizes. Reproducibility is bitwise within a pinned environment
(NumPy 2.4.4 / CPython 3.12.3); cross-NumPy-version bitwise identity is NOT
guaranteed and must not be promised.

**D037 re-evaluated: CORRECTLY DEFERRED.** No production or demo caller constructs
co-seeded independent engines; the absence of independent streams is a capability
gap, not a production limitation. Methodology unrevised.

**D021: CLOSED — SUPERSEDED BY D037.** Methodology RESOLVED (D037); implementation
DEFERRED; trigger = first co-seeded / parallel / ensemble / replica consumer. The
stale OPEN state is retired.

**Evidence.** New RNG suite 72 passed; seam suite 476 passed / 2 skipped;
tests/simlab 281; focused regression + solutions 1636 passed / 4 skipped; E2E
`ALL CHECKS PASSED`; D019 goldens preserved; changed-file lint clean; scope diff =
`schemas.py` + the new RNG test + one superseded test case.

**Remaining gap (not closed here).** Run identity: nine categories of
result-affecting input sit outside the scenario config (historical values and
column order, ParametricRiskData arrays, proxy/shift/shock_scale, risk-measure
config, reporting FX, library versions). Verdict is therefore PRODUCTION CAPABLE
BUT NOT HARDENED; the next increment is the RunManifest defined in the audit.



## D040 — Run identity: content-addressed reproducibility contract for a risk run

Closes the run-identity gap left open by the adversarial audit. A new decision is
warranted because this adds a durable PUBLIC serialized contract (a manifest
format with its own schema version) that no existing decision covers. D036–D039
are unchanged and not rewritten. Full evidence:
`SIMLAB_RUN_IDENTITY_CLOSURE.md`.

**Unit of reproducibility.** The risk run, not the generator and not the scenario
stream. The identity was derived from a mutation inventory over the real path
(config -> generator -> `iter_batches` -> `apply_shocks_to_base_levels` ->
sensitivity P&L -> FX -> `compute_var_es`) BEFORE any manifest was designed.

**Two inventory findings that a "save the config" scheme would have missed.**
(1) `required_shift_type_id`, `source_factor_ids` and `shock_scale` are arguments
to `iter_batches`, not fields of the scenario config, and each changes the risk
number. (2) `batch_size` leaves the scenario stream bitwise invariant (D029) but
can perturb a downstream BLAS reduction by ~1 ulp (measured 1.4e-13 between
widths 7 and 16), so it is recorded rather than assumed neutral.

**Contract.** A risk result is reproducible when its run identity and the
identified input snapshots are available and the environment matches the
manifest's `engine` section (riskquant / python / numpy versions). Equality is
then BITWISE for VaR, ES, the scenario stream and the P&L vector. Cross-version
bitwise identity is NOT GUARANTEED and is not claimed.

**Implementation.** New solution-tier package
`src/riskquant/solutions/run_identity/` (`hashing.py`, `schemas.py`, `api.py`,
empty `__init__.py`). `RunManifest` is a frozen Pydantic v2 model
(`extra="forbid"`) with `manifest_schema_version` (Literal "1", deliberately
distinct from the engine version) plus `engine`, `scenario_config` (full dump,
re-parsable through the frozen D036 seam), `data` (kind + digest), `shocks`,
`pnl`, `reporting`, `risk_measure`, `execution`. Identity is SHA-256
(`hashlib`, no new dependency) over a canonical serialization: keys sorted, no
incidental whitespace, tuples normalised, floats written as exact IEEE-754 hex so
the digest cannot drift with dict order or decimal formatting. `to_json` is a
separate persistence form that round-trips to the same identity.

**Hashing is CONTENT identity.** dtype, shape and the C-ordered native-endian
buffer are folded in: copies and F-vs-C views hash equal; transpose, reshape and
dtype changes do not. factor_ids are hashed as an ORDERED vector, never a set,
because column order is authoritative. Floats are never rounded or stringified —
one-ulp differences have different identities. No collision claim is made beyond
SHA-256's primitive-level assumption.

**Identity vs record.** The manifest carries no timestamp, host, operator or job
id, so the same computation re-run later on another machine has the same
identity. Snapshots are identified, not stored: the JSON is ~1 kB regardless of
matrix size. Rejected as unnecessary: database, ORM, event sourcing, generic
provenance framework, global context, RunManager, registry, plugin mechanism.

**Fail-fast on substitution.** `verify_run_inputs` re-digests supplied inputs and
raises `InputMismatchError` naming the first mismatch, so a stale or substituted
snapshot cannot silently price a different, plausible VaR.

**Core purity (D001).** Manifest code is solution tier, built beside the
computation from the same inputs, and never passed into a generator, a kernel or
a Numba signature. `src/riskquant/simlab` is BYTE-IDENTICAL to the pristine dump;
the entire change is one new `solutions/run_identity` package plus its tests.

**Evidence.** run_identity suite 48 passed (11 result-defining mutations change
result AND identity; 3 false-positive cases change neither; 10 substitution cases
detected; 5 malformed-manifest rejections; fresh-process bitwise reproduction of
VaR/ES via exact float hex). `tests/solutions` + `tests/simlab` 1104 passed / 3
skipped; focused regression 861 passed / 1 skipped; E2E `ALL CHECKS PASSED`; lint
clean. Performance: manifest build 1.78 ms vs 21.3 ms scenario generation (8.3%
one-time) at K=250/S=10k; matrix hash 2.45 ms for a 2 MB matrix; JSON 1.01 kB.

**Documented asymmetry.** The contract forbids "same identity, different result".
It deliberately permits "different identity, same result": a uniform rescale of
the whole history is return-neutral for a log-return model yet changes the data
digest. That conservative direction is asserted by an explicit test.

**Remaining limitations.** Cross-version reproducibility not guaranteed;
`seed=None` runs carry no reproducibility claim; D037 `stream_id` still deferred
(it enters `scenario_config` automatically when implemented); the manifest
identifies snapshots but does not store them; and adoption is caller-side — the
production runner and E2E demo do not yet build a manifest. Wiring it in is the
next increment.

**Status: SUPERSEDED IN PART by D088 (2026-09-25).** `ShockIdentity = (required_shift_type_ids_digest, source_factor_ids_digest, shock_scale_digest)` is no longer the identity of the shock mapping. Under D088 there is exactly one rule: `ShockIdentity = (proxy_model_digest)`, where `proxy_model_digest = ID_proxy = H(canonical V1 proxy rows)` over the target node `rf_id`, the driver node `rf_id`s, the concrete driver chart ids, the canonical merged float64 loadings and the canonical row/term ordering, computed with the existing `digest_named_arrays` mechanism. Legacy fields: `required_shift_type_ids` is derived operational data already identified through `universe_map_digest` and `static_config_digest` and is not an independent proxy-identity input; `source_factor_ids` and `shock_scale` are replaced by the proxy-model digest. No identity field is optional. Unaffected: the run as the unit of reproducibility, the manifest canonical bytes and content-address semantics, the environment section, the bitwise-equality contract and the D041 and D083 relationships. Inventory finding (1) above remains true in substance — the proxy model changes the risk number — and is captured by the proxy-model digest.



## D041 — Deterministic execution: the final risk result is bitwise invariant; batch_size is operational (supersedes part of D040)

Release-gate audit. Full evidence: `SIMLAB_DETERMINISTIC_EXECUTION_AUDIT.md`.

**The D040 1-ULP finding was misattributed and is retracted.** Re-tested against
the OFFICIAL path (E2E demo instrumented on a copy to emit `float.hex()` of VaR
and ES), the final numbers are BITWISE IDENTICAL across batch sizes
1,2,3,4,5,7,11,63,64,65,1000, across three fresh processes, and across
OMP/OPENBLAS/MKL/NUMBA thread-environment settings. The 1.4e-13 divergence
recorded in D040 came from the audit harness's own `sensitivities @ delta` GEMV,
whose BLAS blocking varies with batch width. RiskQuant's P&L performs no such
reduction: the only BLAS call site in the entire numerical path is
`np.matmul(...)` in `simlab/scenarios/parametric.py:402`.

**Why the production path is structurally deterministic.** The Taylor kernel uses
thread-private scratch `(n_threads_max, B_capacity)` indexed by
`nb.get_thread_id()`; `prange` chunks partition the ROW axis so each iteration
owns disjoint whole rows of `out` with no cross-thread write aliasing; and no
reduction across the thread axis exists anywhere in `sensitivity_pnl`. The
arithmetic producing a given output row is therefore fixed however rows are
distributed across threads, even though the chunk partition itself depends on
`NUMBA_NUM_THREADS`.

**Numerical contract (new).** Scenario stream, shocked levels, trade P&L,
portfolio P&L, VaR and ES are ALL **BITWISE** reproducible under a matching
environment. Across differing NumPy/Numba versions: NOT GUARANTEED.

**Environment identity refined.** `EngineIdentity` gains `numba_version`:
evidence-based, because the P&L kernels are Numba/LLVM-compiled and codegen
governs FMA contraction. Thread settings, BLAS configuration, array layout, batch
size and process identity are deliberately NOT in the contract — each was
measured to have no effect. CPU architecture is outside the supported envelope
and is not claimed.

**Supersedes D040 in part.** `execution.batch_size` is REMOVED from the
mathematical run identity and reclassified as an operational parameter;
`ExecutionIdentity` is deleted; `manifest_schema_version` goes "1" -> "2" (a v1
payload and a stray `execution` field are now rejected — this is what the schema
version exists for). `build_run_manifest` no longer accepts `batch_size` and
`verify_run_inputs` no longer checks it. Every other D040 field survived
re-testing unchanged; D040 is not rewritten.

**Seed policy made explicit.** `seed` is the reproducibility root of the RNG, NOT
the run identity — Historical and FilteredHistorical carry no seed yet are fully
identified by the manifest. Production reproducible runs MUST set an explicit
seed; `seed=None` remains permitted for exploratory work and carries no
reproducibility claim. D037 (`stream_id`) stays deferred: nothing in this audit
required independent co-seeded streams.

**Evidence.** `test_deterministic_execution.py` 47 passed (4 models x 10 batch
sizes asserting bitwise equality of stream, shocked levels, P&L, VaR hex and ES
hex; plus layout invariance and fresh-process / thread-env child processes);
run_identity suites 95 passed; `tests/solutions` + `tests/simlab` 1151 passed / 3
skipped; focused regression 861 passed / 1 skipped; E2E `ALL CHECKS PASSED` at
every batch size; lint clean; `src/riskquant/simlab` BYTE-IDENTICAL to pristine.
Manifest build 1.97 ms vs 17.1 ms generation; no determinism remediation was
needed, so determinism costs nothing.

**Open, stated plainly.** (1) Multi-thread determinism is NOT empirically proven:
this host is single-core (`os.cpu_count() == 1`, `NUMBA_NUM_THREADS == 1`), so the
parallel path never executed. The contract is pinned by a test that becomes a
real gate on multi-core CI — running it there is REQUIRED before a multi-core
production release. (2) Manifest adoption remains caller-side; a risk result
without an identity should be considered incomplete. (3) A caller that reduces
P&L with its own BLAS GEMV reintroduces 1-ULP batch sensitivity in its own code.



## D042 — Production run entry point: every risk result is inseparable from its identity

Closes production adoption. Full evidence: `SIMLAB_PRODUCTION_RUN_CLOSURE.md`.
A new decision is warranted because this establishes a durable PUBLIC production
contract; D036–D041 are unchanged and not contradicted.

**Archaeology finding.** Before this increment there was NO production entry
point for scenario-based VaR/ES. No `src/` function produced one; all six
producers were hand-wired demo scripts and none carried an identity. The only
`src/` caller of `compute_var_es` was the credit loss path. "The runner does not
emit a manifest" understated it: there was no runner.

**The entry point.** `riskquant.solutions.risk_run.api.run_scenario_var_es`
composes ONLY existing production functions — the D036 scenario seam,
`apply_shocks_to_base_levels`, the compiled `SensitivityPnlCalculator` (official
Taylor P&L via `FactorAxis.levels`), `build_fx_conversion_plan` /
`convert_scenario_pnl_to_reporting_currency` (D014), and one canonical
`compute_var_es` over `[positions; portfolio]` (D015) — and adds no numerical
logic. Order is fixed: validate -> build identity from the exact objects handed
to numerics -> (if reproducing) verify environment and inputs, FAIL FAST ->
execute -> `RiskRun`.

**`RiskRun` (result + identity, inseparable).** Frozen dataclass: `position_uids`,
`var`/`es` (positions then portfolio LAST), `reporting_ccy`, full `manifest`,
`identity`, `reproducible`, `pnl_provenance`. `__post_init__` rejects an identity
that does not equal `manifest.identity()`; `to_json` writes the numbers as exact
IEEE-754 hex with the manifest inside; `from_json` re-derives the identity and
rejects a tampered payload. A risk number cannot be persisted without its
identity through this API. `SensitivityPnlProvenance` rides along as the
operational RECORD (counts, batch_size); it is not the identity (D040
identity-vs-record distinction preserved).

**Sensitivities identity strengthened.** `hash_compiled_book(calculator.book)`
digests the CSR arrays the Taylor kernel actually consumes (position_uids,
active_factor_ids, offsets, factor_uix, first/second/cross order). The identity
names exactly what priced the run. `build_run_manifest` now takes
`sensitivities_digest`/`fx_digest` (digests, not arrays) so a compiled book and a
plain vector share one path without a nullable alternative; identity values for
the array path are unchanged. `digest_reporting_plan` identifies reporting ccy,
per-position settlement ccys, FX factor names and rows; FX scenario levels are
not re-hashed because they are rows of the shocked levels already fixed by the
data + shock identities.

**Seed policy (explicit).** `seed` is the RNG reproducibility root, NOT the run
identity — Historical/FilteredHistorical carry no seed and are fully identified.
`mode="reproducible"` (default) raises `NonReproducibleRunError` for a stochastic
model with `seed=None` before any numerical work; `mode="exploratory"` permits it
and marks `reproducible=False`. A `seed=None` run is never represented as
reproducible.

**Environment gate (explicit).** `verify_environment` enforces STRICT EQUALITY
on riskquant/python/numpy/numba versions when reproducing. No compatibility
policy: the repository has none, the claim is BITWISE, and no evidence exists
that any two versions are bit-compatible. `EnvironmentMismatchError` before
numerics.

**Stale-input boundary.** When `reproduce=` is given, `verify_run_inputs`
re-digests data, shock vectors, base levels, compiled book, config, confidence
and reporting currency and raises `InputMismatchError` on the first mismatch —
after identity construction, before `build_scenario_generator`. No kernel or
generator is touched. `batch_size` is operational (D041): same identity and same
numbers for bs=3 and bs=S+5 through the production API.

**Bypass audit.** One PUBLIC production entry point (identity enforced). The
scenario seam, generators, `apply_shocks`, `SensitivityPnlCalculator`,
`run_sensitivity_pnl`, FX conversion and `compute_var_es` remain identity-free
low-level building blocks by design (D001). The six demo scripts and the credit
loss path are documented as demo / separate-path, not production bypasses.

**Rejected.** `RunManager`, `RunContext`, global execution context, registry,
database, ORM, event sourcing, plugin, generic provenance framework — none was
needed; the existing solution-tier composition sufficed. A `RiskResult(...,
run_identity=)` extension of `VarEsResult` was rejected because `VarEsResult`
is a core metric type and must stay identity-free; `RiskRun` composes at the
solution tier instead.

**Evidence.** `tests/solutions/risk_run` 29 passed (result always carries
identity; identity tamper-checked on construction and on load; bitwise JSON
round-trip; seed policy; batch_size excluded; stale data / stale book /
substituted shock-scale, source-ids, confidence, base-levels refused before
execution; foreign environment refused; 10 result-defining mutations change
result AND identity; equivalent objects change neither; fresh-process
reproduction of identity, VaR and ES via the production API, and a persisted
`RiskRun` reproducing against its own manifest in a new process).
`tests/solutions/run_identity` 95; `tests/solutions` + `tests/simlab` 1180 passed
/ 3 skipped; focused regression 861 passed / 1 skipped; E2E `ALL CHECKS PASSED`;
lint clean; `src/riskquant/simlab` and `src/riskquant/sensitivity_pnl` core
BYTE-IDENTICAL. Performance at K=250/S=10k/M=200: production run 88.4 ms;
identity 3.3 ms (3.8%); verification 1.6 ms; artifact 11.9 kB.

**Verdict.** PRODUCTION READY WITH EXPLICIT ENVIRONMENT GATE. Open: multi-core
determinism remains structurally argued and empirically unverified on this
single-core host (D041) — a required prerequisite for a multi-core release.



## D043 — SimLab RELEASE-CERTIFIED (corrects D041's multi-core limitation)

Final release-gate certification of `riskquant.simlab` and the official
scenario-VaR path. Full evidence: `SIMLAB_FINAL_RELEASE_CERTIFICATION.md`.
Test-only change; no production code was modified.

**D041's multi-core limitation was WRONG and is corrected.** D041 recorded
multi-core certification as "not performed" because `numba.config.NUMBA_NUM_THREADS
== 1`. That value merely follows `cpu_count()` when the variable is unset. With
`NUMBA_NUM_THREADS=8` on the same single-`nproc` host, Numba's OMP layer spawns
real worker threads and **8 distinct `numba.get_thread_id()` values are observed
INSIDE the parallel region** (`set_num_threads(4)` gives exactly 4). The gate was
executable all along. D041's numerical conclusions are unaffected and stand.

**Multi-thread determinism now EMPIRICALLY certified.** Through the official
`run_scenario_var_es`, at threads {1,2,4,8} actually executing, across models
{historical, sbb, filtered_bootstrap, filtered_historical, parametric} and
batch sizes {1,3,7,64,S,S+1}: scenario stream, shocked levels, trade P&L, VaR, ES
and run identity are all bitwise identical, compared in divergence order
(sha256 over dtype|shape|bytes; `float.hex()` for scalars; no tolerance anywhere).
Fresh processes agree. The one BLAS site (`parametric.py:402 np.matmul`) got its
own gate under five BLAS/OMP thread configurations and did not move.

**Structural backing.** Every `prange` reachable from the risk path partitions an
independent axis; `taylor.py:159` takes thread-private scratch via
`accumulator[nb.get_thread_id()]`; a search for any cross-thread reduction
(shared accumulator, `.sum(axis=0)` over a thread axis, a loop over thread ids,
atomics, critical sections) returns nothing. Arithmetic per output row is
therefore fixed however rows are distributed.

**Bypass audit.** Exactly two `src/` callers of `compute_var_es`:
`risk_run/api.py` (the single public production scenario-VaR entry, identity
enforced) and `credit/aggregate/loss_metrics.py` (separate credit subsystem).
Exactly one function returns `RiskRun`. No unintended production bypass.

**Verdict: RELEASE READY WITH DOCUMENTED ENVIRONMENT LIMITATIONS.** No P0, no P1.
Remaining limitations are the out-of-scope set — cross-version bitwise
reproducibility (gated by `verify_environment`), `seed=None` (refused by default),
D037 co-seeded streams (deferred, no consumer), unsupported CPU architectures —
plus one honest hardware qualifier: threads were time-sliced on a single physical
CPU, so true multi-socket execution is uncertified. No code mechanism is sensitive
to simultaneity, since no cross-thread reduction exists.

**Evidence.** Release certification suite 47 passed; `tests/solutions/risk_run`
76; `run_identity` + `production_run` 124; `tests/solutions` + `tests/simlab` 1180
passed / 3 skipped; focused regression 861 passed / 1 skipped; E2E
`ALL CHECKS PASSED`; lint clean; `simlab`, `sensitivity_pnl`, `scenario_metrics`
and `reporting` BYTE-IDENTICAL to pristine. Performance flat across thread counts
(28–29 ms, ~70 k scen/s at K=250/S=2000/M=200); verification <1 ms.

**SIMLAB = RELEASE-CERTIFIED for the documented scope. SimLab work is closed.**



## D044 — Correction to D043 + portable multi-core release gate

D043 overclaimed. This entry corrects it; D043 is not rewritten.

**The overclaim.** D043 wrote "multi-core determinism now EMPIRICALLY certified"
and "genuine multi-thread execution ... the multi-core gate ran and passed". The
evidence supports the second half only. Three claims must never be collapsed:

```
A. NUMBA MULTI-THREAD EXECUTION VERIFIED  — >= 2 distinct get_thread_id() inside a real prange
B. MULTI-CORE HARDWARE AVAILABLE          — the PROCESS has >= 2 CPUs
C. MULTI-CORE DETERMINISM CERTIFIED       — A and B, plus every check passing
```

On the audit host `len(os.sched_getaffinity(0)) == 1` and `os.cpu_count() == 1`.
Numba's OMP layer still created 8 workers, but they time-slice ONE CPU.
Therefore: **A is PROVEN. B is FALSE. C is NOT PROVEN.** D041's claim that the
parallel path could not be exercised was wrong (A corrects it); D043's claim of
multi-core certification does not follow from A.

**Corrected D043 conclusion.** D043 demonstrated genuine multi-threaded Numba
execution and determinism across the tested worker configurations (1, 2, 4, 8),
but did NOT establish determinism under genuinely concurrent multi-core hardware,
because effective CPU availability was one. D043 is therefore NOT the final
multi-core hardware release certificate. Its determinism observations stand as
supporting evidence; its headline classification does not.

**The portable gate.** `scripts/release/certify_simlab_multicore.py`, run as
`python scripts/release/certify_simlab_multicore.py`. It uses only the official
`run_scenario_var_es`, compares in divergence order (scenario stream -> shocked
levels -> trade P&L -> portfolio P&L -> VaR -> ES -> manifest identity ->
RiskRun identity), exactly (sha256 over dtype|shape|bytes; `float.hex()`; no
tolerance), and emits a human-readable plus a JSON certificate. It reports
PASS / FAIL / NOT EXECUTED and **cannot produce a false multi-core PASS**:
hardware is measured by `sched_getaffinity` (not `cpu_count`), a requested
worker count that is clamped is not counted as executed, and NOT EXECUTED is
never upgraded. Exit codes 0/1/2.

**Run on the audit host: NOT EXECUTED** (effective CPUs = 1), with
"NUMBA MULTI-THREAD EXECUTION VERIFIED: True" recorded separately so the two
claims cannot be confused. Certificate retained as
`simlab_multicore_certificate.NOT_EXECUTED.json`.

**PASS path validated, and it found a real defect.** A machinery dry-run with the
hardware check forced exercised the whole PASS path and surfaced two failures:
one fixture flaw (the "different book" substitution rebuilt an identical book —
corrected in the gate), and one GENUINE production gap — `RiskRun` validated
identity and array lengths but never checked `reporting_ccy` against
`manifest.reporting.reporting_currency`, so a persisted artifact could state a
currency its own identity contradicted. **Smallest correction applied**:
`RiskRun.__post_init__` now rejects that mismatch (solution tier, no numerics).
Regression after the fix: risk_run + run_identity 124 passed; tests/solutions +
tests/simlab 1180 passed / 3 skipped; focused regression 861 passed / 1 skipped;
E2E `ALL CHECKS PASSED`; lint clean; `simlab`, `sensitivity_pnl`,
`scenario_metrics` and `reporting` BYTE-IDENTICAL to pristine.

**Classification: RELEASE READY WITH DOCUMENTED ENVIRONMENT LIMITATIONS**, with
the multi-core determinism gate explicitly **OPEN** and executable locally. SimLab
is release-certified for every claim except C. When the local run reports
`MULTI-CORE DETERMINISM CERTIFIED`, SimLab is closed for the documented envelope;
if it reports FAIL, stop at the first divergent object named by the certificate
and document the mechanism before any patch.


## D045 — Fresh independent valuation audit: NOT PROVEN

**Status:** LOCKED AS AUDIT BASELINE
**Date:** 2026-09-16
**Full evidence:** `RISKQUANT_VALUATION_FRESH_INDEPENDENT_AUDIT.md`

A fresh, independent audit of `riskquant.valuation` was performed against the
current repository dump. `DECISIONS.md` and the prior seven-pass valuation audit
were used only for orientation; material conclusions were re-established from
current source, independent mathematical derivation, targeted probes, and
executed evidence.

**Verdict: NOT PROVEN.**

This is **not** an assertion that the valuation mathematics is wrong. The
mathematics independently checked in this pass is substantially correct. The
package is not certifiable because one live critical production defect exists,
the valuation test suite is substantially stale/non-executable, and most
authored valuation functionality is not reachable through the sanctioned
production construction path.

### Confirmed live valuation defect

**VAL-01 — CRITICAL — `_black76_fv_vega` domain guard absent.**

`pricing_models/black76/model.py::_black76_fv_vega` documents a guard requiring
vega to return `0.0` when `vol <= 0` or `sqrt_tte <= 0`, but the implementation
contains no such guard.

Measured behaviour:

* `vol = IV_SOLVER_FAILED_SENTINEL (-1.0)` returns approximately `35.2065`
  instead of `0.0`;
* ATM `vol = 0` returns `NaN`;
* OTM `vol = 0` happens to return `0.0` only accidentally through tail
  saturation.

The production implied-volatility calibration path contains a comment explicitly
relying on this guard.

This is a genuine valuation/model-primitive defect and is live.

### Assurance failure

**VAL-02 — CRITICAL — current valuation test suite is substantially dead.**

<!-- # TODO: We need to update this. src/riskquant/valuation/pricers/_deprecated was
removed from RiskQuant. It was legacy dead code that survived from a previous
refactor. It is now gone. Any reference to src/riskquant/valuation/pricers/_deprecated
are invalid and stale and should be ignored. -->

### Reachability state

Only `SpotBlock` is currently wired through the sanctioned portfolio path.

The current production compiler registry contains only the Spot compiler.
`CompiledPortfolio` exposes slots for Spot, Forward, and European Option, while
the three cashflow block types have no compiled portfolio slot. The valuation
state currently populates only `effective_spot_prices`.

Consequently:

* Spot valuation is reachable;
* Forward valuation is authored but unreachable;
* European option valuation is authored but unreachable;
* Fixed cashflow valuation is authored but unreachable;
* Floating cashflow valuation is authored but unreachable;
* Principal cashflow valuation is authored but unreachable.

These are primarily buildout/reachability findings, not evidence that the
underlying pricing formulas are incorrect.

### Mathematics independently established

The following were independently derived and/or numerically reconciled:

* Spot valuation matches `Q × effective_spot_price` exactly;
* the forward kernel is the correct two-leg formulation for the currently
  implied physical, two-currency settlement model;
* Black-76 `cp_mult` packing is mathematically correct for both calls and puts;
* put-call parity holds to machine precision;
* fixed, floating and principal cashflow algebra matches the declared equations;
* effective discount-factor folding
  `DF_ccy × FX_spot(ccy→settle)` is mathematically valid for the current
  settlement-currency valuation representation;
* valuation performs no FX inversion itself; inversion is resolved upstream;
* canonical C/F layout handling does not exhibit the SimLab single-row/single-
  column ambiguity defect.

### Additional findings

The fresh audit recorded:

* **VAL-03 HIGH:** `run_valuation` does not validate the full `out.shape`;
* **VAL-04 HIGH:** several `ValuationState` factor matrices default to empty
  arrays and have no producer/validator, creating an unsafe latent state;
* **VAL-05 HIGH:** `ValuationResult.get_trade_values` assumes sorted trade IDs
  although `CompiledPortfolio` constructs them in block-concatenation order;
* **VAL-06 MEDIUM/HIGH:** forward factor documentation describes a raw forward
  price while the state supplies a discounted, FX-converted quantity;
* **VAL-07 MEDIUM:** option documentation describes a nonexistent
  `INVALID_UIX` branch;
* **VAL-08 MEDIUM:** `VanillaEuropeanOptionBlock` lacks the keyword-only
  construction contract used by sibling blocks;
* **VAL-09 MEDIUM:** block/result equality and hashing semantics are not defined
  and produce unsafe behaviour;
* **VAL-10 MEDIUM:** the active Abramowitz–Stegun normal CDF has no documented
  accuracy envelope; measured relative pricing error reaches approximately
  `4.3e-4` for an ordinary deep-OTM put and several percent in very deep tails;
* **VAL-11 MEDIUM:** forward and log-forward option factor universes have no
  enforced pairing invariant;
* **VAL-12 MEDIUM:** read-only kernel inputs use writable aliases despite
  canonical `_RO` aliases existing in `riskquant.jit`;
* **VAL-13 MEDIUM:** five of six authored block types lack current compilation
  reachability;
* **VAL-14/15/16 LOW:** stale or incomplete valuation field/convention
  documentation;
* **VAL-17 LOW:** `resolve_pricer` is dead relative to actual engine dispatch;
* **VAL-18 LOW ACCEPTED-RISK:** denormal far-tail Black-76 values can become
  slightly negative at machine-denormal magnitudes.

### Currency finding

The previously reported valuation/sensitivity inverse-FX difference is **not a
valuation mathematical defect**.

Valuation receives already-resolved effective FX rows and performs multiplication.
Sensitivity requires the functional form of FX to compute derivatives and
therefore has a different admissibility condition for inverse FX.

The remaining issue is a boundary-contract/documentation gap, not a valuation
formula error.

### Canonical valuation theorem

Under the explicitly stated input hypotheses:

1. target factor matrices are populated and C-contiguous `float64`;
2. every block UIX is valid for its target matrix;
3. factor rows carry the economic quantities declared by the valuation state;
4. option factors satisfy the required positivity and forward/log-forward pairing
   invariants;
5. output storage has the required shape and layout;

the authored valuation kernels implement their declared mathematical equations,
up to IEEE-754 arithmetic and the accepted Abramowitz–Stegun CDF approximation
on the option path.

The hypotheses are **not currently guaranteed by the repository**.

### Architectural disposition

No broad redesign is warranted.

The numerical kernels remain consistent with the RiskQuant principle that
kernels perform dumb mathematics and semantic contracts are established above
them.

The next phase is therefore controlled remediation and assurance, not a rewrite
of the valuation architecture.

### Certification gates

Before `riskquant.valuation` can be release-certified:

1. VAL-01 must be closed and regression-tested through the live calibration path.
2. The current valuation test suite must be restored to executable, current APIs.
3. An independent end-to-end valuation oracle must exist and use hostile,
   identifying fixtures.
4. Output-shape and trade-identity contracts must be explicit and enforced.
5. Factor/UIX ownership must be established at the boundary that owns each
   producer.
6. Valuation documentation must match actual economic quantities and currency
   conventions.
7. The Black-76 CDF approximation must have an explicit accuracy contract and
   recorded decision.
8. Authored valuation blocks must either be deliberately documented as
   buildout/unreachable or be wired through a complete, independently tested
   production construction path.

**Conclusion:**

> `riskquant.valuation` is **NOT PROVEN** as a production-certified valuation
> primitive in the current repository. The independently checked pricing
> mathematics is largely sound; the immediate blockers are one live critical
> defect, broken assurance coverage, and incomplete semantic/reachability
> contracts.

## D046 — Black-76 normal-CDF accepted-risk closure

**Status:** LOCKED — ACCEPTED-RISK
**Date:** 2026-09-16
**Finding:** VAL-10
**Full evidence:** `VAL-10_BLACK76_CDF_NUMERICAL_CONTRACT.md`

The fresh valuation audit identified an undocumented numerical approximation in
the Black-76 valuation path: `numba_norm_cdf_abrom_steg` implements Abramowitz &
Stegun 26.2.17 but had no explicit accuracy contract.

An independent measurement was performed using SciPy as the reference and an
independently implemented Black-76 formula. The benchmark also compared the
existing erf-based `numba_norm_cdf`.

### Decision

**Retain `numba_norm_cdf_abrom_steg` on the valuation price path.**

Do not replace it with `numba_norm_cdf` merely to improve tail accuracy.

### Evidence

A&S absolute CDF error is bounded by approximately:

```text
7.5e-8
```

with relative error increasing as the tail probability becomes small:

```text
|d| = 1  → ~3.6e-8 relative
|d| = 2  → ~3.1e-6
|d| = 3  → ~5.1e-5
|d| = 4  → ~4.7e-4
|d| = 8  → ~1.0e-2
```

Across the measured Black-76 price grid:

* worst relative pricing error was approximately `1.75e-2`;
* worst absolute error was approximately `1.97e-5` per 100 of forward.

The large relative errors occur only where the option price itself is extremely
small. For the measured grid, every point with relative pricing error above
`1e-3` had an exact price below approximately `1e-7` per 100 of forward.

For implied-volatility inversion, the measured maximum error was approximately
`0.077 bp`, against market quote precision of approximately 5–25 bp.

### Alternative CDF

The existing `numba_norm_cdf` was independently measured rather than assumed
to be superior.

It is substantially more accurate through much of the central region, but is
not uniformly superior. In particular, `math.erf` causes the probability to
reach exactly zero at sufficiently extreme negative arguments (for example
`|d| = 10`), producing complete relative loss in that tail.

An otherwise identical 8-million-option benchmark measured:

```text
A&S CDF : 195.8 ms best / 218.9 ms median
erf CDF : 344.2 ms best / 371.7 ms median
ratio   : ~1.76x
```

The benchmark was single-threaded and isolates the CDF implementation. The
absolute throughput is environment-dependent; the measured relative cost is
the relevant architectural evidence.

### Parity

The A&S sign branch satisfies:

$$
\Phi(x)+\Phi(-x)\equiv1
$$

identically, so the approximation error cancels in:

$$
C-P=F-K.
$$

Put-call parity is therefore exact under the approximation to floating-point
precision.

### Canonical accepted-risk contract

`numba_norm_cdf_abrom_steg` is accepted with the following documented envelope:

> Implements Abramowitz & Stegun 26.2.17 with absolute CDF error no greater
> than approximately `7.5e-8`. Relative CDF error increases materially in the
> far tail, reaching approximately `5e-5` at `|d|=3`, `5e-4` at `|d|=4`, and
> `1e-2` at `|d|=8`.
>
> For Black-76 valuation, measured absolute price error is bounded at roughly
> `2e-5` per 100 of forward over the audited grid. Relative price error can
> become large for extremely small option values, but the corresponding
> absolute error remains small. Implied-volatility inversion through the
> production approximation was measured to better than `0.1 bp` over the
> audited cases.
>
> The erf-based `numba_norm_cdf` is retained as a separate reference/research
> primitive. It is not uniformly more accurate and measured approximately
> `1.76x` slower in an otherwise identical pricing kernel.

### Disposition

**VAL-10 is CLOSED.**

The remaining remediation is documentation only: attach the accepted-risk
contract to `numba_norm_cdf_abrom_steg` and record this decision.

No numerical implementation change is authorised by VAL-10.

**VAL-01 remains open and independent.** The vega domain-guard defect is
unaffected by this decision because `_black76_fv_vega` uses the normal PDF rather
than the CDF.

## D047 — `VanillaForwardBlock` is a physically settled forward; no settlement enum

**Status:** LOCKED

**Decision:** `VanillaForwardBlock` means one thing: a PHYSICALLY SETTLED
forward. Receive one unit of the asset at T, pay `strike` in the strike
currency, each leg discounted on its own curve and converted at its own spot FX.

    PV(0) = Q * [ F_a DF_a FX(a->s)  -  K DF_k FX(k->s) ]

**Reason:** D045/VAL-15 found the two-leg formula could express both physical
and cash settlement, with no recorded convention. A block with two possible
economic meanings is not a contract. Physical is the general case: when both
legs share a curve, date and currency the form degenerates EXACTLY to
`Q (F - K) DF * FX`, which is also the cash-settled value, so cash settlement
needs no separate representation.

**Implications:** no settlement enum, no second block type. The compiler must
never pair two DIFFERENT discount curves on a cash-settled forward -- the
two-leg form would then silently produce the physical value. Documented on the
block. Proved as an oracle result, not an asserted identity:
`test_single_currency_forward_degenerates_to_q_times_f_minus_k_times_df`.

## D048 — `effective_forward_prices` is a settlement-currency PV, not a forward price

**Status:** LOCKED

**Decision:** a row of `ValuationState.effective_forward_prices` is

    F_ccy(0,T) * DF_ccy(0,T) * FX_SPOT_EFFECTIVE(ccy -> settle)

-- the PV at t = 0 of receiving one unit of the asset at T, in SETTLEMENT
currency. It is not a forward price and not in the asset currency.

**Reason:** D045/VAL-06. The block comment previously described a raw forward
price. A compiler written to that comment would subtract a discounted PV from an
undiscounted price and produce a dimensionally invalid number with no error.

**Implications:** `_fold_effective_forward_prices` produces exactly this, and
the block field comment states the units, currency, date, discounting and FX
state. Closes VAL-06.

## D049 — Forward effective universes: `(curve, date, fx)` and `(price, curve, date, fx)`

**Status:** LOCKED

**Decision:** two new deduplicated tables on `UixMapper`:

    edf_effective_triples  (K,3)  (curve_uix, date_uix, fx_eff_uix)
    efp_effective_quads    (K,4)  (price_uix, curve_uix, date_uix, fx_eff_uix)

built by `build_uix_mapper` from exactly the legs the forward compiler will
resolve, and folded by `market_data.compile` into `effective_discount_factors`
and `effective_forward_prices`.

**Reason:** the forward kernel gathers one pre-multiplied row per leg. Those
rows must exist and be addressable by a dense UIX the compiler can prove valid.

**Implications:** lookup uses a MIXED-RADIX key built from RUNTIME universe
sizes, not hard-coded bit shifts -- a hard-coded shift silently aliases two
distinct factors the day a universe outgrows its field, and the int64 product is
asserted. Both tables are built on `fx_effective_pairs`, so the `fx_eff_uix`
they carry is byte-identical to the one the compiler stamps per trade.
Both fields default to empty, so existing `UixMapper` callers are unaffected.

## D050 — NODE-sourced forwards only; CURVE stays unregistered

**Status:** LOCKED

**Decision:** `VANILLA_FORWARD_NODE_ID` is registered and production-supported.
`VANILLA_FORWARD_CURVE_ID` remains UNREGISTERED.

**Reason:** curve-sourced forwards read `MarketDataMatrix.price_curve_grid`, not
`forward_nodes`, and therefore need a source-kind column in
`efp_effective_quads`. Registering both against a node-only fold would price a
curve forward off the wrong array, silently.

**Implications:** the engine fails loud on a curve forward, per the existing
partial-registration convention. VAL-13 closes FOR NODE FORWARDS ONLY. Do not
read this as "forwards are supported"; read it as "node forwards are".

## D051 — The block compiler owns UIX validity for the forward path

**Status:** LOCKED

**Decision:** `compile_vanilla_forward_block` proves `0 <= uix < n_rows` against
the row count of each effective universe BEFORE emitting a block, and raises
naming the offending trade uids otherwise.

**Reason:** D045/VAL-04. The kernel runs under `boundscheck=False` and cannot
distinguish an out-of-range row from a real one: a negative uix wraps to the
last row, an oversized one reads arbitrary memory. The compiler is the earliest
layer that holds both the trade identity and the universe, so it is the correct
owner.

**Implications:** no bounds check inside any Numba kernel. VAL-04 closes FOR THE
FORWARD PATH ONLY -- the option and cashflow universes are still unpopulated and
their compilers unwritten. Evidence:
`test_unresolved_market_leg_is_rejected_by_the_compiler`,
`test_unresolved_discount_leg_is_rejected_by_the_compiler`,
`test_mapper_built_from_a_different_batch_is_rejected`.

## D052 — Phase 4 closure: forward production-certified against the independent oracle

**Status:** LOCKED

**Decision:** the NODE-sourced vanilla forward is production-certified. A real
`IngestBatch` compiled by the real compiler and priced by the real engine
matches an independent economic oracle to a worst relative error of 4.09e-15
over four trades and three scenarios, on a hostile cross-currency fixture.

**Evidence:** 156 valuation tests pass, including the 10 Phase-3 oracle tests
unchanged. Fixture: `trade_uids = [9007, 41, 120003, 5]`; sparse non-ascending
rf ids with `id != row position` everywhere; USD asset / GBP strike / EUR
settlement; the GBP leg quoted EUR->GBP with `inv = 1` so the resolution must
invert it; `DF_asset != DF_strike` on different dates; mixed long and short.
Seven mutation classes detected against the oracle, worst 3.78e+03 relative
deviation: asset forward row permutation, strike discount row substitution,
asset FX substitution, strike FX substitution, wrong FX inversion, spot row
permutation, block row permutation.

**Implications:** VAL-13 (node forwards), VAL-06 and VAL-15 CLOSED; VAL-04
CLOSED FOR THE FORWARD PATH. Remaining buildout, none of which is a defect:
curve-sourced forwards, the option compiler, cashflow compilers and their
`CompiledPortfolio` slots, the `vols` / `forward_rates` /
`forward_prices_strike_ccy` / `option_log_forward_prices_strike_ccy` producers,
and VAL-11 (forward/log-forward pairing).


## D053 — Cross-currency option volatility basis remains an explicit model-input contract

**Status:** OPEN — MODEL-INPUT CONTRACT, NOT A VALUATION DEFECT

**Context:** Phase 5 option production certification identified an important
distinction between correct volatility-row binding and correctness of the
economic volatility itself. `VanillaEuropeanOptionBlock.vol_uix` is correctly
bound to the volatility row selected for the option, but the current repository
does not contain sufficient stochastic inputs to derive or independently
validate the volatility of a cross-currency strike-currency forward from an
underlying-forward volatility.

### Decision

For the current NODE-sourced vanilla European option implementation, the
valuation contract is:

> **The volatility supplied to Black-76 must already be the volatility of the
> exact strike-currency forward represented by the option's
> `forward_price_strike_ccy_uix`.**

In particular, RiskQuant must NOT silently substitute the volatility of the
asset/underlying forward when the option's Black-76 underlying is the
strike-currency translated forward.

For a cross-currency option, where the priced quantity is conceptually:

$$
F_K
=
F_a
\times
\frac{FX_{a\to s}}{FX_{K\to s}},
$$

the supplied volatility must represent the stochastic volatility of
\(F_K\), not merely that of \(F_a\).

### What RiskQuant proves

The option compiler proves the **identity binding**:

$$
vol\_uix
\rightarrow
\text{the volatility row declared for the option's }F_K.
$$

The production path therefore guarantees that the option does not accidentally
select an unrelated volatility row.

The independent oracle and mutation tests establish that substituting another
volatility row materially changes the economic PV.

### What RiskQuant does not currently prove

The current repository contains no complete model for deriving the volatility of
a cross-currency strike-currency forward from:

* asset-forward volatility;
* FX volatility;
* correlation/dependence;
* or other quanto inputs.

Therefore the repository cannot currently establish that a supplied volatility
value is economically correct merely from an underlying-forward volatility.

This is an **input/model-governance boundary**, not a kernel defect.

### Required fail-safe semantic rule

A future option construction path must not permit the following implicit
substitution:

$$
\sigma(F_a)
\;\not\Rightarrow\;
\sigma(F_K).
$$

Where the feed does not explicitly identify the supplied volatility as belonging
to the priced strike-currency forward, the construction layer must either:

1. reject the option as lacking a valid volatility basis; or
2. use an explicitly defined model that derives the required
   \(\sigma(F_K)\) from sufficient quanto inputs.

No fallback or "best available" volatility is permitted.

### Current supported contract

The current NODE option implementation may continue to be certified under the
following assumption:

> **The market-data/feed layer supplies a volatility already defined for the
> exact strike-currency forward being priced.**

This assumption must be visible in the production contract and option
documentation.

It is deliberately NOT implemented as a new runtime calculation in the Black-76
kernel.

### Future quanto model

A separate future decision is required before RiskQuant claims native
quanto-volatility construction.

That future decision must define, independently:

* the stochastic quantities;
* the FX volatility inputs;
* the dependence/correlation inputs;
* the variance/covariance construction;
* any approximation or model assumption;
* the calibration/data source;
* and the numerical validation oracle.

No such model is introduced by D053.

### Sequencing

D053 does **not block the next valuation construction increment**.

The next increment may proceed to the fixed, floating and principal cashflow
blocks, while D053 remains open.

Before any future claim of unrestricted cross-currency option support, D053 must
either be closed by an explicit supplied-volatility contract enforced at the
input boundary or superseded by a separately approved quanto-volatility model.

### Relationship to Phase 5

Phase 5 therefore remains correctly classified as:

> **OPTION PRODUCTION-CERTIFIED — for NODE-sourced vanilla European options
> with explicitly supplied volatility for the strike-currency forward.**

D053 does not invalidate the Phase 5 construction/binding certification.

It records the boundary of that certification so that a feed supplying
asset-forward volatility cannot be mistaken for a feed supplying
strike-forward volatility.

### Rejected alternatives

**Implicit substitution of asset-forward volatility:** rejected because it can
produce a mathematically valid Black-76 calculation on the wrong stochastic
underlying.

**Adding FX-volatility/correlation inputs to the current option block:** rejected
as premature because no implemented quanto model or product requirement has
established the required specification.

**Embedding a quanto adjustment in the Black-76 kernel:** rejected because the
kernel must remain a mathematical primitive and cannot own market-model
construction or semantic input validation.

**Using a generic volatility-basis string inside the kernel:** rejected by the
DOD/SoA architecture. Any semantic basis must be resolved before the JIT
boundary.

**Verdict — OPEN / EXPLICIT INPUT CONTRACT.**

The current option path is safe only under the explicit assumption that the
volatility feed already represents the exact forward underlying used by the
Black-76 price.


## D054 — Cashflow production construction is blocked at ingest, not at valuation

**Status:** LOCKED (blocker recorded; Phase 6 not completed)

**Decision:** the three authored cashflow block types (`FixedCashflowBlock`,
`FloatingCashflowBlock`, `PrincipalCashflowBlock`) CANNOT be wired to a
production compiler against the current repository, and no compiler was written.

**Evidence:** two independent blockers, both verified from source.

1. `IngestBatch` carries none of the cashflow economics the blocks require --
   no `notional`, `accrual_factor`, `fixed_rate`, `spread`, `amount`,
   `pay_receive_mult`, and no reset/fixing date. This is closed at the schema,
   not merely absent from the SoA container: `ALL_ALLOWED_COLS_SET` is the
   entire permitted ingest universe and is

       uid, instrument, quantity, underlying, settlement_ccy, netting_set_id,
       strike_price, strike_ccy, is_call, expiry_date, delivery_date,
       settlement_date, under_discount_curve, strike_discount_curve,
       volatility_container_name

   A rate-schedule column set does not exist at any layer.

2. `MarketDataMatrix` has no forward-rate producer.
   `ValuationState.forward_rates` exists as a consumer field and measures
   `size == 0` on every run. A simple forward rate is derivable from
   `discount_factor_grid` as `(DF(0,s)/DF(0,T) - 1)/alpha`, but that presupposes
   single-curve discounting -- a modelling decision (OIS/LIBOR basis) outside a
   valuation-hardening phase -- and still needs `alpha` and the reset date from
   an ingest that has neither.

**Implications:** VAL-13 stays OPEN for cashflows; VAL-04 is NOT closed for the
cashflow path; VAL-16's two-convention sign contract remains documented but
untested in production. No ingest columns were added: that is an EDGE/schema
change spanning `schema.py`, `parser.py`, `canonicalize.py`, `materialize.py`,
`validate.py` and `IngestBatch`, plus a rate-curve market-data producer.

Positive findings recorded for whoever does the work:
* the D049 EDF key `(curve_uix, date_uix, fx_eff_uix)` is the correct SHAPE for
  a cashflow -- curve, payment date, FX-to-settlement. Only the dedup POOL needs
  extending with `(cashflow_curve, payment_date, fx_c2s)`. No redesign.
* adding cashflow slots to `CompiledPortfolio` is safe: `simlab/runner.py` and
  `valuation/engine.py` both read `total_trades` as a ROW count, which is
  correct and stays correct, and nothing assumes `num_trades == num_rows`.

Pinned by `test_cashflow_blocks_are_still_not_production_reachable` and
`test_forward_rate_state_field_has_no_producer` so the blocker is a visible
regression rather than folklore.

## D054 — Cashflow production construction is blocked at ingest and rate-model inputs, not at valuation

**Status:** LOCKED (blocker recorded; Phase 6 not completed)

**Decision:** the three authored cashflow block types
(`FixedCashflowBlock`, `FloatingCashflowBlock`, `PrincipalCashflowBlock`)
CANNOT be wired to a production compiler against the current repository, and no
cashflow compiler was written.

The blockage is upstream of `riskquant.valuation` and consists of two separate
missing contracts:

1. **Cashflow economics are absent from the ingest schema.**

   `IngestBatch` carries none of the economic fields required by the authored
   cashflow blocks:

   * `notional`
   * `accrual_factor`
   * `fixed_rate`
   * `spread`
   * `amount`
   * `pay_receive_mult`
   * reset/fixing date

   This is not merely a missing SoA field. The permitted ingest universe is
   closed by `ALL_ALLOWED_COLS_SET`:

   ```
   uid, instrument, quantity, underlying, settlement_ccy, netting_set_id,
   strike_price, strike_ccy, is_call, expiry_date, delivery_date,
   settlement_date, under_discount_curve, strike_discount_curve,
   volatility_container_name
   ```

   No cashflow schedule/rate column set exists elsewhere in the current ingest
   stack.

2. **No forward-rate economic source exists.**

   `ValuationState.forward_rates` exists as a consumer field but is unpopulated.
   `MarketDataMatrix` has no forward-rate producer in the raw market-data tier.

   A forward rate can be mathematically derived from discount factors under a
   particular single-curve convention, for example

   ```
   (DF(0,s) / DF(0,T) - 1) / alpha
   ```

   but that expression is NOT adopted by this decision. Doing so would select a
   modelling convention for the floating-rate source and would still require
   accrual and reset-date inputs that do not exist in the current ingest schema.

**Implications:**

* `VAL-13` remains OPEN for cashflows.
* `VAL-04` remains NOT CLOSED for the cashflow path.
* `VAL-16` sign conventions remain documented but are not production-tested for
  cashflows.
* No ingest columns or rate-market-data producers are added in the valuation
  hardening phases.

Any future cashflow implementation therefore requires a separate upstream
specification covering both:

```text
cashflow ingest/schema
+
forward-rate economic source
```

before the valuation compiler can be implemented.

**Positive findings recorded for the future implementation:**

* The D049 effective-discount-factor key

  ```
  (curve_uix, date_uix, fx_eff_uix)
  ```

  is structurally suitable for cashflows: discount curve, payment date and
  FX-to-settlement. The existing EDF design therefore does not require a
  conceptual redesign; the future implementation needs to extend the
  deduplication pool with the cashflow economic key

  ```
  (cashflow_curve, payment_date, fx_c2s).
  ```

* Extending `CompiledPortfolio` with cashflow slots is safe once the inputs
  exist. `simlab/runner.py` and `valuation/engine.py` consume `total_trades` as
  a ROW count, and current code does not assume `num_trades == num_rows`.

The blocker is pinned by:

* `test_cashflow_blocks_are_still_not_production_reachable`
* `test_forward_rate_state_field_has_no_producer`

so the absence remains a tested repository fact rather than an undocumented
assumption.

## D055 — Multi-row trade identity is PROVEN on the production path

**Status:** LOCKED

**Decision:** the row/trade cardinality model

```
one economic trade
→ one or many valuation rows
→ PV_trade = sum(PV_rows)
```

is proven end-to-end through the production valuation path.

The proof uses one trade UID appearing in TWO different production block types,
rather than relying on multiple cashflow rows.

**Reason:** D045/VAL-05 established that `trade_uids` are neither required to be
unique nor sorted, and `ValuationResult.get_trade_pv` is explicitly designed for
that cardinality model. Phase 6 intended to exercise the same invariant through
cashflow decomposition, but cashflow ingest is blocked by D054.

The invariant itself does not depend on cashflows. A UID appearing in both the
forward and option blocks is:

* non-unique;
* non-contiguous;
* distributed across a `block_offsets` boundary;
* required to aggregate multiple valuation rows into one trade PV.

This is sufficient to prove the result-container cardinality semantics.

**Evidence:** production fixture UIDs:

```
[9007, 41, 120003, 9007]
```

UID `9007` owns both the forward row and the final option row.

```
trade_uids : [9007, 41, 120003, 9007]
num_rows   = 4
num_trades = 3

     uid   rows     trade PV (scen 0)       oracle sum     max|rel|
      41     1          -2062.425636        -2062.429386     2.22e-05
    9007     2           69064.040900         69064.061214     4.64e-07
  120003     1            5565.918426          5565.919014     2.81e-07

portfolio  : 72567.533690
oracle     : 72567.550842
max rel    : 2.36e-07
```

The independent oracle groups rows from the fixture's economic ownership, not
from production lookup results.

The 13 dedicated tests prove, among other things:

* one UID can span multiple block types;
* the UID rows can straddle a `block_offsets` boundary;
* `num_rows != num_trades`;
* `get_trade_values()` returns all rows belonging to the UID;
* `get_trade_pv()` is a true aggregation rather than a first-match lookup;
* row aggregation and trade aggregation produce the same portfolio total;
* incorrect trade grouping is detected against the independent oracle.

**Implications:**

* `VAL-05` is CLOSED on the production valuation path.
* The result container is proven for non-unique, non-sorted trade UIDs.
* Future cashflow decomposition may therefore reuse the established row/trade
  model rather than introducing a special cashflow aggregation mechanism.

199 valuation tests pass with no unexplained golden drift and no tolerance
widening.

D046, D052 and D053 are unchanged.

D053 remains OPEN.


## D056 — Cashflow blocks are first-class reusable valuation primitives

**Status:** LOCKED

**Decision:** `FixedCashflowBlock`, `FloatingCashflowBlock`, and
`PrincipalCashflowBlock` are **first-class reusable valuation primitives** in
`riskquant.valuation`.

Their current lack of production reachability from `IngestBatch` does NOT mean
they are dead code, provisional code, or scope that should be removed,
simplified, or downgraded.

The blocks exist intentionally as reusable economic building blocks for future
instrument decomposition and downstream engines, including PFE/XVA use cases,
where a single economic trade may decompose into multiple dated cashflows.

### Rationale

Cashflows are a canonical valuation primitive:

```text
economic trade
    ↓
one or many dated cashflows
    ↓
cashflow valuation
    ↓
trade aggregation
```

Many derivative products can ultimately be represented, wholly or partly, as
collections of fixed, floating, and principal cashflows.

The block family therefore belongs in the core valuation vocabulary independently
of whether the current ingest schema can construct every such cashflow today.

In particular, future products such as vanilla interest-rate swaps are expected
to use these blocks as reusable decomposition/valuation primitives rather than
introducing product-specific cashflow valuation kernels.

### Current boundary

The current repository does NOT provide the upstream contracts required to
construct these blocks from `IngestBatch`.

Specifically:

* cashflow economic fields are absent from the current ingest schema;
* no forward-rate producer currently populates `ValuationState.forward_rates`;
* no production cashflow compiler is therefore implemented;
* the cashflow blocks remain intentionally unregistered/unreachable from the
  current production ingest path.

This boundary is recorded by D054.

### Architectural consequence

The correct interpretation is:

```text
cashflow block exists
        ≠
cashflow block must currently be production-reachable
```

and:

```text
currently unreachable
        ≠
dead / removable / non-first-class
```

Future upstream work may add:

* a cashflow/schedule ingest representation;
* an explicit forward-rate economic source;
* the corresponding compilers;
* `CompiledPortfolio` cashflow slots;
* state producers and UIX mappings.

Such work must preserve the existing cashflow block semantics and must be
specified as a separate upstream/product-construction effort.

### PFE/XVA relevance

The cashflow blocks are deliberately retained as reusable primitives for future
PFE/XVA construction, where dated cashflows and their aggregation are natural
objects for exposure profiling.

This does NOT commit RiskQuant to a particular PFE/XVA trade decomposition,
interest-rate model, curve framework, or cashflow ingest schema.

Those are separate future decisions.

### Non-goals

This decision does not:

* require immediate cashflow production reachability;
* select a single-curve or multi-curve framework;
* define IRS ingestion;
* define a forward-rate model;
* close VAL-13 for cashflows;
* close VAL-04 for cashflows.

Those remain governed by D054 and future dedicated decisions.

### Invariant

The current valuation architecture must preserve the three cashflow blocks as
stable, reusable valuation primitives while keeping their construction boundary
explicit.

**Verdict: LOCKED — RETAIN AS FIRST-CLASS VALUATION PRIMITIVES.**


## D057 — The supported valuation surface is closed; its production consumer is not

**Status:** LOCKED

**Decision:** the three production-supported blocks -- `SpotBlock`,
`VanillaForwardBlock` (NODE) and `VanillaEuropeanOptionBlock` (NODE) -- are
CLOSED as a system: every state universe they consume is populated, every UIX is
range-proven against its own table, and row/trade/portfolio PVs reconcile to an
independent oracle. `ValuationState` has eight fields; seven are produced and
consumed by a supported block, and `forward_rates` is dormant future state whose
only consumer (floating cashflow) has no `CompiledPortfolio` slot.

**Evidence:** one production path, all three blocks, five rows, uids
[9007, 41, 120003, 9007, 5] with 9007 spanning the spot and option blocks.
Worst relative error 2.22e-05 on the deepest option (|d2| = 2.32, inside the
D046 band); spot and forward at 1.29e-16 and 1.77e-15. Eleven semantic
mutations detected, from 1.33e+03 down to 3.44e-02. 225 valuation tests pass.

**However:** `SimLabRunner` -- the only non-demo production consumer -- CANNOT
call the engine. At `simlab/runner.py:163` and `:210` it calls
`run_valuation(portfolio=..., valuation_state=...)` with no `out=`, assigns the
return, and reads `.values`. The current contract is
`(*, portfolio, valuation_state, out) -> None`, so the call raises TypeError and
the attribute read would raise AttributeError. It also passes a
`MarketDataMatrix` where a `ValuationState` is required: SimLab never calls
`assemble_valuation_state` (neither symbol appears in the module).

This is NOT solely a VAL-03 regression. The MarketDataMatrix/ValuationState
split already orphaned SimLab; VAL-03 added a second break on top. Out of scope
for this phase and deliberately not fixed.

**Implication:** certification is scoped to the supported SURFACE. The valuation
package is not release-certified, and the estate has no working end-to-end
revaluation path until the SimLab caller is repaired.

**Valuation-authority audit (2026-09-22).** After P1-B made this route the producer of every V_0 and V_s, the following properties were verified against the independent scalar oracle (`tests/valuation/oracle/economic_oracle.py`) on the hostile whole-surface fixture, and pinned:
- *Admitted surface.* The three blocks above are the only production-admitted valuation surface: `COMPILERS` registers SPOT, VANILLA_FORWARD (NODE) and VANILLA_EUROPEAN_OPTION (NODE). Cashflow blocks remain blocked at ingest (D054).
- *Output.* `run_valuation(out=)` overwrites every cell of `out` (shape `(total_trades, S)`, validated); no caller may rely on pre-zeroed memory. A pricer that writes nothing is visible to the poison check, and a missing pricer raises; it is never skipped.
- *Row identity.* Row r of `out` belongs to `portfolio.trade_uids[r]`: both are laid out from the same `blocks` tuple and `block_offsets`. Rows of one UID are summed by the consumer (D055).
- *Dispatch.* `BLOCK_PRICERS` is indexed by `BLOCK_TYPE_ID`, with an import-time guard against unregistered types. A block type outside the three typed `CompiledPortfolio` fields is unrepresentable.
- *Linearity, sign, scenarios.* V is linear in quantity (reversal negates exactly, zero gives zero). Each scenario column is valued independently: permuting every per-scenario input permutes the output columns bitwise.
- *As-of.* The option time to expiry follows the mapper as-of, and the moved values stay oracle-exact. Spot and forward rows do not move.
- *FX ownership.* Instrument FX (quote → settlement) is folded into the valuation state (effective prices and discount factors). Settlement → reporting translation is applied once, downstream, by D081.
- *Discounting.* NONE means DF = 1 (D079/D080). CURVE builds the D080 zero curve. A CURVE context over a world without curve pillars fails closed.

Evidence: `tests/valuation/oracle/test_valuation_authority.py`; `tests/solutions/risk_run/test_run_owned_identity.py::test_discounting_mode_is_bound_to_identity_or_refused`.


**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.

## D058 — Dense UIX are range-checked, not type-checked; the compiler is the only guard

**Status:** LOCKED

**Decision:** every block UIX is a bare `int32` into a dense table, and the
per-universe guards (`map_*_to_effective_uix` searchsort + `_require_resolved`
bound) prove RANGE, not SEMANTICS. Where two universes overlap numerically, one
universe's integer is a valid index into the other and the swap is silent at
compile time.

**Measured:** in the option block, every stamped `forward_price_strike_ccy_uix`
and `vol_uix` value is in range for the other's table, and `_require_resolved`
accepts `vol_uix` checked against the 4-row EDF universe.

**Reason it is not a defect today:** a compiler cannot accidentally produce the
swap, because each coordinate is resolved by a `map_*` method that searchsorts
its OWN table and raises on a miss, and forward/log-forward are one integer by
construction (VAL-11). The hazard is a future compiler stamping the wrong
resolved coordinate.

**Implication:** the independent oracle is the ONLY layer that can detect a
cross-universe stamp. Every new block compiler must ship an oracle mutation test
that swaps its UIX fields; a range test is not a substitute. Pinned by
`test_cross_universe_uix_swap_is_detected_by_the_oracle`.

## D059 — A mutation test must be proven to move the output before it proves anything

**Status:** LOCKED

**Decision:** every state-mutation test must first assert that the mutation
CHANGES the production output, and only then assert that the independent oracle
rejects it.

**Reason:** during this phase a spot mutation permuting `effective_spot_prices`
rows 0 and 2 passed vacuously -- the spot block consumes row 1. A mutation that
touches rows no supported block reads changes nothing and proves nothing, while
appearing as green coverage.

**Implication:** `test_every_state_mutation_touches_a_consumed_row` guards the
whole mutation matrix. Apply the same discipline to any future mutation suite.

## D060 — One "no FX rf" sentinel; a missing FX quote fails at ingest

**Status:** LOCKED
**Findings closed:** MD-00, MD-08 (second market-data audit)

**Decision:** `riskquant.constants.NO_FX_RF_REQUIRED` is the only representation
of "this leg needs no FX conversion". `RiskStaticConfig.fx_spot_matrix` is filled
with it; no module re-declares it. A negative FX id other than this constant is
an unresolved quote: `lookup_fx_effective_uix` and `_dedup_fx_effective_pairs`
raise on it. A currency pair must be configured in exactly one orientation;
reciprocal FX nodes are rejected at static-config compile.

**Reason:** `static_config._NO_FX_RF_REQUIRED` was a local alias equal to
`NO_ID = -999_999`, while `fx_resolver` compared against `-1`. Every empty matrix
cell therefore read as a valid direct quote. Inverse-only and missing pairs
resolved to `NO_ID`, which downstream code mapped to identity FX. Measured in a
mixed book: −14.39 % PV. The documented missing-pair error was unreachable.

**Implications:** inverse quotes now emit `inv = 1` from production ingest (the
capability D052 assumed). Missing pairs raise `PrettyKeyError`. Pinned by
`tests/market_data/remediation/test_contract_failure_paths.py` (FX section) and
`test_structural_invariants.py::test_no_fx_rf_required_has_exactly_one_definition`.


## D061 - No non-trivial discount-factor or price-curve producer until the curve-point contract exists

**Status:** OPEN - FUTURE CURVE CONSTRUCTION NOT IMPLEMENTED, CURRENT NO-DISCOUNTING MODE SUPPORTED

**Findings:** MD-02a/b/c/d

### Decision

RiskQuant initially operates all pricing with an explicit **no-discounting** mode:

```text
discounting_mode = NONE

DF(as_of, target_date) = 1.0
```

for every valid pricing date `target_date >= as_of`.

This is an explicit economic state and MUST NOT be represented by a missing curve, `NO_ID`, an invalid UIX, or any other sentinel.

A run requesting a non-trivial discount curve or a constructed price curve remains fail-closed:

```text
discounting_mode = CURVE
    -> raise NotImplementedError(D061)
```

before any curve construction, interpolation, extrapolation, or discount-factor evaluation is called.

The canonical market-data snapshot MAY already contain curve risk factors. Those RFs are market-data inputs, not discount factors. Their presence does not imply that a constructed discount curve exists.

No current pricing product requires market-derived discounting. Therefore the absence of a curve constructor is not a blocker for the initial pricing universe.

### Reason

RiskQuant already supports curve RFs as canonical market-data inputs, for example:

```text
USD_SOFR_OIS[1Y] -> 4.80
USD_SOFR_OIS[5Y] -> 4.10
```

with the configured generic market-data representation such as:

```text
quote_type = RATE
unit_scale = PERCENT
x_axis_kind = LABEL
```

What is not yet implemented is the RiskQuant-owned transformation from those curve RF levels into a canonical pricing curve and, ultimately, resolved discount factors.

The missing future contract includes, as applicable:

* semantic meaning of the curve input;
* distinction between par/instrument rates and zero rates;
* quote units and normalization;
* instrument-specific accrual and compounding conventions;
* tenor / maturity construction;
* conversion from market quotes to zero rates or discount factors;
* curve pillar validation;
* interpolation and extrapolation;
* collateral / curve identity where relevant.

A curve-level `CurveInputType` MUST make the economic input representation explicit rather than overloading generic `QuoteType`.

For example:

```text
quote_type       = RATE
unit_scale       = PERCENT
curve_input_type = PAR_RATE
```

and:

```text
quote_type       = RATE
unit_scale       = PERCENT
curve_input_type = ZERO_RATE
```

are distinct economic contracts.

The existing dormant implementation attempted to perform curve construction implicitly inside the market-data matrix builder. That path is contractually invalid because it conflates market quote interpretation, coordinate construction and numerical curve evaluation.

Previously observed defects include:

* configured `RATE` observations being treated as `ln DF`;
* relative tenor offsets being passed where the interpolation kernel expects a time coordinate;
* absolute `int32` date ordinals being passed against an incompatible kernel contract;
* missing LABEL-to-date construction;
* undefined quote accrual / compounding semantics;
* undefined zero-pillar policy.

These MUST NOT be repaired by selecting arbitrary conventions inside a numerical kernel.

### Architectural contract

D061 establishes the following boundary:

```text
canonical market-data RFs
        |
        v
RiskQuant market-data / curve-construction layer
        |
        +-- interpret CurveInputType
        +-- resolve maturities
        +-- bootstrap / convert
        +-- validate pillars
        +-- interpolate
        +-- extrapolate
        +-- construct scenario curve
        +-- evaluate requested dates
        |
        v
resolved discount-factor / price grid
        |
        v
valuation inputs / blocks
        |
        v
pricer kernels
```

All curve intelligence belongs to the market-data / curve-construction side of the boundary.

Valuation kernels consume already-resolved numerical quantities. They MUST NOT:

* interpret `CurveInputType`;
* interpret tenor labels;
* construct maturities;
* bootstrap market instruments;
* infer day-count conventions;
* interpolate curve pillars;
* extrapolate curves;
* convert rates to discount factors;
* construct discount curves.

Interpolation is a market-data operation, never a valuation operation.

### Current implications

The initial pricing universe uses:

```text
discounting_mode = NONE
```

Accordingly:

```text
DF(as_of, as_of) = 1.0

DF(as_of, T) = 1.0
for every valid T >= as_of
```

NODE forwards and NODE options therefore remain reachable through the market-data producer without requiring a market-derived discount curve, provided their other contracts are satisfied.

Their valuation certification remains independent of future curve construction.

The presence of IR curve RFs in the canonical market-data snapshot does not change this. Those RFs are currently unused for non-trivial discounting.

### Future lifting condition

D061 may be lifted for:

```text
discounting_mode = CURVE
```

only after RiskQuant has a locked curve-construction contract covering:

1. curve input semantics;
2. `CurveInputType`;
3. quote normalization;
4. maturity construction;
5. curve construction / bootstrap method where required;
6. canonical curve representation;
7. interpolation and extrapolation;
8. zero/as-of anchor policy;
9. independent producer oracle;
10. fail-fast validation of invalid curve inputs.

The first implementation MAY be a deliberately small construction model. It MUST NOT require the full future multi-curve / IRS architecture merely to establish the first non-trivial discounting path.


**Amended 2026-09-18 by D080:** the "Future lifting condition" is met for exactly one curve. `discounting_mode = CURVE` is live for `USD_SOFR_OIS` under D080 and remains fail-closed for every other curve; the gate moved from "any discount curve present" to per-curve constructibility evaluated in the layout plan. `NONE` semantics are unchanged. The dormant log-DF discount producer described under "Previously observed defects" has been deleted rather than repaired.

## D062 — Packed effective-factor coordinates are in-domain by contract

**Status:** LOCKED
**Finding closed:** MD-03

**Decision:** `_mixed_radix_key` rejects any column value outside `[0, radix)`.
The only admissible negative optional coordinate is `INVALID_UIX` in the FX
column (identity FX). FX invert flags must be 0 or 1. The FORWARD family requires
`delivery_date`, `settlement_date`, `under_discount_curve` and
`strike_discount_curve` at ingest. The OPTION family additionally requires
`settlement_date`, `delivery_date` and both curves.

**Reason:** a missing settlement or delivery date packed as `INVALID_UIX` in a
non-leading key column produced a valid key of another trade's curve/date row.
The D051 range check ran after the lookup and accepted it.

**Implications:** D051 range proofs remain; they are no longer the only guard
against sentinel aliasing.

## D063 — Market-linkage source kind is part of the contract

**Status:** LOCKED
**Findings closed:** NEW-01 (second audit), and the ingest half of MD-01b

**Decision:** enforced at ingest by `portfolio/ingest/identity.py`:

| Linkage | Admissible source |
|---|---|
| SPOT underlying | node or curve point with quote type PRICE, FX_RATE or INDEX |
| NODE forward / NODE option underlying | curve point of a PRICE curve |
| NODE option volatility | node with quote type VOL |

A container id, a rate, or a volatility is never a spot price.

**Reason:** `_resolve_underlying` accepted "rf OR container" for every instrument.
A SPOT on a discount-curve container read an unwritten `prices` row: PV equal to
prior allocator contents.

**Implications:** a SPOT on an interest-rate curve point is rejected. The
canonical base-valuation demo filter was changed accordingly: it now excludes
rate-point spots and prices FX-rate spots.

## D064 — A MarketDataMatrix is bound to the UixMapper that built it

**Status:** LOCKED
**Finding closed:** NEW-03

**Decision:** `UixMapper.fingerprint` is a derived blake2b digest of every
universe, every effective table and the as-of.
`MarketDataMatrix.uix_fingerprint` is required. `assemble_valuation_state`
raises when the two differ.

**Reason:** a matrix and a mapper with equal universe sizes folded silently into
foreign rows (measured: PV 18 518.00 against oracle 17 521.50).

**Implications:** `UixMapper`, `MarketDataMatrix`, `ValuationState` and
`UniverseMapping` use `eq=False` (identity equality, hashable). No semantic
`__eq__` is provided (MD-13).

## D065 — A relative forward point must be the contract the trade delivers

**Status:** LOCKED (CALENDAR_PERIOD only)
**Finding closed:** MD-06

**Decision:** D065 applies to **relative NODE references**, not to every `VANILLA_FORWARD_NODE`.

For a `VANILLA_FORWARD_NODE` whose underlying is an exact NODE identity, ingest uses that node directly. The node's market level is supplied by the resolved NODE/risk-factor identity, so `delivery_date` is **not required for identity resolution**.

For a `VANILLA_FORWARD_NODE` whose underlying is a **relative forward-point name**, ingest resolves the relative name at the run as-of (`rf_resolution`). The trade's `delivery_date` is then required and must fall in the resolved contract period.

Supported relative NODE contracts:

* `OFFSET_M + CALENDAR_PERIOD` (label `YYYYMM`);
* `OFFSET_Y + CALENDAR_PERIOD` (label `YYYY`).

Every other roll method fails closed because its period definition is not owned here.

`as_of_date` is required at ingest when relative NODE resolution is required.

`VANILLA_FORWARD_CURVE` is a separate contract: its underlying is a curve and `delivery_date` is required because the forward level is obtained by interpolation/extrapolation of that curve at the delivery date. D065's relative-NODE contract-period check does not define the curve-forward interpolation contract.

**Reason:** `BRENT_FWD[M+0]` (Feb-2027 at as-of 2027-01-04) accepted a 2029 delivery and consumed the wrong contract's level.


## D066 — A discount linkage is a discount curve in the leg's currency

**Status:** LOCKED
**Findings closed:** MD-04, NEW-02

**Decision:** a linked curve must be a value of `discount_curve_lookup` (not
`NOT_A_DISCOUNT_CURVE`), and its `container_ccy_id` must equal the currency of
the leg it discounts (`under_ccy` for the asset leg, `strike_ccy` for the strike
leg). CSA curves are not an exception: they are keyed `(ccy, basis)` and carry
the cash-flow currency.

## D067 — One run as-of

**Status:** LOCKED
**Finding closed:** MD-11, AUD-04

**Decision:**

* The as-of used at ingest to resolve relative names is carried on
  `IngestBatch.as_of_ord32`.
* An ingest batch may legitimately have no as-of; `NO_DATE_ORD32` denotes
  "no as-of" and is not a valid run as-of value.
* `build_uix_mapper` derives the run as-of from the batches. An explicit
  argument must agree.
* `build_uix_mapper` must reject `NO_DATE_ORD32` when supplied explicitly as
  the run as-of.
* A real calendar date whose ordinal equals `NO_DATE_ORD32` is invalid and
  must fail fast rather than being interpreted as "no as-of".
* `build_market_data_matrix` requires `uix_mapper.as_of_ord32 == as_of_date`.
* `build_market_data_matrix_from_snapshot` requires
  `snapshot.as_of_date == as_of_date`.

**Reason:** snapshot, mapper and matrix as-ofs were independent and accepted in
any combination. Relative names denote different contracts at different as-ofs.
The `NO_DATE_ORD32` sentinel also occupied the same numeric domain as valid
date ordinals, allowing a sentinel value to masquerade as a real run as-of and,
conversely, allowing a real date to be silently erased into "no as-of".

**Implications:** every producer caller passes the as-of to `build_uix_mapper`.
SPOT ingest rows themselves do not require an ingest as-of; a downstream
producer requiring date-bound market state must have a real run as-of.
`SimLabRunner` was not changed (D057 remains).


**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.


## D068 — Expired options are rejected

**Status:** LOCKED
**Finding closed:** NEW-05

**Decision:** `compile_vanilla_european_option_block` rejects `expiry < as_of`.
`expiry == as_of` is priced with `TTE_FLOOR`, which is the documented
intrinsic-limit representation, not an accident.

## D069 — One unit-normalisation engine for level inputs

**Status:** LOCKED
**Findings closed:** NEW-04, MD-10

**Decision:** `compile_risk_factor_level_snapshot_from_levels_dict` is the only
normalisation point. `build_risk_factor_level_snapshot_from_{dict,df,xlsx}`
require `static_config` and delegate to it. Canonical snapshot levels are
float64. The df route rejects missing levels instead of dropping rows.

Scope extended by D076: the history edge (`build_risk_factor_matrix_from_{df,xlsx}`) uses the same engine.

**Reason:** a PERCENT volatility of 18.3 was stored as 18.3 by one route and 0.183
by another, with no unit state on the snapshot.

## D070 — Market-data level finiteness and factor-domain rules

**Status:** LOCKED
**Finding closed:** MD-09a

**Decision:** The market-data level contract is:

* all market levels must be finite;
* FX spot conversion factors must be strictly positive;
* scalar-node levels are strictly positive where their configured quote type
  requires it, currently VOL;
* PRICE, RATE, SPREAD and INDEX levels may take any finite value, including
  zero or negative values.

`RiskFactorLevelSnapshot` and its compiler perform early validation where those
routes apply. The authoritative enforcement boundary for every supported
market-data production route is `build_market_data_matrix`, as defined by
D075.

**Reason:** Market-data domain validity must be independent of the particular
entry route. Snapshot construction historically rejected non-finite values,
and the snapshot compiler rejected non-positive FX_RATE and VOL levels, but
flat levels and other direct routes could bypass those checks.

**Implications:** D075 owns producer-level enforcement. Snapshot-level checks
remain useful as earlier rejection points but are not sufficient by themselves
to establish the producer invariant. Negative PRICE, RATE, SPREAD and INDEX
values remain legal unless a more specific economic contract states otherwise.
`F_K > 0` before the Black-76 logarithm remains a separate invariant owned by
MD-09b.


## D071 — FX gathers translate market rows to dense FX rows

**Status:** LOCKED
**Finding closed:** MD-01

**Decision:** `UniverseMapping.fx_spot_uix_in_universe` records the market row
of each dense FX-block row. `_build_fx_spots_effective` translates every
`fx_effective_pairs` coordinate through it, with a membership proof. A market
coordinate is never used as an FX-block row. Extends D058 to producer-internal
gathers.

## D072 — Every consumed scalar row has a producer owner

**Status:** LOCKED
**Finding closed:** MD-01b

**Decision:** FX levels are scattered into `prices` alongside scalar nodes. Before
state assembly, `build_market_data_matrix` proves that every market row
referenced by `spot_effective_pairs`, `efp_effective_quads` or
`vol_effective_uix` was written, and raises otherwise. `np.empty` rows (curve and
surface containers) may exist only while nothing consumes them.

**Evidence:** memory-poisoning tests (three poison values, S=1 and S=2000) on the
real producer path. An ingest-bypassing container underlying raises
`ValueError(D072)`.

## D073 — Market-data remediation certification boundary

**Status:** LOCKED AS CERTIFICATION BOUNDARY

**Decision:**

* **Certified on the REAL production producer path against independent
  economic oracles:**

  * `SpotBlock` inputs and PV;
  * direct, inverse and identity FX handling;
  * FX-rate underlying handling;
  * sparse and interleaved market-factor IDs;
  * multiple scenario counts;
  * memory-poisoning cases;
  * D059 identifying mutations.

* **Certified valuation surface:**

  * Spot valuation is certified through the supported production market-data
    path and whole-surface valuation oracle.
  * NODE Forward and NODE Option valuation mathematics remain certified only
    for contract-correct raw state.

* **Not certified:**

  * NODE Forward and NODE Option market-data production, because the discount
    curve producer remains blocked by D061.
  * End-to-end SimLab, because the production consumer remains unresolved
    under D057.

* **Deferred / open:**

  * AUD-03;
  * AUD-04;
  * AUD-06;
  * D061;
  * D053;
  * MD-09b;
  * MD-14;
  * non-`CALENDAR_PERIOD` forward roll methods;
  * the `SimLabRunner` consumer.

**Certification evidence:** The current real-producer Spot/FX certification
has been independently re-run under D074, D075 and D076 using hostile sparse
and interleaved fixtures, direct/inverse/identity FX, multiple scenario counts,
memory poisoning and identifying mutations. Current measured worst relative
error is approximately `3.0e-16`. The whole-surface valuation certification
suite executes again under the current contracts, with Spot and Forward
errors at approximately `1.8e-15` relative and the worst observed option
relative error approximately `2.2e-05`, within the established D046 numerical
envelope.

**Boundary:** "Market-data Spot and FX certified" means certified for the
supported production routes and adversarial fixture domain actually exercised
by this evidence. It does not constitute blanket certification of every
possible market-data source, legacy helper or unresolved contract.

## D074 — A compiled portfolio is bound to the UixMapper that stamped it

**Status:** LOCKED

**Findings closed:** AUD-01 (whole-surface fixture landing), AUD-02
(portfolio ↔ mapper identity)

**Extends:** D051, D058, D064

**Decision:** Portfolio-to-market-data identity rests on two invariants. Both
are required.

### 1. Membership: global id → dense UIX

Every translation of a global identifier into a dense universe coordinate
must prove membership.

`universe.translate_ids_to_uix_or_raise` is the single translator for this
identity path.

* Every non-negative identifier must be an element of the target universe.
  Otherwise the translation raises and returns no coordinate.
* Negative sentinel inputs supported by the translator represent absent
  linkage and map to `INVALID_UIX`.
* The translator is used by:

  * `UixMapper.map_market_ids_to_uix`;
  * `UixMapper.map_discount_ids_to_uix`;
  * `UixMapper.map_dates_to_uix`;
  * stage 1 of `lookup_fx_effective_uix`;
  * every relevant producer translation in `portfolio/engine.py`.
* `_arrays.searchsorted_or_propagate_sentinel_int32` remains an unsafe
  left-join primitive and has no caller in the identity path.

Therefore an absent non-negative `rf_id`, curve id or date cannot silently
become the insertion position of another valid universe member.

### 2. Mapper identity: portfolio ↔ state ↔ mapper

* `CompiledPortfolio.uix_fingerprint` is stamped by
  `compile_portfolio_from_ingest_batch` from the `UixMapper` that stamped the
  portfolio block UIXs.
* `ValuationState.uix_fingerprint` is stamped by
  `assemble_valuation_state`.
* Both fields are required and non-empty.
* `run_valuation` raises when portfolio and state fingerprints differ, before
  output validation and before any pricer runs.
* `run_sensitivities` raises when the portfolio or matrix fingerprint differs
  from the mapper fingerprint.
* The fold retains the D064 matrix ↔ mapper check.

The identity token is the existing `UixMapper.fingerprint` defined by D064.
No mapper copy is stored inside the portfolio.

**Reason:** `np.searchsorted` returns an insertion position for an absent id.
That position can be a valid dense row and therefore pass every D051 range
check described by D058. D064 alone binds the matrix to its mapper but does
not bind a compiled portfolio to the mapper that stamped its UIXs.

Membership and object identity are therefore separate and both necessary:

```text
rf_id absent from mapper
    → membership proof fails

portfolio compiled with mapper A
    +
state / mapper B
    → fingerprint mismatch fails
```

**Implications:**

* A portfolio must be compiled against a mapper whose universes contain every
  identity it consumes.
* A compiled portfolio may only be paired with state produced from the same
  mapper identity, represented by equal fingerprints.
* Market levels are not part of mapper identity. Rebuilding market state with
  new levels on the same universe preserves the fingerprint.
* Hand-built kernel/test portfolios and states may declare one explicit shared
  token because they intentionally bypass production mapper construction.
  This is a raw test boundary; production compilation stamps the token from
  the mapper and does not permit callers to choose an arbitrary identity.
* `SimLabRunner` remains governed by D057 and must eventually provide a
  portfolio, mapper and valuation state from the same production lineage.


**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.

**Evidence:**

* `tests/market_data/remediation/test_portfolio_mapper_identity.py`
  covering absent ids between, below and above the universe, absent FX,
  absent dates, sparse/interleaved ids, same-shape foreign mappers,
  sensitivity pairings, D064 preservation and D059 consumption tests.
* `tests/valuation/oracle/test_whole_surface.py` with the current
  `uix_fingerprint` contract.
* Independent economic evidence demonstrated wrong valid PVs before the fix
  and fail-fast rejection afterward.


## D075 — The market-data producer owns the level-domain contract

**Status:** LOCKED
**Finding closed:** AUD-05
**Amends:** D070 (ownership)

**Decision:** `build_market_data_matrix` validates every level before any
allocation, scatter or effective-FX construction, whatever route supplied it
(compiled snapshot, direct snapshot, history snapshot, flat levels,
scenario-shocked levels):

* all levels finite;
* FX spot rows > 0 (FX_RATE by config; conversion factor by role);
* scalar-node rows > 0 where `UniverseMapping.scalar_node_strictly_positive`
  is set;
* curve and surface levels finite.

The flag is resolved once per universe from `rf_id → quote type` by
`strictly_positive_level_mask`, the same rule set as the snapshot compiler.
PRICE, RATE, SPREAD and INDEX accept any finite value.

**Reason:** flat levels bypassed both snapshot checks. Observed:
`fx_spots_effective = [-1.0989]` for FX −0.91, `[inf]` for FX 0, and VOL ≤ 0
stored in `prices`.

**Implications:** snapshot-layer checks remain as early rejections. Scenario
batches are validated per batch (vectorised) because each carries new levels.
`MarketDataMatrix.__post_init__` stays structural; hand-built matrices are a
test-only raw boundary. `F_K > 0` (MD-09b) remains separate.

**Evidence:** `tests/market_data/remediation/test_level_domain_contract.py`
(50 tests: every category on every route at S ∈ {1, 4096}; builder never
reached; route consistency; D059 mutations; oracle reconciliation).


## D076 — History inputs normalise once, at the matrix edge

**Status:** LOCKED

**Finding closed:** AUD-06

**Extends:** D069, D017

**Decision:** `build_risk_factor_matrix_from_df` and
`build_risk_factor_matrix_from_xlsx` require `static_config`.

They:

* resolve history columns by relative name against the configured factor
  universe;
* reject unknown or unoccupied identifiers;
* apply `canonical_unit_scale` exactly once;
* use the same canonical unit-normalisation engine defined by D069;
* produce `RiskFactorMatrix.values` in canonical base units.

The `RiskFactorMatrix` constructor, `from_snapshot`, `get_snapshot` and
scenario generators do not rescale values.

`name_to_id_map` and integer-column input are removed from the supported
history-builder interface.

**Reason:** The history edge previously copied source-unit values directly
into `RiskFactorMatrix.values`, while downstream consumers treated those
values as canonical. This produced economically material scaling errors,
including:

* PERCENT volatility `18.3` remaining `18.3` instead of `0.183`;
* a PERCENT-quoted price producing a Spot PV approximately 100× too large;
* DIFFERENCE historical shocks being calculated against inconsistent units.

**Implications:**

* The source-unit history edge owns conversion from declared source units to
  canonical units.
* `canonical_unit_scale` is the single unit-normalisation engine shared with
  the snapshot compiler.
* Unit interpretation is metadata-driven from configured `rf_id` semantics;
  numeric magnitude is never used to infer units.
* Canonical units and canonical dtype are separate contracts: unit semantics
  are owned here and by D069, while history dtype and kernel compatibility are
  governed by D017.
* Feeding already-canonical values into the source-unit history edge is a
  caller contract violation and cannot be detected reliably from magnitude.
* LOG_RETURN and SIMPLE_RETURN scenario semantics remain unchanged by this
  normalisation because they are scale-invariant. DIFFERENCE shocks operate
  on canonical base levels after this decision.
* `RiskFactorMatrix.get_snapshot` has no production caller and is retained as
  a helper under the same canonical-unit contract.

**Evidence:** `tests/market_data/remediation/test_history_unit_contract.py`
covers the supported quote families against an independent unit oracle,
dict/df/xlsx/history route equivalence, Spot PV and historical-scenario
reconciliation, D059 mutations, fail-closed metadata handling, AUD-05 domain
validation and D074 identity preservation.

Current evidence shows equivalent supported routes agreeing within the
declared numerical tolerance, with the independent economic oracle deriving
unit scales independently from configured unit semantics.


## D077 — NODE and CURVE are market-data specifications, never pricing products

**Status:** LOCKED
**Findings closed:** F-1 (NODE option delivery), F-3 (D063 justification)
**Amends:** D063 (wording only), D065 (option half)
**Open, deliberately unresolved here:** OQ-1, OQ-2, OQ-3

**Decision:**

1. A configured NODE rf and a configured CURVE/SURFACE POINT rf are both
   **EXACT MARKET POINTS**: each resolves to exactly one `rf_id`, occupies one
   `market_id_universe` row, and is read directly. `MarketDataMatrix.prices`,
   `.forward_nodes` and `.vol_nodes` are one array; `assemble.py` deliberately
   does not branch on `rf_id_to_parent_topology_type` for scalar reads.

2. An exact market point is **NEVER** interpolated, and no instrument date
   selects **WHICH** exact market point is read.

3. A **CURVE or SURFACE CONTAINER** is a market object requiring coordinate
   resolution. Resolution is performed by the market-data layer before the
   resulting numeric market input is consumed by the pricing block.
   Interpolation never occurs in a valuation kernel.

4. A relative name that denotes an exact market point, together with the
   **RUN as-of**, resolves to exactly one `rf_id`, fixed for the run
   (D067, D064, D074). Expiry and delivery never re-resolve that RF.
   Relative container names resolve to their container identity and are
   governed by the container-resolution contract in §3.

5. NODE vs CURVE does not define separate economic pricing products. One
   economic instrument uses one block and one kernel. Any residual source
   distinction required to obtain a valuation input is represented at the
   compile/fold boundary, never in the block taxonomy or the kernel contract.

6. After the fold, a block holds only quantities, strikes, precomputed scalars,
   and integer row coordinates. It holds no names, containers, topologies, or
   lookup dates.

7. **Date requirements:** a date is required only as a container coordinate,
   an argument of the pricing formula, or an independently justified economic
   property. It is never required because a sibling instrument uses it.

   * **NODE forward:** delivery REQUIRED (asset-leg discount date); settlement
     REQUIRED.
   * **NODE option:** expiry REQUIRED; settlement REQUIRED; delivery NOT
     REQUIRED.

8. D063 stands unchanged as a **SCALAR QUOTE SEMANTICS** gate. Its current
   topology condition is a configuration-level proxy for the forward-quote
   semantics supported by the present market-data model. It is **not** an
   interpolation rule. Whether broader scalar quote semantics are permitted
   is deferred to OQ-1/OQ-2/OQ-3.

9. The `InstrumentType` NODE/CURVE split is deferred technical debt. It is
   removed only when CURVE-sourced valuation consumers exist and the required
   compile/fold representation has been established.

**Reason:** delivery was required for NODE options solely because the option
family inherited the forward's `dated` classification. Evidence:
the option compiler never reads `delivery_ord`; on the production-path option
fixture, erasing or moving delivery leaves all four PVs bitwise identical and
still matching the independent oracle at 2.22e-05. By contrast, moving a
forward's delivery moved its PV from 176.154 to 151.070, and erasing it fails
closed.

**Implications:** canonical option rows 304/305 now ingest and compile.
Rows 300-303 remain rejected by D063 pending OQ-1/OQ-2. Canonical forward
fixtures are unchanged pending OQ-1/OQ-3.

**Evidence:** `tests/market_data/remediation/test_node_option_delivery_contract.py`
(5 tests) and `tests/valuation/oracle/test_option_delivery_independence.py`
(2 tests). Full suite: 4,083 passed; the failure and error set is identical
to baseline.

**Open questions (NOT resolved here):**

* **OQ-1:** May a spot-quoted node be a forward/option underlying, carried to
  T by an asset-leg carry curve?

* **OQ-2:** For an FX underlying, the asset is the BASE currency but the
  derived leg currency is the QUOTE currency, so D066 rejects the CIP carry
  curve. What is the correct asset/carry-leg currency contract?

* **OQ-3:** How should an identity (undiscounted) asset leg be represented,
  if such a leg is required by the supported product contract?


## D078 — Scalar Underlying Quote Semantics

**Status:** LOCKED
**Closes:** OQ-1, OQ-2 (forward-quote half), OQ-3; audit findings G-2, G-9
**Amends:** D063 (wording only)
**Depends on:** D077 (market-data topology), D066 (discount linkage)

**Decision**

1. **D078.1** A scalar underlying consumed by the current vanilla forward /
   European-option model MUST represent a **forward price for the relevant
   underlying contract**, expressed in the currency in which that forward price
   is quoted.
2. **D078.2** The current model MUST NOT derive that forward price from a spot
   level through an implicit carry assumption. No carry transformation exists in
   the engine.
3. **D078.3** The asset-leg factor in the current forward-quote model is a
   **discount factor in the currency of the quoted forward**. `F x DF` is the
   present value of that leg, which is what D066 enforces.
4. **D078.4** For FX, `BASEQUOTE` means **quote-currency units per one
   base-currency unit** (`fx_base_ccy` / `quote_ccy` in config, resolved through
   `fx_spot_matrix` with `invert = 1` for the reciprocal). A forward-quoted FX
   underlying uses quote-currency discounting under D078.3.
5. **D078.5** A spot level MUST NOT be accepted as a forward/option underlying
   under the current contract. This is not a universal mathematical
   prohibition: it is outside the supported model because the carry semantics
   that would make it correct (per-asset-class carry object, spot/forward
   declaration) do not exist. Forcing a spot level through the present formula
   prices a carry-free forward; with a EUR/USD example (S = 0.9137,
   K = 0.9000, DF_USD = 0.9560, DF_EUR = 0.9800, Q = 1e7) the result is
   +134,260 EUR against an independent covered-interest-parity value of
   -85,028 EUR, a sign flip.
6. **D078.6** `DF(0,T) = 1` is a valid numerical discount-factor outcome,
   including for `T > as-of`. It is ordinary market data and does not by itself
   imply a SPOT trade or any special state.
7. **D078.7** `NO_ID`, invalid UIX values and other unresolved/missing-data
   sentinels MUST NEVER be interpreted as `DF = 1`. Unresolved discounting
   fails closed at coordinate packing.
8. **D078.8** If a future product requires an explicit "no discounting
   required" economic state, it MUST be represented and validated as an explicit
   economic state, distinct from missing-data sentinels, in the pattern already
   used by `NO_FX_RF_REQUIRED` for identity FX. No currently identified product
   needs it.
9. **D078.9** The following remain **future product decisions**, not current
   implementation defects:
   * **U-1** whether spot-underlying forwards/options are wanted at all;
   * **U-2** which carry model applies by asset class (FX base/quote funding;
     equity dividend or repo; commodity convenience yield), with day count and
     compounding;
   * **U-3** how spot-versus-forward semantics are declared in configuration
     (topology is only a present-day correlate, never a proof — D077);
   * **U-4** whether any product genuinely needs explicit no-discounting
     semantics.

**Reason.** The valuation algebra is the dated-forward model and nothing else.
Control: with a forward quote (F = 82.35, K = 80, DF = 0.956, Q = 1e7)
production returns 22,465,999.999999937 against an independent
`Q(F - K)DF` of 22,465,999.999999944. Substituting a spot level is a sign flip
for FX (above) and for equity (S = 185.50, K = 190, DF = 0.956: production
-4,302 against +3,860 with zero dividends); production would be right there
only if the asset-leg factor were `e^{-qT}` with q = 4.4997%, i.e. a dividend
or repo curve, which the configuration does not contain.

**Implications.** D063 stays restrictive until U-1..U-3 are answered. Canonical
forward fixtures 100-105 are classified, not modified. The option half of D065
(D077) is unaffected.

**Evidence.** `tests/market_data/remediation/test_scalar_underlying_quote_semantics.py`
(20 tests): two-leg forward pricing with non-unit per-leg DFs; `DF = 1` as
ordinary data; `NO_ID` failing closed; the D063 predicate table; the spot
sign-flip counterexample; and the fixture classification for uids 100-105.


## D079 - RiskQuant-owned curve-input, curve-construction and canonical discount-factor contract

**Status:** PARTIAL LOCK

**Purpose:** Define what RiskQuant requires in order to transform configured curve RFs into resolved pricing curves and discount-factor grids. This is a RiskQuant-owned contract. It is not inferred from the implementation of any external market-data producer.

The initial production state is:

```text
discounting_mode = NONE
```

The clauses below therefore distinguish the current production contract from the future:

```text
discounting_mode = CURVE
```

capability.

### A. Current production contract

#### 1. NO-DISCOUNTING MODE

`discounting_mode = NONE` is an explicit supported economic state.

For every valid target date:

```text
DF(as_of, target_date) = 1.0
```

`DF = 1` MUST NOT be inferred from:

* missing data;
* `NO_ID`;
* invalid UIX;
* absent curve;
* NaN / infinity;
* failed lookup.

Those conditions remain invalid and MUST fail fast.

#### 2. CANONICAL PRICING INTERFACE

Valuation consumes resolved numerical discount factors, not raw curve quotes.

The future curve-construction subsystem MUST ultimately produce:

```text
DF(as_of, target_date, scenario)
```

in the currency of the curve.

Pricing kernels MUST NOT consume tenor labels, curve RFs or raw market rates as substitutes for discount factors.

---

### B. Curve RF semantic contract

#### 3. CURVE INPUT TYPE

Every curve specification that is eligible for future curve construction MUST declare an explicit:

```text
CurveInputType
```

at the **curve level**, not independently on each curve point.

`CurveInputType` is orthogonal to the generic market-data `QuoteType`.

For example:

```text
quote_type       = RATE
unit_scale       = PERCENT
curve_input_type = PAR_RATE
```

means the RFs represent market instrument rates requiring the appropriate curve construction / bootstrap.

Whereas:

```text
quote_type       = RATE
unit_scale       = PERCENT
curve_input_type = ZERO_RATE
```

means the RFs already represent zero rates and do not require a par-instrument bootstrap.

A future implementation MAY also support:

```text
curve_input_type = DISCOUNT_FACTOR
curve_input_type = LOG_DISCOUNT_FACTOR
```

where the configured RF values directly represent those quantities.

RiskQuant MUST NOT silently infer one `CurveInputType` from `QuoteType`, tenor labels, curve names, or numerical values.

`CurveInputType` belongs to configuration / market-data metadata.

It MUST NOT leak into valuation blocks or pricing kernels.

#### 4. QUOTE ACCRUAL

Where a construction model is based on quoted market instruments, the instrument's accrual convention MUST be explicit in the RiskQuant construction specification.

Current intended conventions are:

| Curve family     | Accrual  | Spot lag | Calendar                     |
| ---------------- | -------- | -------: | ---------------------------- |
| SOFR OIS         | ACT/360  |      T+2 | US                           |
| LIBOR 3M         | ACT/360  |      T+2 | London + NY                  |
| CSA_EUR, USD leg | ACT/360  |      T+2 | US / relevant CSA convention |
| ESTR OIS         | ACT/360  |      T+2 | TARGET2                      |
| EURIBOR 6M       | ACT/360  |      T+2 | TARGET2                      |
| SONIA OIS        | ACT/365F |      T+0 | London                       |
| SARON OIS        | ACT/360  |      T+2 | Zurich                       |
| AONIA OIS        | ACT/365F |      T+1 | Sydney                       |

Business-day convention is Modified Following with the applicable end-of-month rule.

These are construction-model conventions. They are not required by valuation once the canonical discount-factor grid has been produced.

They are relevant only where the declared `CurveInputType` requires instrument-level interpretation.

#### 5. LABEL MATURITY

Where the curve construction specification represents instruments by tenor labels, LABEL-to-date resolution MUST occur in the market-data / curve-construction layer.

For the currently intended convention:

```text
spot = business_day_add(as_of, spot_lag, calendar)

ON:
    next_business_day(as_of, calendar)

1W:
    adjust(spot + 7 calendar days)

nM / nY:
    adjust(add_months/add_years(spot, n))
```

where `adjust` applies Modified Following with the applicable end-of-month rule.

`3M` MUST NOT be interpreted as `0.25` years or as exactly 90 calendar days.

#### 6. ON IS NOT THE ZERO ANCHOR

`ON` is a market instrument / pillar with its own maturity.

It MUST NOT be conflated with:

```text
DF(as_of, as_of) = 1.0
```

The as-of zero anchor is a mathematical identity and is separate from the ON market observation.

#### 7. ZERO / AS-OF ANCHOR

For a constructed curve:

```text
T(as_of) = 0
DF(as_of, as_of) = 1.0
```

The zero anchor MUST be synthesized by the curve-construction layer.

It MUST:

* participate as the left-most curve point;
* not be sourced from the feed;
* not be scenario-shocked.

#### 8. PILLAR VALIDITY

Before numerical curve evaluation, the market-data / curve-construction layer MUST validate:

* finite coordinates;
* finite curve values;
* strictly increasing curve coordinates;
* no duplicate coordinates;
* no zero-value market pillars;
* valid relationship to the as-of anchor.

A single valid market pillar plus the synthetic anchor is structurally permissible.

No claim that such a structure represents a particular market extrapolation or "flat" economic convention is implied by that structural allowance.

#### 9. CANONICAL GRID MEANING

For a constructed discount curve:

```text
discount_factor_grid[curve, date, scenario]
    = DF(run_as_of, date, scenario)
```

in the curve's `quote_ccy`.

Required invariants:

```text
date == as_of  -> exactly 1.0
date < as_of   -> fail
DF > 0         -> always
DF >= 1        -> legal for non-positive rates
```

This grid is a **market-data output**.

Valuation receives the resolved grid values. It does not perform curve evaluation.

#### 10. TIME COORDINATE

The canonical curve evaluation coordinate SHOULD be:

```text
T = ACT/365F(as_of, target_date)
```

regardless of the quoted instrument's own accrual convention.

The instrument-specific accrual convention belongs to quote interpretation / curve construction.

It MUST NOT be silently reused as the universal curve interpolation coordinate merely because it was used to quote the source instrument.

#### 11. INTERPOLATION

The initial proposed model for a constructed discount curve is:

```text
linear interpolation in ln(DF)
```

on the canonical `T` coordinate, exact at pillars.

This interpolation occurs entirely inside the market-data / curve-construction layer.

It is never performed by valuation or pricing kernels.

This remains a RiskQuant model decision and is not inferred from the existence of the market quotes.

#### 12. EXTRAPOLATION

Left extrapolation is unnecessary because the curve is explicitly anchored at `T = 0`.

For the right side, the proposed initial model is **flat instantaneous forward**:

```text
extend ln(DF) using the final pillar interval slope
```

Flat DF extrapolation MUST NOT be adopted merely because it is convenient numerically.

This remains a model decision until the curve-construction specification is formally locked.

#### 13. SCENARIO SHOCKING

For rate-based market inputs:

```text
normalised decimal RATE
        |
        v
scenario shock
        |
        v
curve construction / rate -> DF
```

Scenario shocks MUST therefore be applied to the normalized rate before the rate-to-curve / DF transformation.

A discount factor MUST NOT be scenario-shocked additively.

Maturity construction is independent of scenario shocking.

#### 14. UNITS

D069 normalization applies exactly once at the normalized market-data boundary.

For a PERCENT rate:

```text
5.25% -> 0.0525
```

No downstream curve-construction or pricing component MAY apply another unit conversion.

#### 15. CURRENCY / CURVE IDENTITY

A discount factor belongs to the curve's declared `quote_ccy` and discounts cash flows denominated in that currency.

Curve identity MUST NOT be reduced to currency alone.

The identity MUST be capable of distinguishing economically distinct curves such as:

```text
USD OIS
USD under EUR CSA
USD projection
```

where such curves exist.

A collateral basis such as `CSA_EUR` may alter curve construction and level, but does not change the currency of the resulting discount factor.

---

### C. External feed boundary

#### 16. RISKQUANT OWNS THE CONTRACT

The canonical snapshot is already a valid RiskQuant market-data boundary.

For example:

```text
USD_SOFR_OIS[1Y] -> 4.80
```

is a curve RF level.

It is not a discount factor.

RiskQuant defines the economic meaning required for that RF through curve metadata, including `CurveInputType`.

A future external producer MUST emit observations satisfying the RiskQuant-owned contract.

The producer does not define the RiskQuant contract.

Whether a future source provides:

```text
par instrument quotes
```

or:

```text
already-constructed zero rates
```

is therefore represented explicitly by the RiskQuant-side curve metadata rather than inferred from the producer implementation.

#### 17. CURVE CONSTRUCTION METHOD

The construction method remains a future implementation decision determined by `CurveInputType` and the corresponding curve specification.

Examples:

```text
PAR_RATE
    -> bootstrap / instrument-based construction

ZERO_RATE
    -> rate-to-DF conversion

DISCOUNT_FACTOR
    -> direct validated curve construction

LOG_DISCOUNT_FACTOR
    -> direct validated curve construction
```

RiskQuant MUST NOT silently reinterpret one input representation as another.

A vendor-supplied zero curve and a RiskQuant-bootstrapped par-rate curve are both valid future implementations provided they produce the same canonical downstream contract.

#### 18. USD_CSA_EUR

`USD_CSA_EUR` requires an explicit construction specification.

Its presence MUST NOT be treated as sufficient evidence that its construction is equivalent to USD SOFR OIS with a different label.

Until that construction is specified, the curve remains unsupported in `discounting_mode = CURVE`.

---

### D. Architectural consequence

The definitive boundary is:

```text
                    CANONICAL SNAPSHOT
                           |
                +----------+----------+
                |                     |
            NODE RFs              CURVE RFs
                |                     |
                |                     v
                |          MARKET-DATA / CURVE LAYER
                |                     |
                |          interpret CurveInputType
                |          resolve maturity
                |          bootstrap / convert
                |          build / validate pillars
                |          interpolate
                |          extrapolate
                |          apply scenario
                |          evaluate requested dates
                |                     |
                |                     v
                |             RESOLVED DF GRID
                |                     |
                +----------+----------+
                           |
                           v
                    VALUATION INPUTS
                           |
                           v
                      PRICER KERNEL
```

The pricer sees only resolved numerical quantities.

The pricer MUST NOT know that the discount factor came from:

* an OIS bootstrap;
* a vendor zero curve;
* a direct DF curve;
* interpolation;
* extrapolation;
* a particular tenor label;
* a particular day-count convention;
* a particular calendar.

The initial production path is:

```text
discounting_mode = NONE
        |
        v
DF = 1
        |
        v
valuation
```

The future production path is:

```text
curve RFs
        |
        v
market-data / curve construction
        |
        v
resolved DF grid
        |
        v
valuation
```

This preserves the "kernels just do dumb math" boundary while allowing RiskQuant to own the full future curve-construction contract.


**Amended 2026-09-18 by D080:** §10 (time coordinate), §11 (interpolation) and §12 (extrapolation) are LOCKED as written for `USD_SOFR_OIS`; §13 gains the clause that DF-valued and log-DF-valued inputs have no shock contract and MUST NOT be constructed; §17 is answered for `ZERO_RATE`: `DF = exp(-zT)` with `curve_compounding = CONTINUOUS`. The axis-method fields resolve to Option B (price-curve/surface only). §18 (`USD_CSA_EUR`) remains OPEN.


## D080 - First non-trivial discount curve: USD_SOFR_OIS as a continuously compounded zero curve

**Status:** LOCKED (Phases 1-3 implemented and certified)
**Date:** 2026-09-18
**Related:** D061 (mode and fail-closed gate), D066 (curve currency), D067 (run as-of), D069 (single normalisation), D077 (topology), D078 (DF semantics), D079 (curve-input contract)
**Supersedes:** the `D080 - TODO` placeholder. Amends D079 §10-§13 and §17 (see D080.9).

### Context
D061 left `discounting_mode = CURVE` fail-closed because no economic convention existed for turning configured curve levels into discount factors. D079 fixed the surrounding contract but left the time coordinate, interpolation, extrapolation and the first input representation as open model decisions. Three implementation gaps blocked any constructor regardless of those decisions: LABEL discount curves contributed no pillars to the run universe, no business-day engine existed, and `curve_meta` could not express calendars, spot lags, business-day conventions, compounding or end-of-month behaviour.

### Decision
**D080.1 Scope.** Exactly one curve is constructible: `USD_SOFR_OIS` (`curve_id 250000000`, USD, basis OIS, LABEL axis, pillars ON/1W/1M/3M/6M/1Y/2Y/5Y/10Y), with `curve_input_type = ZERO_RATE` and `curve_compounding = CONTINUOUS`. Constructibility is tied to **curve identity** (`D080_CONSTRUCTIBLE_CURVE_IDS = {250000000}` in `market_data/assemble.py`), keyed by `curve_id` and never by currency or `(ccy, basis)`: another curve declaring the same metadata fails closed until a decision adds its identity. No other curve, input type, compounding, currency or basis is constructible. *(Clarified at closure: the identity gate is enforced in code, not only by which fixture rows are declared.)*

**D080.2 Metadata.** `CurveMeta` carries six nullable, curve-level construction fields: `curve_input_type`, `curve_compounding`, `maturity_calendar`, `spot_lag_days`, `business_day_convention`, `end_of_month`. They reuse the existing `CurveInputType`, `COMPOUNDING_CONVENTIONS` and `BusinessDayConvention` vocabularies; no new enum was created. A curve is **constructible** only when all six are declared and coherent; absence is `None`, never a default. Curve points inherit through `curve_id`.

**D080.3 Canonical calendar stays sparse (C1, option b).** The canonical US holiday fixture is intentionally sparse (four holidays per year) and is NOT extended by D080. Real-world completeness of any holiday calendar is outside D080. Consequently, for `as_of = 2026-01-15`, `spot = 2026-01-19` (2026-01-19 is not a fixture holiday), and every D080 worked example uses that fixture-true value. Alternative rejected: adding MLK and other federal holidays, because it changes fixture semantics for every future consumer of the calendar.

**D080.4 LABEL vocabulary: rejection, not inference.** `XAxisKind.LABEL` admits `ON, SN, TN, nD, nW, nM, nY`. D080 defines maturities only for `ON, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y`. `SN`, `TN` and `nD` are rejected by the D080 maturity layer with a named error; their semantics are not inferred. The generic LABEL axis is NOT narrowed.

**D080.5 Maturity construction.** `spot = business_day_add(as_of, spot_lag_days, calendar)`; `ON = next_business_day(as_of)` (bypasses spot, never T+3); `1W = adjust(spot + 7 days)`; `nM/nY = adjust(add_months/add_years(spot, n))`; `adjust` = the curve's business-day convention (MOD_FOLLOWING for USD_SOFR_OIS) with the end-of-month rule: if **spot** is its month's last **business** day, the maturity is the target month's last business day. Maturities are scenario-independent and are built once per run. The engine (`riskquant.utils.business_days`) is pandas-free and operates on the compiled `HolidayCalendar.holiday_dates_ord` int32 axis.

**D080.6 Canonical coordinate.** `T = (date_ord - as_of_ord) / 365.0` (ACT/365F, `DAYS_PER_YR_ACT_365F`), float64, for pillars and targets alike. It is independent of the instrument's own accrual convention (SOFR ACT/360 is not used) and distinct from the option `tte_grid` (ACT/365.25), which is untouched.

**D080.7 Mode-gated universe.** `build_universe_mapping` takes `discounting_mode` with no default and carries it on `UniverseMapping`. Under `NONE` no discount pillar is emitted and no discount level is required. Byte identity with the pre-D080 form was measured for exactly four artefacts, and the guarantee is stated at that scope: `UniverseMapping.required_factor_ids`, `shape_discount_curves`, `raw_discount_tenors.shape`, and `MarketDataMatrix.discount_factor_grid` (all ones). Beyond those artefacts, the evidence is semantic rather than a blanket bitwise guarantee: the full-suite failure and error set is unchanged from baseline, and the D019 golden VaR/ES remains unchanged.

**D080.8 Canonical ordering.** Source row order carries no semantic meaning: the layout plan resolves every pillar to its maturity and canonical `T`, then sorts by `T`; an unsorted snapshot is never rejected for being unsorted. After sorting the coordinate sequence MUST be strictly increasing, so duplicate maturities/coordinates are a hard failure (never last-write-wins). Pillars with `T <= 0`, zero market pillars and non-finite levels are hard failures. A single market pillar plus the synthetic anchor is structurally valid. *(Amended at closure: the earlier phrase "unsorted-unfixable input" was misleading and is withdrawn.)*

**D080.9 Mathematics (amends D079 §10-§13, §17 from proposed/open to LOCKED for this curve).**
- `z` is the D069-normalised decimal zero rate, already scenario-shocked additively upstream (`shift_type = DIFFERENCE` on the rate). The constructor never rescales.
- `lnDF_p = -z_p * T_p` per pillar and scenario.
- Anchor `(T = 0, lnDF = 0)` is synthesised per curve, prepended, unshocked, and not a market pillar; `DF(as_of, as_of) = 1.0` exactly.
- Interpolation: **linear in `lnDF` on `T`**, exact at pillars (piecewise-constant instantaneous forwards, `f = -d lnDF/dT`).
- Extrapolation: left does not exist (anchored); right **extends the final `lnDF` slope** (`s_last = (lnDF_K - lnDF_{K-1})/(T_K - T_{K-1})`, `forward_last = -s_last`). Flat-DF extrapolation is forbidden.
- `DF = exp(lnDF)`, strictly positive; `DF > 1` under negative rates is valid and never clipped.
- Targets with `T < 0` fail before construction.

**D080.10 Kernel reuse.** The constructor (`market_data.curves.zero_curve`) reuses the existing `interp_linear_on_log_values_linear_extrap_3d_numba_kernel`, whose semantics are exactly D080.9 (linear in supplied log values, unclamped right tail). The `..._flat_extrap_...` sibling clamps the weight and would assert zero forward rates; it is not used for discount curves. The dormant `builder._build_discount_factor_grid`, which assumed log-DF quotes and int32 targets, is deleted. The builder's level argument is renamed `raw_discount_levels` because it carries rates, not log-DFs.

**D080.11 Axis-method fields (D079 Option B).** `x_interp_method` / `x_extrap_method` govern price-curve and surface axes only. They are not consulted for discount curves; the discount policy is D080.9.

**D080.12 D061 transition.** `NONE -> DF = 1` unchanged. `CURVE + constructible USD_SOFR_OIS -> construct`. `CURVE + any non-constructible curve -> D061 error at the layout plan`. A CURVE plan carrying a discount curve with no pillars is a wiring defect and raises; it never falls back to `DF = 1`.

### Rationale
The canonical fixture values (5.30 … 4.00 PERCENT at ON … 10Y) do **not** mathematically distinguish `ZERO_RATE` from `PAR_RATE`: both readings are consistent with those numbers, and the feed contract that would settle it lives in `stx-data-god`, outside this repository (D079 §16). `ZERO_RATE + CONTINUOUS` was therefore selected on constructibility grounds, not on evidence from the fixture: it is the only candidate that needs no bootstrap or schedule machinery (none exists in `riskquant`), already has a locked shock contract (D079 §13), and admits a low-complexity independent oracle. Declaring `curve_input_type = ZERO_RATE` on `USD_SOFR_OIS` is a **data-contract assertion by RiskQuant** about what its feed publishes. If the feed is later confirmed to publish par rates, this declaration must change and D080 must be reopened for that curve; the remaining clauses (coordinate, anchor, interpolation, extrapolation, shock ordering) are independent of that choice and stand. Linear-in-`lnDF` is the one candidate whose economic content is a statement about forwards and which is positivity-preserving under negative rates. Flat-forward extrapolation avoids the zero-forward artefact of flat DF (503 bp at 12y, 2 165 bp at 20y on the canonical curve). Mode-gating is required because unconditional pillar emission would change the meaning of every existing NONE call.



### Consequences
NODE forwards and options price with market-derived USD discount factors under `CURVE`. Valuation blocks, `ValuationState`, the fold and every kernel are unchanged: the constructor fills the same `(C, D, S)` grid the NONE branch fills. Certified against `tests/market_data/remediation/d080_oracle.py` (stdlib + numpy only, import-guarded) with max absolute error 0.0 at 21 targets including all pillars, every interior gap, the anchor and the right tail; an asymmetric pillar x scenario shock grid `(1, 13, 3)` reproduces the oracle with max absolute error 0.0; five source-row permutations give bitwise-identical grids; a NODE forward and a NODE option priced by the real pricers reconcile to 1e-12 and 6e-7 respectively.

### Non-scope (deferred, explicitly)
`PAR_RATE` (schedules, accrual, solver, bootstrap-time interpolation); `DISCOUNT_FACTOR` / `LOG_DISCOUNT_FACTOR` (no shock contract); every other curve family and currency; `USD_CSA_EUR`; IRS; multi-curve; price-curve construction; per-curve interpolation choices; a curve as-of distinct from the run as-of; real-world holiday completeness.

---

## D081 — Production P&L constitution: FULL_REVAL, Taylor orders, ordinary-MTM reporting P&L

**Status:** LOCKED (mathematics and architecture)
**Date:** 2026-09-20
**Supersedes:** D014 (reporting equation), D018 (FULL_REVAL as oracle only)
**Related:** D019, D029, D040, D057, D067, D069, D079, D080, P1-B run-identity contract

---

### 1. Economic identity and state [IDENTITY]

* `p` is a production position identified by exactly one int64 economic UID. Final production P&L has exactly one row per economic UID.
* The underlying valuation representation may contain multiple rows for one economic UID. Such rows must be aggregated to the economic UID before entering the production P&L grid.
* `x` is the canonical market state on the run's authoritative ordered factor axis.
* `x_0` is the authoritative bound base state.
* `x_s` is the canonical shocked-level state for scenario `s`, exactly as produced by the scenario stream. It is a shocked **level state**, never a raw shock representation.
* `V_p(x)` is the position value in its settlement currency under the declared valuation function.
* `q_p(x)` is the settlement-to-reporting conversion factor for the declared settlement/reporting currency pair. When only the reciprocal quotation exists, the reciprocal conversion is used exactly once.
* `W_p(x)` is the reporting-currency value:

$$
W_p(x)=V_p(x)\,q_p(x)
$$

* `V_0 = V_p(x_0)`, `V_s = V_p(x_s)`, and `\Delta V = V_s - V_0`.

---

### 2. Reporting P&L [IDENTITY, then POLICY]

Production reporting-currency P&L is the ordinary mark-to-market change in reporting-currency value:

$$
\Delta W_{p,s}
=
V_s q_s - V_0 q_0
=
\Delta V\,q_s + V_0(q_s-q_0)
$$

The equality is algebraic:

$$
V_sq_s - V_0q_0
=
(V_s-V_0)q_s + V_0q_s - V_0q_0
$$

The production policy is therefore:

> **Reporting P&L is the change in the position's reporting-currency value, including base-value FX translation.**

The following are consequences of that identity, not independent policy choices:

* `V_s = V_0` and `q_s = q_0` implies `\Delta W = 0`.
* `V_s = V_0` implies `\Delta W = V_0(q_s-q_0)`.
* `q_s = q_0 = q` implies `\Delta W = q\,\Delta V`.
* `V_0 = 0` implies `\Delta W = \Delta V\,q_s`.
* Sign follows ordinary multiplication. For example, `V_0 < 0` and `q_s > q_0` implies a negative translation contribution.
* Reciprocal FX quotation produces the same reporting-currency value `W` when applied consistently.
* Reporting FX is not a Taylor factor and is not included in Taylor displacement coordinates.

---

### 3. FULL_REVAL [METHOD]

FULL_REVAL computes settlement-currency valuation movement directly from the declared valuation function:

$$
\Delta V_{\mathrm{FULL}}
=
V_p(x_s)-V_p(x_0)
$$

and reporting P&L is:

$$
\Delta W_{\mathrm{FULL}}
=
V_p(x_s)q_s - V_p(x_0)q_0
=
\Delta V_{\mathrm{FULL}}q_s
+
V_0(q_s-q_0)
$$

FULL_REVAL is exact with respect to the declared valuation function and its numerical implementation. It is not Taylor-truncated and does not use derivatives.

Base and scenario valuation use the same valuation route, static configuration, valuation contract, and `discounting_mode`. The base valuation is computed once from the authoritative base state.

For Taylor production, **`V_0` is run-computed from the authoritative valuation representation and base state. It is not an independent caller-authoritative Taylor input.**

FULL_REVAL is a production methodology. Where a raw representation exists, FULL_REVAL also provides the reference valuation for Taylor consistency checks.

---

### 4. Taylor displacement and coordinates [IDENTITY]

For each factor with a declared derivative coordinate:

**ABSOLUTE**

$$
\Delta x_i=x_{i,s}-x_{i,0}
$$

**RELATIVE**

$$
\Delta x_i=\frac{x_{i,s}}{x_{i,0}}-1,
\qquad x_{i,0}\neq0
$$

**LOG**

$$
\Delta x_i=\ln\left(\frac{x_{i,s}}{x_{i,0}}\right),
\qquad x_{i,0}>0,\;x_{i,s}>0
$$

`Δx` is recovered from the canonical shocked level state and the authoritative base level using the derivative's declared coordinate.

The scenario stream's shock representation, including DIFFERENCE, SIMPLE, and LOG, is a representation of how shocked levels are generated. It is independent of the Taylor derivative coordinate and does not enter Taylor evaluation directly.

Derivatives are evaluated at `x_0` in their declared coordinate.

Non-native coordinate declarations are either transformed exactly under the declared coordinate contract or rejected. Coordinate relabelling is forbidden.

---

### 5. Settlement-currency Taylor orders [METHOD]

Let:

$$
V_i=\left.\frac{\partial V}{\partial x_i}\right|_{x_0}
$$

$$
V_{ii}=
\left.\frac{\partial^2V}{\partial x_i^2}\right|_{x_0}
$$

$$
V_{ij}=
\left.\frac{\partial^2V}{\partial x_i\partial x_j}\right|_{x_0}
$$

Then:

#### FIRST_ORDER

$$
\Delta V_{\mathrm{FO}}
=
\sum_i V_i\Delta x_i
$$

First order in every settlement factor. All second and higher orders are omitted.

#### SECOND_ORDER_TRUNCATED

$$
\Delta V_{\mathrm{SO,T}}
=
\sum_i V_i\Delta x_i
+
\frac12\sum_iV_{ii}\Delta x_i^2
$$

This is a declared truncation. Cross derivatives are omitted by declaration. It is second order only in the retained diagonal directions. It is **not** complete second order and must never be described as such.

#### SECOND_ORDER_FULL

$$
\Delta V_{\mathrm{SO,F}}
=
\sum_iV_i\Delta x_i
+
\frac12\sum_iV_{ii}\Delta x_i^2
+
\sum_{i<j}V_{ij}\Delta x_i\Delta x_j
$$

This is the reduced symmetric form of:

$$
\frac12\sum_{i,j}V_{ij}\Delta x_i\Delta x_j
$$

The diagonal terms retain coefficient `1/2`. Each unordered pair `i<j` appears exactly once with coefficient `1`, because symmetry gives:

$$
\frac12
\left(
V_{ij}\Delta x_i\Delta x_j
+
V_{ji}\Delta x_j\Delta x_i
\right)
=
V_{ij}\Delta x_i\Delta x_j
$$

The second-order expansion is:

$$
V(x_0+\Delta x)
=
V(x_0)
+
\sum_iV_i\Delta x_i
+
\frac12\sum_{i,j}V_{ij}\Delta x_i\Delta x_j
+
O(\|\Delta x\|^3)
$$

Each Taylor order is the stated truncation of this expansion.

**Implementation constitution:** the cross-term representation stores each unordered factor pair exactly once. A duplicated pair or an additional factor `1/2` on the reduced `i<j` sum is a defect.

---

### 6. Run-owned base state and Taylor `V₀` [IDENTITY]

The production run owns the authoritative base valuation context.

The run-computed base-value path is:

```text
authoritative base levels
    → market-data matrix at the run's authoritative as_of
    → valuation state
    → canonical valuation route
    → economic-UID aggregation
    → V₀
```

The resulting `V₀` is the single production base value used consistently for:

1. FULL_REVAL valuation movement;
2. Taylor DERIVED binding validation;
3. D081 reporting P&L translation.

Taylor DERIVED production therefore requires a valuation representation from which the run can compute the authoritative `V₀`.

A synthetic Taylor representation that contains derivatives but no corresponding valuation representation is not a complete production DERIVED input under this constitution.

Caller-supplied Taylor `V₀` is not an independent production source of truth.

---

### 7. Run context and identity [IDENTITY]

The run identity is tied to the semantic context that defines the result.

At minimum, the run identity must bind the authoritative:

* economic UID universe;
* methodology partition;
* ordered factor axis;
* scenario identity;
* `as_of`;
* static configuration identity;
* valuation contract identity;
* applicable discounting mode.

The effective `as_of` is owned by the authoritative valuation context and must agree with the relevant market-data and Taylor bindings.

The static configuration identity is the canonical content identity of the authoritative `RiskStaticConfig`.

The valuation contract identity is the established canonical valuation-contract fingerprint together with any separately load-bearing valuation mode such as `discounting_mode`.

Equivalent semantic inputs must produce deterministic identity. A semantically material change must either change the run identity or fail closed.

Derived numerical results such as `V₀` are not independently hashed merely because they are available as intermediate results. Identity represents the authoritative definition of the run and the inputs/contracts that determine derived values.

---

### 8. Composition constitution [IDENTITY]

Production composition follows:

```text
authoritative economic UID universe
    → explicit methodology partition
    → one PnlGrid
    → one D081 reporting transformation
    → VaR / ES
```

The economic UID universe is distinct from valuation representation row order.

FULL_REVAL values are aggregated by economic UID before entering the grid.

Taylor DERIVED values are produced on the run's authoritative factor axis.

The composition layer does not repair factor identity or coordinate ordering. Taylor representations must already satisfy the exact ordered factor-axis contract before numerical evaluation.

No methodology may silently:

* drop an economic UID;
* duplicate an economic UID;
* fill a missing economic UID with zero;
* reinterpret a factor by position;
* apply reporting FX twice;
* replace `x_s` by a raw shock representation;
* substitute a caller-owned Taylor `V₀` for the run-owned base value.

---

### 9. Acceptance constitution [IDENTITY / METHOD]

A production implementation is compliant with D081 only when:

* `x_s` is the canonical shocked-level state;
* Taylor displacement is derived from `x_s` and `x_0` in the declared derivative coordinate;
* FULL_REVAL uses the canonical valuation route;
* Taylor `V₀` is run-computed from the authoritative valuation representation;
* reporting P&L is exactly the ordinary mark-to-market identity in §2;
* reporting FX translation is applied exactly once;
* FULL and Taylor share the same canonical shocked-level state;
* economic UID aggregation occurs before final P&L composition;
* Taylor factor axes are exact, ordered, and identity-consistent;
* second-order cross terms follow the reduced symmetric formula in §5;
* no implicit coordinate relabelling or positional factor reinterpretation exists.

Any implementation that produces a plausible number while violating one of these identity or method contracts is incorrect.

---

## D082 — F-1: the run declares the factor axis; the Taylor book adapter aligns to it

**Status:** LOCKED
**Date:** 2026-09-22
**Related:** D081 §5, §10, §11 (Taylor orders, base binding, methodology composition), P1-A (F(p), completeness, known-zero vs missing), D064 (mapper identity).

### Context

The production run's required-factor axis is `UniverseMapping.required_factor_ids`, in layout order. The run asserts it, row for row, against the FULL_REVAL universe map. The certified Taylor calculator requires its compiled axis to equal the market axis row for row (`SensitivityPnlCalculator._require_axis`). Before this decision, `build_sensitivity_book_spec` built its own axis from the sorted-unique factors that the sensitivities reference. A TAYLOR/DERIVED book whose positions depend on a strict subset of the run axis (the normal mixed-run case) was therefore refused by the run. The only work-around re-keyed `factor_axis` and `factor_meta` by hand in the caller. The identity-keyed mapping already existed: `FactorAxis.from_keys` keeps key order (DIRECT or SLOT encoding) and `compile_book` relabels every factor key through `FactorAxis.relabel` (`replace_strict`).

### Decision

1. **Declaration is run-owned.** The ordered factor axis is declared by the production run and by nothing else.
2. **Alignment is adapter-owned.** `build_sensitivity_book_spec(sensitivities=..., required_factor_ids=..., ...)`. `required_factor_ids` is REQUIRED: 1-D `int32`, unique, non-empty, used exactly in the given order. The adapter never sorts, trims or pads it, and never builds an implicit axis. A caller that prices a book on its own factors declares them explicitly (`required_factor_ids=sensitivities.active_factor_rf_ids`).
3. **Alignment is by factor identity.** The spec carries the declared axis and one metadata row per axis factor, keyed by factor id. `compile_book` maps every sensitivity to its axis row through `FactorAxis.relabel`. No positional mapping and no new permutation helper exist.
4. **Fail fast, before any coefficient transform and before numerics.** The adapter refuses: a non-`int32`, non-1-D or empty axis; a duplicated axis id; any factor the sensitivities reference (diagonal or cross) that is absent from the axis (it is never dropped). The compile step still refuses: metadata missing for an axis factor, duplicated `(position, factor)` rows, and a declared F(p) dependency without its derivative (P1-A).
5. **Inactive axis factors are not a zero-fill.** An axis factor that no position references gets no sensitivity entry and no coefficient, and the engine never gathers its row. The Taylor declaration's F(p) states that no position depends on it. A declared dependency without its derivative still fails closed.
6. **The numerical boundary stays strict.** The calculator keeps exact ordered axis equality: market axis == book axis, row for row.
7. **Composition never aligns.** `solutions.risk_run` does not reorder, gather, sort or re-key Taylor data. It refuses, before numerics, a book not built on the run axis.

### Implications

- Migrated callers: `scripts/demos/e2e/_fixtures.py` (the production DERIVED book now uses the run axis; the standalone sensitivity report declares its own axis), `tests/solutions/risk_run/_mixedfix.py` (manual re-keying removed), `tests/solutions/sensitivity_pnl/{test_taylor_contract,test_spot_book_adapter,test_scenario_pnl}.py` (own axis, declared explicitly).
- Evidence: `tests/solutions/sensitivity_pnl/test_book_adapter_factor_axis.py` (hostile unsorted ids, positional-order trap against an independent dict oracle, inactive factor, malformed/missing/duplicate axis, metadata mismatch, wrong-axis market, declared-dependency-without-derivative); `tests/solutions/risk_run/test_methodology_composition.py::{test_f1_adapter_builds_the_subset_book_on_the_run_axis, test_derived_book_on_a_factor_subset_is_refused_before_numerics, test_composition_performs_no_factor_realignment}`.
- Compile-key space (intentional, unchanged): the CORE `CompiledSensitivityBook.active_factor_ids` holds the compiler's keys, i.e. factor ids under DIRECT encoding and axis row positions under SLOT encoding (`pipeline.py` "Slot relabelling"; `factor_key_encoding` in provenance). The encoding-independent factor identity is `calculator.axis.keys[book.active_level_uix]` (the lookup `taylor_validity.py` uses). A consumer that reads `active_factor_ids` as factor ids is valid only for a DIRECT-encoded book.
- Metadata scoping (intentional, safe): `scope_factor_meta` ignores metadata rows for factors that are NOT on the axis, under both encodings. It never removes the metadata of a factor the book references: an off-axis referenced factor fails (adapter, and at compile for a hand-built spec), and a referenced axis factor without metadata fails at compile.
- Hardening evidence: hostile cross terms aligned by factor-pair identity (`test_cross_terms_are_aligned_by_factor_identity`); an unsorted run axis [9007, 12, 101, 4002] produced by the real `build_universe_mapping` end to end (`test_f1_unsorted_authoritative_run_axis_end_to_end`).

### Amendment (2026-09-24, D085): the expanded canonical run axis

8. **One canonical ordered run axis.** A RiskRun has exactly one run axis: the valuation universe prefix (`UniverseMapping.required_factor_ids`, row for row, in its existing layout order) followed by the reporting-only FX suffix that the reporting contract requires (D085). The prefix ordering is unchanged. The suffix is ascending by rf_id. No factor id appears twice: a reporting FX factor already on the prefix is used there.
9. **Suffix factors are scenario state, not valuation factors.** They are shocked by the same scenario stream as the prefix and read only by the D081 reporting conversion. Valuation reads the prefix only.
10. **The Taylor axis is the valuation prefix.** The certified calculator, the book (decision 2) and the Taylor base binding are declared on the prefix, so a reporting FX factor cannot become a Taylor factor. Decision 6 applies to the prefix.
11. **Every axis-aligned run input is bound to the whole run axis.** `required_factor_ids` and `required_shift_type_id` must equal the run-derived axis row for row, and `source_factor_ids`, `shock_scale` and `base_levels` must have one entry per axis row; otherwise the run refuses before identity and numerics. The five inputs must already be 1-D, C-contiguous arrays of exact dtype (int32, int8, int32, float32, float64); nothing is cast, because the identity digests the dtype. Callers obtain the axis from `solutions.risk_run.api.derive_run_factor_axis`.
12. **Identity.** The suffix enters identity through the shock and base-level digests and the reporting FX digest (factor names and axis rows). An empty suffix leaves every existing identity unchanged.

**Clause 11 status: SUPERSEDED IN PART by D088 (2026-09-25).** The requirement that `source_factor_ids` (int32) and `shock_scale` (float32) accompany the axis one entry per row is withdrawn as the canonical proxy representation. Under D088 the compiled canonical proxy model contains exactly one row per V1 target-axis node; an authored configuration may omit a node's model because identity-by-default supplies `(d, τ, β) = (A, κ_A, 1)` at compile time, so users never author identity rows; loadings are float64 and canonicalised per D088. Unaffected: the axis derivation (clauses 8–10), `required_factor_ids` and `required_shift_type_id` bound to the axis row for row, `base_levels` one per row with exact dtype, `derive_run_factor_axis`, the reporting-only suffix (D085) and clause 12's principle that the suffix enters identity.

---

## D083 — Persisted-result integrity: `RiskRun.result_digest`, separate from run identity

**Status:** LOCKED
**Date:** 2026-09-23 (authority, reproduction and provenance clauses made explicit at PR-01 closure)
**Related:** D040, D041, D042 (clarified, not rewritten), D044, D059, D081 §7. Evidence: `PR-01_RiskRun_Result_Integrity_Audit.md`.

### Context

`RiskRun.from_json` checked identity against manifest and `reporting_ccy` against manifest (D044). Nothing bound the stored `var`, `es`, `position_uids` or `reproducible` flag. A payload with a one-ULP VaR change, permuted UIDs, a flipped flag or NaN/inf loaded under the authentic identity. D042's "rejects a tampered payload" held for identity/manifest tampering only.

### Decision

1. **`RunManifest.identity()` stays the identity of the CALCULATION DEFINITION.** Unchanged: manifest, schema version `"4"`, `EngineIdentity`, hashing, identity values. No output enters it (D081 §7).
2. **`RiskRun.result_digest` is the integrity of the persisted RESULT ARTIFACT.** Derived in `__post_init__` (`field(init=False)`), never supplied by a caller. `digest_named_arrays` over `identity` (ASCII bytes), `position_uids` (int64), `var`, `es` (float64), `reproducible` (bool): dtype, shape and exact bytes, so one ULP moves it. Every member is load-bearing (mutation-proven).
3. **`to_json` writes `result_digest`; `from_json` recomputes and compares.** Absent or mismatched → `ResultIntegrityError(ValueError)`, message `RESULT INTEGRITY: …`.
4. **Integrity, NOT authentication.** Unkeyed SHA-256. It detects corruption, inconsistent edits, cross-run transplants and a relabelled manifest with a recomputed identity. It does not detect a forger who rebuilds manifest, identity, numbers and digest consistently; a test pins that non-claim. No keys, signing or provenance framework (D040/D042 rejected list).
5. **Persisted `var` and `es` are authoritative.** The `var` and `es` of a persisted `RiskRun` that `from_json` loads successfully, with a verified `result_digest`, are the authoritative numerical results that the artifact represents. Ordinary deserialization needs no re-execution.
6. **`reproduce=` is a separate mechanism.** It is unchanged: it verifies environment and inputs before numerics and returns fresh numbers. It serves reproducibility and verification where its existing contract applies (D041/D042); it is not a condition of integrity-valid deserialization. Detection of a self-consistent forgery (§4) needs reproduction and a bitwise comparison of the fresh numbers with the stored ones.
7. **`pnl_provenance` is outside `result_digest`.** It is operational provenance metadata (the D042 record), not a result and not identity. It is persisted and validated by its own schema, but not integrity-bound.
8. **A `RiskRun` carries finite risk numbers only.** NaN / ±inf in `var` or `es` is refused on construction and on load.
9. **The digest is well defined by construction.** `position_uids` must be 1-D int64, `var`/`es` 1-D float64 (never cast), `reproducible` a `bool`. The stored arrays are owned, C-ordered, read-only copies, so the digest keeps describing them.
10. **Malformed payloads fail deliberately.** A non-object payload, a missing key or a non-bool `reproducible` raises `ValueError("malformed persisted RiskRun: …")`, never an incidental `KeyError`.

### Semantics preserved

- Exploratory `seed=None`: same identity, possibly different numbers (D040/D042). Each artifact now carries the digest of its own numbers.
- `batch_size` stays operational (D041): same identity and same `result_digest`.
- The constructor signature of `RiskRun` is unchanged; `api.py` and `run_identity/*` are unchanged.

### Compatibility

A persisted `RiskRun` JSON written before D083 has no `result_digest` and is REJECTED on load. This is deliberate: an unverifiable result is not admitted. RiskQuant is a development-stage codebase: historical development artifacts create no compatibility requirement, and no migration or compatibility path exists. Regenerate any stored artifact through `run_scenario_var_es`.

### Evidence

`tests/solutions/risk_run/test_result_integrity.py` (test-owned book `_resultfix.py`, real production path): round trip bitwise; every VaR/ES entry ±1 ULP rejected; UID permutation/replacement/duplicate, flag flip, digest replacement/absence, cross-run transplant and D2-style relabelling rejected; self-consistent forgery accepted (non-claim); NaN/±inf rejected; identity and manifest unchanged; seed=None semantics; fresh-process, batch-size and configured-thread invariance of `result_digest`.

---

## D084 — SPOT Taylor capability boundary: inverted-FX legs are FULL_REVAL only

**Status:** LOCKED
**Date:** 2026-09-23
**Related:** D009, D010, D012, D060, D078.4, D081 §3–§5 and §11, D082. D009 and D081 are not modified. Evidence: `PR-02_Inverted_FX_Taylor_Boundary_Audit.md`.

### Context

A SPOT leg whose underlying currency differs from its settlement currency, where only the reciprocal pair is configured, resolves with `invert = 1` (D060, D078.4). One orientation per pair is configured, so one direction of every pair is inverted by construction. FULL_REVAL values such a leg exactly: `V = q·S/X` on the quoted factor X. The SPOT sensitizer refuses it with `NotImplementedError`. The run admitted a caller-built, certified TAYLOR/DERIVED book for such a leg, because P1-A binds a book's base, order and completeness, not its Greek values. A fabricated bilinear book moved portfolio VaR from −1986.092 to −1860.114 without error.

### Decision

1. **Capability contract.**
   - inverted-FX SPOT + FULL_REVAL: admitted.
   - inverted-FX SPOT + TAYLOR (SUPPLIED or DERIVED): not admitted.
   - direct-FX or no-FX SPOT + TAYLOR/DERIVED: admitted.
2. **The producer refusal is intentional.** `sensitize_spot_block` raises `NotImplementedError` for `invert = 1` before its kernel (D010 capability-gap convention). It is not a defect and gets no stub.
3. **Run admission.** `run_scenario_var_es` refuses any TAYLOR position (U_S ∪ U_D) that is an inverted-FX SPOT leg with `CompositionContractError`. The check (`_require_taylor_capable_spot`) reads the run's own compiled portfolio and `UixMapper` tables. It runs before identity construction and before any valuation or scenario work. It is not a kernel check.
4. **Taylor sensitisation is scoped to U_D.** A TAYLOR/DERIVED book is sensitised on the U_D positions only. Sensitivities are rf_id-keyed (D012) and the book aligns to the run axis by identity (D082), so a U_D sub-portfolio compiled against the same static configuration and base levels is a valid source.
5. **Coexistence.** An inverted-FX SPOT position declared FULL_REVAL does not prevent direct TAYLOR/DERIVED positions in U_D from running.

### Future authoring material (not an implementation requirement)

For `V = q·S/X` on the quoted factor X (D078.4), differentiated on X in ABSOLUTE displacement (D081 §4): `V_S = q/X`, `V_X = −q·S/X²`, `V_SS = 0`, `V_XX = 2·q·S/X³`, `V_SX = −q/X²`, `V_XXX = −6·q·S/X⁴ ≠ 0`. Factor identity, displacement convention, base level and F(p) = {S, X} are unchanged relative to the direct leg; signs, scaling, the own second order and the cross term change. Expressing the leg in `1/X` would be coordinate relabelling (D081 §4) off the run axis (D082). No Taylor order is exact: SECOND_ORDER_FULL leaves an O(Δx³) remainder, so admission would also need a remainder contract. Any future admission requires a new decision, the analytic partials, a finite-difference oracle on the FULL_REVAL route and a remainder test per order.

### Evidence

`tests/solutions/risk_run/test_inverted_fx_taylor_admission.py`: T1 sensitizer refusal names `invert=1`; T2 a certified fabricated book for the inverted leg moves VaR when the guard is disabled (D059) and is refused before manifest construction, base valuation, scenario generation and grid allocation; T3 inverted FULL_REVAL beside direct and no-FX TAYLOR/DERIVED (U_D-scoped) gives VaR/ES bitwise equal to all-FULL_REVAL; T4 a direct-FX DERIVED position passes admission. Mutation: removing the guard fails T2; widening it to every FX leg fails T3 and T4.

---

## D085 — Reporting currency: run-derived source currency and reporting-only FX on the run axis

**Status:** LOCKED
**Date:** 2026-09-24
**Related:** D013, D040, D060, D078.4, D081 (mathematical reporting semantics unchanged), D082 (execution-axis semantics amended to permit a derived reporting-only suffix), D083, D084. Evidence: PR-04 F4 and F4B audits, PR-04 final hardening.

### Context

`ReportingSpec.settlement_ccys` restated each position's settlement currency, which the trade record already fixes (ingest `settlement_ccy`, folded into `fx_u2s`). A false claim was accepted and produced finite, wrong reporting P&L, VaR and ES. Separately, a reporting FX factor that no valuation needs could not enter the run, because the run axis was the valuation universe: a USD position could not be reported in EUR unless an unrelated position happened to need USDEUR. The D081 conversion itself was correct.

### Decision

1. **The caller owns only the reporting currency.** `ReportingSpec` is `reporting_ccy`. `settlement_ccys`, `fx_factor_names`, `fx_factor_rows` and `position_uids` are removed.
2. **The run derives each position's settlement currency** from compact ingest state: `rf_id_to_ccy_id[underlying_ids]` with no FX leg; with a leg, the quote side of the configured pair (`fx_u2s_inv = 0`) or its base side (`fx_u2s_inv = 1`). Admitted instruments: SPOT, VANILLA_FORWARD_NODE, VANILLA_EUROPEAN_OPTION_NODE. Any other instrument, an underlying without a configured currency, an FX leg that is not a configured pair, or one UID with rows in different settlement currencies is refused (`CompositionContractError`).
3. **The run derives the reporting FX route** from `fx_spot_matrix` (D060): settlement = reporting is an identity conversion with no factor; else the direct pair (settlement, reporting) gives q = x; else the reciprocal pair gives q = 1 / x, applied once; else the run is refused.
4. **One canonical run axis** (D082 amendment): the valuation prefix, then the reporting FX factors the prefix lacks, ascending by rf_id. One scenario stream generates both, so historical, bootstrap and parametric dependence is joint. There is no second scenario universe.
5. **Reporting FX is scenario state, not a Taylor factor.** Valuation and the Taylor valuation prefix read rows [0, K_val) only; the reporting conversion reads the exact FX rows it needs from the whole run axis [0, K_run). D081 mathematical reporting semantics are unchanged: ΔW = ΔV q_s + V_0 (q_s − q_0), with ΔV from FULL_REVAL or the settlement-currency Taylor expansion. D082 execution-axis semantics are amended to permit a derived reporting-only suffix.
6. **Axis ownership.** The run derives the axis; callers bind their scenario data to it through `derive_run_factor_axis(valuation_context=..., reporting_ccy=...)`; the run re-derives it and refuses any other binding before identity and numerics. `RunFactorAxis` is a descriptor: frozen fields, owned read-only C-contiguous int32 / int8 arrays. The run type-checks the five axis-aligned inputs exactly (D082 amendment, clause 11) and executes on its own writable copies, so read-only descriptor arrays may be passed as-is and a caller buffer cannot alias run state.
7. **PR-04 R2/R3.** The route is derived during composition, so an unroutable reporting currency fails before identity and numerics (R2). R3, the check of caller-supplied FX rows against their names, lost its premise and is removed.
8. **`source_factor_ids` is shock provenance, bound row by row (D013).** `source_factor_ids[k]` names the scenario-data column whose simulated move drives run-axis row k. It equals `required_factor_ids[k]` when that factor has its own scenario data, and is otherwise the proxy chosen by `resolve_required_factor_ids_to_market_data_factor_ids` from the caller-side `ProxyTopology`; it is therefore NOT required to equal the axis, and it is result-defining and in the identity (D040). The run does not hold the topology and does not re-derive a proxy; it enforces every part of the resolver contract its own inputs determine: every source is a column of the scenario data, and a factor that has its own column is its own source (resolver tier 1). Both are checked during composition, before identity and numerics.

### Consequences

A reporting-only suffix adds K_rep rows (K_rep ≤ number of distinct settlement currencies) to the shocked-level buffer and to the generators' per-factor state; for the parametric model this enlarges the whole-run normal buffer (PR-06). A suffix row (or any row) whose shock source has no column in the scenario data is refused during composition, before identity and numerics.

### Evidence

`tests/solutions/risk_run/test_reporting_axis.py`: single USD position reported in EUR (bitwise against an independent D081 oracle; analytic −2 up to the float32 shock and grid contracts); USD + GBP to EUR with a reciprocal quote and a shared factor; identity conversion; no duplicate of a prefix factor; missing reporting history; no caller-owned routing; parametric correlation shared across prefix and suffix; one bootstrap time index; DERIVED on the prefix equals FULL bitwise and a book on the expanded axis is refused; composition negatives. Mutation: removing the axis gate, dropping the suffix, losing the reciprocal flag, ignoring `fx_u2s_inv`, building Taylor on the full axis or removing the shape check each fails tests. K_rep = 0 regression: six existing compositions keep identity, manifest identity, VaR, ES and `result_digest` bitwise.

`tests/solutions/risk_run/test_axis_input_contract.py` (final hardening): wrong dtype with equal values, 2-D, strided and non-array inputs are refused before any numerical stage (spies on manifest, generator, valuation, shock application, grid, Taylor, conversion and VaR/ES); the run executes on owned writable copies and accepts the read-only descriptor as-is; `RunFactorAxis` ownership, read-only arrays, frozen fields and exact dtypes; a proxied row reads exactly its proxy's float32 shock; two valid proxy routings differ in result and identity; substituted, permuted and tier-1-violating sources and a source without a data column are refused before numerics; poisoning the reporting-only suffix of the one shocked-level buffer leaves V_0 and settlement dV bitwise unchanged (FULL_REVAL and DERIVED) while reported P&L equals the hand D081 value on the poison level; the Taylor axis is the valuation prefix and an expanded-axis book is refused; reporting currency, reporting FX factor, route orientation and suffix base level each move the identity, and a rebuilt identical run matches identity, `result_digest`, VaR and ES bitwise; historical and bootstrap prefix and suffix read one date per scenario (asymmetric markers, exact on float32 shocks); parametric full-axis shocks equal an independent fixed-seed construction exactly; one shared row per settlement→reporting pair. Mutation: removing each new check (dtype, dimensionality, contiguity, owned copy, source column, tier 1, read-only axis, axis ownership) fails tests.

**Clause 8 status: SUPERSEDED by D088 (2026-09-25).** The rule that `source_factor_ids[k]` equals `required_factor_ids[k]` whenever that factor has its own scenario data — resolver tier 1, "a factor that has its own column is its own source" — is no longer authoritative for model selection. Under D088 the model determines the source columns and data availability validates those references: every V1 target-axis node has exactly one compiled model, identity-by-default when none is authored; data availability never chooses between models; there is no fallback tier. Unaffected: every referenced `(d, τ, h)` column must exist in the scenario data; the reporting-only FX suffix (clauses 1–7); row-by-row binding to the run axis; every other data-existence check that is not model-selection logic.

---

## D086 — Taylor order is a certified property of the compiled representation

**Status:** LOCKED
**Date:** 2026-09-24
**Related:** D040, D042, D081 §5 and §7, D082, D084, P1-A. Evidence: PR-05 audit.

### Context

`TaylorOrder` (`FIRST_ORDER`, `SECOND_ORDER_TRUNCATED`, `SECOND_ORDER_FULL`) is declared in `TaylorDeclaration.order` and proven against the spec frames at compile (`validate_taylor_completeness`). The kernel never reads the order: it evaluates the stored terms, `Σ d1·Δx + ½ Σ d2·Δx² + Σ_{i<j} c·Δx_i·Δx_j`. The run identity binds the book through `hash_compiled_book`, which digests those stored terms and not the declaration. For the admitted SPOT sensitizer `V_ii = 0`, so FIRST_ORDER and SECOND_ORDER_TRUNCATED books store identical terms and share one identity. The PR-05 audit also showed that the certificate was coherent with its book only at compile: `dataclasses.replace` or direct construction could attach any order, including a non-member value, to any compiled book, and the run priced it without error.

### Decision

1. **Domain.** The order is exactly one `TaylorOrder` member. A value that merely compares equal (the string `"FIRST_ORDER"`, `1`, `True`) is not a declaration and is refused (`TaylorContractError`).
2. **Certified pair.** A declaration and the book it is attached to form one certified pair wherever a `SensitivityPnlCalculator` is built (compile, direct construction, `dataclasses.replace`): FIRST_ORDER admits no non-zero diagonal second order and no cross term; SECOND_ORDER_TRUNCATED admits no cross term; SECOND_ORDER_FULL requires every unordered pair of each position's factor rows exactly once. Refused at construction, before any run (`verify_declared_order_against_book`).
3. **Execution.** The order selects the method by determining which terms the representation stores; the kernel evaluates exactly those terms. There is no separate execution switch and no inference of the order from data.
4. **Identity.** The order is not hashed separately. The identity names the stored terms (`sensitivities_digest`), which is the method actually evaluated. Two orders that store different terms get different identities; two orders coherent with the same stored terms denote the same function (each is the stated truncation of D081 §5 and the omitted terms are zero) and share one identity. For SPOT, FIRST_ORDER ≡ SECOND_ORDER_TRUNCATED always, and SECOND_ORDER_FULL coincides with both for positions with one factor. The methodology partition (FULL_REVAL / DERIVED) remains separately identity-defining (D081 §7).
5. **Persistence.** A `RiskRun` carries no separate order claim, so no stored label can disagree with the digested representation; `reproduce=` accepts exactly the books whose stored terms match the manifest.

### Evidence

`tests/solutions/risk_run/test_taylor_order_semantics.py`: order domain (six non-member values refused at compile and at construction); four incoherent re-labellings and FIRST_ORDER over curvature refused at construction; coherent re-labelling admitted with an identical digest; identity follows stored terms (FIRST vs FULL differ, FIRST vs TRUNCATED equal, including `result_digest`); changed Greeks under one order and a methodology change with equal numbers both change identity; a recompiled identical book keeps it; an independent D081 §5 oracle per order with non-zero diagonal and cross terms (one float32 ulp bound); FULL_REVAL rows independent of the Taylor order; reporting-only suffix poison never reaches Taylor under any order; historical, SBB, filtered bootstrap and parametric worlds with batch invariance per order; round trip and reproduction per order, a different term set refused under `reproduce=`. Mutation: removing the construction hook, the FIRST_ORDER curvature or cross check, the TRUNCATED cross check, the FULL pair count or the domain check each fails tests.

---

## D087 — The parametric scenario world is one whole-run realization; a batch only partitions it

**Status:** LOCKED
**Date:** 2026-09-24
**Related:** D026 (F7), D029, D037, D039, D040, D041, D081 §8, D082, D086. Evidence: PR-06 audit.

### Context

PR-06 asked whether the parametric generator's whole-run `(total_scenarios, K_total)` float64 normal buffer is the right contract and whether its memory is a production ceiling. The audit confirmed the D029 mechanism, wrote down its exact draw order, and found one missing check: the production run concatenated whatever batches the generator yielded without checking that they tile the scenario range. A repeated, skipped, reordered or truncated batch completed silently, and a skipped batch changed VaR under an unchanged identity.

### Decision

1. **Stochastic contract (unchanged, now explicit).** Each `iter_batches` call builds `numpy.random.default_rng(seed)` (the D037 stream 0) and fills ONE C-ordered `(S, K_total)` float64 buffer with `standard_normal`; cell `(t, k)` is draw `t·K_total + k`. A non-normal law then draws ONE size-`S` auxiliary vector (`chisquare`, `exponential`, `gamma` or `random`) after all normals. Row `t` of the shocks is `vols ∘ (row_scale(aux_t) · (Z_t Lᵀ)[:, cols])`, with the crisis multipliers applied on the crisis rows of the regime-switching law, rounded ONCE to float32 after all scaling. Generation is serial NumPy; only the deterministic transformation touches BLAS (D041).
2. **The world is defined by the scenario configuration and the scenario data.** `K_total` is the factor set of `ParametricRiskData`, not the run axis; the run axis only selects columns through `source_factor_ids`. Adding a row to the axis (a reporting-only suffix, D085) or permuting it leaves every other column bitwise unchanged. The Taylor order never enters generation (D086).
3. **A batch is only an execution partition.** `batch_size` chooses row slices of the whole-run buffer and nothing else (D029). The production run now requires the yielded batches to tile `[0, S)` exactly once, in order, each declaring the same total: `scenario_offset` must equal the scenarios already written and the stream must end at `generator.total_scenarios`. A violation raises `RuntimeError` before any P&L is allocated.
4. **Identity and reproduction (unchanged).** The seed, scenario count and law are identity inputs through `scenario_config`, the correlation and volatilities through the data digest, and the NumPy / Numba versions through `EngineIdentity`. Under D090–D092 the chart ids, the effective period and, when present, the per-factor crisis multipliers also enter through the data digest, while `prob_crisis`, the scalar crisis multiplier and the `"per_factor"` declaration enter through `scenario_config` (amended by D092). `batch_size` is operational (D041). The realised buffer is not hashed: it is a deterministic function of those inputs in a matching environment.
5. **Memory: no streaming change.** Measured on this host (1 CPU) at the certification dimensions (`K_total = 250`, `K_run = 140`, `M = 200`, batch 4 096): at `S = 100 000` the buffer is 190.7 MiB and lives only during generation. Parametric peak RSS was 1 310.8 MiB against 1 307.4 MiB for SBB on the same world; at `S = 20 000`, 533.4 MiB against 530.6 MiB. The peak is set by the downstream whole-run arrays that D081 §8 already requires (one shocked-level buffer, one PnlGrid, the reporting rows), about 10 KB per scenario here. Streaming the normals would not lower that peak, and for a non-normal law it could not keep the D029 stream without generating the normals twice. Reassess only if `8·K_total` bytes per scenario approaches the downstream per-scenario footprint (at `M = 200`, `K_total` of the order of 1 300 factors).

### Evidence

`tests/simlab/scenarios/test_parametric_whole_run_stream.py`: for all five laws, the recorded whole-run buffer equals an independent `default_rng(seed).standard_normal(S·K).reshape(S, K)` and the auxiliary draw bitwise, with one normal fill and one auxiliary call in that order; buffer and concatenated shocks are bitwise identical for batch sizes 1, 2, 3, 7, 13, S−1, S and S+1; the shocks equal an explicit-sum construction from the model's formulas within one float32 ulp, and exactly under an identity Cholesky factor (single rounding point); added or permuted requested columns leave the others bitwise unchanged; the seed alone selects the buffer. `tests/solutions/risk_run/test_parametric_run_world.py`: repeated, skipped, reordered and truncated batch streams are refused before P&L allocation; every partition yields the same shocked levels, identity, `result_digest`, VaR and ES; a reporting-only suffix leaves the valuation row bitwise; FIRST and FULL Taylor orders see one world; a new seed changes world and identity; `reproduce=` with another batch size is bitwise. Mutation: a fresh draw per batch, a reseed per batch, a batch-size-derived seed, reversed factor columns, reversed scenario rows, a slice shifted by one, an F-ordered buffer, an earlier float32 rounding point, and removing either run check each fails tests.

**Clause 2 status: SUPERSEDED IN PART by D088 (2026-09-25).** `K_total` is the column set of the parametric world keyed `(d, τ, h)` (D034 as superseded), not the factor-id set alone, and the run axis selects columns through the compiled proxy model's term references rather than `source_factor_ids`. Unaffected: clause 1 (whole-run draw order, single float32 rounding point), clause 3 (batch as partition; tiling check), clause 4 (seed, count and law through `scenario_config`; `batch_size` operational per D041), clause 5 (memory) and D029.

**Status: AMENDED IN PART by D092 (2026-09-29).** Clause 4's identity composition is extended as marked; the stochastic contract (clauses 1–3) and RNG semantics are unchanged.

## D088 — RiskQuant Proxy V1: a proxy is a model for the target's shock on world columns

**Status:** LOCKED — contract and design FROZEN 2026-09-25
**Supersedes:** D013 (definition of proxying). **Amends:** D026 F7, D034, D036, D040 `ShockIdentity`, D082 clause 11, D085 clause 8, D087 clause 2, each by the status note recorded in that decision. Unrelated clauses of those decisions remain locked.
**Implementation status:** D088 STAGE D COMPLETE (2026-09-29). `run_scenario_var_es` executes Proxy V1: it compiles the authored `proxy_terms` over the full run axis (§13.8(a)), lowers the model onto the axis, binds every `(d, τ)` to the world's `(d, τ, h)` columns before identity and generation, and applies `X_A = Σ β Y` then `T⁻¹_κA` (Slices 1–3). `ShockIdentity` is `proxy_model_digest` (manifest schema v5, SHA-256), and a v5 manifest without `RunContextIdentity` is refused (§13.3). Repository callers (scripts, release certifiers, tests) pass `proxy_terms`; the D013 path (`resolve_required_factor_ids_to_market_data_factor_ids`, `apply_shocks_to_base_levels`, `proxy_config`) is unreachable from production and remains in the repository for Stage E removal, together with the generators' neutral `shock_scale` protocol argument.

**Implementation status, Stage E (2026-09-29):** D088 STAGE E COMPLETE. Proxy V1 is the exclusive production proxy architecture. D013 production semantics, implementation and ingest/configuration are removed: `proxy_resolver`, `config/proxy_config` and its canonical ingest fixture, `simlab/shocks/apply.py`, `shock_kernel.py`, `simlab/builder/shock_universe.py` and the transitional world-column stream no longer exist. Generators take a unique `(d, τ)` world request (`iter_batches(driver_rf_ids, chart_ids, batch_size)`, validated by `build_world_column_plan`) and return the columns in request order; `shock_scale`, `source_factor_ids` and `required_to_shock_uix` no longer exist. The Proxy V1 authoring/file format and its ingest edge remain OPEN (§12.3; specification §13.2); no canonical Proxy V1 ingest fixture exists yet.

**Stage E certification correction (2026-09-30).** The Stage E certification (2026-09-29) did not detect the D013 frame schema `RiskProxyConfigData` (`proxying_level`, `required_name`, `proxy_name`, `vol_scaling_factor`) in `src/riskquant/ingest_fixtures/schemas.py` and its reference in `src/riskquant/ingest_fixtures/utils.py`: the structural test matched exact D013 identifiers only. A later repository-wide audit found it. It was removed and replaced by the D093 schema `RiskProxyModelTermData`. `tests/test_proxy_v1_exclusivity.py` now also refuses `RiskProxyConfigData`, `proxying_level`, `vol_scaling_factor`, `_kernel_apply_shocks` and string references to removed modules (dynamic imports). Corrected Stage E state: no D013 code, schema or frame vocabulary remains in `src/` or `scripts/`.

### 1. Scope

The V1 proxy-model universe is the set of run-axis factor-nodes whose configured `shift_type` is a V1 reconstruction chart, `κ_A ∈ {LOG_RETURN, SIMPLE_RETURN, DIFFERENCE}`. A node whose configured `shift_type` is `ABSOLUTE_BASE_RETURN` or `NO_SHIFT` is outside the V1 target universe and keeps its D025 behaviour: ABSOLUTE_BASE has no V1 target reconstruction contract and no V1 column contract; NO_SHIFT is a configured shift type and an execution behaviour (a zero column, skipped by `validate_shift_domain`, reconstructed to the base) — it is not a chart and not a node status. The V1 driver-chart catalogue is `τ ∈ {LOG_RETURN, SIMPLE_RETURN, DIFFERENCE}`.

### 2. Mathematical object

    X_A(ω) = Σ_{j ∈ J_A} β_Aj · Y_(d_j, τ_j, h)(ω)
    A_s    = T⁻¹_κA(A_0, X_A(ω_s))

`A` and `d_j` are resolved factor-node identities; `κ_A` is the target's reconstruction chart; `τ_j` is the explicit driver chart of the term; `β_Aj` is a signed finite float64 modelling coefficient; `A_0` is the run-bound base level; `h` is the world-declared elapsed horizon in the world's declared step unit. The identity model `J_A = {(A, κ_A)}, β = 1` is the compiled default for every V1 node with no authored model. Charts: LOG forward `ln(v1/v0)` (inputs > 0), inverse `A_0·e^x`; SIMPLE forward `v1/v0 − 1` (v0 ≠ 0), inverse `A_0(1+x)`; DIFFERENCE forward `v1 − v0`, inverse `A_0 + x`. Reconstruction output must be finite; whether the resulting level is admissible for the factor (finite; > 0 for FX_RATE and VOL) is decided by the existing factor-state guard (D070, D075), not by the chart.

### 3. Ownership

| Fact | Owner |
|---|---|
| Factor-node identity `A`, `d` | factor definition; resolved run axis; canonical identity is the `rf_id` under the run as-of (D067) |
| Reconstruction chart `κ_A` | target factor definition (configured `shift_type`); never the proxy model |
| Base level `A_0` | run-bound base state (`base_levels`; D081 §1) |
| Driver chart `τ_j`, loading `β_Aj`, term set `J_A` | proxy model |
| Horizon `h`, step unit, scenario law | scenario world (configuration and data; D087) |
| Physical quantity unit | factor definition where declared; absent today for PRICE/SPREAD/INDEX |
| FX quotation convention | pair definition and orientation matrix (D078.4, D060) |
| Provenance (calibration method, window, rationale) | governance; never numerical |

The configured `shift_type` of a driver node does not imply `τ_j`. A node supplies `Y_(d, τ, h)` only when the world supplies that column and the applicable rules, including D025, permit it.

### 4. World columns

A world is a finite probability space with one declared horizon and jointly defined real random variables keyed `(d, τ, h)`. `Y_(d, τ, h)` is fundamentally a random variable in the declared shock space. A column may be supplied directly in shock space or constructed from levels; the distinction is per column. A level-backed column equals `T_τ(v_start, v_end)` and its level inputs must satisfy the forward chart's input domain; for a level-backed column, any valid sequential decomposition whose intermediate states satisfy that domain composes to the endpoint transformation (LOG and DIFFERENCE by sum, SIMPLE by product-minus-one). A direct column implies no level path and no level-domain obligation. The proxy contract does not prescribe how any engine constructs a column; each engine's own contract shows that its columns satisfy this definition. Loadings apply after the h-horizon column exists; the model carries no horizon and no hidden horizon or volatility scaling; horizon-invariance of `β` is a V1 modelling assumption, not a theorem. World invariants: common scenario index; common horizon; column values a function of the world identity alone; no silent missing values; no clipping; no scenario dropping, reordering or resampling by the model; a model changes no column and no other row; the probability weights belong to the risk-measure layer.

### 5. Loadings and units

`β_Aj = ∂X_A/∂Y_j`; `[β_Aj] = u_κA / u_τj` with `u = 1` for LOG and SIMPLE and `u = ` the node's canonical level unit for DIFFERENCE (D069). Physical-unit metadata is validation and interpretation metadata where available; missing physical-unit metadata does not invalidate a mathematically defined proxy, including reduced-form cross-currency and cross-unit terms. If physical-unit metadata is added to the compiled static configuration, the existing static-configuration digest includes it. No magnitude bound is part of the contract.

### 6. Canonicalisation (deterministic, not symbolic)

(1) Resolve every target and driver to node identity. (2) Remove authored coefficients that compare equal to zero (`== 0.0`, which includes `-0.0`). (3) Group terms by identical `(d, τ)`. (4) Order the coefficients of a group by their IEEE-754 binary64 bit pattern. (5) Sum in that order with plain float64 left-to-right addition. (6) Remove merged coefficients that compare equal to zero. (7) Reject a row with no remaining term. (8) Sort terms by `(rf_id, chart id)`; order rows by the run axis. (9) Digest the canonical representation. No epsilon tolerance, no Decimal, no compensated summation. Canonicalisation is deterministic over the resolved authored float64 term multiset; it is not symbolic algebraic normalisation: `0.1·Y + 0.2·Y` merges to a coefficient one ULP from authored `0.3`, and the contract does not promise one identity for every real-equivalent expression.

### 7. Identity

`ID_proxy = H(canonical rows in run-axis order)`, each row `(A, canonical {(d_j, τ_j, β_Aj)})`, computed with `digest_named_arrays` over `row_target_rf_ids` (int32), `row_term_offsets` (int32), `term_driver_rf_ids` (int32), `term_chart_ids` (int8), `term_loadings` (float64), mirroring `hash_compiled_book`. `ID_target = (A, κ_A)`, carried by `RunContextIdentity.static_config_digest` and `RunContextIdentity.universe_map_digest`. `ID_run = RunManifest.identity()` over `ID_proxy` (`ShockIdentity = (proxy_model_digest)`), `ID_target`, the world identity (`scenario_config` and the data digest including parametric column keys, horizon, step unit and Case-1/Case-2 declaration), `PnlIdentity.base_levels_digest`, and the as-of. Changing a canonical merged coefficient by one ULP, a driver chart, a driver node or a target node changes `ID_proxy` and `ID_run`; changing only non-semantic authoring syntax does not.

### 8. Mapping, containers, graph

Mapping selects nodes and never becomes weighting. `EXACT` means equal coordinate kind and equal resolved economic coordinate at the run as-of (`[HY+3]` → `2028_FH` and `[Y+3]` → `2028` are not equal; two `[M+k]` tokens with different `anchor_offset` are not equal); `EXPLICIT` means the author selects the pairing; `NEAREST` and automatic interpolation weights are deferred; level-space interpolation `w·B1 + (1−w)·B2` is not shock-space weighting `w·Y1 + (1−w)·Y2`. A container row is authoring shorthand that expands to node-level terms at compile time; the compiled model contains node identities only. A term's input is a world column; a generated shock `X_B` is never an input, so target-to-target chains cannot exist at runtime; a term may reference the data column of a node that also has a model.

### 9. FX and data selection

Reporting FX (D081, D085) is outside the proxy model. The exact LOG identity `A_t = B_t·P_t ⇒ X_A = Y_(B,LOG) + s·Y_(P,LOG)` uses `s = +1` when the configured pair is quoted as target currency per driver currency and `s = −1` for the reciprocal quotation (D078.4, D060); reconstruction reproduces the level identity only if `A_0 = B_0·P_0`; otherwise the same terms are a reduced-form model; omitting an FX term attributes no explicit FX dynamics. The DIFFERENCE-space identity is bilinear and deferred; reduced-form DIFFERENCE terms remain admissible. The model determines the source columns; data availability validates those references and never chooses between models; there is no fallback tier.

### 10. Deferred (coherent, not in V1)

ABSOLUTE_BASE as driver or target (requires an explicit h-horizon ABS contract; on the path −10 → −9 → −8.1 the endpoint statistic is 0.19 and the sign-blind product is 0.21); residual terms as world variables; nonlinear and bilinear families; edge flattening of chart-aligned chains; NEAREST and automatic interpolation; horizon-dependent loadings; physical-unit metadata for PRICE/SPREAD/INDEX; multi-chart parametric columns as a producer capability.

### 11. Acceptance oracles

Closed-form, prior to implementation: identity default (`80 → 82` gives `82`); LOG←LOG (`70·(82/80)^0.95 = 71.6615`); LOG←DIFFERENCE (spread `2.00 → 2.50`, `β = 0.04`: `70·e^0.02 = 71.4141`); DIFFERENCE←LOG (`0.025 + 0.002·ln(88/80) = 0.0251906`); DIFFERENCE←DIFFERENCE (`0.045 + 0.0025 = 0.0475`); two drivers (`70·e^(0.7·ln 1.025 + 0.3·ln 0.98) = 70.790`); mixed charts (`70·e^(0.9·ln(82/80) + 0.02·0.5) = 72.292`); negative loading with explicit base `A_0 = 0.90/0.92`: `EURCHF = USDCHF/USDEUR`, loadings `(+1, −1)`, `A_s = 0.978261·e^0.029853 = 1.00791 = 0.918/0.9108`; FX LOG identity with `A_0 = B_0·P_0 = 73.6`: `A_s = 76.26 = 82·0.93`, and with `A_0 = 70` the model still runs but the level identity does not hold; same node under two charts (row `B` under DIFFERENCE gives `B_0 + 2`, row `A` via `(B, LOG)` gives `71.6615`); SIMPLE aggregation order (`y = (0.10, 0.10)`, `β = 0.5`: `0.105`, not `0.1025`); direct shock-space column with no level path; level-backed domain failure only when levels are used; chart reconstruction versus factor-state admissibility (`A_0 = 1`, `X = −2` reconstructs to `−1`; an FX node is then rejected by the factor-state guard); node identity (`A_2027` and `A_2028` never collide); duplicate terms (`0.3·Y_B + 0.2·Y_B` and `0.5·Y_B` share one identity because `0.3 + 0.2` is exact); authoring-order invariance over one float64 multiset; signed zero (`+0.0` and `−0.0` authored terms disappear identically); identity sensitivity and non-sensitivity; NO_SHIFT node: own column zero and held fixed while a separately requested `(N, LOG, h)` column gives `ln(82/80)` and `A_s = 71.6615`; three-world horizon test (level-backed `h = 5`, direct `h = 5`, direct `h = 1` with `Y = ln(84/80)`, `β = 0.95`, all give `73.321`); ABS target rejected at compile; missing column rejected before numerics; parametric keying (`(B, DIFF, 1)` cannot bind a model that requests `(B, LOG)`; with `(B, LOG, 1)`, `σ = 0.02`, `σ_A = 0.019`).

### 12. Clarifications (2026-09-25; precision only, no semantic change)

1. **The compiled model is run-bound.** The compiled proxy model is `M = f(authored terms, static configuration, as-of, V1 run axis)`: the canonical resolved operator after totality closure against the run axis, including the materialised identity rows that closure requires. It is not a universal, run-independent object. `proxy_model_digest = hash_proxy_model(M)` identifies that operator, not the authoring syntax. The digest is a function result; it is not a stored field of the model.
2. **As-of resolution is upstream of identity.** The dependency order is: authoring → resolution at the as-of → resolved `rf_id`s → totality closure over the run axis → canonical model → digest. An EXACT specification that resolves differently at another as-of legitimately yields a different canonical model and a different digest. The digest is not made invariant to legitimate resolution changes.
3. **Authoring schema versus compiler contract.** The compiler's input contract is the resolved semantic information it needs: authored node terms (target, driver, `τ`, `β`) and authored container terms (target container, driver container, `τ`, `β`), the static configuration, the as-of and the run axis. The compiler owns admissibility, resolution, totality closure, canonicalisation, the canonical representation and the digest function. It does not own file parsing, scenario generation, world construction, valuation or P&L. The authoring file format is a separate edge concern.
4. **Native rows stay outside the V1 proxy model.** `ABSOLUTE_BASE_RETURN` and `NO_SHIFT` axis nodes keep their D025 representation and numerical path. They are not rows of the V1 proxy model and not inputs to `proxy_model_digest`. A later numerical kernel may lower them to its seam as one-term, unit-loading rows; that is a numerical representation only and does not make them part of the authored model, the canonical proxy identity or the proxy digest.
5. **World horizon binding.** The proxy model declares `(driver rf_id, driver chart id)` pairs. The run binds those pairs to the world's declared horizon `h`; the world supplies or validates `(d, τ, h)`; the generator validates the requested `h` against the world declaration. No layer infers `h` from the target, the driver, a resolver, a fallback, source selection or the target chart. `h` is not part of a compiled proxy term.

### 13. Closure clarifications (2026-09-28; precision only, no change to the model)

1. **Identity scope.** A digest identifies only the object whose canonical content it hashes. `proxy_model_digest` identifies the canonical proxy operator of one run. It does not identify a world, a run or a result, and it must not be used alone as a run, result or cache key. `ID_run = RunManifest.identity()` is the identity that must change whenever an input capable of changing the result changes. The manifest section named `ShockIdentity` will carry only `proxy_model_digest` (§7); despite its historical name it identifies the proxy operator, not the world shocks. This record does not rename the schema field.
2. **Horizon.** `β` is horizon-invariant by the V1 modelling assumption (§4); horizon-dependent loadings are not supported in V1 (§10). Two runs at different horizons may share `proxy_model_digest`; their run identities differ through the world identity (`scenario_config` and the data digest).
3. **Target-definition identity is mandatory with the V1 proxy identity (Slice 4 requirement).** `ID_target` (§7) is carried by `RunContextIdentity` inside `PnlIdentity.composition`, which the current manifest schema allows to be `None`. A manifest whose `ShockIdentity` carries `proxy_model_digest` must carry `RunContextIdentity` (`static_config_digest`, `universe_map_digest`); a manifest without it must be refused. `κ_A` is not added to `proxy_model_digest`: for an identity row `κ_A` is part of the operator as the row's driver chart; for an authored row it enters the run identity only through the target-definition identity. The schema change is implemented in Slice 4.
4. **NO_SHIFT and ABSOLUTE_BASE at the numerical seam.** `T⁻¹_NO_SHIFT(A_0, X) = A_0` for every `X`. A NO_SHIFT row needs no world column: at the numerical seam it is a zero-term row, or an equivalent explicit no-shock path, with output `A_0`. The phrase "a zero column" in §1 describes the historical-family kernels only; no world is required to supply a zero column. An ABSOLUTE_BASE row keeps its native world column `(A, ABSOLUTE_BASE_RETURN, h)` with unit loading (D025). Neither row is part of the V1 proxy model (§12.4). The request form for native ABSOLUTE_BASE columns at the world seam is a Slice 3 item.
5. **Slice 2 numerical contract.** For row `n` and scenario `b`: `X_n^(b) = Σ_{t = o_n}^{o_{n+1} − 1} β_t · Y_(b, c_t)`, evaluated in ascending canonical term index. Shocks are stored as float32 and converted to float64 before multiplication; `β`, the products, the accumulation and the output are float64. The reduction kernel uses `riskquant.jit.stable_math_parallel_njit` (no `reassoc`). Parallelism is over independent rows only, never over the term reduction of one row. No `fsum`, pairwise or compensated summation. Reconstruction applies `T⁻¹_κn` after the complete sum. A zero-term row is admissible only for `κ_n = NO_SHIFT`. Offsets, column indices, dtypes and layouts are validated at the numerical edge before the kernel. D041 batch invariance applies.
6. **Oracle tolerance.** The closed-form values in §11 are real-number identities. A test that compares the numerical kernel with a Python float64 reference uses a relative tolerance of `1e-12` and an absolute tolerance of `1e-15`, because FMA contraction (`contract`) makes bitwise comparison with a Python expression inappropriate. Determinism tests (batch size, thread count, repetition) compare kernel output with kernel output bitwise.
7. **Slice 1 behaviours recorded.** (a) EXACT between unequal coordinate kinds (curves: `x_axis_kind`; surfaces: `(x_axis_kind, y_axis_kind)`) or unequal topologies is rejected at compile time, because no pillar can match (§8). (b) Container and node terms that resolve to the same target, driver and chart are merged by §6; no override rule exists. (c) A merged loading that is not finite is rejected (§2).
8. **Reporting-only FX suffix rows; parametric representation.** (a) Reporting-only FX suffix rows (D085 decision 4) are run-axis factor nodes and are rows of the proxy compiler axis under §1 on the same terms as valuation-prefix rows: the compiler's run axis is the whole run axis, prefix then suffix. A suffix node whose configured `shift_type` is a V1 reconstruction chart has exactly one compiled row, the identity row `(q, κ_q, 1.0)` unless terms are authored for it, and it enters `proxy_model_digest`; a suffix node whose `shift_type` is `ABSOLUTE_BASE_RETURN` or `NO_SHIFT` keeps its D025 native path (§12.4, §13.4). "Reporting FX" in §9 denotes the reporting conversion — the settlement currency, the route and its orientation, and the D081 formula `ΔW = ΔV·q_s + V_0(q_s − q_0)` (D085 decisions 1–3 and 5) — which the proxy model neither performs nor alters; the reporting layer reads `q_s` from the run axis and does not define how it is generated. A node has one reconstruction chart and one compiled row whether it is in the prefix or the suffix; its reporting role and its axis position do not change that. Consequences: `proxy_model_digest` depends on the reporting currency only through the run axis (§12.1, §13.1); a change of route orientation, settlement currencies or suffix base level changes `ID_run` but not `proxy_model_digest`. **§13.8(a) closed. Status: LOCKED (2026-09-29).** (b) The parametric world representation of `(d, τ, h)` (declared horizon, step unit, chart, D034 Case declaration) was open when this decision was recorded. The parametric horizon contract (meaning, ownership of the two declarations, exact matching) is fixed by D089; D090 closes the forecast-period representation and D091 closes the chart representation for V1, in which the D034 Case declaration does not arise.

**Status: AMENDED IN PART by D090 and D091 (2026-09-29).** Only §13.8(b) is updated, to record that the representation it left open is now decided. §13.8(a) is closed (LOCKED, 2026-09-29) by the text recorded in §13.8(a); every other clause is unchanged.

---


## D089 — Parametric two-sided horizon contract

**Status:** LOCKED (owner instruction, 2026-09-28).
**Scope:** V1 direct parametric worlds (`ParametricGeneratorConfig`, `ParametricRiskData`, `ParametricGenerator`). **Relation to D088:** applies §2 (world-declared elapsed horizon), §3 (horizon owned by the scenario world, configuration and data), §4 (one horizon per world; no hidden horizon or volatility scaling) and §7 (the parametric data digest includes the horizon). D088 §12.5 is unchanged: once this contract holds, the matched horizon is the world declaration against which requested column horizons are validated. **Does not decide:** the representation of the horizon (field names, type, `(magnitude, unit)` or other form, unit vocabulary, serialization); parametric chart representation; calibration methodology. When this decision was recorded, D088 §13.8(b) was open for the first two; D090 closes the horizon representation and D091 the parametric chart representation.

1. **One world horizon, two declarations.** A V1 parametric world has exactly one forecast horizon `H_W`. Two independently produced artifacts declare it. `ParametricGeneratorConfig` declares the requested forecast horizon `H_requested`: the run requests a parametric world whose shock distribution represents this horizon. `ParametricRiskData` declares the effective period `H_effective`: the supplied volatility, dependence and distributional parameters represent a shock distribution at this horizon. The contract is `H_requested = H_effective = H_W`. It is a consistency relation, not a transformation.
2. **Why both declarations exist.** The arrays cannot reveal the period they represent. A configuration-only declaration could not detect data that represents another period; a data-only declaration would leave the run without a statement of what it requests. The two values belong to independently produced artifacts (the run specification and the supplied data), and their equality establishes that the artifacts are compatible.
3. **Horizon-matching rule.** A V1 direct parametric world represents one stochastic law at one forecast horizon. Its scale, dependence and distributional parameters jointly define that horizon-specific law; the horizon belongs to the law as a whole, not to each parameter. Parameters calibrated from n-period returns define an n-period parametric shock. Weekly parameters with `H_requested = H_effective = 1 week` give a one-week shock; five-week parameters with both declarations equal to 5 weeks give a five-week shock.
4. **Generator boundary.** The generator computes `Y_b = GSM_b(z_b Lᵀ) ⊙ σ` from the supplied parameters. Its multiplications by the mixing scale and by σ are distributional. It performs no temporal/horizon scaling or conversion: it does not aggregate shorter-horizon shocks, simulate temporal paths, apply √h scaling, rescale dependence for another horizon, interpolate between horizons, convert daily parameters to n-day parameters or weekly parameters to multi-week parameters, or infer a horizon from `total_scenarios`. It does not know how many days make up a week. Weekly parameters paired with a 5-week request are not turned into `√5 · σ_1w`: generic temporal aggregation can change variance, covariance (`Cov(Σᵢ rᵢ, Σⱼ qⱼ) = Σᵢⱼ Cov(rᵢ, qⱼ)`), correlation and distributional shape, and deriving one horizon's law from another is outside V1.
5. **Exact matching and fail-fast ordering.** `H_requested = H_effective` is checked exactly, with no conversion, at the configuration/data composition seam: in the run's composition contract (the checks `run_scenario_var_es` performs before `build_run_manifest`), that is before identity, before any RNG draw, before scenario generation and before P&L. The generator does not own this check. A mismatch fails. Examples: requested 10 business days with effective 1 week fails; requested 5 weeks with effective 1 week fails; requested 1 week with effective 1 week passes.
6. **Lookback is not horizon.** The calibration lookback is distinct from the effective return period, which equals the forecast horizon. Example: 500 business-day observations are the lookback; 10-business-day returns are the effective return period; a 10-business-day parametric shock is the forecast horizon. The lookback is not part of the direct generator's horizon. Its storage is not decided here.
7. **Calibration boundary.** `RiskFactorMatrix` → future calibration methodology → n-period returns → horizon-matched parameters → `ParametricRiskData` → `ParametricGenerator` → n-period shocks. Calibration creates parameters for the desired horizon; the generator does not change the horizon of supplied parameters. Estimator, weighting and λ, lookback, overlapping or non-overlapping returns, dependence estimator, tail fitting, filtering and missing-data policy are not decided here.
8. **Risk measures.** VaR and ES are both calculated from the P&L distribution generated for the world's forecast horizon: `H_VaR = H_ES = H_W`. No separate risk-measure horizon exists.
9. **Identity.** The requested horizon is part of `ParametricGeneratorConfig` and therefore of the serialized `scenario_config` in `RunManifest`; changing it changes the run identity. The effective period is part of the meaning of the supplied data and must enter `DataIdentity`. The run identity combines both; neither identity alone identifies the complete world. A run is valid only when the two declarations match.
10. **Proxy V1 unchanged.** Slice 1 (`X_A = Σ_j β_Aj Y_j`), Slice 2 (numerics and `(d, τ, h)` binding) and `proxy_model_digest` are unchanged; the proxy operator does not own a horizon. β is not rescaled when the world horizon changes; its horizon-invariance remains the separate D088 §4 modelling assumption.

---

## D090 — Parametric forecast-period representation

**Status:** LOCKED (owner approval, 2026-09-29).
**Implementation status:** Implemented 2026-09-29: `ForecastPeriodUnit`, `ParametricRiskData.effective_period_count` / `effective_period_unit`, `ParametricGeneratorConfig.forecast_horizon`, and the equality gate `resolve_parametric_binding_horizon`, applied before identity and generation.
**Scope:** V1 direct parametric worlds. **Relation:** represents the two D089 declarations (`H_W = H_requested = H_effective`); D089 is unchanged. Closes the horizon part of D088 §13.8(b); the chart part is closed by D091. Matrix-backed scenario configurations are unaffected: their `horizon` keeps counting intervals of the matrix's own date axis.

1. **Meaning.** A `ForecastPeriod = (count, unit)` is the forecast/return period represented by a direct parametric law: the period over which the supplied law describes the change. It is not a number of observations, a lookback, a sample size, a scenario count, a number of simulation steps or a calendar-day duration; no temporal path exists. `H_W`, `H_requested` and `H_effective` (D089) are ForecastPeriods, and a V1 period is `(n, BUSINESS_DAY)` with `n > 0`.
2. **V1 vocabulary.** Exactly {`BUSINESS_DAY`}. One `BUSINESS_DAY` step is one step of business-day time: the interval from one business-day close to the next. `(n, BUSINESS_DAY)` is the change over n consecutive business-day steps and spans n + 1 closes (six consecutive closes give five steps, `(5, BUSINESS_DAY)`). The unit does not identify a holiday calendar, session schedule or observation schedule. For a multi-factor law the n steps are one common business-day grid; the calendar defining that grid is producer provenance. Equal ForecastPeriods produced under different calendars therefore satisfy the same request; calendar-specific periods are outside V1. WEEK, MONTH, CALENDAR_DAY, calendar-qualified and sub-daily units are excluded; each may be added only by a later decision that defines its sampling convention and guarantees that no two encodings denote the same period.
3. **Single definition.** The vocabulary and its mapping between the configuration string and the core code (`"BUSINESS_DAY"` ↔ int8 `1`) are defined once and used by configuration validation, data validation, hashing and the pairing check.
4. **Equality.** Two ForecastPeriods are equal if and only if count and unit are both equal. There is no normalisation, no unit equivalence and no conversion.
5. **Representation and ownership.** `ParametricGeneratorConfig.forecast_horizon = {count: StrictInt > 0, unit: "BUSINESS_DAY"}` (required, no default) declares `H_requested` and is serialized in `scenario_config`. `ParametricRiskData` carries `effective_period_count` (int64-compatible positive integer, not bool) and `effective_period_unit` (int8 code), both required with no default, declaring `H_effective`. `hash_parametric_risk_data` includes both as named arrays (int64 and int8, shape (1,)). `RiskMeasureIdentity`, `RiskRun` and `proxy_model_digest` are unchanged; VaR and ES are computed from the P&L distribution generated for `H_W` (D089).
6. **Producer assertion.** `H_effective` is an assertion by the producer of the data that the supplied numerical law, including the scale of `volatilities`, represents that period. The system validates its form and its equality with `H_requested`; it cannot verify the assertion from the arrays. A false assertion (for example annualised σ, or one-day estimates declared as five business days) is a producer error.
7. **Validation.** Construction rejects missing, non-integer, bool, out-of-range or out-of-vocabulary values. The exact equality `H_requested = H_effective` is checked in the run's composition contract before `build_run_manifest` (D089). The generator performs no temporal aggregation, √h scaling, annualisation, horizon inference or path construction (D089).
8. **Binding reduction.** After the equality check, the binder's world and requested horizons are both `h = count(H_W)`. The reduction is legal because one binding concerns one world, whose unit is fixed and whose declared horizon is the only horizon source (D088 §4, §12.5). The `(d, τ, h)` contract and the Slice 2 kernel are unchanged. A future run-level horizon source, or binding across worlds, must compare full ForecastPeriods before reducing.
9. **Calibration obligation.** A producer declares its target ForecastPeriod and emits data whose effective period equals it. Before emitting `(n, BUSINESS_DAY)` it must establish that its changes span n business-day steps of a declared business-day grid; `RiskFactorMatrix.dates` alone does not establish business-day sampling, since the date axis admits weekends and irregular gaps. The mechanism, the source-grid declaration format, the estimator, lookback, overlap and missing-data policies and the calendar implementation are outside this decision. A producer that cannot justify the period must fail rather than emit it.

---

## D091 — Parametric column chart, scale and dependence contract

**Status:** LOCKED (owner approval, 2026-09-29).
**Implementation status:** Implemented 2026-09-29: `ParametricRiskData.chart_ids` and the unit-diagonal check on the dependence factor; the charts enter the data digest, and the world binding keys parametric columns by `(factor_ids[i], chart_ids[i])`.
**Relation:** applies D088 §2–§5, §10, §12.5; D089 and D090 unchanged; clauses 3 and 8 use D090's `ForecastPeriod`. **Amends D034 additively:** adds `chart_ids` and `R_ii = 1` to the `ParametricRiskData` data invariants; D034's sorted-unique keys, finite square factor and finite strictly positive scales are retained, and invariant ownership is unchanged. D034's Case-1/Case-2 declaration cannot arise in V1, because no node has several parametric charts (D088 §10); this decision enables no multi-chart world.

1. **Driver chart.** `ParametricRiskData` carries `chart_ids`: int8, shape (K,), aligned with `factor_ids`, values in {LOG_RETURN = 1, SIMPLE_RETURN = 2, DIFFERENCE = 3} (`ShiftType` codes). NO_SHIFT, ABSOLUTE_BASE_RETURN and any other code are rejected. `chart_ids[i]` is the driver chart τ of column i. `ParametricGeneratorConfig` carries no driver-chart declaration. The target chart κ remains the target factor's configured `shift_type` (D088 §3) and is never inferred from, or copied into, the data.
2. **One chart per node.** A V1 parametric world supplies exactly one driver chart per factor: multi-chart parametric production is deferred (D088 §10), and sorted unique `factor_ids` with an aligned `chart_ids` represents that restriction. A request for another chart of that factor fails as a missing column; no chart conversion exists.
3. **Column meaning.** Column i is the direct column `Y_(d, τ, H_W)` with `d = factor_ids[i]` and `τ = chart_ids[i]`: LOG `ln(V(t+H_W)/V(t))`; SIMPLE `V(t+H_W)/V(t) − 1`, the compound simple return over `H_W`; DIFFERENCE `V(t+H_W) − V(t)` in the node's canonical level unit (D069). These are D088 §2's forward charts applied to the change over `H_W` (D090). The column implies no level path (D088 §4). The producer asserts this meaning together with the effective period (D090).
4. **Numerical law.** `W_b = L z_b`, `z_b ~ N(0, I_K)`; `Y_{b,i} = σ_i · s_b · c_{b,i} · W_{b,i}`. For the normal, Student-t (ν > 2), Laplace and variance-gamma laws, `s_b` is the implemented mixing scale, independent of `z_b`, with `E[s²] = 1`, and `c_{b,i} = 1`. For regime switching, `s_b = 1`, `C_b ~ Bernoulli(p)` with p ∈ [0, 1] independent of `z_b`, and `c_{b,i} = 1` in the base/calm component (`C_b = 0`) and `m_i` in the crisis component (`C_b = 1`), with `m_i` finite and strictly positive.
5. **Dependence.** `L` is a dimensionless dependence factor with `R = L Lᵀ`; the semantic requirement is `R_ii = 1` (`UNIT_DEPENDENCE_DIAGONAL`). `R` is the correlation parameterisation of the standardised base/calm component. No additional strict-positive-definiteness or lower-triangularity invariant is imposed on `L`; since `R = L Lᵀ`, `R` is necessarily symmetric positive semidefinite. The name `cholesky_matrix` is a label. Acceptance rule: `|Σ_k L_ik² − 1| ≤ NUMERICAL_EXCURSION_TOLERANCE` (riskquant.constants), computed in float64 as an absolute comparison. The constant's value and rationale (floating-point error of an inner product of unit-norm rows) are reused; its clamp-inside-tolerance behaviour does not apply to input data, because `L` is never coerced (D034). Within tolerance `L` is accepted unchanged; beyond it construction raises; no clamping or renormalisation occurs.
6. **Scale.** Non-regime laws: `σ_i = Std(Y_i)` at `H_W` and `Cov(Y) = D_σ R D_σ`; for Student-t, σ is the standard deviation, not the t-scale parameter. Regime switching: `σ_i` is the standard-deviation parameter of the base/calm component, and `m_i σ_i` that of the crisis component. For each component with positive probability (base/calm when p < 1, crisis when p > 0), the conditional law has standard deviations `σ_i`, respectively `m_i σ_i`, covariance `D_σ R D_σ`, respectively `D_σ D_m R D_m D_σ`, and correlation `R`, because each component applies a strictly positive deterministic diagonal scale. A component of probability zero defines no conditional law; its parameters remain part of the model. For all p ∈ [0, 1], `Cov(Y) = D_σ[(1−p)R + p D_m R D_m]D_σ` and `Var(Y_i) = σ_i²[(1−p) + p m_i²]`. For i ≠ j, the unconditional correlation equals `R_ij` if and only if at least one of `R_ij = 0`, `p = 0`, `p = 1` or `m_i = m_j` holds; otherwise it has the sign of `R_ij` and strictly smaller magnitude. σ carries its chart's units; `L`, `R`, `s` and `m` are dimensionless, so mixed charts form a coherent joint law.
7. **Separation.** The data describes driver variables only. The generator is chart-blind: it applies no chart-dependent transformation. Chart semantics are established by the producer and consumed through binding.
8. **Binding.** World column keys are `(factor_ids[i], chart_ids[i])` at `h = count(H_W)` (D090), passed unchanged to `bind_term_columns`. Slice 2 is unchanged.
9. **Identity.** `hash_parametric_risk_data` includes `chart_ids` as a named int8 array alongside `factor_ids`, `L`, `σ` and the effective period (D090). `DataIdentity` identifies the data artifact, not an equivalence class of parameterisations: distinct `L` arrays with equal `L Lᵀ` are distinct artifacts. The law is identified jointly with `scenario_config`. `proxy_model_digest` is unchanged. The identity of the regime multipliers is fixed by D092.
10. **Calibration semantics.** A producer declares τ per factor explicitly and supplies, at `H_W`, the standard deviation and the correlation of the standardised chart-τ changes (base/calm component under regime switching), with the crisis multipliers as crisis-component scale modifiers. No estimation methodology is fixed here.

---

## D092 — Regime-switching crisis multiplier ownership and identity

**Status:** LOCKED (owner approval, 2026-09-29).
**Implementation status:** Implemented 2026-09-29: the scalar or `"per_factor"` configuration form, the optional data vector, and the composition gate `resolve_parametric_crisis_multipliers`, applied before identity.
**Relation:** fixes the regime-multiplier identity left open by D091 clause 9; D088–D091 unchanged; the stochastic law of D091 and the RNG semantics of D087 are unchanged. **Amends:** D034 (strict typing, validation split, construction-data invariants, and its D088 status note), D038 (multiplier typing sentence) and D087 (clause 4 identity composition); each amended sentence is marked "D092" in place.

1. **Configuration level.** `RegimeSwitchingDistribution` in `ParametricGeneratorConfig` keeps `prob_crisis` (p ∈ [0, 1]) and `crisis_vol_multipliers`. `crisis_vol_multipliers` is required and has exactly two admissible states: a finite, strictly positive real scalar (the uniform form; D038 typing), or the literal `"per_factor"` (per-factor values supplied by `ParametricRiskData`). A sequence is rejected and never reinterpreted. Absence does not imply the per-factor form. Non-regime distribution variants cannot carry the field. A uniform scalar is a model-level parameter: with `m_i = m` for all i the regime law is a common two-point scale mixture.
2. **Data level.** Heterogeneous crisis multipliers are factor-law parameters (the crisis-component scale `m_i σ_i` of named columns). `ParametricRiskData` gains the optional field `crisis_vol_multipliers`: float64, shape (K,), C-contiguous, finite, strictly positive, aligned positionally with the canonical `factor_ids`, so that `crisis_vol_multipliers[i]` belongs to `factor_ids[i]`. `factor_ids` remains the sole factor key. `ParametricRiskData` never sorts; a producer places all aligned arrays (`factor_ids`, `chart_ids`, rows and columns of `L`, `volatilities`, `crisis_vol_multipliers`) in canonical order together. There is no silent factor matching. The data object validates the vector whenever it is present.
3. **Composition.** The run's composition contract, before `build_run_manifest`, enforces: scalar with no data vector is valid; `"per_factor"` with a data vector is valid; scalar with a data vector fails; `"per_factor"` without a data vector fails; any non-regime law with a data vector fails. There is no broadcasting, no fallback and no positional reinterpretation across artifacts.
4. **Validation domain.** `m_i = 0`, negative values, ±∞ and NaN are rejected; any finite strictly positive value is accepted; no upper bound is introduced. Multipliers are required for regime switching at every p ∈ [0, 1]: at p = 0 or p = 1 the zero-probability component's parameters remain model parameters (D091).
5. **Identity.** When present, the heterogeneous vector enters `hash_parametric_risk_data` as the named float64 array `crisis_vol_multipliers`; its presence or absence changes the data digest. `prob_crisis`, the scalar multiplier and the `"per_factor"` declaration enter through `scenario_config`. `proxy_model_digest` is unchanged. A scalar and an equal uniform data vector represent the same stochastic law but are distinct artifacts.
6. **Horizon.** `p`, the scalar and the per-factor multipliers describe the law at `H_W` (D089). The per-factor values are part of the data producer's assertion (D090); `p` and the scalar are asserted by the configuration's author.
7. **Generator.** The stochastic law and its numerical application are unchanged. The generator continues to consume either a scalar or a positional array; the composition layer establishes that array position is canonical `factor_ids` position before the generator receives it.

---


## D093 — Proxy V1 tabular authoring edge (proposed closure of D088 §13.2 for the frame representation)

**Status:** PROPOSED (2026-09-30). Requires owner approval. Until approval, D088 §13.2 remains OPEN.

**Context.** D088 §12.3 fixes the compiler's authored input contract: node terms (target, driver, τ, β) and container terms (target container, driver container, τ, β). The authoring file format is a separate edge concern (§13.2, OPEN). Stage E removed the D013 proxy-configuration edge and its canonical ingest fixture. RiskQuant configuration inputs enter through Polars frames with a canonical ingest fixture under `riskquant.ingest_fixtures.config` (the static configuration convention). Proxy V1 had no such edge, so a Proxy V1 model could be authored only as Python objects.

**Decision (proposed).**

1. **Frame.** A `polars.DataFrame` with exactly five columns, identified by name; physical column order is not semantic. Dtypes, compared exactly and never cast: `term_kind`, `target_name`, `driver_name`, `driver_chart` are `pl.String` (`pl.Utf8` is the same dtype); `loading` is `pl.Float64`. Any other dtype is refused, including Int64, Float32, Decimal, String or Object loadings and Categorical or Enum text columns. A frame with zero rows is valid and authors no terms.
2. **Values.** No null in any column. `target_name` and `driver_name` are non-empty and not whitespace-only; they are passed verbatim (no trimming, case change, resolution or existence check). `term_kind` is exactly `"NODE"` or `"CONTAINER"`; `driver_chart` is exactly `"LOG_RETURN"`, `"SIMPLE_RETURN"` or `"DIFFERENCE"`; matching is case-sensitive, with no aliases and no whitespace normalisation. `loading` is finite (NaN, +inf and −inf are refused); `0.0`, `−0.0` and negative values are admissible and are preserved bit-exactly.
3. **Mapping and preservation.** Each row maps to exactly one `NodeProxyTerm` (`NODE`: factor-node names) or `ContainerProxyTerm` (`CONTAINER`: curve or surface names), in input row order. The edge preserves row count, row order, duplicate rows and duplicate `(target, driver, chart)` terms, zero loadings with their sign, negative loadings, mixed kinds and several charts of one driver. It does not resolve names, read an as-of or the static configuration, infer `κ` or the horizon, create identity rows, expand containers, merge, drop, sort or canonicalise.
4. **Compiler ownership.** Name resolution at the run as-of, container expansion (`EXACT`), admissibility, identity rows, canonicalisation (D088 §6, including the grouping of duplicate terms and exact-zero elimination) and the digest belong to `compile_proxy_model`. Consequences: `proxy_model_digest` is invariant under row permutation; a frame and the equivalent typed terms compile to identical canonical arrays and digest and give identical run results.
5. **Not authored.** The target chart `κ` (owned by the static configuration), the horizon (owned by the world), an identity marker (a V1 target without terms receives its identity row at compile time), weights normalisation and any D013 field.
6. **Canonical fixture.** `riskquant.ingest_fixtures.config.risk_proxy_model.canonical_risk_proxy_model` (schema `RiskProxyModelTermData`). Tests pin its compiled rows (derived by hand) against the canonical static configuration at `CANONICAL_AS_OF_DATE` and its `proxy_model_digest`, and re-derive that digest from the hand-derived rows without the production hashing code.
7. **Not decided.** File loaders (xlsx, csv or other) remain OPEN; they would read into this frame. The production API stays typed (`run_scenario_var_es(proxy_terms=...)`); the frame is an authoring adapter to it.

**Consequences.** No change to D088 semantics, canonicalisation or identity. The D013 frame schema `RiskProxyConfigData` is removed.

---

# Decision Record for `riskquant.credit`

## D-C01 — `obligor_uids_target` ordering (credit / exposure→JTD bridge)

`obligor_uids_target` may be supplied in **any order**. Uids must be unique; duplicates raise. The output rows are emitted in exactly the caller's target order. Sorting is an internal implementation detail: `aggregate_ead_to_obligor` creates a sorted copy of the target, performs the `searchsorted` lookup against that sorted-unique array, and maps the returned ranks back through the sort permutation.

**Rationale.** The canonical target is `jtd_portfolio.obligor_uids`, which `CompiledJtdPortfolio` builds by concatenating per-model block uid arrays. Band order is structurally unsorted, and `run_jtd` requires `ead_by_obligor` rows in exactly that order. Requiring a sorted target would force every caller to sort and then restore band order, unnecessarily exposing an internal ordering constraint at the composition boundary. `align_obligor_ead_to_portfolio` already supports UID-safe alignment independently of source ordering.

**Superseded behaviour.** Before this decision, the bridge used `searchsorted` directly against `obligor_uids_target`, despite `riskquant._arrays` defining the `universe` argument as sorted-unique and leaving that invariant to the caller. When the documented JTD band order was supplied, the lookup could silently lose or mis-attribute exposure. In the reproduced two-block case, 60% of portfolio EAD was lost with no exception.

**Unchanged.**

* An EAD row naming an unknown netting set still fails fast.
* A netting set whose obligor is absent from the target is still dropped, because modelling a subset of the obligor universe is legitimate.
* The output remains `float64`, C-contiguous, with row `i` corresponding exactly to `obligor_uids_target[i]`.

---

## D-C03 — a modelled obligor with no netting-set exposure

A JTD-modelled obligor with no corresponding netting-set exposure is an **input/composition error**, not an implicit zero. `aggregate_ead_to_obligor` raises and names the offending uid(s).

**Rationale.** Zero EAD means zero loss in every scenario, so silently zeroing a modelled obligor removes it from EL, VaR and ES without any diagnostic. `align_obligor_ead_to_portfolio` already enforced this semantic and documents it explicitly: a modelled obligor with no exposure is a gap in the inputs, not something to silently zero. The two exposure-to-JTD alignment paths must therefore have the same answer to the same semantic question.

**Effect on solutions-tier callers.** There is no effect on the default `run_credit_exposure` path. Its default target is `np.unique(master.obligor_uid)`, so every target obligor has at least one corresponding netting set by construction. The error is reachable when a caller supplies an explicit target containing an obligor not covered by the netting-set master. That state represents an inconsistent composition and must be surfaced.

**Unchanged.**

* A netting set whose obligor is absent from the modelled target remains a legitimate soft case and is dropped.
* A genuinely modelled subset of the obligor universe remains supported.
* No JTD-modelled obligor may acquire an implicit zero EAD merely because its exposure mapping is missing.

---

## D-C04 — exact scenario coverage for batched credit execution

All batched scenario producers consumed by the credit execution layer must cover the declared scenario range **exactly once**:

```text
[0, total_scenarios)
```

A valid batch stream therefore has:

```text
no gaps
no overlaps
no duplicate writes
no negative offsets
no zero-sized batches
no out-of-range writes
no truncation
```

Batch arrival order itself is not part of the contract. Valid batches may be yielded out of scenario order provided their spans still form an exact partition of `[0, total_scenarios)`.

**Rationale.** `build_credit_factor_matrix` and `run_jtd_streaming` use `np.empty` for their destination accumulators. An unwritten scenario therefore contains uninitialised memory which can propagate directly into factor values, portfolio loss, EL, VaR and ES. A final length check is insufficient because overlapping batches can have the expected total size while leaving other scenarios unwritten. The implementation therefore records per-scenario write coverage, rejects overlap and overrun at the offending batch, and rejects gaps or truncation at completion.

**Enforcement.**

* Coverage is tracked at the Python streaming boundary.
* Nothing related to coverage bookkeeping crosses a JIT boundary.
* Successful return implies every scenario was written exactly once.
* The producer cannot certify its own correctness through self-declared batch metadata; the consumer's declared `total_scenarios` is authoritative.

**`CreditFactorBatch.scenario_count_total`.** This field is currently redundant metadata. It is not read anywhere in `src/` and is not authoritative for coverage. A later contract review may either remove it or explicitly make it authoritative and validate it. No decision to change it is made here.

---

## D-C15 — batch size is not a stochastic parameter

Internal batch size is a **performance/execution parameter**, not a stochastic or risk parameter. At fixed economic inputs, seed and scenario count, changing the internal batch size MUST NOT change the generated scenario stream.

`MixedCreditFactorGenerator` achieves this by using exactly one RNG consumer, the C-contiguous `standard_normal` fill, with every non-normal marginal derived deterministically from the normal already occupying its own cell through the corresponding probability-integral transform.

### Reproducibility scope: environment-relative

Cross-environment bitwise reproducibility is **not** claimed.

Within a fixed supported NumPy/SciPy environment, with the required C-order buffer layout, the stream is contractually batch-invariant and bitwise reproducible. This has been verified across multiple factor dimensions, scenario counts and batch sizes, including mixed normal, uniform, indicator, chi and beta marginals.

Across NumPy versions, reproducibility is not guaranteed because `numpy.random.Generator` explicitly provides no general version compatibility guarantee for its bit stream. In addition, the equality between one large `standard_normal(out=...)` call and concatenated smaller calls is an implementation property of the current NumPy implementation, not a cross-version API guarantee.

The PIT construction additionally uses `scipy.special.ndtr`, `gammaincinv` and `betaincinv`. Consequently, the environment relevant to bitwise scenario reproduction now includes **both NumPy and SciPy**.

The C-contiguous layout of the RNG output buffer is part of this implementation contract. It is not merely a performance choice: changing the output memory order can change the generated stream.

### Backward compatibility

The pre-D-C15 and post-D-C15 generators define different stochastic constructions. Therefore a historical scenario stream generated before this change is **not expected to reproduce from the same seed under the new construction**.

Historical credit VaR/ES results are not thereby deemed mathematically invalid. The issue is reproducibility and provenance: a stored result must be distinguishable from results generated under the new stochastic construction.

Any release containing this change must therefore identify the scenario-construction version in provenance. The reproducibility identity should include, at minimum:

```text
scenario construction identifier
seed
scenario count
NumPy version
SciPy version
```

and the package/release version must distinguish the pre-D-C15 and post-D-C15 implementations.

**Required release condition.** The provenance change is part of the release contract for D-C15. The stochastic construction must not be treated as reproducible solely from `(seed, input_hash, engine_version)` when those fields do not identify the generator construction and numerical environment.

---

## D-C16 — SA-CCR public trade representation: direction, maturity, and the house-calibration boundary

**Status:** LOCKED
**Findings closed:** P2-11A, P2-11B, P2-11C

The public SA-CCR exposure model (`solutions.credit_exposure`, `model="saccr"`) implements the CRE52 Standardised Approach for its declared supported population: **unmargined netting sets, linear (non-option, non-CDO) trades, and equity and commodity asset classes**.

For the supported equity/commodity population, `TradeRow.volume` represents the **number of units referenced by the trade**, and adjusted notional is:

$$
d_i = current\_price_i \times volume_i.
$$

The trade-level effective notional is the CRE52 structural product:

$$
\boxed{
D_i=d_i\times MF_i\times\delta_i
}
$$

and the resulting signed `D_i` flows into the P2-03 signed subclass aggregation unchanged.

### P2-11A — supervisory delta / direction (structural)

Each SA-CCR trade carries an explicit `direction` in `{"long", "short"}`.

For the supported linear population:

$$
\delta_i =
\begin{cases}
+1 & \text{long},\\
-1 & \text{short}.
\end{cases}
$$

Direction is a **STRUCTURAL CRE52 term**, not a house-calibrated parameter.

Direction is required per SA-CCR trade and validated fail-fast. An absent direction is rejected at the request boundary, and an unexpected value is never silently mapped to short. The public schema constrains the value to `Literal["long", "short"]`, and the recipe-side mapping independently rejects any other value.

Options and CDO-tranche supervisory-delta treatment are outside the supported population.

### P2-11B — maturity factor (structural)

Each SA-CCR trade carries an explicit `remaining_maturity_years`, representing the CRE52 remaining maturity \(M_i\), in **years**, before the floor/cap transformation, supplied by the caller.

It is not:

```text
an already-floored maturity
a tenor measured from trade inception
a raw day count
the maturity factor itself
```

The unmargined CRE52 maturity factor is:

$$
\boxed{
MF_i=
\sqrt{
\operatorname{clamp}
\left(
M_i,\frac{10}{250},1
\right)
}
}
$$

The 10-business-day floor (`10/250` years, using 250 business days per year) and the one-year cap are **STRUCTURAL CRE52 rules**, not house parameters.

A valid positive \(M_i\) below `10/250` years is clamped to the regulatory floor. A non-positive \(M_i\) is a boundary error and is rejected; it is never silently clamped.

The credit-exposure engine does not derive maturity from dates, does not own an as-of/calendar calculation for this purpose, and does not depend on `market_data` calendar infrastructure. The caller supplies the remaining maturity \(M_i\) directly.

Margined maturity-factor treatment is outside the supported population.

### P2-11C — meaning of "house-calibrated"

"House-calibrated" refers only to RiskQuant-owned **CONFIGURATION**:

```text
asset-subclass definitions
subclass → hedging-set mapping
supervisory factors SF
per-hedging-set correlation rho
alpha
multiplier floor
```

It does **not** mean:

```text
direction
supervisory delta
remaining maturity
maturity factor
maturity formula
or any other SA-CCR structural equation
```

The SA-CCR hedging-set aggregation uses the existing CRE52 single-factor structure, including full signed offset within a subclass and partial correlation-based offset across subclasses within one hedging set, with no offset across hedging sets.

### Supported population and explicit scope boundary

The current public model supports only:

```text
unmargined netting sets
linear non-option / non-CDO trades
equity and commodity asset classes
explicit long / short direction
unmargined remaining maturity down to the CRE52 10-business-day floor
```

The following populations are **outside the supported public model contract and are not representable by its current public trade model**:

```text
options
CDO tranches
interest-rate asset classes
FX asset classes
credit asset classes
margined transactions
```

No conclusion is made here about whether any of those populations should be supported by a future model extension.

### Structural / house boundary

The distinction between CRE52 structure and RiskQuant-owned configuration is normative:

$$
\boxed{
\text{house calibration/configuration}
\neq
\text{replacement of SA-CCR structural mathematics}
}
$$

In particular, \(\delta\) and \(MF\) are never to be described as house-calibrated parameters.

### P2-03 relationship

The P2-03 signed subclass aggregation remains unchanged:

$$
D_{jk}=\sum_iD_i
$$

$$
A_{jk}=SF_kD_{jk}.
$$

P2-11 supplies the public trade-level signed effective notionals that make this existing signed aggregation reachable from the public SA-CCR path. No change to the P2-03 aggregation or downstream SA-CCR kernel is implied by this decision.
