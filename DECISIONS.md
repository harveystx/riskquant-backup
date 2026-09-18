# DECISIONS.md — RiskQuant Architectural Decision Record

Durable decisions and rationale. New decisions append sequentially. Do not turn
an unresolved discussion into LOCKED. Mark superseded decisions clearly.

> Scope note: D001–D0xx below cover both the long-running audit workstream
> (`RISKQUANT_AUDIT_LEDGER.md`, RQ-AUD-###) and the E2E sensitivity-VaR demo
> build-out. Audit remediations are recorded compactly; the demo/pipeline
> decisions carry full rationale because they most constrain ongoing work.

---

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

---

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
float64 1-D length-K C-contig finite strictly-positive. Placement proven from
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
each parameter's domain preserved (df>2, nu>0, prob_crisis in [0,1], multipliers
strictly positive scalar-or-per-factor). NOTE: D032/D033 use lax `float` +
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
(total_scenarios, seed, distribution selection, distribution parameters, and the
per-factor `crisis_vol_multipliers` length-vs-K check, which is data-coupled and
only knowable at construction). The generator's own array checks remain because
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
values stored as Python `float`. `crisis_vol_multipliers` keeps its scalar-or-
sequence semantics with the rule applied to its numerical leaves. Evidence: new
`tests/solutions/scenario_models/test_continuous_typing.py` 160 passed / 2
skipped; seam suite 405 passed / 2 skipped; tests/simlab 281; focused regression
+ solutions 1565 passed / 4 skipped; E2E `ALL CHECKS PASSED`; D019 goldens
preserved (no re-baseline); changed-file lint clean; scope diff = `schemas.py`
plus the new test. D037 remains deferred: archaeology confirms no production or
demo caller constructs co-seeded independent engines. See
`SIMLAB_PRODUCTION_CLOSURE.md`.

---

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

---

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

---

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

---

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

---

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

---

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

Six of eight modules beneath `tests/valuation/` import modules/classes that do not
exist in the current repository, including:

* `riskquant.valuation.models.black_scholes`;
* `pricers._deprecated.*`;
* `VanillaForwardNodeBlock`;
* `VanillaForwardCurveBlock`;
* `VanillaEuropeanOptionNodeBlock`;
* `VanillaEuropeanOptionCurveBlock`.

`test_engine.py` also constructs the current `SpotBlock` using fields that no
longer exist and calls `run_valuation` without its required `out=` argument.

These are collection/API failures rather than failed mathematical assertions.

The only live mathematical valuation test found is a small Spot-kernel smoke
test and does not establish correctness of the broader valuation package.

Therefore prior mathematical conclusions are audit evidence, not repository
assurance.

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

## D061 — No discount-factor or price-curve producer until the curve-point contract exists

**Status:** OPEN — UPSTREAM ECONOMIC CONTRACT MISSING (fail-closed)
**Findings:** MD-02a/b/c/d

**Decision:** `build_market_data_matrix` raises `NotImplementedError(D061)` for any
run whose universe references a discount curve or a price curve, before any
interpolation kernel is called.

**Reason:** the repository does not define:

* what a discount-curve point level means for discounting (configured as
  `quote_type=RATE`, `unit_scale=PERCENT`, but with no compounding and no day
  count);
* how a LABEL tenor (`[ON]`, `[3M]`, `[1Y]`) maps to a date or year fraction.
  Every canonical discount curve is LABEL;
* how relative OFFSET tenors map onto the kernel's coordinate domain.

The builder previously read the levels as `ln DF`, passed relative offsets as
tenors and absolute int32 ordinals as targets, and hit a Numba `TypeError`.
Fixing only the dtype would have produced DF ≈ 73.7 or 1.044 against ≈ 0.99 and
collapsed all dates onto the last pillar. Selecting a convention here would
invent market semantics.

**Implications:** NODE forwards and NODE options are NOT production-reachable
through the market-data producer. Their valuation certification (D052, D057)
stays scoped to contract-correct raw state. Lifting D061 requires a separate
decision recording the quote convention, the day count, the tenor→date rule and
the zero-pillar policy, followed by a producer oracle through the real path.

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

**Decision:** for NODE forwards and options, ingest resolves the underlying's
relative name at the run as-of (`rf_resolution`). The trade's `delivery_date`
must fall in the resolved contract period. Supported:

* `OFFSET_M + CALENDAR_PERIOD` (label `YYYYMM`);
* `OFFSET_Y + CALENDAR_PERIOD` (label `YYYY`).

Every other roll method fails closed, because its period definition is not owned
here. Forward/option rows require `as_of_date` at ingest.

**Reason:** `BRENT_FWD[M+0]` (Feb-2027 at as-of 2027-01-04) accepted a 2029
delivery and consumed the wrong contract's level.

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
   required" economic state, it MUST be represented and validated as its own
   economic value, distinct from missing-data sentinels, in the pattern already
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
