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
