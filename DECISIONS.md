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

## D038 — Continuous configuration strictness harmonization

### Methodology RESOLVED; implementation DEFERRED

Archaeology + decision memo only; no production edits, no test edits.

Defines the canonical continuous-numerical-input contract for the
`scenario_models` configuration seam.

Full methodology memo: `D038_CONTINUOUS_CONFIG_STRICTNESS.md`.

### Problem

Two incompatible input-typing policies currently coexist.

**Lax policy** in D032/D033/D035:

* `expected_block_size`
* `decay_factor`
* `max_vol_ratio`

These use ordinary Pydantic `float` fields with explicit bool rejection.
Pydantic therefore permits representation coercion such as `"0.9"`,
`Decimal`, or `Fraction` before the value reaches the stored Python `float`.

**Strict policy** in D034:

* Parametric `df`
* Parametric `nu`
* Parametric `prob_crisis`
* Parametric `crisis_vol_multipliers`

These use an explicit `isinstance(x, (int, float))` check. That criterion is
itself inconsistent across NumPy scalar types because its accepted set does
not explicitly model the intended numerical family.

`numbers.Real` is also unsuitable as the canonical predicate because it admits
types such as `bool` and `Fraction`, while `Decimal` does not participate in
the desired hierarchy.

The result is not an observed numerical-corruption defect. It is a
configuration-boundary typing inconsistency: equivalent numerical inputs can
follow different acceptance/coercion paths depending on the scenario model.

### Canonical policy

The scenario-model solution boundary adopts one **strict numerical scalar
contract** for continuous numerical values.

For each scalar continuous value:

```text
accept:
    int
    float
    np.integer
    np.floating

reject:
    bool
    str
    Decimal
    Fraction
    complex
    None
    NaN
    +inf
    -inf

store:
    Python float
```

`bool` is explicitly excluded even though Python considers it an integer
subclass.

Integer-valued numerical inputs are valid continuous inputs and are stored as
Python `float`. Examples include:

```text
expected_block_size = 1
max_vol_ratio       = 8
df                  = 3
nu                  = 1
prob_crisis         = 0 or 1
```

Domain constraints remain field-specific and unchanged.

### Scalar-versus-vector fields

The strict numerical contract applies to **numeric leaves**, not necessarily
to the container representation of a field.

Where a configuration field has legitimate scalar-or-per-factor semantics,
such as `crisis_vol_multipliers`, the surrounding container semantics remain
unchanged. A valid scalar representation must satisfy the scalar contract
above; a valid vector/sequence representation must preserve its existing
shape, cardinality, positivity, finiteness, and factor-alignment semantics
while applying the same strict numerical contract to each contained numeric
value.

The implementation must therefore inspect the existing declared type and
validator path for `crisis_vol_multipliers` before changing `_require_real`.
A scalar-only validator must not be applied to the field in a way that
invalidates an existing, intentional per-factor representation.

No new container representation is introduced by D038.

### Explicit representation policy

The solution boundary validates already-typed numerical representations. It
does not perform representation parsing.

The architectural sequence is:

```text
raw external representation
        |
        v
EDGE parsing / normalization
        |
        v
strict numerical type validation
        |
        v
canonical Python float(s)
        |
        v
field-level domain / shape constraints
        |
        v
stored solution configuration
```

Therefore:

* textual `"0.9"` is an EDGE representation that must be parsed before the
  solution boundary;
* JSON numeric values arrive already typed and follow the normal path;
* callers supplying `Decimal` or `Fraction` convert them at the appropriate
  EDGE boundary rather than relying on solution-tier coercion.

This preserves the D001/D006 EDGE ownership of representation parsing.

### Non-finite values

`NaN`, positive infinity, and negative infinity are rejected.

`allow_inf_nan=False` remains the canonical Pydantic field-level mechanism,
with the strict numerical validator operating alongside it.

No non-finite sentinel is introduced by this decision.

### Mechanism

The smallest correct implementation is confined to:

```text
scenario_models/schemas.py
```

The intended implementation is:

1. Correct the existing `_require_real` validation mechanism so that scalar
   values satisfy the explicit numerical-family predicate:

   ```python
   isinstance(value, bool) or not isinstance(
       value,
       (int, float, np.integer, np.floating),
   )
   ```

2. Ensure the accepted scalar is canonicalized to Python `float` at the
   solution boundary rather than relying on accidental downstream coercion.

3. Apply the same scalar policy consistently to all applicable scalar
   continuous fields.

4. For fields with scalar-or-vector semantics, such as
   `crisis_vol_multipliers`, preserve the existing container contract and apply
   the strict scalar rule to the contained numerical values rather than
   replacing the field with a scalar-only validator.

5. Remove the superseded `_reject_bool_real` validators where the unified
   contract makes them redundant.

6. Preserve all existing `float` declarations, `Field(...,
   allow_inf_nan=False)` settings, and field-specific domain bounds unless
   inspection proves that a change is required to implement the canonical
   contract correctly.

No changes are permitted in:

```text
api.py
generators
kernels
scenario-model unions
ParametricRiskData
```

unless repository archaeology proves that the current schema cannot enforce
the already-decided contract without crossing that boundary. Such a finding
must be reported before broadening scope.

D036 validation ownership remains unchanged:

```text
config boundary:
    representation typing
    numerical type admissibility
    finiteness
    field domain constraints
    existing structural constraints

generator/runtime:
    model-specific runtime invariants

kernel:
    numerical computation only
```

No mathematical logic moves into configuration validation.

### Backward compatibility

Repository archaeology performed for D038 established that no existing test
uses a `str`, `Decimal`, or `Fraction` as a successful continuous-config input.

Therefore:

* rejecting those representations removes zero currently-supported test
  cases that intentionally depend on coercion;
* accepting `np.int64` and `np.float32` where they represent valid numerical
  inputs is a safe widening of the intended numerical type domain;
* accepted values remain canonical Python `float` values, so numerical
  behavior is unchanged.

No D019 golden re-baseline is required.

No D029 RNG or batch-equivalence behavior is affected.

This is an input-contract change, not a numerical-method change.

### Required implementation verification

Before editing, verify the actual runtime representation and validator path of
every continuous field.

In particular, explicitly verify whether `crisis_vol_multipliers` currently
accepts:

```text
scalar
sequence
NumPy array
scalar-or-sequence
```

and where element-level validation currently occurs.

The implementation must preserve legitimate existing semantics while enforcing
the D038 numerical contract.

After implementation, tests must explicitly cover:

```text
Python int
Python float
np.integer
np.floating
bool
str
Decimal
Fraction
complex
None
NaN
+inf
-inf
```

for applicable scalar fields, plus the existing valid container cases for
`crisis_vol_multipliers`.

Tests must also verify that accepted values are canonical Python `float`
values and that existing domain constraints remain authoritative.

### Supersession

D038 supersedes the **continuous-input typing portions** of D032, D033, and
D035.

D038 also corrects the NumPy-scalar asymmetry in D034's continuous numerical
typing rule.

Those decisions are not rewritten. This entry records the supersession.

Scope is limited to the `scenario_models` configuration seam.

The credit-tier configuration seam, which currently has separate numerical
typing behavior, is explicitly outside D038 and requires a separate future
decision.

### Verdict

**METHODOLOGY RESOLVED. IMPLEMENTATION DEFERRED.**

The current architecture is not considered unsafe: existing validation
already rejects bool and non-finite values, and the observed lax behavior is
a representation/coercion inconsistency rather than demonstrated numerical
corruption.

D038 nevertheless fixes the canonical policy:

> Solution configuration accepts explicitly defined numerical types, does
> not perform representation coercion, canonicalizes accepted scalar values
> to Python `float`, and preserves existing container semantics for
> scalar-or-vector numerical fields.

Implementation is a later mechanical increment.

D036 scenario-model seam remains **FROZEN**.

D037 remains **methodology RESOLVED / implementation DEFERRED**.
