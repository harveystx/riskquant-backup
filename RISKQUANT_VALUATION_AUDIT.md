# RISKQUANT — FRESH INDEPENDENT AUDIT

## `riskquant.valuation`

**Status:** AUDIT ONLY — no production code modified, no tests modified, no remediation.
**Date:** 2026-09-16
**Basis:** the current repository dump (`context_riskquant_all.txt`, 5.72 MB; `project_structure.txt`, both dated 2026-09-16). The attached `DECISIONS.md` and the prior seven-pass audit were read for orientation only; every conclusion below was re-established from current source, independent derivation, or executed probe.

---

## 1. Executive Verdict

### **NOT PROVEN**

Not *invalidated* — the pricing mathematics I was able to check independently is correct, and in several places it is more carefully constructed than its documentation admits. But "not proven" is the accurate classification, for three reasons that are independent of one another.

**First, there is a confirmed CRITICAL defect, and it is reachable in production.** `_black76_fv_vega` carries a docstring asserting a domain guard, naming the remediation ticket that added it, and describing the exact regression it prevents. The guard is not in the code. `vol = −1.0` (the documented `IV_SOLVER_FAILED_SENTINEL`) returns **35.207** where the contract requires `0.0`, and ATM `vol = 0` returns **NaN**. The live consumer — `_kernel_black76_iv_from_undiscounted_call_prices` in the implied-vol calibration path — states in a code comment that it relies on exactly this guard.

**Second, the valuation test suite does not run against the current API.** Six of the eight test modules under `tests/valuation/` import modules and classes that no longer exist in the repository: `riskquant.valuation.models.black_scholes`, `riskquant.valuation.pricers._deprecated.*`, `VanillaForwardNodeBlock`, `VanillaForwardCurveBlock`, `VanillaEuropeanOptionNodeBlock`, `VanillaEuropeanOptionCurveBlock`. `tests/valuation/test_engine.py` constructs a `SpotBlock` from `spot_price_uix`, `fx_uix` and `fx_invert` — none of which are fields of the current `SpotBlock` — and calls `run_valuation` without the `out=` argument the current signature requires. These are collection-time `ImportError`s and `TypeError`s, not assertion failures. **Every mathematical claim about forwards, options, cashflows and the engine is currently carried by dead code.** The only executable mathematical test on the package is a 3×4 spot smoke test with ascending indices and no FX.

**Third, five of the six registered block types are unreachable through the sanctioned production path.** Only `compile_spot_block` is registered in `portfolio/compilers/registry.py`; only `effective_spot_prices` is populated by `assemble_valuation_state`; and the three cashflow block types have no slot in `CompiledPortfolio` at all. The package's forward, option and cashflow mathematics is authored but not wired.

### What *is* proven

Under an independent scipy oracle the `cp_mult` Black-76 formulation is exact for both calls and puts; put-call parity holds to machine precision; the fixed, floating, principal and spot algebra reduce exactly to their economic definitions; and the forward kernel is the correct two-leg form for a physically-settled cross-currency forward, degenerating correctly to `Q(F−K)·DF·FX` in one currency. The kernels are genuinely dumb, genuinely allocation-free, and constitutionally compliant.

### The shape of the problem

This package's mathematics is better than its contracts, and its contracts are far better than its assurance. Four of the eighteen findings below are cases where a docstring asserts something the code does not do — including the CRITICAL one. That is a specific and correctable failure mode, and it is not an aesthetic complaint: under an architecture whose safety argument is *"kernels may assume their documented preconditions"*, a false precondition claim is load-bearing.

---

## 2. Mathematical Contract

Derived from the economic object first, then compared with the implementation. Currency symbols: `a` = asset/quote currency, `k` = strike currency, `s` = settlement currency.

### Spot

A spot position of signed quantity `Q` in an asset quoted at `S` in currency `a`, reported in `s`:

$$V = Q \cdot S_a \cdot FX_{a\to s}, \qquad \text{no discounting}$$

Implementation: `out[i,s] = qty[i] · effective_spot_prices[p, s]`, with `effective_spot_prices = SPOT_PRICE_CCY(0) · FX_SPOT_EFFECTIVE(CCY→SettleCCY)` per `state.py`. **Exact match.** Correctly undiscounted — a spot position has no time value to discount.

### Vanilla forward

Long `Q` units, delivery at `T`, strike `K`. Deriving from the two legs rather than from the kernel:

$$V = Q\Big[\underbrace{PV_s(\text{receive asset at }T)}_{F_a(0,T)\,DF_a(0,T)\,FX_{a\to s}} - \underbrace{PV_s(\text{pay }K\text{ at }T)}_{K\,DF_k(0,T)\,FX_{k\to s}}\Big]$$

Implementation: `out[i,s] = qty[i] · (effective_forward_prices[p,s] − strike[i] · effective_discount_factors[d,s])`, with `effective_forward_prices = FWD_PRICE_CCY(0,T) · DF_CCY(0,T) · FX_SPOT_EFFECTIVE(CCY→SettleCCY)` and `effective_discount_factors = DF_CCY(0,T) · FX_SPOT_EFFECTIVE(CCY→SettleCCY)`.

**Exact match**, and strictly more general than the single-currency identity `Q(F−K)·DF·FX`, which it recovers when `a = k`. The generality encodes a **physical, two-currency settlement** assumption that is stated nowhere — see VAL-15.

### Vanilla European option

Black-76 forward measure, undiscounted future value at expiry in strike currency, then discounted and converted:

$$FV = \phi\left[F\,\Phi(\phi d_1) - K\,\Phi(\phi d_2)\right], \quad \phi = \pm 1$$
$$d_1 = \frac{\ln(F/K) + \tfrac12\sigma^2 T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T$$
$$PV = Q \cdot FV \cdot DF_k(0, T_{settle}) \cdot FX_{k\to s}$$

Verification of the `cp_mult` packing, derived rather than assumed:

* `φ = +1`: `+1·(FΦ(d₁) − KΦ(d₂))` = call. ✔
* `φ = −1`: `−1·(FΦ(−d₁) − KΦ(−d₂))` = `KΦ(−d₂) − FΦ(−d₁)` = put. ✔

The block stores `sqrt_tte = √T` and `half_tte = T/2`, and the kernel forms `vol_sqrt_tte = σ√T` and `vol2_half_tte = σ²·T/2`. Substituting: `d₁ = (ln(F/K) + ½σ²T)/(σ√T)`. ✔ FV, not PV, exactly as the docstring states; the discount and FX are applied once by the caller.

### Cashflows

$$V_{\text{fixed}} = N\,\alpha\,r\,\varepsilon \cdot DF\cdot FX, \qquad V_{\text{float}} = N\,\alpha\,(f+\sigma_{sp})\,\varepsilon \cdot DF\cdot FX, \qquad V_{\text{principal}} = A\cdot DF\cdot FX$$

with `ε = pay_receive_mult ∈ {−1,+1}`. All three implementations match exactly, and each hoists its scenario-independent product outside the scenario loop. FX conversion occurs **per cashflow**, folded into `effective_discount_factors` — which is correct, because each cashflow may sit on a different curve and date.

`PrincipalCashflowBlock` carries **no** `pay_receive_mult`; direction must be signed into `amount`. Its two siblings carry an explicit multiplier. See VAL-16.

---

## 3. Architecture Map

Reconstructed from `project_structure.txt` and the current source, not from the prior audit.

```
src/riskquant/valuation/
    __init__.py                  (empty ✔)
    engine.py                    run_valuation(portfolio, valuation_state, out) -> None
    pricer_registry.py           BLOCK_PRICERS tuple, Pricer protocol, resolve_pricer
    result.py                    ValuationResult
    blocks/
        _base_block.py           Block (frozen, slots, kw_only) + __init_subclass__ guard
        block_type.py            BlockType IntEnum 0..5, sentinel -999, lookup tables
        spot_block.py            SpotBlock                       BLOCK_TYPE_ID = 0
        vanilla_forward_block.py VanillaForwardBlock             BLOCK_TYPE_ID = 1
        vanilla_european_option_block.py
                                 VanillaEuropeanOptionBlock      BLOCK_TYPE_ID = 2
        discounted_cashflow_block.py
                                 FixedCashflowBlock              BLOCK_TYPE_ID = 3
                                 FloatingCashflowBlock           BLOCK_TYPE_ID = 4
                                 PrincipalCashflowBlock          BLOCK_TYPE_ID = 5
    pricers/                     six thin unpack-and-call wrappers (no arithmetic)
    pricer_kernels/              six parallel_njit kernels
    pricing_models/
        math.py                  numba_norm_cdf, numba_norm_pdf, numba_norm_cdf_abrom_steg
        black76/{api,kernels,model}.py
```

**No `valuation/models/` package. No `pricers/_deprecated/` package.** Both are imported by the current test tree.

### Actual data flow

```
SimLabRunner / demo scripts / credit exposure runner
        │
        ├── build_uix_mapper(ingest_batches)              portfolio/engine.py
        ├── compile_portfolio_from_ingest_batch(...)      -> CompiledPortfolio
        │        COMPILERS[SPOT_ID] = compile_spot_block          ◀── only registration
        │        all other instrument ids: UNREGISTERED, fail loud
        ├── build_market_data_matrix_from_flat_levels(...)-> MarketDataMatrix
        ├── assemble_valuation_state(matrix, uix_mapper)  -> ValuationState
        │        populates: effective_spot_prices ONLY            ◀── one of six factors
        └── out = np.empty((total_trades, batch_size))
                 run_valuation(portfolio=..., valuation_state=..., out=out)
                     for block_idx in range(total_blocks):
                         BLOCK_PRICERS[block.BLOCK_TYPE_ID](block, state, out[start:end])
                             pricer unpacks dataclass -> flat arrays
                                 kernel: prange over rows, inner loop over scenarios
                 ValuationResult(trade_uids=portfolio.trade_uids, values=out)
```

`CompiledPortfolio` exposes exactly three block slots — `spot`, `vanilla_forward`, `vanilla_european_option` — and `BlockT` is the union of those three. The three cashflow types cannot be placed in a portfolio at all.

### Scope boundary

| | |
|---|---|
| **IN SCOPE** | everything under `src/riskquant/valuation/`, plus `tests/valuation/` |
| **BOUNDARY DEPENDENCIES** | `market_data.state.ValuationState` (input contract), `portfolio.compiled.CompiledPortfolio` (input contract), `riskquant.jit` (Numba policy), `market_data.calibrate...black76` (consumer of `_black76_fv_vega`) |
| **OUT OF SCOPE** | the market-data fold's internal construction, the shock/scenario engine, the sensitivity engine, `scenario_metrics` |

---

## 4. Findings

| ID | Sev | Cat | Status | Location | Observed | Expected | Evidence | Owner |
|---|---|---|---|---|---|---|---|---|
| **VAL-01** | **CRITICAL** | A+E | **CONFIRMED** | `pricing_models/black76/model.py::_black76_fv_vega` | No guard; `log`/divide run unconditionally. `vol=−1.0` → **35.2065**; ATM `vol=0` → **NaN** | `0.0` iff `vol ≤ 0 or sqrt_tte ≤ 0`, per the function's own docstring | Probe A-4; source contains no `if`. Consumer comment at `black76_implied_vol_batch.py` relies on it | KERNEL / model primitive |
| **VAL-02** | **CRITICAL** | D | **CONFIRMED** | `tests/valuation/` (6 of 8 modules) | Collection-time `ImportError` / `TypeError` against deleted modules and classes | Tests import and exercise the current API | `project_structure.txt` shows no `valuation/models/`, no `pricers/_deprecated/`; grep finds no `VanillaForwardNodeBlock`, `VanillaEuropeanOptionCurveBlock`, `fv_black_scholes_vanilla_3d` anywhere in `src/` | TEST / ASSURANCE |
| **VAL-03** | HIGH | C | **CONFIRMED** | `engine.py::run_valuation` | `out[start:end]` clamps silently; kernels size work from `out.shape` | Assert `out.shape == (total_trades, batch_size)` at entry | Probe A-8: `np.zeros((3,4))[2:6].shape == (1,4)`, no exception | CORE boundary (engine) |
| **VAL-04** | HIGH | C | **CONFIRMED** | `market_data/state.py::ValuationState` + `engine.py` | Five of seven factor fields default to `np.empty((0,0))` and are never populated; dispatch accepts them | A block's `_uix` cannot address an unpopulated universe | Probe A-6: `(0,0)` is C- **and** F-contiguous, passes `_F64_2D_C`, kernel returned `2.28e-316` heap | UPSTREAM (fold) + CORE |
| **VAL-05** | HIGH | B | **CONFIRMED** | `result.py::ValuationResult.get_trade_values` | `np.searchsorted` on `trade_uids`, which `CompiledPortfolio` builds in **block-concatenation order**, not sorted | Return the values for the requested uid | Probe A-10: uids `[9007,41,120003,5]` → `KeyError` for three of four **present** uids | CORE (result container) |
| **VAL-06** | HIGH | C | **CONFIRMED** | `blocks/vanilla_forward_block.py` field comment | "the underlying forward price in the FORWARD PRICE CURRENCY" | It is `F·DF·FX` — a **settlement-currency PV** | `state.py` definition vs block comment; kernel is correct under the PV reading | EDGE/contract (block doc) |
| **VAL-07** | MEDIUM | C | **CONFIRMED** | `blocks/vanilla_european_option_block.py` docstring | Documents `fx_*_to_settle_effective_uix` fields and an `INVALID_UIX (<0)` branch "the pricer kernel performs" | Neither exists; no kernel branches | Source; probe A-5: `uix = −1` silently returns the **last** row | EDGE/contract (block doc) |
| **VAL-08** | MEDIUM | C | **CONFIRMED** | `blocks/vanilla_european_option_block.py` | `@dataclass(frozen=True, slots=True)` — no `kw_only=True`, unlike every sibling | Keyword-only construction | Probe A-9: positional construction **accepted**; six adjacent same-dtype `float64` arrays | EDGE/contract (block) |
| **VAL-09** | MEDIUM | C | **CONFIRMED** | all `blocks/*.py`, `result.py` | `eq=False` absent; `frozen=True` generates `__hash__` | Defined `==`/`hash` semantics, or none | Probe A-9: `b1 == b2` → `ValueError`; `hash(b)` → `TypeError`; `b in [..]` → `ValueError` | EDGE/contract |
| **VAL-10** | MEDIUM | A | **NUM OBS** | `pricing_models/math.py::numba_norm_cdf_abrom_steg` | Abramowitz–Stegun 26.2.17; **no docstring, no stated accuracy envelope** | An accuracy contract, since this is the CDF the price path uses | Probe A-3: rel err `5.1e-5` at `x=−3`, `4.7e-4` at `x=−4`, `2.0e-2` at `x=−10`. Probe A-1: **−4.3e-4 on a deep-OTM put at F=100, K=60, σ=25 %, T=0.5** | KERNEL / model primitive |
| **VAL-11** | MEDIUM | B | **LATENT** | option block + `state.py` | `forward_prices_strike_ccy` and `option_log_forward_prices_strike_ccy` are two independent factor universes with two independent UIX; the invariant `ln(A[i,s]) = B[j,s]` is enforced nowhere | One universe, or a compile-time pairing assertion | Source; no option compiler exists to establish it | UPSTREAM (unwritten compiler) |
| **VAL-12** | MEDIUM | C | **CONFIRMED** | all six `pricer_kernels/*.py` | Read-only inputs pinned with writable aliases (`_F64_2D_C`, `_I32_1D_C`, …) | `_RO` aliases, which `riskquant.jit` exports and `scenario_metrics` kernels use | Source comparison | KERNEL (signature) |
| **VAL-13** | MEDIUM | D | **CONFIRMED** | `portfolio/compilers/registry.py`, `portfolio/compiled.py` | 5 of 6 block types have no compiler; 3 also have no `CompiledPortfolio` slot | Either reachability or an explicit buildout record | Registry `# --- PENDING ---`; `CompiledPortfolio` has three slots | UPSTREAM (buildout) |
| **VAL-14** | LOW | C | **CONFIRMED** | `pricer_kernels/spot.py`, `pricers/spot.py` | Docstrings describe `fx_spots_effective`, `fx_effective_uix < 0` branch, and an empty "Numba layout-pinned aliases" banner | Describe the single-gather form actually implemented | Source | EDGE/contract (doc) |
| **VAL-15** | LOW | C | **CONFIRMED** | `blocks/vanilla_forward_block.py` | Two legs may carry different `DF`/`FX`; no settlement convention stated | State whether the block models physical two-currency or cash settlement, and whether the compiler may pair different curves | §2 derivation | EDGE/contract (doc) |
| **VAL-16** | LOW | C | **CONFIRMED** | option block; `PrincipalCashflowBlock` | `sqrt_tte`/`half_tte` redundantly encode one `T` with no invariant; principal block's sign convention diverges from its siblings undocumented | An asserted invariant; a stated convention | Source | EDGE/contract (doc) |
| **VAL-17** | LOW | C | **CONFIRMED** | `pricer_registry.py::resolve_pricer` | Dead code — `run_valuation` indexes `BLOCK_PRICERS` directly; return type is unparameterised `Pricer` | One dispatch surface | Source | CORE |
| **VAL-18** | LOW | A | **ACCEPTED-RISK** | `math.py` | A&S admits `FV < 0` at denormal magnitudes in the far tail | `FV ≥ 0` | Prior measurement `−2.7e−322`; immaterial | KERNEL |

---

## 5. Mathematical Oracle Evidence

Oracle: `scipy.stats.norm` with an independently written Black-76 closed form, structurally distinct from the production code (which uses the A&S polynomial and a `cp_mult` packing). No production symbol was used to derive an expected value.

### Black-76, call and put

| F | K | σ | T | cp | implementation | oracle | rel err |
|---|---|---|---|---|---|---|---|
| 100 | 100 | 0.20 | 1 | +1 | 7.96557924 | 7.96556746 | 1.48e-06 |
| 100 | 100 | 0.20 | 1 | −1 | 7.96557924 | 7.96556746 | 1.48e-06 |
| 100 | 60 | 0.25 | 0.5 | +1 | 40.0076566 | 40.0076598 | −8.19e-08 |
| **100** | **60** | **0.25** | **0.5** | **−1** | **0.0076565661** | **0.00765984107** | **−4.28e-04** |
| 100 | 180 | 0.30 | 2 | +1 | 2.12117958 | 2.12117356 | 2.84e-06 |
| 100 | 180 | 0.30 | 2 | −1 | 82.1211796 | 82.1211736 | 7.34e-08 |
| 100 | 100 | 0.20 | 1/365 | +1 | 0.417632589 | 0.41762996 | 6.30e-06 |
| 100 | 100 | 0.01 | 1 | +1 | 0.398943141 | 0.398940618 | 6.32e-06 |
| 100 | 100 | 2.50 | 1 | +1 | 78.8700322 | 78.8700453 | −1.66e-07 |
| 100 | 100 | 0.20 | 30 | +1 | 41.6117641 | 41.6117579 | 1.49e-07 |
| 10⁷ | 10⁷ | 0.20 | 1 | +1 | 796557.924 | 796556.746 | 1.48e-06 |
| 10⁻⁴ | 10⁻⁴ | 0.20 | 1 | +1 | 7.96557924e-06 | 7.96556746e-06 | 1.48e-06 |
| 100 | 400 | 0.20 | 1 | +1 | 1.13059182e-11 | 1.15067259e-11 | **−1.75e-02** |
| 100 | 800 | 0.20 | 1 | +1 | 6.54655019e-25 | 6.78888499e-25 | **−3.57e-02** |
| 400 | 100 | 0.20 | 1 | −1 | 1.13059182e-11 | 1.15067259e-11 | −1.75e-02 |

**Classification.** The `cp_mult` formulation is **mathematically exact** — every deviation above is attributable to the A&S CDF, not to the option formula. Call and put at identical parameters give identical prices where parity requires it. Scale invariance is exact: `F = K = 10⁷` and `F = K = 10⁻⁴` return the *same* relative error as `F = K = 100`, so the implementation has no absolute-scale sensitivity.

The row that matters most is the fourth: a **deep-OTM put at entirely ordinary parameters** (50 % out of the money, 25 % vol, six months) already carries `−4.3 × 10⁻⁴` relative error. This is not a far-tail curiosity. Any option whose `d`-value sits beyond roughly three standard deviations inherits ≥10⁻⁴ relative price error.

**Classification: numerical approximation, not a mathematical defect** — but an approximation whose envelope is **undocumented**, in a module where the exact `erf`-based `numba_norm_cdf` sits unused three lines above it. See VAL-10.

### Put-call parity under the implemented CDF

| F | K | σ | T | `C − P` | `F − K` | error |
|---|---|---|---|---|---|---|
| 100 | 100 | 0.2 | 1 | 0.0 | 0 | 0.000e+00 |
| 100 | 150 | 0.5 | 0.1 | −50.00000000000001 | −50 | −7.11e-15 |
| 80 | 120 | 0.9 | 5 | −40.0 | −40 | 0.000e+00 |
| 10⁶ | 9×10⁵ | 0.3 | 2 | 100000.00000000006 | 100000 | 5.82e-11 (5.8e-16 relative) |

Parity is exact by construction: the A&S branch satisfies `Φ(x) + Φ(−x) ≡ 1` identically, so the approximation error cancels in `C − P`. **Proven correct.**

### Vega

| input | contract | measured |
|---|---|---|
| `vol = −1.0` (`IV_SOLVER_FAILED_SENTINEL`), F=K=100, T=1 | `0.0` | **35.2065326764** |
| `vol = 0.0`, ATM (`log_fk = 0`) | `0.0` | **NaN** (`0/0` under `error_model="numpy"`) |
| `vol = 0.0`, OTM (`log_fk ≠ 0`) | `0.0` | `0.0` — accidental: `d₁ = −inf`, `φ(−inf) = 0` |
| `vol = 0.20` | positive | `39.695` ✔ |

The formula itself, `∂C/∂σ = F·φ(d₁)·√T`, is correct and correctly call/put-symmetric (via `Fφ(d₁) = Kφ(d₂)`). **The defect is exclusively the missing domain guard, which the function's own docstring says is present.**

---

## 6. Identity / Index-Space Findings

### Semantic integer universes

| Integer | Semantic universe | Producer | Consumer | Representation | Boundary invariant |
|---|---|---|---|---|---|
| `trade_uids` | global forged trade UID | ingest / `UixMapper` | `ValuationResult` | `int64` | **none** — not sorted, not unique-checked |
| block row index `i` | position within one block | `Block.size` | kernel `prange` | implicit | `i < block.size` by construction |
| portfolio row index `m` | `out` row | `CompiledPortfolio._block_offsets` | `out[start:end]` | `int64` offsets | **none** at `run_valuation` (VAL-03) |
| `BLOCK_TYPE_ID` | dense dispatch tag 0–5 | `BlockType` IntEnum | `BLOCK_PRICERS[·]` | `int` ClassVar | `__init_subclass__` + import-time registry guard ✔ |
| `effective_spot_price_uix` | row of `effective_spot_prices` | `UixMapper.map_spot_to_effective_uix` | spot kernel | `int32` | mapper raises on drift ✔; **kernel gather unchecked** |
| `effective_discount_factor_uix` | row of `effective_discount_factors` | *no producer exists* | 5 kernels | `int32` | **none** |
| `effective_forward_price_uix` | row of `effective_forward_prices` | *no producer exists* | forward kernel | `int32` | **none** |
| `forward_price_strike_ccy_uix` | row of `forward_prices_strike_ccy` | *no producer exists* | option kernel | `int32` | **none** |
| `option_log_forward_price_strike_ccy_uix` | row of a **separate** log universe | *no producer exists* | option kernel | `int32` | **none**; coupling to the above unenforced (VAL-11) |
| `vol_uix` | row of `vols` | *no producer exists* | option kernel | `int32` | **none** |
| `forward_rate_uix` | row of `forward_rates` | *no producer exists* | floating kernel | `int32` | **none** |

**One of ten UIX universes has a producer, and it is the only one with a validator.**

### Unsafe crossings, measured

Kernels run under `boundscheck=False`. Against a four-row market matrix:

```
uix = -1                     -> [9.0, 10.0, 11.0]              silently the LAST row
uix = INVALID_UIX (-999)     -> [1e-323, 0.0, nan]             out-of-bounds read, no error
uix = 4  (first invalid +ve) -> [0.0, 2.06e-321, 7.26e-320]    out-of-bounds read, no error
uix = 9999                   -> [2.27e-318, 2.99e-316, 1e-323] out-of-bounds read, no error
```

The `−1` case is the dangerous one: it is **in range** under NumPy/Numba wrap semantics, returns a well-formed finite row, and is exactly the value the option block's docstring says the kernel treats as a sentinel (VAL-07).

Under "kernels do dumb math" this is **not a kernel defect**. It is an unowned contract: `UixMapper.map_spot_to_effective_uix` genuinely validates the one UIX that has a producer, and nothing validates the other nine because nothing produces them yet. The correct owner is the block compilers, at UIX-stamping time.

### Trade identity

`CompiledPortfolio` builds `trade_uids` by concatenating each block's `trade_uids` in slot order. `ValuationResult.get_trade_values` uses `np.searchsorted`, whose precondition is a sorted array. With adversarial identifiers `[9007, 41, 120003, 5]`:

```
get_trade_values(  9007) -> KeyError      <- uid IS present
get_trade_values(    41) -> KeyError      <- uid IS present
get_trade_values(120003) -> [12.0003]     <- correct by coincidence
get_trade_values(     5) -> KeyError      <- uid IS present
```

Three of four present uids raise. This is **loud but wrong**. Worse, with duplicate uids — which the block model explicitly permits, since `trade_uids[i]` identifies "the originating trade that produced row `i`" and one IRS produces many cashflow rows — `searchsorted` can land on a matching value and return **one leg's PV as the trade's PV**, silently. `ValuationResult` has no row→trade aggregation and `num_trades` returns a row count.

**`PV[m] ≡ trade_uids[m]` holds inside the matrix.** The identity that fails is the *lookup by uid*, and the row/trade cardinality claim that `num_trades` makes.

---

## 7. Currency / FX Findings

### Direct vs inverse

The current valuation tier performs **no FX arithmetic at all**. `ValuationState` has `fx_spots_effective` **commented out**; the spot, forward, option and cashflow kernels each perform one gather and one multiply. Inversion is resolved upstream: `MarketDataMatrix.fx_spots_effective` rows are "already the resolved transform (direct or pre-inverted)", and the fold pre-multiplies them into the effective factors.

So `FX_{A→B} = 1/FX_{B→A}` is represented **once**, upstream, and the valuation tier multiplies only. This is the correct placement: it removes a branch from the hot path and makes double-inversion structurally impossible below the fold.

### The "effective discount factor" abstraction

Derived independently rather than accepted. For a cashflow `C` in currency `c` at `T`, reported in `s`:

$$PV_s = C \cdot DF_c(0,T) \cdot FX^{spot}_{c\to s}(0)$$

Both operands are scenario-dependent and both are per-`(currency, date)`. Pre-multiplying them into a single row is therefore **mathematically valid**, not merely a memory optimisation: the product is exactly the quantity the pricing equation requires, and no pricing path ever needs `DF` and `FX` separately. The row key is `(curve_currency, date)` and the value is a pure number.

One subtlety worth stating explicitly, since the abstraction hides it: converting an *asset-currency PV at t=0* into settlement currency correctly uses the **spot** FX, not a forward FX — and `state.py` uses `FX_SPOT_EFFECTIVE`. That is right. By contrast `forward_prices_strike_ccy` uses the **forward** FX ratio `FX(ccy→settle)/FX(strike→settle)` with the two discount curves, which is the correct no-arbitrage conversion for a *forward price*. The two conversions are different and both are correct for their respective objects. **No defect.**

### The valuation/sensitivity inverse-FX discrepancy

The prior audit recorded that valuation supports inverse FX while the sensitivity path raises `NotImplementedError` on it. Re-checked against current source:

* **Valuation** never sees an invert flag. Inversion is consumed by `_build_fx_spots_effective` upstream; the valuation kernels multiply a resolved rate.
* **Sensitivity** consumes `MarketDataMatrix.fx_spots_effective` *and* `UixMapper.fx_effective_pairs`, and refuses `invert = 1` because `∂PV/∂FX` for a payoff of the form `qty·S/F` is not the bilinear partial it emits.

**This is not an inconsistency in valuation.** The two layers differ because they need different things from the same data: valuation needs the *rate*, differentiation needs the *functional form*. The convention is owned upstream, at `_build_fx_spots_effective`. What is missing is any statement of that asymmetry at either boundary — a portfolio that values cleanly cannot be risked, with nothing at the valuation boundary warning of it. **Classification: boundary contract defect, owned upstream; not a valuation defect.**

---

## 8. Numerical Findings

### Stability, measured

| regime | behaviour |
|---|---|
| `T → 0` (T = 1/365) | stable; rel err 6.3e-06 |
| `σ → 0` (σ = 0.01) | stable; rel err 6.3e-06 |
| `σ√T → 0` exactly | `d₁ = ±inf` (OTM) → intrinsic ✔; `d₁ = 0/0 = NaN` (ATM). **Unreachable in the option kernel** if the market-data `TTE_FLOOR = 1e-9` holds and `σ > 0`; reachable in `_black76_fv_vega` (VAL-01) |
| `σ → ∞` (σ = 2.5) | `FV → F`; rel err −1.7e-07 |
| `T` large (30 y) | rel err 1.5e-07 |
| `F/K → 1` | no cancellation; the price is `≈0.4·σ√T·F` and both CDF terms are near 0.5 |
| deep ITM (`F/K = 100`) | returns intrinsic; A&S saturates to exactly `1.0`/`0.0`, so no cancellation |
| `F` from 10⁻⁴ to 10⁷ | scale-invariant; identical relative error |
| `Φ(±inf)`, `Φ(±10³⁰⁰)` | saturate to `1.0`/`0.0`; no overflow |
| `Φ(NaN)` | propagates NaN; does not mask |

The one genuine numerical weakness is the A&S tail (VAL-10), and it is a property of the CDF, not of the option formula.

### Degenerate shapes against the pinned `_F64_2D_C` contract

| shape | C | F | dispatch |
|---|---|---|---|
| `(1, 4)` | ✔ | ✔ | **OK** |
| `(3, 1)` | ✔ | ✔ | **OK** |
| `(1, 1)` | ✔ | ✔ | **OK** |
| F-ordered `(4, 3)` | ✘ | ✔ | **rejected loudly** (`TypeError`) |
| `(0, 0)` | ✔ | ✔ | **OK — and this is VAL-04** |

Numba resolves the ambiguous single-row/single-column cases to `C`, which is what every valuation signature declares, so there is **no C/F ambiguity trap** in this package. Layout violations fail loud. **Size violations do not** — the empty case and the `out` slice both pass.

### Dtype / Numba policy

Compliant, with one exception. All six kernels use `parallel_njit` from `riskquant.jit` with `_NB_VOID` signatures and canonical `_F64_2D_C` / `_F64_1D_C` / `_I32_1D_C` / `_I8_1D_C` aliases; the nine model primitives use `inline_njit`. No raw `nb.njit`, no local alias redefinition, no per-kernel `cache=`/`fastmath=`/`parallel=` override. `import numba as nb` appears only for `nb.prange`. Zero allocation in every kernel. No Python objects, no strings, no Pandas, no Polars anywhere in the package. `valuation/__init__.py` and all sub-package `__init__.py` files are empty.

The exception is **VAL-12**: read-only inputs are pinned with writable aliases. `riskquant.jit` exports `_F64_2D_C_RO` and friends, and `scenario_metrics/_kernels/_tail_metrics.py` uses them — so the policy exists and is applied elsewhere.

Minor inconsistencies, recorded but not raised as findings: `# type: ignore[import-untyped]` appears on the `numba` import in `spot.py` and `vanilla_forward.py` and not in the other four; `_kernel_price_vanilla_european_option` lacks a docstring and any precondition statement, while `_kernel_black76_fv_batch` documents its preconditions fully.

---

## 9. Test / Assurance Findings

### The coverage map

| Test module | Imports | Status against current API |
|---|---|---|
| `blocks/test_base_block_slots_regression.py` | `Block` internals | **probably live** |
| `pricer_kernels/test_spot_kernel_run.py` | `_kernel_price_spot` (4 args) | **LIVE** — matches the current signature |
| `models/test_black_scholes.py` | `riskquant.valuation.models.black_scholes` | **DEAD** — package does not exist |
| `pricers/test_spot.py` | `MarketDataMatrix`, `price_spot_block`, `NO_FX_RF_REQUIRED` | **DEAD** — `price_spot_block` takes `ValuationState`; `SpotBlock` has no `fx_uix`/`fx_invert` |
| `pricers/test_forward_node.py` | `VanillaForwardNodeBlock`, `pricers._deprecated.vanilla_forward_node_v1` | **DEAD** — neither exists |
| `pricers/test_forward_curve.py` | `VanillaForwardCurveBlock`, `pricers._deprecated.vanilla_forward_curve_v1` | **DEAD** — neither exists |
| `pricers/test_vanilla_european_option_node.py` | `VanillaEuropeanOptionNodeBlock`, `models.black_scholes`, `pricers._deprecated...` | **DEAD** |
| `pricers/test_vanilla_european_option_curve.py` | `VanillaEuropeanOptionCurveBlock`, `pricers._deprecated...` | **DEAD** |
| `test_engine.py` | `VanillaForwardNodeBlock`; `SpotBlock(spot_price_uix=, fx_uix=, fx_invert=)`; `run_valuation` without `out=` | **DEAD** |

Verified structurally: `project_structure.txt` shows `src/riskquant/valuation/` contains `blocks/`, `pricer_kernels/`, `pricers/`, `pricing_models/`, `engine.py`, `pricer_registry.py`, `result.py` — and **no `models/`, no `pricers/_deprecated/`**. A repository-wide grep finds no definition of `VanillaForwardNodeBlock`, `VanillaForwardCurveBlock`, `VanillaEuropeanOptionNodeBlock`, `VanillaEuropeanOptionCurveBlock` or `fv_black_scholes_vanilla_3d`.

### What the one live mathematical test establishes

`test_spot_kernel_compiles_and_runs`: three trades, two prices, four scenarios, `uix = [0, 1, 0]`, no FX, and three `assert_allclose` calls against `qty × price`. It proves the spot kernel compiles under the canonical aliases and multiplies correctly. It cannot detect a permuted index, a wrong currency, a wrong discount factor, or anything about forwards, options or cashflows.

### Non-identifying fixtures

Even setting aside the dead modules, the surviving fixture is non-identifying in the specific ways §17 asks about:

* indices are ascending (`[0, 1, 0]`), so a positional/identifier confusion is invisible;
* FX is absent entirely;
* there is one block type;
* all three quantities differ but all map through the same trivial `qty × price`, so a row permutation of `prices` would be caught only because `uix` is not constant — a weak accident, not a designed property.

### Assurance verdict

There is **no independent end-to-end valuation oracle** in the repository, and — given VAL-02 — no executable test of the forward, option or cashflow mathematics at all. The prior seven-pass audit's favourable mathematical conclusions rest on derivation and on probes run inside that audit, **not on any repository test**, and this fresh pass reaches the same conclusions by the same means. That is legitimate evidence for an audit and is *not* a substitute for assurance.

---

## 10. Production-Path Assessment

**Can current callers safely rely on valuation? For spot only, and conditionally.**

The sanctioned production path is `compile_portfolio_from_ingest_batch → build_market_data_matrix → assemble_valuation_state → run_valuation`. Along that path:

* **Spot** is fully wired: compiler registered, fold populates `effective_spot_prices`, `UixMapper` validates the UIX and raises on mapper/batch drift, kernel is correct. Conditional on the upstream fold delivering a correct `effective_spot_prices` — outside this audit's scope.
* **Forward, option, and the three cashflow types** are unreachable. `COMPILERS` has one registration; the registry comment states plainly that the others "stay unregistered and the engine fails loud on encounter". `assemble_valuation_state` passes exactly two keyword arguments. So the engine cannot currently be handed a block whose factor universe is empty — **VAL-04 is latent, not live**, and it is latent by ordering coincidence rather than by invariant.
* `run_valuation` has no output contract (VAL-03). A caller that mis-sizes `out` gets partial pricing and `np.empty` garbage in the remainder, silently.

### Callers and bypasses

| Caller | Kind | Assessment |
|---|---|---|
| `SimLabRunner.run` | full-revaluation runner | intended production path |
| `demo_e2e_canonical_base_valuation.py` and siblings | demo | intended, documented |
| credit-exposure runner | production | uses `run_valuation` |
| `pricing_models.black76.api.price_black76_fv_batch` | validated batch entry point | **intentional low-level primitive**, not a bypass — it validates dtype, positivity and finiteness before calling its kernel, and D001 permits lower-level building blocks |
| `market_data.calibrate...black76` → `_black76_fv_vega` | production | **legitimate reuse of a model primitive; this is the VAL-01 exposure path** |
| `resolve_pricer` | dead | VAL-17 |

No production caller manipulates `BLOCK_PRICERS` directly or calls a `pricer_kernels.*` symbol outside its pricer. The single public production entry point is `run_valuation`, and the model-primitive reuse is deliberate and correctly placed.

---

## 11. Remediation Gate

The minimum evidence that must exist before `riskquant.valuation` could be certified. **Not implemented here; listed as gates only.**

1. **VAL-01 closed** — the vega domain guard present, with a regression test that asserts `0.0` for `vol = IV_SOLVER_FAILED_SENTINEL`, `vol = 0` ATM, `vol = 0` OTM, and `sqrt_tte = 0`. Until then no calibrated vega from this estate can be trusted as a quality weight.
2. **VAL-02 closed** — `tests/valuation/` importing and exercising the current API. This is the gate that subsumes most of the others: nothing below can be *demonstrated* until the suite runs.
3. **An independent end-to-end oracle** with: non-ascending trade uids; sparse, non-ascending, high factor ids; at least two currencies; a non-unit FX and a non-unit discount factor; at least three block types; mixed long/short quantities; and enough asymmetry that swapping any two factor rows or any two block rows changes the expected PV. Expected values derived without any production symbol.
4. **VAL-03 closed** — `run_valuation` asserts `out.shape == (portfolio.total_trades, valuation_state.batch_size)`.
5. **VAL-05 closed** — a stated row/trade cardinality contract for `ValuationResult`, and a lookup that is sound for the uid ordering the portfolio actually produces.
6. **VAL-06, VAL-07, VAL-14, VAL-15 closed** — every block field comment stating the true units, currency and date of the factor it indexes. These gate the *unwritten compilers*, so they are cheap now and expensive later.
7. **VAL-10 documented** — a stated accuracy envelope for `numba_norm_cdf_abrom_steg` covering both the near-ATM and the ≥3σ regimes, and a recorded decision on why the exact `numba_norm_cdf` beside it is not used on the price path.
8. **VAL-04 owned** — a named layer that rejects a block whose factor universe is unpopulated, before the first non-spot compiler lands.

---

## 12. Final Theorem

The evidence supports a **conditional** theorem, and the hypotheses are doing real work.

> **Theorem (pricing paths).** Let `B` be a compiled block of type SPOT, VANILLA_FORWARD, VANILLA_EUROPEAN_OPTION, FIXED_CASHFLOW, FLOATING_CASHFLOW or PRINCIPAL_CASHFLOW, and let `V` be a `ValuationState` such that:
>
> **(H1)** every factor matrix `B` indexes is populated, C-contiguous, `float64`, and has `shape[1] ≥ out.shape[1]`;
> **(H2)** every `_uix` field of `B` satisfies `0 ≤ uix < M` for its target matrix's row count `M`;
> **(H3)** each factor row carries the economic quantity declared in `market_data/state.py` — in particular `effective_discount_factors = DF_ccy(0,T)·FX^{spot}_{ccy→settle}` and `effective_forward_prices = F_ccy(0,T)·DF_ccy(0,T)·FX^{spot}_{ccy→settle}`;
> **(H4)** for option rows, `σ > 0`, `√T > 0`, `F > 0`, `K > 0`, and `option_log_forward_prices_strike_ccy[j,s] = ln(forward_prices_strike_ccy[i,s])` for the paired UIX;
> **(H5)** `out` is C-contiguous `float64` with `out.shape[0] ≥ B.size`;
>
> then the corresponding pricer computes, for every row `i` and scenario `s`, the economic value stated in §2 **exactly**, up to:
>
> * IEEE-754 double rounding, and
> * for the option path only, the Abramowitz–Stegun 26.2.17 CDF approximation, whose measured effect is ≤1.5 × 10⁻⁶ relative near the money, ~5 × 10⁻⁵ at `|d| = 3`, and up to 3.6 × 10⁻² at `|d| ≳ 8`.
>
> Put-call parity `C − P = F − K` holds to machine precision independently of that approximation.

### What the hypotheses cost

**H1 is not established** — five of seven factor matrices have no producer, and the container's defaults pass dispatch (VAL-04).
**H2 is not established** — one of ten UIX universes has a validator (§6).
**H3 is asserted by `state.py` and contradicted by two block field comments** (VAL-06, VAL-07).
**H4's log-coupling clause is enforced nowhere** (VAL-11).
**H5 is not checked** (VAL-03).

**No theorem is available for `_black76_fv_vega`.** Its stated contract — `vega == 0.0 iff vol ≤ 0 or sqrt_tte ≤ 0` — is **disproven** by measurement.

---

## Final Question

> **Can `riskquant.valuation` be treated as a trustworthy mathematical valuation primitive on the basis of the current repository evidence, and exactly what hypotheses must hold for that statement to be true?**

**No — not on the basis of the current repository evidence.** The qualifier is the whole answer: the mathematics is very likely right, and I have derived and measured it independently, but the *repository* does not currently demonstrate it.

**Proven.** The Black-76 `cp_mult` formulation is exact for calls and puts against an independent scipy oracle across 15 parameter regimes spanning 11 decades of scale. Put-call parity is exact. The spot, fixed, floating and principal cashflow algebra matches its economic definition exactly. The forward kernel is the correct two-leg form. FX is resolved once, upstream, and the valuation tier multiplies only — a correct placement that makes double-inversion structurally impossible. Layout contracts fail loud; degenerate `(1,S)`, `(N,1)` and `(1,1)` shapes dispatch correctly. Dispatch is dense, guarded at class creation and at import, and free of silent fall-through. Constitutional compliance is clean but for `_RO` aliases.

**Disproven.** `_black76_fv_vega`'s documented domain contract. `ValuationResult.get_trade_values`'s correctness on the uid ordering the portfolio actually produces. The option block's documented `INVALID_UIX` sentinel branch. The forward block's documented field units. The claim, implicit in the existence of `tests/valuation/`, that the package's mathematics is tested.

**Merely assumed.** Every one of H1–H5. Nine of ten UIX universes have neither producer nor validator; five of seven `ValuationState` factor matrices have no producer at all; `run_valuation` checks no shape. These are assumed by the kernel tier under "kernels do dumb math" — which is the right architecture — but the layer that is supposed to establish them mostly does not exist yet.

**Accepted approximation.** The A&S CDF, at ≤1.5 × 10⁻⁶ near the money and ~5 × 10⁻⁵ at three standard deviations. Defensible for revaluation; its envelope is undocumented and the exact alternative sits unused in the same module.

**Assurance gap.** Six of eight valuation test modules do not import. There is no independent end-to-end oracle. No test in the repository uses non-ascending identifiers, more than one currency, a non-unit FX, or a non-unit discount factor on this package.

**Upstream / downstream.** The empty-factor-matrix hazard (VAL-04), the unenforced UIX validity (§6), the forward/log-forward coupling (VAL-11) and the unreachability of five block types (VAL-13) all belong to the market-data fold and the unwritten block compilers, not to `riskquant.valuation`. The valuation/sensitivity inverse-FX asymmetry belongs to `_build_fx_spots_effective`. None of these should be repaired inside a numerical kernel.

**The one defect that is unambiguously valuation's own, and unambiguously live, is VAL-01.**
