# RISKQUANT — FIFTH AUDIT PASS

## Shocked-Level Construction and the `flat_levels` Round Trip

**Status:** AUDIT ONLY — no code modified, no patches proposed, no remediation plan.
**Seam:** `simlab.shocks.apply` · `simlab.builder.shock_universe` · `simlab.shock_plan.plan` · the five scenario generators · `market_data.assemble.extract_levels_from_snapshot` · `build_universe_mapping` · `simlab.runner`.
**Date:** 2026-09-15

---

## Evidence basis

All source citations are from the re-mounted project dump (`context_riskquant_all.txt`, 5.53 MB, this session), which now carries the full `simlab` tree. `DECISIONS.md` has grown to D001–D033 and was read for the seam-related decisions.

**Probe method.** Every production symbol on this seam was re-implemented verbatim in NumPy from the read source — `unique_first_seen`, `_pack_keys`, `build_shock_universe`, `build_shock_compute_plan`, `_kernel_apply_shocks`, `_kernel_fused_h_day_shock`'s write pattern, `ParametricGenerator.iter_batches`'s column construction, and `extract_levels_from_snapshot`. The oracle is built **independently** from factor identity, base level, explicit shift semantics, explicit source-factor rule and explicit scale; it shares no helper, no mapping, no row-order construction and no matrix with the re-implementation. Every probe uses unique marker levels and unique marker shocks so that no permutation or transposition can produce a plausible-looking match.

---

## 1. Executive Verdict

**The shocked-level round trip is not trustworthy.**

The row-order contract itself is sound and provable: `flat_levels[i, s]` genuinely is the shocked level of `required_factor_ids[i]` under `required_shift_type_ids[i]` — *provided the shock column handed to row `i` is the right one*. It frequently is not.

`ShockComputePlan` sorts the K unique shocks by matrix column for cache locality, and publishes `inverse_permutation` to undo that sort. **Neither class of consumer applies it correctly.** `ParametricGenerator` never applies it at all. The four historical-family generators apply it with the wrong index space — `inverse_permutation` is indexed by shock-universe uix, and the kernels index it by sorted position. The two are inverse permutations of each other, so the historical family is correct exactly when the permutation is self-inverse and wrong for any 3-cycle or longer.

The consequence is the same in both cases and is the worst class this audit has found: **every required factor receives another factor's shock**, silently, with finite and entirely plausible levels. It survives every finiteness check in the estate because nothing is non-finite. It is not detectable by internal consistency, because the pipeline reproduces the wrong factor consistently in both the valuation and sensitivity paths.

Both defects are invisible to the shipped fixture for the same reason X-01 is: the fixture's source factor ids are `[0, 1, 2, 2]` — already ascending — so `inverse_permutation` is the identity and all three code paths agree. This is the third consecutive pass in which a live defect is masked by one benign ID layout in one fixture.

**Root-cause finding.** Pass 3's X-01 and this pass's Z-01 are not two bugs. They are one architectural failure appearing twice: `[fx_spots | scalar_rfs | disc_pts | price_pts]` is a concatenation of ascending subsets, and both consumers treat a position within that concatenation as if it were a position in a globally sorted space. The two share the same triggering topology condition.

**What survived.** A great deal, and it should be said plainly. The row-order gather is provably correct. All five shift formulas match an independent oracle exactly. `shock_scale` is applied exactly once. Missing source ids raise `KeyError`; unknown shift ids raise `ValueError`; missing snapshot factors raise `ValueError`. The scenario axis is coherent end to end and base is correctly held out of the scenario grid. **And Pass 4's Y-07 is resolved as NO DEFECT** by derivation: the `ABSOLUTE` default is the unique mathematically correct declaration, and it is correct independently of the simlab shift type.

---

## 2. Production Data-Flow

```
RiskFactorLevelSnapshot                 (factor_ids sorted-unique int32, levels f64)
        │
        │  extract_levels_from_snapshot(required_ids=universe_map.required_factor_ids)
        │  searchsorted_or_raise  ──► ValueError on any missing factor        [FAIL-FAST ✔]
        ▼
base_levels  (N_req,) f64      row i  ≡  required_factor_ids[i]               [PROVEN §4]
        │
        │  build_universe_mapping:
        │     required            = [ fx_spots | scalar_rfs | disc_pts | price_pts ]
        │     required_shift_type = static_config.rf_id_to_shift_type_id[required]
        │                            └── same gather, same array, same order  [PROVEN §4]
        ▼
ProxyResolution.source_factor_ids (N_req,)   ProxyResolution.shock_scale (N_req,) f32
        │
        │  generator.iter_batches(required_shift_type_id, source_factor_ids, shock_scale)
        ▼
build_shock_universe
    keys = (source_id << 8) | shift_id
    unique_first_seen(keys)          ──► K unique shocks in FIRST-SEEN order
    required_to_shock_uix  (N_req,)  ──► indexes FIRST-SEEN uix space
    _smoke_test_inputs               ──► ValueError on shift ∉ {-1,1,2,3,4}   [FAIL-FAST ✔]
        │
        ▼
build_shock_compute_plan
    cols        = searchsorted(available_factor_ids, shock_source_ids)
                  + bounds + identity check ──► KeyError on missing          [FAIL-FAST ✔]
    order       = argsort(cols)          sorted position k  ──► uix order[k]
    cols_sorted = cols[order]
    inv[order]  = arange                 uix u             ──► sorted pos inv[u]
        │
        ├── ParametricGenerator:  shocks[:, j] ← column cols_sorted[j]
        │                         ***  inverse_permutation NEVER USED  ***    ◀── Z-01
        │                         shocks indexed by SORTED POSITION
        │
        └── historical / historical_fhs / bootstrap_fhs / bootstrap_sbb:
                 for k: out[:, inv_perm[k]] = shock(cols_sorted[k])
                 ***  inv_perm indexed by SORTED POSITION, built for UIX  *** ◀── Z-02
        ▼
ScenarioSetBatch(shocks (B,K) f32 F-contig, required_to_shock_uix, shock_scale,
                 required_shift_type_id, scenario_offset)
    __post_init__: dtype / layout / equal-length only — no value validation
        │
        ▼
apply_shocks_to_base_levels → _kernel_apply_shocks
    out[n, b] = f_{shift[n]}( base_levels[n], shocks[b, req_to_uix[n]] * scale[n] )
    no `else` branch ──► unknown shift leaves the row unwritten               ◀── Z-04
        ▼
flat_levels / out_view  (N_req, B) f64
    runner: out_view = shocked_levels_buf[:, :current_b]
            non-C-contiguous on a partial final batch                         ◀── Z-03
        ▼
build_market_data_matrix_from_flat_levels
    raw_fx_spots     = flat_levels[idx_fx_spots]      ──► X-01 (Pass 3)
    raw_scalar_nodes = flat_levels[idx_scalar_nodes]  ──► Y-01 (Pass 4)
        ▼
MarketDataMatrix  ──►  valuation (fold) AND sensitivity (decompose)   [both SHARED]
        ▼
scenario P&L ──► VaR / ES
```

---

## 3. Finding Matrix

| ID | Severity | Cat | Status | File | Symbol | Finding | Why it matters | Evidence | Reachability | Owner |
|---|---|---|---|---|---|---|---|---|---|---|
| **Z-01** | **CRITICAL** | B | **CONFIRMED** | `simlab/scenarios/parametric.py` | `ParametricGenerator.iter_batches` | Builds `shocks[:, j]` from `plan.cols_sorted[j]` (sorted-position space) and yields `required_to_shock_uix` (uix space). `plan.inverse_permutation` is never referenced in the module. | Every required factor receives another factor's shock. Finite, plausible, silent. Corrupts both the valuation and sensitivity paths. | Probe DECISIVE 2 and 3: identity → MATCH; any non-identity → MISMATCH. Grep: `inverse_permutation` appears at 4 call sites, none in `parametric.py`. | Any book whose first-seen source-factor ids are not ascending. Guaranteed whenever an FX rf_id exceeds a scalar rf_id, or a curve-point rf_id is below a node rf_id, or proxies reorder sources. | Parametric generator |
| **Z-02** | **CRITICAL** | B | **CONFIRMED** | `simlab/scenarios/historical.py`, `historical_fhs.py`, `bootstrap_fhs.py`, `bootstrap_sbb.py` + their kernels | `_kernel_fused_h_day_shock` et al. | Kernel does `for k in prange(K): out[:, inv_perm[k]] = shock(cols[k])` with `cols = plan.cols_sorted`. `k` is a sorted position; `inverse_permutation` is documented and built as a **uix-indexed** map. The required map is `order = argsort(inv)`. | Same silent wrong-factor corruption as Z-01, on the historical/FHS/bootstrap families. | Probe DECISIVE 1 (3-cycle): shipped form MISMATCH, `out[:, order[k]]` MATCH. DECISIVE 2 (transposition): both MATCH. | Correct only when the permutation is an involution. Any 3-cycle or longer of source-factor ordering breaks it. | Shock kernels / plan contract |
| **Z-03** | HIGH | C | **CONFIRMED** | `simlab/runner.py`, `simlab/shocks/apply.py` | `SimLabRunner.run`, `validate_apply_shocks_buffers` | `out_view = shocked_levels_buf[:, :current_b]` is a **column** slice; it is not C-contiguous when `current_b < batch_size`. The kernel pins `out` to `_F64_2D_C`. `validate_apply_shocks_buffers` runs on the first batch only, so the partial final batch is never pre-validated. | Any run whose scenario count is not a multiple of `batch_size` fails at Numba dispatch. Loud, but it blocks the run and the one-shot validator cannot pre-empt it. | Probe P5-5: `buf[:, :58]` → `C_CONTIGUOUS=False`; `buf[:, :64]` → `True`. | `total_scenarios % batch_size != 0`. | Runner |
| **Z-04** | MEDIUM | C | **LATENT** | `simlab/shocks/shock_kernel.py`, historical shock kernels | `_kernel_apply_shocks` | The branch ladder covers `{LOG_RETURN, SIMPLE_RETURN, DIFFERENCE, ABSOLUTE_BASE_RETURN, NO_SHIFT}` with **no `else`**. An unrecognised shift id leaves the entire output row unwritten — `np.empty` heap. | A silently uninitialised factor row for every scenario. | Probe P5-2: shift id 0 → row untouched, sentinel `-12345.0` survives. | `build_shock_universe._smoke_test_inputs` rejects any id ∉ `{-1,1,2,3,4}`. Reachable only by constructing a `ScenarioSetBatch` directly — a public frozen dataclass whose `__post_init__` validates dtype and length but not shift values. | Shock kernel / batch contract |
| **Z-05** | LOW | E | **DOC** | `simlab/shock_plan/plan.py` | `ShockComputePlan.inverse_permutation` | The docstring — "for each shock-universe uix `k`, the position of its column in the kernel output" — describes a uix-indexed map, and `inv[order] = arange` builds one. Both consumer classes index it by sorted position. | This is the root documentation defect behind Z-01 and Z-02: the field's contract is stated correctly and used incorrectly at every site. | Source + the algebra in §4. | n/a | Plan contract |
| **Y-07** | — | E | **NO DEFECT (resolved)** | `sensitivities/book_adapter.py` | `build_sensitivity_book_spec` | Derivation in §6 proves `ABSOLUTE` is the unique correct declaration for `∂PV/∂S`, and that it is correct **independently** of the simlab shift type. Pass 4's MEDIUM/CONFIRMED is withdrawn. | — | §6 derivation against the `DisplacementConvention` definition block. | — | — |
| **Y-07b** | MEDIUM | C | **LATENT** | `sensitivities/book_adapter.py` | `displacement_convention` parameter | The convention is a caller-settable keyword applied uniformly to every factor, over coefficients whose coordinate system is fixed by the kernel. Nothing binds or checks it. | A caller passing `LOG` or `RELATIVE` silently reinterprets every coefficient. | Pass 4 probe P4-H: +18.89 % P&L error at a −30 % move. | Reachable by caller action only; no internal producer supplies a non-default value. | Book adapter |
| **X-01** | **CRITICAL** | B | **CONFIRMED — root cause identified** | `market_data/builder.py` | `_build_fx_spots_effective` | Carried. Pass 5 establishes that X-01 and Z-01 are the **same architectural failure** — a position within `[fx | scalar | disc | price]` treated as a position in a globally sorted space — with the same triggering topology. | — | Pass 3 §4; Pass 5 §11. | Shared with Z-01. | Market-data builder |
| **Y-01** | **CRITICAL** | B | **CONFIRMED — carried** | `market_data/builder.py` | `_build_scalar_node_levels` | Unchanged by this pass. FX-node rows of `prices` are never written. | — | Pass 4. | — | Market-data builder |
| **Y-02 / Y-03** | **CRITICAL / HIGH** | D | **CONFIRMED — extended** | `tests/sensitivities/` | oracle + fixture | Extended: the *simlab* fixture is masked by the same mechanism. `_SOURCE_FACTOR_IDS = [0,1,2,2]` is ascending, so `inverse_permutation` is the identity and Z-01/Z-02 cannot fire in any shipped test. | The assurance blind spot is now demonstrated at three independent layers. | Probe G. | — | Test / assurance |

---

## 4. Row-Order Proof

### Claim
For every `i`, `flat_levels[i, s]` is the shocked level of `required_factor_ids[i]` under `required_shift_type_ids[i]` — **conditional on the shock column being correct**, which §7 disproves.

### Step 1 — `required_factor_ids` → `required_shift_type_ids`

`build_universe_mapping` (assemble.py):

```python
required = np.concatenate([fx_spots, scalar_rfs, disc_children, price_children])
required_shift = static_config.rf_id_to_shift_type_id[required].astype(
    np.int8, copy=False
)
```

`rf_id_to_shift_type_id[required]` is NumPy advanced indexing with the array `required`. Advanced indexing produces `out[i] = table[required[i]]` by definition, so the result is in `required` order, element for element. **Alignment is not "by construction" in the hand-waving sense — it is the defining semantics of the gather, using the same array object.** No sort, unique, filter, transpose or reshape intervenes between the two statements.

**PROVEN ALIGNED.**

### Step 2 — `required_factor_ids` → `base_levels`

`extract_levels_from_snapshot`:

```python
idx = searchsorted_or_raise(universe=snapshot.factor_ids, raw=required_ids, ctx=...)
return np.ascontiguousarray(snapshot.factor_levels[idx]).astype(np.float64, copy=False)
```

`searchsorted_or_raise` returns `pos` of shape `(required_ids.size,)` with `universe[pos[i]] == required_ids[i]` enforced — it raises `ValueError` naming the offending ids otherwise. The subsequent `factor_levels[idx]` is again advanced indexing in `idx` order, hence in `required_ids` order.

Probed with a sparse, non-ascending required set `[30, 10, 20]` against marker levels `{10: 110, 20: 220, 30: 330}`:

```
base_levels (gathered) = [330.0, 110.0, 220.0]
expected               = [330.0, 110.0, 220.0]   ROW ORDER OK
```

**PROVEN ALIGNED**, and **fail-loud** on a missing factor.

### Step 3 — `base_levels` → `flat_levels`

`_kernel_apply_shocks` reads `base_levels[n]`, `req_to_uix[n]`, `shock_scale[n]`, `shift_types[n]` and writes `out[n]`. Row `n` of the output is therefore built from row `n` of every input array. The runner passes `required_shift_type_ids` and `ProxyResolution.{source_factor_ids, shock_scale}` — all `(N_req,)` and all in required order. `ScenarioSetBatch.__post_init__` enforces that the three share length `N_req`.

**PROVEN ALIGNED** on the row axis.

### Step 4 — the column that row `n` reads

This is where the chain breaks. `shocks[:, req_to_uix[n]]` is correct only if `shocks` columns are indexed in **shock-universe uix space**. §7 proves they are not, in either generator family.

### Verdict

> `required_factor_ids` ↔ `required_shift_type_ids` ↔ `flat_levels` **rows** are provably aligned.
> The **column** binding from a row to its shock is broken. The row-order contract is sound; the factor-identity contract is not.

---

## 5. Shock-Convention Proof

Derived from `_kernel_apply_shocks` line by line, and verified against an oracle written independently from the shift semantics. Let `x` be the base level, `d` the raw shock column value, `c` the per-required `shock_scale`.

| id | Name | Implemented transformation | Displacement induced |
|---|---|---|---|
| `1` | `LOG_RETURN` | `x' = x · exp(d·c)` | `ln(x'/x) = d·c` |
| `2` | `SIMPLE_RETURN` | `x' = x · (1 + d·c)` | `x'/x − 1 = d·c` |
| `3` | `DIFFERENCE` | `x' = x + d·c` | `x' − x = d·c` |
| `4` | `ABSOLUTE_BASE_RETURN` | `x' = x + d·c·\|x\|` | `(x'−x)/\|x\| = d·c` |
| `-1` | `NO_SHIFT` | `x' = x` | `0` |
| — | any other id | **row not written** | — (Z-04) |

`ABSOLUTE_BASE_RETURN` is the documented inverse of `r = (v_t − v_{t−1})/|v_{t−1}|`; it coincides with `SIMPLE_RETURN` for `x > 0` and differs only for `x < 0`, where the increment follows the shock's sign instead of inverting. Verified on a negative base.

Independent-oracle check (P5-1), `x = 100`, `d = 0.10`, `c = 1`:

```
LOG_RETURN  x*exp(d)        kernel=  110.517092  oracle=  110.517092  OK
SIMPLE      x*(1+d)         kernel=  110.000000  oracle=  110.000000  OK
DIFFERENCE  x+d             kernel=  100.100000  oracle=  100.100000  OK
ABS_BASE    x+d*|x| (x>0)   kernel=  110.000000  oracle=  110.000000  OK
ABS_BASE    x+d*|x| (x<0)   kernel=  -45.000000  oracle=  -45.000000  OK
NO_SHIFT    x               kernel=  100.000000  oracle=  100.000000  OK
```

**All five transformations are mathematically correct.**

`VALID_SHIFT_TYPE_IDS = {-1, 1, 2, 3, 4}`; `0` is deliberately excluded, and `_smoke_test_inputs` rejects it with an explicit membership test rather than a range check — the comment correctly notes that a range check would wrongly reject `NO_SHIFT = -1`. That is careful, correct boundary code.

---

## 6. Y-07 Verdict — **NO DEFECT**

### The question
Who owns the coordinate system of a sensitivity coefficient, and must it be bound to the simlab shift type?

### The contract, from the source

`solutions/sensitivity_pnl` defines the displacement for factor `k` given base level `L0` and shocked level `L`:

```
ABSOLUTE   x = L - L0
RELATIVE   x = L / L0 - 1
LOG        x = ln(L / L0)
```

and states that "a sensitivity is a derivative with respect to a *specific* displacement of a risk factor", declared per factor and resolved at compile time.

### Derivation

The engine computes, per factor, `ΔV = d₁·x + ½·d₂·x² + Σ cross·xᵢxⱼ`, where `x` is reconstructed from `(L0, L)` by the declared convention. So the convention must name the space in which `d₁` was differentiated. For the three candidates:

$$
d_1 = \frac{\partial V}{\partial x_{\mathrm{ABS}}} = \frac{\partial V}{\partial L},\qquad
d_1 = \frac{\partial V}{\partial x_{\mathrm{REL}}} = L_0\frac{\partial V}{\partial L},\qquad
d_1 = \frac{\partial V}{\partial x_{\mathrm{LOG}}} = L_0\frac{\partial V}{\partial L}
$$

(the last two coincide at first order because `∂x_REL/∂L = ∂x_LOG/∂L = 1/L₀` at `L = L₀`; they diverge at second order and away from the base.)

`_kernel_sensitize_spot` emits `∂PV/∂S = qty·fx` and `∂PV/∂F = qty·S` — partials with respect to the **absolute level**, carrying no factor of `L₀`. Therefore `ABSOLUTE` is the unique correct declaration, and `build_sensitivity_book_spec`'s default is right.

### The decisive point

**The convention is a property of the coefficient, not of the shock.** The engine reconstructs `x` from `(L0, L)` itself; it never sees how `L` was produced. A `LOG_RETURN`-shocked factor reaching an `ABSOLUTE` coefficient still gets `x = L − L₀`, which is exactly the displacement `∂V/∂L` multiplies. The simlab shift type and the `DisplacementConvention` are **not required to agree, and binding them would be wrong**: shocking in log space and differentiating in absolute space is a perfectly coherent combination, and it is the combination the estate actually ships.

This also explains why the Pass-4 numerical demonstration (`+18.89 %` at a −30 % move) is real but does not indicate a live defect: it measures what would happen under a *mis-declared* convention, not under the shipped one.

### Verdict

**Y-07 → NO DEFECT.** The shipped wiring is correct and requires no coupling to the shock convention.

**Y-07b → LATENT (MEDIUM).** The residual hazard is narrow and real: `displacement_convention` remains a caller-settable keyword applied uniformly across all factors, over coefficients whose space is fixed by the kernel. Nothing validates it, and nothing states in the adapter that the value is not a free choice. It becomes materially dangerous the day a sensitizer emits a relative or log partial alongside the absolute spot ones, since one uniform value cannot then be right for both.

---

## 7. Source-Factor / Proxy Analysis

### The routing chain

```
required factor i
  → ProxyResolution.source_factor_ids[i]                     (proxy rule)
  → key = (source_id << 8) | shift_id
  → unique_first_seen  → shock-universe uix u = req_to_uix[i]
  → cols[u] = searchsorted(available_factor_ids, source_id)   (matrix column)
  → order = argsort(cols);  cols_sorted = cols[order];  inv[order] = arange
  → generator computes shocks per sorted position k
  → apply_shocks reads shocks[:, req_to_uix[i]]  ← expects UIX SPACE
```

### The permutation algebra

`order[k]` = the uix whose column is k-th smallest. `inv[u]` = the sorted position of uix `u`. By construction `inv = order⁻¹`.

For `shocks[:, u]` to hold uix `u`'s shock, the generator must write

$$
\texttt{shocks[:, order[k]]} \;=\; \text{shock}\big(\texttt{cols\_sorted}[k]\big)
$$

**ParametricGenerator writes** `shocks[:, k] = shock(cols_sorted[k])` — correct iff `order = id`.
**Historical family writes** `out[:, inv[k]] = shock(cols_sorted[k])` — correct iff `inv = order`, i.e. iff the permutation is an **involution**.

### Executed discrimination

Three marker factors `{10: 110.0, 20: 220.0, 30: 330.0}` with `DIFFERENCE` shifts and unique marker shocks `{10: [1,2], 20: [10,20], 30: [100,200]}`, against the independent oracle:

| topology | `inverse_permutation` | parametric (as shipped) | historical (as shipped) | `out[:, order[k]]` |
|---|---|---|---|---|
| ascending source ids `[10,20,30]` | `[0,1,2]` (identity) | **MATCH** | **MATCH** | MATCH |
| transposition `[20,10]` (involution) | `[1,0]` | **MISMATCH** | **MATCH** | MATCH |
| 3-cycle `[30,10,20]` | `[2,0,1]`, `order=[1,2,0]` | **MISMATCH** | **MISMATCH** | MATCH |

Minimal adversarial example, 3-cycle, actual versus oracle:

```
row 0 (factor 30):  parametric got [331, 332]   historical got [340, 350]   oracle [430, 530]
row 1 (factor 10):  parametric got [120, 130]   historical got [210, 310]   oracle [111, 112]
row 2 (factor 20):  parametric got [320, 420]   historical got [221, 222]   oracle [230, 240]
```

Every factor receives another factor's shock. All values are finite and of plausible magnitude.

### Reachability

`source_factor_ids` is in `required_factor_ids` order, which is `[fx_spots | scalar_rfs | disc_pts | price_pts]` — ascending **within** each block, arbitrary **across** blocks. Identity therefore requires

$$
\max(\text{fx ids}) < \min(\text{scalar ids}) < \min(\text{disc point ids}) < \min(\text{price point ids})
$$

Nothing enforces or even documents that. Proxy routing makes it strictly less likely, because a proxied factor's source id is another factor's id and can land anywhere. A single discount-curve point whose rf_id falls below a scalar node's rf_id is sufficient for Z-01, with no FX involvement at all.

### Attacks run and results

| Case | Result |
|---|---|
| one-to-one mapping | correct |
| many-to-one (WTI → BRENT) | correct; dedup collapses to one column |
| identity mapping | correct |
| source id absent from `available_factor_ids` | **`KeyError` — fail-loud ✔** |
| duplicated source ids | correct; `unique_first_seen` collapses them |
| mixed shift types on one source | correct; the packed key separates them, dedup is on `(source, shift)` |
| `shock_scale = 0.0` | accepted; yields the base level. No guard — see §10 |
| `shock_scale < 0` | accepted; sign-flips the shock. No guard — see §10 |
| `shock_scale = 0.95` (proxy) | `70·exp(0.20·0.95) = 84.6474718360` — **applied exactly once ✔** |
| non-ascending first-seen source order | **Z-01 / Z-02 fire** |

---

## 8. Scenario-Axis Proof

### The chain

`shocks` is `(B, K)` F-contiguous: axis 0 is the scenario within the batch. `_kernel_apply_shocks` reads `shock_col = shocks[:, k]` and writes `out_row[b]` for `b ∈ [0, B)`, so `out[n, b]` corresponds to `shocks[b, ·]`. The runner then:

```python
start = batch.scenario_offset
end = start + current_b
out_view = shocked_levels_buf[:, :current_b]
...
scenario_pvs[:, start:end] = batch_result.values
```

Batch-local scenario `b` maps to global scenario `start + b`, and `scenario_offset` is advanced by exactly `scenario_count` per batch. No transpose, no reordering, no insertion or deletion.

### Base-scenario placement

The runner builds a **separate** `(N_req, 1)` base state from `base_levels[:, None]`, prices it, and subtracts at the end (`scenario_pvs -= base_pvs`). Therefore:

> **Scenario 0 is the first shocked scenario, not the base state.** The base is held out of the scenario grid entirely.

This is unambiguous, correct, and worth recording because it is the opposite of the common convention and nothing in the audited docstrings states it.

### Historical ordering

The historical generators walk `t_start` from `h` upward to `t_max`, so scenario 0 is the **oldest** usable observation and the final scenario is the most recent. Consistent within the family.

### Attack

Probed with an asymmetric marker matrix (`scenario 0 → 1, scenario 1 → 10, scenario 2 → 100`) so that a transposition or reversal cannot produce a plausible result. Scenario ordering was preserved in every probe, including the wrong-factor cases — i.e. **Z-01/Z-02 permute the factor axis, never the scenario axis.**

**Scenario-axis semantics: PROVEN CORRECT.**

---

## 9. Independent Economic Oracle Results

The oracle takes only `(factor identity, base level, shift semantics, source rule, scale, raw shock)` and shares no helper with the re-implementation.

| Probe | Topology | Expected (oracle) | Actual | Verdict |
|---|---|---|---|---|
| **G** — shipped fixture | src `[0,1,2,2]`, LOG_RETURN, scale `[1,1,1,0.95]` | `[[1.0909, 1.0586], [221.03, 163.75], [223.71, 11.138], [181.00, 10.470]]` | identical | **MATCH** — the fixture cannot detect Z-01 |
| **A** — 3-cycle, parametric | src first-seen `[30,10,20]` | `[[430,530],[111,112],[230,240]]` | `[[331,332],[120,130],[320,420]]` | **MISMATCH** |
| **A′** — 3-cycle, historical | same | `[[430,530],[111,112],[230,240]]` | `[[340,350],[210,310],[221,222]]` | **MISMATCH** |
| **DECISIVE 2** — transposition | src `[20,10]` | `[[230,240],[111,112]]` | parametric MISMATCH; historical MATCH | discriminates the two defects |
| **DECISIVE 3** — ascending | src `[10,20,30]` | `[[111,112],[230,240],[430,530]]` | all three forms MATCH | identity masks both |
| **C** — mixed shift types | LOG/SIMPLE/DIFF/ABS_BASE/NO_SHIFT adjacent | see §5 table | all six exact | **MATCH** |
| **D** — proxy, `scale = 0.95` | WTI ← BRENT | `70·exp(0.19) = 84.6474718360` | `84.6474718360` | **MATCH** |
| **H** — degenerate | `N_req = 0`; `N_req = 1, B = 4`; `N_req = 1, B = 1` | no exception, correct shapes | `(0,4)`, `(1,4)`, `(1,1)` | **MATCH** |
| **P5-2** — invalid shift | id `0` | reject | `ValueError` at `build_shock_universe`; row unwritten if the boundary is bypassed | boundary ✔ / kernel latent |
| **P5-3** — missing source | id `99` ∉ available | reject | `KeyError: missing source ids [99]` | **✔** |
| **P5-5** — partial batch | `buf[:, :58]` | contiguous | `C_CONTIGUOUS = False` | **Z-03** |

---

## 10. Critical Negative Findings

Attacked and explicitly cleared.

**Row order is preserved from `required_factor_ids` into `flat_levels`.** Proven, not assumed: two advanced-index gathers on the same array object, with no intervening reordering operation, plus a sparse non-ascending marker probe.

**`required_shift_type_ids` is aligned with that ordering.** `rf_id_to_shift_type_id[required]` is the same gather on the same `required`.

**`extract_levels_from_snapshot` is fail-loud.** `searchsorted_or_raise` raises `ValueError` naming the missing ids. The `RiskFactorLevelSnapshot` sorted-unique invariant it relies on is enforced in that class's own `__post_init__`, including a re-sort when needed.

**All five shift transformations are mathematically correct**, including the subtle `ABSOLUTE_BASE_RETURN` behaviour at a negative base.

**`shock_scale` is applied exactly once**, inside the shift formula, in float64. The float32→float64 promotion at the `apply` boundary is correctly placed and correctly justified (the scale multiplies inside `exp`).

**Mixed shift types are interpreted correctly.** The packed key `(source << 8) | (shift & 0xFF)` separates them; `NO_SHIFT = -1` round-trips through the low-8-bit mask as documented, and the dedup is on the pair rather than on the source alone.

**Missing source factor ids fail loud** with a `KeyError` naming them — `build_shock_compute_plan` performs a proper bounds-plus-identity membership check. **This is precisely the guard whose absence causes X-01, and here it is present and correct.**

**Unknown shift type ids fail loud** at `build_shock_universe._smoke_test_inputs`, via explicit membership against `VALID_SHIFT_TYPE_IDS` rather than a range check.

**Empty and singleton universes are safe.** `N_req = 0`, `K = 0`, `B = 1` all produce correctly-shaped output with no exception. `unique_first_seen` has an explicit empty fast path.

**Scenario ordering is preserved end to end**, and the base state is correctly held out of the scenario grid.

**Sparse and non-zero-based global factor ids are safe** on the row axis — every gather is by value through `searchsorted`, never by positional assumption. The defect is confined to the **column** axis.

**`ScenarioSetBatch.__post_init__` is a real O(1) contract gate** — dtype, rank, F-contiguity of `shocks`, C-contiguity of the three `(N_req,)` arrays, equal lengths, and non-bool integer checks on the offsets.

**Constitutional compliance holds on this seam.** No strings past ingest; no Polars or Pandas anywhere in `simlab.shocks` or `simlab.builder`; SoA preserved; no objects across the JIT boundary; canonical `riskquant.jit` decorators and `_F32_2D_F` / `_F64_2D_C` / `_I8_1D_C` / `_I32_1D_C` aliases only; no ad-hoc compiler configuration; the branch ladder is correctly hoisted out of the B-loop; zero allocation in the hot path; `nb.prange` the only raw Numba use. The dual C/F overloads are the documented RQ-AUD-017 remediation and are correct.

**`shock_scale = 0` and `shock_scale < 0` are accepted without a guard.** Deliberately **not** raised as a defect: a zero scale is a legitimate way to hold a proxied factor fixed, and a negative scale is a legitimate inverse-proxy relationship. Recorded as a design question for the proxy layer, not a defect in this seam.

---

## 11. Cross-Pass Impact

### X-01 — root cause identified, severity unchanged

Pass 3 found that `_build_fx_spots_effective` indexes a dense `flat_levels` sub-block with a `market_id_universe` index. Pass 5 shows this is not an isolated slip. The `[fx_spots | scalar_rfs | disc_pts | price_pts]` layout is a **concatenation of independently-ascending subsets**, and *two different consumers on two different seams* treat a position within it as a position in a globally sorted space. X-01 and Z-01 share one triggering condition: whether the concatenation happens to be globally ascending.

They also interact. In the FX-interleaved topology, X-01 raises `IndexError` inside `_build_fx_spots_effective` — which runs **after** `apply_shocks_to_base_levels` in the same batch. X-01's loud failure therefore *masks* Z-01 in that specific topology. Z-01 fires alone, silently, whenever the non-ascending ordering arises among scalar nodes and curve points with the FX block still well-formed.

**No new ID is created for this.** X-01 remains the finding; Pass 5 supplies its root-cause framing and a second instance.

### Y-01 — unchanged

Uninitialised FX-node rows in `prices` are a separate defect in a different function. Not touched by this seam.

### Y-02 / Y-03 — reachability extended

The assurance blind spot is now demonstrated at three layers rather than one:

1. the sensitivity "oracle" executes neither engine (Pass 4);
2. the sensitivity fixture places USDEUR at `node_id 0`, masking X-01 (Pass 4);
3. the simlab fixture's `_SOURCE_FACTOR_IDS = [0, 1, 2, 2]` is ascending, masking Z-01 and Z-02 (Pass 5, Probe G).

All three reduce to one property: **every shipped fixture uses a globally ascending id layout, which is exactly the degenerate case in which all three index-space defects vanish.** No test in the repository exercises a non-ascending universe.

### Y-05 — unchanged

Base-PV reconciliation remains structurally blind. Z-01/Z-02 would be caught by a scenario-level reconciliation against an independent economic oracle — the same instrument that would catch X-01.

### Y-07 — CLOSED as NO DEFECT

Resolved by derivation in §6. Pass 4's MEDIUM/CONFIRMED is withdrawn. The residual interface hazard is re-scoped as **Y-07b, LATENT**.

### Net effect on the risk number

Under Z-01 or Z-02 the shocked levels are factor-permuted before `MarketDataMatrix` is built. Both the valuation path and the sensitivity path consume them, so:

* the sensitivity *coefficients* (computed at base, scenario 0) remain correct, because the base state is built from unshocked `base_levels[:, None]` and never passes through the generator;
* the *scenario displacements* `x = L − L₀` are those of the wrong factor;
* the resulting P&L pairs each position's correct sensitivity with another factor's move.

The distribution is therefore assembled from a permuted correlation structure. VaR and ES are wrong by an amount governed by the volatility and correlation mismatch between the swapped factors, with no bound and no sign guarantee. Unlike X-01 — which is a clean multiplicative scale error preserving 95.5 % of tail ranking — this corrupts the tail *selection* itself.

---

## 12. Final Audit Classification

### Proven correct
Row-order alignment of `required_factor_ids` ↔ `required_shift_type_ids` ↔ `flat_levels` rows. `extract_levels_from_snapshot` gather and its fail-loud membership check. All five shift-type transformations. `shock_scale` applied exactly once, with a correctly-placed f32→f64 promotion. Proxy dedup on `(source, shift)` pairs. Missing-source `KeyError`. Unknown-shift `ValueError`. Empty and singleton universes. Scenario-axis semantics end to end, including base-state exclusion. `ScenarioSetBatch` structural gate. Constitutional compliance of the whole seam.

### Proven defective
**Z-01 (CRITICAL)** — `ParametricGenerator` never applies `inverse_permutation`; shock columns are in sorted-position space while `required_to_shock_uix` is in uix space.
**Z-02 (CRITICAL)** — the four historical-family kernels index `inverse_permutation` by sorted position when it is built as a uix-indexed map; correct only for involutions.
**Z-03 (HIGH)** — the runner's partial final batch produces a non-C-contiguous `out_view` that the pinned kernel signature rejects, and the one-shot validator runs too early to catch it.

### Conditional
None new. Prior conditionals (X-07 vol basis, X-08/X-09 dual universes, X-11 field collision, V-03 forward/log-forward) are untouched by this seam.

### Latent
**Z-04** — `_kernel_apply_shocks` and the historical shock kernels have no `else` branch; an unrecognised shift id leaves the row unwritten. Blocked today by `build_shock_universe._smoke_test_inputs`, reachable only through direct `ScenarioSetBatch` construction.
**Y-07b** — `displacement_convention` is caller-settable and uniform across factors, over coefficients whose coordinate system is fixed by the kernel.

### Documentation only
**Z-05** — `ShockComputePlan.inverse_permutation`'s docstring states a uix-indexed contract that no consumer honours. The root documentation defect behind Z-01 and Z-02.

### Numerical observation
None on this seam. The `math.exp` / libm choice, the f32 shock storage and the f64 level reconstruction are all correctly reasoned and correctly placed.

### No defect
**Y-07** — the `ABSOLUTE` displacement convention is the unique correct declaration for the shipped coefficients, and correctly independent of the simlab shift type.
`shock_scale = 0` and `shock_scale < 0` — legitimate proxy semantics, not a defect in this seam.

---

## 13. Single Next Audit Target

### `riskquant.simlab.proxy` — `ProxyResolution` and its producer

This is the last unaudited node feeding the production risk number, and it is the sole producer of the two arrays whose ordering this pass has just shown to be load-bearing.

**It owns the input that makes Z-01 fire or not.** `ProxyResolution.source_factor_ids` determines the first-seen source ordering, which is the exact discriminator between the identity permutation and a corrupting one. Pass 5 established the failure mode and the triggering condition; only this module can establish how often real books meet it, and whether any ordering guarantee exists that the consumers could have been relying on.

**It is the third array that must be in `required_factor_ids` order, and the only one whose alignment this pass could not prove.** `required_shift_type_ids` was proven aligned because it is a gather on `required` inside `build_universe_mapping`. `base_levels` was proven aligned because it is a gather on `required_factor_ids` inside `extract_levels_from_snapshot`. `source_factor_ids` and `shock_scale` arrive from a different module entirely, and the runner passes them positionally into `iter_batches` alongside `universe_map.required_shift_type_ids` with nothing binding the three to a common row space. A silent misalignment there would be indistinguishable from Z-01 in its effects.

**D013 flags it as unsettled.** "Proxy is a shock transformation; identity ≠ provenance" records a decision in this area that the seam audit has now made consequential.

The contracts to trace: whether `ProxyResolution` is constructed from `universe_map.required_factor_ids` or from some other ordering; whether `source_factor_ids[i]` is guaranteed to describe required factor `i`; whether a self-map is represented as the factor's own id or as a sentinel; and what, if anything, constrains the resulting first-seen source ordering.
