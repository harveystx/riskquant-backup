# SimLab Production Audit

Scope: `src/riskquant/simlab` (complete package). Method: source inspection +
empirical probes (fresh Numba compilation, captured `nopython_signatures`,
degenerate-shape dispatch, failure-mode injection, RNG determinism). Tree state:
reconstruction + D024. No production code was edited for this audit.

Evidence tags: `[RUN]` = executed; `[SIG]` = captured compiled signature;
`[SRC]` = source inspection.

---

## 0. Flagged defect verdict — float64 array index at `col_vals[t1]`

**Verdict: NOT reproducible in this tree. NOT introduced by D024. NOT exposed by
D024. Root-cause CLASS is real (F3 below).**

Evidence:
- `[SIG]` `_kernel_fused_h_day_shock` compiles to three overloads; the scalar
  params `t_start, h, batch_len` are `int64` in **every** overload. No float64
  index type exists in the compiled kernel.
- `[RUN]` `HistoricalGenerator` and `FilteredHistoricalGenerator` both run
  (9 scenarios) after a cache-cleared fresh compile.
- `[SRC]` caller wraps `np.int64(t_start)`, `np.int64(h)`, `np.int64(batch_len)`
  (historical.py:213-215) and coerces `h = int(self.horizon)` (historical.py:174).
- `[RUN]` constructive reproduction: an **unpinned** `@njit` kernel with the same
  body, called with a **float** `t_start`, raises exactly
  `getitem(array(float64, 1d, F), float64) → Unsupported array index type
  float64`. The same kernel with an int scalar compiles and runs.

Therefore the reported error requires BOTH (a) an unpinned/lazy kernel signature
AND (b) a float scalar reaching the kernel. The current kernel has a pinned
`int64` signature AND an int-coercing, `np.int64`-wrapping caller — two
independent guards. D024 added a `(values C, out C)` overload with identical
`int64` scalar params; it cannot introduce a float index.

The **latent cause class** is real and is finding **F3**: `horizon` integrality
is never validated. `[RUN]` `HistoricalGenerator(horizon=1.0)` yields
`total_scenarios = 9.0` (float). Today the kernel still runs because
`iter_batches` re-coerces `h = int(...)`. If a future refactor removed either the
pinned signature or the `int()` coercion, the unvalidated float horizon would
produce exactly the reported error. Fix the class at the boundary (F3), not the
kernel.

Operational note: kernels use `cache=True`. A stale on-disk Numba cache from a
prior source revision is a plausible way to observe a mismatched compiled kernel;
clearing `__pycache__`/`*.nbi`/`*.nbc` is the first diagnostic step if the error
recurs in a specific checkout.

---

## A. Architecture map (actual)

```
EDGE / CONFIG                         CORE (Numba/NumPy)                     ORCHESTRATION
─────────────                         ──────────────────                     ─────────────
RiskStaticConfig ┐
ProxyTopology    ┼─> proxy_resolver.resolve_required_factor_ids...
required_ids     ┘        │  -> ProxyResolution{source_factor_ids i32,
                          │                      shock_scale f32,
                          │                      required_factor_ids i32}
                          v
        simlab.builder.build_shock_universe(required_shift_type_id i8,
                          │                  source_factor_ids i32,
                          │                  available_factor_ids i32)
                          │   dedup (bit-packed i64 keys) -> ShockUniverse{
                          │       required_to_shock_uix i32, compute_plan, ...}
                          v
RiskFactorMatrix ──> ScenarioGenerator.iter_batches(...) ──> ScenarioSetBatch{
 (T,K) f64 F-contig        │  [historical | filtered-historical |               shocks (B,Kreq) f32 F,
 (fortran_order=True)      │   parametric | SBB-bootstrap | filtered-bootstrap]  required_to_shock_uix,
                          │  kernels: h-day-shock / AR1 / EWMA / stitch          shock_scale, shift_type,
                          v                                                       offset, count_total}
        simlab.shocks.apply_shocks_to_base_levels(base_levels f64 1d C,
                          │  batch, out (Nreq,B) f64 C)  [kernel _kernel_apply_shocks]
                          │  (scale f32->f64 promotion at this boundary)
                          v
                  shocked_levels (Nreq,B) f64 C
                          │
        ┌─────────────────┴───────────────────────────────┐
        v (main risk path: sensitivity)                    v (oracle path: full reval)
   FactorAxis.levels -> SensitivityPnlCalculator      SimlabRunner.run(snapshot):
   -> reporting.fx -> scenario_metrics.compute_var_es   per batch -> run_valuation -> PnL
                                                         (D018 oracle; ZERO test coverage)
```


**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.


- **Producers**: scenario generators (shocks); proxy resolver (source/scale).
- **Consumers**: `apply_shocks_to_base_levels`; `SimlabRunner`; the sensitivity
  demo constituents.
- **Boundaries (validate here)**: `RiskFactorMatrix.__post_init__`,
  generator `__post_init__`, `build_shock_universe._smoke_test_inputs`,
  `validate_apply_shocks_buffers`, `proxy._smoke_test_inputs`.
- **Kernels (dumb math)**: `_kernel_fused_h_day_shock`,
  `_kernel_fused_filtered_h_day_shock`, `_kernel_fused_sbb_h_day_shock`,
  `_kernel_fused_filtered_bootstrap_h_day_shock`, `kernel_ar1_residuals`,
  `kernel_ewma_vol_ratios[_from_residuals]`, `_kernel_stitch_sbb_paths`,
  `_kernel_apply_shocks`.
- **Orchestration**: `SimlabRunner`; generator `iter_batches` loops.
- **Shared contracts**: `ScenarioGenerator` protocol, `ScenarioSetBatch`,
  `ShockUniverse`/`ShockComputePlan`, `RiskFactorMatrix`.

No `SimulationConfig`/`GeneratorFactory`/`ModelRegistry` exists. None is present
to remove or preserve.

---

## B. Module inventory

| Module | Purpose | Public surface | Kernel deps | Input contract | Output contract | Risk |
|---|---|---|---|---|---|---|
| `scenarios/base.py` | Protocol + batch payload | `ScenarioGenerator`, `ScenarioSetBatch` | — | — | frozen dataclass, **no validation** | F7 |
| `scenarios/historical.py` | Historical h-day shocks | `HistoricalGenerator` | `_kernel_fused_h_day_shock` | F64 F matrix, i8/i32/f32 req arrays | `(B,Kreq)` f32 F batch | F2,F3 |
| `scenarios/historical_fhs.py` | AR1+EWMA filtered HS | `FilteredHistoricalGenerator` | AR1, EWMA-resid, filtered h-day | as above | as above | F2,F3 |
| `scenarios/parametric.py` | MC (normal/t/VG/regime) | `ParametricGenerator` + factories | (draws in Python via `Generator`) | cholesky, vols, model params | `(B,Kreq)` f32 F | F6 |
| `scenarios/bootstrap_sbb.py` | Stationary block bootstrap | `StationaryBlockBootstrapGenerator` | `_kernel_fused_sbb_h_day_shock`, stitch | F64 F matrix | `(B,Kreq)` f32 F | F2,F4,F6 |
| `scenarios/bootstrap_fhs.py` | Filtered block bootstrap | `FilteredBootstrapGenerator` | AR1, EWMA-resid, filtered-bootstrap | F64 F matrix | `(B,Kreq)` f32 F | F2,F4,F6 |
| `scenarios/core_math/ar1_filter.py` | AR(1) residuals | `kernel_ar1_residuals` | self | `(T,Ktot)` F64, cols/shifts | `(T,Kreq)` F64 residuals | — (D024 fixed) |
| `scenarios/core_math/ewma_volatility.py` | EWMA Z-ratios | `kernel_ewma_vol_ratios[_from_residuals]` | self | `(T,K)` F64 | `(T,K)` F64 ratios | F9 (raw variant) |
| `scenarios/core_math/sbb_block_generator.py` | SBB block layout | `_kernel_stitch_sbb_paths` | self | i32 arrays, i32 scalar | i32 path matrix | F10 |
| `builder/shock_universe.py` | Dedup shock space | `build_shock_universe`, `ShockUniverse` | — | i8/i32 req arrays, sorted i32 avail | `ShockUniverse` | ok (validates) |
| `shock_plan/plan.py` | Compute-plan resolution | `ShockComputePlan`, `build_shock_compute_plan` | — | source ids, shifts, factor ids | plan (cols/perm/shifts) | (see G) |
| `shocks/apply.py` | Apply shocks (hot) + validator | `apply_shocks_to_base_levels`, `validate_apply_shocks_buffers` | `_kernel_apply_shocks` | f64 base, batch, f64 out | mutates `out` | F8 |
| `shocks/shock_kernel.py` | Apply-shocks kernel | `_kernel_apply_shocks` | self | f64 1d C, f32 2d shocks, i32/f64/i8, f64 2d C out | mutates out | F2 (shift ladder) |
| `runner.py` | Full-reval streaming VaR | `SimlabRunner` | via valuation | config/portfolio/generator | `PnlMatrix` f32 | F3-crash, **zero coverage** |

---


**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.

## C. Canonical contract table (from code)

| Concept | dtype | shape | layout | ownership |
|---|---|---|---|---|
| factor / source / required ids | `int32` | `(N,)`/`(K,)` | C | matrix sorted-unique ascending |
| shift type id | `int8` | `(Nreq,)` | C | `0..3`; `>=0` enforced in universe |
| `required_to_shock_uix` | `int32` | `(Nreq,)` | C | index into unique shock columns |
| `shock_scale` | `float32` | `(Nreq,)` | C | proxy owns; promoted to f64 at apply |
| risk-factor levels (matrix) | `float64` | `(T,Ktot)` | **F** (`fortran_order=True`) | matrix normalizes |
| historical obs matrix | `float64` | `(T,K)` | F | kernel-pinned (D017) |
| AR1 residuals / EWMA ratios | `float64` | `(T,Kreq)` | F (or C at Kreq==1) | scratch, generator-owned |
| scenario shocks (`ScenarioSetBatch.shocks`) | `float32` | `(B,Kreq)` | **F** (C at B==1) | generator; consumed by apply |
| base levels | `float64` | `(Nreq,)` | C | snapshot extraction |
| shocked levels (`out`) | `float64` | `(Nreq,B)` | **C** | caller pre-allocates |
| scenario/PnL (runner) | `float64`→`float32` | `(n_trades,S)` | C | runner |
| scalar counts `t_start/h/batch_len` | `int64` (kernel) | scalar | — | caller `np.int64(...)` |
| SBB stitch scalar | `int32` | scalar | — | inconsistent (F10) |
| `total_scenarios` | **declared `int`, not enforced** | scalar | — | **leaks float (F3)** |

Inconsistencies: `total_scenarios`/`horizon` integrality unenforced (F3);
SBB stitch scalar `int32` vs h-day `int64` (F10); `ScenarioSetBatch` carries all
of the above with zero runtime enforcement (F7).

---

## D. Kernel audit (captured signatures)

All `int64` scalars; no float index anywhere `[SIG]`.

| Kernel | overloads | f64 in layout | out layout | scalar | parallel | index typing | notes |
|---|---|---|---|---|---|---|---|
| `_kernel_fused_h_day_shock` | 3 | F, F, **C** | f32 F/C/C | i64 | prange over cols | int (`t1=t_start`) | D024 |
| `_kernel_fused_filtered_h_day_shock` | 3 | (F,F)/(F,F)/(**C,C**) | f32 F/C/C | i64 | prange | int | D024 |
| `_kernel_fused_sbb_h_day_shock` | **2** | F only | f32 F/C | i64 | prange | int | **F4 single-col** |
| `_kernel_fused_filtered_bootstrap_h_day_shock` | **2** | F only | f32 F/C | i64 | prange | int | **F4 single-col** |
| `kernel_ar1_residuals` | 3 | F/F/**C** | F/C/C | — | prange | int | D024 |
| `kernel_ewma_vol_ratios_from_residuals` | 2 | F/**C** | F/C | f64 | prange | int | D024 |
| `kernel_ewma_vol_ratios` (raw) | **1** | F only | F | f64 | prange | int | **F9 unused, single-col latent** |
| `_kernel_stitch_sbb_paths` | 1 | i32 arrays | i32 2d C | **i32** | — | int | F10 |
| `_kernel_apply_shocks` | 2 | f64 1d C base; f32 F/C shocks | f64 2d C | — | prange | int | ok |

All four fused shock kernels branch `if LOG / elif SIMPLE / elif DIFFERENCE`
with **no `else`** (historical:116, fhs:358, sbb:638, boot-fhs:877). An
unhandled shift id (e.g. `NO_SHIFT=0`, or any `>=4`) leaves the output column
**unwritten** → uninitialized `np.empty` garbage. **F2**.

---

## E. Layout-dispatch matrix (empirical `[RUN]`)

Single-factor matrix `K_total==1` (`(T,1)` C=F ambiguity):

| Generator | result |
|---|---|
| `HistoricalGenerator` | **OK** `(11,1)` (D024) |
| `FilteredHistoricalGenerator` | **OK** `(11,1)` (D024) |
| `StationaryBlockBootstrapGenerator` | **RAISES** `TypeError: No matching definition` (F4) |
| `FilteredBootstrapGenerator` | **RAISES** `TypeError: No matching definition` (F4) |

Single-required-from-multi (`K_total>1, Kreq==1`): historical + FHS OK and
value-identical to the multi-factor column at `max_abs_diff = 0.0` (D024 tests).
Partial / single-row trailing batch: covered for historical/FHS/SBB/parametric
by `test_partial_batch_contiguity.py` + `test_filtered_historical_scenarios.py`.
`_kernel_apply_shocks`: single-row batch handled by its f32 F/C dual overload.

---

## F. Scenario-generator matrix

| Generator | constructor | model params | horizon | scenario count | RNG | batch | validation | failure modes |
|---|---|---|---|---|---|---|---|---|
| Historical | `matrix, horizon` | — | `T-h` scenarios | `T - horizon` | none (deterministic) | streaming F buffer | `s_total<=0` only | **F3** (no horizon-int), F2 |
| FilteredHistorical | `matrix, horizon, decay, max_vol_ratio` | EWMA λ, clamp | `T-h` | `T - horizon` | none | streaming | `T<3`, `s_total<=0` | **F3**, F2 |
| Parametric | factories `standard_normal/student_t/vg/regime`; `available_factor_ids, cholesky, vols, total_scenarios, seed` | df, ν, prob_crisis, mult | n/a | explicit `total_scenarios` | `default_rng(seed)` per `iter_batches` | `(bs,k)` block/batch | **strong** (finite chol, pos vols, seed type, model params) | int() **not persisted**; F6 |
| SBB bootstrap | `matrix, total_scenarios, expected_block_size, horizon, seed` | geometric block p | `>=1` | explicit | `default_rng(seed)` per call | per call | `>0`, `block>=1`, `h>=1` | F2, F4, **F6** |
| Filtered bootstrap | `matrix, total_scenarios, horizon, seed, decay, max_vol_ratio` | EWMA + block | `>=1` | explicit | `default_rng(seed)` per call | per call | `T<3`, `>0`, `h>=1` | F2, F4, **F6** |

**Is `ScenarioGenerator` sufficient?** Yes — for the wiring contract. Every
generator satisfies the same `iter_batches(*, required_shift_type_id,
source_factor_ids, shock_scale, batch_size) -> Iterator[ScenarioSetBatch]`
signature `[SRC/RUN]`; D024 tests + partial-batch tests exercise multiple
implementations through it uniformly. The protocol does **not** need redesign.
The insufficiency is not the protocol but the **absence of a shared
construction-time validation contract** (F3/F7) — a base `__post_init__` helper,
not a new abstraction.

---

## G. Proxy / shock trace (one required factor)

Traced WTI_CRUDE (proxied to BRENT_CRUDE ×0.95), evidenced by the green demo's
25 scenario golden checks `[RUN, prior]`:

1. required `WTI rf_id` → `resolve_required_factor_ids_...` → `source_factor_id =
   BRENT rf_id`, `shock_scale = 0.95` (f32). WTI keeps its own base level 70.00.
2. `build_shock_universe` bit-packs `(BRENT, LOG_RETURN)`; WTI and any other
   BRENT-sourced requireds **dedup to one shock column**; `required_to_shock_uix`
   points WTI's row at that column.
3. Generator computes the BRENT shock **once** per scenario.
4. `apply_shocks` reads `required_to_shock_uix` → WTI reads BRENT's shock column,
   scales inside the exponent: `WTI_level = 70·exp(0.95·ΔBRENT)`.

Checks: no duplicate simulation (dedup collapses to one column); no missing
simulation (searchsorted-or-raise on absent source, `[RUN]` KeyError proven); no
double scaling (scale applied once, inside `exp`, per D013); no silent broadcast
(shapes `(Nreq,)`↔`(K,)` via explicit uix map); no ID mismatch (int32 throughout,
sorted-unique matrix ids); no transpose (F-shocks/C-out layout pinned); dtype: the
only conversion is the documented f32→f64 `shock_scale` promotion at the apply
boundary. **Proxy/shock mapping is correct.**

---

## H. Numerical oracle coverage

| Numerical unit | independent oracle | status |
|---|---|---|
| EWMA vol ratios (from residuals) | yes (two-pass, from definition) | **present** (`test_filtered_historical_scenarios.py`, D024) |
| AR(1) residuals | `test_ar1_filter.py` | present (verify independence) |
| Historical h-day shocks | `test_historical_math.py` | present |
| FHS end-to-end (AR1+EWMA+aggregation composed) | — | **MISSING** |
| Bootstrap / SBB numerical | `test_bootstrap_scenarios.py` | present (verify independence) |
| Apply-shocks (level reconstruction) | demo golden (deterministic) | partial; no unit oracle |
| Shock-universe dedup / packing | — | **MISSING (gap)** |
| Parametric moments (cov/vol recovery) | `test_parametric_scenarios.py` | present (verify independence) |

Priority gaps: FHS composed oracle; apply-shocks unit oracle; shock-universe
dedup property test. (Independence of the existing suites was not re-verified line
by line in this audit — flagged for the remediation test phase.)

---

## I. Failure-mode audit `[RUN]`

| Input | behaviour | verdict |
|---|---|---|
| float32 matrix | `TypeError: No matching definition` (deep in kernel) | rejects, but **cryptic, not boundary** — F5 |
| C-order matrix (`fortran_order=True`) | OK — matrix converts to F | correct |
| non-positive level + LOG_RETURN | **OK, produces NaN silently** | **F1 fail-loud violation** |
| `horizon=1.0` | OK; `total_scenarios=9.0` (**float**) | **F3** |
| `horizon >= T` | `ValueError: Not enough history` | fail-fast ✓ |
| duplicate required ids | OK (deduped) | acceptable; undocumented |
| empty required set | OK, `(B,0)` shocks | **no fail-fast** — F11 |
| required id absent from matrix | `KeyError: ...missing from available_factor_ids` | fail-fast ✓ |
| single-factor SBB/FHS-bootstrap | `TypeError` | **F4** |

---

## J. RNG audit `[RUN]`

- **Seed ownership**: streaming generators build `np.random.default_rng(self.seed)`
  **inside `iter_batches`** (sbb:182, boot-fhs:243, parametric per-batch draw).
- **Determinism**: same seed + same batch_size → identical output (SBB proven
  `identical=True`).
- **Batch-size dependence**: same seed + **different** batch_size → **different**
  output (SBB proven `identical=False`). Documented for parametric (CLAUDE.md §6);
  observable in bootstrap too. **F6** — VaR depends on an orchestration knob.
- **Independence across co-seeded generators**: not guaranteed (D021 OPEN).
- **Parallelism**: kernels are `prange` over factor columns; RNG draws happen in
  Python (not in prange), so parallelism does not perturb the stream.

---

## K. Performance audit (no optimisation performed)

- Hot kernels: the four fused h-day-shock kernels + `_kernel_apply_shocks`, all
  `prange` over factors, `cache=True`, zero in-kernel allocation `[SRC]`.
- Hidden allocation: `apply_shocks_to_base_levels` does
  `np.ascontiguousarray(batch.shock_scale, dtype=np.float64)` **per batch** — an
  `(Nreq,)` alloc in the streaming loop (**F8**; documented as cheap, not proven).
- FHS caches AR1+EWMA `(T,Kreq)` scratch lazily and reuses across batches `[SRC]`
  — good.
- No benchmarks recorded in-repo; no latency SLA. Perf claims must be measured
  before any change (none made here).

---

## L. Findings register

| ID | Sev | Module | Finding | Evidence | Root cause | Risk | Remediation | Test | Status |
|---|---|---|---|---|---|---|---|---|---|
| F1 | **P1** | historical, fhs, sbb, boot-fhs kernels | non-positive level in LOG/SIMPLE return → silent NaN in shocks | `[RUN]` `has_nan=True` | no `level>0` gate before `log`/`div` | NaN VaR, silent | validate levels>0 at matrix/generator boundary for return shift types | fail-fast + oracle | open |
| F2 | **P1** | all four fused kernels + apply | shift-ladder has no `else` → unhandled shift id leaves output **unwritten** (np.empty garbage) | `[SRC]` if/elif, no else | missing exhaustiveness guard | silent garbage shocks | reject unknown shift id at boundary (or write+raise) | fail-fast | open |
| F3 | **P2** | historical, fhs (parametric, bootstrap partial) | `horizon`/`total_scenarios` integrality unvalidated → float leaks to `total_scenarios`; latent cause of reported kernel error; crashes runner `np.empty` | `[RUN]` `total_scenarios=9.0` | no int/positive validation in `__post_init__`; parametric `int()` not persisted | wrong-tier crash, non-reproducible bug class | validate+coerce horizon∈ℤ≥1 and total_scenarios∈ℤ≥1 at construction | fail-fast | open |
| F4 | P2 | bootstrap_sbb, bootstrap_fhs | single-factor (K==1) fused kernel raises cryptic `TypeError` | `[RUN]` both RAISE | f64 inputs pinned F-only (D024 twins) | crash on 1-factor book | add `_F64_2D_C` overloads (D024 pattern) | dispatch matrix | open (noted D024) |
| F5 | P2 | matrix / generators | float32 matrix rejected only by deep Numba `TypeError`, not boundary error | `[RUN]` | RiskFactorMatrix accepts any floating dtype | poor diagnosability | reject non-f64 obs matrix at generator construction w/ clear message | fail-fast | open |
| F6 | P2 | parametric, sbb, boot-fhs | scenarios depend on `batch_size` (same seed) | `[RUN]` SBB diff-bs `False` | `default_rng(seed)` rebuilt per `iter_batches`; batch-partitioned draws | VaR varies with orchestration knob | D021 RNG-ownership redesign | reproducibility | open (D021) |
| F7 | P2 | base.py | `ScenarioSetBatch` has no `__post_init__` validation | `[SRC]` | contract informal | contract drift | add dtype/shape/contiguity assertions (debug) or one validator | contract | open |
| F8 | P3 | shocks/apply.py | per-batch `shock_scale` f32→f64 alloc in hot loop | `[SRC]` | promotion at exec boundary | minor alloc | hoist promotion to runner-build (once) | benchmark | open |
| F9 | P3 | ewma_volatility.py | raw `kernel_ewma_vol_ratios` unused by any generator; single-col latent defect | `[SRC/SIG]` | dead-ish public surface | none active | delete or overload when a caller appears | — | open |
| F10 | P3 | sbb_block_generator.py | stitch kernel scalar is `int32` while h-day kernels use `int64` | `[SIG]` | inconsistency | none observed | unify on int64 | — | open |
| F11 | P3 | historical (all) | empty required set → `(B,0)` silently; duplicate ids deduped silently | `[RUN]` | no edge guard | ambiguous empties | decide + document; fail-fast on empty if required | fail-fast | open |
| F0 | — | historical | reported float64 index **not reproducible / not D024** | §0 | see F3 | none in this tree | none (see F3) | — | closed |

---

## Answers to the 16 required questions

1. **What is SimLab?** A streaming scenario-generation + shock-application layer:
   generators emit `(B,Kreq)` f32 shock batches; `apply_shocks` reconstructs f64
   shocked levels; `SimlabRunner` is a full-reval oracle consumer. It is a
   boundary+kernel package, not a config framework.
2. **Canonical contracts?** §C. int32 ids/uix, int8 shifts, f32 shocks/scale, f64
   levels/PnL, F-shocks/C-out layout, int64 kernel scalars.
3. **All generators?** Historical, FilteredHistorical, Parametric (4 factories),
   StationaryBlockBootstrap, FilteredBootstrap (§F).
4. **Is `ScenarioGenerator` sufficient?** Yes; do not redesign. The gap is a
   shared construction-time validation helper, not the protocol (F3/F7).
5. **Where does validation occur?** matrix/generator/`build_shock_universe`/
   `validate_apply_shocks_buffers`/proxy `__post_init__`/smoke-tests. Gaps: F1,
   F2, F3, F7, F11.
6. **Where do Python reprs become typed arrays?** At generator construction
   (matrix) and at `iter_batches` (int64 scalar wrapping + F32/F buffer alloc);
   `apply` promotes scale f32→f64.
7. **All kernel signatures?** §D (captured `[SIG]`).
8. **Are all kernel indices integral?** Yes in this tree (§0, §D). The reported
   float index does not exist here.
9. **C/F layout contracts correct?** Yes, and matrix normalizes to F. D024 added
   no widening. Gaps are the F4 twins (missing C overloads), not incorrectness.
10. **Degenerate dimensions covered?** Historical/FHS yes (D024). Bootstrap twins
    **no** (F4). Single-row batch yes.
11. **Proxying correct?** Yes (§G).
12. **Shock application correct?** Yes for handled shift types; **F1/F2** break
    correctness for non-positive levels and unhandled shift ids.
13. **RNG reproducible?** Deterministic per `(seed, batch_size)`; **batch-size
    dependent** and not cross-generator-independent (F6/D021).
14. **Independently validated?** Partly (§H); FHS-composed, apply-shocks,
    shock-universe oracles missing.
15. **Remaining defects?** F1–F11 (§L); F0 closed.
16. **Justified architecture changes?** None new. No config/registry/factory seam
    is justified until F1–F3 (and F7) close. The package is **not** internally
    coherent on fail-fast grounds until then.


**Update (2026-09-28):** `riskquant.simlab.runner.py` (and the contained `SimlabRunner`)
have been removed from the code base as they were a legacy path.

---

## Remediation sequence (grouped by concern; smallest-first; not yet executed)

**R1 — Kernel/boundary correctness (P1). One concern.**
- F2: reject unknown shift id at the boundary (generator or shock-universe) so no
  kernel can silently leave output unwritten. Smallest: validate
  `required_shift_type_id ∈ {LOG,SIMPLE,DIFFERENCE}` (and NO_SHIFT policy) at
  `build_shock_universe`. Test: fail-fast on an out-of-set id.
- F1: validate `level > 0` for return-based shift types at the matrix/generator
  boundary (fail loud, not NaN). Test: non-positive level raises; oracle unchanged.

**R2 — Construction-time contract (P2). One concern.**
- F3: validate+coerce `horizon ∈ ℤ≥1` and `total_scenarios ∈ ℤ≥1` in a shared
  helper applied by every generator `__post_init__` (persist the coercion).
  Closes the reported error class. Test: `horizon=1.0` raises; `total_scenarios`
  is int.
- F5: reject non-float64 observation matrices at generator construction with a
  clear message. F7: add `ScenarioSetBatch` debug-assert validation.

**R3 — Layout completeness (P2). One concern (defer per current directive).**
- F4: add `_F64_2D_C` overloads to the two bootstrap fused kernels (D024 pattern).
  Test: single-factor dispatch matrix for both bootstrap generators.

**R4 — RNG methodology (P2). One concern (D021).**
- F6: generator-owned `np.random.Generator` at construction; remove batch-size
  dependence. Separate, larger change; keep for its own increment.

**R5 — Hygiene (P3).** F8 (hoist scale promotion), F9 (dead raw EWMA), F10 (int64
scalar), F11 (empty-required policy). Batched only when a neighbouring concern is
already open; never as standalone churn.

**Do not** begin the simulation-model configuration seam until R1–R2 (and R3)
land: the package is not fail-fast-coherent until then.
```
