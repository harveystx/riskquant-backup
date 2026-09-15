# RISKQUANT — SEVENTH AUDIT PASS

## Compiled Sensitivity Book, Taylor P&L, Chunking, VaR / ES, and Final Risk-Number Integrity

**Status:** AUDIT ONLY — no code modified, no patches, no remediation plan.
**Seam:** `sensitivity_pnl.{compile, engine, contracts, conventions, scheduling, grid}` · `sensitivity_pnl.kernels.{displacement, gather, taylor}` · `scenario_metrics.*` · `solutions.sensitivity_pnl.{pipeline, calculator, market}`.
**Date:** 2026-09-15

---

## Evidence basis

Source read from the project dump (`context_riskquant_all.txt`, 5.55 MB, this session). Every production symbol on this seam was re-implemented verbatim in NumPy — `compile_sensitivity_book` including `_compile_cross_block`, `_collapse_duplicates`, `_csr_offsets` and the quote-base folding; `kernel_build_displacement`; `kernel_taylor_pnl` including its four-arm seed and the 4-wide / 2-wide entry blocking; `row_costs` and `balanced_chunk_offsets`; `_compute_lower_tail_parameters` and `_kernel_lower_tail_var_es`.

The independent oracle is plain arithmetic over raw semantic objects only — position identity, factor identity, coefficient, base level, shocked level, convention, scenario values. It uses no `SensitivityBookSpec`, no `FactorAxis`, no production CSR, no production displacement helper, no production kernel, no production quantile or ES helper.

One probe-design error was found and corrected mid-pass and is reported as a finding in its own right (§14, degenerate case): my first Z-01 propagation probe returned a 0.00 % error, and the reason turned out to be a genuine mathematical property of the bilinear payoff rather than a harness fault.

---

## 1. Executive Verdict

### Is the final engine mathematically correct?

**Yes.** This is the cleanest module in seven passes, and it is not close.

The CSR compiler collapses every identifier into one coherent index space — the active-factor axis — and `searchsorted_or_raise` guards each of the five translations into it. Cross terms are canonicalised to `i < j` by `min`/`max`, and `i == j` is rejected with the exact reason ("accepting it here would double-count against the 1/2 factor the Taylor kernel applies to the diagonal block"). Duplicates raise by default. The Taylor kernel's four-arm seed, its 4-wide diagonal block and its 2-wide cross block reproduce the independent oracle at every `nnz` in {0,1,2,3,4,5,8,9,12} and on a cross-only book. Chunking is bit-identical across 1, 2, 3, 4, 8, 16 and 64 chunks. Batch size is invariant. The scenario axis carries asymmetric markers `[1, 10, 100, 1000, 10000]` through unchanged. VaR and ES match a hand-derived empirical tail geometry at five confidence levels, and the sign convention matches D015 exactly.

The boundary code is the best in the estate. `require_positive_levels_for_log_factors` does not merely check positivity — it names the precise failure it prevents, including that `np.sort` places NaN last so a poisoned catastrophe would be sorted *out* of the lower tail, and records the measured consequence (99 % VaR moving from −100 to −10). That is a boundary author who has thought about exactly the class of defect this audit has been chasing for six passes.

### Is the overall production number trustworthy?

**No** — and the reason is entirely upstream.

The final engine faithfully reproduces whatever it is given. That is the correct behaviour for a kernel tier, and it is also why the corruption already proven in Passes 3 and 5 arrives at the reported number undiminished. Measured through the production chain on a 100,000-scenario book:

| corruption | 99 % VaR error | tail-set overlap vs truth |
|---|---|---|
| **X-01** (wrong FX rate in the spot coefficient) | **+15.71 %** | 95.5 % |
| **Z-01 / Z-02** (3-cycle shock permutation) | **−57.07 %** | **0.0 %** |

Z-01/Z-02 is far worse than X-01 and worse than Pass 5 was able to show. X-01 is a multiplicative scale error that preserves the tail *ordering*; the shock permutation destroys tail identity outright — not one of the worst 1 % scenarios under truth is among the worst 1 % under the permuted book — and it understates VaR, which is the dangerous direction.

### Which defects belong to the final engine itself?

Two, both minor, neither reachable through the sanctioned pipeline:

* **Q-01 (LATENT)** — a SLOT factor axis whose integer keys happen to be a permutation of `[0, K)` compiles silently with the wrong mapping *if* `axis.relabel` is bypassed. `pipeline.resolve` applies it unconditionally, so this requires calling `compile_sensitivity_book` directly.
* **Q-02 (DOC)** — the contribution footing identity is documented as the caller's responsibility and is not verified. Deliberate, reasoned, and stated in the docstring.

### Which assurance mechanisms can detect the upstream defects?

**None.** `tests/sensitivity_pnl/` is a genuinely strong suite with a real Layer-2 independent oracle (`test_frozen_book_pnl_matches_hand_computation` — a test-owned frozen book with hand-computed P&L), and `tests/scenario_metrics/test_var_es.py` has a real independent sorted reference. But both take the book and the levels as *given*. They verify the transformation, not the inputs. No test anywhere in the repository derives an expected risk number from factor identity and base levels through the whole chain.

---

## 2. Final Production Data-Flow

```
BlockSensitivities (rf_id-keyed SoA)
        │  book_adapter.build_sensitivity_book_spec
        ▼
SensitivityBookSpec { sensitivities, cross_sensitivities, factor_meta, factor_axis }
        │  pipeline.resolve:  axis.relabel(frames, factor columns)     ◀── SLOT→row positions
        ▼
compile_sensitivity_book
   active_factor_ids  = np.unique(diag ∪ cross_i ∪ cross_j)            sorted-unique
   active_level_uix   = searchsorted_or_raise(level_universe, active)  ──► ValueError
   convention_ids, quote_base = _gather_factor_meta(active)            same axis
   position_uids      = np.unique(diag_pos ∪ cross_pos ∪ extra)        sorted-unique
   diag_row  = searchsorted_or_raise(position_uids, diag_position)     ──► ValueError
   diag_uix  = searchsorted_or_raise(active, diag_factor)              ──► ValueError
   lexsort (row, uix) → _collapse_duplicates(RAISE|SUM) → _csr_offsets
   d1 *= quote_base ; d2 *= quote_base²
   cross: min/max → i<j ; reject i==j ; lexsort (row, lo, hi) ; collapse ; h *= qb_i·qb_j
        ▼
CompiledSensitivityBook  (__post_init__: CSR monotonicity, terminator, uix ranges)
        │
        │  SensitivityPnlEngine.__post_init__
        │    out dtype/ndim/C-contiguity/rows-vs-positions
        │    derive_min_source_rows (from the ACTIVE-FACTOR axis, not nnz)
        │    validate_base_levels_for_ratio_factors  ──► ValueError on L0 ≤ 0 for REL/LOG
        │    _chunk_offsets = balanced_chunk_offsets(book, default_num_chunks(...))
        ▼
absorb_levels(shocked_levels, batch_len, scenario_offset, levels_offset)
   check_2d_batch_source: dtype, ndim, rows, batch_len, batch_size,
                          source_offset, scenario_offset, out_num_scenarios
   require_positive_levels_for_log_factors  ──► ValueError on L ≤ 0 for LOG
        ▼
kernel_build_displacement  (Ka, B) f32, computed f64 and narrowed on store
   ABSOLUTE  x = L − L0      RELATIVE  x = L/L0 − 1      LOG  x = ln(L/L0)
        ▼
kernel_taylor_pnl   prange over CHUNKS (contiguous row ranges, never reordered)
   PnL_m(s) = Σ_j (d1 + ½·d2·x_j)·x_j  +  Σ_{i<j} h_ij·x_i·x_j
   accumulate f64 in per-thread scratch, narrow-cast to f32 on scatter
        ▼
out (M, S_total) f32   row m ≡ position_uids[m]
        ▼
scenario_metrics.compute_var_es(group_value_matrix, confidence_level)
   _compute_lower_tail_parameters: α=1−C, m=αS, 16·eps snap, var_index, full, frac
   _kernel_lower_tail_var_es: quickselect to var_index; ES = (Σ_{s<full} + frac·VaR)/m
        ▼
VarEsResult.var  =  SIGNED lower-tail P&L order statistic  (negative = loss, D015)
VarEsResult.es   =  exact empirical ES with fractional boundary mass
```

---

## 3. Mathematical Contract

Stated before reading the arithmetic, then verified against it.

**Factor identity.** A factor is a global `rf_id` under DIRECT encoding, or a row position in `arange(K)` under SLOT. Inside the compiled book it is an index into `active_factor_ids` — the *active-factor axis* — and that single space indexes `active_level_uix`, `displacement_convention_ids`, `quote_base`, the displacement matrix rows, `factor_uix`, `cross_factor_uix_i` and `cross_factor_uix_j`.

**Position identity.** A position is an `int64` uid. Inside the book it is a row index into `position_uids`, sorted-unique, and `out[m] ≡ position_uids[m]`.

**Displacement.** `x_i` is the factor's move in its own declared `DisplacementConvention`, reconstructed per scenario from `(base_levels[n], shocked_levels[n, s])` where `n = active_level_uix[i]`.

**Coefficient convention.** `d1` is `∂V/∂x`, `d2` is `∂²V/∂x²`, `h_ij` is `∂²V/∂x_i∂x_j` for `i < j`, each already scaled by `quote_base**n`.

**The claimed object.**

$$
\Delta V_p(s) = \sum_i d_{p,i}\,x_i(s) + \tfrac12\sum_i g_{p,i}\,x_i(s)^2 + \sum_{i<j} h_{p,ij}\,x_i(s)\,x_j(s)
$$

**Units.** `x` carries the factor's own units under ABSOLUTE and is dimensionless under RELATIVE/LOG. `d1·x`, `½·g·x²` and `h·x_i·x_j` are all in the position's P&L currency. The `quote_base**n` folding reconciles a quoted coefficient to a natural-unit one.

**Where the ½ and the missing 2 live.** The kernel evaluates each diagonal term in Horner form `(d1 + 0.5·d2·x)·x`, so the ½ is applied **in the kernel**, exactly once, and `second_order` must be the raw gamma. Cross terms are evaluated as `h·x_i·x_j` with **no** ½ and **no** factor of 2, because the ½ from the Taylor expansion and the double-counting of `(i,j)`/`(j,i)` cancel — and the compiler enforces one entry per unordered pair, which is what makes that cancellation exact.

---

## 4. CSR / FactorAxis Identity Proof

### DIRECT versus SLOT

`FactorAxis.from_keys` produces exactly one of two states, never a mixture:

| | condition | `universe_ids` | `slot_map` |
|---|---|---|---|
| **DIRECT** | integer keys, sorted and unique | the keys themselves | `None` |
| **SLOT** | anything else — unsorted integers or labels | `arange(K)` | `{key: row}` |

The choice is a property of the whole axis. **A mixed book is structurally impossible**, which answers §6.4 and §6.8 without a probe: `universe_ids` is one array, built one way. Under SLOT, `pipeline.resolve` calls `axis.relabel(...)` on `sensitivities`, `factor_meta` and `cross_sensitivities` before compilation, rewriting key columns to row positions via `replace_strict` — which raises on an unmapped key.

### The identity proof

Every translation into the active-factor axis is a `searchsorted_or_raise`, and `active_factor_ids` is `np.unique(...)`, hence sorted-unique, which is that helper's precondition. The CSR block is `np.lexsort((diag_uix, diag_row))`, i.e. row-major with columns ascending within a row — the CSR invariant — followed by `_csr_offsets` from a `bincount` cumsum.

Executed with hostile sparse identifiers and unique markers — positions `{100, 7, 9000, 42}`, factors `{3, 70000, 90001}`:

```
position_uids     = [7, 42, 100, 9000]
active_factor_ids = [3, 70000, 90001]
active_level_uix  = [0, 1, 2]
position_offsets  = [0, 2, 3, 5, 6]
factor_uix        = [0, 1, 0, 0, 2, 1]
first_order       = [7.03, 7.7, 42.03, 100.03, 101.01, 900.7]

recovered {(position_uids[m], active_factor_ids[factor_uix[e]]) -> d1}
    == the input {(position, factor) -> d1}   ->  True
```

Every nonzero still denotes exactly `(position_i, factor_j)` after sorting, compaction and CSR construction.

### SLOT, with and without relabel

```
keys = [90001, 3, 70000]   slot_map = {90001:0, 3:1, 70000:2}   universe_ids = [0,1,2]

WITH relabel    : recovered == input                                    -> True
WITHOUT relabel : ValueError "factor_id -> level universe: 3 id(s) absent"  [FAIL-FAST]
```

The general case fails loud. **One window does not** — see Q-01, §20.

### Duplicates and cross canonicalisation

With `h_12 = 3` and `h_21 = 7` supplied as separate rows:

```
policy=RAISE : ValueError "1 duplicate (position_id, factor_id_i, factor_id_j) row(s)"
policy=SUM   : canonicalised to i<j, cross_second_order = [10.0]
i == j       : ValueError "cross: factor_id_i == factor_id_j"   [prevents the ½ double-count]
```

The asymmetry is not silently symmetrised and not silently halved. `RAISE` is the default.

---

## 5. Displacement Reconstruction Proof

From `kernel_build_displacement`, with the convention branch hoisted above the scenario loop and the terminal arm being LOG rather than a guard (unknown ids being rejected at compile time):

| convention | implemented | arithmetic |
|---|---|---|
| ABSOLUTE | `dst[s] = f32(src[s] − base)` | `x = L − L₀` |
| RELATIVE | `inv = 1/base; dst[s] = f32(src[s]·inv − 1)` | `x = L/L₀ − 1` |
| LOG | `inv = 1/base; dst[s] = f32(ln(src[s]·inv))` | `x = ln(L/L₀)` |

Arithmetic is f64, narrowed on store — deliberate, because `L − L₀` for a large level and a small shock is a cancelling subtraction.

### Pathological domain, gate versus arithmetic

| `L₀` | `L` | ABSOLUTE | RELATIVE | LOG |
|---|---|---|---|---|
| 100 | 110 | `10.0` | `0.1` | `0.0953…` |
| 100 | 0 | `−100.0` | `−1.0` | **rejected** (shocked-level gate) |
| 100 | −50 | `−150.0` | `−1.5` | **rejected** (shocked-level gate) |
| 0 | 10 | `10.0` | **rejected** (base gate) | **rejected** (base gate) |
| 0 | 0 | `0.0` | **rejected** | **rejected** |
| −100 | −50 | `50.0` | **rejected** | **rejected** |
| −100 | 50 | `150.0` | **rejected** | **rejected** |

Two gates, correctly scoped:

* `validate_base_levels_for_ratio_factors` — once per engine, over `RATIO_DISPLACEMENT_IDS = {RELATIVE, LOG}`, requiring a finite strictly-positive base. Its docstring names the reason: `error_model="numpy"` emits `inf` rather than raising, so this gate "is the only thing standing between a zero base level and a silent inf in the output."
* `require_positive_levels_for_log_factors` — per batch, over LOG only, scanning the shocked levels.

**RELATIVE deliberately gets the first gate and not the second.** That is correct, not an inconsistency: `L/L₀ − 1` is finite for any finite `L` once `L₀ > 0`, and clamping a below-−100 % return "would silently replace the caller's scenario with a different one". ABSOLUTE accepts a zero or negative base, which is right — an absolutely-shifted rate, spread or basis has no positivity requirement.

**Handling is consistent across conventions once each convention's mathematical domain is taken as the contract.** No defect.

---

## 6. First-Order / Second-Order / Cross-Term Proof

### The seed structure

`kernel_taylor_pnl` reuses one per-thread accumulator row across every position in a chunk, so the first write must be a **write**, not an accumulate. Four arms, all outside the scenario loop:

| arm | condition | seeds with |
|---|---|---|
| 1 | `hi − lo ≥ 4` | four diagonal entries, `lo += 4` |
| 2 | `lo < hi` | one diagonal entry, `lo += 1` |
| 3 | `qlo < qhi` | one cross entry, `qlo += 1` |
| 4 | else | zeros |

Verified against the independent oracle at every boundary:

```
nnz= 0  MATCH      nnz= 1  MATCH      nnz= 2  MATCH      nnz= 3  MATCH
nnz= 4  MATCH      nnz= 5  MATCH      nnz= 8  MATCH      nnz= 9  MATCH      nnz=12  MATCH
```

`nnz = 0` with an `extra_position_uids` row emits exact zeros. `nnz ∈ {1,2,3}` takes arm 2 and the remainder loop. `nnz = 4, 8, 12` exercise arm 1 plus whole 4-blocks. `nnz = 5, 9` exercise arm 1 plus a remainder.

### Cross-only book

A book with `nnz == 0` and one cross entry exercises arm 3, which only exists for this case:

```
engine = [6.0, 24.0, 54.0]      oracle h·(L_i − L0_i)·(L_j − L0_j) = [6.0, 24.0, 54.0]
```

This also confirms `derive_min_source_rows` is derived from the active-factor axis and not from `nnz` — a cross-only book has a fully populated factor axis and zero diagonal entries, and the engine sizes its scratch correctly.

### The ½

`(d1 + 0.5·d2·x)·x` expands to `d1·x + 0.5·d2·x²`. Applied once, in the kernel, never in the coefficient. The `nnz` sweep above used `d2 = 0.1·(i+1)` throughout, so a missing, doubled or misplaced ½ would have shown at every `nnz`.

### The missing factor of 2 on cross terms

Confirmed correct, and the compiler is what makes it correct: one entry per unordered pair with `i < j`, and `i == j` rejected. `h·x_i·x_j` is then the complete contribution of that pair.

---

## 7. Chunk Scheduler Proof

`balanced_chunk_offsets` partitions the row axis by cost prefix-sum `searchsorted`, emitting strictly-increasing bounds from 0 to M. `row_costs = diag + 2·cross + PER_ROW_SCATTER_COST`, which correctly weights a cross entry at twice a diagonal entry (two displacement row loads versus one). Colliding cuts are merged by `np.unique` rather than emitted as empty chunks.

**The scheduler cannot reorder.** Chunks are *contiguous ranges* of the row axis, so `prange` over chunks assigns each `out[m]` to exactly one chunk, and position order is preserved by construction. The kernel docstring is explicit that sorting positions by `nnz` is the opposite of a fix, and why. Reordering is therefore not merely restored before output — it never occurs.

Executed on a 17-position, 6-factor random book:

```
num_chunks= 1 -> [0, 17]                                        identical to 1-chunk: True
num_chunks= 2 -> [0, 6, 17]                                     identical: True
num_chunks= 3 -> [0, 4, 11, 17]                                 identical: True
num_chunks= 4 -> [0, 3, 6, 12, 17]                              identical: True
num_chunks= 8 -> [0, 2, 3, 4, 6, 10, 12, 14, 17]                identical: True
num_chunks=16 -> [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 17]  identical: True
num_chunks=64 -> [0, 1, …, 16, 17]  (capped at one chunk/position)     identical: True
```

Bit-identical, not merely close — because each position's accumulation order is fixed by its own CSR slice regardless of which chunk holds it. Chunking is execution policy only.

The repository pins this independently: `test_chunking_is_output_invariant`, `test_balanced_chunk_offsets_structure`, `test_chunk_offsets_must_{start_at_zero, be_strictly_increasing, cover_every_row}`, `test_engine_writes_only_its_own_rows`, and `test_thread_determinism` (bit-identical at 1 thread and at N).

---

## 8. Scenario-Axis Proof

`out[m][scenario_offset + s]` for `s ∈ [0, batch_len)`, written from `acc[s]` which came from `displacement[·][s]`, which came from `shocked_levels[n][levels_offset + s]`. One offset in, one offset out, no transpose.

Asymmetric markers, ABSOLUTE convention, `d1 = 1`, `base = 0`:

```
batch_size=1: [1.0, 10.0, 100.0, 1000.0, 10000.0]   matches full: True
batch_size=2: [1.0, 10.0, 100.0, 1000.0, 10000.0]   matches full: True
batch_size=3: [1.0, 10.0, 100.0, 1000.0, 10000.0]   matches full: True
batch_size=5: [1.0, 10.0, 100.0, 1000.0, 10000.0]   matches full: True
```

A reversal, transpose or off-by-one is impossible to miss against these markers, and none occurred — including at batch size 3 against 5 scenarios, where the final batch is partial.

`check_2d_batch_source` validates `batch_len`, `batch_size`, `source_offset`, `scenario_offset` and `out_num_scenarios` on every call. The scenario index cannot go out of range at this layer.

---

## 9. VaR Definition and Quantile Proof

### The estimator, derived from source

Let `α = 1 − C` and `m = α·S`. `_compute_lower_tail_parameters` snaps `m` to the nearest integer within `16·eps·max(1, |m|)` — guarding the canonical `0.99 × 100 → 1.0000000000000009` case — then:

```
full_tail_count   = floor(m)
fractional_weight = m − full_tail_count
var_index         = full_tail_count − 1   if fractional_weight == 0   (clamped at 0)
                    full_tail_count       otherwise
```

So **VaR is the `ceil(m)`-th smallest observation**: a pure **order statistic with no interpolation**, not a NumPy percentile method, not nearest-rank-with-averaging.

`_kernel_lower_tail_var_es` uses a Hoare-partition quickselect targeting `var_index`. The partition guarantee is exactly what the ES loop needs, and the two cases dovetail:

* `frac == 0`: `var_index = full − 1`, so `row[0 … var_index]` are the `full` smallest. ES sums `row[0 … full−1]` — the same set.
* `frac > 0`: `var_index = full`, so `row[0 … full−1]` are the `full` smallest. ES sums those plus `frac · row[var_index]`.

The quickselect target index is correct in both cases, and no full sort is performed.

### Sign

`var_out[g] = row[var_index]` — the **signed P&L order statistic**. `compute_var_es` applies no negation, and no caller in the chain does either.

**D015:** *"VaR is the lower P&L quantile itself (negative for a loss), NOT a positive loss magnitude."* Implementation and decision agree exactly.

*Correction to my own Pass-4 report:* I presented VaR as positive magnitudes (`488.54`). The engine returns `−488.56`. That was my presentation convention, not the engine's, and it should not be read as evidence of a negation anywhere in the code.

### Verified against marker P&L `[-100, -50, -10, 0, 20, 40, 100]`, S = 7

| C | `m = αS` | var_index | full | frac | VaR | ES |
|---|---|---|---|---|---|---|
| 0.99 | 0.0700 | 0 | 0 | 0.0700 | −100.0000 | −100.0000 |
| 0.95 | 0.3500 | 0 | 0 | 0.3500 | −100.0000 | −100.0000 |
| 0.90 | 0.7000 | 0 | 0 | 0.7000 | −100.0000 | −100.0000 |
| 0.75 | 1.7500 | 1 | 1 | 0.7500 | −50.0000 | −78.5714 |
| 0.50 | 3.5000 | 3 | 3 | 0.5000 | 0.0000 | −45.7143 |

Each independently derived: `ES(0.75) = (−100 + 0.75·(−50)) / 1.75 = −78.5714`. Exact.

### Small-sample and ties

```
N=1          VaR= -7.0000  ES= -7.0000
N=2          VaR= -5.0000  ES= -5.0000
N=3          VaR= -3.0000  ES= -3.0000
all equal    VaR=  4.0000  ES=  4.0000
ties at VaR  VaR= -9.0000  ES= -9.0000
```

When `m < 1` the tail is a fraction of the single worst observation and ES collapses to VaR — mathematically exact for the empirical measure, not a degenerate fallback. `var_index` can never exceed `S − 1`, since `m < S` for any `α < 1`.

---

## 10. ES Definition and Tail Proof

ES is **not** "the mean of P&Ls below VaR". It is the exact empirical tail integral:

$$
ES = \frac{1}{m}\left(\sum_{s<\lfloor m\rfloor} X_{(s)} \;+\; (m - \lfloor m\rfloor)\, X_{(\text{var\_index})}\right)
$$

— `full_tail_count` observations at unit weight plus a fractional residual on the boundary observation, normalised by `m` rather than by an integer count. This is the correct finite-sample ES for a tail mass that is not an integer number of observations, and it is what makes `ES(0.75) = −78.5714` rather than the naive `(−100 + −50)/2 = −75`.

**Tail selection is by order statistic, not by comparison.** There is no `<` versus `≤` question, because the tail is the first `full_tail_count` entries of a partitioned row plus a weighted boundary. Ties are therefore handled by position, deterministically, and the repository pins this with `test_quickselect_matches_reference_with_ties` and `test_adversarial_orderings_match_reference`.

---

## 11. Streaming versus Full Metrics

There is **no separate streaming metrics path**. `compute_var_es` takes a fully materialised `(n_groups, n_scenarios)` matrix.

Streaming exists one layer down, in the P&L engine: `SensitivityPnlEngine.absorb_levels` / `absorb_displacement` fill one `(M, S_total)` grid batch by batch, and metrics run once on the completed grid. So the question "is the streaming quantile exact or approximate?" does not arise — **the quantile is always computed on the full sample**, and batch-size invariance of the P&L grid (verified in §8, and pinned by `test_batch_size_invariance`) is sufficient for metric equality.

This is a sound design choice and worth recording as such: no approximate streaming quantile, no reservoir sampling, no t-digest, and therefore no numerical observation to quantify.

---

## 12. Contribution / Decomposition Proof

`_compute_tail_contribution` defines the tail scenario set from `total_row`, argselects on it, and gathers **each component at those same scenario indices**. Therefore:

$$
\sum_k C^{VaR}_k = \sum_k \text{component}_k[s^\*] = \text{total}[s^\*] = VaR
$$

and likewise for ES with the same weights. The footing identity is **exact by construction**, provided `total_row` is the column sum of `component_matrix`.

`_validate_component_matrix_and_total` validates shape, dtype, contiguity and finiteness — and explicitly declines to verify the column-sum, with a stated reason:

> "The footing identity `sum_k contribution[k] == metric(total_row)` only holds when `total_row` is the column sum of `component_matrix`. That is the caller's responsibility (the total is frequently accumulated online, independently of any materialised component matrix), so it is documented rather than re-derived and checked here."

That is a deliberate, reasoned, documented contract — Euler-style additive contributions under a stated precondition. **Q-02** records that nothing verifies the precondition; it is not a defect.

---

## 13. Independent End-to-End Oracle

The adversarial fixture required by §30, run through the re-implemented production chain and compared against arithmetic that shares nothing with it.

**Bilinear spot, the already-proven identity, exercised through the actual compile → displacement → Taylor path:**

`V = Q·S·F`, `Q = −50`, `S₀ = 82.30`, `F₀ = 1.08`, factors `{2, 7}`, position `1020`, 8 lognormal scenarios.

```
engine = [12.996405, -10.019889, 11.290367, 69.100075, 37.443439, 111.529320, -4.568144, -139.818420]
truth  = [12.996404, -10.019890, 11.290367, 69.100071, 37.443439, 111.529325, -4.568144, -139.818427]
max|err| = 6.989e-06     max relative = 6.462e-08
```

`truth` is `Q·S·F − Q·S₀·F₀` — the closed form, never touching a coefficient. The residual is float32 *output storage* only; the accumulation is f64 throughout.

**Combined fixture** (§30 checklist): 4 positions with sparse/high ids `{7, 42, 100, 9000}`; 3 factors with sparse ids `{3, 70000, 90001}`; DIRECT and SLOT encodings both compiled and both recovered; first-order, second-order and cross terms; `nnz` from 0 to 12 crossing every entry-blocking boundary; cross-only book; 17-position book crossing chunk boundaries at seven chunk counts; asymmetric scenario markers; ties in the metric tests. Every one matched the independent oracle.

---

## 14. X-01 / Z-01 / Z-02 Corruption Propagation

100,000 scenarios, 99 % confidence, through the production chain.

### Case A — correct upstream

```
max |engine − closed form| = 5.478e-05        VaR99 = -488.56   ES99 = -562.20
tail-set overlap vs truth  = 100.0%
```

### Case B — X-01: the spot coefficient built from the wrong FX rate

```
VaR99 = -565.32   (+15.71 %)      ES99 = -651.76   (+15.93 %)
tail-set overlap vs truth = 95.5 %
```

A clean multiplicative overstatement. The tail *ordering* largely survives, because the spot factor dominates and its shock is correct. The engine did not amplify, damp, sign-flip or reorder it — it computed exactly the P&L the corrupted coefficient implies.

### Case C — Z-01 / Z-02: a 3-cycle shock permutation

Two positions, three factors (`SPOT_A`, `SPOT_B`, `FX`), vols 2.0 %, 1.8 %, 0.6 %, permuted so each factor receives another's shock process, rebased to its own level:

```
truth      VaR99 =  -931.39   ES99 = -1066.14
permuted   VaR99 =  -399.85   ES99 =  -458.63       VaR err = -57.07 %   ES err = -56.98 %
tail-set overlap vs truth = 0.0 %
```

**Not one of the worst 1 % scenarios under truth is among the worst 1 % under the permuted book.** This is categorically worse than X-01: tail identity is destroyed, not merely rescaled, and the error understates risk. Pass 5 established the mechanism; this is the first measurement of its consequence at the reported number.

### The degenerate case — a genuine negative finding

My first Case-C probe returned **0.00 %** error, and the reason is mathematical, not a harness fault.

For a single pure-bilinear position `V = Q·S·F`, second-order Taylor with the cross term reduces exactly to `Q·(S·F − S₀·F₀)`. Swapping the spot and FX *shock processes* leaves `S·F` unchanged, because the product of the two lognormal exponentials is commutative. So:

> **A spot↔FX shock permutation is undetectable on a single bilinear position, at any confidence level, by any metric.**

Detection requires at least three factors, or two positions that do not share the same factor pair. This is worth recording precisely because the shipped demo book is small: a permutation defect can be invisible on a toy fixture and catastrophic on a real one.

### Verdict on the final engine's role

In all three cases the engine consumed the corrupted state **faithfully**. It neither masked nor amplified nor sign-flipped anything beyond what the corrupted inputs themselves imply. That is the correct behaviour for a kernel tier under "kernels do dumb math", and it is precisely why upstream correctness is load-bearing.

---

## 15. Test-Suite Assurance Audit

### Coverage map

| Test file | Production symbols invoked | Layer | Oracle class |
|---|---|---|---|
| `test_golden_master_sensitivity_pnl.py` | `compile_sensitivity_book`, `CompiledSensitivityBook`, `SensitivityPnlEngine`, `balanced_chunk_offsets`, `allocate_pnl_grid` | compile + engine + scheduler | **Layer 1 structural invariants + Layer 2 independent hand-computed frozen book** |
| `unit/test_kernels.py` | `kernel_build_displacement`, `kernel_gather_displacement`, `kernel_taylor_pnl` | kernels | direct arithmetic |
| `unit/test_mathematical_properties.py` | real compiler + real one-shot engine, over randomly generated books | pipeline | **property-based** — linearity, negation, superposition, independence, zero-displacement |
| `unit/test_compile.py` | `compile_sensitivity_book` | CSR | structural |
| `unit/test_scheduling.py` | `row_costs`, `default_num_chunks`, `balanced_chunk_offsets` | scheduler | structural |
| `unit/test_batch_contract.py` | `check_2d_batch_source`, the level gates | boundary | structural |
| `unit/test_conventions.py` | `DisplacementConvention` | conventions | structural |
| `scenario_metrics/test_var_es.py` | `compute_var_es`, `_compute_lower_tail_parameters` | metrics | **independent sorted reference** (`_reference_var_es`, `_reference_tail_mass`) + property tests |
| `scenario_metrics/test_contribution.py` | `compute_var_es_contribution` | metrics | footing identity |
| `solutions/.../test_calculator.py`, `test_pipeline.py`, `test_market.py` | `SensitivityPnlCalculator`, `pipeline.resolve`, `FactorAxis` | solutions edge | structural + axis-mismatch |

### What is genuinely independent

Two real oracles, and they are real.

**`test_frozen_book_pnl_matches_hand_computation`** — the module docstring states it plainly: *"a tiny test-OWNED frozen book with hand-computed PnL. Six numbers, derivable on paper from the docstrings. Owned here, not sourced from a canonical fixture, so it cannot drift when the canonical book does."* That is an independent-implementation oracle for the engine.

**`test_randomized_against_independent_sorted_reference`** plus `_reference_var_es` — an independently written sorted reference, exercised on randomised inputs, adversarial orderings and ties.

Both are the genuine article, and they stand in sharp contrast to Y-02.

### Specific coverage questions

| § | Question | Answer |
|---|---|---|
| 32.4 | Can the fixtures trigger the known index-space defects? | **No** — every test supplies the book and the levels directly. X-01, Y-01, Z-01, Z-02 corrupt the inputs, and the engine's correct behaviour is to reproduce them. |
| 32.5 | DIRECT/SLOT exercised? | **Yes** — `test_market.py` exercises `relabel` on diagonal and cross columns; `test_conventions.py` covers axis construction. |
| 32.6 | Nontrivial CSR permutations? | **Yes** — `test_mathematical_properties.py` runs over randomly generated books rather than one fixture. |
| 32.7 | Chunk boundaries? | **Yes** — `test_chunking_is_output_invariant`, `test_entry_blocking_boundaries`, `test_balanced_chunks_equalise_skewed_work`, `test_thread_determinism`. |
| 32.8 | Ties? | **Yes** — `test_quickselect_matches_reference_with_ties`, `test_all_scenarios_equal`. |
| 32.9 | Sign conventions? | **Yes** — `test_es_is_not_above_var_for_lower_tail_pnl`, `test_translation_invariance`, `test_positive_homogeneity`, `test_monotonicity_under_elementwise_increase`. |
| 32.10 | Streaming path? | **Yes** — `test_batch_size_invariance`, `test_levels_path_equals_displacement_path`. |

### Assurance independence classification (§33)

| Reconciliation | Classification | Can it detect… |
|---|---|---|
| `test_frozen_book_pnl_matches_hand_computation` | **independent implementation oracle** | CSR corruption ✔, displacement errors ✔, Taylor errors ✔. X-01 ✘, Y-01 ✘, Z-01/Z-02 ✘ — book and levels are given. |
| `test_randomized_against_independent_sorted_reference` | **independent implementation oracle** | quantile sign ✔, tie handling ✔, ES geometry ✔. Upstream ✘. |
| `test_mathematical_properties.py` | **property-based, same implementation** | structural pipeline defects ✔. Absolute correctness ✘ by design — the docstring says so. |
| `tests/sensitivities/test_simulated_scenarios_and_revaluation.py` | **non-exercising** (Y-02, Pass 4) | nothing. |
| `tests/sensitivities/test_spot_sensitizer.py` | **shared-input, layout-degenerate** (Y-03) | would catch a wrong FX rate, but the fixture's `node_id 0` layout makes X-01 unreachable. |

**There is still no end-to-end oracle that derives an expected risk number from factor identity and base levels through the whole chain.** Every oracle in the repository begins downstream of where the defects live.

---

## 16. Z-03 Resolution

### The exact production chain

**Producer.** `SimLabRunner.run` allocates `shocked_levels_buf = np.empty((n_req_factors, batch_size), dtype=np.float64, order="C")` and passes `out_view = shocked_levels_buf[:, :current_b]` — a **column** slice.

**Flags.** For `n_req > 1` and `current_b < batch_size`, the slice has shape `(n_req, current_b)` with strides `(batch_size·8, 8)`. C-contiguity requires `(current_b·8, 8)`. Measured: `buf[:, :64]` → `C_CONTIGUOUS = True`; `buf[:, :58]` → `False`; `buf[:, :1]` → `False`.

**Consumer signature.** `_kernel_apply_shocks` pins `out` to `_F64_2D_C` in both registered overloads. No F-order or non-contiguous overload exists for `out`.

**Selected overload.** None — the dispatcher finds no matching definition and raises `TypeError`.

**Validation.** `validate_apply_shocks_buffers` would reject it, but the runner calls it under `if first_batch` only. The partial final batch is therefore never pre-validated and fails at dispatch instead.

**Outcome:** loud and deterministic; the run aborts whenever `total_scenarios % batch_size != 0`.

### Scope

`SimLabRunner` is the **full-revaluation** runner. D016 states it explicitly: *"`SimlabRunner` is a full-revaluation runner requiring a `ScenarioGenerator` and is not reusable for the sensitivity path, but its constituents are."* D018 makes full revaluation an oracle, not the risk path.

The sensitivity engine takes the opposite approach by design. `absorb_levels` accepts a full-width `(N_req, S)` matrix plus a `levels_offset`, and the docstring gives the reason: *"a column slice would not be C-contiguous and would break the pinned kernel signature."* The sensitivity demo likewise allocates a fresh C-contiguous `np.empty` per batch rather than slicing.

### Classification

**Z-03 → CONFIRMED, scope NARROWED.** A real defect, loud, confined to the full-revaluation runner. **It is not on the production risk path**, and the sensitivity engine demonstrates the correct pattern for the same problem.

---

## 17. Y-08 Resolution

Y-08 concerns `run_sensitivities(portfolio, uix_mapper, market_data, scenario=0)` in `riskquant/sensitivities/engine.py` — the *coefficient-evaluation* scenario column, which `_kernel_sensitize_spot` uses to index `prices` and `fx_spots_effective` under `boundscheck=False`.

The `sensitivity_pnl` engine's `check_2d_batch_source` validates `scenario_offset`, `batch_len`, `source_offset` and `out_num_scenarios` on every absorb call. But that is a **different index at a different layer**: it bounds the P&L grid's scenario axis, not the market-data column from which coefficients are read.

No higher boundary guarantees the coefficient scenario index. **Y-08 stands unchanged** — MEDIUM, CONFIRMED, owned by `sensitivities/engine.py`. Not duplicated here.

---

## 18. P-03 / P-04 / P-05 Interaction

**P-03 — upstream-only, does not propagate as a distinct defect.** By the time the final engine is reached, `source_factor_ids`, `shock_scale` and `required_shift_type_ids` have been fully reduced into `shocked_levels`. The engine consumes `(base_levels, shocked_levels)` and a book keyed by `rf_id`; it has no view of, and no dependency on, the three-array positional binding. A P-03 misalignment therefore arrives here as a *corrupted levels matrix* — observationally identical to Z-01 — and is already measured in §14. **P-03 is upstream-only.**

**P-04 — not a numerical defect at this layer.** The reported number is computed correctly without any provenance: the book carries `factor_rf_ids` (the correct *identity*), and the engine needs nothing about where a shock originated. Provenance is required for explainability and audit — "why did WTI move?" — not for arithmetic. **Classified as an explainability gap, not a numerical defect**, exactly as §34 asks.

**P-05 — cannot reach this engine silently.** A non-finite proxy scale produces non-finite shocked levels. `validate_base_levels_for_ratio_factors` rejects non-finite bases for RELATIVE/LOG; `require_positive_levels_for_log_factors` rejects non-positive shocked levels for LOG; `_require_finite` rejects non-finite *coefficients* at compile. **Gap:** an ABSOLUTE-convention factor with a NaN *shocked level* is not gated — `x = L − L₀` is NaN, the Taylor accumulation is NaN, and `_validate_group_value_matrix` then raises `"group_value_matrix must contain only finite values"` at the metrics boundary. So it fails loud, at the last boundary rather than the first, and P-05's silent-NaN concern does **not** reach the risk number.

---

## 19. Critical Negative Findings

Attacked and cleared, with the reason each passed.

### CSR

**DIRECT encoding.** Sorted-unique integer keys index themselves; `searchsorted_or_raise` makes key→row the identity. Verified on sparse ids `{3, 70000, 90001}` with unique markers.
**SLOT encoding.** `relabel` rewrites key columns to row positions via `replace_strict`, which raises on an unmapped key. Verified: recovered mapping equals input.
**Mixed encoding.** Structurally impossible — `universe_ids` is one array built one way per axis.
**Sparse / high / non-zero-based ids.** All translations are by value through `searchsorted_or_raise`, never by positional assumption.
**Reordered factors and positions.** `np.unique` + `lexsort` canonicalise both axes; the recovered `(position, factor) → coefficient` map is identical to the input.
**Duplicate keys.** `RAISE` by default with a count and a remedy; `SUM` via `np.add.reduceat` over group starts. Quote-base folding is applied after collapse and is linear, so the two orders agree.
**Empty book.** `nnz = 0` with `extra_position_uids` emits exact zeros and correctly-shaped CSR terminators.

### Mathematics

**First order, second order, cross terms, the ½.** Verified against the independent oracle at nine `nnz` values crossing every entry-blocking boundary, plus a cross-only book.
**Mixed conventions.** ABSOLUTE, RELATIVE and LOG in one book, branch hoisted above the scenario loop.
**Zero displacement.** Exactly zero P&L — pinned independently by `test_zero_displacement_is_exactly_zero`.
**Negative displacement.** Handled; ABSOLUTE has no positivity requirement.
**Pathological base levels.** Seven `(L₀, L)` combinations across three conventions; every rejection traceable to a named gate, every acceptance mathematically defined.

### Execution

**Chunking.** Bit-identical across seven chunk counts.
**Scheduling.** Contiguous row ranges only; cannot reorder positions.
**Scenario ordering.** Asymmetric markers preserved at four batch sizes.
**Position ordering.** `out[m] ≡ position_uids[m]` throughout.
**Full versus streaming.** Metrics always run on the full grid; batch-size invariance is sufficient.
**Partial final batch.** Handled correctly here via `levels_offset` — the pattern that breaks `SimLabRunner` (Z-03) is explicitly avoided.

### Metrics

**VaR sign.** Signed lower-tail order statistic, matching D015 exactly.
**Quantile definition.** `ceil(m)`-th order statistic, no interpolation, with a documented `16·eps` snap for binary-representation noise.
**Ties.** Resolved by position through the quickselect partition; deterministic.
**ES tail.** Exact empirical integral with fractional boundary mass, normalised by `m` — not a naive mean below VaR.
**Small samples.** `N ∈ {1, 2, 3}`, all-equal, ties-at-VaR all produce mathematically defined results.
**Contribution identities.** Additive by construction under the documented precondition.

### Constitutional

No strings, no objects, no Pandas in any numerical path. Polars confined to `compile.py`'s edge resolution and the solutions tier. Canonical `riskquant.jit` decorators and aliases throughout — including `_F64_2D_C_RO` and `_F64_1D_C_RO` on the metrics kernels' read-only inputs, which is the `_RO` usage Pass 1 found missing in `valuation`. `stable_math_parallel_njit` on the metrics kernels, correctly distinguished from the ordinary `parallel_njit` used elsewhere. Zero allocation in every hot path; scratch is engine-owned and pre-sized. The chunk scheduler is execution policy, computed once per engine, never on the hot path.

The kernel documentation records measured alternatives that were built and rejected — scenario-tiled register blocking at 0.21–0.30×, split linear/quadratic CSR at 0.58–0.90×, position sorting by `nnz` at 0.17× — with instructions to re-run the throughput script before reintroducing either. This is the empirical-benchmarking discipline the estate claims, actually practised.

---

## 20. Finding Matrix

| ID | Severity | Category | Status | File | Symbol | Finding | Why it matters | Evidence | Reachability | Owner |
|---|---|---|---|---|---|---|---|---|---|---|
| **Q-01** | MEDIUM | B | **LATENT** | `sensitivity_pnl/compile.py`, `solutions/.../market.py` | `compile_sensitivity_book`, `FactorAxis.relabel` | A SLOT axis whose integer keys are a permutation of `[0, K)` compiles **silently with the wrong factor mapping** if `relabel` is bypassed. Every other SLOT layout raises. | The only silent mismapping window in the final engine. | Probe: keys `[2,0,1]`, unrelabelled ids `[2,0,1]` → accepted, `factor_uix = [2,0,1]` instead of `[0,1,2]`. Contrast keys `[90001,3,70000]` → `ValueError`. | Requires calling `compile_sensitivity_book` directly. `pipeline.resolve` applies `relabel` unconditionally to all three frames. | Compiler / solutions edge |
| **Q-02** | LOW | E | **DOC** | `scenario_metrics/_validate.py` | `_validate_component_matrix_and_total` | The footing identity `Σ_k C_k == metric(total_row)` holds only if `total_row` is the column sum of `component_matrix`; this is documented as the caller's responsibility and not verified. | A caller supplying an inconsistent total gets contributions that do not foot, with no error. | Source: the docstring states the precondition and the reason for not checking it. | Caller-supplied inputs only. | Metrics boundary |
| **Z-03** | HIGH | C | **CONFIRMED — NARROWED** | `simlab/runner.py` | `SimLabRunner.run` | Resolved exactly (§16). Partial final batch → non-C-contiguous `out_view` → `TypeError` at Numba dispatch. Loud. **Confined to the full-revaluation runner; not on the production risk path.** | Blocks any full-reval run whose scenario count is not a multiple of `batch_size`. | Measured flags + pinned `_F64_2D_C` signature + `first_batch`-only validation. | `total_scenarios % batch_size != 0`. | SimLab runner |
| **Y-08** | MEDIUM | C | **CONFIRMED — unchanged** | `sensitivities/engine.py` | `run_sensitivities(scenario=…)` | The coefficient-evaluation scenario index is unguarded. The P&L engine's `check_2d_batch_source` bounds a **different** index at a different layer and does not cover it. | An out-of-range coefficient scenario reads out of bounds under `boundscheck=False`. | §17. | Caller-supplied. | Sensitivity engine |
| **Y-07b** | MEDIUM | C | **LATENT — unchanged** | `sensitivities/book_adapter.py` | `displacement_convention` | Now with exact consequence: the value lands in `displacement_convention_ids` and selects the `kernel_build_displacement` branch. Declaring LOG over absolute coefficients silently rebases every displacement. | Silent mis-scaling of the entire book. | §5 + Pass 4 probe (+18.89 % at a −30 % move). | Caller action only. | Book adapter |
| **P-03** | HIGH | C | **UPSTREAM-ONLY** | `simlab/runner.py`, `scenarios/base.py` | — | Does not propagate to this engine as a distinct defect; its effect arrives as corrupted levels, measured as Case C in §14. | — | §18. | — | Upstream |
| **P-04** | MEDIUM | D | **EXPLAINABILITY ONLY** | `scenarios/base.py` | `ScenarioSetBatch` | The reported number is computed correctly without provenance. Reclassified per §34: an audit/explainability gap, not a numerical defect. | — | §18. | — | Batch contract |
| **P-05** | LOW | C | **CANNOT REACH SILENTLY** | `proxy_topology`, resolver | `vol_scaling_factor` | A NaN scale produces NaN levels; ABSOLUTE factors are not gated at the engine, but `_validate_group_value_matrix` rejects a non-finite P&L grid at the metrics boundary. Fails loud, late. | — | §18. | — | Config compile |
| **D015** | — | A | **NO DEFECT** | `scenario_metrics/api/var_es.py` | `compute_var_es` | Implementation and decision agree exactly: VaR is the signed lower P&L quantile, no negation anywhere. | — | §9. | — | — |
| **X-01 / Y-01 / Z-01 / Z-02** | **CRITICAL** | B | **CONFIRMED — impact quantified** | carried | — | Not new findings. §14 measures their effect at the reported number: X-01 → VaR +15.71 %, tail overlap 95.5 %; Z-01/Z-02 3-cycle → VaR −57.07 %, tail overlap **0 %**. | Z-01/Z-02 destroys tail identity and understates risk. | §14. | Established in Passes 3 and 5. | Upstream |

---

## 21. Cross-Pass Reconciliation

### The complete production chain

| Arrow | Verdict | Pass |
|---|---|---|
| static configuration → proxy resolution | **proven correct** | 6 |
| proxy resolution → `source_factor_ids` / `shock_scale` row binding | **proven correct** | 6 |
| positional crossing into `iter_batches` | **proven defective** (P-03) | 6 |
| provenance → downstream | **proven defective** (P-04), explainability only | 6, 7 |
| shock universe / compute plan construction | **proven correct** | 5, 6 |
| compute plan → generator shock columns | **proven defective** (Z-01, Z-02) | 5 |
| shift-type application | **proven correct** | 5 |
| `required_factor_ids` → `flat_levels` rows | **proven correct** | 5 |
| scenario-axis semantics | **proven correct** | 5, 7 |
| `flat_levels` → `MarketDataMatrix` (FX) | **proven defective** (X-01) | 3 |
| `flat_levels` → `MarketDataMatrix` (scalar rows) | **proven defective** (Y-01) | 3, 4 |
| `MarketDataMatrix` → valuation kernels | **proven correct** | 1, 2 |
| `MarketDataMatrix` → sensitizer | **proven correct** | 4 |
| sensitizer → `BlockSensitivities` | **proven correct** | 4 |
| `BlockSensitivities` → `SensitivityBookSpec` | **proven correct**; latent convention hazard (Y-07b) | 4, 7 |
| `SensitivityBookSpec` → compiled CSR | **proven correct**; latent SLOT window (Q-01) | 7 |
| compiled CSR → Taylor P&L | **proven correct** | 7 |
| Taylor P&L → chunking / scenario assembly | **proven correct** | 7 |
| scenario P&L → VaR / ES | **proven correct** | 7 |
| VaR / ES → contribution | **proven correct** under a documented precondition (Q-02) | 7 |
| assurance across the whole chain | **proven defective** (Y-02, Y-03) | 4, 5, 6, 7 |

No production arrow is unclassified.

### Changes made by Pass 7

**New root causes:** Q-01 (latent), Q-02 (documentation).
**New downstream impact:** §14 — the first quantification of X-01 and Z-01/Z-02 at the reported risk number, and the discovery that Z-01/Z-02 is far more destructive than X-01.
**Architectural clarification:** Z-03 narrowed to the full-revaluation branch; Y-08 confirmed as a distinct layer and not duplicated; P-03 classified upstream-only; P-04 reclassified as explainability; P-05 shown to fail loud at the metrics boundary; D015 confirmed.
**Self-correction:** my Pass-4 report presented VaR as positive magnitudes; the engine returns signed quantiles.

---

## 22. Final Mathematical Theorem

> **Theorem.** Let `B` be a `CompiledSensitivityBook` produced by `compile_sensitivity_book` from a `SensitivityBookSpec` whose factor keys were resolved through `FactorAxis` and `pipeline.resolve`. Let `(base_levels, shocked_levels)` satisfy the engine's admitted domain — finite, correctly shaped, strictly positive base for RATIO-convention factors, strictly positive shocked levels for LOG-convention factors. Then for every position `m` and scenario `s`, `SensitivityPnlEngine` computes
>
> $$\text{out}[m,s] \;=\; \sum_{j\in \text{row}(m)} \left( d^{(1)}_{mj} x_j(s) + \tfrac12 d^{(2)}_{mj} x_j(s)^2 \right) \;+\; \sum_{(i,j)\in \text{cross}(m),\, i<j} h_{mij}\, x_i(s)\, x_j(s)$$
>
> with `x` reconstructed per factor in its declared `DisplacementConvention`, **exactly**, up to:
>
> * float32 representation of the displacement matrix (~6 × 10⁻⁸ relative, three to five orders below the Taylor truncation error it feeds);
> * float32 narrowing of the output grid (~6 × 10⁻⁸ relative; accumulation is f64);
>
> and **independently of** batch size, chunk count, thread count, position order, factor order and CSR entry order. Measured: max relative error 6.46 × 10⁻⁸ against a closed form; bit-identical across 1–64 chunks; bit-identical across 1 and N threads (pinned by `test_thread_determinism`).
>
> **Corollary.** For a payoff that is exactly bilinear in two factors, second-order Taylor with the cross term is not an approximation: the engine reproduces `Q(SF − S₀F₀)` to float32 output precision.

### Upstream assumptions currently violated

The theorem's hypothesis requires a *semantically correct* book and a *semantically correct* shocked state. Four established defects violate it:

1. **X-01** — `fx_spots_effective` built across two index spaces, so `d1` for the spot factor carries the wrong FX rate. Measured: VaR +15.71 %.
2. **Y-01** — FX-node rows of `prices` never written, so `∂PV/∂FX = qty · prices[·]` can be uninitialised memory for an FX pair held as a position.
3. **Z-01** — `ParametricGenerator` never applies `inverse_permutation`.
4. **Z-02** — the historical family applies it in the wrong index space.

(3) and (4) violate the shocked-state hypothesis. Measured on a 3-cycle: VaR −57.07 %, tail-set overlap 0 %.

**The theorem holds. Its hypothesis does not.**

---

## 23. Final Audit Classification

### Proven correct
CSR identity under DIRECT and SLOT, sparse and high ids, reordered factors and positions. Cross canonicalisation to `i < j` and `i == j` rejection. Duplicate policy, both arms. Quote-base folding. All three displacement conventions and their domain gates. The Taylor form including the ½ placement and the cross-term coefficient convention, at every entry-blocking boundary and on a cross-only book. Chunking invariance, bit-identical. Scenario-axis preservation. Position-order preservation. Batch-size invariance. VaR as the `ceil(m)`-th order statistic with the `16·eps` snap. ES as the exact empirical tail integral with fractional boundary mass. Tie handling. Contribution footing under its documented precondition. D015 sign convention. Constitutional compliance including `_RO` aliases on the metrics kernels.

### Proven defective
**Z-03** (HIGH) — confined to the full-revaluation runner, loud, off the production risk path.
**Y-08** (MEDIUM) — coefficient-evaluation scenario index unguarded, distinct layer, unchanged.
Carried CRITICAL: **X-01**, **Y-01**, **Z-01**, **Z-02** — all upstream, all now quantified at the reported number.
Carried CRITICAL/HIGH assurance: **Y-02**, **Y-03**.

### Conditional
None.

### Latent
**Q-01** (MEDIUM) — SLOT permutation-of-`[0,K)` silent mismapping when `relabel` is bypassed.
**Y-07b** (MEDIUM) — caller-settable displacement convention over fixed-coordinate coefficients.
**Z-04** (MEDIUM) — unknown shift id leaves a row unwritten if `ScenarioSetBatch` is constructed directly.

### Documentation only
**Q-02** — contribution footing precondition documented but unverified.
**P-07** (Pass 6) — unused source-only dedup in `ProxyResolution`.
**D013** — resolved in implementation, unrecorded in the decision log.

### Numerical observation
**float32 displacement and output storage** — ~6 × 10⁻⁸ relative, deliberate, measured (1.35× end-to-end), and three to five orders below the Taylor truncation error. Not a defect; the reasoning is recorded in the kernel docstrings.
**A&S normal CDF** (Pass 1, D020) — carried; deep-tail relative error up to 2.9 %, near-ATM ~10⁻⁶. Accepted risk, quantification incomplete in the decision record.

### No defect
**D015 / VaR sign.** **Contribution additivity** under its stated precondition. **RELATIVE convention exempt from the per-scenario positivity gate.** **ABSOLUTE convention accepting non-positive base levels.** **No streaming quantile** — the full sample is always materialised. **Chunk scheduler cannot reorder.** **Mixed DIRECT/SLOT** — structurally impossible.

### Closed
**P-01** (Pass 6) — proxy row binding. **P-06** (Pass 6) — proxy chains and cycles. **Y-07** (Pass 5) — shock convention independent of displacement convention. **V-05** (Pass 3) — cashflow reachability as buildout state. **W-03 / X-10** on the sensitivity path (Pass 4). **P-03** — upstream-only. **P-04** — explainability, not numerical. **P-05** — fails loud at the metrics boundary.

---

## 24. Final Production-Path Verdict

### Can the reported VaR/ES be considered mathematically trustworthy?

**No — but the reason is now precisely located, and it is not the engine.**

1. **Is the final engine mathematically correct?** Yes. Proven to float32 output precision, invariant to every execution parameter, with the best boundary code and the only genuinely independent oracles in the estate.

2. **Is the overall production number trustworthy?** No.

3. **Which upstream defects invalidate it?** X-01 (wrong FX rate → VaR +15.71 %), Z-01 and Z-02 (shock permutation → VaR −57.07 %, tail overlap 0 %), and Y-01 (uninitialised `prices` rows for FX-node positions).

4. **Which defects belong to the final engine itself?** Q-01 (latent, requires bypassing the pipeline) and Q-02 (documented contract). Neither affects a number produced through the sanctioned path.

5. **Which assurance mechanisms can detect them?** None. The two real oracles that exist — the hand-computed frozen book and the independent sorted VaR reference — both begin downstream of where the defects live. Every upstream fixture sits in an ascending-id layout, which is the one topology in which all three index-space defects vanish.

### The seven-pass shape

The estate's kernels are excellent and its containers are weak. Every defect found in seven passes lives at an index-space boundary between two subsystems — `[fx | scalar | disc | price]` misread as globally sorted (X-01), sorted-column position misread as shock-universe uix (Z-01, Z-02), uix-indexed permutation applied by sorted position (Z-02), a factor class partition never written (Y-01), a SLOT permutation of `[0,K)` (Q-01) — and every one of them is invisible to a fixture whose identifiers happen to be ascending.

### Verdict

> **AUDIT COMPLETE FOR THIS PRODUCTION PATH.**

Every arrow from static configuration to the reported VaR/ES is classified in §21. No material production seam remains unaudited. The remaining work is remediation and assurance, not investigation — and the single highest-value assurance artefact the estate lacks is an end-to-end oracle that derives an expected risk number from factor identity and base levels, exercised on a **non-ascending** identifier layout.
