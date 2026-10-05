# Handoff: position-chunked run path for `run_scenario_var_es`

Date: 2026-10-02
Repo: `riskquant`
Tier: SOLUTION (`riskquant.solutions.risk_run`), with small CORE additions
Status: not started. This file is the full brief for a new workstream.
Origin: the SPOT scaling benchmark (`scripts/demos/e2e/bench_spot_scaling.py`).

## 0. How to use this file

1. Read sections 1 to 4. They state the problem and the evidence.
2. Read section 5. It lists the rules the solution must obey.
3. Run work package WP0 (section 8) before any production change. WP0 answers
   the open questions in section 7. Several design choices depend on the answers.
4. Do not change `DECISIONS.md` without the owner's approval (CLAUDE.md).
   `DECISIONS.md` was not available when this file was written. Every
   decision ID below comes from code comments only. Check each ID against
   `DECISIONS.md` before you cite it.

Evidence labels used in this file:

* `[READ]` The author read it in the code. The location is given.
* `[DERIVED]` The author computed it from `[READ]` facts. It is not measured.
* `[MEASURED]` A tool recorded it. The source is given.
* `[UNVERIFIED]` The author believes it. A probe in section 7 must confirm it.

## 1. What problem this workstream solves

`run_scenario_var_es` holds the complete `(M, S)` scenario P&L grid in memory.
M is the number of positions. S is the number of scenarios. Memory grows with
M x S. A machine with enough RAM for the other inputs can still fail on the grid.

The benchmark brief asked for these cells:

* Sweep A: historical simulation, S = 520, M up to 1,000,000.
* Sweep B: SBB, S = 100,000, M up to 1,000,000.
* Sweep C: SBB, M = 50,000, S up to 1,000,000.

A 16 GiB machine can run only part of this grid. The benchmark skips the other
cells and labels them. The sentence "No change to production code. A
position-chunked run path in `solutions.risk_run` would make all 15 cells
feasible" meant this. The limit is in the run, not in the benchmark. Only a
change to the run lifts it. That change is this workstream.

Goal: add an operational option that processes the book in position chunks.
Peak memory must depend on the chunk size, not on M. The result must stay
bitwise identical to the unchunked run (section 5, R1 and R2).

## 2. Facts about the current run

All facts are `[READ]` from `src/riskquant/solutions/risk_run/api.py` unless
stated otherwise.

E1. `run_scenario_var_es` builds `shocked_levels` as one `(K, S)` float64
array. K is the number of factors on the run axis. The scenario stream is
consumed once, in batches that must tile `[0, S)` exactly once (PR-06).

E2. `allocate_pnl_grid` creates one `(M, S)` float32 array. It lives in
`src/riskquant/sensitivity_pnl/grid.py` as `PnlGrid`. Block order is
`[U_F | U_S | U_D]`. U_F is FULL_REVAL, U_S is TAYLOR/SUPPLIED and U_D is
TAYLOR/DERIVED.

E3. `_write_full_reval_block` loops over scenario batches. For each batch it
calls `_value_positions` for ALL FULL_REVAL positions. It writes
`vs - v0_full[:, None]` into the float32 block. The cast from float64 to
float32 happens in that assignment. The batch size limits the valuation
temporaries. It does not limit the grid.

E4. `_value_positions` allocates `rows` of shape `(total_trades, batch)`. It
then makes `rows[row_order]` and `np.add.reduceat(...)`. These are three
`(M, B)` float64 arrays, plus two more in `_write_full_reval_block`.

E5. After the grid, the run allocates `settlement_pnl`, an `(M, S)` float64
array. It fills it with `settlement_pnl[production_rows] = grid.pnl`. The
variables `grid` and `settlement_pnl` stay alive until the function returns.

E6. `convert_scenario_pnl_to_reporting_currency` is in
`src/riskquant/reporting/fx_conversion.py`. It allocates `rates`,
`translation`, one product temporary and `converted`. Each is `(M, S)`
float64. It also gathers FX rows for positions that need conversion.
`portfolio_pnl = converted.sum(axis=0)` is the portfolio row.

E7. The run builds `groups = np.vstack([converted_pnl, portfolio_pnl[None]])`.
This is another `(M + 1, S)` float64 copy.

E8. `compute_var_es` is in `src/riskquant/scenario_metrics/api/var_es.py`. It
allocates `scratchpad`, an `(M + 1, S)` float64 array, and runs a Numba kernel
from `scenario_metrics/_kernels/_tail_metrics.py`. It validates with
`np.isfinite(...).all()` first.

E9. The run returns only `(M + 1,)` VaR and `(M + 1,)` ES arrays. It returns
no `(M, S)` data. A chunked run needs no `(M, S)` output.

E10. A position is not always one valuation row. `_row_aggregation` maps
valuation rows to economic UIDs (D045, D055). A UID can own several rows.
`_value_positions` sums the rows of one UID in compiled row order.

E11. `IngestBatch.reorder` (`src/riskquant/portfolio/ingest/batch.py`) accepts
only a full permutation. It rejects an index array whose length differs from
`size`. No subset method exists.

E12. `compile_portfolio_from_ingest_batch` stamps every block with
`uix_mapper.fingerprint`. `run_valuation` raises when the portfolio
fingerprint differs from the valuation state fingerprint (AUD-02).

E13. `convert_scenario_pnl_to_reporting_currency` requires
`num_positions == plan.num_positions`. `FxConversionPlan` holds per-position
arrays and a tuple of settlement currency strings.

E14. `scripts/release/spot/certify_spot_production_scale.py` records stage
times. `[MEASURED]` in `spot_production_scale_certifcate.json` for M = 200,
S = 2,000 and filtered bootstrap. Wall time is 0.095 s. Valuation takes
0.009 s, D081 takes 0.012 s, VaR/ES takes 0.003 s and other work takes
0.069 s. Fixed costs dominate at this size. It is not a scale measurement.

## 3. Memory model

All figures in this section are `[DERIVED]` from E1 to E8. WP0 replaces them
with measurements.

Live arrays at the peak, per `(position, scenario)` cell:

| Run stage | Live bytes per cell |
|---|---|
| Float32 grid (E2) | 4 |
| Float64 settlement P&L (E5) | 8 |
| Inside D081: `rates`, `translation`, product temporary, `converted` (E6) | 32 |
| Inside D081: FX gather, fraction f of positions that need FX | 16 f |
| Peak inside D081 | 44 + 16 f |
| After D081: grid, settlement, converted, `groups` (E7) | 28 |
| Inside VaR/ES: the 28 above plus `scratchpad` (E8) | 36 to 37 |

The peak is about 44 to 60 bytes per cell. The benchmark uses 52. The
benchmark replaces 52 with a measured value after the first cell of at least
10 million cells. It never goes below 28, the minimum live at VaR/ES time.

Cells of the original request, at 52 bytes per cell (grid-bound part only):

| Cell | M x S | GiB at 52 B/cell |
|---|---|---|
| A, M = 100k | 5.2e7 | 2.5 |
| A, M = 1M | 5.2e8 | 25 |
| B, M = 1k | 1e8 | 4.8 |
| B, M = 10k | 1e9 | 48 |
| B, M = 100k | 1e10 | 484 |
| B, M = 1M | 1e11 | 4,843 |
| C, S = 10k | 5e8 | 24 |
| C, S = 100k | 5e9 | 242 |
| C, S = 1M | 5e10 | 2,421 |

A machine with R GiB of RAM runs at most (R - 2) x 2^30 / 52 cells. This gives
2.9e8 cells for 16 GiB, 1.3e9 for 64 GiB and 5.2e9 for 256 GiB.

With position chunks of Mc positions the peak becomes:

```
peak ~ fixed + 8 K S + 52 Mc S + 8 S (portfolio accumulator) + O(M) + 40 Mc B
```

`8 K S` is `shocked_levels`. `40 Mc B` is the valuation temporaries for batch
size B. `O(M)` is the per-position inputs and the `(M + 1,)` outputs.

Choose Mc = floor((budget - fixed - 8 K S) / (52 S)). With an 8 GiB budget,
2 GiB fixed and K = 55:

| Cell | Mc | Chunks |
|---|---|---|
| B, M = 1M, S = 100k | 1,230 | 814 |
| C, M = 50k, S = 1M | 115 | 435 |
| A, M = 1M, S = 520 | 238,247 | 5 |

Time is not bounded by memory. Time is about `ns_per_cell x M x S`. E14 gives
60 ns per cell as an order of magnitude (valuation + D081 + VaR/ES). At that
rate the cell with 1e11 cells needs about 1.7 hours. This is illustrative.
Run the benchmark on the target machine to get the real rate.

A second bound remains after chunking: `shocked_levels` needs `8 K S` bytes.
For K = 55 and S = 1M it is 0.41 GiB. For K = 2,000 and S = 1M it is 15 GiB.
Section 11 lists this as out of scope.

## 4. What can and cannot be chunked

Position-independent (safe to chunk):

* Valuation of one position at one scenario.
* The float64 difference `V(x_s) - V(x_0)` and its float32 cast.
* The D081 conversion of one position (row-wise formula).
* VaR and ES of one position. Each row is an independent empirical variable.

Cross-position (need care):

* The portfolio P&L row is the sum over all positions, per scenario. The
  floating-point result depends on the order of the additions.
* The run identity (`manifest`, `identity`). It must not depend on the chunk size.
* The route, the FX plan and the row aggregation. They are built once for the
  whole book and then sliced.
* `shocked_levels`. The run computes it once. All chunks read it.

Loop order. The scenario batch loop must be INSIDE the position chunk loop.
A position's VaR needs all S scenarios of that position. A chunk can release
its grid only after it has seen every scenario batch. The reverse order would
keep all chunks alive.

## 5. Requirements

R1. Invariance. For any chunk size, the identity, every per-position VaR and
every per-position ES are bitwise equal to the unchunked run. This extends
D041 (batch size) to chunk size. The chunk size stays out of the identity.

R2. Portfolio row. The portfolio VaR and ES are bitwise equal to the unchunked
run. This needs the chunked sum to add rows in the same order as the current
`converted.sum(axis=0)`. Open question Q1 decides how. If the order cannot
be reproduced, use option R2-b in section 6. The owner must approve a new
canonical order, a new certification and a note in `DECISIONS.md`.

R3. Memory bound. Peak memory follows the formula in section 3. It does not
contain a term `M x S`. A test must measure it (WP5).

R4. Float32 grid semantics. Each chunk must quantise `dV` through float32 and
back to float64, as the current run does (PnlGrid contract).

R5. Coverage proof. The chunks must tile the sorted production UIDs exactly
once, in order. The run must raise when they do not. This mirrors PR-06.

R6. Whole UIDs. A chunk must contain all valuation rows of each of its UIDs
(E10). Row order inside a UID must equal the compiled order.

R7. Fail loud. A chunked run with a non-empty U_D or U_S raises in phase 1.
It does not fall back to the unchunked path. A silent fallback hides the
memory failure the user wanted to avoid.

R8. Tier model. CORE gets Numba/NumPy only. No strings in new core code. Use
`riskquant.jit` decorators. `__init__.py` stays empty. Public facade stays in
`solutions/risk_run/api.py`.

R9. Default behaviour. With the new option unset, the run takes the current
code path. All existing tests and certificates stay valid unchanged.

## 6. Proposed architecture (phase 1: FULL_REVAL only)

New operational option on `run_scenario_var_es`:

```
position_chunk_size: int | None = None
```

`None` keeps the current path (R9). A positive integer sets the maximum
number of economic positions per chunk. The option is operational. The
manifest does not contain it. Add a pure helper
`suggest_position_chunk_size(num_positions, num_scenarios, num_axis_factors,
budget_bytes)` that implements the Mc formula in section 3. The Excel terminal
can pass a budget. The run itself stays deterministic.

Per-run steps, in order:

1. Everything up to and including `shocked_levels` stays as it is.
   The identity, the route, the proxy operator and V0 do not change.
2. Plan the chunks. Cut the ascending economic UID axis into contiguous
   ranges of at most `position_chunk_size` UIDs. Prove coverage (R5).
3. For each chunk:
   1. Take the ingest rows of the chunk's UIDs, in original row order (P1).
   2. Compile the sub-portfolio with the SAME `UixMapper` (P2, Q2).
   3. For each scenario batch, build market data from `shocked_levels`, value
      the sub-portfolio, and write `dV` into the chunk's float32 grid.
   4. Widen to float64, convert with the sliced FX plan (P3), and compute
      per-position VaR/ES for the chunk's rows.
   5. Add the chunk's converted rows into the portfolio accumulator, row by
      row, in canonical order (P4, Q1).
4. Compute portfolio VaR/ES from the accumulator. Concatenate the
   per-position results in sorted UID order, then append the portfolio entry.
5. Build `RiskRun` as today.

New primitives:

P1. `IngestBatch.take(indices)`. Subset with order preserved, for all 18
parallel arrays and the scalar `as_of_ord32`. CORE, `portfolio/ingest/batch.py`.

P2. Sub-portfolio compile under the full mapper. Probe Q2 first. If
`compile_portfolio_from_ingest_batch` accepts a sub-batch with the full
mapper, no new code is needed. If it does not, add a mapper-preserving entry
point. A second option is to slice compiled blocks. It needs a slice method
for every block type (SpotBlock, VanillaForwardBlock, and later ones). It is
more code. Do not choose it without a measurement that justifies it.

P3. `FxConversionPlan` row slice. `fx_row_index` and `invert` slice directly.
`settlement_ccys` is a tuple of strings. Slice it as a tuple. The plan is
EDGE data used by a CORE function today. Keep that boundary as it is.

P4. Portfolio accumulation kernel. A Numba kernel with the `riskquant.jit`
decorator. Signature idea: `accumulate_rows(acc, rows)`, with `acc` shape
`(S,)` float64 and `rows` shape `(Mc, S)` float64. It adds rows in order.
No allocation (`out`-style argument).

P5. A chunk-coverage validator, next to the PR-06 batch-tiling check.

The reverse loop order is rejected. With scenario batches outside and
position chunks inside, every chunk's grid stays alive (section 4).

Option R2-a: reproduce the order of `converted.sum(axis=0)` exactly. Needs
Q1 to show that NumPy adds rows sequentially on axis 0.
Option R2-b: define a new canonical order and certify it. It changes the last
bits of portfolio VaR and ES in stored results. It needs owner approval.

## 7. Open questions and probes

Run each probe in WP0. Record the answer and the machine in this file.

Q1. How does NumPy sum `(M, S)` C-contiguous float64 on axis 0? `[UNVERIFIED]`
The author expects sequential row addition. Probe:

```python
import numpy as np
rng = np.random.default_rng(0)
a = rng.standard_normal((10_000, 520)) * 1e3
ref = a.sum(axis=0)
seq = a[0].copy()
for i in range(1, a.shape[0]):
    seq += a[i]
chunked = np.zeros(520)
for k in range(0, 10_000, 1_000):
    chunked += a[k:k + 1_000].sum(axis=0)
print(np.array_equal(ref, seq), np.array_equal(ref, chunked))
```

Decision rule. `(True, False)`: use R2-a with a sequential kernel. `(True,
True)`: chunk sums are safe. Still test several chunk sizes and several
shapes (S = 1, S = 7, S = 100,000). `(False, *)`: use R2-b. Repeat the probe on
the production machine and on the numpy version pinned in `pyproject.toml`.

Q2. Does `compile_portfolio_from_ingest_batch(ingest_batch=sub, uix_mapper=full)`
work and give a portfolio whose `uix_fingerprint` equals the full mapper's?
Probe: build the full portfolio and a sub-portfolio, value both on the same
state, and compare the shared rows bitwise.

Q3. Is `compute_var_es` row-independent and thread-count independent? Probe:
compare one call on `(M + 1, S)` with chunked calls, bitwise, with 1 and N
Numba threads.

Q4. What does one market-data build cost per scenario batch? The chunk loop
repeats it `chunks x batches` times. For B, M = 1M, S = 100k there are
814 x 98 = about 80,000 builds. If a build costs 1 ms, that is 80 s. Measure
`build_market_data_matrix_from_flat_levels + assemble_valuation_state` per
batch. If the cost matters, cache the valuation states per batch across
chunks. The cache needs memory about O(U S). Decide with the number.

Q5. What is the real peak memory per stage? Replace the section 3 model.
Use `tracemalloc` or a subprocess peak RSS (the benchmark has a helper) at
M = 2,000 and S = 5,000.

Q6. (Only for WP8.) Does a fused Numba D081 kernel match NumPy bitwise? The
risk is FMA contraction of `dV * rate + V0 * (rate - rate0)`. NumPy does the
multiplications and the addition as separate operations. A fused kernel must
not contract them. Verify on random data before any use.

Q7. How small can a chunk be before Numba thread parallelism stops paying?
Time `run_valuation` at Mc = 1, 8, 64, 512 and pick a floor for the helper.

## 8. Work packages, in priority order

WP0. Probes Q1 to Q5 and Q7. No production change. Output: a results section
in this file. Exit: the owner picks R2-a or R2-b.

WP1. P1 and the chunk planner (P5) with unit tests. Tests: `take` with the
identity index equals the original batch. A multi-row UID is never split.
The planner proves coverage and rejects gaps and overlaps.

WP2. P3 and P4 with unit tests. Tests: converting a chunk equals the matching
rows of the full conversion, bitwise. The accumulator equals the unchunked
portfolio row, bitwise (R2-a). Under R2-b it equals the new canonical
reference.

WP3. Chunked FULL_REVAL path in `solutions/risk_run/api.py`. New option,
phase-1 refusal (R7), identity unchanged. Keep the old path untouched (R9).

WP4. Invariance tests, in the style of demo `06_spot_batching_and_scale.py`.
Chunk sizes 1, 3, 7, 64, M and M + 1. Compare identity, all VaR and all ES,
bitwise. Books: the realistic SPOT book, and a book with multi-row UIDs
(`tests/solutions/risk_run/_fwdfix.py` has forward fixtures). Test the
refusal with a non-empty U_D.

WP5. Memory test. Run a chunked run in a subprocess. Assert peak RSS stays
below the section 3 formula plus a stated margin. Assert it does not grow
with M at fixed Mc.

WP6. Certification and benchmark. Extend
`scripts/release/spot/certify_spot_production_scale.py` with a chunked run at
the release dimensions. Then run
`python scripts/demos/e2e/bench_spot_scaling.py --requested-grid --mem-budget-gib <N>`
with the chunked option wired into the benchmark worker (section 9). Record
the measured table.

WP7. Phase 2: TAYLOR/DERIVED chunking. `SensitivityPnlCalculator.run_into_grid`
writes the whole U_D block. It needs a compiled-book row slice, with its
digest rules (P1-A, P1-B) intact. Lift R7 for U_D only after this passes.

WP8. Optional memory and time work, only after WP6. Free `grid` after the
scatter and `settlement_pnl` after the conversion. Replace the D081 NumPy
temporaries with a zero-allocation kernel (Q6). Remove the `vstack` copy by
calling `compute_var_es` per chunk and once for the portfolio row.

WP9. `DECISIONS.md` entry for chunking and for the R2 choice. Only with owner
approval. Use the next free decision number.

## 9. How the benchmark connects

`scripts/demos/e2e/bench_spot_scaling.py` is the acceptance instrument.

* It runs one cell per subprocess and records process-lifetime peak RSS.
* `--requested-grid` selects the original 15 cells.
* The memory guard marks a cell SKIP when the estimate exceeds
  `--mem-budget-gib` (default 14).
* After WP3, add `--position-chunk-size N` (or `auto`). The worker passes it
  to `run_scenario_var_es`. The guard then uses the chunked formula from
  section 3. The estimate function is `estimate_gib` in that file.
* The stage split wraps `_value_positions`, `convert_scenario_pnl_to_reporting_currency`
  and `compute_var_es` by module attribute. A chunked path may rename or
  move them. The split then prints `n/a`. Update `StageTimer.TARGETS`.
* Success test: every requested cell runs, and measured peak RSS stays
  within the chunked formula. Compare ns per cell with the unchunked
  numbers at overlapping sizes. A large rise means Q4 matters.

## 10. Risks

K1. Summation order changes the last bits of the portfolio P&L.
`RiskRun.to_json` stores floats as IEEE-754 hex, and `result_digest` binds the
stored numbers. Demos 06 and 07 compare results bitwise. A silent change
breaks these claims even when tolerance-based oracle tests pass. Guard: R2 and
WP2.

K2. A UID with several valuation rows (E10) is split across chunks. Guard: R6
and the WP1 test.

K3. Rows reorder inside a sub-compile. `compile_portfolio_from_ingest_batch`
sorts by `instrument_id` with a stable sort. Row order inside a UID must
still match the full compile. Guard: P1 preserves original order; WP4 tests it.

K4. Per-chunk overhead (Q4, Q7) makes the run slower than a linear model.
Guard: measure before choosing the default chunk size.

K5. Boolean-mask scatter writes to a copy. Use explicit index arrays in every
new scatter.

K6. A fallback to the unchunked path on a refusal hides a memory failure
(R7). Do not add one.

K7. The benchmark and the certificate monkeypatch private names. A refactor
renames them. The benchmark degrades to `n/a`. The certificate script fails
loudly. Update both in WP6.

## 11. Out of scope

* Streaming the scenario axis. Exact empirical VaR/ES per position needs all
  S values of that position. The `K x S` array of `shocked_levels` stays
  resident. This is a second bound for large K and S.
* Euler VaR/ES contributions (`compute_var_es_contribution`). They need the
  total row first and then the component rows. This forces a second pass. It
  is a different design.
* Changes to scenario models, the identity, the manifest or the VaR/ES
  definition.
* Credit exposure, JTD and sensitivity-PnL engines.
* Non-SPOT instrument tests beyond the multi-row-UID check in WP4.
* TAYLOR/DERIVED and TAYLOR/SUPPLIED chunking until WP7.

## 12. File map

| Path | Role |
|---|---|
| `src/riskquant/solutions/risk_run/api.py` | `run_scenario_var_es`, `_value_positions`, `_write_full_reval_block`, `_row_aggregation` |
| `src/riskquant/solutions/risk_run/schemas.py` | `RiskRun`, `MethodologyDeclaration`, `ValuationContext` |
| `src/riskquant/sensitivity_pnl/grid.py` | `PnlGrid`, `allocate_pnl_grid` |
| `src/riskquant/reporting/fx_conversion.py` | `FxConversionPlan`, D081 conversion |
| `src/riskquant/scenario_metrics/api/var_es.py` | `compute_var_es` |
| `src/riskquant/scenario_metrics/_kernels/_tail_metrics.py` | tail kernel |
| `src/riskquant/portfolio/ingest/batch.py` | `IngestBatch` (P1) |
| `src/riskquant/portfolio/engine.py` | `compile_portfolio_from_ingest_batch`, `build_uix_mapper` |
| `src/riskquant/valuation/engine.py` | `run_valuation` (fingerprint check) |
| `src/riskquant/market_data/assemble.py` | `build_market_data_matrix_from_flat_levels` |
| `scripts/demos/e2e/06_spot_batching_and_scale.py` | invariance pattern to copy for chunk size |
| `scripts/demos/e2e/bench_spot_scaling.py` | acceptance benchmark |
| `scripts/release/spot/certify_spot_production_scale.py` | release certification |
| `tests/solutions/risk_run/` | existing run tests and fixtures |

Decision IDs seen in code comments. Verify each in `DECISIONS.md`.

* D041: batch invariance.
* D045 and D055: multi-row UIDs.
* D081: reporting conversion.
* D082 and D085: run axis.
* D084: inverted FX.
* D087: scenario world.
* D088: Proxy V1.
* D102: forward admission.
* PR-06: batch-tiling rule.
* P1-A and P1-B: Taylor certification gates.

## 13. Delivery rules for this workstream

Follow `CLAUDE.md` and the project working protocol.

* Audit before code.
* Deliver complete drop-in files for package-local modules.
* Use surgical edits for shared files.
* Start every `.py` file with a comment that holds its repo-relative path.
* Run the gates in this order: `py_compile`, import-all, `ruff check`,
  `ruff format --check`, `mypy --strict` (non-kernel files), `pytest`.
* Return changed `.py` files together with a zip of the project structure.
