# WS_RISK_RUN_POSITION_CHUNKING: Appendix A_PROBES

Used by: S1. Probes change no file in `src/`, `tests/`, `scripts/` or `DECISIONS.md`.
Record for every probe: `numpy.__version__`, `numba.__version__`, Python version,
`NUMBA_NUM_THREADS`, `os.cpu_count()`, and the raw output.

## P-Q1: Portfolio-row summation order

Sandbox result (2026-10-06, NumPy 2.4.4, Python 3.12.3), Observed:

| S | ref == sequential | Cases |
|---|---|---|
| 1 | no in 19 of 30 | seeds 0..4, M in {2,9,130,1003,10000,50000} |
| 2, 3, 7, 8, 16, 520 | yes in 150 of 150 | same |

Block partial sums `acc += a[k:k+mc].sum(axis=0)` differ from the reference for
mc in {3, 7, 64, 1000}. They match for mc = 1 and mc >= M - 1.

Required in the project environment:
1. Repeat the sweep above with the pinned or installed NumPy.
2. Compare the reference with a sequential row accumulator built with the
   `riskquant.jit` decorator, for S in {1, 2, 7, 520} and chunk sizes
   {1, 3, 7, 64, M - 1, M, M + 1}.
3. Report each mismatch with (seed, M, S, chunk size).

## P-Q2: Sub-compile under the full mapper

1. Build the full `UixMapper` and the full portfolio from one ingest batch.
2. Take a row subset in original order, including a multi-row UID (D055).
3. Compile the subset with the full mapper.
4. Assert `uix_fingerprint` equality.
5. Value both on one valuation state. Compare shared rows bitwise.

## P-Q3: Tail kernel row independence

1. Build a finite `(M + 1, S)` float64 C-contiguous matrix.
2. Call `compute_var_es` once on all rows, then on row blocks.
3. Compare VaR and ES hex values. Repeat with 1 and N Numba threads.

## P-Q4: Per-batch market-data cost

Time `build_market_data_matrix_from_flat_levels` plus `assemble_valuation_state`
per batch at the certification axis size. Report the median of at least 20 calls.
Multiply by chunks x batches for the cells of H section 3.

## P-Q5: Peak memory per stage

Run the unchunked path in a subprocess at M = 2000, S = 5000. Record peak RSS
after each stage: shocked levels, FULL_REVAL block, scatter, D081, `vstack`,
`compute_var_es`. Compare with the count from source:

| Stage | Bytes per (position, scenario) cell, from source |
|---|---|
| FULL_REVAL write | 4 (grid) |
| Scatter | 12, or 20 if NumPy makes a cast temporary |
| D081 peak | 44 (grid 4, settlement 8, rates 8, translation 8, product 8, converted 8); 36 if temporary elision applies |
| After D081 with `vstack` | 28 |
| `compute_var_es` | 36, plus 1 transient (isfinite mask) |

Per scenario, add `16 K` bytes of shocked levels (F-7) and `8 K_fx` bytes of FX rows.

## P-Q7: Small-chunk floor

Time `run_valuation` at chunk sizes 1, 8, 64, 512 with the project thread
count. Report ns per cell.

P-Q6 (fused D081 kernel) is out of scope for this workstream.
