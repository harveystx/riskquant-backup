# WS_RISK_RUN_POSITION_CHUNKING: Position-chunked FULL_REVAL run path

Path: docs/workstreams/WS_RISK_RUN_POSITION_CHUNKING.md
Status: DRAFT
Revision: 1
Revision note: First draft from investigator PLAN CHANGE on HANDOFF_position_chunked_run.md (2026-10-02).
Date: 2026-10-06
Branch: ws/risk-run-position-chunking
Snapshot: SNAPSHOT commit=043a7f4 dirty=false content_hash=12995a03dacaa471
Drafted by: investigator
Independent review: NONE
DECISIONS.md reconciled: NO
Evidence limits: No test, gate or benchmark ran. Q2 to Q5 and Q7 did not run (full environment not available). Q1 ran only in a sandbox with NumPy 2.4.4 and Python 3.12.3; pyproject.toml declares `numpy>=1.26` with no pin. All memory figures are counts from source, not measurements. The scripts profile and the credit / solutions_credit_var profiles were not attached: benchmark and certificate monkeypatches and credit machinery are not verified. `spot_production_scale_certifcate.json` was not read. Production M, S and RAM are not stated in the snapshot or in the handoff.

## 1. Goal and scope

Goal: Peak memory of `run_scenario_var_es` depends on a position chunk size, not on M, with identity and per-position VaR/ES bitwise equal to the unchunked run, and with the current path unchanged when the option is unset.

In scope:
- A new operational option on `run_scenario_var_es` for FULL_REVAL-only runs.
- Order-preserving subset of `IngestBatch` and a chunk planner with a coverage proof.
- Row slice of the reporting conversion plan, and a portfolio-row accumulator kernel.
- Invariance tests, a memory test, certification and benchmark wiring.

Out of scope:
- TAYLOR/DERIVED and TAYLOR/SUPPLIED chunking (H WP7).
- Changes to the default (unchunked) path, including H WP8.
- Streaming of the scenario axis; the `K x S` shocked-level buffer stays resident.
- Euler contributions (`compute_var_es_contribution`).
- Changes to scenario models, identity, manifest, VaR/ES definition.
- Credit exposure, JTD, sensitivity-PnL engines.

Owner decisions in force:
- none recorded for this workstream.

Depends on: O-1, O-2, O-3 (section 5).
Blocks: none.

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | `allocate_pnl_grid` allocates one `(M, S)` float32 C-contiguous grid, blocks `[U_F, U_S, U_D]`. | `src/riskquant/sensitivity_pnl/grid.py:allocate_pnl_grid` | Observed |
| F-2 | The run allocates `settlement_pnl` `(M, S)` float64 and scatters `grid.pnl` into it. `grid` and `settlement_pnl` live until return. | `src/riskquant/solutions/risk_run/api.py:run_scenario_var_es` | Observed |
| F-3 | D081 conversion allocates `rates`, `translation`, a product temporary and `converted`, each `(M, S)` float64. `portfolio = converted.sum(axis=0)`. | `src/riskquant/reporting/fx_conversion.py:convert_scenario_pnl_to_reporting_currency` | Observed |
| F-4 | The run builds `groups = vstack([converted_pnl, portfolio_pnl[None]])` and makes one `compute_var_es` call. | `api.py:run_scenario_var_es` | Observed |
| F-5 | `compute_var_es` allocates `scratchpad` `(n_groups, S)` float64. The kernel uses `prange` over rows; each row is processed alone. | `src/riskquant/scenario_metrics/_computer.py:_compute_tail_var_es`; `_kernels/_tail_metrics.py:_kernel_lower_tail_var_es` | Observed |
| F-6 | Per-row VaR/ES does not depend on other rows or on the thread count. | F-5 source | Inferred |
| F-7 | `shocked_levels = np.hstack(chunks)`; the `chunks` list is not released. The run holds `16 K S` bytes of shocked levels until return. | `api.py:run_scenario_var_es` | Observed |
| F-8 | Each scenario batch values every row of `context.portfolio` (U_D rows included), then gathers `full_positions`. | `api.py:_value_positions`, `_write_full_reval_block` | Observed |
| F-9 | A UID can own several valuation rows. `_row_aggregation` groups rows by UID with a stable sort; `_value_positions` sums each group with `np.add.reduceat`. | `api.py:_row_aggregation`, `_value_positions` | Observed |
| F-10 | `IngestBatch.reorder` rejects an index array whose length differs from `size`. No subset method exists. It has 18 parallel arrays and scalar `as_of_ord32`. | `src/riskquant/portfolio/ingest/batch.py:IngestBatch.reorder` | Observed |
| F-11 | `compile_portfolio_from_ingest_batch` sorts by `instrument_id` with `kind="stable"`, compiles through mapper gathers only, and stamps `CompiledPortfolio.uix_fingerprint`. Blocks carry no fingerprint. | `src/riskquant/portfolio/engine.py:compile_portfolio_from_ingest_batch` | Observed |
| F-12 | A sub-batch compiled under the full mapper keeps the relative row order and the full fingerprint; `run_valuation` accepts it. | F-11; `src/riskquant/valuation/engine.py:run_valuation` fingerprint check | Inferred |
| F-13 | `assemble_valuation_state` depends on the mapper, not on the portfolio. A chunk loop rebuilds the same state per batch and per chunk. | `api.py:_value_positions` | Inferred |
| F-14 | `convert_scenario_pnl_to_reporting_currency` requires `num_positions == plan.num_positions`. `FxConversionPlan` holds `fx_row_index` int32, `invert` int8, `settlement_ccys` tuple of str. | `fx_conversion.py:FxConversionPlan` | Observed |
| F-15 | The identity is built before numerics from full objects: `build_run_manifest` has no operational parameter; `_context_identity` hashes `context.portfolio`; `digest_reporting_plan` uses the full plan. | `api.py:run_scenario_var_es`; `src/riskquant/solutions/run_identity/api.py:build_run_manifest` | Observed |
| F-16 | `result_digest` binds `identity`, `position_uids`, `var`, `es`, `reproducible`. | `src/riskquant/solutions/risk_run/schemas.py:_result_digest` | Observed |
| F-17 | Sandbox, NumPy 2.4.4: `sum(axis=0)` on C-contiguous `(M, S)` float64 equals sequential row addition for S in {2,3,7,8,16,520} (150 of 150 cases). For S = 1 it differs in 19 of 30 cases. Block partial sums (mc = 3, 7, 64, 1000) differ from the reference. | sandbox probe, appendix A_PROBES.md P-Q1 | Observed |
| F-18 | Tests monkeypatch `api._value_positions`, `_write_full_reval_block`, `convert_scenario_pnl_to_reporting_currency`, `compute_var_es`, `allocate_pnl_grid`. `test_methodology_composition` reads `seen["groups"][i]` from the one `compute_var_es` call. | `tests/solutions/risk_run/test_*.py` (grep `setattr(api,`) | Observed |
| F-19 | The SPOT certification fixture declares a split FULL_REVAL / TAYLOR_DERIVED book. | `tests/solutions/risk_run/test_spot_production_certification.py` module docstring, `DERIVED` | Observed |
| F-20 | `_require_block_wiring` raises `CompositionContractError` before identity and numerics. | `api.py:_require_block_wiring` | Observed |

## 3. Contract

- C.1 With `position_chunk_size` unset, the run executes the current code path. No existing test or certificate changes. [Proposed] Test: S5.T1
- C.2 For every admitted chunk size, `identity`, `manifest` and `result_digest` inputs other than `var`/`es` are bitwise equal to the unchunked run. [Proposed] Test: S5.T2
- C.3 For every admitted chunk size, every per-position VaR and ES is bitwise equal to the unchunked run. [Proposed] Test: S5.T3
- C.4 Portfolio VaR and ES follow the rule that O-2 selects. [Unresolved] Test: S3.T2, S5.T4
- C.5 A chunk contains all valuation rows of each of its UIDs, in compiled order. [Proposed] Test: S2.T2
- C.6 The chunks tile the ascending economic UID axis exactly once, in order. A gap, overlap or reorder raises before numerics. [Proposed] Test: S2.T3
- C.7 A chunked run with a non-empty U_S or U_D raises `CompositionContractError` before identity and numerics. It does not fall back. [Proposed] Test: S4.T2
- C.8 The chunk size is not an input of `build_run_manifest`, `_context_identity` or `digest_reporting_plan`. The identity is computed before the chunk loop from full objects. [Proposed] Test: S5.T2
- C.9 Peak RSS of a chunked run has no term in M x S. At fixed chunk size and S, peak RSS does not grow with M beyond the O(M) per-position inputs. [Proposed] Test: S6.T1
- C.10 Each chunk quantises `dV` through float32 and widens to float64, as F-1 and F-2. [Proposed] Test: S5.T3
- C.11 Non-positive chunk size raises `ValueError`. [Proposed] Test: S4.T1

## 4. Decisions

Decisions relied on: D041, D042, D055, D081 (§1, §3, §8), D083, D084, D085, D087 (clauses 3 and 5), D102. Each checked against `DECISIONS.md`: yes.

Conflicts to resolve:
- D081 §8 states "one PnlGrid -> one D081 reporting transformation -> VaR / ES". A chunked path makes one grid and one conversion per chunk.
- D042 states one canonical `compute_var_es` over `[positions; portfolio]`.
- D087 clause 5 reads D081 §8 as whole-run arrays.
- D041 makes `batch_size` operational. It does not name a position chunk size.
- No decision fixes the summation order of the portfolio row. D041 claims bitwise portfolio P&L only in a matching environment.

### P-RISK_RUN_POSITION_CHUNKING-001: Position chunk size is operational (PROPOSED)

```text
Related: D041, D042, D081 §8, D083, D087
Scope: An optional position chunk size on run_scenario_var_es for FULL_REVAL-only runs.
1. The chunk size is an operational parameter. It is absent from the manifest, the identity and the result digest.
2. A chunked run is one D081 §8 composition partitioned by economic UID. Each partition holds the grid, conversion and VaR/ES rows of its UIDs only.
3. Per-position VaR and ES are bitwise invariant to the chunk size in a matching environment.
4. A chunked run admits FULL_REVAL only. A non-empty U_S or U_D is refused before identity and numerics.
Evidence: S2.T2, S2.T3, S4.T2, S5.T1 to S5.T4, S6.T1.
```

### P-RISK_RUN_POSITION_CHUNKING-002: Portfolio-row summation order (PROPOSED)

```text
Related: D041, D042, D083
Scope: The canonical floating-point order of the portfolio scenario row.
1. Option R2-a: the portfolio row is the sequential float64 sum of converted rows in ascending UID order. The chunked accumulator reproduces it. S = 1 needs a separate rule (F-17).
2. Option R2-b: a new canonical order applies to both paths. Stored portfolio VaR/ES bits change.
The owner selects one option at O-2.
Evidence: S1 P-Q1, S3.T2, S5.T4.
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Is a UID-partitioned composition (P-001.2) compliant with D081 §8 and D042, or is a D081/D042 amendment required? | OPEN | | |
| O-2 | R2-a or R2-b for the portfolio row (P-002)? For R2-a: how is S = 1 handled? | OPEN | | |
| O-3 | What are the production M, S, K and RAM? Does the need exist outside the benchmark grid? | OPEN | | |
| O-4 | Pin the NumPy version in pyproject.toml, or accept that R2-a holds only per environment (D041)? | OPEN | | |
| O-5 | P2 route: sub-ingest compile (F-12) or compiled-block slicing? | OPEN | | |
| O-6 | Must the chunked path release the duplicated shocked-level buffer (F-7)? It changes the default path if shared. | OPEN | | |

## 6. Slices

Order: S1, then S2 and S3 in parallel, then S4, S5, S6, S7. S2 and S3 may run in parallel after S1.

### S1: Probes in the full environment

Objective: Replace inferred facts F-6, F-12, F-13 and the sandbox result F-17 with measurements in the pinned environment.
Contract: none (evidence for C.3, C.4, C.9, O-2, O-5).
Depends on: none
Files:
- create: docs/workstreams/WS_RISK_RUN_POSITION_CHUNKING/R_PROBES_<YYYY-MM-DD>.md
- modify: none
Prohibited: any change in `src/`, `tests/`, `scripts/`, `DECISIONS.md`.
Gates: the probe scripts in appendix A_PROBES.md, run in the project environment; record `numpy.__version__`, `numba.__version__`, `NUMBA_NUM_THREADS`, CPU count.
Exit: Report records each probe P-Q1 to P-Q7 with raw output and machine data.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | P-Q1 in project env, S in {1,2,7,520}, M in {9,1003,10000} | Table of equalities | Wrong R2-a premise |
| S1.T2 | P-Q2 sub-compile vs full compile, shared rows bitwise | Equal | Sub-compile reorders or remaps rows |
| S1.T3 | P-Q3 `compute_var_es` full vs row chunks, 1 and N threads | Equal | Row coupling in tail kernel |
| S1.T4 | P-Q4, P-Q5, P-Q7 cost and peak per stage | Recorded numbers | Hidden per-chunk overhead |

Appendix: A_PROBES.md

### S2: IngestBatch subset and chunk planner

Objective: Give an order-preserving batch subset and a proven UID-chunk plan.
Contract: C.5, C.6
Depends on: S1, O-5 (sub-ingest route)
Files:
- modify: src/riskquant/portfolio/ingest/batch.py
- create: tests/portfolio/test_ingest_batch_subset.py; planner module and test under src/riskquant/solutions/risk_run/ and tests/solutions/risk_run/ (names set by the implementer)
Prohibited: change to `IngestBatch.reorder` semantics; change to `compile_portfolio_from_ingest_batch`.
Gates: py_compile; import-all; `ruff check`; `ruff format --check`; `mypy --strict` on changed non-kernel files; `pytest tests/portfolio tests/solutions/risk_run -q`.
Exit: All gates pass. Subset with identity index equals the batch field by field.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | Subset with `arange(size)` | Every field equal, `as_of_ord32` kept | Dropped or reordered field |
| S2.T2 | Multi-row UID (forward + spot, D055) near a chunk edge | UID never split; row order equals full compile | K2, K3 of H |
| S2.T3 | Plans with gap, overlap, reorder, empty chunk | Raise before numerics | Silent coverage error |

### S3: Conversion-plan row slice and portfolio accumulator

Objective: Convert one chunk exactly as the full conversion does, and accumulate the portfolio row in the O-2 order.
Contract: C.3, C.4, C.10
Depends on: S1, O-2
Files:
- modify: src/riskquant/reporting/fx_conversion.py (row slice only, if O-2 and the implementer require it)
- create: accumulator kernel under the owning core domain with a `riskquant.jit` decorator; its unit test
Prohibited: change to `convert_scenario_pnl_to_reporting_currency` arithmetic or validation; a fused D081 kernel (H Q6); local Numba aliases.
Gates: as S2, plus `pytest tests/reporting -q`.
Exit: All gates pass.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | Chunk conversion vs rows of full conversion, inverted and identity FX rows | Bitwise equal | Plan slice misaligned |
| S3.T2 | Accumulator vs `converted.sum(axis=0)` (R2-a) or the canonical reference (R2-b), S in {1,2,7,520}, chunk sizes {1,3,7,64,M-1,M,M+1} | Bitwise equal | Summation order drift (K1 of H) |

### S4: Chunked FULL_REVAL path

Objective: Add the operational option and the chunk loop with scenario batches inside position chunks.
Contract: C.1, C.2, C.7, C.8, C.11
Depends on: S2, S3, O-1
Files:
- modify: src/riskquant/solutions/risk_run/api.py
Prohibited: any change to the default path statements; per-chunk identity; a fallback to the unchunked path; boolean-mask scatter.
Gates: as S2, plus `pytest tests/solutions -q`.
Exit: All gates pass with no change to existing tests.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | Chunk size 0, -1 | `ValueError` | Missing domain check |
| S4.T2 | Chunked run with non-empty U_D; with non-empty U_S | `CompositionContractError` before manifest and generator (stage spies) | Silent fallback (R7, K6 of H) |

### S5: Invariance tests

Objective: Prove C.1 to C.4 on realistic books.
Contract: C.1, C.2, C.3, C.4, C.8, C.10
Depends on: S4
Files:
- create: tests/solutions/risk_run/test_position_chunked_run.py
Prohibited: change to existing tests or fixtures.
Gates: `pytest tests/solutions/risk_run -q`; full `pytest -q`.
Exit: All pass.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | Option unset vs current path, all existing risk_run tests | Unchanged | Default path drift |
| S5.T2 | Chunk sizes {1,3,7,64,M,M+1}: identity, manifest JSON | Bitwise equal to unchunked | Chunk size in identity |
| S5.T3 | Same sizes: every per-position VaR/ES hex | Bitwise equal | Row coupling, f32 quantisation drift |
| S5.T4 | Same sizes: portfolio VaR/ES hex; `result_digest` | Equal under O-2 rule | Summation order drift |

### S6: Memory test

Objective: Show that peak RSS has no M x S term.
Contract: C.9
Depends on: S4
Files:
- create: test in a subprocess under tests/solutions/risk_run/ (marker set by the implementer)
Prohibited: tolerance tuned to pass; claims from estimates.
Gates: the new test; record raw RSS figures.
Exit: Peak RSS at fixed chunk size grows only with the O(M) term across at least three M values.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S6.T1 | M in {a, 4a, 16a}, fixed S and chunk size | Peak RSS slope consistent with O(M), not O(M S) | Grid kept alive across chunks |

### S7: Certification and benchmark

Objective: Record the chunked run at release dimensions and on the benchmark grid.
Contract: C.3, C.9
Depends on: S5, S6, O-3
Files:
- modify: scripts/release/spot/certify_spot_production_scale.py; scripts/demos/e2e/bench_spot_scaling.py
Prohibited: change to the unchunked certificate values.
Gates: the certificate script; `python scripts/demos/e2e/bench_spot_scaling.py --requested-grid --mem-budget-gib <N>` with the chunk option.
Exit: Measured table recorded in a report. Monkeypatch targets updated (K7 of H).
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S7.T1 | Certificate with chunked run | Chunked VaR/ES equal unchunked | Release regression |

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| Summation order differs (F-17) | Portfolio bits change; `result_digest` changes | O-2, S3.T2, S5.T4 |
| NumPy version not pinned | R2-a premise holds per environment only | O-4; D041 environment scope |
| Market-data rebuild per chunk and batch (F-13) | Run time grows with chunk count | S1 P-Q4; owner sets default size |
| All rows valued per batch (F-8) | U_D rows valued in every chunk if not excluded | Chunk compile holds U_F rows only (C.7 makes U_D empty) |
| Shocked levels held twice (F-7) | Residual 16 K S bound | O-6 |
| Private-name monkeypatches (F-18) | Refactor breaks tests | S4 prohibits default-path change |

Deferred:
- TAYLOR/DERIVED chunking: opens when this workstream is CLOSED and an owner gate admits it.
- H WP8 default-path memory work: opens when O-1 and a separate decision allow a change to the D042 single call.
- Scenario-axis streaming: opens when O-3 shows `16 K S` binds.

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.
