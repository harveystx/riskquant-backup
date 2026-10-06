# WS_SOLUTIONS_SPOT_INPUTS: Solution-tier input builders for a SPOT VaR run

Path: docs/workstreams/WS_SOLUTIONS_SPOT_INPUTS.md
Status: DRAFT
Revision: 1
Revision note: First draft from the investigator PLAN CHANGE on the non-credit solutions surface.
Date: 2026-10-06
Branch: ws/solutions-spot-inputs
Snapshot: SNAPSHOT commit=043a7f4 dirty=false content_hash=12995a03dacaa471
Drafted by: investigator
Independent review: NONE
DECISIONS.md reconciled: NO
Evidence limits: No command, test or gate was executed. All facts come from static reading and AST scans of the snapshot profiles. The scripts profile has no SNAPSHOT line; its identity with commit 043a7f4 is not proven. stx-data-god and riskquant-excel sources were not available. Credit profiles were not read.

## 1. Goal and scope

Goal: A consumer builds and runs a SPOT VaR with `riskquant.solutions.*` imports only, and the e2e demos and their helpers do the same.

In scope:
- New solution-tier builders for the six inputs of a SPOT run: static config, positions, level snapshot, history matrix, proxy terms, run request.
- Migration of `scripts/demos/e2e` (12 files: `_fixtures.py`, `_spot_world.py`, `00` to `07`, `bench_spot_scaling.py`, `demo_e2e_sensi_based_var.py`) to solutions-only imports.
- Decoupling of tests that read core symbols through `scripts/demos/e2e/_fixtures.py`.
- A Taylor certification builder (after the SPOT FULL_REVAL path).
- A boundary rule, after the owner selects the mechanism.

Out of scope:
- `credit_exposure`, `credit_var` and all credit code.
- Kernels, numerics and performance.
- The `run_scenario_var_es` signature and the position-chunked run path (O-15).
- Any change to `DECISIONS.md`.
- The stx-data-god and riskquant-excel repositories.

Owner decisions in force:
- 2026-10-06: Consumers and the e2e demos reach riskquant only through `riskquant.solutions.*`. Credit is out of scope.
- 2026-10-06: The first consumer is stx-data-god. Every position is SPOT and FULL_REVAL. No Taylor block.

Depends on: D001, D067, D093, D094.
Blocks: stx-data-god SPOT production export (workstream outside this repository).

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | `ValuationContext` fields are `RiskStaticConfig`, `IngestBatch`, `UixMapper`, `DiscountingMode`; `portfolio: CompiledPortfolio` is compiled in `__post_init__`. | `src/riskquant/solutions/risk_run/schemas.py:ValuationContext` | Observed |
| F-2 | The risk_run contracts are frozen dataclasses. Pydantic exists only in `run_identity/schemas.py` and `scenario_models/schemas.py`. | `src/riskquant/solutions/risk_run/schemas.py` | Observed |
| F-3 | `run_scenario_var_es` takes `data: RiskFactorMatrix \| ParametricRiskData` and `proxy_terms: Sequence[AuthoredProxyTerm]`. | `src/riskquant/solutions/risk_run/api.py:run_scenario_var_es` | Observed |
| F-4 | `compile_raw_df_into_ingest_batch` accepts `pl.DataFrame` only and raises `TypeError` otherwise. | `src/riskquant/portfolio/ingest/pipeline.py:compile_raw_df_into_ingest_batch` | Observed |
| F-5 | The positions frame contract: required `uid`, `instrument`; market `quantity`, `underlying`, `settlement_ccy`; credit `netting_set_id`; nine extension columns. Unknown columns raise. | `src/riskquant/portfolio/ingest/schema.py:ALL_ALLOWED_COLS_SET`; `canonicalize.py`; `validate.py:validate_spot_family` | Observed |
| F-6 | `build_uix_mapper(ingest_batches, as_of_ord32)` builds the mapper. `date_to_ord32` converts the date. | `src/riskquant/portfolio/engine.py:build_uix_mapper`; `src/riskquant/utils/dates.py:date_to_ord32` | Observed |
| F-7 | `compile_risk_static_config_from_dict_of_dfs(dict[str, pl.DataFrame])` compiles the static config. `load_risk_static_config_xlsx_into_dict` reads xlsx. | `src/riskquant/config/static_config/compile.py`; `loader.py` | Observed |
| F-8 | `build_risk_factor_level_snapshot_from_dict` takes `Mapping[str, float]`. `extract_levels_from_snapshot` returns float64 levels for given rf ids. | `src/riskquant/market_data/risk_factors/builder_snapshot.py`; `src/riskquant/market_data/assemble.py` | Observed |
| F-9 | `build_risk_factor_matrix_from_df` takes `pd.DataFrame` and raises `ValueError` for other types. | `src/riskquant/market_data/risk_factors/builder_matrix.py:build_risk_factor_matrix_from_df` | Observed |
| F-10 | `proxy_terms_from_df(pl.DataFrame) -> tuple[AuthoredProxyTerm, ...]` exists. | `src/riskquant/config/proxy_model/ingest.py:proxy_terms_from_df` | Observed |
| F-11 | `parse_scenario_model_config`, `MethodologyDeclaration`, `ReportingSpec` and `derive_run_factor_axis` exist in solutions. | `solutions/scenario_models/api.py`; `solutions/risk_run/schemas.py`; `solutions/risk_run/api.py` | Observed |
| F-12 | Chapter 07 states that no solution facade exists for ingest, mapper, snapshot or proxy terms. | `scripts/demos/e2e/07_spot_production_example.py` module docstring | Observed |
| F-13 | The 12 demo files import 25 non-solutions riskquant modules. | AST import scan, gap map in A_GAP_MAP.md | Observed |
| F-14 | `tests/test_architecture.py` scans `src/riskquant` only. pyproject has no import-linter and no ruff TID rule. | `tests/test_architecture.py:SOURCE_ROOT`; `pyproject.toml:[tool.ruff.lint]` | Observed |
| F-15 | Tests load `_fixtures.py` as module `FX` and read core symbols through it. | `tests/solutions/risk_run/_mixedfix.py`; users `_fixture.py`, `_certfix.py`, `test_run_owned_identity.py`, `test_methodology_composition.py`; `tests/e2e/test_demo_e2e_sensi_based_var.py` | Observed |
| F-16 | `_forward_world.py`, `10_forward_node_full_reval.py`, `11_forward_node_taylor_derived.py` import `_spot_world`. | scripts profile grep | Observed |
| F-17 | `00`, `demo_e2e_sensi_based_var.py` and `_spot_world.py` re-derive the run with core blocks for teaching. | `00` docstring; `_spot_world.py:shocked_levels_for`, `revalue`, `scenario_pnl_anatomy` | Observed |
| F-18 | The run derives the expected Taylor base binding privately. | `src/riskquant/solutions/risk_run/api.py:_value_positions` and the `TaylorBaseBinding` built in `run_scenario_var_es` | Observed |
| F-19 | Chapter 05 constructs `RiskFactorMatrix` directly. | `scripts/demos/e2e/05_spot_reproducibility_and_identity.py` | Observed |
| F-20 | A run with `proxy_terms=()` is used by tests. | `tests/solutions/risk_run/_fixture.py` | Inferred (not executed) |
| F-21 | Removing core imports from `_fixtures.py` breaks the tests in F-15. | F-15 | Inferred (not executed) |

## 3. Contract

- C.1 A static config builder in solutions returns `RiskStaticConfig` from `dict[str, pl.DataFrame]`, through F-7 only. [Proposed] Test: S1.T1
- C.2 The static config builder refuses a missing sheet with the existing compiler error. It adds no validation. [Proposed] Test: S1.T2
- C.3 A valuation context builder takes a `pl.DataFrame` positions frame, the static config, one as-of `date` and a discounting mode. It returns `ValuationContext`. [Proposed] Test: S2.T1
- C.4 The valuation context builder passes the one as-of to ingest and to the mapper (D067). [Proposed] Test: S2.T2
- C.5 The valuation context builder refuses a non-Polars frame with the D094 `TypeError`. [Proposed] Test: S2.T3
- C.6 A base level builder takes name to level data, the as-of, the static config and a `RunFactorAxis`. It returns float64 levels row-aligned to `RunFactorAxis.factor_ids`. [Proposed] Test: S3.T1
- C.7 The base level builder refuses a missing factor with the existing snapshot error. [Proposed] Test: S3.T2
- C.8 A history builder returns `RiskFactorMatrix` in canonical units, through F-9, with the input type that O-7 selects. [Unresolved] Test: S4.T1
- C.9 A proxy term builder returns `tuple[AuthoredProxyTerm, ...]` through F-10 only. [Proposed] Test: S5.T1
- C.10 A SPOT FULL_REVAL run built only from solutions imports gives a `RiskRun` bitwise equal to the same run built with core imports. [Proposed] Test: S6.T1
- C.11 No new builder changes `run_scenario_var_es` or any core module. [Proposed] Test: S6.T2
- C.12 After S7, no test reads a core symbol through a demo module. [Proposed] Test: S7.T1
- C.13 After S6 to S9, the 12 demo files import no non-solutions riskquant module, except the exemptions O-10, O-13 and O-17 approve. [Unresolved] Test: S10.T1
- C.14 The D019 golden numbers of `demo_e2e_sensi_based_var.py` do not change. [Proposed] Test: S9.T1

## 4. Decisions

Decisions relied on: D001, D006, D019, D022, D042, D067, D076, D081, D082, D085, D088, D093, D094. Each checked against `DECISIONS.md`: yes (headings and decision text read; implementation not verified).

### P-SOLUTIONS_SPOT_INPUTS-001: Solution-tier input builders (PROPOSED)

```text
Related: D001, D067, D093, D094
Scope: One solutions package holds one builder per run input. Each builder wraps existing core machinery.
1. Each builder calls existing core functions only. It adds no validation, conversion or default that core lacks.
2. Each builder that needs an as-of takes one datetime.date and passes it to every as-of consumer.
3. Builders return existing solution or core types. No new result type is introduced.
4. The package name and home follow O-1.
Evidence: S1.T1 to S5.T1, S6.T1, S6.T2
```

### P-SOLUTIONS_SPOT_INPUTS-002: Import boundary rule for scripts (PROPOSED)

```text
Related: D001, D002
Scope: A mechanical rule that refuses a non-solutions riskquant import in a named path set.
1. The path set and the exemptions follow O-10, O-11, O-12, O-13 and O-17.
2. The mechanism follows O-11.
Evidence: S10.T1
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Where do the builders live, and what is the package name? New package or `risk_run/api.py`? | OPEN | | |
| O-2 | Where does the positions frame contract live? `portfolio/ingest/schema.py` only, or also a solutions re-export? | OPEN | | |
| O-3 | May `ValuationContext` keep core field types (F-1)? | OPEN | | |
| O-4 | How is `DiscountingMode` exposed? Solutions re-export or string literal? | OPEN | | |
| O-5 | Who authors the static config? stx-data-god sheets, or a riskquant-owned file? | OPEN | | |
| O-6 | How does a consumer get the name universe for `underlying`, history columns and levels? | OPEN | | |
| O-7 | History input type: pandas (F-9), Polars with conversion in solutions, or a new Polars core builder (D006, D022, D076)? | OPEN | | |
| O-8 | Level input type: `Mapping[str, float]` or a frame? | OPEN | | |
| O-9 | Proxy input: frame only (F-10), or also typed terms (needs `ShiftType`)? | OPEN | | |
| O-10 | Glass-box demos (F-17): solutions anatomy facade, move to tests, or exemption? | OPEN | | |
| O-11 | Boundary mechanism: AST test, ruff TID251, or import-linter? Scope: scripts, tests? | OPEN | | |
| O-12 | Public solutions module: `api.py` only, or any module under `solutions`? | OPEN | | |
| O-13 | Are `riskquant.utils.printer` and `riskquant_lab` allowed in demos? | OPEN | | |
| O-14 | Do run-request contracts become Pydantic for terminals (F-2)? | OPEN | | |
| O-15 | May any builder touch `run_scenario_var_es` or the position-chunked path? | OPEN | | |
| O-16 | May `_fixtures.py` stop exposing core symbols that tests read (F-15)? | OPEN | | |
| O-17 | May chapter 05 construct `RiskFactorMatrix` directly (F-19)? | OPEN | | |

## 6. Slices

Order: S7 first if O-16 allows helper changes; else S7 runs before S6. S1 to S5 run in parallel after O-1. S6 after S1 to S5. S8 after S6. S9 after O-10. S10 last.

Common gates (changed files only, D023):
`python -m py_compile <changed .py>` ; import-all gate ; `ruff check <changed>` ; `ruff format --check <changed>` ; `mypy --strict <changed non-kernel>` ; `python -m pytest -q`.

### S1: Static config builder

Objective: Expose static config compile and load through solutions.
Contract: C.1, C.2
Depends on: O-1, O-5
Files:
- create: `src/riskquant/solutions/<O-1>/api.py` (static config part), `tests/solutions/<O-1>/test_static_config.py`
- modify: none
Prohibited: changes to `src/riskquant/config/**`; new validation.
Gates: common gates; `python -m pytest tests/solutions/<O-1> -q`
Exit: S1.T1 and S1.T2 pass with raw output.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | Canonical sheets through builder and through F-7 | Equal compiled content (`static_config_content_digest`) | Facade alters config |
| S1.T2 | One sheet removed | Same error type as the core compiler | Facade masks a missing sheet |

Appendix: none

### S2: Valuation context builder

Objective: Build `ValuationContext` from a Polars positions frame and one as-of.
Contract: C.3, C.4, C.5
Depends on: O-1, O-2, O-3, O-4
Files:
- create: builder in `src/riskquant/solutions/<O-1>/api.py`; `tests/solutions/<O-1>/test_valuation_context.py`
- modify: none
Prohibited: changes to `src/riskquant/portfolio/**` and `solutions/risk_run/**`.
Gates: common gates
Exit: S2.T1 to S2.T3 pass.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | SPOT frame from `_spot_world.TINY_POSITIONS` | `hash_compiled_portfolio` equal to the core-built context | Facade changes compile |
| S2.T2 | As-of 2027-01-04 | `uix_mapper.as_of_ord32 == date_to_ord32(as_of)` and batch as-of equal | Split as-of (D067) |
| S2.T3 | pandas frame | `TypeError` naming D094 | Silent conversion |

Appendix: none

### S3: Base level builder

Objective: Return base levels aligned to the run axis.
Contract: C.6, C.7
Depends on: O-1, O-8
Files:
- create: builder in `src/riskquant/solutions/<O-1>/api.py`; `tests/solutions/<O-1>/test_base_levels.py`
Prohibited: changes to `src/riskquant/market_data/**`.
Gates: common gates
Exit: S3.T1 and S3.T2 pass.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | Levels for every axis factor | Bitwise equal to F-8 composition; dtype float64; length = `len(factor_ids)` | Row misalignment |
| S3.T2 | One axis factor missing | Existing snapshot error | Invented default level |

Appendix: none

### S4: History builder

Objective: Return `RiskFactorMatrix` from consumer history.
Contract: C.8
Depends on: O-1, O-7
Files:
- create: builder in `src/riskquant/solutions/<O-1>/api.py`; `tests/solutions/<O-1>/test_history.py`
Prohibited: change to unit normalisation (D076); new pandas use unless O-7 approves it.
Gates: common gates
Exit: S4.T1 passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | Wide history of 3 factors, PERCENT unit on one | `values`, `factor_ids`, `dates` bitwise equal to F-9 | Double scaling |

Appendix: none

### S5: Proxy term builder

Objective: Expose the D093 frame edge through solutions.
Contract: C.9
Depends on: O-1, O-9
Files:
- create: builder in `src/riskquant/solutions/<O-1>/api.py`; `tests/solutions/<O-1>/test_proxy_terms.py`
Prohibited: changes to `src/riskquant/config/proxy_model/**`.
Gates: common gates
Exit: S5.T1 passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | Canonical proxy frame and zero-row frame | Equal to `proxy_terms_from_df` output; `()` for zero rows | Reorder or drop |

Appendix: none

### S6: SPOT FULL_REVAL consumer path

Objective: Prove one run with solutions-only imports. Migrate the FULL_REVAL steps of chapter 07.
Contract: C.10, C.11
Depends on: S1 to S5, O-15
Files:
- create: `tests/solutions/<O-1>/test_spot_full_reval_consumer_path.py`
- modify: `scripts/demos/e2e/07_spot_production_example.py`
Prohibited: changes to `src/riskquant/solutions/risk_run/**`; changes to the Taylor steps of 07.
Gates: common gates; `python -m pytest tests/e2e -q`
Exit: S6.T1 and S6.T2 pass; 07 runs.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S6.T1 | Realistic SPOT book, all FULL_REVAL, built two ways | Equal `identity`, `var`, `es` bitwise | Facade changes inputs |
| S6.T2 | AST scan of the new test file | Only `riskquant.solutions.*` riskquant imports | Hidden core import |

Appendix: none

### S7: Test decoupling from demo helpers

Objective: Remove test reads of core symbols through `_fixtures.py`.
Contract: C.12
Depends on: O-16
Files:
- modify: `tests/solutions/risk_run/_mixedfix.py`, `_fixture.py`, `_certfix.py`, `test_run_owned_identity.py`, `test_methodology_composition.py`, `tests/e2e/test_demo_e2e_sensi_based_var.py`
Prohibited: test logic, tolerance or golden value changes.
Gates: common gates
Exit: S7.T1 passes; the full suite passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S7.T1 | grep `FX\.` and `_FIXTURES\.` for core symbol names | No match | Test coupling to demo imports |

Appendix: none

### S8: Taylor certification builder

Objective: Build a certified `TaylorBlock` for U_D through solutions.
Contract: C.13 (Taylor part)
Depends on: S6, O-1
Files:
- create: builder in solutions (O-1); tests
- modify: `01`, `02`, `07`, `_fixtures.py`, `_spot_world.py` Taylor steps
Prohibited: a second binding derivation; reuse F-18. No change to D081 or D086 semantics.
Gates: common gates; `python -m pytest tests/solutions tests/e2e -q`
Exit: S8.T1 passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S8.T1 | Mixed book of chapter 02 | `RiskRun.identity` and VaR/ES unchanged | Binding drift |

Appendix: none

### S9: Glass-box demos

Objective: Apply the O-10 decision to `00`, `demo_e2e_sensi_based_var.py` and the `_spot_world` anatomy functions.
Contract: C.13, C.14
Depends on: O-10
Files: per O-10
Prohibited: change to D019 constants or golden checks.
Gates: common gates; `python -m pytest tests/e2e -q`
Exit: S9.T1 passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S9.T1 | Golden checks of `demo_e2e_sensi_based_var.py` | Unchanged | D019 drift |

Appendix: none

### S10: Boundary rule

Objective: Enforce P-SOLUTIONS_SPOT_INPUTS-002 on the 12 files.
Contract: C.13
Depends on: S6 to S9, O-11, O-12, O-13, O-17
Files: per O-11
Prohibited: rules on credit demos; rules outside the approved path set.
Gates: common gates
Exit: S10.T1 passes; forward demos `10`, `11` still run (F-16).
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S10.T1 | Rule run on the 12 files | Zero violations outside the approved exemptions | Boundary regression |

Appendix: none

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| Helper change breaks tests (F-15, F-21) | Hidden red suite | S7 before helper migration |
| `_spot_world` change breaks forward demos (F-16) | Forward demos fail | Run `scripts/run_all_scripts_e2e.py` in S6 to S10 |
| Pandas enters solutions (O-7) | Conflict with D006 and CLAUDE.md section 4 | O-7 before S4 |
| Facade duplicates core validation | Two sources of truth | P-SOLUTIONS_SPOT_INPUTS-001 clause 1 |
| Scripts profile identity unproven | Facts F-12 to F-19 may be stale | Regenerate the scripts profile with a header |

Deferred:
- riskquant-excel migration: opens when its source is attached.
- stx-data-god import audit: opens when its source is attached.
- Forward demos `10`, `11`, `_forward_world.py`, `bench_forward_scaling.py`: opens after S10.
- Parametric data builder (`ParametricRiskData`): opens when a consumer needs the parametric model.

## 8. Closure record

Fill this section when the status becomes CLOSED.
