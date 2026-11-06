# WS_CREDIT_POLARS: Freeze reconciliation

Path: docs/workstreams/WS_CREDIT_POLARS/R_FREEZE_RECONCILIATION_R1.md
Date: 2026-11-05
Spec: WS_CREDIT_POLARS.md, revision 2
Round: 1 of 2
Clauses covered: C.1, C.2, C.3, C.4, C.5, C.6, C.7, C.8, C.9, C.10, C.11, C.12, C.13, C.14, C.17; P-CREDIT_POLARS-001 clauses 1 to 6
Clauses not covered: C.15, C.16 (S7: profiles not attached; C.15 also waits for O-7)
Snapshot: SNAPSHOT commit=1ae80e3 dirty=false content_hash=c00f240898f32d74
Mode: read-only
Evidence base: WS_CREDIT_POLARS.md (revision 2), A_INVENTORY.md, WS_TEMPLATE.md, CLAUDE.md, DECISIONS.md, WORKFLOW.md, context_riskquant_index.txt, context_riskquant_credit.txt, context_riskquant_solutions_credit_var.txt, context_riskquant_tests_credit.txt, context_riskquant_tests_solutions_credit_var.txt, context_riskquant_scripts.txt. For caller searches and for S2 and S3 clauses, a tree was rebuilt from every src profile in the project files (solutions_credit_exposure, utils, portfolio, market_data, ingest_fixtures, lab and others).
Evidence limits: No repository gate, test, demo or engine ran. Every test statement comes from source. No pass claim. Probes marked "sandbox" ran in the audit sandbox with polars 2.1.0, pandas 3.0.6 and numpy 2.5.4; numba and pydantic were not installed. The tests profiles for portfolio, market_data, e2e, utils, config, ingest_fixtures, simlab, sensitivities, root and the risk_run, run_identity and scenario_models solutions were not available. riskquant-emissary and riskquant-excel were not read.

## A. Verdict

NOT READY

Eight BLOCKING findings exist (section F). B-4 and B-5 need an owner choice.
Tripwire (WS_TEMPLATE section 1.8): the revision that applies B-1 to B-8 is Revision 3 with `Code merged: no`. That stops the cycle before round 2. The owner chooses: build v1 now, descope, or stop.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| O-1 | VERIFIED | Spec section 5, owner record | none |
| O-2 | VERIFIED | Owner record. `portfolio/ingest/batch.py:IngestBatch.to_df` imports pandas inside the function (line 255) and under TYPE_CHECKING (line 48). | R-12 |
| O-3 | VERIFIED | Owner record. Sandbox: `pl.Series([3.5]).cast(pl.Int8)` gives 3 without error (F-11 holds). | R-9 |
| O-4 | VERIFIED | Owner record. Solution may import core (D001). | none |
| O-5 | VERIFIED as recorded; text conflicts with S6.T3 | `schemas.py:_dataframe_uid`; `test_request_contract.py:test_s2_t12_from_dataframe_non_integral_uid_raises` | B-4, B-5 |
| O-6 | VERIFIED | Owner record. Owner action on consumer callers stays open. | Owner check before S6 |
| O-7 | OWNER APPROVAL REQUIRED | Section E, O-7 input | Owner chooses (a) or (b). C.15 waits. |
| O-8 | VERIFIED | Owner record | none |
| D001, D002 | VERIFIED | C.5 placement (O-4) keeps the one-way rule | none |
| D006 | VERIFIED | Goal of the spec | none |
| D094, D095 | VERIFIED | C.7 uses the D094 type-guard form and the D095 no-mutation form | B-8, R-11 |
| D096 | VERIFIED | P-001 clause 5 supersedes it; consistent with O-2 (b) | Owner promotes P-001 |
| D097 | VERIFIED | `utils/dataframes.py` has no caller (search in section D) | none |
| D098, D100 | VERIFIED | C.8 uses them as a precedent only. D098 scope is `netting_set_id` of portfolio ingest. | none |
| D099 | VERIFIED | Not cited by C.8; row positions do not change | none |
| D106 | VERIFIED; conflict with C.4 | D106.5: `RunProvenance.engine_version` is `"<V>+src.<H>"` (`provenance.py:compute_engine_version`, `source_tree_digest`) | B-1 |
| D107 | VERIFIED | `schemas.py:_float_or_int` stores an int input as float, so `rho` is float or None (F-13 holds) | none |
| D-C15 | VERIFIED | Reproducibility is environment-relative | B-1 |
| D017, D064, D076 | NOT VERIFIED | S7 clauses not covered | Round 2 |

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | AST scan of the rebuilt src tree: 57 files import pandas, equal to A_INVENTORY.md section 1. The index MANIFEST lists 508 `.py` files under the two trees; the clause states 660. No dynamic `import_module("pandas")` in src. | MATCH WITH REQUIRED CHANGE | R-4 |
| C.2 | `pyproject.toml`: `pandas>=2.0,<3.1` and `pandas-stubs>=2.0` in `[project].dependencies`; groups dev, scripts, notebooks exist; seaborn in scripts and notebooks | MATCH | none |
| C.3 | `riskquant_lab/plotting/credit_var.py` and `plot_utils.py` import matplotlib at module level. matplotlib is only in the scripts and notebooks groups. | MATCH WITH REQUIRED CHANGE | R-8 |
| C.4 | `provenance.py:compute_engine_version`; `schemas.py:RunProvenance.engine_version` | CONTRADICTION | B-1 |
| C.4 (set R) | `tests/solutions/credit_var/test_result_contract.py:_book`, `_BOOK_KEYS` | MATCH WITH REQUIRED CHANGE | R-1 |
| C.5 | `validate.py:_semantic_mf_gaussian` lines 612 to 614; `schemas.py:CreditVarRequest._check_loading_sum_sq` | MATCH WITH REQUIRED CHANGE | R-5 |
| C.6 | Spec only; golden recorded in S1 | NOT VERIFIED | none |
| C.7 | Entries and callers in section D; `validate.py:validate_dtypes`, `validate_no_required_nulls`, `validate_mf_loadings_structural` | MATCH WITH REQUIRED CHANGE | B-2, B-8, R-11 |
| C.8 | `masters/obligors/ingest.py:_rating_id`, `_sector_id`, `_country_id`, `_parent_uid`, `_i64`, `_f64` | MATCH WITH REQUIRED CHANGE | R-9, R-10 |
| C.9 | `masters/*/ingest.py` sort on the key with `kind="stable"`; sandbox: `np.argsort(kind="stable")` and a Polars sort with `maintain_order=True` give `[1, 2, 3, 0]` for `[3, 1, 1, 2]` | MATCH WITH REQUIRED CHANGE | R-10 |
| C.10 | `run.py:_obligor_frame`, `_loadings_frame`; `recipes.py:assemble_inputs` and every assembler | MATCH WITH REQUIRED CHANGE | B-3 |
| C.11 | `schemas.py:CreditVarRequest.from_dataframe`, `_dataframe_uid` | CONTRADICTION | B-4, B-5 |
| C.12 | `solutions/credit_exposure/handoff.py:attach_exposure_ead` (has a `uid_col` keyword); `schemas.py:CreditExposureRequest.from_dataframe`; `run.py:_trades_frame` and four other row-dict frames | MATCH WITH REQUIRED CHANGE | B-7, R-13 |
| C.13 | `utils/printer.py:print_df` uses `select_dtypes`, `to_frame`, `to_string`; `print_dict_of_dfs` uses `df.T` | MATCH | R-14 |
| C.14 | `batch.py:IngestBatch.to_df` builds columns that mix int values and the text "NONE" (`_debug_mask_int_sentinel`) | MATCH WITH REQUIRED CHANGE | R-12 |
| C.15 | Not covered | NOT VERIFIED | O-7 |
| C.16 | Not covered | NOT VERIFIED | none |
| C.17 | D106.1 to D106.3; `pyproject.toml` version 0.2.0 | MATCH | none |
| P-001 | Clauses 1 to 6 follow C.1 to C.17 | MATCH WITH REQUIRED CHANGE | follows B-1 to B-8 |

## D. Blast radius of shared interfaces

Search: Python AST walk for `Name`, `Attribute` and `ImportFrom` nodes over the rebuilt `src/` (every src profile), `tests/` (tests_credit and tests_solutions_credit_var only) and `scripts/` (scripts profile). Equivalent command: `rg -n "\b<symbol>\b" src tests scripts`. Files not searched: every test file outside the two attached tests profiles.

| Interface | Slice | Callers (`path:symbol`) |
|---|---|---|
| `compile_obligor_master` | S4 | `src/riskquant/solutions/credit_var/recipes.py:_masters`; `scripts/demos/credit/jtd/e2e/demo_e2e_full_jtd_portfolio_{core,mf_chi,mf_chi_streaming}.py:run_demo`; `scripts/demos/credit/jtd/validation/demo_validate_{asset_correlation,beta_binomial_jtd_engine,forced_default_engine,independent_bernoulli_engine,marshall_olkin_engine,mf_gaussian_copula_jtd_engine,sf_copula_jtd_engine_sf_gaussian,sf_copula_jtd_engine_sf_students_t,stress_regime_bernoulli_engine}.py:run_demo` |
| `compile_netting_set_master` | S4 | none outside its module (F-16 holds) |
| `compile_credit_factor_universe` | S4 | `recipes.py:_universe`; `src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_universe_master.py:build_canonical_credit_factor_universe`; the 3 e2e demos and 8 validation demos above (`run_demo`), not `demo_validate_mf_gaussian_copula_jtd_engine.py` |
| `compile_obligor_ead` | S4 | `recipes.py:_ead` |
| `compile_add_on_factors` | S3 | `src/riskquant/solutions/credit_exposure/recipes.py:_add_on`; `scripts/demos/credit/exposure/demo_e2e_cem_trade_to_exposure.py:main`; `scripts/demos/credit/exposure/demo_e2e_saccr_trade_to_exposure.py:main` |
| `compile_saccr_config` | S3 | `credit_exposure/recipes.py:_assemble_saccr`; `demo_e2e_saccr_trade_to_exposure.py:main` |
| `JtdIngestBundle`, `from_instrument_frame` | S5 | `recipes.py:_assemble_forced_default`, `_assemble_independent_bernoulli`, `_assemble_beta_binomial`, `_assemble_stress_regime_bernoulli`, `_assemble_sf`, `_mf_universe_and_bundle`, `_assemble_beta_binomial_grouped`, `_assemble_stress_regime_bernoulli_grouped`, `_assemble_marshall_olkin`; `minimal_mixed.py:build_canonical_mixed_jtd_bundle`; 12 JTD demos (`run_demo`); `scripts/tests/credit/jtd/test_ingest_bundle_validation.py:_expect_ok`, `_expect_fail`, `main`; `tests/credit/test_dataclass_equality_contract.py:test_identity_equality_holds_whatever_the_arrays_contain` |
| `assemble_inputs` (not named in spec section 1) | S4, S5 | `src/riskquant/solutions/credit_var/run.py:_simulate`; `tests/credit/solutions/test_catalog_default_ownership.py:_assembled_universe`; `tests/credit/solutions/test_marshall_olkin_default_ownership.py:_effective_systemic_pd_at_construction`; `tests/solutions/credit_var/test_request_contract.py:_engine_accepts` |
| `CreditVarRequest.from_dataframe` | S6 | the 15 `scripts/solutions/credit_var/demo_e2e_*.py`; `tests/solutions/credit_var/test_request_contract.py` (S2.T12 tests, 14 uses); `tests/credit/factors/test_batch_size_reproducibility.py`; `tests/credit/jtd/test_beta_binomial_contract.py` (6 uses); `tests/credit/solutions/test_catalog_default_ownership.py`; `tests/credit/solutions/test_marshall_olkin_default_ownership.py`; `tests/credit/solutions/test_marshall_olkin_request_validation.py` (2 uses) |
| `CreditExposureRequest.from_dataframe` | S3 | `src/riskquant/solutions/credit_exposure/golden_master_credit_exposure.py`; `tests/credit/exposure/saccr/test_saccr_trade_level_contract.py` (4 uses) |
| `attach_exposure_ead` | S3 | re-export in `src/riskquant/solutions/credit_exposure/api.py` only |
| `print_df` | S2 | about 40 files: 23 src fixture modules and `printer.py:print_dict_of_dfs`; 17 scripts (the JTD demos, `demo_spot_rc_end_to_end.py`, `demos/e2e/_fixtures.py`, `demo_e2e_sensi_based_var.py`, `demo_snapshot.py`, `demo_e2e_mf_gaussian.py`, `demo_e2e_mf_students_t.py`). F-17 states 38. |
| `print_dict_of_dfs` | S2 | `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py:<module>` only. F-17 states 2. |
| `IngestBatch.to_df`, `.df` | S2 | `demo_spot_rc_end_to_end.py:<module>` (line 212) |
| RiskFactorMatrix and snapshot views | S7 | section E, O-7 input |

## E. Hidden dependency findings

E-1. Test files that import pandas (AST, attached profiles): `tests/solutions/credit_var/test_request_contract.py`, `tests/credit/test_dataclass_equality_contract.py`, `tests/credit/factors/test_batch_size_reproducibility.py`, `tests/credit/solutions/test_catalog_default_ownership.py`, `tests/credit/solutions/test_marshall_olkin_default_ownership.py`, `tests/credit/solutions/test_marshall_olkin_request_validation.py`, `tests/credit/jtd/test_beta_binomial_contract.py`, `tests/credit/jtd/kernels/test_mf_factor_independence.py` (inside functions; calls `validate.py:_assert_distinct_columns`), `tests/credit/exposure/saccr/test_saccr_trade_level_contract.py` (inside functions). The last two are inside the S5 and S3 Files globs. Consequence: B-2, B-6.

E-2. `test_request_contract.py:_engine_grouped_sum` is a pandas group-sum oracle for WS_CREDIT_VAR_API S2.T6. It fails when pandas is not installed (S8). No slice owns it.

E-3. `solutions/credit_var/recipes.py:_assemble_beta_binomial_grouped` (line 528) and `_assemble_stress_regime_bernoulli_grouped` (line 581) read `obligor_table["group"].astype(str)`. `pl.Series` has no `astype`. Consequence: B-3.

E-4. Polars row-dict inference (sandbox, polars 2.1.0). `pl.DataFrame(rows)` with 120 rows `{"v": None}` then one `{"v": 2.5}` raises `ComputeError`. With 150 rows of int 1 then one 1.5, the column is Int64 and the last value is 1 without an error. `infer_schema_length=None` gives Float64. `credit_exposure/run.py:_trades_frame`, `_netting_sets_frame`, `_add_on_frame`, `_hedging_sets_frame` and `_correlations_frame` build frames from `model_dump()` row dicts with optional fields. Consequence: B-7. Column lists (`run.py:_obligor_frame`) scan the full list: `[None]*150 + [1.5]` gives Float64 (sandbox).

E-5. `jtd/ingest/validate.py:validate_no_required_nulls` uses `Series.isna()`, which is true for float NaN. Polars `is_null()` is false for NaN. `validate_dtypes` and `validate_mf_loadings_structural` use `pd.api.types.is_integer_dtype`, `is_numeric_dtype` (true for bool) and `is_object_dtype or is_string_dtype`. Polars has no object dtype, and `Boolean` is not numeric. Consequence: B-8.

E-6. `pl.Series.to_numpy()` returns a read-only array for Float64 (sandbox). `masters/obligors/ingest.py:_i64`, `_f64`, `masters/netting_set/ingest.py:_i64`, `_f64`, `obligor_ead/ingest.py` and `jtd/ingest/_shared.py:_i64` do not copy a read-only array. Only `_shared._f64` copies. Section 7 of the spec names `_shared._f64` only. A read-only array that reaches a kernel typed for writable arrays raises at run time. C.4 detects it (loud). Consequence: R-10.

E-7. `scripts/demos/market_data/risk_factors/demo_snapshot.py:<module>` (line 68) and `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py:<module>` (line 174) call `print_df(rf_lvl_snapshot.df.T, index=True, ...)`. `pl.DataFrame` has no `T` attribute (sandbox). `demo_snapshot.py` imports no pandas and is in no S7 Files list. Under O-7 (a) and under O-7 (b) the demo breaks.

E-8. `RiskFactorMatrix.plot` (`matrix.py` line 140) calls `self.to_df().plot(**kwargs)`. F-9 omits it. `pl.DataFrame.plot` exists but has a different backend and signature.

E-9. `scripts/demos/credit/jtd/validation/demo_validate_mf_gaussian_copula_jtd_engine.py:run_demo` imports no pandas. It passes the fixture frames `CANONICAL_OBLIGOR_CREDIT_INGEST_DF`, `CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF` and `CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF` to S4 and S5 entries. The S5 glob `scripts/demos/credit/jtd/**` covers it; the A_INVENTORY method (import scan) does not.

O-7 input (STOP AND ASK; not resolved here):

* Members with pandas types. `src/riskquant/market_data/risk_factors/matrix.py`: `RiskFactorMatrix.to_df`, `head`, `tail`, `plot`, `index_dates`, `index_factors`, `df`. `src/riskquant/market_data/risk_factors/snapshot.py`: `RiskFactorLevelSnapshot.to_df`, `to_series`, `df`, `sr`.
* Search 1: `rg -n "\.(index_dates|index_factors|to_series|sr)\b" src tests scripts`. Hit: `snapshot.py:RiskFactorLevelSnapshot.sr` (calls `self.to_series`). No other caller.
* Search 2: `rg -n "\.(to_df|df|head|tail|plot)\b" src tests scripts`, then a receiver check in the 29 files that name `RiskFactorMatrix`, `RiskFactorLevelSnapshot`, a builder, `get_snapshot`, `from_snapshot` or the canonical history. Hits on a matrix or snapshot receiver: `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py:<module>` (`rf_lvl_snapshot.df`, line 174); `scripts/demos/market_data/risk_factors/demo_snapshot.py:<module>` (`rf_lvl_snapshot.df`, line 68). Both receivers come from `compile_snapshot.py:compile_risk_factor_level_snapshot_from_levels_dict` (Inferred: it returns `RiskFactorLevelSnapshot`). Other hits are internal delegation inside `matrix.py` and `snapshot.py`, or other types (`scenario_models/api.py:_build_parametric` reads `dist.df`, a Student-t parameter).
* Result for src and scripts: `RiskFactorMatrix` views have no caller outside `matrix.py`. `RiskFactorLevelSnapshot.df` has 2 script callers. [Observed]
* Not searched: tests. The owner choice needs `context_riskquant_tests_market_data.txt`, `context_riskquant_tests_simlab.txt`, `context_riskquant_tests_e2e.txt`, `context_riskquant_tests_solutions_risk_run.txt`, `context_riskquant_tests_solutions_scenario_models.txt` and `context_riskquant_tests_solutions_run_identity.txt`. C.15, S7.T3 and the S7 Files list wait for them.

## F. Findings and required changes

Carried from earlier rounds: none.

BLOCKING:

* **B-1 (C.4, S1.T3, S8.T3; D106.5, D-C15).** Observed: the C.4 comparison excludes only `provenance.generated_at`. `engine_version` is `"<V>+src.<H>"`, where H hashes every `.py` file of the package. Consequence: each slice changes src, and S8 sets V to 0.2.1, so C.4 fails by construction from S1. Required: exclude `provenance.engine_version` as well. Record and compare with equal `numpy_version`, `scipy_version` and `numba_version`; S8.T3 compares in the same environment with pandas removed.
* **B-2 (C.7, C.10, C.11; S1, S4, S5, S6).** Observed: test files outside the Files lists pass pandas frames to changed entries (E-1, E-2, section D). `test_request_contract.py:_engine_accepts` breaks in S4 and S5, `_engine_grouped_sum` breaks in S8, and only S6 lists the file. `tests/credit/solutions/**`, `test_batch_size_reproducibility.py` and `test_dataclass_equality_contract.py` are in no slice. `test_beta_binomial_contract.py` changes with S6, and S6 has no FIRST step. Consequence: the Exit "full suite passes" of S4, S5 and S6 cannot be met within their Files lists. Required: add each file to the slice that changes its entry, give S6 a FIRST step with STOP AND ASK, and add `context_riskquant_tests_credit.txt` to the S6 Attach field.
* **B-3 (C.10, S4).** Observed: `recipes.py:assemble_inputs` (in `recipes.__all__`) changes type in S4 (`obligor_table`) and S5 (`loadings_table`). Section 1 and C.10 do not name it. The S4 member list omits `_assemble_beta_binomial_grouped` and `_assemble_stress_regime_bernoulli_grouped` (E-3). Consequence: two catalog keys fail in S4. Required: name the `assemble_inputs` signature in C.10 per slice, and add the two members to S4 Files.
* **B-4 (C.11, O-5, S6.T3).** Observed: O-5 (b) admits only integer dtypes for `obligor_uid`; any other dtype raises TypeError. S6.T3 expects a `pl.Float64` uid column with 7.0 to be admitted and 7.9 to raise R. `test_s2_t12_from_dataframe_non_integral_uid_raises` and `test_s2_t12_from_dataframe_non_integral_loadings_uid_raises` expect `ValidationError` at `("obligor_uid",)`. Sandbox: `pl.Series([7.0, 7.9]).cast(pl.Int64)` gives `[7, 7]`. Consequence: the clause and its test contradict each other. Required (owner choice): (i) a float uid column raises TypeError, and the S2.T12 tests change in S6; or (ii) a float uid column is admitted after an integral check, and a non-integral cell raises R.
* **B-5 (C.11, O-5).** Observed: an optional `group` or `rho` column that holds only None has dtype `pl.Null` (F-12). O-5 (b) as written raises TypeError for it. v0.2.0 admits it (`pd.notna`). The rule for a null cell in a required column, and for a null cell in `group` or `rho`, is not stated. Consequence: a request that v0.2.0 admits is refused, without a clause. Required (owner approval): state that `pl.Null` is admitted for `group` and `rho` (all rows absent), state the null-cell rules, and add the cases to S6.T3.
* **B-6 (S8 Exit, C.2).** Observed: the S8 Exit requires the full suite to pass with pandas uninstalled, and C.2 removes pandas from the dev group. No clause or slice removes pandas from test files that call no changed entry (E-2). D094 cites a differential corpus against pandas runtimes in `tests/portfolio/ingest` (not read). Consequence: the S8 Exit cannot be met, or S8 must change files outside its Files list. Required: add a clause for the tests tree (no test imports pandas, or an owner-named exception) and assign each file to a slice. The list for the tests profiles not attached is NOT VERIFIED.
* **B-7 (C.12, S3; C.10).** Observed: E-4. `credit_exposure/run.py` builds five frames from `model_dump()` row dicts with optional fields. A direct Polars port raises `ComputeError` on a valid request when the first 100 rows have None in an optional float field, and can truncate a float to an int when the first 100 rows hold ints. Consequence: a valid production request fails, or a value changes. Required: C.12 and C.10 state that every frame built from request rows has an explicit schema (or is built column by column with stated dtypes); S3.T2 adds a book of more than 100 rows in which the first 100 rows have None in `volume`.
* **B-8 (C.7, S5.T3).** Observed: E-5. The S5 admission checks depend on pandas dtype predicates and on `isna()`, which includes NaN. C.7 states only the frame type. Consequence: a NaN in a required column can pass the null check, and the dtype admission changes without a clause (WS_TEMPLATE section 1.5: a change of admission is HEAVY). Required: state the Polars rules in C.7: null means null or NaN for float columns; the key column is an integer dtype; value columns are integer or float dtypes (state Boolean); factor-name columns are `pl.Utf8` (state `pl.Categorical`, `pl.Enum`). Add a NaN case and a Boolean case to S5.T3.

REFINEMENT:

* R-1: The F-21 builder is `tests/solutions/credit_var/test_result_contract.py:_book` with `_BOOK_KEYS` (N = 4, S = 2_000, seed 4242). It already holds `beta_binomial_grouped` with 2 groups, `forced_default` with `in_stress_test` obligors 30 and 40 and `force_default` obligor 20, and `sf_gaussian` with `rho` on 1 obligor. Name it in C.4 and keep only the C.4 additions that it does not hold.
* R-2: S2.T12 builds pandas frames (`test_request_contract.py` lines 823, 854, 855, 875, 895). F-21 is closed.
* R-3: The spec Snapshot field gives 19d097a. The index gives 1ae80e3 with the same content_hash. Update the field.
* R-4: The C.1 count is 660. The MANIFEST lists 508. S8.T1 enumerates by walk, so the count is information only.
* R-5: `_semantic_mf_gaussian` reports rejected uids in ascending order (pandas groupby `sort=True`; sandbox: index `[10, 20, 30]` for keys `[30, 10, 20]`). C.5 returns first-seen order, and S5.T3 needs the v0.2.0 list. State in C.5 that the engine caller sorts the rejected uids.
* R-6: Add `RiskFactorMatrix.plot` to F-9 and to O-7 (E-8).
* R-7: S5 states "13 files" under `scripts/demos/credit/jtd`. The profile holds 15; 11 import pandas. The demo-full gate count must match.
* R-8: C.3 and S8.T3 need matplotlib in the environment, or must exclude `riskquant_lab/plotting`. State the environment of the S8 gate.
* R-9: C.8 omits overflow. Sandbox: `pl.Series([200]).cast(pl.Int8)` raises `polars.exceptions.InvalidOperationError`; pandas `astype("Int8")` raises `TypeError`. State the exception type. C.8 also omits a fractional float or a null in the required integer columns (`uid`, `obligor_uid`, `factor_id`, `subclass_uix`, `hedging_set_uix`); today numpy truncates or wraps without an error. Add these cases to S4.T2 or name them as intended changes.
* R-10: C.9 compares value and dtype only. Add the writeable flag, or require a copy at each `to_numpy` site (E-6). `factor_id` is cast int64 to int32: numpy wraps 2**40 to 0 without an error; a Polars strict cast raises (sandbox). Name it as an intended change.
* R-11: C.7 tests one frame per entry. `compile_saccr_config` takes two frames, and `JtdIngestBundle.from_instrument_frame` takes `frame` and `mf_loadings_df`. Test each frame argument.
* R-12: C.14 and S2.T2 do not state the column dtypes of `to_df()` under O-2 (b). Today `und_ids`, `vol_ids` and the FX id columns mix int values and the text "NONE". State null or `pl.Utf8` for these columns.
* R-13: C.12 omits the `uid_col` keyword of `attach_exposure_ead`, and gives no cell admission rule for `CreditExposureRequest.from_dataframe` (today `int(...)` truncates a float). Owner may align it with O-5.
* R-14: `print_dict_of_dfs` transposes with `df.T`; `pl.DataFrame` has no `T`. The backlog item covers it.
* R-15: S3 has no v0.2.0 record for `run_credit_exposure` results. S3.T2 compares two new paths. `golden_master_credit_exposure.py:test_frozen_case` is the only independent oracle. Consider a v0.2.0 JSON record as in C.4.
* R-16: The v0.2.0 tag is not visible in the snapshot. `pyproject.toml` holds version 0.2.0.
* R-17: Section 7 says `_shared._f64` mitigates read-only arrays. Other `to_numpy` sites do not copy (E-6).

Owner approval only: O-7; the options of B-4 and B-5; the tripwire decision (section A).

Profiles for the next round (C.15, C.16, and the test sweep of B-6). More than 5 profiles, so split by clause range (section 1.6):
* Part 1 (C.15, C.16, O-7 test callers): `context_riskquant_market_data.txt`, `context_riskquant_ingest_fixtures.txt`, `context_riskquant_utils.txt`, `context_riskquant_tests_market_data.txt`, `context_riskquant_tests_e2e.txt`.
* Part 2 (O-7 test callers, B-6 sweep, S8): `context_riskquant_tests_simlab.txt`, `context_riskquant_tests_solutions_risk_run.txt`, `context_riskquant_tests_solutions_scenario_models.txt`, `context_riskquant_tests_solutions_run_identity.txt`, `context_riskquant_tests_portfolio.txt`.
* Part 3 (B-6 sweep, S8): `context_riskquant_tests_root.txt`, `context_riskquant_tests_utils.txt`, `context_riskquant_tests_config.txt`, `context_riskquant_tests_ingest_fixtures.txt`, `context_riskquant_tests_sensitivities.txt`.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no (B-4, B-6) |
| Consistent with `DECISIONS.md` | no (B-1 against D106.5) |
| Implementable with existing architecture | yes |
| Blast radius bounded | no (B-2, B-3; O-7 test callers not searched) |
| Existing behavior protected | conditionally (B-5, B-7, B-8) |
| Value case holds: the smallest alternative fails, with evidence | yes (57 src modules import pandas, AST scan; S1 alone removes only the cap) |
| Remaining issues are approvals, not technical gaps | no |

Verdict: NOT READY
