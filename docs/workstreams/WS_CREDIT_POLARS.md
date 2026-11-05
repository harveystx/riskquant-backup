# WS_CREDIT_POLARS: Credit edge on Polars and one admission loading sum (v1)

Path: docs/workstreams/WS_CREDIT_POLARS.md
Class: HEAVY
Status: FROZEN
Revision: 4
Revision note: Owner decision of 2026-11-05 (S1 implementer finding): C.5 states the pandas 3.0.x compensation reset and the non-finite input guard; the non-finite total admission defect goes to S5. Revision 3 was frozen by owner decision. No audit round.
Date: 2026-11-05
Branch: ws/credit-polars
Snapshot: SNAPSHOT commit=19d097a dirty=false content_hash=c00f240898f32d74. The index read by the round 1 audit shows commit=1ae80e3 with the same content_hash; the two snapshots hold the same content.
Drafted by: investigator
Independent review: PENDING (round 1 NOT READY; the owner freezes Revision 3 without round 2, decision 2026-11-05)
Audit rounds used: 1 of 2
Code merged: no
DECISIONS.md reconciled: NO
Evidence limits: No repository gate, test, demo or benchmark ran. Code facts come from the 19d097a profiles (src, scripts, lab) rebuilt in the investigator sandbox. Facts F-21 to F-29 come from R_FREEZE_RECONCILIATION_R1.md, which read context_riskquant_tests_credit.txt and context_riskquant_tests_solutions_credit_var.txt; the drafter did not read those profiles. Other tests profiles were read by no chat; each slice has a FIRST step with STOP AND ASK. Facts marked "sandbox" ran with Python 3.12.3, numpy 2.5.4, polars 2.1.0, pandas 3.0.6, and venvs with pandas 2.2.3, 3.0.2 and 3.1.1; numba and pydantic were not installed, so no engine ran. The v0.2.0 tag is not visible in the snapshot; the WS_CREDIT_VAR_API closure record names it. riskquant-emissary and riskquant-excel were not read.

## 1. Goal and scope

Goal: The credit edge of riskquant (credit ingest, `solutions.credit_var`, `solutions.credit_exposure`) uses Polars at the edge and numpy arrays in core, the book admission and request rule 14 use one frame-free loading sum, the pandas upper bound leaves pyproject.toml, and the results of `riskquant.solutions.credit_var.api` stay bitwise equal for all 13 catalog keys. pandas stays a declared dependency until WS_PANDAS_REMOVAL.

Value case:
- Benefit: the cap pandas>=2.0,<3.1 (D106 pins line) goes, so pandas 3.1 installs (F-3). Rule 14 and the engine admission check call one function (F-2, F-5). 44 of the 57 src modules that import pandas (F-1) stop the import; the credit edge follows D006 and D094. The WS_CREDIT_VAR_API backlog item on bool, Decimal and str cells of `from_dataframe` closes (C.11).
- Cost proxies: 6 slices. 44 src modules; 31 scripts files (15 credit_var demos, 12 JTD demos, 2 exposure demos, 2 config and rc demos, 1 scripts test); 10 test files named by the round 1 audit (F-22). Shared interfaces that change type: 7 credit ingest entries and the `JtdIngestBundle` slots, `recipes.assemble_inputs`, 2 `from_dataframe` members, `attach_exposure_ead`, `print_df`, `print_dict_of_dfs`, `IngestBatch.to_df` and `.df`. 1 new D entry (P-CREDIT_POLARS-001), which supersedes D096 and edits the D106 pins line. 1 audit round used.
- Smallest alternative and why it fails: S1 alone. It removes the cap and gives one admission sum, but 55 modules keep pandas (F-1) and the credit edge stays against D006 and D094. WS_PANDAS_REMOVAL cannot remove the dependency while the credit edge takes pandas frames.
- Case against: v1 does not meet the estate goal "no pandas in riskquant": 13 src modules keep pandas (A_INVENTORY.md rows tagged S7) and pyproject.toml keeps `pandas>=2.0` and `pandas-stubs`. Two workstreams cost two closure cycles. Mitigation: section 7 lists every item that WS_PANDAS_REMOVAL takes over.
- Recommendation at drafting: PROCEED SMALLER (owner descope, 2026-11-05) [Proposed]

In scope:
- One frame-free per-obligor loading sum for admission, called by the engine and by rule 14; removal of the pandas upper bound (S1).
- `riskquant.utils.printer`, `riskquant.utils.dataframes`, `riskquant.config.static_config.audit`, `IngestBatch.to_df` and `.df`, the demo import smoke tool (S2).
- Credit exposure chain: `credit/exposure/{add_on,saccr}/ingest.py`, `solutions/credit_exposure`, `riskquant_lab/fixtures/credit_exposure.py` (S3).
- Credit masters, `credit/exposure/obligor_ead/ingest.py`, the obligor-table side of `solutions/credit_var` (S4).
- JTD ingest: `credit/jtd/ingest/*`, the 9 builders (frame reads only), the bundle side of `solutions/credit_var` (S5).
- `CreditVarRequest.from_dataframe` on `pl.DataFrame`, the 15 credit_var demos, release 0.2.1 (S6).

Out of scope:
- Every item of section 7 "WS_PANDAS_REMOVAL": the market-risk edge, `utils/dates.py`, the portfolio and market-data fixtures, the e2e demos, the test-tree sweep, the removal of pandas and pandas-stubs from pyproject.toml, and the package enforcement tests.
- The credit_var JSON contract: the 22 names of D106.1, the request and result golden schemas, the result values (C.4).
- riskquant-emissary, riskquant-excel, the ruff upgrade.
- Any change of a kernel, of `credit/jtd/ingest/builders/*` arithmetic (F-7), or of `riskquant.jit`.
- The text of D-C15, D094, D095, D097 to D100, D106 clauses 1 to 9 and D107.

Owner decisions in force:
- 2026-11-03 (WS_CREDIT_VAR_API, pandas cap): WS_CREDIT_POLARS replaces the pandas group sum with one shared core loading-sum function and removes the cap.
- 2026-11-04 (owner rules): run the full demo set only when a slice changes numbers or demo files; otherwise use the demo import smoke. Every implementer prompt states "Read DECISIONS.md before you write code" and "Use ruff 0.15.21 for format and lint".
- 2026-11-05 (O-1 to O-6, O-8): see section 5.
- 2026-11-05 (tripwire, descope): v1 is S1 to S6. S7 of Revision 2 (C.15, C.16, O-7), the test-tree sweep (B-6) and S8 of Revision 2 (C.1, C.2, C.3, P-CREDIT_POLARS-001 clause 1) move to WS_PANDAS_REMOVAL. Round 1 covered every clause that stays; no round 2. The owner freezes Revision 3 by explicit decision.
- 2026-11-05 (release): 0.2.1 (O-6 (a)). pandas stays in pyproject.toml until WS_PANDAS_REMOVAL. S1 removes the upper bound <3.1. S6 carries the version bump.
- 2026-11-05 (B-1): C.4 excludes `provenance.generated_at` and `provenance.engine_version`, and compares only at equal numpy, scipy and numba versions.
- 2026-11-05 (B-4, option (ii)): a float `obligor_uid` column is admitted after an integral check; a non-integral cell raises R at `("obligor_uid",)`. The WS_CREDIT_VAR_API S2.T12 tests keep their assertions.
- 2026-11-05 (B-5): a `pl.Null` column for `group` or `rho` means absent for all rows; a null cell in `group` or `rho` means absent for that obligor; a null in a required column raises R at the field path.
- 2026-11-05 (B-8): for float columns, null means null or NaN; Boolean is not a numeric dtype and raises; factor-name columns admit `pl.Utf8`, `pl.Categorical` and `pl.Enum` (D094).
- 2026-11-05 (R-9): integer overflow raises TypeError, as a fractional float does.
- 2026-11-05 (R-10, R-17): every `to_numpy` site that feeds an engine array gives a writeable, C-contiguous array.
- 2026-11-05 (B-2, B-3, B-7, R-1 to R-8, R-11 to R-16): applied as the report states, where they concern S1 to S6.
- 2026-11-05 (P-CREDIT_POLARS-001): keeps only the clauses that v1 delivers.
- 2026-11-05 (freeze): the owner freezes Revision 3 by explicit decision under the tripwire, with no audit round 2; round 1 covered every clause that v1 keeps. Accepted deviations of the Revision 3 author: the change list stays outside the file (WS_TEMPLATE 1.3); C.12 admits CreditExposureRequest.from_dataframe cells by dtype, as C.11 does; the S1 gate also runs under pandas 3.1.x; the C.4 request set is named Q; A_INVENTORY.md stays at Revision 1. The S1 golden records are made on the owner machine with a recorder tool outside the repository (_gates), on a v0.2.0 worktree.
- 2026-11-05 (S1, C.5 reset and guard; Revision 4): the C.5 function applies the pandas 3.0.x reset of a non-finite compensation to 0.0, and raises ValueError for a NaN or infinite loading. Reason (S1 implementer, Observed in the sandbox): without the reset, loadings [1e154, 1e154, 0.1] give NaN and the obligor is admitted; pandas 3.0.6 gives inf and rejects it; with the reset the function equals pandas 3.0.6 bit for bit on 20,000 near-bound groups. The guard is stricter than pandas, on a path that no caller reaches.
- 2026-11-05 (S5, non-finite total): finite loadings whose squares overflow can give a NaN total (for example [1e154, 1e154, 1e154]: inf + (-inf)); the engine check `total > bound` then admits the obligor. S5 changes the engine admission so that a total that is not finite rejects the obligor, and S5.T3 adds that case. Rule 14 is not exposed (loadings in [-1, 1]).
- 2026-11-05 (S1 mypy baseline): validate.py has 15 mypy errors in untyped functions that exist at 19d097a, on lines that S1 does not change. The S1 mypy gate accepts exactly these 15 errors when the same 15 appear at 19d097a; S5, which rewrites validate.py, removes them.
- 2026-11-05 (S1 imports): S1 adds the imports that its new calls need (schemas.py: numpy and loading_sum_sq; test_request_contract.py: pathlib.Path).

Depends on: WS_CREDIT_VAR_API (CLOSED; D106, D107).
Blocks: WS_PANDAS_REMOVAL.

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | 57 modules under `src/` import pandas: 55 at module level; `utils/dates.py` under TYPE_CHECKING and inside two functions; `portfolio/ingest/batch.py` under TYPE_CHECKING and inside `to_df`. 24 further files in pandas_inventory.txt name pandas in comments or docstrings only. `riskquant_lab/fixtures/credit_var.py` names a variable `pd` and imports no pandas. 35 files under `scripts/` import pandas. v1 covers 44 of the 57; the 13 rows tagged S7 in A_INVENTORY.md belong to WS_PANDAS_REMOVAL. | AST walk over the rebuilt snapshot; A_INVENTORY.md; R1 section C (C.1 row) | Observed |
| F-2 | `credit/jtd/ingest/validate.py:_semantic_mf_gaussian` decides sum(loading^2) <= 1 + 1e-9 with `pd.Series(sq).groupby(uid).sum()`. | `validate.py` lines 612 to 614 | Observed |
| F-3 | pandas 2.2.3 and 3.0.2 evaluate that group sum as a Kahan compensated sum in row order from (0.0, 0.0) with compensation `(t - total) - y`: 0 bitwise mismatches against a Python loop over 14,467 groups near the bound. pandas 3.1.1 gives 25 mismatches. | sandbox probe | Observed |
| F-4 | Polars `group_by().agg(pl.col().sum())` is not that sum: it equals the Kahan loop in 216 of 300 single-group cases. | sandbox probe, polars 2.1.0 | Observed |
| F-5 | `schemas.py:CreditVarRequest._check_loading_sum_sq` (rule 14) reimplements the Kahan state per uid in request order and uses the first row at which the running sum exceeds the bound for the loc. | `schemas.py` lines 516 to 571 | Observed |
| F-6 | `run._loadings_frame` keeps request order and drops only the rows of force_default obligors; `recipes._mf_universe_and_bundle` keeps the order. | `run.py:_loadings_frame`; `recipes.py:_mf_universe_and_bundle` | Observed |
| F-7 | The engine number `idio_weight = sqrt(max(0, 1 - sum_sq))` uses `np.add.reduceat` over band-sorted rows, a naive sequential sum. It differs bitwise from the Kahan admission sum in 4,924 of about 14,000 groups; the admission decision did not differ in that sample. | `builders/mf_gaussian.py` line 64; `builders/mf_students_t.py` line 68; sandbox probe | Observed |
| F-8 | All other pandas uses on the credit_var path are frame assembly or typed extraction: `pd.DataFrame(...)`, `to_numpy(dtype=...)`, `astype(str)`, `sort_values(key, kind="stable")` on unique keys, `astype("Int8"/"Int32"/"Int64").fillna(sentinel)`, `notna`, `pd.to_numeric(errors="coerce")` on a validated float-or-None column, boolean filter and `drop(columns)` in `bundle.from_instrument_frame`. None sums, merges or deduplicates. | `credit/masters/*/ingest.py`; `credit/exposure/obligor_ead/ingest.py`; `credit/jtd/ingest/{bundle,_shared}.py`; `solutions/credit_var/{run,recipes}.py` | Observed |
| F-9 | `market_data/risk_factors/*.py` import pandas at module level; `RiskFactorMatrix` is on the import path of simlab, risk_run, run_identity and scenario_models. This is the reason for the descope. | Revision 2 F-9; R1 O-7 input | Observed |
| F-10 | D096 (LOCKED) keeps `IngestBatch.to_df` and `.df` as a pandas output adapter. One caller: `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py` line 212. `to_df` builds `und_ids`, `vol_ids`, `fx_u2s_ids`, `fx_k2s_ids` as object columns that mix int values and the text "NONE" (`_debug_mask_int_sentinel`), and `k`, `cp_mult` as float columns with NaN for the sentinel (`_masked_float_copy`). | DECISIONS.md D096; `portfolio/ingest/batch.py:IngestBatch.to_df` | Observed |
| F-11 | pandas `astype("Int8")` raises TypeError on 3.5 and on 200; Polars `cast(pl.Int8)` gives 3 for 3.5 without an error and raises `InvalidOperationError` for 200. numpy `astype(np.int32)` wraps 2**40 to 0 without an error. | sandbox probes; R1 R-9, R-10 | Observed |
| F-12 | `pl.DataFrame({"rho": [None, None]})` has dtype `pl.Null`. `pl.Series.to_numpy()` of an Int64 or Float64 column with nulls returns float64 with NaN, and the array is read-only. `pl.from_dict` refuses a list that mixes str and float NaN. | sandbox probe | Observed |
| F-13 | `run.py:_obligor_frame` builds `group` and `rho` columns from `None` values; `recipes.py:_per_obligor_rho` reads `col.notna()` and `pd.to_numeric(col, errors="coerce")`. After request validation `rho` is a float or None and `group` is a str or None (D107.4). | `run.py`; `recipes.py`; `schemas.py:ObligorRow`, `_float_or_int` | Observed |
| F-15 | `solutions/credit_exposure` uses pandas for frame assembly only: `pd.DataFrame([row.model_dump() ...])` in `run.py` (5 frames), `to_numpy(dtype=...)` and `np.argsort(kind="stable")` in `recipes.py`, `from_dataframe` with `int(...)` casts and `pd.isna` in `schemas.py`, `obligor_table.copy()` in `handoff.py`. `golden_master_credit_exposure.py` holds `test_*` functions; its `groupby().sum()` is a test oracle with a tolerance. | `solutions/credit_exposure/*.py` | Observed |
| F-16 | Callers outside `src/riskquant/credit/`: `compile_obligor_master`: `credit_var/recipes.py:_masters` and 12 JTD demos (3 e2e, 9 validation). `compile_credit_factor_universe`: `recipes.py:_universe`, `minimal_factor_universe_master.py`, 11 of those demos. `compile_obligor_ead`: `recipes.py:_ead`. `compile_netting_set_master`: none. `JtdIngestBundle` and `from_instrument_frame`: `recipes.py` (9 assemblers), `minimal_mixed.py`, the 12 demos, `scripts/tests/credit/jtd/test_ingest_bundle_validation.py`. `compile_add_on_factors`: `credit_exposure/recipes.py:_add_on`, the cem and saccr demos. `compile_saccr_config`: `credit_exposure/recipes.py:_assemble_saccr`, the saccr demo. | grep over src and scripts; R1 section D | Observed |
| F-17 | `utils/printer.py:print_df` accepts `pd.Series | pd.DataFrame` and uses `select_dtypes`, `to_frame`, `to_string`; about 40 files call it, among them src fixtures and scripts that stay on pandas in v1. `print_dict_of_dfs` uses `df.T`; its only caller outside `printer.py` is `demo_spot_rc_end_to_end.py`. `pl.DataFrame` has no `T`. `utils/dataframes.py:validate_string_cols` has no caller. `config/static_config/audit.py` builds pandas frames only to print. | `utils/printer.py`; R1 section D, R-14 | Observed |
| F-18 | The 15 `scripts/solutions/credit_var/demo_e2e_*.py` import pandas and pass `pd.DataFrame` tables to `CreditVarRequest.from_dataframe`. `riskquant_lab/fixtures/credit_var.py:generate_credit_portfolio_data` returns `dict[str, np.ndarray]`. | scripts profile; lab profile | Observed |
| F-19 | pyproject.toml declares `pandas>=2.0,<3.1` and `pandas-stubs>=2.0` in `[project].dependencies`. | pyproject.toml | Observed |
| F-20 | The 13 catalog keys are forced_default, independent_bernoulli, stress_regime_bernoulli, stress_regime_bernoulli_grouped, beta_binomial, beta_binomial_grouped, sf_gaussian, sf_gaussian_basel, sf_students_t, sf_students_t_basel, mf_gaussian, marshall_olkin, mf_students_t. | `solutions/credit_var/catalog.py:MODEL_CATALOG` | Observed |
| F-21 | WS_CREDIT_VAR_API S2.T12 builds pandas frames (`test_request_contract.py` lines 823, 854, 855, 875, 895). `test_s2_t12_from_dataframe_non_integral_uid_raises` and `test_s2_t12_from_dataframe_non_integral_loadings_uid_raises` expect `ValidationError` at `("obligor_uid",)`. | R1 R-2, B-4 | Observed |
| F-22 | Test files that pass pandas frames to a changed entry or use a pandas oracle: `tests/solutions/credit_var/test_request_contract.py` (`_engine_accepts` calls `assemble_inputs`; `_engine_grouped_sum` is a pandas group-sum oracle; the WS_CREDIT_VAR_API S2.T12 tests call `from_dataframe`), `tests/credit/test_dataclass_equality_contract.py` (`JtdIngestBundle`), `tests/credit/factors/test_batch_size_reproducibility.py` (`from_dataframe`), `tests/credit/solutions/test_catalog_default_ownership.py` (`from_dataframe`, `assemble_inputs`), `tests/credit/solutions/test_marshall_olkin_default_ownership.py` (`from_dataframe`, `assemble_inputs`), `tests/credit/solutions/test_marshall_olkin_request_validation.py` (`from_dataframe`), `tests/credit/jtd/test_beta_binomial_contract.py` (`from_dataframe`), `tests/credit/jtd/kernels/test_mf_factor_independence.py` (`validate._assert_distinct_columns`), `tests/credit/exposure/saccr/test_saccr_trade_level_contract.py` (`CreditExposureRequest.from_dataframe`). Only the two tests profiles were searched. | R1 E-1, E-2, section D | Observed |
| F-23 | `pl.DataFrame(rows)` from row dicts infers the schema from the first 100 rows: 120 rows with `{"v": None}` then `{"v": 2.5}` raise `ComputeError`; 150 rows of int 1 then 1.5 give Int64 and the value 1 without an error. A column built from one list scans the full list. | R1 E-4, sandbox | Observed |
| F-24 | `validate.py:validate_no_required_nulls` uses `isna()`, which is true for NaN; Polars `is_null()` is false for NaN. `validate_dtypes` and `validate_mf_loadings_structural` use `is_integer_dtype`, `is_numeric_dtype` (true for bool) and `is_object_dtype or is_string_dtype`. | R1 E-5 | Observed |
| F-25 | `pl.Series.to_numpy()` returns a read-only array. `masters/obligors/ingest.py:_i64`, `_f64`, `masters/netting_set/ingest.py:_i64`, `_f64`, `obligor_ead/ingest.py` and `jtd/ingest/_shared.py:_i64` do not copy a read-only array; only `_shared._f64` copies. | R1 E-6 | Observed |
| F-26 | `recipes.py:_assemble_beta_binomial_grouped` (line 528) and `_assemble_stress_regime_bernoulli_grouped` (line 581) read `obligor_table["group"].astype(str)`. `recipes.assemble_inputs` is in `recipes.__all__`; its callers are `run.py:_simulate` and three test helpers (F-22). | R1 E-3, B-3, section D | Observed |
| F-27 | `RunProvenance.engine_version` is `"<V>+src.<H>"`; H hashes every `.py` file of the package (D106.5). Any src change changes it. | `provenance.py:compute_engine_version`; R1 B-1 | Observed |
| F-28 | `_semantic_mf_gaussian` reports rejected uids in ascending order (pandas groupby `sort=True`). | R1 R-5, sandbox | Observed |
| F-29 | `tests/solutions/credit_var/test_result_contract.py:_book` with `_BOOK_KEYS` (N = 4, S = 2_000, seed 4242) holds `beta_binomial_grouped` with 2 groups, `forced_default` with `in_stress_test` obligors 30 and 40 and `force_default` obligor 20, and `sf_gaussian` with `rho` on 1 obligor. | R1 R-1 | Observed |

## 3. Contract

C.1, C.2, C.3, C.15 and C.16 of Revision 2 belong to WS_PANDAS_REMOVAL (section 7). The other IDs keep their numbers.

Results (S1 to S6). R is `pydantic.ValidationError` and E is `CreditVarEngineError` (D106.8):
- C.4 For the request set Q, the JSON of `run_credit_var` without `provenance.generated_at` and `provenance.engine_version` (F-27) is byte-equal to the record made at tag v0.2.0. Q is `test_result_contract.py:_book(key)` for each key of `_BOOK_KEYS` (F-29), plus mf_gaussian with one obligor at sum(loading^2) = 1 + 0.5e-9 (the WS_CREDIT_VAR_API S2.T6 accepted case). The record stores `numpy_version`, `scipy_version` and `numba_version` of `RunProvenance`; the test compares only when the runtime versions equal the recorded versions, and otherwise fails with a message that names each difference (D-C15). [Proposed] Test: S1.T3; the full suite of S2 to S6 runs it

Loading sum (S1):
- C.5 One function in `src/riskquant/credit/jtd/ingest/loading_sum.py` (O-4) computes the per-obligor loading sum of squares for admission. Input: `obligor_uid` int64 (n,), `loading` float64 (n,), in row order. Per uid the state (total, compensation) starts at (0.0, 0.0); for each row y = x * x - compensation, t = total + y, compensation = (t - total) - y, total = t; when the compensation is not finite it is set to 0.0 (the pandas 3.0.x group_sum rule, GH#53606, GH#60303). A NaN or infinite loading raises ValueError: both callers admit finite loadings only. Output: the running total after each row (n,), the uids in first-seen order and their final totals. The bound comparison stays in the callers. `validate.py:_semantic_mf_gaussian` and `schemas.py:_check_loading_sum_sq` call this function and hold no other summation of loading squares. The engine caller reports the rejected uids in ascending order (F-28). `builders/mf_gaussian.py` and `builders/mf_students_t.py` keep `np.add.reduceat` (F-7). [Proposed] Test: S1.T1, S1.T2, S1.T4
- C.6 The function output equals, under `float.hex()`, the pandas `groupby().sum()` record made with pandas 3.0.x on the owner machine for 2,000 groups near 1 + 1e-9, including groups whose rows in two orders give two different totals. [Proposed] Test: S1.T1
- C.18 After S1, pyproject.toml declares `pandas>=2.0` with no upper bound; `pandas-stubs>=2.0` does not change. No test or src code compares a value with the pandas group sum. [Proposed] Test: S1.T5, S1.T6; S1 gate under pandas 3.1.x

Credit edge (S3, S4, S5):
- C.7 Each of `compile_obligor_master`, `compile_netting_set_master`, `compile_credit_factor_universe`, `compile_obligor_ead`, `compile_add_on_factors`, `compile_saccr_config` (both frame arguments), `JtdIngestBundle.from_instrument_frame` (`frame` and `mf_loadings_df`) and each frame slot of `JtdIngestBundle` accepts `pl.DataFrame` only. A `pandas.DataFrame`, `pl.LazyFrame`, dict or None (where None is not the documented default) raises TypeError before any column is read (D094 form). The caller's frame is never changed (D095 form). JTD frame admission: the key column (`uid`, `obligor_uid`) is an integer dtype; a value column is an integer or float dtype, and `pl.Boolean` is not admitted; a factor-name column is `pl.Utf8`, `pl.Categorical` or `pl.Enum`; in a float column, null means null or NaN. Each refusal raises the exception type that the same check raises at v0.2.0 (ValueError for the `validate.py` checks). [Proposed] Test: S3.T1, S4.T1, S5.T1, S5.T3
- C.8 Column rules of the credit masters and the exposure ingest. Required integer columns (`uid`, `obligor_uid`, `factor_id`, `subclass_uix`, `hedging_set_uix` and the netting-set `uid`): an integer dtype, or a float dtype whose values are integral; a fractional value or a value outside the target dtype range raises TypeError (R-9; `factor_id` int64 to int32 overflow is an intended change, F-11); a null raises ValueError. Optional integer columns (`rating_id`, `sector_id`, `country_id`, `parent_uid`): a null compiles to the sentinel of `riskquant.constants`; an integral float compiles to its integer; a fractional value or an overflow raises TypeError; an all-null `pl.Null` column and an absent column mean all missing. Required float columns: null or NaN raises ValueError; `pl.Boolean` raises TypeError. `nu`, `alpha` and `beta` of the factor universe map null to their sentinels as today's NaN does. [Proposed] Test: S3.T1, S4.T2
- C.9 Every array that an entry of C.7 places in a compiled container is C-contiguous and writeable, and has the dtype of v0.2.0. On the canonical credit fixtures the compiled arrays of `compile_obligor_master`, `compile_netting_set_master`, `compile_credit_factor_universe` and `compile_obligor_ead` are equal, element for element and dtype for dtype, to the arrays recorded at v0.2.0. The sort permutation is `np.argsort(key, kind="stable")`, or an equal Polars sort with `maintain_order=True`. [Proposed] Test: S4.T3, S5.T4
- C.10 `recipes.assemble_inputs` takes `obligor_table: pl.DataFrame` after S4 and `loadings_table: pl.DataFrame | None` after S5; `group_params` does not change. `run._obligor_frame` (S4) and `run._loadings_frame` (S5) return `pl.DataFrame`, built column by column with stated dtypes (F-23): `uid` Int64, `ead`, `pd`, `lgd`, `rho`, `loading` Float64, `group` and `factor` Utf8. `recipes._per_obligor_rho` reads an explicit `rho` from a `pl.Float64` column with nulls; a `pl.Null` column means no explicit value. The grouped assemblers read `group` as `pl.Utf8` or `pl.Null`. [Proposed] Test: S4.T4, S4.T5, C.4
- C.11 `CreditVarRequest.from_dataframe(obligor_table: pl.DataFrame, *, model, loadings_table: pl.DataFrame | None = None, group_params=None, n_scenarios=DEFAULT_N_SCENARIOS, alphas=None, seed=12345, **params)`. Any other table type raises TypeError before any read. Column admission by dtype (O-5 (b)): `obligor_uid` (both tables) is an integer dtype, or a float dtype admitted after an integral check, where a non-integral, NaN or infinite cell raises R at `("obligor_uid",)` (B-4 (ii)); `ead`, `pd`, `lgd`, `rho`, `loading` are integer or float dtypes; `group` and `factor` are `pl.Utf8`; `group` and `rho` may also be `pl.Null`, which means absent for all rows. Any other dtype, `pl.Boolean` and `pl.Decimal` included, raises TypeError that names the column. A null cell in `group` or `rho` means absent for that obligor; a null cell in a required column (`obligor_uid`, `ead`, `pd`, `lgd`, `factor`, `loading`) raises R at the field path of the row model, as `("obligor_uid",)` (B-5). The docstring names the columns the code reads; the default n_scenarios equals the request default (WS_CREDIT_VAR_API C.21). [Proposed] Test: S6.T1, S6.T2, S6.T3
- C.12 `CreditExposureRequest.from_dataframe` takes `pl.DataFrame` for each table; each field follows the C.11 rule of its type (int fields as `obligor_uid`, float fields as `ead`, str fields as `group`); a null cell in an optional field means absent. `attach_exposure_ead(obligor_table: pl.DataFrame, result, *, uid_col: str = "obligor_uid") -> pl.DataFrame` returns a new frame with an `ead` Float64 column and raises ValueError for a missing uid as today. The five frames of `credit_exposure/run.py` are built with an explicit schema (F-23). `run_credit_exposure` results equal the v0.2.0 record, without `provenance.engine_version`. [Proposed] Test: S3.T2, S3.T3, S3.T4

Utilities (S2):
- C.13 `print_df` accepts `pl.DataFrame | pl.Series`; `print_dict_of_dfs` accepts `dict[str, pl.DataFrame]` and prints a frame wider than `cols_before_transpose` as `transpose(include_header=True)`. In v1 an object that is not Polars and has `to_string` is printed through `to_string`, with no pandas import in `printer.py`; WS_PANDAS_REMOVAL removes that branch. [Proposed] Test: S2.T1, S2.T3
- C.14 `IngestBatch.to_df()` and `.df` return `pl.DataFrame` (O-2 (b)) with the columns of v0.2.0. `und_ids`, `vol_ids`, `fx_u2s_ids` and `fx_k2s_ids` are `pl.Int64` with null where the SoA holds the sentinel (today the text "NONE"); `k` and `cp_mult` are `pl.Float64` with null where today they hold NaN. `batch.py` imports no pandas. [Proposed] Test: S2.T2
- C.19 `python scripts/tools/demo_import_smoke.py [paths]` executes only the top-level import statements of each demo (AST) and exits non-zero when one fails; default paths: the 15 `scripts/solutions/credit_var/demo_e2e_*.py` (O-8). [Proposed] Test: S2.T4

Release (S6):
- C.17 The version is 0.2.1 under D106.2 (O-6 (a)): the result values (C.4), the 22 names, the golden schemas and the error types do not change; `from_dataframe` is outside the surface (D106.1). [Proposed] Test: S6.T4

## 4. Decisions

Decisions relied on: D001, D002, D006, D094, D095, D096, D097, D098, D100, D106, D107, D-C15. Each checked against `DECISIONS.md`: yes (text at 19d097a; R1 section B).

### P-CREDIT_POLARS-001: Credit edge on Polars and one admission loading sum (PROPOSED)

```text
Related: D006, D094, D095, D096 (superseded), D098, D100, D106 (pins line), D-C15
Scope: the frame type of the credit edge, the admission loading sum, the null and
dtype rules of the credit edge, and the IngestBatch debug adapter.
1. The credit ingest entries of C.7 accept polars.DataFrame only; any other type
   raises TypeError before a stage runs; the caller's frame is never changed.
   CreditVarRequest.from_dataframe and CreditExposureRequest.from_dataframe take
   polars.DataFrame tables and admit columns by dtype (C.11, C.12).
2. The per-obligor loading sum of squares that admits or rejects a book is one
   function (C.5). Its arithmetic is the Kahan compensated sum in row order that
   pandas 3.0.x used in groupby().sum(), including its reset of a non-finite
   compensation to 0.0 (pandas 2.2.3 resets on NaN only and differs only for
   overflowing squares). The engine admission check and
   the credit_var request rule 14 both call it. The builder sum that gives
   idio_weight is np.add.reduceat and does not change. The pins line of D106
   reads pandas>=2.0 (no upper bound) until WS_PANDAS_REMOVAL.
3. Null and dtype rules of the credit edge: an integer column admits an integer
   dtype or integral floats; a fractional value or an overflow raises TypeError;
   a null in a required column raises (ValueError at ingest, R at the request
   edge); a null in an optional integer column compiles to its sentinel; in a
   float column null means null or NaN; Boolean is not numeric; an all-null
   column is all missing.
4. D096 is superseded: IngestBatch.to_df and .df return polars.DataFrame (C.14).
Evidence: S1.T1 to S1.T6, S2.T2, S3.T1 to S3.T4, S4.T1 to S4.T5, S5.T1 to S5.T4,
S6.T1 to S6.T3; C.4 record in every slice.
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Scope of the market-risk edge (F-9). | CLOSED | 2026-11-05 | Owner: first (a); then, on R1 under the tripwire, descope: S7 moves to WS_PANDAS_REMOVAL. |
| O-2 | `IngestBatch.to_df` and `.df` (F-10, D096). | CLOSED | 2026-11-05 | Owner: (b). `pl.DataFrame`; D096 superseded by P-CREDIT_POLARS-001 clause 4; dtypes in C.14 (R-12). |
| O-3 | C.8 null rules. | CLOSED | 2026-11-05 | Owner: a fractional float raises TypeError; an all-null column is all missing; a null in a required column raises ValueError. R-9: overflow raises TypeError. |
| O-4 | Placement of the C.5 function. | CLOSED | 2026-11-05 | Owner: (a) `src/riskquant/credit/jtd/ingest/loading_sum.py`. |
| O-5 | `from_dataframe` cell admission. | CLOSED | 2026-11-05 | Owner: (b) admission by column dtype; B-4 (ii) and B-5 refine it (C.11). |
| O-6 | Release version. | CLOSED | 2026-11-05 | Owner: (a) 0.2.1. Owner action before S6: check riskquant-excel and riskquant-emissary for `from_dataframe` callers that pass pandas frames. |
| O-7 | `RiskFactorMatrix` and `RiskFactorLevelSnapshot` pandas views. | CLOSED | 2026-11-05 | Owner: moved to WS_PANDAS_REMOVAL with S7 (section 7). |
| O-8 | Full demo set and demo import smoke. | CLOSED | 2026-11-05 | Owner: smoke is `python scripts/tools/demo_import_smoke.py [paths]` (top-level imports only, AST); default paths are the 15 credit_var demos; a slice adds the demo files it changes. S2 creates the tool from `_gates/demo_import_smoke.py`. Full set: the 15 credit_var demos plus every demo file the slice changes. |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1 and S2 first, in parallel. S3 and S4 after S1; S3 and S4 may run in parallel. S5 after S4. S6 after S1 to S5 are VALIDATED. The owner may deliver S4 and S5 in one prompt; the review stays FULL.

Review depth: FULL for S1, S3, S4, S5, S6. QUICK for S2; the S2 review checks the C.14 dtypes.

A_INVENTORY.md (Revision 1) tags each pandas importer with a slice. Rows tagged S7 belong to WS_PANDAS_REMOVAL. `scripts/reporting/credit_risk/stress_tests/calc_deteministic_jtd.py` calls no entry of v1 (it imports `ingest_fixtures.credit.jtd.deterministic_jtd`); it is not in S5.

Every implementer prompt states "Read DECISIONS.md before you write code" and "Use ruff 0.15.21 for format and lint".

Gate block (PowerShell, repository root, `WORKFLOW.md` section 6). Each slice sets `$f` to its Files and `$t` to its test paths:

```powershell
& {
  "== ruff version";        ruff --version
  "== ruff check";          ruff check $f;          "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f; "exit=$LASTEXITCODE"
  "== mypy";                mypy $f;                "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;           "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;              "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

Demo gate (O-8): `smoke` = `python scripts/tools/demo_import_smoke.py [demo files the slice changes]` (default: the 15 credit_var demos). Before S2 is merged, S1 runs `python _gates/demo_import_smoke.py`. `full` = `Get-ChildItem <paths> | ForEach-Object { python $_.FullName; "$($_.Name) exit=$LASTEXITCODE" }` over the 15 credit_var demos and every demo file the slice changes, each with exit=0.

### S1: Frame-free loading sum and the pandas upper bound

Objective: Replace the pandas admission sum with one function that the engine and rule 14 call, prove it bit for bit, and remove the upper bound <3.1.
Contract: C.4, C.5, C.6, C.18
Depends on: none
Files:
- create: src/riskquant/credit/jtd/ingest/loading_sum.py; tests/credit/jtd/test_loading_sum.py; tests/credit/jtd/golden/loading_sum_kahan.json; tests/solutions/credit_var/test_pandas_free_results.py; tests/solutions/credit_var/golden/credit_var_results_v0_2_0.json
- modify: src/riskquant/credit/jtd/ingest/validate.py (lines 612 to 614 and the import only); src/riskquant/solutions/credit_var/schemas.py (`_check_loading_sum_sq`, its docstring, the `LOADING_SUM_SQ_TOLERANCE` comment); tests/solutions/credit_var/test_request_contract.py (`_engine_grouped_sum` only: compare with the S1 golden records, not with pandas and not with the C.5 function); pyproject.toml (`pandas>=2.0,<3.1` to `pandas>=2.0` only)
FIRST: on the v0.2.0 tag (the WS_CREDIT_VAR_API closure record names it; the snapshot does not show tags), before any src change: record the C.6 golden with pandas 3.0.x on the owner machine (report `pandas.__version__`); record the C.4 golden from Q with the numpy, scipy and numba versions. VERIFY FIRST that `_BOOK_KEYS` holds all 13 keys of F-20; add each missing key to Q in `test_pandas_free_results.py` and report it.
Prohibited: `builders/*`; every other pandas use of validate.py (S5); `pandas-stubs` and every other pyproject.toml line; DECISIONS.md; any kernel.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_tests_credit.txt ; context_riskquant_tests_solutions_credit_var.txt ; pyproject.toml
Gates: gate block; `$t = "tests/credit/jtd", "tests/solutions/credit_var"`; demo smoke. Then, in a second environment with the latest pandas 3.1.x (record `pandas.__version__`): `pytest $t -q` and `pytest -q`, raw output. STOP AND ASK for a failure outside the S1 Files under pandas 3.1.x.
Exit: every S1 test passes in both environments; full suite passes in both environments; `grep -n "groupby" src/riskquant/credit/jtd/ingest/validate.py src/riskquant/solutions/credit_var/schemas.py` returns no hit.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | 2,000 groups near 1 + 1e-9 from a fixed seed, including 200 groups given in two row orders | each final total equals the golden under `float.hex()` | a sum other than the pandas Kahan sum (F-3, F-4) |
| S1.T2 | WS_CREDIT_VAR_API S2.T6 cases through `api` and through `validate_ingest_bundle_semantics`; three obligors rejected with uids given as 30, 10, 20 | 1 + 2e-9 rejected by both; 1 + 0.5e-9 accepted by both; api loc `("loadings", j, "loading")`; the engine message lists `[10, 20, 30]` | predicate drift; uid order change (F-28) |
| S1.T3 | Q through `run_credit_var` | JSON without `generated_at` and `engine_version` byte-equal to the v0.2.0 record; a version mismatch fails with each difference named | any result change (C.4) |
| S1.T4 | AST of validate.py and schemas.py | one call each to the C.5 function; no `groupby` attribute; no other loop over `loading * loading` | a second summation (C.5) |
| S1.T5 | `tomllib` parse of pyproject.toml | the pandas requirement is `pandas>=2.0` with no upper bound; `pandas-stubs>=2.0` present | cap left behind (C.18) |
| S1.T6 | AST of `test_request_contract.py:_engine_grouped_sum` | no pandas reference; no call to the C.5 function | an oracle that depends on pandas or on the code under test |

### S2: Utilities and the demo import smoke tool

Objective: Make the printer, the config audit and the ingest batch adapter pandas-free, and add the demo import smoke tool.
Contract: C.13, C.14, C.19
Depends on: none
Files:
- create: tests/utils/test_printer_polars.py; scripts/tools/demo_import_smoke.py (from the owner gate tool `_gates/demo_import_smoke.py`, with optional path arguments)
- modify: src/riskquant/utils/printer.py; src/riskquant/config/static_config/audit.py; src/riskquant/portfolio/ingest/batch.py; scripts/demos/config/static_config/demo_full.py; scripts/demos/credit/rc/demo_spot_rc_end_to_end.py (the `.df` print at line 212 only); tests/portfolio/test_pandas_free_boundary.py (the `to_df` case only)
- delete: src/riskquant/utils/dataframes.py (no caller, F-17)
FIRST: list every test that calls `IngestBatch.to_df`, `.df`, `print_df` or `print_dict_of_dfs` with a pandas object. STOP AND ASK for a hit outside the Files list.
Prohibited: `src/riskquant/credit/**`; `src/riskquant/market_data/**`; `src/riskquant/utils/dates.py`; the snapshot print of `demo_spot_rc_end_to_end.py` (line 174, WS_PANDAS_REMOVAL); pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_utils.txt ; context_riskquant_config.txt ; context_riskquant_portfolio.txt ; context_riskquant_tests_portfolio.txt ; context_riskquant_tests_utils.txt ; _gates/demo_import_smoke.py
Gates: gate block; `$t = "tests/utils", "tests/portfolio", "tests/config"`; demo full over `scripts/demos/config/static_config/demo_full.py`, `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py` and the 15 credit_var demos; `python scripts/tools/demo_import_smoke.py`.
Exit: every S2 test passes; full suite passes; `utils/dataframes.py` is absent; `grep -n "pandas" src/riskquant/portfolio/ingest/batch.py src/riskquant/utils/printer.py src/riskquant/config/static_config/audit.py` returns docstring hits only.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | `print_df` on a `pl.DataFrame` with int, float, str and null cells, `float_format` set and None, `max_rows=2`; on a `pl.Series` | output contains each column name; no exception; the series prints as one column; `"pandas" not in sys.modules` in a child process | pandas-only formatting path |
| S2.T2 | `IngestBatch.to_df()` and `.df` on the canonical SPOT batch | `pl.DataFrame`; the v0.2.0 column names; `und_ids`, `vol_ids`, `fx_u2s_ids`, `fx_k2s_ids` are Int64 with null at the sentinel rows; `k`, `cp_mult` are Float64 with null at the sentinel rows; no pandas import | D096 adapter left in place; mixed int and text column (R-12) |
| S2.T3 | `print_dict_of_dfs` with one `pl.DataFrame` of 25 columns and one of 3 | the wide frame prints transposed with a header column; no exception | `df.T` on a Polars frame (R-14) |
| S2.T4 | `demo_import_smoke.py` on a temporary demo whose body writes a file and whose imports succeed; on a temporary demo with a top-level import of a missing module | exit 0 and no file written; non-zero exit that names the demo | tool that runs demo bodies or hides an import failure (C.19) |

### S3: Credit exposure chain

Objective: Move the exposure ingest entries and `solutions.credit_exposure` to Polars with explicit schemas and a v0.2.0 result record.
Contract: C.4, C.7, C.8, C.12
Depends on: S1
Files:
- create: tests/solutions/credit_exposure/golden/credit_exposure_results_v0_2_0.json; tests/solutions/credit_exposure/test_polars_edge_contract.py
- modify: src/riskquant/credit/exposure/add_on/ingest.py; src/riskquant/credit/exposure/saccr/ingest.py; src/riskquant/solutions/credit_exposure/{schemas,recipes,run,handoff,golden_master_credit_exposure}.py; src/riskquant_lab/fixtures/credit_exposure.py; src/riskquant/ingest_fixtures/credit/credit_exposure/minimal_spot_with_netting_sets.py; scripts/demos/credit/exposure/demo_e2e_cem_trade_to_exposure.py; scripts/demos/credit/exposure/demo_e2e_saccr_trade_to_exposure.py; tests/credit/exposure/saccr/test_saccr_trade_level_contract.py
FIRST: on the v0.2.0 tag, record `run_credit_exposure` results for the lab canonical book and for the `golden_master_credit_exposure.py:_frozen_book` book, for each method of the catalog. Then list every other test file that passes a pandas frame to an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: `credit/exposure/obligor_ead/ingest.py` (S4); every kernel and evaluator; `credit_exposure/api.py` names; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_exposure.txt ; context_riskquant_lab.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/credit/exposure", "tests/solutions/credit_exposure", "tests/solutions/credit_var/test_pandas_free_results.py"`; `python src/riskquant/solutions/credit_exposure/golden_master_credit_exposure.py`; demo full over the 2 changed demos and the 15 credit_var demos.
Exit: every S3 test passes; full suite passes; the golden master exits 0.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | `compile_add_on_factors` with a pandas frame, a `pl.LazyFrame`, a dict, None; `compile_saccr_config` with each of these in each frame argument; a float `subclass_uix` column with 1.5; an Int64 `subclass_uix` of 2**40; a null `factor` | TypeError before any read, naming `pl.DataFrame`; TypeError; TypeError; ValueError | silent acceptance (C.7, R-11); truncation or wrap (C.8) |
| S3.T2 | `CreditExposureRequest.from_dataframe` and `attach_exposure_ead` (default and explicit `uid_col`) on the lab book; a book of 120 trades whose first 100 have None in `volume` | result JSON byte-equal to the dict-built request; new frame with `ead`; input frame unchanged; missing uid raises ValueError; the 120-trade book runs | assembly drift; frame mutation; schema inference failure (B-7) |
| S3.T3 | Each method on the two FIRST books | JSON without `provenance.engine_version` byte-equal to the v0.2.0 record | exposure value change (R-15) |
| S3.T4 | `CreditExposureRequest.from_dataframe` with a Float64 `trade_uid` column 7.0; 7.9; a `pl.Boolean` `volume` column | admitted as 7; R; TypeError naming `volume` | silent `int()` truncation (R-13) |

### S4: Credit masters and the credit_var obligor table

Objective: Move the three masters, `compile_obligor_ead` and the obligor-table side of `solutions.credit_var` to Polars.
Contract: C.4, C.7, C.8, C.9, C.10
Depends on: S1
Files:
- create: tests/credit/masters/test_polars_edge_contract.py; tests/credit/masters/golden/compiled_masters_v0_2_0.npz
- modify: src/riskquant/credit/masters/obligors/ingest.py; src/riskquant/credit/masters/netting_set/ingest.py; src/riskquant/credit/masters/factor_universe/ingest.py; src/riskquant/credit/exposure/obligor_ead/ingest.py; src/riskquant/solutions/credit_var/run.py (`_obligor_frame` only); src/riskquant/solutions/credit_var/recipes.py (`assemble_inputs` `obligor_table` type, `_uids`, `_universe`, `_masters`, `_ead`, `_per_obligor_rho`, `_assemble_beta_binomial_grouped`, `_assemble_stress_regime_bernoulli_grouped`, and the `obligor_table` annotation of every assembler; the bundle frames stay pandas until S5); src/riskquant/ingest_fixtures/credit/obligors/minimal_obligors_master.py; src/riskquant/ingest_fixtures/credit/netting_set/minimal_netting_set_master.py; src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_universe_master.py; src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_data.py; the 12 JTD demos of F-16 (master and universe frames only); tests/solutions/credit_var/test_request_contract.py (`_engine_accepts`, `obligor_table` only); tests/credit/solutions/test_catalog_default_ownership.py (`_assembled_universe` only); tests/credit/solutions/test_marshall_olkin_default_ownership.py (`_effective_systemic_pd_at_construction` only)
FIRST: on the v0.2.0 tag, record the C.9 arrays from the canonical fixtures. Then list every other test file that passes a pandas frame to an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: `credit/jtd/**` (S5); `run._loadings_frame`; `recipes._mf_universe_and_bundle` and every `pd.DataFrame(frame_data)` bundle frame; `schemas.py`; every kernel; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_ingest_fixtures.txt ; context_riskquant_tests_credit.txt ; context_riskquant_tests_solutions_credit_var.txt
Gates: gate block; `$t = "tests/credit", "tests/solutions/credit_var"`; demo full over the 12 changed JTD demos and the 15 credit_var demos.
Exit: every S4 test passes; full suite passes; S1.T3 passes unchanged.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | Each of the 4 entries with a pandas frame, a `pl.LazyFrame`, a dict and None | TypeError before any column read | C.7 |
| S4.T2 | `compile_obligor_master`: `rating_id` 3.0, 3.5, 200, null, a `pl.Null` column, an absent column; `uid` 7.5, null, 2**63 in a UInt64 column; `pd` NaN, null, a `pl.Boolean` column. `compile_credit_factor_universe`: `factor_id` 2**40; `nu` null | 3; TypeError; TypeError; sentinel; sentinel for every row; sentinel for every row; TypeError; ValueError; TypeError; ValueError; ValueError; TypeError; TypeError; `NO_CHI_NU` | truncation, wrap, NaN read as data (C.8, F-11) |
| S4.T3 | The 4 entries on the canonical fixtures against the npz record; a fixture in reversed row order | arrays equal in value and dtype; each array C-contiguous and writeable; same arrays for the reversed fixture | permutation drift; read-only array (C.9, F-25) |
| S4.T4 | sf_gaussian with `rho` on 2 of 5 obligors; forced_default with no `rho` and no `group`; beta_binomial_grouped and stress_regime_bernoulli_grouped | explicit rho rows use the request value, others the fill; `pl.Null` columns accepted; results equal the C.4 record | null handling (F-12, F-13); `astype` on a Polars series (F-26) |
| S4.T5 | `run._obligor_frame` with 150 obligors whose first 120 have `rho` None and the rest a float | schema `uid` Int64, `ead`, `pd`, `lgd`, `rho` Float64, `group` Utf8 or Null; values equal the request | schema inference failure (F-23) |

### S5: JTD ingest

Objective: Move the JTD bundle, the structural and semantic validators, the shared primitives and the 9 builders to Polars, with the credit_var bundle frames.
Contract: C.4, C.7, C.9, C.10
Depends on: S4
Files:
- modify: src/riskquant/credit/jtd/ingest/bundle.py; src/riskquant/credit/jtd/ingest/validate.py; src/riskquant/credit/jtd/ingest/_shared.py; src/riskquant/credit/jtd/ingest/builders/{beta_binomial,forced_default,independent_bernoulli,marshall_olkin,mf_gaussian,mf_students_t,sf_gaussian,sf_students_t,stress_regime_bernoulli}.py (frame reads only); src/riskquant/solutions/credit_var/run.py (`_loadings_frame`); src/riskquant/solutions/credit_var/recipes.py (`assemble_inputs` `loadings_table` type, every bundle frame, `_mf_universe_and_bundle`); src/riskquant/ingest_fixtures/credit/jtd/** (8 files of A_INVENTORY.md section 1); the 12 JTD demos of F-16 (bundle frames); scripts/tests/credit/jtd/test_ingest_bundle_validation.py; tests/solutions/credit_var/test_request_contract.py (`_engine_accepts`, `loadings_table` only); tests/credit/test_dataclass_equality_contract.py; tests/credit/jtd/kernels/test_mf_factor_independence.py
FIRST: on the v0.2.0 tag, record the S5.T2 arrays and the exception type and message of each S5.T3 case. Then list every other test file that passes a pandas frame to an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: the `np.add.reduceat` lines and the CSR packing of `builders/mf_gaussian.py` and `builders/mf_students_t.py` (F-7); `loading_sum.py` and its call sites; every kernel and evaluator; `schemas.py`; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_ingest_fixtures.txt ; context_riskquant_tests_credit.txt ; context_riskquant_tests_solutions_credit_var.txt
Gates: gate block; `$t = "tests/credit", "tests/solutions/credit_var"`; `python scripts/tests/credit/jtd/test_ingest_bundle_validation.py`; demo full over the 12 changed JTD demos and the 15 credit_var demos.
Exit: every S5 test passes; full suite passes; S1.T3 passes unchanged; `grep -rln "import pandas" src/riskquant/credit src/riskquant/solutions/credit_var` returns only `solutions/credit_var/schemas.py`.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | `JtdIngestBundle.from_instrument_frame` with a pandas frame, a `pl.LazyFrame` and a dict in `frame` and in `mf_loadings_df`; each frame slot with a pandas frame | TypeError before any column read | C.7 (R-11) |
| S5.T2 | The mixed canonical fixture through `compile_jtd_portfolio`: block order, `obligor_uids`, `loading_indptr`, `loading_values`, `idio_weight` | equal in value and dtype to the v0.2.0 arrays | order or packing drift |
| S5.T3 | Each structural and semantic check of `validate_ingest_bundle` and `validate_ingest_bundle_semantics` with one fault each; a NaN in `factor_loading`; a NaN in a required float column of a per-model frame; a `pl.Boolean` `factor_loading` column; factor names as `pl.Categorical` and as `pl.Enum`; an Int64 factor-name column | the v0.2.0 exception type and uid list for each check; ValueError (NaN/inf); ValueError (null); ValueError (not numeric); accepted; accepted; ValueError | NaN missed by `is_null`; Boolean admitted as numeric (F-24, B-8) |
| S5.T4 | Every array in each compiled JTD block of the mixed fixture | C-contiguous and writeable | read-only array from `to_numpy` (F-25) |

### S6: from_dataframe on Polars, the credit_var demos and release 0.2.1

Objective: Move `CreditVarRequest.from_dataframe` to `pl.DataFrame` with dtype admission, move the 15 demos and the test callers, and set version 0.2.1.
Contract: C.4, C.11, C.17
Depends on: S1, S2, S3, S4, S5 VALIDATED; O-6 owner action (consumer check)
Files:
- modify: src/riskquant/solutions/credit_var/schemas.py (`from_dataframe`, `_dataframe_uid`, docstring; drop the pandas import); scripts/solutions/credit_var/demo_e2e_*.py (15 files); tests/solutions/credit_var/test_request_contract.py (the frames of the WS_CREDIT_VAR_API S2.T12 tests only; assertions unchanged, B-4); tests/credit/factors/test_batch_size_reproducibility.py; tests/credit/jtd/test_beta_binomial_contract.py; tests/credit/solutions/test_catalog_default_ownership.py; tests/credit/solutions/test_marshall_olkin_default_ownership.py; tests/credit/solutions/test_marshall_olkin_request_validation.py (each: `from_dataframe` frames only); tests/solutions/credit_var/test_public_surface.py (the version of S5.T3 only); pyproject.toml (`version` only)
FIRST: report the owner consumer check of O-6. List every other test file that calls `from_dataframe` with a pandas frame. STOP AND ASK for a hit outside the Files list.
Prohibited: every other `schemas.py` member; `api.py` names; result models; the golden schema files; every pyproject.toml line except `version`; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_lab.txt ; context_riskquant_tests_solutions_credit_var.txt ; context_riskquant_tests_credit.txt ; context_riskquant_scripts.txt (the 15 credit_var demos only) ; pyproject.toml
Gates: gate block; `$t = "tests/solutions/credit_var", "tests/credit"`; demo full over the 15 credit_var demos; `pip install -e .` and `python -c "import importlib.metadata as m; print(m.version('riskquant'))"`, raw output of both.
Exit: every S6 test passes; full suite passes; 15 of 15 demos exit=0; WS_CREDIT_VAR_API S5.T1 and S5.T2 pass unchanged; installed version prints 0.2.1.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S6.T1 | `from_dataframe` with a pandas frame, a `pl.LazyFrame`, a dict, in `obligor_table` and in `loadings_table` | TypeError before any read | C.11 |
| S6.T2 | `pl.DataFrame` tables for each of the 13 keys, with `group`, `rho` and `loadings_table` where the model uses them | the request equals the dict-built request field for field; `input_hash` equal | assembly drift |
| S6.T3 | Columns: `ead` `pl.Boolean`; `ead` `pl.Decimal`; `ead` `pl.Utf8` "7.9"; `obligor_uid` Float64 7.0; 7.9; NaN; a `group` column of `pl.Null`; a `rho` column of `pl.Null`; a null cell in `group`; a null cell in `rho`; a null cell in `ead`; a null cell in `obligor_uid` (Int64); a null cell in `factor` | TypeError naming `ead` (3 cases); admitted as 7; R at `("obligor_uid",)` (2 cases); accepted, no group; accepted, fill rho; that obligor has no group; that obligor uses the fill; R at `("ead",)`; R at `("obligor_uid",)`; R at `("factor",)` | bool, Decimal, str cells (WS_CREDIT_VAR_API backlog); null and pl.Null rules (B-4, B-5) |
| S6.T4 | `tomllib` parse of pyproject.toml; installed version | version 0.2.1; the pandas requirement still `pandas>=2.0`; numpy, scipy, numba, pydantic specs unchanged; metadata version 0.2.1 | wrong release version (C.17) |

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| A consumer passes a pandas frame to `from_dataframe` (riskquant-emissary, riskquant-excel, notebooks) | TypeError after 0.2.1 | The member is outside the stable surface (D106.1); O-6 owner check before S6; the release note names `pl.from_pandas` |
| Tests profiles other than tests_credit and tests_solutions_credit_var were not read | A slice finds test files outside its Files list | Every slice has a FIRST step with STOP AND ASK |
| pandas 3.1.x installs after S1 | A pandas-based path outside v1 changes behavior | S1 runs the full suite under pandas 3.1.x and stops on a failure outside its Files |
| Read-only arrays from `to_numpy` (F-25) | A Numba kernel typed for writeable arrays raises | C.9, S4.T3, S5.T4 |
| Transitional state of `recipes.py` and the 12 JTD demos between S4 and S5 | Two reviews of the same files | The owner may deliver S4 and S5 in one prompt |
| polars 2.x is current (sandbox 2.1.0); pyproject declares `polars>=1.0` | API differences between 1.x and 2.x | Each report records `polars.__version__`; a pin decision is in the backlog |

Deferred: WS_PANDAS_REMOVAL. Opens when WS_CREDIT_POLARS is CLOSED. It takes over:
- S7 of Revision 2: the market-risk edge `market_data/risk_factors/{matrix,snapshot,builder_matrix,builder_snapshot}.py` (C.15 of Revision 2, O-7, F-9), including `RiskFactorMatrix.plot` (R-6, R1 E-8); `utils/dates.py`; `ingest_fixtures/market_data/canonical_market_data_history.py`; the 7 `ingest_fixtures/portfolio/*.py` modules and the NaN-to-None change of the exhaustive dict (C.16 of Revision 2, Revision 2 F-14); `scripts/demos/e2e/{_fixtures,_spot_world,03_spot_fx_and_reporting_currency,demo_e2e_sensi_based_var}.py`; the snapshot print of `demo_spot_rc_end_to_end.py` and `scripts/demos/market_data/risk_factors/demo_snapshot.py` (`.df.T`, R1 E-7). Audit input: R1 "O-7 input" and the profile parts 1 and 2 of R1 section F.
- The test-tree sweep (B-6): every test file that imports pandas and calls no v1 entry, among them the D094 differential corpus of `tests/portfolio/ingest`. Audit input: R1 profile parts 2 and 3.
- S8 of Revision 2: no pandas import under `src/riskquant` and `src/riskquant_lab` at any depth, enumerated by walk (C.1 of Revision 2, R-4); no pandas or pandas-stubs in pyproject.toml (C.2 of Revision 2); an import-blocked process imports every module and runs the 13 keys, in an environment with matplotlib or with `riskquant_lab/plotting` excluded (C.3 of Revision 2, R-8); removal of the C.13 legacy branch of `print_df`; docstrings that name pandas; the D106 pins line without pandas; P-CREDIT_POLARS-001 clause 1 of Revision 2 ("riskquant has no pandas dependency").
- Scripts that import pandas and call no v1 entry, for example `scripts/reporting/credit_risk/stress_tests/calc_deteministic_jtd.py`.

Backlog (REFINEMENT findings and items that v1 omits):
- `from_dataframe` `group` and `factor` columns: admit `pl.Categorical` and `pl.Enum`, as C.7 does for factor names.
- `golden_master_credit_exposure.py` holds tests inside `src/`; move to `tests/`.
- `minimal_factor_data.py:_debug_print` uses `DataFrame.corr(method="spearman")`; S4 keeps the print with numpy or removes it.
- A Numba `njit` version of the C.5 function: opens when a benchmark shows the Python loop above 1 ms per 10,000 loading rows.
- A polars upper bound in pyproject.toml: opens on the first polars 2.x incompatibility.
- The C.4 golden ties to one numpy, scipy and numba triple; a version change needs a new record by owner decision.

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.

| Item | Value |
|---|---|
| Final commit | |
| Release tag | |
| Decisions locked | |
| Independent validation | |

Delivered:
-

Deferred:
-

Open work: none
