# WS_CREDIT_POLARS: Remove pandas from riskquant

Path: docs/workstreams/WS_CREDIT_POLARS.md
Class: HEAVY
Status: DRAFT
Revision: 1
Revision note: First draft from the investigator chat of 2026-11-04/05.
Date: 2026-11-05
Branch: ws/credit-polars
Snapshot: SNAPSHOT commit=19d097a dirty=false content_hash=c00f240898f32d74
Drafted by: investigator
Independent review: NONE
Audit rounds used: 0 of 2
Code merged: no
DECISIONS.md reconciled: NO
Evidence limits: No repository gate, test or benchmark ran. Code facts come from the 19d097a profiles (src, scripts, lab) rebuilt in an investigator sandbox. The tests profiles (tests_credit, tests_portfolio, tests_solutions_credit_var, tests_market_data, tests_e2e, tests_root, tests_ingest_fixtures, tests_utils, tests_config, tests_simlab, tests_sensitivities) were not attached; every statement about a test file comes from pandas_inventory.txt and WS_CREDIT_VAR_API.md, not from source. Facts marked "sandbox" ran in the sandbox (Python 3.12.3, numpy 2.5.4, polars 2.1.0, pandas 3.0.6, plus venvs with pandas 2.2.3, 3.0.2 and 3.1.1); numba and pydantic are not installed there, so no engine ran. riskquant-emissary and riskquant-excel were not read.

## 1. Goal and scope

Goal: riskquant declares no pandas dependency, no module under `src/riskquant` or `src/riskquant_lab` imports pandas, the credit edge uses Polars, and the results of `riskquant.solutions.credit_var.api` stay bitwise equal for all 13 catalog keys.

Value case:
- Benefit: pandas and pandas-stubs leave pyproject.toml, with the temporary cap pandas>=2.0,<3.1 (D106 pins line). The engine admission sum and rule 14 call one frame-free function (F-5, F-6). The credit ingest edge follows D006 and D094. `import riskquant.solutions.credit_var.api` no longer loads pandas (WS_CREDIT_VAR_API F-23).
- Cost proxies: 8 slices. 57 src files and 35 scripts files import pandas (F-1, A_INVENTORY.md). Shared interfaces that change type: 8 credit ingest entries, 2 `from_dataframe` members, `attach_exposure_ead`, `print_df` and `print_dict_of_dfs` (38 and 2 callers), `IngestBatch.to_df`, `build_risk_factor_matrix_from_df`, `build_risk_factor_level_snapshot_from_df` and the pandas view members of `RiskFactorMatrix` and `RiskFactorLevelSnapshot`. 1 new D entry (P-CREDIT_POLARS-001), which supersedes D096 and edits the D106 pins line. At most 2 audit rounds.
- Smallest alternative and why it fails: S1 alone (one frame-free loading sum). It removes the reason for the pandas cap and nothing else: 55 other modules still import pandas (F-1), so no goal item except the cap is met. "Do nothing" keeps pandas 3.1 blocked and keeps the D006 debt.
- Case against: the market-risk edge (`market_data/risk_factors`, F-9) is outside the credit intent of the workstream name. Its contracts D017, D064, D076 and the 12 remediation test files in `tests/market_data/remediation/` (inventory) make S7 the largest review risk, and the credit golden of C.4 cannot detect a market-risk defect. Mitigation: S7 has its own golden (C.15) and the owner gate O-1 can move S7 to a separate workstream.
- Recommendation at drafting: PROCEED [Proposed]

In scope:
- One frame-free per-obligor loading sum, called by the engine admission check and by rule 14 (S1).
- `riskquant.utils.printer`, `riskquant.utils.dataframes`, `riskquant.config.static_config.audit`, `IngestBatch.to_df` (S2).
- Credit exposure chain: `credit/exposure/{add_on,saccr}/ingest.py`, `solutions/credit_exposure`, `riskquant_lab/fixtures/credit_exposure.py` (S3).
- Credit masters, `credit/exposure/obligor_ead/ingest.py`, the obligor-table frames of `solutions/credit_var` (S4).
- JTD ingest: `credit/jtd/ingest/*`, the 9 builders, the bundle frames of `solutions/credit_var` (S5).
- `CreditVarRequest.from_dataframe` on `pl.DataFrame` and the 15 credit_var demos (S6).
- Market-risk edge, `utils/dates.py`, portfolio and market-data fixtures, e2e demos (S7, gate O-1).
- pyproject.toml, the enforcement tests, the release (S8).

Out of scope:
- The credit_var JSON contract: the 22 names of D106.1, the request and result golden schemas, the result values (C.4 keeps them equal).
- riskquant-emissary, riskquant-excel, the ruff upgrade.
- Any change of a kernel, of `credit/jtd/ingest/builders/*` arithmetic (F-7), or of `riskquant.jit`.
- The text of D-C15, D094, D095, D097 to D100, D106 clauses 1 to 9 and D107.
- `scripts/` files that import pandas but call no riskquant entry whose type changes (A_INVENTORY.md section 3 lists which scripts change and why).

Owner decisions in force:
- 2026-11-03 (WS_CREDIT_VAR_API, pandas cap): WS_CREDIT_POLARS replaces the pandas group sum with one shared core loading-sum function and removes the cap.
- 2026-11-04 (owner rules): run the full demo set only when a slice changes numbers or demo files; otherwise use the demo import smoke. Every implementer prompt states "Read DECISIONS.md before you write code" and "Use ruff 0.15.21 for format and lint".

Depends on: WS_CREDIT_VAR_API (CLOSED, D106, D107).
Blocks: any consumer release that needs pandas 3.1 or a pandas-free environment.

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | 57 modules under `src/` import pandas: 55 at module level; `utils/dates.py` under TYPE_CHECKING and inside two functions; `portfolio/ingest/batch.py` under TYPE_CHECKING and inside `to_df`. 24 further files in pandas_inventory.txt only name pandas in comments or docstrings. `riskquant_lab/fixtures/credit_var.py` names a variable `pd`; it does not import pandas. 35 files under `scripts/` import pandas. | AST walk over the rebuilt snapshot; A_INVENTORY.md | Observed |
| F-2 | `credit/jtd/ingest/validate.py:_semantic_mf_gaussian` decides sum(loading^2) <= 1 + 1e-9 with `pd.Series(sq).groupby(uid).sum()`. | `validate.py` line 613 | Observed |
| F-3 | pandas 2.2.3 and 3.0.2 evaluate that group sum as a Kahan compensated sum in row order from (0.0, 0.0) with compensation `(t - total) - y`: 0 bitwise mismatches against a Python loop over 14,467 groups near the bound. pandas 3.1.1 gives 25 mismatches. | sandbox probe, numpy 2.5.4 | Observed |
| F-4 | Polars `group_by().agg(pl.col().sum())` is not that sum: it equals the Kahan loop in 216 of 300 single-group cases and the naive loop in 181 of 300. | sandbox probe, polars 2.1.0 | Observed |
| F-5 | `solutions/credit_var/schemas.py:CreditVarRequest._check_loading_sum_sq` (rule 14) reimplements the Kahan state per uid in request order and uses the first row at which the running sum exceeds the bound for the loc. | `schemas.py` lines 516 to 571 | Observed |
| F-6 | `run._loadings_frame` keeps request order and drops only the rows of force_default obligors; `recipes._mf_universe_and_bundle` keeps the order. | `run.py:_loadings_frame`; `recipes.py:_mf_universe_and_bundle` | Observed |
| F-7 | The engine number `idio_weight = sqrt(max(0, 1 - sum_sq))` uses `np.add.reduceat(loading_values * loading_values, loading_indptr[:-1])` over band-sorted rows, a naive sequential sum. It differs bitwise from the Kahan admission sum in 4,924 of about 14,000 groups; the admission decision at 1 + 1e-9 did not differ in that sample. | `builders/mf_gaussian.py` line 64; `builders/mf_students_t.py` line 68; sandbox probe | Observed |
| F-8 | All other pandas uses on the credit_var path are frame assembly or typed extraction: `pd.DataFrame(...)` from lists and arrays, `to_numpy(dtype=...)`, `astype(str)`, `sort_values(key, kind="stable").reset_index(drop=True)` on unique keys, `astype("Int8"/"Int32"/"Int64").fillna(sentinel)`, `notna`, `pd.to_numeric(errors="coerce")` on a validated float-or-None column, boolean filter and `drop(columns)` in `bundle.from_instrument_frame`. None sums, merges or deduplicates. | `credit/masters/*/ingest.py`; `credit/exposure/obligor_ead/ingest.py`; `credit/jtd/ingest/{bundle,_shared}.py`; `solutions/credit_var/{run,recipes}.py` | Observed |
| F-9 | `market_data/risk_factors/{matrix,snapshot,builder_matrix,builder_snapshot}.py` import pandas at module level. `RiskFactorMatrix` is imported by `simlab/scenarios/{historical,historical_fhs,bootstrap_fhs,bootstrap_sbb}.py`, `solutions/risk_run/api.py`, `solutions/run_identity/api.py`, `solutions/scenario_models/{api,world_columns}.py`, `market_data/assemble.py`. The pandas members are the views `to_df`, `head`, `tail`, `df`, `index_dates`, `index_factors`, `to_series`, `sr`, the entry `build_risk_factor_matrix_from_df(df: pd.DataFrame)` with `pd.to_datetime(df.index)`, `build_risk_factor_level_snapshot_from_df`, and two `pd.read_excel` entries. The only src caller of the builders is `ingest_fixtures/market_data/canonical_market_data_history.py`; `scripts/demos/e2e/{_fixtures,_spot_world}.py` also call them. | `market_data/risk_factors/*.py`; grep over src and scripts | Observed |
| F-10 | D096 (LOCKED) keeps `IngestBatch.to_df` and `.df` as a pandas output adapter with one caller, `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py`. | DECISIONS.md D096; `portfolio/ingest/batch.py` | Observed |
| F-11 | pandas `Series.astype("Int8")` raises TypeError on 3.5 and admits 3.0. Polars `Series.cast(pl.Int8)` turns 3.5 into 3 without an error. | sandbox probe | Observed |
| F-12 | `pl.DataFrame({"rho": [None, None]})` has dtype `pl.Null`; with one float the dtype is `pl.Float64`. `pl.Series.to_numpy()` of an Int64 or Float64 column with nulls returns float64 with NaN, and the array is read-only. `pl.from_dict` refuses a list that mixes str and float NaN. | sandbox probe | Observed |
| F-13 | `run.py:_obligor_frame` builds `group` and `rho` columns from `None` values; `recipes.py:_per_obligor_rho` reads `col.notna()` and `pd.to_numeric(col, errors="coerce")`. After request validation `rho` is a float or None and `group` is a str or None. | `run.py`; `recipes.py`; `schemas.py:ObligorRow` | Observed |
| F-14 | `ingest_fixtures/portfolio/exhaustive.py:_get_data` builds the DATA dict with `pd.concat` after `dropna(axis=1, how="all")`; missing cells become float NaN, also in str columns. `scripts/demos/credit/rc/demo_spot_rc_end_to_end.py` wraps that dict in `pd.DataFrame` and converts with `pl.from_pandas`. Polars `pl.concat(how="diagonal")` gives `None` for the same cells. | `exhaustive.py` lines 64 to 90; demo line 79; sandbox probe | Observed |
| F-15 | `solutions/credit_exposure` uses pandas for frame assembly only: `pd.DataFrame([row.model_dump() ...])` in `run.py`, `to_numpy(dtype=...)` and `np.argsort(kind="stable")` in `recipes.py`, `from_dataframe` and `pd.isna` in `schemas.py`, `obligor_table.copy()` in `handoff.py`. `golden_master_credit_exposure.py` holds `test_*` functions and a `main()`; its `groupby().sum()` is a test oracle. `CreditExposureResult.obligor_ead_df` already returns `pl.DataFrame`. | `solutions/credit_exposure/*.py` | Observed |
| F-16 | `credit_exposure.recipes` calls `compile_add_on_factors` and `compile_saccr_config` and builds `CompiledNettingSetMaster` with numpy. `credit_var.recipes` calls `compile_obligor_master`, `compile_credit_factor_universe`, `compile_obligor_ead` and `JtdIngestBundle.from_instrument_frame`. No module in `src/` other than `netting_set/ingest.py` calls `compile_netting_set_master`. | grep over src | Observed |
| F-17 | `utils/printer.py:print_df` accepts `pd.Series | pd.DataFrame` and prints with `DataFrame.to_string`; 38 files in src and scripts call it. `print_dict_of_dfs` has 2 callers. `utils/dataframes.py:validate_string_cols` has no caller (D097 noted this). `config/static_config/audit.py` builds pandas frames only to print. `utils/dates.py` imports pandas inside `coerce_as_of_date_to_pd_timestamp` (caller: `builder_matrix.py`) and `get_previous_business_date_as_iso_str` (no caller in src or scripts). `riskquant_lab/polars_display.py:configure_polars_display` exists. | `utils/*.py`; grep over src and scripts | Observed |
| F-18 | `scripts/solutions/credit_var/demo_e2e_*.py` (15 files) import pandas at line 3 or 4 and pass `pd.DataFrame` tables to `CreditVarRequest.from_dataframe`; `demo_e2e_mf_gaussian.py` and `demo_e2e_mf_students_t.py` also use `groupby().agg` for a printed summary. `riskquant_lab/fixtures/credit_var.py:generate_credit_portfolio_data` returns `dict[str, np.ndarray]`. | scripts profile; lab profile | Observed |
| F-19 | pyproject.toml declares `pandas>=2.0,<3.1` and `pandas-stubs>=2.0` in `[project].dependencies`. The optional groups `scripts` and `notebooks` declare seaborn, which requires pandas; no riskquant module imports seaborn. `observations/preprocess.py` imports `sklearn.impute` inside a function; scikit-learn does not require pandas. | pyproject.toml; grep over src | Observed |
| F-20 | The 13 catalog keys are forced_default, independent_bernoulli, stress_regime_bernoulli, stress_regime_bernoulli_grouped, beta_binomial, beta_binomial_grouped, sf_gaussian, sf_gaussian_basel, sf_students_t, sf_students_t_basel, mf_gaussian, marshall_olkin, mf_students_t. | `solutions/credit_var/catalog.py:MODEL_CATALOG` | Observed |
| F-21 | WS_CREDIT_VAR_API S2.T12 checks the `from_dataframe` docstring columns and the default n_scenarios against the code; whether it builds a pandas frame is not visible without the tests profile. | WS_CREDIT_VAR_API.md S2 table | Unresolved |

## 3. Contract

Package (S8):
- C.1 No `.py` file under `src/riskquant` or `src/riskquant_lab` contains an `Import` or `ImportFrom` node of `pandas` or `pandas.*` at any level, including inside functions and under `TYPE_CHECKING`. The set is every `.py` file under the two trees (660 at the snapshot). [Proposed] Test: S8.T1
- C.2 pyproject.toml declares no requirement whose distribution name is `pandas` or `pandas-stubs` in `[project].dependencies` or in the optional groups dev, scripts and notebooks. A transitive pandas install through seaborn in the scripts and notebooks groups is allowed and is not a riskquant dependency (F-19). [Proposed] Test: S8.T2
- C.3 In a fresh process where `import pandas` raises ImportError, every module under the two trees imports, and `run_credit_var` returns a result for each of the 13 keys of F-20. [Proposed] Test: S8.T3
- C.4 For the request set R, the JSON of `run_credit_var` without `generated_at` is byte-equal to the record made at tag v0.2.0. R is the WS_CREDIT_VAR_API S1.T12 request builder of `tests/solutions/credit_var/` (13 keys, N <= 5, S = 2_000; F-21 applies to its name) plus: sf_gaussian with a `rho` column of 2 floats and 3 nulls; beta_binomial_grouped with 2 groups; mf_gaussian with one obligor at sum(loading^2) = 1 + 0.5e-9 (the WS_CREDIT_VAR_API S2.T6 accepted case); forced_default with one `in_stress_test` obligor. [Proposed] Test: S1.T3; re-run in S3 to S8

Loading sum (S1):
- C.5 One function computes the per-obligor loading sum of squares for admission. Input: `obligor_uid` int64 (n,), `loading` float64 (n,), in row order. Per uid the state (total, compensation) starts at (0.0, 0.0); for each row y = x * x - compensation, t = total + y, compensation = (t - total) - y, total = t. Output: the running total after each row (n,), the uids in first-seen order and their final totals. The bound comparison stays in the callers. `validate.py:_semantic_mf_gaussian` and `schemas.py:_check_loading_sum_sq` call this function and hold no other summation of loading squares. `builders/mf_gaussian.py` and `builders/mf_students_t.py` keep `np.add.reduceat` (F-7). [Proposed] Test: S1.T1, S1.T2, S1.T4
- C.6 The function output equals, under `float.hex()`, the pandas `groupby().sum()` record made with pandas 3.0.x on the owner machine for 2,000 groups near 1 + 1e-9, including groups whose rows in two orders give two different totals. [Proposed] Test: S1.T1

Credit edge (S3, S4, S5):
- C.7 Each of `compile_obligor_master`, `compile_netting_set_master`, `compile_credit_factor_universe`, `compile_obligor_ead`, `compile_add_on_factors`, `compile_saccr_config`, `JtdIngestBundle.from_instrument_frame` and the frame slots of `JtdIngestBundle` accepts `pl.DataFrame` only. A `pandas.DataFrame`, `pl.LazyFrame`, dict or None raises TypeError before any column is read (D094 form). The caller's frame is never changed (D095 form). [Proposed] Test: S3.T1, S4.T1, S5.T1
- C.8 Null rules of the credit masters: a null in a required column raises ValueError; in `rating_id`, `sector_id`, `country_id` and `parent_uid` a null compiles to the sentinel of `riskquant.constants`, an integral float compiles to its integer, a float with a fractional part raises TypeError (F-11, D098 form), an all-null `pl.Null` column counts as all missing (D100 form); `nu`, `alpha` and `beta` of the factor universe map null to their sentinels as today's NaN does. [Proposed, O-3] Test: S4.T2
- C.9 The compiled arrays of `compile_obligor_master`, `compile_netting_set_master`, `compile_credit_factor_universe` and `compile_obligor_ead` on the canonical credit fixtures are equal, element for element and dtype for dtype, to the arrays recorded at v0.2.0. The sort permutation is `np.argsort(key, kind="stable")` on the key column, or an equal Polars sort with `maintain_order=True`. [Proposed] Test: S4.T3
- C.10 `run._obligor_frame` and `run._loadings_frame` return `pl.DataFrame`; `recipes._per_obligor_rho` reads an explicit `rho` from a `pl.Float64` column with nulls and treats a `pl.Null` column as no explicit value; `group` is `pl.Utf8` or `pl.Null`. [Proposed] Test: S4.T4, C.4
- C.11 `CreditVarRequest.from_dataframe(obligor_table: pl.DataFrame, *, model, loadings_table: pl.DataFrame | None = None, group_params=None, n_scenarios=DEFAULT_N_SCENARIOS, alphas=None, seed=12345, **params)`. Any other table type raises TypeError before any read. Column admission follows O-5. The docstring names the columns the code reads and the default n_scenarios equals the request default (WS_CREDIT_VAR_API C.21). [Proposed, O-5] Test: S6.T1, S6.T2, S6.T3
- C.12 `CreditExposureRequest.from_dataframe` takes `pl.DataFrame` for each table; `attach_exposure_ead(obligor_table: pl.DataFrame, result) -> pl.DataFrame` returns a new frame with an `ead` column and raises ValueError for a missing uid as today. [Proposed] Test: S3.T2
- C.13 `print_df` accepts `pl.DataFrame | pl.Series`; `print_dict_of_dfs` accepts `dict[str, pl.DataFrame]`. Until S8 a non-Polars object is printed through its own `to_string` without a pandas import; S8 removes that branch. [Proposed] Test: S2.T1, S8.T4
- C.14 `IngestBatch.to_df` and `.df` follow O-2. [Proposed, O-2] Test: S2.T2

Market-risk edge (S7, O-1):
- C.15 `build_risk_factor_matrix_from_df(df: pl.DataFrame, ...)` reads a `date` column (`pl.Date`, or tz-naive `pl.Datetime`) and factor columns; values are float64 (D017); D076 normalisation is unchanged. `build_risk_factor_level_snapshot_from_df` likewise. The `..._from_xlsx` entries read with `pl.read_excel`. The pandas view members of F-9 become Polars members with the same names or are removed (O-7). The `RiskFactorMatrix` built from `CANONICAL_HISTORICAL_MARKET_DATA` has values and date ordinals bitwise equal to the record at v0.2.0. [Proposed, O-1, O-7] Test: S7.T1, S7.T2, S7.T3
- C.16 The fixture dict `CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA` carries `None` where it carries float NaN today (F-14); `compile_raw_df_into_ingest_batch(pl.from_dict(...))` gives an `IngestBatch` equal to the one from the v0.2.0 dict through `pl.from_pandas`. [Proposed] Test: S7.T4

Release (S8):
- C.17 The version is 0.2.1 under D106.2 (O-6): the result values (C.4), the 22 names, the schemas and the error types do not change; `from_dataframe` is outside the surface (D106.1). The pandas cap leaves pyproject.toml with pandas. [Proposed, O-6] Test: S8.T2, S8.T5

## 4. Decisions

Decisions relied on: D001, D002, D006, D094, D095, D096, D097, D098, D099, D100, D106, D107, D-C15, D017, D064, D076. Each checked against `DECISIONS.md`: yes (text at 19d097a).

### P-CREDIT_POLARS-001: riskquant has no pandas dependency (PROPOSED)

```text
Related: D006, D094, D095, D096 (superseded), D098, D100, D106 (pins line), D-C15
Scope: the pandas dependency of riskquant and the credit edge frame type.
1. No module under src/riskquant or src/riskquant_lab imports pandas, at module level, inside a function or under TYPE_CHECKING. pyproject.toml declares no pandas and no pandas-stubs requirement. A test enforces both by AST and by an import-blocked process.
2. The credit ingest entries of C.7 accept polars.DataFrame only; any other type raises TypeError before a stage runs; the caller's frame is never changed. CreditVarRequest.from_dataframe and CreditExposureRequest.from_dataframe take polars.DataFrame tables.
3. The per-obligor loading sum of squares that admits or rejects a book is one function of C.5. Its arithmetic is the Kahan compensated sum in row order that pandas 2.2.3 to 3.0.x used in groupby().sum(). The engine admission check and the credit_var request rule 14 both call it. The builder sum that gives idio_weight is np.add.reduceat and does not change.
4. Nullable integer columns of the credit masters: null compiles to the sentinel; a float with a fractional part raises TypeError; an all-null column is all missing.
5. D096 is superseded: IngestBatch.to_df and .df follow O-2.
6. The pins line of D106 drops pandas. The result values of credit_var.api at a fixed request are unchanged; the release is 0.2.1 (O-6).
Evidence: S1.T1 to S1.T4, S4.T2, S4.T3, S8.T1 to S8.T5; C.4 golden in every slice from S3.
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Scope of the market-risk edge (F-9): (a) keep S7 in this workstream; (b) move S7 to a separate workstream and keep pyproject.toml with pandas until it closes. Investigator view: (a), because the goal items on pyproject.toml and C.1 need it. | OPEN | | |
| O-2 | `IngestBatch.to_df` and `.df` (F-10, D096): (a) delete both and the demo print; (b) `to_df() -> pl.DataFrame`, `.df` the same. Investigator view: (b). | OPEN | | |
| O-3 | C.8 null rules: confirm the TypeError on a fractional float (F-11) and the all-null column rule. | OPEN | | |
| O-4 | Placement of the C.5 function: (a) `src/riskquant/credit/jtd/ingest/loading_sum.py`, imported by `validate.py` and `schemas.py`; (b) `src/riskquant/_arrays.py`. Investigator view: (a); the function carries credit semantics and `_arrays.py` holds domain-free joins. | OPEN | | |
| O-5 | `from_dataframe` cell admission: (a) keep the per-cell `int`, `float`, `str` casts; a bool cell becomes 1, a Decimal cell is truncated, a str cell "7.9" raises ValueError (WS_CREDIT_VAR_API backlog); (b) admit by column dtype: `obligor_uid` integer dtypes; `ead`, `pd`, `lgd`, `rho`, `loading` integer or float dtypes; `group`, `factor` `pl.Utf8`; any other dtype raises TypeError that names the column. Investigator view: (b); one check per column closes the backlog item. | OPEN | | |
| O-6 | Release version: (a) 0.2.1, no D106.3 item is met; (b) 0.3.0 because the installed environment loses pandas. Investigator view: (a). | OPEN | | |
| O-7 | `RiskFactorMatrix` and `RiskFactorLevelSnapshot` pandas views (F-9): (a) Polars members with the same names; (b) remove the views and keep the arrays. | OPEN | | |
| O-8 | "Full demo set" for the owner rule: the 15 `scripts/solutions/credit_var/demo_e2e_*.py` plus every demo file the slice changes. Demo import smoke: `runpy.run_path(path, run_name="demo_import_smoke")` for each of the 15, which runs module-level code and not `main()`. Confirm or name the repository command. | OPEN | | |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1, S2 (parallel with S1), S3, S4, S5, S6, S7, S8. S3 may run in parallel with S4. S5 depends on S4. S6 depends on S5. S7 depends on S2 and on O-1 (a). S8 depends on every other slice VALIDATED. S4 and S5 may be delivered in one prompt when the owner prefers; the review depth stays FULL.

Review depth: FULL for S1, S3, S4, S5, S7, S8; QUICK for S2 and S6.

Gate block (PowerShell, repository root, `WORKFLOW.md` section 6, ruff 0.15.21). Each slice sets `$f` to its Files and `$t` to its test paths:

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

Demo gate (O-8): `smoke` = `Get-ChildItem scripts/solutions/credit_var/demo_e2e_*.py | ForEach-Object { python -c "import runpy; runpy.run_path(r'$($_.FullName)', run_name='demo_import_smoke')"; "$($_.Name) exit=$LASTEXITCODE" }`. `full` = `Get-ChildItem <paths> | ForEach-Object { python $_.FullName; "$($_.Name) exit=$LASTEXITCODE" }` over the 15 credit_var demos and every demo file the slice changes, each with exit=0.

### S1: Frame-free loading sum

Objective: Replace the pandas admission sum with one function that the engine and rule 14 call, with bitwise proof.
Contract: C.4, C.5, C.6
Depends on: O-4
Files:
- create: src/riskquant/credit/jtd/ingest/loading_sum.py (O-4); tests/credit/jtd/test_loading_sum.py; tests/credit/jtd/golden/loading_sum_kahan.json; tests/solutions/credit_var/test_pandas_free_results.py; tests/solutions/credit_var/golden/credit_var_results_v0_2_0.json
- modify: src/riskquant/credit/jtd/ingest/validate.py (line 613 only, plus the import); src/riskquant/solutions/credit_var/schemas.py (`_check_loading_sum_sq` and its docstring)
FIRST: on the v0.2.0 checkout, before any src change, record the two golden files: the C.6 record with pandas on the owner machine (report `pandas.__version__`), and the C.4 record from R.
Prohibited: `builders/*`; every other pandas use of validate.py (S5); pyproject.toml; DECISIONS.md; any kernel.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_tests_credit.txt ; context_riskquant_tests_solutions_credit_var.txt
Gates: gate block; `$t = "tests/credit/jtd", "tests/solutions/credit_var"`; demo smoke.
Exit: every S1 test passes; full suite passes; `grep -n "groupby" src/riskquant/credit/jtd/ingest/validate.py src/riskquant/solutions/credit_var/schemas.py` returns no hit.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | 2,000 groups near 1 + 1e-9 from a fixed seed, including 200 groups given in two row orders | each final total equals the golden under `float.hex()` | a sum other than the pandas Kahan sum (F-3, F-4) |
| S1.T2 | S2.T6 cases of WS_CREDIT_VAR_API through `api` and through `validate_ingest_bundle_semantics` | 1 + 2e-9 rejected by both; 1 + 0.5e-9 accepted by both; the api loc is `("loadings", j, "loading")` | predicate drift between rule 14 and the engine |
| S1.T3 | R through `run_credit_var` | JSON without `generated_at` byte-equal to the v0.2.0 record | any result change (C.4) |
| S1.T4 | AST of validate.py and schemas.py | one call each to the C.5 function; no `groupby` attribute; no other loop over `loading * loading` | a second summation (C.5) |

### S2: Utilities

Objective: Make the printer, the config audit and the ingest batch adapter pandas-free.
Contract: C.13, C.14
Depends on: O-2
Files:
- create: tests/utils/test_printer_polars.py
- modify: src/riskquant/utils/printer.py; src/riskquant/config/static_config/audit.py; src/riskquant/portfolio/ingest/batch.py; scripts/demos/config/static_config/demo_full.py; scripts/demos/credit/rc/demo_spot_rc_end_to_end.py (the `.df` print only); tests/portfolio/test_pandas_free_boundary.py (the `to_df` case)
- delete: src/riskquant/utils/dataframes.py (no caller, F-17)
Prohibited: `src/riskquant/credit/**`; `src/riskquant/market_data/**`; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_utils.txt ; context_riskquant_config.txt ; context_riskquant_portfolio.txt ; context_riskquant_tests_portfolio.txt ; context_riskquant_tests_utils.txt
Gates: gate block; `$t = "tests/utils", "tests/portfolio", "tests/config"`; demo smoke; `python scripts/demos/config/static_config/demo_full.py`.
Exit: every S2 test passes; full suite passes; `utils/dataframes.py` is absent.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | `print_df` on a `pl.DataFrame` with int, float, str and null cells, `float_format` set and None, `max_rows=2` | output contains each column name; no exception; a `pl.Series` prints as one column | pandas-only formatting path |
| S2.T2 | `IngestBatch.to_df()` and `.df` on the canonical SPOT batch | per O-2: AttributeError (a) or `pl.DataFrame` with the SoA columns (b); `import pandas` not in `sys.modules` after the call | D096 adapter left in place |

### S3: Credit exposure chain

Objective: Move the exposure ingest entries and `solutions.credit_exposure` to Polars.
Contract: C.4, C.7, C.12
Depends on: S1
Files:
- modify: src/riskquant/credit/exposure/add_on/ingest.py; src/riskquant/credit/exposure/saccr/ingest.py; src/riskquant/solutions/credit_exposure/{schemas,recipes,run,handoff,golden_master_credit_exposure}.py; src/riskquant_lab/fixtures/credit_exposure.py; src/riskquant/ingest_fixtures/credit/credit_exposure/minimal_spot_with_netting_sets.py; scripts/demos/credit/exposure/demo_e2e_cem_trade_to_exposure.py; scripts/demos/credit/exposure/demo_e2e_saccr_trade_to_exposure.py; tests/credit/exposure/** and tests of `solutions/credit_exposure` that build pandas frames (FIRST step lists them)
FIRST: list every test file under tests/ that imports pandas and calls an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: `credit/exposure/obligor_ead/ingest.py` (S4); every kernel and evaluator; `credit_exposure/api.py` names; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_exposure.txt ; context_riskquant_lab.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/credit/exposure", "tests/solutions/credit_var/test_pandas_free_results.py"` plus the credit_exposure test paths the FIRST step finds; demo full over the 2 changed demos and the 15 credit_var demos.
Exit: every S3 test passes; full suite passes; `credit_exposure` golden master `main()` exits 0.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | `compile_add_on_factors` and `compile_saccr_config` with a pandas frame, a `pl.LazyFrame`, a dict and None | TypeError before any column read; the message names `pl.DataFrame` | silent acceptance of another type (C.7) |
| S3.T2 | `CreditExposureRequest.from_dataframe` and `attach_exposure_ead` on the lab fixture book, each method of the catalog | result JSON byte-equal to the same request built from dicts; `attach_exposure_ead` returns a new frame with `ead`, input frame unchanged, missing uid raises ValueError | assembly drift; frame mutation (D095 form) |

### S4: Credit masters and the credit_var obligor table

Objective: Move the three masters, `compile_obligor_ead` and the obligor-table frames of `solutions.credit_var` to Polars.
Contract: C.4, C.7, C.8, C.9, C.10
Depends on: S1, O-3
Files:
- modify: src/riskquant/credit/masters/obligors/ingest.py; src/riskquant/credit/masters/netting_set/ingest.py; src/riskquant/credit/masters/factor_universe/ingest.py; src/riskquant/credit/exposure/obligor_ead/ingest.py; src/riskquant/solutions/credit_var/run.py (`_obligor_frame` only); src/riskquant/solutions/credit_var/recipes.py (`_uids`, `_universe`, `_masters`, `_ead`, `_per_obligor_rho`; the bundle frames stay pandas until S5); src/riskquant/ingest_fixtures/credit/obligors/minimal_obligors_master.py; src/riskquant/ingest_fixtures/credit/netting_set/minimal_netting_set_master.py; src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_universe_master.py; src/riskquant/ingest_fixtures/credit/credit_factors/minimal_factor_data.py; tests/credit/masters/** and tests/credit/exposure/obligor_ead tests that build pandas frames (FIRST step)
- create: tests/credit/masters/test_polars_edge_contract.py; tests/credit/masters/golden/compiled_masters_v0_2_0.npz
FIRST: on the v0.2.0 checkout, record the C.9 arrays from the canonical fixtures. Then list every test file under tests/ that passes a pandas frame to an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: `credit/jtd/**` (S5); `run._loadings_frame`; `recipes._mf_universe_and_bundle` and the `pd.DataFrame(frame_data)` calls; `schemas.py`; every kernel; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_ingest_fixtures.txt ; context_riskquant_tests_credit.txt ; context_riskquant_tests_solutions_credit_var.txt
Gates: gate block; `$t = "tests/credit", "tests/solutions/credit_var"`; demo smoke.
Exit: every S4 test passes; full suite passes; S1.T3 passes unchanged.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | Each of the 4 entries with a pandas frame, a `pl.LazyFrame`, a dict and None | TypeError before any column read | C.7 |
| S4.T2 | `compile_obligor_master` with `rating_id` 3.0, 3.5, null, an all-null `pl.Null` column, an absent column; `nu` null in the factor universe | 3; TypeError; sentinel; sentinel for every row; sentinel for every row; `NO_CHI_NU` | silent truncation (F-11); null read as data |
| S4.T3 | The 4 entries on the canonical fixtures against the npz record; a fixture in reversed row order | arrays equal in value and dtype; same arrays | permutation drift (C.9) |
| S4.T4 | sf_gaussian request with `rho` on 2 of 5 obligors; forced_default with no `rho` and no `group`; beta_binomial_grouped | explicit rho rows use the request value, others the fill; `pl.Null` columns accepted; results in C.4 record | F-12, F-13 null handling |

### S5: JTD ingest

Objective: Move the JTD bundle, the structural and semantic validators, the shared primitives and the 9 builders to Polars, with the credit_var bundle frames.
Contract: C.4, C.7
Depends on: S4
Files:
- modify: src/riskquant/credit/jtd/ingest/bundle.py; src/riskquant/credit/jtd/ingest/validate.py; src/riskquant/credit/jtd/ingest/_shared.py; src/riskquant/credit/jtd/ingest/builders/{beta_binomial,forced_default,independent_bernoulli,marshall_olkin,mf_gaussian,mf_students_t,sf_gaussian,sf_students_t,stress_regime_bernoulli}.py (frame reads only); src/riskquant/solutions/credit_var/run.py (`_loadings_frame`); src/riskquant/solutions/credit_var/recipes.py (bundle frames, `_mf_universe_and_bundle`); src/riskquant/ingest_fixtures/credit/jtd/** (8 files, A_INVENTORY.md); scripts/demos/credit/jtd/** (13 files); scripts/tests/credit/jtd/test_ingest_bundle_validation.py; scripts/reporting/credit_risk/stress_tests/calc_deteministic_jtd.py; tests/credit/jtd/** that build pandas frames (FIRST step)
FIRST: list every test file under tests/ that passes a pandas frame to an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: the `np.add.reduceat` lines and the CSR packing of `builders/mf_gaussian.py` and `builders/mf_students_t.py` (F-7); the C.5 function and its call sites; every kernel and evaluator; `schemas.py`; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_ingest_fixtures.txt ; context_riskquant_tests_credit.txt ; context_riskquant_tests_solutions_credit_var.txt
Gates: gate block; `$t = "tests/credit", "tests/solutions/credit_var"`; demo full over the 13 changed demos and the 15 credit_var demos.
Exit: every S5 test passes; full suite passes; S1.T3 passes unchanged; `grep -rn "import pandas" src/riskquant/credit src/riskquant/solutions/credit_var/run.py src/riskquant/solutions/credit_var/recipes.py` returns no hit.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | `JtdIngestBundle.from_instrument_frame` and each frame slot with a pandas frame, a `pl.LazyFrame`, a dict | TypeError before any column read | C.7 |
| S5.T2 | The mixed canonical fixture through `compile_jtd_portfolio`: block order, `obligor_uids`, `loading_indptr`, `loading_values`, `idio_weight` | equal in value and dtype to the v0.2.0 arrays (recorded in the FIRST step) | order or packing drift |
| S5.T3 | Every structural and semantic ValueError of `validate_ingest_bundle` and `validate_ingest_bundle_semantics` on a Polars fixture with one fault each | the same exception type and the same uid list in the message as at v0.2.0 | silent repair in a validator |

### S6: from_dataframe on Polars and the credit_var demos

Objective: Move `CreditVarRequest.from_dataframe` to `pl.DataFrame` and the 15 demos with it.
Contract: C.11
Depends on: S5, O-5
Files:
- modify: src/riskquant/solutions/credit_var/schemas.py (`from_dataframe`, `_dataframe_uid` per O-5, docstring); scripts/solutions/credit_var/demo_e2e_*.py (15 files); tests/solutions/credit_var/test_request_contract.py (WS_CREDIT_VAR_API S2.T12 and every case that builds a pandas frame, F-21)
Prohibited: every other `schemas.py` member; `api.py` names; result models; the golden schema files; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_lab.txt ; context_riskquant_tests_solutions_credit_var.txt ; context_riskquant_scripts.txt (the 15 demos only)
Gates: gate block; `$t = "tests/solutions/credit_var"`; demo full over the 15 credit_var demos; S5.T1 of WS_CREDIT_VAR_API (`sorted(api.__all__)`) and S5.T2 (golden schemas) pass unchanged.
Exit: every S6 test passes; full suite passes; 15 of 15 demos exit=0.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S6.T1 | `from_dataframe` with a pandas frame, a `pl.LazyFrame`, a dict | TypeError before any read | C.11 |
| S6.T2 | `pl.DataFrame` tables for each of the 13 keys, with `group`, `rho` and `loadings_table` where the model uses them | the request equals the dict-built request field for field; `input_hash` equal | assembly drift |
| S6.T3 | Per O-5: a `pl.Boolean` ead column; a `pl.Decimal` uid column; a `pl.Utf8` ead column "7.9"; a `pl.Float64` uid column with 7.0 and 7.9 | (b): TypeError naming the column for the first three; 7.0 admitted as 7, 7.9 raises R. (a): the v0.2.0 behaviour, recorded case by case | WS_CREDIT_VAR_API backlog item on bool, Decimal and str cells |

### S7: Market-risk edge, dates and fixtures

Objective: Move the risk-factor builders and views, `utils/dates.py`, and the portfolio and market-data fixtures to Polars.
Contract: C.15, C.16
Depends on: S2, O-1 (a), O-7
Files:
- modify: src/riskquant/market_data/risk_factors/{matrix,snapshot,builder_matrix,builder_snapshot}.py; src/riskquant/utils/dates.py (`coerce_as_of_date_to_pd_timestamp` becomes a `datetime.date` or `np.datetime64[D]` coercion; `get_previous_business_date_*` uses `riskquant.utils.business_days` or is deleted after a caller search); src/riskquant/ingest_fixtures/market_data/canonical_market_data_history.py; src/riskquant/ingest_fixtures/portfolio/{exhaustive,spot,vanilla_irs,vanilla_forward_node,vanilla_forward_curve,vanilla_european_option_node,vanilla_european_option_curve}.py; scripts/demos/e2e/{_fixtures,_spot_world,03_spot_fx_and_reporting_currency,demo_e2e_sensi_based_var}.py; scripts/demos/credit/rc/demo_spot_rc_end_to_end.py; tests/market_data/**, tests/portfolio/**, tests/e2e/**, tests/simlab/**, tests/sensitivities/**, tests/utils/test_dates.py and tests/ingest_fixtures/** that build pandas frames (FIRST step)
- create: tests/market_data/risk_factors/test_polars_edge_contract.py; tests/market_data/risk_factors/golden/canonical_history_matrix_v0_2_0.npz
FIRST: on the v0.2.0 checkout, record the C.15 and C.16 goldens. Then list every test file that passes a pandas frame to an entry of this slice. STOP AND ASK for a hit outside the Files list.
Prohibited: `market_data/assemble.py`; `simlab/**`; `solutions/{risk_run,run_identity,scenario_models}/**`; the D076 normalisation code paths; `portfolio/ingest/**`; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/workstreams/WS_CREDIT_POLARS/A_INVENTORY.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_market_data.txt ; context_riskquant_utils.txt ; context_riskquant_ingest_fixtures.txt ; context_riskquant_portfolio.txt ; context_riskquant_tests_market_data.txt ; context_riskquant_tests_portfolio.txt ; context_riskquant_tests_e2e.txt
Gates: gate block; `$t = "tests/market_data", "tests/portfolio", "tests/utils", "tests/ingest_fixtures", "tests/e2e"`; demo full over the 5 changed demos and the 15 credit_var demos; `scripts/release/simlab/certify_simlab_multicore.py` per D044 if `RiskFactorMatrix` members change.
Exit: every S7 test passes; full suite passes; `grep -rn "import pandas" src/riskquant src/riskquant_lab` returns no hit.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S7.T1 | `build_risk_factor_matrix_from_df` with a pandas frame, a `pl.LazyFrame`, a frame without `date`, a tz-aware `pl.Datetime` date | TypeError; TypeError; ValueError; TypeError (D100 form) | silent acceptance |
| S7.T2 | The canonical history fixture through the builder against the npz record | `values` bitwise equal, float64, same layout; date ordinals equal; `factor_ids` equal | D017 or D076 drift |
| S7.T3 | Each Polars view member of O-7 (a) on the canonical matrix and snapshot | column names equal `factor_ids`; `date` column equals the ordinals; no pandas in `sys.modules` | view drift |
| S7.T4 | `CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA` through `pl.from_dict` and `compile_raw_df_into_ingest_batch` against the v0.2.0 batch | every SoA array equal; the dict holds `None`, no float NaN, in str columns | F-14 |

### S8: Dependency removal, enforcement and release

Objective: Remove pandas and pandas-stubs from pyproject.toml, enforce the package rule, and release 0.2.1.
Contract: C.1, C.2, C.3, C.13 (branch removal), C.17
Depends on: S1 to S7 VALIDATED, O-6
Files:
- create: tests/test_pandas_free_package.py
- modify: pyproject.toml (dependencies, version); src/riskquant/utils/printer.py (remove the C.13 legacy branch); src/riskquant/portfolio/ingest/batch.py (docstring, D096 citation); src/riskquant/solutions/credit_var/api.py (docstring, if it names pandas); docs under `docs/` that name pandas as a dependency
Prohibited: every other src file; DECISIONS.md (the owner promotes P-CREDIT_POLARS-001 and edits the D106 pins line).
Attach: docs/workstreams/WS_CREDIT_POLARS.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_tests_root.txt ; context_riskquant_tests_portfolio.txt ; pyproject.toml
Gates: gate block; `$t = "tests/test_pandas_free_package.py", "tests/solutions/credit_var"`; `pip uninstall -y pandas pandas-stubs`, `pip install -e .`, then `pytest -q` again in that environment; `python -c "import importlib.metadata as m; print(m.version('riskquant'))"`; demo full over the 15 credit_var demos in the environment without pandas.
Exit: every S8 test passes; full suite passes with pandas uninstalled; installed version prints 0.2.1; the final report records `pip show pandas` returning not found.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S8.T1 | AST of every `.py` file under `src/riskquant` and `src/riskquant_lab` | no `Import` or `ImportFrom` of `pandas` or `pandas.*` at any depth | an in-function or TYPE_CHECKING import (C.1) |
| S8.T2 | `tomllib` parse of pyproject.toml | no requirement named `pandas` or `pandas-stubs` in dependencies or any optional group; version 0.2.1; numpy, scipy, numba and pydantic specs as D106.7 and C.30 of WS_CREDIT_VAR_API | cap or dependency left behind |
| S8.T3 | Child process with a `sys.meta_path` finder that raises ImportError for `pandas` and `pandas.*`; `pkgutil.walk_packages` over both trees; `run_credit_var` for the 13 keys of R | every module imports; 13 results; JSON equal to the C.4 record; `"pandas" not in sys.modules` | a lazy pandas import on the run path (C.3) |
| S8.T4 | `print_df` with an object that has `to_string` and is not Polars | TypeError | legacy branch left in (C.13) |
| S8.T5 | `api.py` and `batch.py` docstrings | no line matches `(?i)\bpandas\b` except a D096 history note | stale documentation |

Appendix: A_INVENTORY.md (S3, S4, S5, S7)

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| A consumer passes a pandas frame to `from_dataframe` (riskquant-emissary, riskquant-excel, notebooks) | TypeError after 0.2.1 | The member is outside the stable surface (D106.1); the release note names `pl.from_pandas`; owner input on consumers (section 1 value case) |
| The tests profiles were not read (F-21, I-009) | A slice finds test files outside its Files list | Every slice has a FIRST step with STOP AND ASK |
| `pl.Series.to_numpy()` is read-only and null columns become float64 with NaN (F-12) | A Numba kernel rejects a read-only array; an int column with a null compiles as float | `_shared._f64` already copies a read-only view; C.8 and S4.T2 reject or map nulls before the cast |
| Polars diagonal concat changes NaN to None in the exhaustive fixture dict (F-14) | A test that compares the dict literally fails | C.16 compares the compiled batch, not the dict |
| The transitional mixed state of `recipes.py` between S4 and S5 | Two reviews of one file | The owner may deliver S4 and S5 in one prompt |
| pandas stays installed through seaborn in the scripts and notebooks groups (F-19) | A pandas import could survive unseen | S8.T1 is static and S8.T3 blocks the import by name |
| S7 touches D017, D064 and D076 contracts | A market-risk value changes without a credit test noticing | C.15 golden; D044 certification script when `RiskFactorMatrix` members change; O-1 may split S7 out |
| polars 2.x is current (sandbox 2.1.0); pyproject declares `polars>=1.0` | API differences between 1.x and 2.x | The implementer records `polars.__version__` in each report; a pin decision goes to the backlog |

Deferred:
- A Numba `njit` version of the C.5 function: opens when a benchmark shows the Python loop above 1 ms per 10,000 loading rows.
- A polars upper bound in pyproject.toml: opens on the first polars 2.x incompatibility.

Backlog (REFINEMENT findings and items that v1 omits):
- WS_CREDIT_VAR_API backlog items that S6 (b) closes: bool, Decimal and str cells of `from_dataframe`; the `loadings_table` docstring says `uid` (R1 R-19).
- `golden_master_credit_exposure.py` holds tests inside `src/`; move to `tests/` (adjacent, not this workstream).
- `minimal_factor_data.py:_debug_print` uses `DataFrame.corr(method="spearman")` and matplotlib; S4 keeps the print with numpy or removes it.
- `print_dict_of_dfs` transposes frames wider than 20 columns; keep or drop in S2.
- `scripts/` files that import pandas for their own tables and call no changed entry stay as they are (A_INVENTORY.md section 3).

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
