# WS_FACTOR_REGRESSION: Factor regression research library

Path: docs/workstreams/WS_FACTOR_REGRESSION.md
Class: HEAVY
Status: FROZEN
Revision: 3
Revision note: Audit R1 (R_FREEZE_RECONCILIATION_R1.md) BLOCKING B-1 to B-7 applied (B-1 with the owner fitted-value amendment); owner gate O-8 option 1 applied; owner-directed refinements R-1 to R-5, R-13, R-14, R-16 applied; other refinements moved to the backlog; snapshot changed from d4ba609 dirty=true to the audit snapshot 05314f8 dirty=false. Owner 2026-10-12: section 1.8 option A, build v1 now. Known gap I-001: the S6 and S7 Exit text omits the mypy baseline line, which prints exit=1 by design (13 errors in hashing.py). Owner edits after revision 3, non-contract: section 7 mypy risk row, A.6 heading, A.7 centring rows, appendix Status lines.
Date: 2026-10-12
Branch: ws/factor-regression
Snapshot: SNAPSHOT commit=05314f8 dirty=false content_hash=358d3f657c81a072
Drafted by: investigator
Independent review: NONE
Audit rounds used: 1 of 2
Code merged: no
DECISIONS.md reconciled: NO
Evidence limits: No code exists and no test, mypy or ruff gate ran. Sandbox: Python 3.12.3, numpy 2.4.4, scipy 1.17.1; numpy 1.26
not tested. No import-all command exists in the repository (F-17), so each gate imports the slice modules explicitly. stx-data-god
was not attached.

## 1. Goal and scope

Goal: A researcher fits and tests linear relations between observed series and factors with explicit, reproducible,
fail-loud contracts, and the result binds to its exact sample and specification.

Value case:
- Benefit: One reusable fit path with recorded alignment, a rank gate, a conditioning report, joint Wald tests and
  digests. Today each fit is an ad-hoc NumPy call (F-2). The failure that it prevents: unscaled `lstsq` silently drops a
  rank at a mean to sd ratio of 1e7 (F-11).
- Cost proxies: 7 slices; 16 new `.py` files (2 empty) and 11 new test files; 1 modified shared file
  (`tests/test_architecture.py`); 1 read-only reuse of `digest_named_arrays`; 1 new decision (P-FACTOR_REGRESSION-001);
  at most 2 audit rounds.
- Smallest alternative and why it fails: "Do nothing" keeps ad-hoc fits with no sample binding. One script under `scripts/`
  can meet every numerical clause but fails on reuse: `pyproject.toml` packages no `scripts/` path (F-9) and no
  mechanical gate covers `scripts/` (F-8).
- Case against: The facade adds a public surface under `riskquant.solutions` for a research tool. `riskquant.dependence`
  has no facade and no D entry that exempts it (F-13, F-14), so the repository has no single precedent for this shape.
- Class: HEAVY (owner-confirmed). Criteria: a new D decision (P-FACTOR_REGRESSION-001) and a new public signature under
  `src/riskquant/solutions`.
- Recommendation at drafting: PROCEED SMALLER (no D088 export, no stationarity tests, no Polars adapter) [Proposed]

In scope (v1):
- Core `riskquant.factor_regression`: integer alignment, rows and transforms, scaled SVD rank gate, condition number,
  OLS, covariance IID, HC0, HAC_BARTLETT with a declared lag, integer-coded diagnostics, joint Wald
- Facade `riskquant.solutions.factor_regression`: `ObservedSeries`, `ModelSpec`, results, names, units, labels,
  digests, and the public functions `align_levels`, `fit_ols`, `wald_test`

Out of scope:
- D088 proxy loading export (O-4); stationarity, cointegration and break tests (O-5); automatic lag selection,
  HC1 to HC3 (O-3); Polars adapter (O-6); lagged regressors, GLS, WLS, panel fits, rolling windows; persistence
- Any change to existing `src/` files

Owner decisions in force:
- 2026-10-12: O-1 B2, O-1a, O-2 (A), O-3 (A), O-4 (A), O-5 (A), O-6 (A), O-7 (A), O-8 (option 1). See section 5.

Depends on: `riskquant.exceptions.RiskQuantError`; `riskquant.solutions.run_identity.hashing.digest_named_arrays`.
Blocks: none.

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | The only regression kernel in `src/` is `_ar1_one_col`: one regressor, raw moments, `df` clamped to 1, `beta = 0` when `abs(t) < 1.96`. | `src/riskquant/simlab/scenarios/core_math/ar1_filter.py:_ar1_one_col` | Observed |
| F-2 | Other regression code is in scripts or tests only: `linregress` in two demo scripts, `np.linalg.lstsq` in one test, closed-form OLS in one oracle. No library. | `grep -n -i -E` for linregress, lstsq, polyfit over the scripts, tests, simlab, lab, dependence profiles | Observed |
| F-3 | `observations.compile.compile_sample` takes a `(K, S)` matrix with `K >= 2`, has no date axis, copies and freezes, and reports cell addresses as indices. The validators are private. | `src/riskquant/observations/compile.py:compile_sample`, `_validate.py:reject_degenerate_shape`, `reject_non_finite` | Observed |
| F-4 | `digest_named_arrays` accepts arrays only and folds in name, dtype, shape and C-order bytes. | `src/riskquant/solutions/run_identity/hashing.py:digest_named_arrays` | Observed |
| F-5 | D025 locks the level domains at the generator edge: LOG above 0, SIMPLE non-zero, DIFFERENCE finite. `validate_shift_domain` raises with a factor id and no date. | `DECISIONS.md` D025; `src/riskquant/simlab/builder/shift_domain.py:validate_shift_domain` | Observed |
| F-6 | D088 section 2 has no intercept term. Section 3 gives the loading to the proxy model and calibration provenance to governance. Authoring format is OPEN. No decision defines a calibration method. | `DECISIONS.md` D088 sections 2, 3, 5 and the Stage E note | Observed |
| F-7 | `riskquant_lab` holds fixtures, plotting and display only, with no tests and a non-empty `__init__.py`. | `context_riskquant_lab.txt` | Observed |
| F-8 | `test_every_init_file_is_empty`, `test_every_module_declares_its_repository_relative_path` and `tests/test_no_stale_path_comments.py` scan `src/riskquant` recursively. The FORBIDDEN rules cover the package tuple of `test_core_packages_obey_the_mechanical_rules` only. | `tests/test_architecture.py:SOURCE_ROOT`, `FORBIDDEN`; `tests/test_no_stale_path_comments.py:SRC_ROOT` | Observed |
| F-9 | Wheel packages: `src/riskquant`, `src/riskquant_lab`. Declared: numpy>=1.26, scipy>=1.15, polars, pandas. Not declared: statsmodels, pandera, pydantic. | `pyproject.toml` | Observed |
| F-10 | `utils.dates` converts dates to int32 ordinals and imports pandas only inside functions. | `src/riskquant/utils/dates.py:date_to_ord32` | Observed |
| F-11 | At mean to sd 1e7 with an intercept, unscaled `lstsq(rcond=None)` returns rank 1 and a wrong slope. Scaled solves match the exact Fraction value. | Appendix A.7, sandbox | Observed |
| F-12 | With A.1 ratio-form log returns, 250 rows and tolerance 250*eps, the scaled-SVD gate detects the exact log identity at vol 1%, 0.5% and 0.1% (200 of 200) and misses it at vol 0.05%, 0.02% and 0.01% (0 of 200, kappa 1.9e12 to 1.7e13). The scaled simple-return design has rank 3 and kappa * vol between 2.4 and 3.6. | Appendix A.7, sandbox | Observed |
| F-13 | `riskquant.dependence` has no caller in `src/` outside its package. `riskquant.observations` is imported outside its package by `dependence_stress.engine`, `dependence_stress.permutations` and `market_state.indices` only. No `solutions/dependence/api.py` exists. | index DEPENDENCIES section: grep for edges into `riskquant.dependence` or `riskquant.observations`, minus edges that start in those two packages; index MANIFEST grep for `src/riskquant/solutions/dependence` gives 0 | Observed |
| F-14 | No D entry exempts `riskquant.dependence` from the facade rule of D001. | `grep -n -i "dependence"` in `DECISIONS.md`: 27 lines, none names the package; `grep "riskquant.dependence"` gives 0 | Observed |
| F-15 | No module under `src/riskquant` outside `riskquant.solutions` imports `riskquant.solutions`. | index DEPENDENCIES grep `-> .*riskquant.solutions` minus left side `riskquant.solutions`: 0 lines; source grep over all attached src profiles: 0 | Observed |
| F-16 | `hashing.py` imports `riskquant.config.proxy_model.compiled` under `if TYPE_CHECKING:` only. At runtime it imports stdlib and numpy only. | `src/riskquant/solutions/run_identity/hashing.py` lines 22 to 33; index line `riskquant.solutions.run_identity.hashing -> riskquant.config.proxy_model.compiled` | Observed |
| F-17 | No import-all command exists in the repository. | `grep -n -i -E "import.all, walk_packages, iter_modules, compileall, py_compile"` over the tests and scripts profiles: only a config-enums loop and one docstring | Observed |

## 3. Contract

All clauses describe new code. Each is Proposed. Formulas are in `A_NUMERICS.md`. Module placement is in A.8.

- C.1 `ObservedSeries(name, unit, date_ord, values)`: `name` and `unit` are non-empty `str` without control characters;
  `date_ord` is 1-D `int32`, strictly increasing; `values` is 1-D `float64`, all finite; lengths are equal and at least 1.
  The constructor copies both arrays and marks them read-only. It never sorts, casts, deduplicates or drops.
  A violation raises `SeriesContractError` naming the series and, for a value, its `date_ord`. [Proposed]
  Test: S6.T1, S6.T2, S6.T3, S6.T4
- C.2 `align_levels(series, *, start_ord, end_ord)` keeps exactly the dates `d` with `start_ord <= d < end_ord` that occur in every
  series. It does no fill, interpolation, as-of join, sort or value change. Series names must be unique.
  An empty result, `start_ord >= end_ord` or a duplicate name raises `AlignmentError`. [Proposed]
  Test: S1.T5, S1.T7, S7.T1, S7.T10
- C.3 `AlignedLevels.exclusions` holds one record per input observation that is not kept: series index, date ordinal and reason
  in {OUTSIDE_WINDOW, NOT_COMMON_DATE}. For every input series, kept count plus excluded count equals its length.
  No observation appears twice. [Proposed] Test: S1.T6, S7.T1
- C.4 Rows are built after alignment from aligned levels only. The library never transforms a series on its own calendar and
  aligns afterwards. [Proposed] Test: S2.T1
- C.5 `ModelSpec` has no default for any field: response, regressors, intercept, step, stride, covariance, `rank_rtol`,
  `condition_threshold`. An omitted field raises `TypeError`. Invalid values raise `ModelSpecError`: empty design,
  `step < 0`, `stride < 1`, step 0 with a return term, step above 0 with only LEVEL terms, `rank_rtol` outside (0, 1),
  `condition_threshold` not None and not above 1, a lag with IID or HC0, a negative lag, an unknown term series.
  [Proposed] Test: S6.T5, S6.T6, S7.T8
- C.6 Row construction follows A.1: end index `h + r*s`, endpoint two-point read, recorded row start and end dates, overlap
  order `q`. Aligned dates that no row uses are recorded as NOT_USED_BY_ROWS. Fewer than one row raises
  `InsufficientObservationsError`. Under LOG_RETURN, `r_Q = r_Z + r_S` holds to 1e-15 absolute when `Q = Z*S`; under
  SIMPLE_RETURN it does not. [Proposed] Test: S2.T2, S2.T3, S2.T4, S2.T8, S2.T9
- C.7 Domain rules, evaluated before any return: LOG_RETURN needs every aligned level of that series above 0;
  SIMPLE_RETURN needs every aligned level non-zero; LEVEL and DIFFERENCE need finite levels only (C.1).
  A violation raises `TransformDomainError` naming series, transform, first `date_ord` and level.
  The rules follow the D025 convention. D025 does not lock them for this library. [Proposed]
  Test: S2.T5, S2.T6, S2.T7, S7.T2
- C.8 The design `X` has the intercept column of ones first if declared, then the regressors in declared order. Each column is
  divided by its 2-norm. A zero column raises `RankDeficiencyError` naming it. Rank, conditioning and solve use the scaled design.
  [Proposed] Test: S3.T1
- C.9 The rank gate is `numpy.linalg.svd` of the scaled design with `rank_rtol * s[0]` as tolerance. `rank < k` raises
  `RankDeficiencyError` before any solve. The core error carries rank, singular values, a participation score per column,
  the involved column indices and the null relations in original units (A.2). The facade adds the column names.
  `numpy.linalg.matrix_rank` agrees in the tests. [Proposed] Test: S3.T2, S3.T3, S3.T4, S7.T3
- C.10 `kappa = s[0] / s[k-1]` of the scaled design is reported for every full-rank design. There is no default threshold.
  If `condition_threshold` is not None and `kappa` is above it, the diagnostic CONDITION_ABOVE_THRESHOLD is recorded.
  [Proposed] Test: S3.T5, S3.T6, S3.T7
- C.11 The core reports diagnostics as an `int8` code array and a `float64` value array. The facade maps them to frozen
  `Diagnostic(code, value, message)` records. The library calls no `warnings.warn`. The code set is closed:
  CONDITION_ABOVE_THRESHOLD, LEVEL_TERM_PRESENT, R2_ABOVE_DURBIN_WATSON, OVERLAP_NOT_CORRECTED. Rules are in A.4.
  A diagnostic never blocks a fit. [Proposed] Test: S3.T8, S3.T9, S3.T10, S7.T6
- C.12 The core fit (`fit.fit_core`, called by `api.fit_ols`) needs `R >= k + 1`, otherwise it raises `InsufficientObservationsError`
  naming `R` and `k`. It solves on the scaled design from the same SVD as the gate (A.3). Coefficients match an exact
  Fraction solution to 1e-12 relative on a small case. At a mean to sd ratio of 1e7 the slope matches to 1e-8 relative and
  the centred intercept a + b * mean(x) matches to 1e-12 relative, and the fitted values match to 1e-12 relative. The raw
  intercept is not compared, because its error equals the slope error times mean(x).
  [Proposed] Test: S4.T1, S4.T2, S4.T13
- C.13 Covariance types are IID, HC0 and HAC_BARTLETT with a declared lag (A.3). HAC with lag 0 equals HC0 to 1e-12 relative.
  A lag above `R - 1` raises `ModelSpecError`. p-values use Student t with `dof = R - k`. [Proposed]
  Test: S4.T3, S4.T4, S4.T5, S4.T6, S7.T11
- C.14 `DegenerateFitError` is raised when `rss <= (R * eps)^2 * sum(y^2)` with `eps` the float64 machine epsilon. This rule
  includes `rss == 0.0` and an all-zero `y`, so an exact fit raises whether rounding gives 0.0 or a value near 1e-26. It is
  also raised when the response has zero variance with an intercept, or when a covariance diagonal is not positive.
  The result never holds NaN. [Proposed] Test: S4.T7, S4.T8
- C.15 `OlsResult` carries: `coef`, `se`, `t_stats`, `p_values`, `cov`, `resid`, `fitted`, `dof`, `n_obs`, `rss`, `r_squared`,
  `durbin_watson`, `condition_number`, `singular_values`, `diagnostics`, column names, units, transform per column, response
  term, window, step, stride, overlap order, row start and end dates, exclusions, covariance spec, `rank_rtol`,
  `condition_threshold`, three digests and `ENGINE_VERSION`. All arrays are read-only. It has no field named accepted,
  approved or valid: the library approves no assumption. [Proposed] Test: S6.T12, S7.T9
- C.16 `sample_digest` covers row start and end dates, column names, `y` and `X` (A.6). `spec_digest` covers the canonical
  spec bytes of A.6, including units and window. `row_digest` covers row dates only. All three are stable across fresh
  processes and change under the mutations of S6.T8. [Proposed] Test: S6.T7, S6.T8, S6.T9
- C.17 Two results are comparable when `row_digest` and the response term are equal. Models on one `AlignedLevels` with equal step
  and stride share `row_digest`. [Proposed] Test: S7.T4
- C.18 `wald_test(result, R, r)` tests `R @ coef = r` as one joint statistic with `result.cov` (A.5). It never combines
  separate t-tests. It returns `W`, `F = W/J`, both p-values, the digests of the result and of `(R, r)`, and
  `ENGINE_VERSION` (O-8). Shape errors,
  non-finite entries, rank-deficient `R` or a non-positive-definite `R @ cov @ R.T` raise `RestrictionError`. [Proposed]
  Test: S5.T1, S5.T2, S5.T3, S5.T4, S5.T5, S7.T5
- C.19 No library function mutates a caller array. This covers the series arrays, the fit inputs, and `R` and `r` of the Wald test.
  Retained arrays are copies and read-only. [Proposed]
  Test: S6.T4, S4.T8, S7.T5
- C.20 Every exception class that the library defines subclasses `riskquant.exceptions.RiskQuantError`. Core exceptions carry
  no str data: integer indices, integer codes and numeric values only. The facade catches each core exception and raises
  the exception that the clause names, with series names and dates, chained as `__cause__` (a `raise` with `from`). The cell-addressed messages of C.1 and C.7 are facade messages.
  [Proposed] Test: S1.T9, S6.T11, S7.T2
- C.21 Direct imports (any `import` or `from` statement, `TYPE_CHECKING` blocks included). The core imports only the standard
  library, numpy, `scipy.stats`, `riskquant.exceptions` and its own modules. It imports no `riskquant.solutions`,
  `riskquant.config`, `riskquant.simlab`, `riskquant.valuation`, `riskquant.credit`, pandas, polars, matplotlib or numba.
  The facade imports only the standard library, numpy, the core, `riskquant.exceptions` and, in `digest.py` only,
  `riskquant.solutions.run_identity.hashing`.
  The facade has no direct import of `riskquant.config`; the `TYPE_CHECKING` edge of `hashing.py` to
  `riskquant.config.proxy_model.compiled` (F-16) is the only transitive edge to `riskquant.config` and is allowed.
  No module outside the two packages and `tests/` imports `riskquant.factor_regression`. Both packages pass the existing
  empty-init and path-comment tests; the core passes the FORBIDDEN rules. [Proposed]
  Test: S1.T10, S1.T11, S1.T12, S6.T13, S6.T14
- C.22 Tier split. The core holds no `str`-typed field, parameter or return and no Enum; it holds no names, units or labels.
  It takes `int32` date ordinals and `float64` arrays, and it reports positions as integer indices and diagnostics, transforms
  and reasons as integer codes (`codes.py`). The facade holds `ObservedSeries`, names, units, the spec, result labels, all digests
  and the mapping from core indices and codes to names and dates. Placement per module: A.8. [Proposed]
  Test: S1.T13, S6.T10
- C.23 `riskquant.solutions.factor_regression.api` is the only public import surface (D001). Its public functions are
  `align_levels`, `fit_ols` and `wald_test`. Its `__all__` also lists the contract types and facade exceptions. The contract types
  are frozen dataclasses with read-only arrays, not Pydantic (O-1a). Array-bearing contracts declare `eq=False` (D064
  precedent). A facade module imports the core; the core never imports the facade. [Proposed] Test: S7.T7, S1.T11, S6.T12

## 4. Decisions

Decisions relied on: D001, D002, D004, D023, D025, D088. Each checked against `DECISIONS.md`: yes.

### P-FACTOR_REGRESSION-001: Factor regression core and facade boundary (PROPOSED)

```text
Related: D001, D002, D025, D088
Scope: Fixes the tier split, the facade form and the limits of the factor regression library.
1. riskquant.factor_regression is a private CORE package (D001). It holds int32 date ordinals, float64 arrays,
   integer indices and integer codes only. It imports no riskquant.solutions and no riskquant.config.
2. riskquant.solutions.factor_regression.api is the only public import surface. Facade modules import the core.
   The core never imports the facade. No other module outside tests imports the core.
3. Exemption from Pydantic v2 solution contracts: the facade contracts are frozen dataclasses with read-only arrays.
   Reason: no JSON boundary exists, the result holds arrays, and pydantic is not declared in pyproject.toml (F-9).
4. The library approves no model assumption. It reports numbers and diagnostics only.
5. No D088 feed: no fit becomes a proxy loading. A loading enters a proxy model only through the D088 authoring path.
6. The D025 level-domain convention applies to every return transform in the library.
Evidence: S1.T10, S1.T11, S1.T13, S2.T5 to S2.T7, S6.T12, S7.T7
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Location: B2, private core `src/riskquant/factor_regression` plus facade `src/riskquant/solutions/factor_regression/api.py`. The name avoids "regression", which names the test suite in `DECISIONS.md`. | CLOSED | 2026-10-12 | owner |
| O-1a | Facade contract form: frozen dataclasses with read-only arrays, not Pydantic; recorded in P-FACTOR_REGRESSION-001 clause 3. | CLOSED | 2026-10-12 | owner |
| O-2 | Estimator source: own numpy and scipy code; no new dependency. | CLOSED | 2026-10-12 | owner |
| O-3 | HAC scope: HAC_BARTLETT with a declared lag; Student t p-values. | CLOSED | 2026-10-12 | owner |
| O-4 | D088 link: none. | CLOSED | 2026-10-12 | owner |
| O-5 | Stationarity tests: none in v1. | CLOSED | 2026-10-12 | owner |
| O-6 | Public data interface: arrays and `str` only at the facade; Polars adapter in the backlog. | CLOSED | 2026-10-12 | owner |
| O-7 | Rank tolerance: the caller declares `rank_rtol`; no default. | CLOSED | 2026-10-12 | owner |
| O-8 | Provenance: option 1. `OlsResult` and `WaldResult` carry `ENGINE_VERSION` only. The environment record and the timestamp (audit R1 options 2 and 3) are in the section 7 backlog. Reason: revision 3 is not audited, and a field added later is additive. | CLOSED | 2026-10-12 | owner |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1, then S2 and S3 and S6 in parallel; S4 after S2 and S3; S5 after S4; S7 after S5 and S6.
Every Attach list also holds `CLAUDE.md`, `DECISIONS.md` and `WORKFLOW.md`. Each slice gate block is in `A_TESTS.md`
(WORKFLOW.md section 6 order: py_compile, import, ruff check, ruff format --check, mypy --strict, pytest slice, pytest full).

### S1: Core skeleton, integer alignment, architecture gates

Objective: Create the core package, its exceptions and codes, integer alignment and the architecture gates.
Contract: C.2, C.3, C.20, C.21, C.22
Depends on: none
Files:
- create: `src/riskquant/factor_regression/__init__.py` (empty), `errors.py`, `codes.py`, `align.py`
- create: `tests/factor_regression/test_errors.py`, `tests/factor_regression/test_align.py`
- modify: `tests/test_architecture.py` (add `"factor_regression"` to the package tuple of
  `test_core_packages_obey_the_mechanical_rules`; add `test_factor_regression_import_direction` and
  `test_factor_regression_core_holds_no_str_type`)
Prohibited: any other change under `src/` or `tests/`; `pyproject.toml`; any `str` annotation or Enum in the core
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`, repo files `tests/test_architecture.py`
and `tests/test_no_stale_path_comments.py`
Gates: `A_TESTS.md` section S1, gate block.
Exit: Every gate line prints `exit=0`. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: commit 25f9ba0; Windows gates exit=0 on every line, 5834 passed, 36 skipped (gate_s1_head.txt); independent review ACCEPT, R-001 to R-005 closed.
Tests: S1.T5 to S1.T7, S1.T9 to S1.T13. The table is in `A_TESTS.md`, section S1.
Appendix: A_NUMERICS.md (A.8); A_TESTS.md (S1 table and gates)

### S2: Core rows and transforms

Objective: Create core row construction and the domain checks on integer-coded transforms.
Contract: C.4, C.6, C.7
Depends on: S1
Files:
- create: `src/riskquant/factor_regression/rows.py`, `tests/factor_regression/test_rows.py`
- modify: none
Prohibited: import of `ShiftType` or `shift_domain` (F-5: they carry NO_SHIFT, ABSOLUTE_BASE and no date)
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`
Gates: `A_TESTS.md` section S2, gate block.
Exit: Every gate line prints `exit=0`. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: none
Tests: S2.T1 to S2.T9. The table is in `A_TESTS.md`, section S2.
Appendix: A_NUMERICS.md (A.1, A.7); A_TESTS.md (S2 table and gates)

### S3: Core rank gate, conditioning and diagnostics

Objective: Create the scaled-design SVD gate, the condition number and the integer-coded diagnostic functions.
Contract: C.8, C.9, C.10, C.11
Depends on: S1
Files:
- create: `src/riskquant/factor_regression/design.py`, `diagnostics.py`
- create: `tests/factor_regression/test_design.py`, `tests/factor_regression/test_diagnostics.py`
- modify: none
Prohibited: a default for `rank_rtol` or `condition_threshold`; `numpy.linalg.lstsq` in `src/`; `warnings.warn`
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`
Gates: `A_TESTS.md` section S3, gate block.
Exit: Every gate line prints `exit=0`. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: none
Tests: S3.T1 to S3.T10. The table is in `A_TESTS.md`, section S3.
Appendix: A_NUMERICS.md (A.2, A.4, A.7); A_TESTS.md (S3 table and gates)

### S4: Core OLS fit and covariance

Objective: Create the core fit, the three covariance types and the core fit result.
Contract: C.12, C.13, C.14, C.19
Depends on: S2, S3
Files:
- create: `src/riskquant/factor_regression/covariance.py`, `fit.py`
- create: `tests/factor_regression/test_covariance.py`, `tests/factor_regression/test_fit.py`
- modify: none
Prohibited: `lstsq`; unscaled solves; any digest in the core
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`
Gates: `A_TESTS.md` section S4, gate block.
Exit: Every gate line prints `exit=0`. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: none
Tests: S4.T1 to S4.T8, S4.T13. The table is in `A_TESTS.md`, section S4.
Appendix: A_NUMERICS.md (A.3, A.4, A.7); A_TESTS.md (S4 table and gates)

### S5: Core joint Wald test

Objective: Create the core joint Wald statistic.
Contract: C.18
Depends on: S4
Files:
- create: `src/riskquant/factor_regression/wald.py`, `tests/factor_regression/test_wald.py`
- modify: none
Prohibited: combining separate t-tests; a library oracle for the p-values in the tests
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`
Gates: `A_TESTS.md` section S5, gate block.
Exit: Every gate line prints `exit=0`. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: none
Tests: S5.T1 to S5.T5. The table is in `A_TESTS.md`, section S5.
Appendix: A_NUMERICS.md (A.5, A.7); A_TESTS.md (S5 table and gates)

### S6: Facade contracts, labels and digests

Objective: Create the facade types, the code-to-label mapping, the facade exceptions and the digests.
Contract: C.1, C.5, C.15, C.16, C.19, C.20, C.21, C.22
Depends on: S1
Files:
- create: `src/riskquant/solutions/factor_regression/__init__.py` (empty), `contracts.py`, `errors.py`, `labels.py`, `digest.py`
- create: `tests/solutions/factor_regression/test_contracts.py`, `tests/solutions/factor_regression/test_digest.py`
- modify: none
Prohibited: Pydantic; import of `hashing` outside `digest.py`; changes to `riskquant.solutions.run_identity`
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`, `context_riskquant_solutions_run_identity.txt`
Gates: `A_TESTS.md` section S6, gate block.
Exit: Every gate line prints `exit=0`, except the `mypy --strict` line. That line passes when its `: error:` lines equal
the `: error:` lines of the `mypy baseline` line. The cross-process digest test (S6.T7) passes. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: none
Tests: S6.T1 to S6.T14. The table is in `A_TESTS.md`, section S6.
Appendix: A_NUMERICS.md (A.6, A.8); A_TESTS.md (S6 table and gates)

### S7: Facade entry points

Objective: Create `api.py` with `align_levels`, `fit_ols` and `wald_test`, and the core-to-facade error mapping.
Contract: C.2, C.3, C.5, C.7, C.9, C.11, C.13, C.15, C.17, C.18, C.19, C.20, C.23
Depends on: S5, S6
Files:
- create: `src/riskquant/solutions/factor_regression/api.py`, `tests/solutions/factor_regression/test_api.py`
- modify: none
Prohibited: numerical logic in the facade; a second public module; Pydantic
Attach: `docs/workstreams/WS_FACTOR_REGRESSION.md`, `docs/workstreams/WS_FACTOR_REGRESSION/A_TESTS.md`,
`docs/workstreams/WS_FACTOR_REGRESSION/A_NUMERICS.md`, `context_riskquant_core.txt`
Gates: `A_TESTS.md` section S7, gate block.
Exit: Every gate line prints `exit=0`, except the `mypy --strict` line. That line passes when its `: error:` lines equal
the `: error:` lines of the `mypy baseline` line. pytest full: 0 failed, 0 errors, 36 skipped or fewer.
Status: NOT_STARTED
Evidence: none
Tests: S7.T1 to S7.T11. The table is in `A_TESTS.md`, section S7.
Appendix: A_NUMERICS.md (A.7, A.8); A_TESTS.md (S7 table and gates)

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| A second public module appears in the facade | D001 surface leaks | S7.T7; C.23 |
| numpy 1.26 lacks `rtol` in `matrix_rank` | none: library and tests use `svd` and `tol` | run S3.T3 on numpy 1.26 if the floor stays |
| rank gate misses a near-identity | a singular fit proceeds | `kappa` always reported (C.10, S3.T4) |
| mypy `--strict` on the facade follows the import of `hashing.py` | gate output shows 13 errors from `hashing.py` itself (owner baseline, exit=1) | S6 and S7 Exit compare the error lines with the mypy baseline line |

Deferred:
- Numba kernel for the HAC meat: opens when the owner states a throughput need. Benchmark to write first: `fit_ols` with
  HAC_BARTLETT at `(R, k, L)` of `(2500, 8, 10)`, `(25000, 8, 50)`, `(250000, 8, 100)`, best of 5 wall times. No claim is made now.

Backlog (REFINEMENT findings and items that v1 omits):
- HC1 to HC3; automatic lag selection; stationarity, cointegration, Chow and CUSUM tests; lagged regressors
- Named restriction syntax; unit algebra checks; explicit NaN exclusion policy; model comparison helper; JSON persistence
- Polars adapter (O-6); adapter from `RiskFactorMatrix`; D088 candidate-loading export (O-4)
- Adjacent: the non-empty `src/riskquant_lab/__init__.py` (D002) is outside the empty-init test (F-7, F-8)
- O-8 option 2: environment record (riskquant, python, numpy, scipy versions) outside every digest, as in D040
- O-8 option 3: UTC timestamp outside every digest (D042 identity versus record); S7.T6 must then exclude it
- R-6: S4.T1, S4.T4, S4.T5 elementwise 1e-12 relative fails on exact zeros; use `max abs error <= 1e-12 * max abs exact`
- R-7: S2.T9 SIMPLE identity to 1e-15 absolute holds up to vol 10% only; state the vol in the case
- R-8: C.14 threshold base and factor (centred base misses exact fits; `4*R*eps^2` candidate from a sample only)
- R-9: `wald_test(result, R, r)` and uppercase locals of A.3 and A.5 fail ruff N803 and N806; rename or permit noqa
- R-10: state tolerances for S3.T1 ("unchanged"), S4.T3 (p-value "equals") and S5.T3 ("equals")
- R-11: state whether the S3 diagnostic function takes `q` as input or computes it from `h` and `s`
- R-12: "AlignedLevels" names a core type (A.1) and a facade type (C.17, S7.T8); use two names
- R-15: C.1 "lengths are equal and at least 1" has no test; add cases to S6.T2
- R-17: value case: a `scripts/` module imports in pytest and from the repository root; it fails outside the checkout
- R-18: `numpy.linalg.matrix_rank(tol=)` exists in numpy 1.26; `rtol` came in numpy 2.0 [Inferred]
- S1 guard limits (owner OD-3): the C.21 check does not cover getattr, rebinding the scipy name, quoted annotations, importlib.import_module or __import__. Reviewers check these forms by eye.
- S1 guard residual (owner OD-3): the outside-import rule (no module outside the two packages imports the core) has no committed negative test. The core guard also misses a riskquant.config read after import riskquant.exceptions. Only a slice that lists tests/test_architecture.py in its Files can add these.
- Core exceptions are not picklable: their __init__ takes keyword-only arguments. Add __reduce__ in errors.py only if a process pool is ever used.
- S1 guard residual (owner OD-4, R-006): the no-str and no-Enum check (S1.T13) misses aliases (Text, LiteralString, Label = str), an aliased Enum import, functional Enum(...) and module-level str constants or label tuples. Reviewers check these forms by eye. Only a slice that lists tests/test_architecture.py in its Files can add them.

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.

| Item | Value |
|---|---|
| Final commit | none yet |
| Release tag | none yet |
| Decisions locked | none yet |
| Independent validation | none yet |

Delivered:
- none yet

Deferred:
- none yet

Open work: all slices
