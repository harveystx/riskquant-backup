# WS_FACTOR_REGRESSION_COINTEGRATION: Unit-root test of a declared linear combination

Path: docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION.md
Class: HEAVY
Status: CLOSED
Revision: 4
Revision note: Revision 4 (2026-10-19). Owner decision under WS_TEMPLATE.md 1.8: Revision 4 without merged code is approved, and the decision of 2026-10-19 (build v1, no round-2 audit) stays in force. At the owner's request Revision 4 folds the audit round 1 refinements R-1 to R-11 into the contract and the slices (WS_TEMPLATE.md 1.6 would have put them in the backlog), and corrects the value case (R-12). The new contract text is not independently audited. B-1 to B-5 were applied in Revision 3. Owner 2026-10-20: S1 to S4 VALIDATED; P-FACTOR_REGRESSION_COINTEGRATION-001 promoted to D105.
Date: 2026-10-19
Branch: ws/factor-regression-cointegration
Snapshot: SNAPSHOT commit=b7b088a dirty=false content_hash=72b2cb891c2269d9
Drafted by: investigator
Independent review: slice reviews S1 to S4 in a separate reviewer chat (S4 in three rounds; every slice finding is closed or in the section 7 backlog; none is open); freeze audit round 1 in a separate chat (R_FREEZE_RECONCILIATION_R1.md): NOT READY. B-1 to B-5 applied in Revision 3. R-1 to R-11 applied in Revision 4 without review. No round 2 (owner decision)
Audit rounds used: 1 of 2
Code merged: no
DECISIONS.md reconciled: YES (D105 added 2026-10-20)
Evidence limits: The owner Windows runs decide (gate_s1.txt to gate_s4_r2.txt, kept untracked; last full run 7117 passed, 36 skipped). The reviewer and
implementer sandboxes lacked the sources of other packages, so their full runs compared the failing test IDs with and without the slice files. No gate ran on
Python 3.11, the mypy target of `pyproject.toml` (R-15), and none ran on the numpy 1.26 floor. statsmodels 0.15.0 was installed in sandboxes only, as a cross-check (O-3).
The 23 N = 1 constants equal the PDF text layer at https://ageconsearch.umn.edu/record/273723/files/qed_wp_1227.pdf and `statsmodels.tsa.adfvalues.tau_2010s`
(24 cells with the empty b3 cell, 0 differences). The implementer compared all 24 cells of the six rows with page images in S2; no other check compared the page images
(the reviewer compared the text layer). The N >= 2 cells of the MacKinnon tables are not transcribed in v1; two of them differ between the PDF text and statsmodels (A.2 note).
The power and size numbers of A.5 come from scripts, not from library code.

## 1. Goal and scope

Goal: A researcher tests the null hypothesis of a unit root in a declared linear combination of aligned level or
log-level series, and the result binds to its exact sample and specification.

Value case:
- Benefit: The owner can state, with a tabulated reference distribution, whether the known log spread
  `ln P - ln Z - ln M` of the THG proxy hypothesis rejects a unit root (F-8), and whether each log series
  has a unit root. Today no unit-root code exists in `src/` (F-1). The landed levels path gives invalid inference (F-10).
- Cost proxies: 4 slices; 3 new core modules; 9 modified files (core `codes.py`, `errors.py`; facade `contracts.py`,
  `labels.py`, `digest.py`, `errors.py`, `api.py`; tests `test_api.py`, `test_errors.py`); 1 new test module per slice; 1 new decision (P-FACTOR_REGRESSION_COINTEGRATION-001);
  23 transcribed constants; at most 2 audit rounds.
- Smallest alternative and why it fails: An external `statsmodels` script on the output of `align_levels`. It can
  compute the statistic today, and `adfuller(autolag=None)` reproduces A.1 (F-17). It fails on one ground only: no sample
  binding (the digest functions are private, F-6; D001, D104.2), together with the owner decision O-3 (no `statsmodels`
  dependency). It does not fail on the lag default, because `autolag=None` gives the declared lag. The 2 disputed constant
  cells of `statsmodels` are N = 2 and N = 3 cells, which v1 does not use (F-7). (R_FREEZE_RECONCILIATION_R1.md, R-12)
- Case against: The motivating case is one hypothesis (THG). A one-off script with a verified constant table answers it.
  The library pays for a public surface, a new decision and 4 slices before any second user exists (owner input).
- Class: HEAVY. Criteria: new public signature in `riskquant.solutions.factor_regression.api`; new D decision.
- Recommendation at drafting: PROCEED SMALLER (ADF of a declared combination only; no Engle-Granger, Johansen,
  ECM, DOLS or p-value in v1) [Proposed]

In scope (v1):
- Core: linear combination of aligned levels on scale LEVEL or LOG; ADF regression with CONSTANT or CONSTANT_TREND
  and a declared lag; MacKinnon (2010) N = 1 critical values at 1 %, 5 % and 10 %
- Facade: `CombinationTerm`, `LinearCombination`, `UnitRootSpec`, `UnitRootResult`, digests, labels, the public function
  `unit_root_test(levels, spec)`

Out of scope:
- Engle-Granger with free coefficients (O-4), Phillips-Ouliaris, Johansen, error-correction model, DOLS or FM-OLS
- p-values (O-5); automatic lag selection (O-7); KPSS; break-robust tests
- Any change to the behaviour, signature or digest of `align_levels`, `fit_ols`, `wald_test`
- `statsmodels` in `pyproject.toml` (O-3)

Owner decisions in force:
- 2026-10-12 (WS_FACTOR_REGRESSION): O-2 own numpy and scipy code, no new dependency; O-7 caller declares `rank_rtol`.
- O-5 of WS_FACTOR_REGRESSION excluded stationarity tests from v1 of that workstream only. This workstream reopens the scope.

Depends on: D104; `riskquant.factor_regression.fit.fit_core`; `riskquant.solutions.run_identity.hashing.digest_named_arrays`.
Blocks: none.

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | No source in the attached profiles contains adfuller, dickey, unit root, cointegrat, kpss, johansen, mackinnon or statsmodels. | `grep -i -E` over `/mnt/project/*.txt` (all 29 attached .txt files): 0 files | Observed |
| F-2 | `fit_core(design, y, *, intercept, rank_rtol, covariance_kind, covariance_lag)` takes any `(R, k)` design with the ones column first when `intercept` is True, and solves on the scaled SVD. | `src/riskquant/factor_regression/fit.py:fit_core` | Observed |
| F-3 | The transform codes are LEVEL, LOG_RETURN, SIMPLE_RETURN, DIFFERENCE. No log-level code exists. | `src/riskquant/factor_regression/codes.py:TRANSFORM_CODES` | Observed |
| F-4 | D104.1 to D104.6 do not forbid a new function. D104.6 covers return transforms only. The D104 Implementation note lists the public functions `align_levels`, `fit_ols`, `wald_test`. | `DECISIONS.md` D104 | Observed |
| F-5 | D104.2: no module outside the two packages and tests imports the core. | `DECISIONS.md` D104.2 | Observed |
| F-6 | The digest functions live in `digest.py`; `api.py` is the only public surface. | `src/riskquant/solutions/factor_regression/digest.py`; D104.2 | Observed |
| F-7 | `statsmodels` 0.15.0 `adfvalues.tau_2010s` differs from the PDF text in 2 cells: tau_c N=2 1 % beta2 (PDF -22.527, statsmodels -33.527) and tau_c N=3 5 % beta1 (PDF -8.5631, statsmodels -8.5632). The N=1 rows agree. | sandbox print of `tau_2010s`; PDF Table 2 | Observed |
| F-8 | The known log spread test is an ordinary DF test with the N=1 tables, because no coefficient is estimated. | MacKinnon (2010) section 2, regressions (6) and (7) | Observed |
| F-9 | Under a unit root with MA(1) errors, theta -0.5, T=500, the 5 % tau_c test rejects 0.629 with lag 0, 0.224 with lag 1, 0.056 with lag 4. | A.5 sandbox | Observed |
| F-10 | In a cointegrated levels regression, the HAC t-test of the true slope rejects 0.109 to 0.466 at a nominal 5 %. | A.5 sandbox | Observed |
| F-11 | The tau_c table assumes no drift under the null (West 1988). The tables are strictly valid for IID normal errors at finite T. T is the number of observations in the test regression. | MacKinnon (2010) sections 2, 3 and appendix | Observed |
| F-12 | MacKinnon (2010) does not recommend tau_nc, because it requires highly unrealistic assumptions. The smallest simulated T is 20. | MacKinnon (2010) appendix | Observed |
| F-13 | `pyproject.toml` declares no `statsmodels`. The `dev` extra holds ruff, pytest, pytest-cov, mypy, hypothesis. | `pyproject.toml` | Observed |
| F-14 | `tests/solutions/factor_regression/test_digest.py` (S6.T10) takes every name in `dir(codes)` that ends in `_CODES`, requires a labels group for each, and a second test requires `set(labels.ALL_CODE_GROUPS)` to equal that set. | `test_t10_all_codes_labeled`, `_codes_tuple_names` | Observed |
| F-15 | `tests/solutions/factor_regression/test_api.py` (S7.T7) asserts `set(api.__all__) == _EXPECTED_PUBLIC`, and that every function and class of `api.py` is in that set or is private. | `test_t7_all_lists_the_public_surface_and_resolves`, `test_t7_helpers_are_private_and_no_other_module_is_public` | Observed |
| F-16 | `codes.py` holds 50 `np.int8` constants in 11 `*_CODES` groups. `LEVEL` is a transform code. `tests/factor_regression/test_errors.py` checks `int8` and uniqueness for a fixed list of those 11 groups, so a new group is not covered by it. The same file fixes the set of core exception classes: `test_the_module_defines_exactly_the_core_exception_set` asserts `set(_defined_classes()) == _EXPECTED_CLASSES` and `set(_FACTORIES) == _EXPECTED_CLASSES` (8 classes). | `codes.py`; `test_every_code_group_is_int8_and_unique`; `test_the_module_defines_exactly_the_core_exception_set` | Observed |
| F-17 | `adfuller(autolag=None)` gives the A.1 statistic (relative difference 2e-14), the same `n` and the same critical values at 1 %, 5 % and 10 % as A.1 and A.2, for (c, 0), (c, 4), (ct, 4) and (ct, 12) on 600 points. | pre-audit sandbox script | Observed |
| F-18 | `codes.SAMPLE_SIZE_ERROR_CODES` holds `SAMPLE_STEP_NOT_BELOW_COUNT` (0) and `SAMPLE_ROWS_NOT_ABOVE_COLUMNS` (1). `api.py` maps a core sample-size reason with `labels.SAMPLE_SIZE_ERROR_NAMES[reason]`, a dict lookup that raises `KeyError` for an unlabelled reason. | `codes.py`; `labels.py`; `api.py:_map_core_error` | Observed |
| F-19 | `scripts/run_all_scripts_factor_regression.py` finds files with `rglob("*.py")`. The tests runners run pytest on the `tests` directory. None holds a fixed module or test list. | pre-audit check of the scripts dump | Observed |

## 3. Contract

All clauses describe new code. Each is Proposed. Formulas and constants are in `A_NUMERICS.md`.

- C.1 `CombinationTerm(series, weight)` and `LinearCombination(terms, scale)` have no default; an omitted field raises `TypeError`.
  `series`: a value that is not a `str` raises `TypeError`; an empty `str`, or one with a character of Unicode category `Cc`, raises
  `ModelSpecError`. `weight`: `bool`, `np.bool_` and every type other than `int`, `float`, `np.integer` and `np.floating` raise `TypeError`
  (as `contracts.py:_check_float_param`); an accepted value is stored as `float`; `0.0`, a non-finite value (`nan`, `inf`, `-inf`) and an `int`
  that `float()` cannot represent (for example `10**400`) raise `ModelSpecError`. `terms`: a value that is not a `tuple` of `CombinationTerm` raises `TypeError`; an empty tuple or a repeated series name
  raises `ModelSpecError`. `scale`: a value that is not a `Scale` member raises `TypeError`. [Proposed] Test: S4.T1
- C.2 The combination value is `z_t = sum_i w_i * g(v_i[t])` on the aligned date axis, with `g` the identity (LEVEL) or the
  natural logarithm (LOG). It uses `AlignedLevels` only, after alignment. An unknown series raises `ModelSpecError`.
  [Proposed] Test: S1.T1, S1.T2, S4.T1
- C.3 LOG domain: every aligned level of every term is above 0, checked before any logarithm. The scan is term-major, as in
  `rows.py:_validate_domains`: every aligned level of term 0 in ascending date order, then term 1, and so on. The first offending pair
  `(term, aligned_index)` raises `CoreScaleDomainError`. No logarithm of a non-positive value is computed. The facade maps the error to
  `ScaleDomainError` naming series, date and level (C.10). The rule follows the D025 convention. [Proposed] Test: S1.T3, S4.T5
- C.4 `UnitRootSpec(combination, deterministic, lag, rank_rtol)` has no default; an omitted field raises `TypeError`. `deterministic`: a value
  that is not a `Deterministic` member raises `TypeError`. `lag`: `bool`, `np.bool_` and every type without an integer index (a `float`, a
  `str`, `None`) raise `TypeError` (as `contracts.py:_check_int_param`); `lag < 0` and `lag > 2**63 - 1` (`_INT64_MAX`) raise `ModelSpecError`, as
  `CovarianceSpec` does. `rank_rtol`: as `ModelSpec.rank_rtol`: `bool` and a non-numeric type raise `TypeError`; a value outside (0, 1), `nan`
  included, raises `ModelSpecError`. [Proposed] Test: S4.T2
- C.5 The ADF regression follows A.1. The core builds the design and the response and calls `fit.fit_core` with IID
  covariance. The statistic is `tau = coef[j] / se[j]` for the column `z_{t-1}`. The library adds no second solver.
  [Proposed] Test: S3.T1, S3.T2, S3.T3
- C.6 The row count `n = m - 1 - p` must be at least 20 and at least `k + 1`. After its type and dtype checks, `adf.py` checks `n < 20` first,
  before it builds the design and before any call of `fit.fit_core`, and raises `CoreSampleSizeError(reason=codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM,
  n_available=n, n_required=20)` (A.8). `n` is zero or negative when `p >= m - 1`. When both `n < 20` and `n < k + 1` hold, this reason is raised.
  `rank_gate` inside `fit_core` raises `SAMPLE_ROWS_NOT_ABOVE_COLUMNS` only when `n >= 20` and `n < k + 1`. The facade maps both reasons to
  `InsufficientObservationsError` naming `n` and the requirement (C.10). [Proposed] Test: S3.T4, S4.T6
- C.7 Critical values follow A.2: `c(n) = b_inf + b1/n + b2/n^2 + b3/n^3` with the MacKinnon (2010) N = 1 rows, at
  levels 0.01, 0.05, 0.10, with `n` from C.6. The constants exist once, in `mackinnon.py`. [Proposed]
  Test: S2.T1, S2.T2, S2.T3, S2.T4, S2.T5
- C.8 `UnitRootResult` carries: `statistic`, `gamma`, `critical_levels` `(0.01, 0.05, 0.10)`, `critical_values` `(3,)`,
  `deterministic`, `lag`, `n_obs`, the ADF regression `coef`, `se` and column labels, `condition_number`,
  `null_hypothesis` (fixed text per deterministic case, A.4), `critical_value_source` (fixed text per deterministic case, A.2; it names N=1
  and declared weights), combination terms, weights, scale, units, window, row start and end dates, exclusions, `rank_rtol`, three digests
  and `ENGINE_VERSION`. `window` is `(levels.start_ord, levels.end_ord)`. `exclusions` is `levels.exclusions`, the alignment exclusions only,
  as `OlsResult.exclusions`. The first aligned dates `d_0 .. d_p`, which the lag lookback uses, are not added to `exclusions`: `levels.date_ord`
  with `row_start_date_ord` and `row_end_date_ord` define them. All arrays are read-only. It has no p-value and no field named rejected,
  accepted, stationary or valid (D104.4). [Proposed] Test: S4.T3, S4.T4
- C.9 `unit_root_spec_digest`, `unit_root_sample_digest` and `row_digest` follow A.3, use `digest_named_arrays`, are
  stable across fresh processes and change under the mutations of S4.T8. The bytes are fixed by A.3 and A.4. [Proposed] Test: S4.T7, S4.T8, S4.T13
- C.10 `unit_root_test` maps these core exceptions, each chained as `__cause__` and naming series, columns and dates where the core gives
  indices: `CoreScaleDomainError` to `ScaleDomainError`; `CoreSampleSizeError` to `InsufficientObservationsError`; `CoreRankError` to
  `RankDeficiencyError`; `CoreDegenerateFitError` to `DegenerateFitError`; `CoreParameterError` to `ModelSpecError`; `CoreInputError` with the
  reason `INPUT_NON_FINITE` (an overflow of the combination or of a fit quantity) to `DegenerateFitError`. Any other `CoreInputError`, and any
  `TypeError` or `ValueError` from the core, is an internal contract violation and propagates unchanged, as in `fit_ols`
  (`api.py:_map_core_error` returns `None`). The result never holds NaN. [Proposed] Test: S4.T5, S4.T6, S4.T14
- C.11 New core modules hold no `str`-typed field, parameter or return and no Enum. They import only the standard
  library, numpy, `riskquant.exceptions` and their own package. The facade adds no import outside the D104 allow-list.
  [Proposed] Test: S1.T5, S4.T9
- C.12 `align_levels`, `fit_ols` and `wald_test` keep their signatures, results and digests. `ENGINE_VERSION` is
  unchanged (O-8). [Proposed] Test: S4.T10
- C.13 No library function mutates a caller array. Retained arrays are read-only copies. [Proposed] Test: S1.T4, S4.T4
- C.14 New code constants and labels. Each new `*_CODES` tuple of `codes.py` holds `np.int8` constants with unique values and has a labels group in
  the same slice that adds the tuple (F-14). A code appended to an existing tuple gets its label entries in the same slice. Names and fixed
  values follow A.8. No new constant has the name of an existing constant (F-16).
  The values of the scale, deterministic and test-kind codes enter the digests (A.3). [Proposed] Test: S1.T6, S1.T7
- C.15 Public names. `api.__all__` and `_EXPECTED_PUBLIC` in `tests/solutions/factor_regression/test_api.py` hold the same set. The set adds
  `unit_root_test`, `CombinationTerm`, `LinearCombination`, `UnitRootSpec`, `UnitRootResult`, `Scale`, `Deterministic` and `ScaleDomainError`.
  No other name changes (F-15). Each `Scale` and `Deterministic` member value equals its core code (A.8). [Proposed] Test: S4.T11, S4.T12

## 4. Decisions

Decisions relied on: D001, D002, D004, D025, D104. Each checked against `DECISIONS.md`: yes.

### P-FACTOR_REGRESSION_COINTEGRATION-001: Unit-root test in the factor regression library (PROMOTED to D105 on 2026-10-20, clauses 1 to 6 unchanged)

```text
Related: D001, D002, D025, D104
Scope: Adds a unit-root test of a declared linear combination to riskquant.factor_regression.
1. The test lives in the D104 packages. D104.1 to D104.5 apply unchanged.
2. The result reports the statistic and MacKinnon (2010) critical values. It reports no p-value and no verdict.
3. Critical values come only from MacKinnon (2010), Tables 2 and 3, transcribed once and named in the result.
4. A LOG-scale combination needs every level above 0 (D025 convention, as D104.6).
5. The caller declares the deterministic case and the lag. The library has no default and no automatic selection.
6. statsmodels is neither a runtime dependency nor a test dependency.
Evidence: S1.T3, S1.T5, S2.T1, S2.T2, S4.T2, S4.T3, S4.T9
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Placement: A, extend `riskquant.factor_regression`. | CLOSED | 2026-10-19 | owner |
| O-2 | Decision form: a new decision (D105) from P-FACTOR_REGRESSION_COINTEGRATION-001. D104 gets one line "Amended by D105" in its Implementation note, as D101 did for D062. | CLOSED | 2026-10-19 | owner |
| O-3 | statsmodels: not in the repository, neither runtime nor test dependency. A reviewer sandbox may use it as a cross-check. | CLOSED | 2026-10-19 | owner |
| O-4 | Engle-Granger with free coefficients: backlog. The F-7 cells must be resolved first. | CLOSED | 2026-10-19 | owner |
| O-5 | p-values: none in v1. A p-value needs the MacKinnon (1994, 1996) coefficients from a verified source. | CLOSED | 2026-10-19 | owner |
| O-6 | Deterministic cases: CONSTANT and CONSTANT_TREND only (F-12). | CLOSED | 2026-10-19 | owner |
| O-7 | Lag: declared, no default, no automatic selection. | CLOSED | 2026-10-19 | owner |
| O-8 | `ENGINE_VERSION` unchanged; the new spec digest holds a test-kind field. | CLOSED | 2026-10-19 | owner |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1 and S2 in parallel; S3 after S1 and S2; S4 after S3.
Every Attach list also holds `CLAUDE.md`, `DECISIONS.md` and `WORKFLOW.md`.
Gate sequence. This workstream defines its own 7 gates. They extend WORKFLOW.md section 6, which lists 4: (1) py_compile, (2) import of the
slice modules, (3) ruff check, (4) ruff format --check, (5) mypy --strict, (6) pytest slice, (7) pytest full. Slice S4 adds the line `mypy baseline` (`mypy --strict src/riskquant/solutions/run_identity/hashing.py`)
before `mypy --strict`. That line prints 13 errors in `hashing.py` and `exit=1` by design. The S4 `mypy --strict` line passes when every
`: error:` line that it prints also appears in the `mypy baseline` output (I-001 rule of WS_FACTOR_REGRESSION, stated here in full).

### S1: Core combination and LOG domain

Objective: Build `z_t` from aligned levels with declared weights and scale, and reject LOG levels at or below 0.
Contract: C.2, C.3, C.11, C.13, C.14
Depends on: O-1, O-2, O-6
Files:
- create: `src/riskquant/factor_regression/combination.py`, `tests/factor_regression/test_combination.py`
- modify: `src/riskquant/factor_regression/codes.py` (additive: scale codes, deterministic codes, test-kind code, the scale-domain reason group and the
  one appended `SAMPLE_ROWS_BELOW_TABLE_MINIMUM`, named by A.8);
  `src/riskquant/factor_regression/errors.py` (additive: one class, `CoreScaleDomainError`, A.8; one docstring line of `CoreSampleSizeError`);
  `src/riskquant/solutions/factor_regression/labels.py` (additive: for every new `*_CODES` tuple the bijective maps (`*_NAMES` and
  `*_NAME_TO_CODE`) and the `ALL_CODE_GROUPS` entry, and the entries of the appended `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` in the `SAMPLE_SIZE_ERROR`
  maps; names only; required by the existing S6.T10 tests, F-14);
  `tests/factor_regression/test_errors.py` (additive only: the name `CoreScaleDomainError` in `_EXPECTED_CLASSES` and one factory in `_FACTORIES`, F-16)
Prohibited: change of any existing code value, tuple member order (one append at the end of `SAMPLE_SIZE_ERROR_CODES` is allowed) or exception
signature; `str` or Enum in the core; a new constant with the name of an existing constant (F-16); a change to any existing test file except the
one additive edit of `test_errors.py` named in Files
Attach: `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION.md`, `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION/A_NUMERICS.md`, `context_riskquant_factor_regression.txt`,
`context_riskquant_solutions_factor_regression.txt`, `context_riskquant_tests.txt`
Gates: section 6 order; pytest slice `tests/factor_regression/test_combination.py`, `tests/factor_regression/test_errors.py`,
`tests/solutions/factor_regression/test_digest.py` and `tests/test_architecture.py`; pytest full.
Exit: Every gate line prints `exit=0`. pytest full: 0 failed, 0 errors.
Status: VALIDATED
Evidence: Windows gates exit=0 on all 7 lines (gate_s1.txt); 267 passed in the slice; pytest full 6881 passed and 36 skipped (baseline 6790 + 91 new tests: 84 in
test_combination.py, 3 in test_errors.py, 4 in test_digest.py); independent review ACCEPT with no finding; reviewer mutation run 22 of 22 killed; the four
edits to existing files are additive (0 removed lines); backlog lines in section 7.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | LEVEL, weights (1, -2.5), Fraction oracle | equal to exact value | wrong weight or order |
| S1.T2 | LOG, `Q = Z*M`, weights (1, -1, -1) on (Q, Z, M) | max abs z below 1e-14 | product instead of log sum |
| S1.T3 | LOG, two terms. Term 0 holds 0.0 at index 3 and -1.0 at index 1. Term 1 holds -2.0 at index 0. | `CoreScaleDomainError` with term 0, aligned_index 1 and level -1.0 (term-major, ascending date: not term 1, not index 3); no logarithm of a non-positive value | wrong scan order or log of non-positive level |
| S1.T4 | input arrays flagged read-only, then compared after the call | unchanged | caller mutation |
| S1.T5 | existing `tests/test_architecture.py:test_core_packages_obey_the_mechanical_rules` scans the new module | pass | str or forbidden import in core |
| S1.T6 | existing S6.T10 tests of `tests/solutions/factor_regression/test_digest.py`, run after S1 | pass | new `*_CODES` tuple without a labels group (F-14) |
| S1.T7 | in `test_combination.py`: each new `*_CODES` tuple holds `np.int8` constants with unique values; no name clash with the 50 existing constants; the values of A.8, including `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` = 2 as the last member of `SAMPLE_SIZE_ERROR_CODES` | exact | clash, duplicate value, wrong digest value |
| S1.T8 | existing tests of `tests/factor_regression/test_errors.py` after the additive edit (exception set, `RiskQuantError` subclass, no `str` field or parameter, message tests) | pass for `CoreScaleDomainError` | unlisted or `str`-bearing core exception (F-16) |

### S2: Core MacKinnon critical values

Objective: Transcribe the N = 1 tau_c and tau_ct rows of A.2 and evaluate `c(n)`.
Contract: C.7
Depends on: O-5, O-6
Files:
- create: `src/riskquant/factor_regression/mackinnon.py`, `tests/factor_regression/test_mackinnon.py`
- modify: none
Prohibited: constants from memory or from statsmodels; scipy in this module
Attach: `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION.md`, `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION/A_NUMERICS.md`, the source PDF (A.2 URL)
Gates: section 6 order; pytest slice `tests/factor_regression/test_mackinnon.py` and `tests/test_architecture.py`; pytest full.
Exit: Every gate line prints `exit=0`. The report lists each of the 23 N = 1 constants with its PDF page.
Status: VALIDATED
Evidence: Windows gates exit=0 on all 7 lines (gate_s2.txt); 111 passed in the slice; pytest full 6930 passed and 36 skipped (6881 + 49 new tests); independent review
ACCEPT with no finding. All 24 cells of the six rows: the implementer reports a comparison with the page images of the PDF (printed pages 13 and 14), 0 differences; the
reviewer compared the PDF text layer and statsmodels 0.15.0, 0 differences. Check by the owner's assistant: `critical_values` against `statsmodels.tsa.adfvalues.mackinnoncrit`
over 834 (table, n) cases, worst difference 8.9e-16. Reviewer mutants killed by S2.T5 (b2 and b3 swapped, one digit changed, tables swapped). Backlog lines in section 7.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | each constant against the A.2 table typed in the test file | exact float equality | transcription error |
| S2.T2 | seeded Monte Carlo, 200000 random walks, n = 100, lag 0, closed-form DF regression in the test | quantile within 0.05 (1 %) and 0.03 (5 %, 10 %) of `c(100)` | wrong case or wrong row |
| S2.T3 | n = 1e9 | `c(n)` equal to `b_inf` within 1e-8 | wrong asymptote |
| S2.T4 | n from 20 to 1e6 | `c_1% < c_5% < c_10% < 0` | level order swapped |
| S2.T5 | `c(n)` for the six rows at n in {20, 25, 100, 1000}, against an exact `fractions.Fraction` evaluation of the constants typed in the test | relative error at most 1e-15 | wrong or swapped power of 1/n |

### S3: Core ADF statistic

Objective: Build the ADF design and response of A.1, call `fit_core` with IID covariance, return tau and `c(n)`.
Contract: C.5, C.6, C.7, C.13
Depends on: S1, S2
Files:
- create: `src/riskquant/factor_regression/adf.py`, `tests/factor_regression/test_adf.py`
- modify: none
Prohibited: `lstsq`; a second OLS solver; HAC or HC0 covariance in the ADF regression; p-values of `fit_core` in the core result
Attach: `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION.md`, `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION/A_NUMERICS.md`, `context_riskquant_factor_regression.txt`
Gates: section 6 order; pytest slice `tests/factor_regression/test_adf.py` and `tests/test_architecture.py`; pytest full.
Exit: Every gate line prints `exit=0`.
Status: VALIDATED
Evidence: Windows gates exit=0 on all 7 lines (gate_s3.txt); 146 passed in the slice; pytest full 7014 passed and 36 skipped (6930 + 84 new tests); independent review
ACCEPT with two P2 findings in the backlog (R-001, R-002). Reviewer checks: A.1 rebuilt row by row (design, response, row indices, n, k, gamma column) bit-exact in 4 cases;
`adfuller(autolag=None)` on 40 cases: tau 1.3e-13 relative, critical values 4.4e-16 absolute, `n` equal; mutation run 42 of 43 non-equivalent mutants killed (the survivor
is a one-configuration mutant, recorded as no finding). Backlog lines in section 7.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | lag 0, CONSTANT, 25 rows, Fraction normal equations in the test | tau within 1e-12 relative | wrong row offset or dof |
| S3.T2 | lag 4, CONSTANT_TREND, 600 rows, float normal equations in the test | tau within 1e-10 relative | wrong lag columns |
| S3.T3 | source scan of `adf.py` | no `numpy.linalg`, one call of `fit_core` | second solver |
| S3.T4 | n = 19 (lag 0); n = 20 with k = 20 (lag 18, 39 aligned dates); n = 9 with k = 32 (lag 30, 40 aligned dates, both rules fail); n = -3 (lag 5, 3 aligned dates) | `CoreSampleSizeError`: `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` with n_available 19 and n_required 20; `SAMPLE_ROWS_NOT_ABOVE_COLUMNS` from `rank_gate`; `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` with n_available 9 (C.6 order); `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` with n_available -3 | missing size rule, wrong check order or a failure on n at or below 0 |

### S4: Facade contracts, digests and entry point

Objective: Add the contract types, labels, errors, digests and `unit_root_test` to the facade.
Contract: C.1, C.3, C.4, C.8, C.9, C.10, C.11, C.12, C.13, C.15
Depends on: S3, O-3, O-8
Files:
- create: `tests/solutions/factor_regression/test_unit_root.py`
- modify: `src/riskquant/solutions/factor_regression/contracts.py`, `labels.py`, `errors.py`, `digest.py`, `api.py`;
  `tests/solutions/factor_regression/test_api.py` (the set `_EXPECTED_PUBLIC` only, C.15)
`labels.py` receives only the fixed text constants of `null_hypothesis` and `critical_value_source` (A.2, A.4). It changes no code-to-name map and no
`ALL_CODE_GROUPS` entry that S1 added.
Prohibited: numerical logic in the facade; a second public module; Pydantic; any change to existing digests; a change to `test_api.py` other than
`_EXPECTED_PUBLIC`
Attach: `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION.md`, `docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION/A_NUMERICS.md`, `context_riskquant_factor_regression.txt`, `context_riskquant_solutions_run_identity.txt`,
`context_riskquant_solutions_factor_regression.txt`, `context_riskquant_tests.txt`
Gates: section 6 order, with the line `mypy baseline` before `mypy --strict`; pytest slice `tests/solutions/factor_regression/` and `tests/test_architecture.py`; pytest full.
Exit: Every gate line prints `exit=0`, except the `mypy baseline` line (`exit=1` by design, 13 errors in `hashing.py`). The `mypy --strict` line passes
when every `: error:` line that it prints also appears in the `mypy baseline` output.
Status: VALIDATED
Evidence: Windows gates exit=0 on every gate line except `mypy baseline` (exit=1 by design, 13 errors in `hashing.py`; the strict line adds no error) (gate_s4_r2.txt); slice 471 passed;
pytest full 7117 passed and 36 skipped (7014 + 103 new tests in test_unit_root.py). Independent review in three rounds: round 1 ACCEPT WITH FIXES (R-001 to R-006), round 2 BLOCKED on a
missing attachment, round 3 ACCEPT; R-001 to R-005 closed. Reviewer checks: `unit_root_test` against `adfuller(autolag=None)` on 18 cases (statistic 2.8e-13 relative, `n` equal, critical
values 4.4e-16); an end-to-end hashlib oracle for the three digests in 4 cases; the S4.T8 matrix; 30 facade mutants in round 1 (11 survived and led to R-002 and R-003) and 15 required
mutants in round 3, all killed. The first delivery had known defects and was committed as a rollback point; two fix rounds followed (F-1 to F-13; the bound of the ADF label loop). The owner
checked the F-16 diff against the pre-S4 text with git, and an independent dump comparison gave the same line counts. Backlog lines in section 7.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | `TypeError`: weight `True`, `np.bool_(True)`, `"1"`, `None`, `Fraction(1)`; series `1`; `terms` that is a list; `scale` that is a string. `ModelSpecError`: weight `0.0`, `nan`, `inf`, `-inf`, `10**400`; series `""`; series with `\x85`; empty `terms`; repeated series; unknown series. Weight `2` (an `int`) is accepted and stored as `2.0`. | as listed | missing spec check or wrong exception type |
| S4.T2 | `TypeError`: an omitted field; lag `True`, `np.bool_(True)`, `1.5`, `"2"`; `deterministic` `"NONE"`. `ModelSpecError`: lag -1; lag `2**63`; rank_rtol 1.0; rank_rtol `nan`. Lag `2**63 - 1` is accepted. | as listed | defaults, open set or wrong exception type |
| S4.T3 | field names of `UnitRootResult` | no p-value, no verdict field | approval leak (D104.4) |
| S4.T4 | arrays of the result | read-only; inputs unchanged | mutation |
| S4.T5 | LOG level 0.0 on a named date; `issubclass(ScaleDomainError, RiskQuantError)` | `ScaleDomainError` names series and date; `__cause__` is the core error; the subclass check holds | lost context or class outside the exception tree |
| S4.T6 | 21 aligned dates with lag 2 | `InsufficientObservationsError` names n = 18 and 20 and the reason `SAMPLE_ROWS_BELOW_TABLE_MINIMUM` | wrong n or unlabelled reason (F-18) |
| S4.T7 | digests in two fresh processes | equal | non-determinism |
| S4.T8 | change one weight, the scale, the lag, the case, one level, one date | the matching digests change | digest omits a field |
| S4.T9 | imports of the facade modules; text of `pyproject.toml` | allow-list of D104 only; `pyproject.toml` holds no `statsmodels` | new dependency |
| S4.T10 | the existing golden digest tests of `test_digest.py` (S6.T9) and the existing S7 tests of `test_api.py`, run after S4 with no change to any test except the `_EXPECTED_PUBLIC` set of C.15 | pass | collateral change to an existing digest or result |
| S4.T11 | existing S7.T7 tests of `test_api.py` after the change of `_EXPECTED_PUBLIC` | pass; the set equals `api.__all__` | unlisted public name or missing name (F-15) |
| S4.T12 | each `Scale` and `Deterministic` member value against the core code of A.8 | equal | Enum out of step with the core |
| S4.T13 | `unit_root_spec_digest` and `unit_root_sample_digest` rebuilt with `hashlib` and numpy only, from the text of A.3 and A.4, as `test_t9_oracle_spec_digest_no_threshold` does for the fit digests | equal | digest bytes differ from the contract |
| S4.T14 | one case per mapping of C.10: a rank-deficient ADF design (an exactly collinear combination series); an exact fit; an overflow of the combination (levels near 1e308 with weights 10 and -10); a `CoreParameterError`; a core `CoreInputError` with another reason (patched in) | `RankDeficiencyError` naming the ADF column labels of A.3; `DegenerateFitError`; `DegenerateFitError`; `ModelSpecError`; the error propagates unchanged. Every mapped error has the core error as `__cause__`. | lost mapping, wrong class or swallowed internal error |

Appendix: A_NUMERICS.md (A.1 to A.7)

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| Low power at 250 rows: half-life 20 days gives power 0.26 (A.5) | the owner reads "no rejection" as "no relation" | A.6 text; report `n_obs`; use the longest window |
| Lag too short under MA errors (F-9) | false rejection, false proxy support | caller declares the lag; coefficients of all lags are reported |
| Drift in the spread under the null (F-11) | tau_c invalid | CONSTANT_TREND is available; A.6 text |
| Transcription error in a constant | wrong critical value | S2.T1 and S2.T2; reviewer re-reads the PDF images |
| A new `*_CODES` tuple or a new public name breaks an existing enumerating test (F-14, F-15) | pytest full fails from S1 or S4 | C.14, C.15; S1.T6; S4.T11; the audit searches `tests/` for other closed-set tests |

Deferred:
- Engle-Granger with free coefficients: opens on O-4. Needs the N = 2 and N = 3 rows and the EG dof convention (A.2 note).
- p-values: opens when a verified copy of the MacKinnon (1994) or (1996) coefficients exists.
- Johansen trace and maximum eigenvalue: opens when the owner needs more than one cointegrating vector.
- ECM, DOLS and FM-OLS: open when the owner needs inference on the vector or the reversion speed.

Backlog (REFINEMENT findings and items that v1 omits):
- LOG_LEVEL transform in `fit_ols` (needed by Engle-Granger on log levels; changes `rows.py`, R-009 applies)
- Half-life `ln(0.5) / ln(1 + gamma)` for `-1 < gamma < 0`; KPSS confirmation; break-robust tests (Gregory-Hansen)
- Lag rules (Schwert, Ng-Perron) as a facade helper that reports, never selects silently
- Audit round 1 REFINEMENT findings kept in the backlog (source: `R_FREEZE_RECONCILIATION_R1.md` section F). R-1 to R-11 are applied in the contract and the slices:
  - R-13 S2.T3: the margin is 9.5e-10 under the tolerance and no power defect changes the result. It stays an asymptote check only (its "Defect caught" text is changed by B-4).
  - R-14 A.8: the reason group of the scale exception shares the prefix `SCALE_` with `SCALE_CODES`. No test conflicts. The group is named `SCALE_DOMAIN_ERROR_CODES` in A.8.
  - R-15 Run the gates on Python 3.11 (the mypy target of `pyproject.toml`), not only on 3.12.
- S1 review, accepted points (source: slice review of S1):
  - `combine_core` documents that `levels` is C-order but does not check it. Any layout gives the same values, because each term reads one row view.
  - The module docstring of `codes.py` names only the old group kinds, not the new groups SCALE, DETERMINISTIC, TEST_KIND and SCALE_DOMAIN_ERROR.
  - The finiteness check of `combine_core` covers every row of `levels`. The LOG domain check covers only the rows that the terms read. The facade always passes finite levels.
  - `combine_core` checks dtype for all three operands before it checks ndim for any of them (owner decision 2 read as phases; a test fixes the order).
- S2 review, accepted points (source: slice review of S2):
  - `critical_values` raises `OverflowError` for an `int` above the float64 range (about 1.8e308), because it converts `n` with `float()`. `n` is a row count, far outside the valid range. The docstring documents it.
  - `test_mackinnon.py` reads the private name `_COEFFICIENTS` to compare it with independently typed literals.
  - The label rule at the top of `A_NUMERICS.md` still says that the constants are not compared with the page images. The S2 evidence records the comparison. Update the label rule at closure.
- S3 review, accepted points (source: slice review of S3):
  - R-001: the exact-fit fixture `z = 5 + 0.8**t` (60 points) in `test_adf.py` has `rss / bound` 0.833 with `fit_core`, and 0.972 and 0.918 with other solvers; the rule fires at 1.0 or less, and with 40 points it does not fire. Replace it by `z = 0.5**t` with 40 points (`rss / bound` 3.3e-4). S4.T14 uses the dyadic fixture.
  - R-002: `adf_core` builds `row_start_index` and `row_end_index` with `np.arange(..., dtype=np.int32)`, which wraps without an error when `m - 1 > 2**31 - 1`. The facade path is bounded by `INPUT_TOO_LONG` in `align.py`. A later LIGHT change adds the same check to `adf_core`.
  - A lag below the float64 range (for example `-10**400`) gives `OverflowError` from `float()` in `adf_core`, not `CoreParameterError`. `covariance.py` does the same. The facade rejects a negative lag first (C.4).
- S4 review, accepted points (source: slice review of S4):
  - R-006: `unit_root_spec_digest` has no int8 range check for `test_kind`, `scale` and `deterministic`. A value outside int8 raises `OverflowError`. Same class as the accepted scalar item of the S6 digests.
  - R-007: the "Input contract" paragraph of the `digest.py` module docstring omits the two unit-root functions and their A.3 dtypes.
  - `UnitRootSpec(rank_rtol=10**400)` raises `OverflowError`. `ModelSpec` does the same (C.4 parity).
  - The `CombinationTerm` message says "above the float64 range" also for `-10**400`.
  - No static check keeps numerical logic out of the facade. A mutant that computes `coef / se` in `api.py` survives the tests. Code reading finds no numerical operation in `unit_root_test`.
  - The child-process test `test_r001_huge_lag_raises_size_error_in_child` has no memory cap on Windows (the module `resource` is absent). Its 20 s timeout is the guard. Measure the import margin if the test flakes.

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.

| Item | Value |
|---|---|
| Final commit | 9a5bd68, head of ws/factor-regression-cointegration before the closure commit (S4 fix round 2) |
| Release tag | none (the owner tags the merge commit) |
| Decisions locked | D105, promoted from P-FACTOR_REGRESSION_COINTEGRATION-001 with clauses 1 to 6 unchanged. D104 carries an "Amended by D105" line. |
| Independent validation | Slice reviews S1 to S4 in a separate reviewer chat, each ACCEPT (S3 with two P2 findings; S4 ACCEPT WITH FIXES in round 1, BLOCKED in round 2 on a missing attachment, ACCEPT in round 3). Reviewer mutation runs: S1 22 of 22 killed; S2 all required mutants killed; S3 42 of 43 (the survivor is a one-configuration mutant); S4 round 3 15 of 15. Independent oracles: Fraction arithmetic (the A.2 constants, ADF lag 0), hashlib digests. Reviewer cross-checks with `adfuller(autolag=None)` in sandboxes (statsmodels 0.15.0, sandbox only): 40 core cases (statistic 1.3e-13 relative) and 18 facade cases (2.8e-13). The owner reports the last full pytest run: 7117 passed, 36 skipped (gate_s4_r2.txt, kept). |

Delivered:
- Core package `src/riskquant/factor_regression/`: new modules `combination`, `mackinnon` and `adf`; additive changes to `codes` and `errors` (scale, deterministic, test kind and scale-domain codes; `CoreScaleDomainError`).
- Facade `src/riskquant/solutions/factor_regression/`: additive changes to `contracts`, `errors`, `labels`, `digest` and `api`. New public function `unit_root_test`. Eight new public names: `unit_root_test`, `CombinationTerm`, `LinearCombination`, `UnitRootSpec`, `UnitRootResult`, `Scale`, `Deterministic`, `ScaleDomainError`.
- 327 new tests: S1 91 (`test_combination.py` 84, `test_errors.py` 3, `test_digest.py` 4), S2 49 (`test_mackinnon.py`), S3 84 (`test_adf.py`), S4 103 (`test_unit_root.py`). Full suite: 6790 passed and 36 skipped before; 7117 passed and 36 skipped after (owner run). Existing test files changed: `test_errors.py` and `test_digest.py` (additive, S1) and `test_api.py` (the set `_EXPECTED_PUBLIC`, 8 names, S4).
- Documents: this workstream document, `A_NUMERICS.md`, `R_FREEZE_RECONCILIATION_R1.md` and D105 in `DECISIONS.md` (with the "Amended by D105" line in D104).

Deferred:
- Engle-Granger with free coefficients (O-4): opens with the N = 2 and N = 3 rows and the EG dof convention.
- p-values (O-5): open when a verified copy of the MacKinnon (1994) or (1996) coefficients exists.
- Johansen, ECM, DOLS and FM-OLS: open when the owner needs them (section 7).
- Section 7 backlog: R-13 to R-15; the accepted items of S1 to S4.

Open work: merge of `ws/factor-regression-cointegration` into `prod` (owner). No slice and no review finding is open.
