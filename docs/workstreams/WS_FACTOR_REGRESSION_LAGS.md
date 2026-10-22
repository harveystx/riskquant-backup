# WS_FACTOR_REGRESSION_LAGS: Lagged and leading regressors in fit_ols

Path: docs/workstreams/WS_FACTOR_REGRESSION_LAGS.md
Class: LIGHT (owner decision 2026-10-22: HEAVY by WS_TEMPLATE.md 1.5, run on the LIGHT path)
Status: FROZEN
Revision: 1
Code merged: no
Date: 2026-10-22
Branch: ws/factor-regression-lags
Snapshot: commit=df8bb69 dirty=false content_hash=93e1f4e1e329dc19
Drafted by: owner
Evidence limits: Designed from the source of prod 828c402 (rows.py:build_rows, digest.py:spec_digest,
contracts.py:Term and ModelSpec). Nothing of the new behaviour was executed.

## 1. Goal and value

Goal: fit_ols accepts a signed lag per regressor term, on a declared common sample.
Benefit: loadings for non-synchronous closes at daily frequency (the Dimson estimator: lags 0 and 1, then
the sum of the coefficients), and tests of delayed price formation. The demo shows that a longer step h
removes the timing bias only like 1/h (population US loading 0.28 against 0.30 at h = 20). wald_test
already tests a sum or a block of lag coefficients.
Case against: Proxy V1 rows have no lag (D088), so the hand-off is the summed loading. A longer step h is
the existing alternative, with fewer independent intervals.
Reclassify trigger: stop and tell the owner if the work needs a new D decision or a change outside
section 4 Files.

## 2. Scope

In scope (v1):
- Term.lag (signed integer, default 0) and ModelSpec.max_lag (integer >= 0, default 0).
- core rows.py:build_rows: a lag per term and the common sample from max_lag.
- Column names, spec digest fields, docstrings, one new parameter reason code with its label.

Out of scope (see section 6):
- Lag selection; lags of the response; lags in unit_root_test, DOLS or ECM; rolling windows; the notebook.

Parity: none.

## 3. Owner decisions in force

- 2026-10-22: a lag counts aligned dates (rows of the common calendar), as the ADF lag does.
- 2026-10-22: a lag is a signed integer per term. Positive: the term reads an earlier interval (lag).
  Negative: a later interval (lead).
- 2026-10-22: one series may appear in several terms with different lags (distributed lag). An exact
  duplicate term (series, transform, lag) gives RankDeficiencyError at the fit, as today.
- 2026-10-22: no lag selection in the library. The caller declares every lag (as D105.5 for the ADF lag).
- 2026-10-22: a declared max_lag L fixes the common sample: every spec with the same L uses the same rows.
- 2026-10-22: a lag shifts a term's interval by `lag` aligned dates, independently of the step h.
- 2026-10-22: lag 0 everywhere and L = 0 change nothing: rows, results, column names and all digests stay
  as today. ENGINE_VERSION stays unchanged.
- 2026-10-22: process: one slice, one FULL code review, no spec audit.

## 4. Acceptance

- C.1 Term.lag is an int, default 0. A bool or a non-integer raises TypeError; a numpy integer is stored as
  int. [Proposed] Verified by: S1.T1
- C.2 ModelSpec.max_lag is an int >= 0, default 0, stored as int. ModelSpec raises ModelSpecError when
  max_lag < |lag| of a regressor, or when the response lag is not 0. [Proposed] Verified by: S1.T2
- C.3 Rows. With m aligned dates, step h, stride s and L = max_lag: row k (k = 0 .. R-1) ends at aligned
  index e_k = h + L + k*s, with R = floor((m - 1 - h - 2L) / s) + 1. The response reads [e_k - h, e_k]. A
  regressor with lag l reads [e_k - l - h, e_k - l]. The row start and end dates are those of the response.
  If m - 1 < h + 2L, fit_ols raises InsufficientObservationsError with SAMPLE_STEP_NOT_BELOW_COUNT,
  n_available m and n_required h + 2L + 1. [Proposed] Verified by: S1.T3
- C.4 The NOT_USED_BY_ROWS exclusions list exactly the aligned dates that no term reads. [Proposed]
  Verified by: S1.T3
- C.5 With every lag 0 and L = 0, rows, coefficients, covariance, diagnostics, column names and the spec,
  sample and row digests equal today's. [Proposed] Verified by: S1.T4 (the existing tests pass unchanged;
  spec_digest golden literals recorded before the change; rows equal an in-test reconstruction of the
  current formula bit for bit)
- C.6 The column name of a regressor with lag l != 0 is its current name followed by "[-l]" for l > 0 and
  by "[+|l|]" for l < 0. Lag-0 names do not change. [Proposed] Verified by: S1.T5
- C.7 spec_digest: a regressor line with lag l != 0 carries a fifth field "\x1f{l}" (signed decimal). When
  L > 0, a named array max_lag int64 [L] follows the existing arrays. A lag-0 line and L = 0 add nothing.
  [Proposed] Verified by: S1.T6 (hashlib oracle; specs that differ in one lag or in L give different digests)
- C.8 Oracle: two drivers, one at lags {0, 1}, one at lags {-1, 0}, L = 1, LOG_RETURN, h = 1: coefficients
  and IID standard errors equal a hand-built design solved in the test with numpy.linalg.lstsq, within
  1e-12 relative. [Proposed] Verified by: S1.T7
- C.9 core build_rows takes term_lags (int32, (n_terms,), response first with lag 0) and max_lag. A lag with
  |lag| > max_lag raises CoreParameterError with the new reason PARAM_LAG_OUTSIDE_WINDOW (appended last to
  its code group, with its label). [Proposed] Verified by: S1.T8
- C.10 wald_test on a lagged fit: a restriction with ones on the lag columns of one driver tests the sum of
  its lag coefficients; W equals a Fraction oracle within 1e-11 relative. [Proposed] Verified by: S1.T9

Files:
- modify: src/riskquant/factor_regression/rows.py; src/riskquant/factor_regression/codes.py;
  src/riskquant/solutions/factor_regression/contracts.py, labels.py, digest.py, api.py
- create: tests/factor_regression/test_rows_lags.py; tests/solutions/factor_regression/test_lags.py
- modify (closed-set lists only): existing tests that pin code groups, labels or contract field lists
Prohibited: adf.py, combination.py, mackinnon.py, wald.py, unit_root_test; ENGINE_VERSION; a change to a
  lag-0 digest; lag selection; a new public name; a change to an existing test other than a closed-set list.
Attach: CLAUDE.md; DECISIONS.md; WORKFLOW.md; pyproject.toml; this brief; context_riskquant_index.txt;
  context_riskquant_factor_regression.txt; context_riskquant_solutions_factor_regression.txt;
  context_riskquant_solutions_run_identity.txt; context_riskquant_tests.txt

## 5. Gates

The WORKFLOW.md section 6 block on the changed and new files, with a "mypy baseline" line on
src/riskquant/solutions/run_identity/hashing.py first; then pytest on tests/factor_regression,
tests/solutions/factor_regression and tests/test_architecture.py; then pytest -q (full). On the owner
machine: the floor gate (Python 3.11, numpy 1.26.4) on the two test directories, with the mypy command
printed and run on the changed src files.

Evidence format: raw output of each command, in the final report.

## 6. Backlog (v2)

- Notebook section on the Dimson estimator with the proxy fixture: a demo change after v1 lands.
- Rolling windows: a separate brief (rolling_fit), after v1 lands, because both change api.py.
- Information criteria (AIC, BIC) on a fixed sample: the caller computes them from rss, n and k today.
- Lags in unit_root_test, DOLS and ECM.

## 7. Closure

| Item | Value |
|---|---|
| Final commit | <hash, dirty=false> |
| Date | <YYYY-MM-DD> |
| Open work | none |
