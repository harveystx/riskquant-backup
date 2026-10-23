# WS_FACTOR_REGRESSION_ROLLING: Rolling-window fits

Path: docs/workstreams/WS_FACTOR_REGRESSION_ROLLING.md
Class: LIGHT (owner decision 2026-10-23: HEAVY by WS_TEMPLATE.md 1.5, run on the LIGHT path)
Status: FROZEN
Revision: 1
Code merged: no
Date: 2026-10-23
Branch: ws/factor-regression-rolling
Snapshot: commit=08d72b6 dirty=false content_hash=3b2f8fc5aa66f67e
Drafted by: owner
Evidence limits: Designed from the source of prod 828c402 and the lags change (contracts.py:AlignedLevels,
api.py:fit_ols). An owner probe ran the loop "align_levels and fit_ols per window" on the credit fixture
(26 windows, 0.22 s). Nothing of the new function was executed.

## 1. Goal and value

Goal: rolling_fit runs fit_ols on successive windows of one AlignedLevels and returns one record per window.
Benefit: loadings over time (stability of proxy and credit factor loadings) with one window definition, one
alignment and one failure rule, instead of a loop that every caller writes.
Case against: the loop is about ten lines today. The function adds no numerics, only a contract.
Reclassify trigger: stop and tell the owner if the work needs a new D decision or a change outside
section 4 Files.

## 2. Scope

In scope (v1):
- rolling_fit(levels, spec, *, window, move) in api.py; RollingWindow and RollingFitResult in contracts.py.
- Window slicing of AlignedLevels; per-window failure records.

Out of scope (see section 6):
- Rolling unit_root_test; tests across windows; plots; a new digest; the notebook.

Parity: the owner loop (align_levels and fit_ols per window). v1 gives the same results without re-alignment.

## 3. Owner decisions in force

- 2026-10-23: a window counts aligned dates. Window i covers the aligned indices i*move to
  i*move + window - 1. A last, incomplete window is dropped.
- 2026-10-23: each window is a slice of the given AlignedLevels, with the exclusions whose date_ord lies in
  the window. rolling_fit does no re-alignment.
- 2026-10-23: each window is a full fit_ols call: the same checks, diagnostics and digests. Lagged specs
  work unchanged; each window drops max_lag rows at both ends, as a single fit does.
- 2026-10-23: a window that fails on its data (InsufficientObservationsError, RankDeficiencyError,
  DegenerateFitError, TransformDomainError) is recorded for that window. The other windows continue. Any
  other error propagates.
- 2026-10-23: a wrong type raises TypeError. A bad window or move raises ModelSpecError. Both raise before
  the first window.
- 2026-10-23: the result holds one record per window: bounds, and either the OlsResult or the error. No array
  holds NaN. No new digest; ENGINE_VERSION stays unchanged.
- 2026-10-23: process: one slice, one FULL code review, no spec audit.

## 4. Acceptance

- C.1 api.py defines rolling_fit(levels: AlignedLevels, spec: ModelSpec, *, window: int, move: int) ->
  RollingFitResult. contracts.py defines RollingWindow (start_ord: int, end_ord: int, result: OlsResult |
  None, error_type: str | None, error_message: str | None) and RollingFitResult (window: int, move: int,
  windows: tuple[RollingWindow, ...]), both frozen, slots, kw_only, eq=False (D064). Exactly one of result and
  error_type is None. [Proposed] Verified by: S1.T1
- C.2 levels not an AlignedLevels or spec not a ModelSpec raises TypeError. window or move that is a bool or
  not an integer raises TypeError (the integer rule of ModelSpec; a numpy integer is stored as int). window <
  1, move < 1 or window > m (no complete window) raises ModelSpecError. [Proposed] Verified by: S1.T2
- C.3 With m aligned dates d: the number of windows is W = floor((m - window) / move) + 1. Window i covers
  d[i*move] to d[i*move + window - 1]; its start_ord is d[i*move] and its end_ord is d[i*move + window - 1] + 1
  (the half-open window of OlsResult.window). [Proposed] Verified by: S1.T3
- C.4 Oracle: for every window, the OlsResult equals fit_ols(align_levels(series, start_ord, end_ord), spec)
  on the original ObservedSeries: coefficients, standard errors, row dates, exclusions and the spec, sample and
  row digests are equal (bit for bit). [Proposed] Verified by: S1.T4
- C.5 A window with too few rows records error_type "InsufficientObservationsError", a window with a constant
  regressor records "RankDeficiencyError"; error_message is the str of the error; result is None; the other
  windows hold results. An error of any other class propagates unchanged. [Proposed] Verified by: S1.T5
- C.6 A spec with lags and max_lag L: every window that holds a result has n_obs equal to the C.3 row formula
  of the lags brief for its window length. [Proposed] Verified by: S1.T6
- C.7 api.__all__ gains exactly rolling_fit, RollingFitResult and RollingWindow. Every existing test passes;
  only the closed-set list _EXPECTED_PUBLIC changes. [Proposed] Verified by: S1.T7 and the full suite

Files:
- modify: src/riskquant/solutions/factor_regression/contracts.py; src/riskquant/solutions/factor_regression/api.py
- create: tests/solutions/factor_regression/test_rolling.py
- modify (closed-set lists only): tests/solutions/factor_regression/test_api.py (_EXPECTED_PUBLIC) and any
  other test that pins the public names or the contract type list
Prohibited: the core package; digest.py; labels.py; a change to fit_ols, wald_test, align_levels or
  unit_root_test behaviour; re-alignment inside rolling_fit; an array with NaN; a new digest; ENGINE_VERSION.
Attach: CLAUDE.md; DECISIONS.md; WORKFLOW.md; pyproject.toml; this brief; context_riskquant_index.txt;
  context_riskquant_factor_regression.txt; context_riskquant_solutions_factor_regression.txt;
  context_riskquant_solutions_run_identity.txt; context_riskquant_tests.txt;
  context_riskquant_ingest_fixtures.txt

## 5. Gates

The WORKFLOW.md section 6 block on the changed and new files, with a "mypy baseline" line on
src/riskquant/solutions/run_identity/hashing.py first; then pytest on tests/factor_regression,
tests/solutions/factor_regression and tests/test_architecture.py; then pytest -q (full). On the owner
machine: the floor gate (Python 3.11, numpy 1.26.4) on the two test directories, with the mypy command
printed and run on the changed src files.

Evidence format: raw output of each command, in the final report.

## 6. Backlog (v2)

- Rolling unit_root_test: low power on short windows; needs its own decision on how to present it.
- A notebook section with rolling loadings on the credit and proxy fixtures: a demo change after v1 lands.
- Tables across windows (coefficient paths as arrays): the caller builds them from the records today.

## 7. Closure

| Item | Value |
|---|---|
| Final commit | <hash> |
| Date | <YYYY-MM-DD> |
| Review | <verdict> |
| Open work | none |
