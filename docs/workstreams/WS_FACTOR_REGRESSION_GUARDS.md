# WS_FACTOR_REGRESSION_GUARDS: Wald overflow guard and AlignedLevels admission

Path: docs/workstreams/WS_FACTOR_REGRESSION_GUARDS.md
Class: HEAVY
Status: FROZEN
Revision: 2
Revision note: Revision 2 (2026-10-21). Audit round 1 (R_FREEZE_RECONCILIATION_R1.md): B-1 and B-2 applied; owner rewording of the first sentence of O-2 (O-A-1). Revision 1 (2026-10-20): investigator draft from a PLAN CHANGE on defects N-1 and B-2.
Date: 2026-10-21
Branch: ws/factor-regression-guards
Snapshot: SNAPSHOT commit=dbcb827 dirty=false content_hash=7461aad8accf3519
Drafted by: investigator
Independent review: PASSED (commit=dbcb827 dirty=false content_hash=7461aad8accf3519); rounds
1 and 2; B-3 fixed by the owner header edit at freeze.
Audit rounds used: 2 of 2
Code merged: no
DECISIONS.md reconciled: NO
Evidence limits: Sandbox only (Python 3.12.3, numpy 2.4.4, scipy 1.17.1), source rebuilt from the factor_regression,
solutions_factor_regression and core profiles. No gate ran. No run on Python 3.11 or numpy 1.26.
The tests profile has no SNAPSHOT line. scripts/ is not in the snapshot. The verbatim text of Owner decision 4 is not
attached.

## 1. Goal and scope

Goal: wald_test never returns a W made wrong by overflow of m or gap, and AlignedLevels rejects at construction every
input that fit_ols or unit_root_test cannot use correctly.

Value case:
- Benefit: removes a silent wrong W (W = 0 with finite inputs) for every J, raw IndexError, KeyError, ValueError and
  OverflowError from a hand-built AlignedLevels, and digest collisions from control characters and float bounds.
- Cost proxies: 2 slices; wald.py, contracts.py, docstrings in api.py; test_wald.py, test_contracts.py, test_api.py,
  test_unit_root.py; 0 new public names (if O-2 = AlignmentError); 0 new codes; 0 or 1 D record; at most 2 audit rounds.
- Smallest alternative and why it fails: document "AlignedLevels from align_levels only". It does not stop W = 0 from a
  true matmul overflow (F-3). It contradicts the owner decision that a hand-built AlignedLevels is a supported input.
- Case against: no deadline; the first unit-root run is not affected; N-1 needs regressor 2-norms near 1e-154.
- Recommendation at drafting: PROCEED [Proposed]

In scope:
- wald_core finiteness check of gap and of the matrix passed to numpy.linalg.cholesky; symmetrization form per O-1.
- AlignedLevels.__post_init__ value and type checks; docstrings of fit_ols, unit_root_test and wald_test.

Out of scope:
- v2 unit-root surface (E-1, E-2), LOG transform (E-3), change of the A.4 rule (E-4), the N-3 warning change.
- Overflow guard in combine_core (Owner decision 4). Conditioning of cov. Exclusion-record consistency with date_ord.
- Any change to the CLOSED documents WS_FACTOR_REGRESSION.md and WS_FACTOR_REGRESSION_COINTEGRATION.md.

Owner decisions in force:
- 2026-10-20 (proposed in the request): wald_test does not compute W on a scaled covariance. It raises
  RestrictionError(RESTRICTION_NON_FINITE) only where m or gap overflows (see O-1 = B below).
- 2026-10-20 (proposed in the request): a hand-built AlignedLevels is a supported input. The window is the declared
  half-open window [start_ord, end_ord), not the date range.
- 2026-10-21: O-1 = B. The symmetrization is `0.5 * m + 0.5 * m.T`. Owner evidence (F-17): a 400-case sweep against a Fraction
  oracle; B returns the exact W in all 400 cases (largest relative error 6.8e-14); A raises in 158 of them. The earlier
  count of 34 J=3 cases that the current code computes correctly was wrong: the current code raises there.
  Imports: one changed expression in wald.py and the golden test S1.T8. Cost proxies: 0 new files, 0 new names, 1 test.
- 2026-10-21: O-2 = SeriesContractError for the value checks of C.9 (first sentence reworded by the owner on 2026-10-21, O-A-1). The required `series` string is
  the series name when the violation concerns one series, otherwise "AlignedLevels.<field>". align_levels keeps
  AlignmentError for duplicate names. Scope: the value checks of C.9 raise SeriesContractError. The type checks of C.8
  keep TypeError (owner confirmed 2026-10-21). Rule: a wrong Python type raises TypeError, as the array checks do
  today and test_r019_* pins; a wrong value raises SeriesContractError. ObservedSeries raises SeriesContractError for
  type faults too; AlignedLevels does not, so that test_r019_* stays unchanged. The SeriesContractError docstring
  change of S2 states the rule.
  Imports: a docstring change in errors.py (one more S2 file) that states the convention. Cost proxies: 0 new public
  names, 0 code change in errors.py, message assertions in S2.T2.
- 2026-10-21: O-3 = the known exclusion reasons are OUTSIDE_WINDOW and NOT_COMMON_DATE.
  Imports: one set of two reason names in contracts.py and one rejection case (NOT_USED_BY_ROWS) in S2.T2. Cost
  proxies: 0 new codes, 0 new labels.
- 2026-10-21: O-4 = no D record. The closure record carries the change.
  Imports: nothing. Cost proxies: 0 D records, 0 changes to DECISIONS.md.
- 2026-10-21: O-5 = yes. SCOPE INCREASE: C.8 and C.9 cover finite levels, m >= 1, n_series >= 1, tuple and item types,
  integer bounds in the int32 range, and the type and int32 range of Exclusion.date_ord.
  Imports: about 8 more check items in contracts.py and about 15 more cases in S2.T2 and S2.T3. Cost proxies: 0 new
  files, 0 new public names, 0 new codes, 0 change to slice count.

Depends on: D104, D105
Blocks: none

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | `0.5 * (m + m.T)` turns a finite m entry 1.67646806e308 into inf (owner case, scale 2.0e-156, seed 1) | `src/riskquant/factor_regression/wald.py:wald_core`; sandbox | Observed |
| F-2 | Owner case: J=1 W = 0 (exact 216.715); J=2 W = 426.911 (exact 215476); one warning, "overflow encountered in add". Generator: dates 20000 to 20249; `rng = numpy.random.default_rng(seed)`; `x1 = scale * (1 + 0.5 * rng.random(250))`; `x2` likewise, drawn second; `y = 1 + 0.3 * x1 / scale + 0.2 * x2 / scale + 0.05 * rng.standard_normal(250)`; series Y, X1, X2 with unit "u"; ModelSpec: Y LEVEL on X1 and X2 LEVEL, intercept, step 0, stride 1, IID, rank_rtol 1e-12, condition_threshold None | `src/riskquant/solutions/factor_regression/api.py:wald_test`; Fraction oracle | Observed |
| F-3 | True matmul overflow (cov diag 1e308, off-diagonal 0.9e308): J=1 W = 0.0; J=2 W = 9.0 (exact about 9.0000000105); one warning, "overflow encountered in matmul" | `wald.py:wald_core`; sandbox | Observed |
| F-4 | An overflowed gap (row norm 1e-300, rhs 1e10) already raises RESTRICTION_NON_FINITE at step 10, no warning, J=1 and J=2 | `wald.py:wald_core` step 10; sandbox | Observed |
| F-5 | numpy.linalg.cholesky returns inf and raises no LinAlgError, on numpy 2.4.4 and 1.26.4 | request fact I-003 | Observed |
| F-6 | Option A (check after the current symmetrization) rejects computable cases: scale 2.0e-156 to 2.5e-156 for R=[0,1,1], 1.9e-156 to 2.2e-156 for R=[0,1,-1]. Option B (check plus `0.5*m + 0.5*m.T`): oracle error at most 5.81e-14 relative; rejects only scale 1.9e-156, R=[0,1,1], where m overflows | sandbox prototypes, Fraction oracle | Observed |
| F-7 | `0.5*(a+b)` equals `0.5*a + 0.5*b` bit for bit for exponents -1000 to 1020 (4e6 pairs, 0 differences). 644204 of 2e6 pairs differ for exponents -1074 to -1015 | sandbox, numpy rng seed 7 | Observed |
| F-8 | Precedent: RESTRICTION_NON_FINITE with no warning under `warnings.simplefilter("error")` | `tests/factor_regression/test_wald.py:TestR012NonFiniteWald` | Observed |
| F-9 | AlignedLevels checks only exact type, dtype and byte order of date_ord and levels | `src/riskquant/solutions/factor_regression/contracts.py:AlignedLevels.__post_init__` | Observed |
| F-10 | Hand-built inputs give raw IndexError, KeyError, ValueError, AttributeError, raw Core*Error, or silent success (descending dates, extra level row, start_ord > end_ord, date_ord[-1] == end_ord) | sandbox probe, 29 variants of one align_levels output | Observed |
| F-11 | start_ord 0.5 is accepted; result.window is (0.5, 1000); spec_digest casts the window to int32 0. Exclusion date_ord 5.5 is accepted. start_ord -2**40 raises raw OverflowError | `src/riskquant/solutions/factor_regression/digest.py` (window_arr); sandbox | Observed |
| F-12 | NaN level: fit_ols raises TransformDomainError for a LEVEL term; unit_root_test leaks CoreInputError(reason=9). Empty date axis and 1-D levels leak CoreInputError | sandbox probe | Observed |
| F-13 | EXCLUSION_NAME_TO_CODE holds OUTSIDE_WINDOW, NOT_COMMON_DATE and NOT_USED_BY_ROWS. align_core emits only OUTSIDE_WINDOW and NOT_COMMON_DATE | `src/riskquant/solutions/factor_regression/labels.py`; `src/riskquant/factor_regression/align.py:CoreAlignedLevels` | Observed |
| F-14 | Hand-built AlignedLevels in tests/: `tests/solutions/factor_regression/test_contracts.py:test_t12_aligned_levels_eq_false` (2 instances) and helper `_aligned_levels` (test_r019_* , 3 tests). Every valid one has dates 1, 2, 3 in [0, 10) and levels shape (1, 3). All other tests use align_levels (45 calls). `tests/factor_regression/test_rows.py` builds CoreAlignedLevels, a core type | grep "AlignedLevels" over context_riskquant_tests.txt: 32 hits, 3 constructions | Observed |
| F-16 | Restriction rows [1,0,0] and [1,1e-3,0] (unit-row singular value ratio 5.0e-4, above tau 1e-4), cov identity: relative error of W about 5e-11 (owner: 4e-11) in 2000 random draws of coef and rhs. The bound eps * (1 / ratio^2) is 8.9e-10 | `wald.py:wald_core`; sandbox, Fraction oracle | Observed |
| F-17 | Owner sweep with `wald_overflow_sweep.py` (8 seeds x 10 scales from 3.2e-156 to 1.95e-156 x 5 restrictions = 400 cases; fit_ols fits of the F-2 generator; Fraction oracle; numpy 2.4.4): option B returns W in all 400 cases and raises in 0; largest relative error 6.83e-14 (J=2) and 6.52e-14 (J=3). Option A raises in 158 of the 400 cases, where B returns the exact W. In 124 of them (J=1 and J=2) the current code returns a silently wrong W. In the other 34 (J=3 identity) the current code already raises RESTRICTION_NON_FINITE. The earlier owner statement "A raises in 34 of 80 J=3 cases that the current code computes correctly" was wrong and is withdrawn. An investigator sweep (scale 1.9e-156 to 3.0e-156, J = 1, 2, 3) gave largest errors 6.84e-14 (J=2) and 7.24e-14 (J=3); B raised only where m overflows (1 case per design, scale 1.9e-156) | `wald.py:wald_core`; sandbox, Fraction oracle | Observed |
| F-15 | AlignmentError has a facade-only reason precedent (`_DUPLICATE_SERIES_NAME`, OD-12) and applies `int()` to both bounds | `api.py:align_levels`; `errors.py:AlignmentError.__init__` | Observed |

## 3. Contract

- C.1 After step 8, wald_core raises CoreRestrictionError(RESTRICTION_NON_FINITE, shape) when any entry of gap or of the
  matrix passed to numpy.linalg.cholesky is not finite. numpy.linalg.cholesky is not called. The rule applies for every
  J >= 1. [Proposed] Test: S1.T3, S1.T4
- C.2 For the fits and restrictions of S1.T1, S1.T2 and S1.T7, wald_core returns W within 1e-11 relative of the
  Fraction oracle, or raises RESTRICTION_NON_FINITE. The observed worst error is 7.24e-14 (F-17), so the tolerance has
  a margin above 100. Outside this regime the clause states only that no finite W comes from an overflowed m or gap.
  The rounding error of W otherwise grows with the condition number of m, about 1 / ratio^2 for unit rows (F-16). This
  clause sets no general accuracy bound. [Proposed] Test: S1.T7
- C.3 The symmetrization is `0.5 * m + 0.5 * m.T` (O-1). The owner case returns W within 1e-11 relative of the
  oracle. [Proposed] Test: S1.T1, S1.T2
- C.4 No RuntimeWarning leaves wald_core or wald_test for the inputs of C.1 and for the inputs of S1.T1 and S1.T2 (F-8 precedent). [Proposed] Test: S1.T5
- C.5 wald_test maps the C.1 error to RestrictionError(reason="RESTRICTION_NON_FINITE", shape as received). Its
  `__cause__` is the core error. [Proposed] Test: S1.T6
- C.6 For every existing test_wald.py fixture, W is bit-identical to the value at 3b39de7. [Proposed] Test: S1.T8
- C.7 The check order in the wald.py module docstring and in the wald_core docstring names the new step. The Raises
  section of wald_test names it. [Proposed] Test: S1.T9
- C.8 AlignedLevels raises TypeError, after the existing array checks, when: names, units or exclusions is not exactly
  tuple; a name or unit is not str; an exclusion is not Exclusion; Exclusion.series or Exclusion.reason is not str;
  start_ord, end_ord or Exclusion.date_ord is a bool or not an integer. The integer check reuses
  `contracts._check_int_param`. contracts.py does not import api. [Proposed] Test: S2.T3, S2.T4
- C.9 AlignedLevels raises SeriesContractError when: date_ord is not 1-D, is empty, or is not strictly increasing;
  len(names) == 0; levels is not 2-D with shape (len(names), m); levels is not all finite; len(units) != len(names); a
  name occurs twice; a name or unit is empty or holds a character of Unicode category Cc; an exclusion series is not
  in names; an exclusion reason is not OUTSIDE_WINDOW or NOT_COMMON_DATE; start_ord, end_ord or an exclusion date_ord
  is outside the int32 range; date_ord[0] < start_ord; date_ord[-1] >= end_ord. The `series` field is the series name
  when the violation concerns one series, otherwise "AlignedLevels.<field>". [Proposed] Test: S2.T2
- C.10 Every align_levels output passes C.8 and C.9 unchanged. [Proposed] Test: S2.T1
- C.11 For every AlignedLevels that constructs, fit_ols and unit_root_test return a result, or raise TypeError or a
  class of src/riskquant/solutions/factor_regression/errors.py. They never raise IndexError, KeyError, ValueError,
  OverflowError, AttributeError or a Core*Error. [Proposed] Test: S2.T5
- C.12 The window is the declared [start_ord, end_ord), not the date range. OlsResult.window and UnitRootResult.window
  equal it. [Proposed] Test: S2.T6
- C.13 Each C.8 and C.9 message names the field and, where one exists, the series name or the exclusion index.
  [Proposed] Test: S2.T2, S2.T3

## 4. Decisions

Decisions relied on: D001, D002, D064, D104, D105. Each checked against DECISIONS.md: yes.

No new P- decision. The change stays inside D104.1 to D104.6 (Inferred). O-4 decides a record line.

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Symmetrization: A (keep `0.5*(m+m.T)`; the owner case raises) or B (`0.5*m + 0.5*m.T`; the owner case returns the exact W)? Investigator: B [Proposed] | CLOSED | 2026-10-21 | B. Owner sweep (400 cases); F-1, F-6, F-7, F-17 |
| O-2 | Error class for C.9: AlignmentError with facade-only reasons (no new public name), SeriesContractError, or a new class? Investigator: AlignmentError [Proposed] | CLOSED | 2026-10-21 | SeriesContractError; series convention in section 1 |
| O-3 | Known exclusion reasons: {OUTSIDE_WINDOW, NOT_COMMON_DATE}, or all three names of EXCLUSION_NAME_TO_CODE? Investigator: the two [Proposed] | CLOSED | 2026-10-21 | OUTSIDE_WINDOW, NOT_COMMON_DATE; F-13 |
| O-4 | Add a record line to D104 (Implementation), or none? | CLOSED | 2026-10-21 | none; the closure record carries the change |
| O-5 | Extend the G2 list with: finite levels; m >= 1; n_series >= 1; tuple and item types; integer bounds in the int32 range; type and int32 range of Exclusion.date_ord? | CLOSED | 2026-10-21 | yes, SCOPE INCREASE (section 1); F-10, F-11, F-12 |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1, then S2. No parallel run: both slices modify api.py, and delivery is full files.

### S1: Wald overflow guard

Objective: wald_core and wald_test never return a W made wrong by an overflow of m or gap.
Contract: C.1 to C.7
Depends on: O-1 (CLOSED)
Files:
- create: none
- modify: src/riskquant/factor_regression/wald.py; src/riskquant/solutions/factor_regression/api.py (wald_test
  docstring only); tests/factor_regression/test_wald.py; tests/solutions/factor_regression/test_api.py
Prohibited: a new code constant; a scaled cov; scipy.linalg; a change to steps 1 to 8 or to the rank rule (OD-10); a
  change to an existing test; any other file.
Attach: docs/workstreams/WS_FACTOR_REGRESSION_GUARDS.md; context_riskquant_index.txt;
  context_riskquant_factor_regression.txt; context_riskquant_solutions_factor_regression.txt;
  context_riskquant_tests.txt; DECISIONS.md
Gates:
1. WORKFLOW.md section 6 gate block on the changed files.
2. `pytest -q` (full suite).
3. Floor gate (below).
Exit: every S1 test passes on Python 3.12 and on Python 3.11 with numpy 1.26.4. Every existing test passes.
Status: VALIDATED
Evidence: S1 land 2c9fef5 and fix cc336ab on ws/factor-regression-guards. Windows gates exit=0 (gate_g1.txt,
gate_g1_fix.txt; mypy baseline exit=1 by design); slice 278 passed; pytest full 7153 passed and 36 skipped (7117 + 36
new tests). Floor gate (gate_g1_floor.txt): Python 3.11.9, numpy 1.26.4, scipy 1.15.3; 1387 passed; mypy exit=0.
Independent review: ACCEPT WITH FIXES (R-001 and R-003, P1), then ACCEPT at depth QUICK after the fix. Reviewer
mutation run: 9 mutants; M6 and M8 survived the first review and the fix tests kill them. Owner ruling (OD-R4, R-004):
S1.T8 compares W bit for bit with the pre-change symmetrization run on the same machine and checks the recorded Linux
values to 1e-12 relative. Reason: the recorded literals differed on Windows in 8 of 15 cases, by up to 1.5e-14
relative, with the pre-change code too. The reviewer confirmed from git diff dbcb827 2c9fef5 that the symmetrization
line is the only arithmetic change in step 9, so S1.T8 equals C.6 on each machine. Backlog lines in section 7.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | fit_ols owner case (n 250, LEVEL, intercept, IID, scale 2.0e-156, seed 1), R=[[0,1,1]], rhs 0 | W within 1e-11 relative of the Fraction oracle (216.715) | W = 0 (F-2) |
| S1.T2 | same fit, R=[[0,1,1],[1,0,0]], rhs 0 | W within 1e-11 relative of the oracle (215476) | finite wrong W, J=2 |
| S1.T3 | synthetic cov diag 1e308, off-diagonal 0.9e308; J=1 and J=2 (F-3); monkeypatch spy on numpy.linalg.cholesky | RESTRICTION_NON_FINITE; spy not called | matmul overflow; check after Cholesky |
| S1.T4 | gap overflow (row norm 1e-300, rhs 1e10), J=1 and J=2 | RESTRICTION_NON_FINITE; spy not called | gap not checked |
| S1.T5 | S1.T1 to S1.T4; only the wald_core call runs under `warnings.simplefilter("error")` | S1.T1 and S1.T2: W as in C.3; S1.T3 and S1.T4: CoreRestrictionError(RESTRICTION_NON_FINITE); no RuntimeWarning in any of the four | a warning escapes, or a warning replaces the result or the raise |
| S1.T6 | S1.T1 case through wald_test | WaldResult with W within 1e-11 relative of the oracle | facade drops the C.3 result |
| S1.T7 | the F-2 generator with `scale` in `numpy.geomspace(3.2e-156, 1.95e-156, 10)` and seeds 1 to 8; R in [[0,1,1]], [[0,1,-1]], [[0,1,1],[1,0,0]], [[0,1,-1],[0,0,1]] and the 3 x 3 identity (400 cases; the owner sweep had 0 raises and 0 skips). A case where fit_ols raises (DEGENERATE_NON_POSITIVE_VARIANCE, below about 1.9e-156) is skipped; the final report states the skip count | each outcome is W within 1e-11 relative of the oracle, or RESTRICTION_NON_FINITE | any third outcome |
| S1.T8 | W of each existing test_wald.py fixture; golden values recorded at 3b39de7 before the change | equal bit for bit | silent numerical drift (O-1 = B) |
| S1.T9 | read `inspect.getdoc(wald)`, `wald.wald_core.__doc__` and `api.wald_test.__doc__` | both wald.py check-order lists name the finiteness check of gap and of the matrix passed to numpy.linalg.cholesky before the Cholesky step; the wald_test Raises section names RESTRICTION_NON_FINITE for an overflow of that matrix or of gap | stale docstring (C.7) |

Appendix: none

### S2: AlignedLevels admission

Objective: AlignedLevels rejects at construction every input of C.8 and C.9.
Contract: C.8 to C.13
Depends on: S1; O-2, O-3, O-5 (CLOSED)
Files:
- create: none
- modify: src/riskquant/solutions/factor_regression/contracts.py; src/riskquant/solutions/factor_regression/api.py
  (fit_ols and unit_root_test docstrings only); src/riskquant/solutions/factor_regression/errors.py
  (SeriesContractError docstring only: the series convention and the type and value rule of O-2); tests/solutions/factor_regression/test_contracts.py;
  tests/solutions/factor_regression/test_api.py; tests/solutions/factor_regression/test_unit_root.py
Prohibited: a change to Exclusion, align_levels, _core_aligned_levels, the core package, the digests or ENGINE_VERSION;
  a code change in errors.py; a change to an existing test; a new public name.
Attach: as S1
Gates: as S1
Exit: every S2 test and every existing test pass on both environments.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | AlignedLevels(**fields) of align_levels outputs: the existing fixtures, plus one case with both exclusion reasons | constructs; the existing rebuild tests pass | a check rejects valid output |
| S2.T2 | one case per C.9 item: descending, duplicate, 2-D and empty date_ord; date_ord longer and shorter than levels; extra and missing level row; 1-D levels; NaN and inf level; short units; duplicate names; empty name and unit; `\x1f`, `\x7f`, `\x85` in name and in unit; unknown exclusion series; unknown reason (NOT_USED_BY_ROWS per O-3); date_ord[0] < start_ord; date_ord[-1] == end_ord; date_ord[-1] > end_ord; bounds and exclusion date_ord outside int32 | SeriesContractError; `series` is the series name or "AlignedLevels.<field>" (O-2); the message names the field | each case of request facts I-007, I-008, I-009 |
| S2.T3 | names, units, exclusions as list; int name; str "junk" as exclusion; start_ord 0.5, True, "0", numpy.float64(0); Exclusion.date_ord 5.5, True | TypeError; the message names the field and, where one exists, the exclusion index | silent cast (F-11) |
| S2.T4 | wrong date_ord dtype plus duplicate names | TypeError (array checks first); test_r019_* unchanged | check order |
| S2.T5 | every hand-built AlignedLevels that constructs (window wider than the dates, exclusions present) through fit_ols and unit_root_test | a result, TypeError or a facade error class; no class named in C.11 | raw error path |
| S2.T6 | start_ord = date_ord[0] - 5, end_ord = date_ord[-1] + 10 | constructs; result.window == (start_ord, end_ord); spec_digest differs from the digest of the tight window | window read as date range |

Appendix: none

### Floor gate (S1 and S2)

PowerShell, repository root. The commands create `.venv-floor` only. Do not commit it.

```powershell
py -3.11 -m venv .venv-floor
.\.venv-floor\Scripts\python -m pip install "numpy==1.26.4" "scipy>=1.15,<1.16" "pytest>=8.0" "hypothesis>=6.0" "mypy>=1.10" "ruff>=0.6"
.\.venv-floor\Scripts\python -c "import sys, numpy, scipy; print(sys.version, numpy.__version__, scipy.__version__)"
.\.venv-floor\Scripts\python -m pytest tests/factor_regression tests/solutions/factor_regression -q
.\.venv-floor\Scripts\python -m mypy <changed src files>
```

VERIFY FIRST: the two test directories import no package outside this install list (Inferred). If an import fails,
report it. Add only the version that pyproject.toml declares for that package.

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| O-1 = B changes W when an entry of m is near the subnormal range (F-7) | last-bit change of W | S1.T8 golden values |
| A hand-built caller outside tests/ (scripts/ is not in the snapshot) | a new raise for that caller | the freeze audit attaches the scripts profile |
| The floor environment does not install (scikit-learn floor in pyproject.toml) | the R-15 gate does not run | the floor gate installs only numpy, scipy and the test tools |
| The new check order breaks an existing TypeError test | pytest fails | C.8 order; S2.T4 |

Deferred:
- Consistency of exclusion records with date_ord and the window: opens when a consumer reads exclusion dates numerically.

Backlog (REFINEMENT findings and items that v1 omits):
- Audit round 1 (R_FREEZE_RECONCILIATION_R1.md) refinements R-1 to R-15 are not applied in the spec body. R-4, R-5, R-6, R-7, R-8, R-10, R-13 and R-15 are carried into the S1 and S2 implementer prompts as prompt-level owner decisions. R-1, R-2, R-3, R-9, R-11, R-12 and R-14 are wording or evidence notes. O-A-1 is applied in revision 2.
- Owner decision 4 (WS_FACTOR_REGRESSION_COINTEGRATION, S1), code text in
  `src/riskquant/factor_regression/combination.py:combine_core`: "The core has no overflow guard, as rows.py: a product
  or sum outside the float64 range gives inf with a numpy RuntimeWarning, and the ADF stage rejects a non-finite z
  (S3)." Pinned by `tests/factor_regression/test_combination.py:test_an_overflow_gives_inf_with_a_runtime_warning`.
- E-1, E-2: v2 unit-root surface (E-1: residuals and fit diagnostics of adf_core; E-2: access to the combination
  series z). E-3: LOG transform in fit_ols. E-4: change of the A.4 rule (R2_ABOVE_DURBIN_WATSON with intercept=False).
  N-3: missing np.errstate guards in combination.py and rows.py (warning change). Source:
  AUDIT_REPORT_FACTOR_REGRESSION_AND_COINTEGRATION.md, Part II, findings E-1, E-2, E-3, E-4 and N-3.
- `api.py:_core_aligned_levels` uses `setdefault` for duplicate names. S2 makes the path unreachable.
- AlignedLevels keeps the input memory order. combine_core documents C-order and does not check it.
- Conditioning of cov in wald_core (outside C.2).
- S1 review R-002: the S1 implementer report showed no archive listing, gave condensed gate lines and counted 37 S1 tests instead of 34.
- S1 review R-005: the floor mypy ran on wald.py only. The S2 floor gate prints its mypy command and covers every changed src file.
- S1 U-5: the wald_core Raises text "Steps 6-10" does not name the step 9 cause.
- S1 U-6: pyproject.toml does not declare pydantic, which tests import. This is outside this workstream.

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.
