# WS_FACTOR_REGRESSION_GUARDS: Freeze reconciliation

Path: docs/workstreams/WS_FACTOR_REGRESSION_GUARDS/R_FREEZE_RECONCILIATION_R1.md
Date: 2026-10-21
Spec: WS_FACTOR_REGRESSION_GUARDS.md, revision 1
Round: 1 of 2
Clauses covered: C.1, C.2, C.3, C.4, C.5, C.6, C.7, C.8, C.9, C.10, C.11, C.12, C.13
Clauses not covered: none. C.5, C.7 and C.12 were checked from source only.
Snapshot: SNAPSHOT commit=3b39de7 dirty=true content_hash=7461aad8accf3519
Mode: read-only
Evidence base: the spec; DECISIONS.md (D001, D002, D004, D025, D064, D104, D105); WS_TEMPLATE.md; CLAUDE.md;
WORKFLOW.md; wald_overflow_sweep.py; the profiles index, factor_regression, solutions_factor_regression and tests;
project_structure.txt; pyproject.toml.
Evidence limits: The sandbox source was rebuilt from all attached profiles: 591 files, 0 conflicts between profiles.
The runs used Python 3.12.3 with numpy 2.4.4 and scipy 1.17.1, then Python 3.12.3 with numpy 1.26.4 and scipy 1.15.3.
No run used Python 3.11. No gate ran. The scripts profile was not attached, so its file contents were not searched.
The tests profile has no SNAPSHOT line. AUDIT_REPORT_FACTOR_REGRESSION_AND_COINTEGRATION.md was attached but not
read, because no clause cites it. The sandbox run of tests/factor_regression and tests/solutions/factor_regression
gave 1351 passed on the rebuilt source. That run is a probe. It is not a gate result.

## A. Verdict

NOT READY

Two BLOCKING findings exist. B-1: the S1.T5 expected result contradicts C.3, S1.T1 and S1.T2. B-2: half of C.13
has no test. Both fixes are text edits. Next action: revision 2 applies B-1 and B-2 only. Then round 2 runs.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| Tripwire (section 1.8) | VERIFIED | Revision 1, Code merged no | none |
| D001 | VERIFIED | S1 changes the core only. C.8 forbids an import of api in contracts.py. `tests/test_architecture.py:test_factor_regression_import_direction` stays applicable | none |
| D002 | VERIFIED | No new file in either slice | none |
| D064 | VERIFIED | AlignedLevels keeps eq=False. No change to equality | none |
| D104.1 to D104.6 | VERIFIED | The wald.py change uses float64 arrays and integer codes only. There is no new public name: O-2 uses the existing SeriesContractError. The OD-10 rank rule is a prohibited change in S1 | none |
| D105 | VERIFIED | unit_root_test gets a docstring change only. The ADF path and the critical values do not change | none |
| O-1 = B | VERIFIED | The F-17 sweep reproduced exactly (section C) | none |
| O-2 scope | VERIFIED | test_r019_* assert only `pytest.raises(TypeError)` with no message match. Each valid construction passes C.9. ObservedSeries raises SeriesContractError for a type fault (`contracts.py:ObservedSeries.__post_init__`, "name must be str") | wording: see O-A-1 |
| O-3, O-4, O-5 | VERIFIED | `labels.py:EXCLUSION_NAME_TO_CODE` holds 3 names. `align.py:CoreAlignedLevels` documents 2 reasons. No D104 clause changes, so a D record is not required | none |
| Status move to REVIEW or FROZEN | OWNER APPROVAL REQUIRED | the B-1 and B-2 fixes come first | owner |

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | `wald.py:wald_core` step 9 calls numpy.linalg.cholesky with no check before it. F-3 was reproduced in kind: J=1 gives W 0.0, and J=2 gives W 11.25 against an exact 11.3158. Each run gave the warning "overflow encountered in matmul" and called the cholesky spy once. Variant B raises reason 1 with 0 spy calls and 0 warnings. F-4 was reproduced: the current code raises reason 1 after 1 Cholesky call, and B raises with 0 calls | MATCH WITH REQUIRED CHANGE | R-15 |
| C.2 | Sweep: B returns W in 400 of 400 cases and stays within 1e-11. The worst error is 6.83e-14 on numpy 2.4.4 and 1.11e-13 on numpy 1.26.4 | MATCH WITH REQUIRED CHANGE | R-3 |
| C.3 | Owner case: B gives W = 216.71490314600163, which equals the Fraction oracle. J=2 gives 215475.55709090008 against the exact 215475.55709088963 (relative error 4.8e-14) | MATCH WITH REQUIRED CHANGE | none |
| C.4 | With variant B, every sweep case gave 0 warnings. fit_ols gives no warning at scales 1.95e-156, 2.0e-156 and 3.2e-156. S1.T5 contradicts C.3 | CONTRADICTION | B-1 |
| C.5 | `api.py:_facade_errors` maps CoreRestrictionError to RestrictionError and chains the core error `from` it | MATCH | none |
| C.6 | 35 wald_core calls in test_wald.py and test_api.py reach step 9. The A form and the B form are bit-identical in all 35. abs(m) lies in [1e-200, 2.14] | MATCH WITH REQUIRED CHANGE | R-9 |
| C.7 | The module docstring and the wald_core Notes each list 10 steps. The wald_test Raises section has no overflow text | MATCH WITH REQUIRED CHANGE | none |
| C.8 | `contracts.py:_check_int_param` exists. It accepts numpy integers and returns int | MATCH WITH REQUIRED CHANGE | R-6, R-7 |
| C.9 | `contracts.py:AlignedLevels.__post_init__` checks type, dtype and byte order only (F-9) | MATCH WITH REQUIRED CHANGE | R-8 |
| C.10 | align_core rejects an empty intersection, a bound outside int32 and start_ord >= end_ord (`align.py`). ObservedSeries rejects an empty name, an empty unit, a Cc character and a non-finite value. align_levels passes tuples and Python int bounds | MATCH | none |
| C.11 | A fuzz used 6000 hand-built AlignedLevels that satisfy C.8 and C.9 by construction, with fit_ols and unit_root_test. The inputs included F-order levels, exclusions that do not agree with date_ord, levels of 1e300 and 1e-300, and subnormal levels. Result: 0 IndexError, KeyError, ValueError, OverflowError, AttributeError or Core*Error. Every outcome was a result or a facade class | MATCH WITH REQUIRED CHANGE | R-14 |
| C.12 | `api.py:fit_ols` and `api.py:unit_root_test` both set `window=(levels.start_ord, levels.end_ord)` | MATCH | none |
| C.13 | The Test field names S2.T2 only. S2.T2 holds no C.8 case | CONTRADICTION | B-2 |

The sweep was run as follows (Observed). The command was `python wald_overflow_sweep.py src`, with the working
directory at the rebuilt root. The result matches F-17 exactly on numpy 2.4.4:

| Row | Current, W out of tolerance | A raises | B raises | B worst error |
|---|---|---|---|---|
| J1 [0,1,1] | 30 | 30 | 0 | 5.23e-16 |
| J1 [0,1,-1] | 31 | 31 | 0 | 1.10e-15 |
| J2 [x1+x2, c] | 30 | 30 | 0 | 6.83e-14 |
| J2 [x1-x2, x2] | 33 | 33 | 0 | 9.99e-16 |
| J3 identity | 0 | 34 | 0 | 6.52e-14 |

* The totals are: 124 wrong W from the current code (J=1, J=2), 158 raises from A, and 0 raises from B.
* In all 34 J=3 cases where A raises, the current code raises reason 1.
* numpy 1.26.4 gives the same counts. Only the B worst errors change: 1.11e-13 for J=2 and 6.11e-14 for J=3.

## D. Blast radius of shared interfaces

Search command: `grep -rn --include=*.py -E "AlignedLevels\(" tests | grep -v "isinstance\|CoreAlignedLevels"`, then
the same pattern over src. Files searched: the rebuilt src/ and tests/. scripts/ was not available.

* wald_core callers: `api.py:wald_test` (src). Tests: tests/factor_regression/test_wald.py and
  tests/solutions/factor_regression/test_api.py.
* AlignedLevels constructions in src: `api.py:align_levels` only.
* AlignedLevels constructions in tests: there are 3 (F-14 confirmed).
  * `test_contracts.py:test_t12_aligned_levels_eq_false`, at lines 550 and 559.
  * `test_contracts.py:_aligned_levels`, at line 679.
  * Every valid construction has names ("P",), units ("EUR",), dates 1, 2, 3 in [0, 10) and levels of shape (1, 3).
* Other figures: there are 45 `align_levels(` calls. The grep for "AlignedLevels" over context_riskquant_tests.txt
  gives 32 hits.
* No `dataclasses.replace` acts on an AlignedLevels. The only call is on an OlsResult, at `test_api.py:925`.
* AlignedLevels consumers: `api.py:fit_ols`, `api.py:unit_root_test` and `api.py:_core_aligned_levels`.
* scripts/: project_structure.txt shows `scripts/demos/factor_regression/` with no file. This is Observed from the
  tree only. The file contents were not searched.

## E. Hidden dependency findings

* E-1. `test_wald.py:_oracle_w` is an existing Fraction oracle, at line 63. test_api.py has none. S1.T6 needs one in
  test_api.py. (Observed)
* E-2. No file in tests/factor_regression/ imports riskquant.solutions today. If S1.T1, S1.T2 and S1.T7 go into
  test_wald.py, they add the first facade import in a core test. The architecture test scans src/ and scripts/ only,
  so it does not fail. (Observed)
* E-3. No existing test pins the symmetrization or a W near the float64 limit. A grep for symmetr, m.T, 1e30x, finfo
  and overflow in both test files finds only `TestR012NonFiniteWald`. That test covers a W overflow at step 10 with
  m = 1e-200, and its outcome does not change. (Observed)
* E-4. `_map_core_error` lets an unmapped core error propagate unchanged. For this reason, C.11 depends on C.9 for
  every leak path. The fuzz in section C found no remaining leak path. (Observed; the fuzz is a sample.)

## F. Findings and required changes

Carried from earlier rounds: none.

BLOCKING:

* B-1. S1.T5 contradicts C.3.
  * S1.T5 runs S1.T1 to S1.T4 and expects "CoreRestrictionError".
  * Under O-1 = B, S1.T1 and S1.T2 return W and do not raise. The section C evidence for C.3 shows this.
  * Consequence: the test cannot pass as written.
  * Required change: set the expected result case by case. S1.T1 and S1.T2 give W as in C.3. S1.T3 and S1.T4 give
    CoreRestrictionError(RESTRICTION_NON_FINITE). No RuntimeWarning occurs in any of the four. Put only the
    wald_core call under the "error" filter.
  * Also extend C.4 to name the S1.T1 and S1.T2 inputs. As written, C.4 covers only "the inputs of C.1", and the
    owner case is not a C.1 input under B.
* B-2. The C.8 part of C.13 has no test.
  * C.13 names only S2.T2, and S2.T2 holds only C.9 cases.
  * Consequence: the message rule for C.8 TypeError cases cannot be verified (section 1.3: each clause names a test).
  * Required change: add S2.T3 to the Test field of C.13. Add "the message names the field and, where one exists,
    the exclusion index" to the Expected column of S2.T3.

REFINEMENT:

* R-1. F-6 ranges differ from the reproduction.
  * Reproduction grid: seed 1, scale 1.85e-156 to 3.00e-156, step 0.05e-156.
  * With [0,1,1], A raises from 1.90e-156 to 2.70e-156. With [0,1,-1], A raises from 1.90e-156 to 2.40e-156.
  * F-6 states 2.0e-156 to 2.5e-156 and 1.9e-156 to 2.2e-156.
  * The B part of F-6 matches: B raises only at 1.90e-156 with [0,1,1].
  * F-6 states no seed or grid. Add them, or correct the ranges. The difference makes the case for O-1 stronger.
* R-2. "Silently wrong W" (F-17, section 1) is not accurate. The current code emits the RuntimeWarning "overflow
  encountered in add" (Observed). Write "a wrong W with no error".
* R-3. The C.2 claim "a margin above 100" does not hold on the floor. On numpy 1.26.4 the worst error is 1.11e-13,
  which gives a margin of about 90. The 1e-11 tolerance still holds.
* R-4. F-3 and S1.T3 do not state coef, rhs or R. State the arrays in the test row.
* R-5. Placement of the S1 tests (E-1, E-2). Name the file for S1.T1, S1.T2 and S1.T7. Name the oracle for S1.T6.
  F-2 omits the align_levels window. The sweep uses [20000, 20250). Every window that contains dates 20000 to 20249
  gives the same fit.
* R-6. C.8 and C.9 do not state their relative order. Add "C.8 before C.9". The single-fault tests do not depend on
  the order.
* R-7. C.8 does not say whether the int that `_check_int_param` returns is stored. A numpy integer bound would
  otherwise stay in `window`.
* R-8. Choice of the `series` field. Give one value per C.9 item. Three items need a decision: an unknown exclusion
  series (exclusion.series, or "AlignedLevels.exclusions"), an empty name, and a non-finite level (also say whether
  `date_ord` is set).
* R-9. S1.T8 says "golden values recorded at 3b39de7". The snapshot is dirty=true. Write "recorded on the pre-change
  working tree".
  * F-7 is exact arithmetic. Halving is exact outside the subnormal range, so 0.5*(a+b) and 0.5*a + 0.5*b round to
    the same value when both are finite.
  * The section C instrumented run found no differences.
* R-10. Make S1.T7 assert skips == 0. Observed: 0 skips on numpy 2.4.4 and 1.26.4. As written, S1.T7 can pass when
  all 400 cases are skipped.
* R-11. Floor gate: Python 3.11 did not run. The sweep on numpy 1.26.4 with Python 3.12 gives the same counts.
* R-12. The scripts/ risk row: attach the scripts profile at S2 review, or record the project_structure.txt evidence.
* R-13. S2.T5 samples a universal clause (C.11). Use a property-based test (hypothesis is declared in dev). Use the
  fuzz in section C as a model.
* R-14. The C.11 evidence is a sample. Label it Inferred in the spec.
* R-15. C.1 changes the reason for one input class: a non-finite gap together with a non-positive-definite m. Before
  the change, the result was RESTRICTION_NOT_POSITIVE_DEFINITE. After the change, it is RESTRICTION_NON_FINITE. No
  existing test reaches this class: all 35 logged calls had a finite gap and a finite m. State the change in the C.7
  docstring text.

Owner approval only:

* O-A-1. The O-2 text begins "SeriesContractError for every AlignedLevels admission violation". The Scope sentence of
  the same decision keeps TypeError for C.8. C.8 and C.9 follow the confirmed scope. The audit does not change the
  owner text (STOP AND ASK). The owner may reword the first sentence.

Value case challenge:

* S1 holds. The smallest alternative fails, because F-3 was reproduced: the current code returns W = 0 for J=1 and a
  wrong finite W for J=2, both from a true matmul overflow. The cost is one expression and one check.
* S2 is weaker.
  * In the repository, align_levels and 3 test constructions are the only producers of AlignedLevels.
  * The fuzz shows that the C.9 checks remove every leak path that the audit found.
  * "PROCEED SMALLER" (S1 only) is a valid option against the value case. The 2026-10-20 owner decision makes a
    hand-built AlignedLevels a supported input. That decision rejects this option. The owner may review it.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no (B-1, B-2) |
| Consistent with `DECISIONS.md` | yes |
| Implementable with existing architecture | yes |
| Blast radius bounded | yes |
| Existing behavior protected | conditionally (R-15; scripts/ not searched) |
| Value case holds: the smallest alternative fails, with evidence | yes (S1); conditionally (S2, owner decision) |
| Remaining issues are approvals, not technical gaps | no |

Verdict: NOT READY
