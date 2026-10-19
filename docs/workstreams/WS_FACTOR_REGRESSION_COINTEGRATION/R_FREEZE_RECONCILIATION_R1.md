# WS_FACTOR_REGRESSION_COINTEGRATION: Freeze reconciliation

Path: docs/workstreams/WS_FACTOR_REGRESSION_COINTEGRATION/R_FREEZE_RECONCILIATION_R1.md
Date: 2026-10-19
Spec: WS_FACTOR_REGRESSION_COINTEGRATION.md, revision 2 (with A_NUMERICS.md, revision 2)
Round: 1 of 2
Clauses covered: C.1 to C.15; P-FACTOR_REGRESSION_COINTEGRATION-001; S1 to S4; A.1 to A.8
Clauses not covered: none
Snapshot: SNAPSHOT commit=b7b088a dirty=false content_hash=72b2cb891c2269d9
Mode: read-only
Evidence base: WS_FACTOR_REGRESSION_COINTEGRATION.md; A_NUMERICS.md; context_riskquant_index.txt;
context_riskquant_factor_regression.txt; context_riskquant_solutions_factor_regression.txt; context_riskquant_tests.txt;
context_riskquant_solutions_run_identity.txt; pyproject.toml; WS_TEMPLATE.md; CLAUDE.md; DECISIONS.md (D001, D002, D004,
D025, D104); WORKFLOW.md; text layer of https://ageconsearch.umn.edu/record/273723/files/qed_wp_1227.pdf (web fetch).
Evidence limits: No repository test ran. No repository checkout exists in the sandbox. Every statement about a test comes
from source text. No pass claim is made for a repository test. The page images of the PDF were not compared: the sandbox
cannot resolve ageconsearch.umn.edu, and the web fetch returns the text layer only. The snapshot manifest holds no
`scripts/*.py` file, so script runners were not searched. Sandbox: Python 3.12.3, numpy 2.4.4, scipy 1.17.1,
statsmodels 0.15.0 (sandbox only). `pyproject.toml` sets mypy `python_version = "3.11"`; no check ran on 3.11.
Tripwire (WS_TEMPLATE.md 1.8): Revision 2, Code merged no. The audit is permitted.

## A. Verdict

NOT READY

Five BLOCKING findings (B-1 to B-5): two existing closed-set tests break or lack a code, digest bytes are not fixed, and
C.7 has no detecting test. Next action: the owner reads section F, Owner approval only. A revision that applies B-1 to
B-5 is Revision 3 with no merged code, so WS_TEMPLATE.md 1.8 fires before round 2.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| D001 four tiers, one-way | VERIFIED | Index graph: no module outside `riskquant.factor_regression` and `riskquant.solutions.factor_regression` imports the core. A.7 places 3 modules in core, 5 in facade. | none |
| D002 empty `__init__.py`, path line | VERIFIED | `tests/test_architecture.py:test_every_init_file_is_empty`, `test_every_module_declares_its_repository_relative_path` scan all modules, new ones included. | none |
| D004 no NaN in core | VERIFIED | C.10 "never holds NaN"; `fit.py:fit_core` rejects non-finite output (`DEGENERATE_NON_POSITIVE_VARIANCE`). No new sentinel is proposed. | none |
| D025 level domains | VERIFIED | C.3 and P-001.4 apply LOG > 0, as D025 and D104.6 do for LOG returns. | none |
| D104.1 to D104.6 | VERIFIED | Core holds float64 constants and int8 codes only (A.7, C.11). No public module other than `api.py` (C.15). No verdict field (C.8, D104.4). No conflict needs an owner choice. | none |
| D104 amendment line and D105 number (O-2) | OWNER APPROVAL REQUIRED | Precedent: "Amended 2026-10-01 by D101" lines in DECISIONS.md (lines 3018, 3533). | Owner assigns D105 at freeze. |
| O-1 to O-8 | VERIFIED | Closed by the owner on 2026-10-19. Reported as approvals only. | none |
| F-14 | VERIFIED | `tests/solutions/factor_regression/test_digest.py:_codes_tuple_names`, `test_t10_all_codes_labeled`, `test_t10_no_labels_group_without_code_tuple`. The label must equal the constant name (`getattr(codes, name) is code`). | none |
| F-15 | VERIFIED | `tests/solutions/factor_regression/test_api.py:_EXPECTED_PUBLIC` (22 names), `test_t7_all_lists_the_public_surface_and_resolves`, `test_t7_helpers_are_private_and_no_other_module_is_public`. | none |
| F-16 | VERIFIED, INCOMPLETE | `codes.py`: 50 `np.int8` constants, 11 tuples (counted: 4, 3, 3, 4, 10, 3, 12, 2, 2, 3, 4). `test_errors.py:test_every_code_group_is_int8_and_unique` lists 11 tuples. The fact omits `test_the_module_defines_exactly_the_core_exception_set` in the same file (B-1). | B-1 |
| F-17 | VERIFIED | Sandbox C2: own A.1 code against `adfuller(autolag=None)`, 25 cases, max relative tau difference 8.13e-14, equal `n`. | none |
| F-7 | VERIFIED (text layer) | PDF text: tau_c N=2 1 % beta2 -22.527; N=3 5 % beta1 -8.5631. statsmodels 0.15.0: -33.527, -8.5632. | none |
| A.2 N = 1 constants against text layer and statsmodels (VERIFY FIRST 2) | VERIFIED | Sandbox C1: 24 cells (23 constants and the empty b3 cell) typed from A.2, from the PDF text layer and from `tau_2010s`: 0 differences. | none |
| A.2 N = 1 constants against page images (VERIFY FIRST 2) | NOT VERIFIED | Page images not available (see Evidence limits). | S2 verification plan (A.2 steps 1 and 4) stays mandatory. |
| fit_core IID standard error (VERIFY FIRST 3) | VERIFIED | Source: `covariance.py:coefficient_covariance` IID branch `C = (rss/dof) * B`, `B = (Xs'Xs)^-1`, `cov = C / outer(d, d)`, `dof = R - k`, k counts the ones column; `fit.py:fit_core` `se = sqrt(diag(cov))`. So `se_j = sqrt(rss/(n-k) * [(X'X)^-1]_jj)` in original units. Sandbox C3: literal transcription against the formula, 20 A.1 designs, max relative difference 7.07e-14. | none |
| Error mapping (VERIFY FIRST 3) | VERIFIED WITH GAPS | `api.py:_map_core_error`, `_facade_errors`, `_ErrorContext.column_names` map every core class used by `fit_core` and `rank_gate`. The ADF path needs column names (B-3), a new sample-size label (B-2) and the new core class in `_CORE_ERRORS` (S4 `api.py`). | B-2, B-3 |
| A.8 against A.3 (VERIFY FIRST 4) | VERIFIED | Values unique in each group: SCALE (0, 1), DETERMINISTIC (0, 1), TEST_KIND (0). The int8 bytes are stable when S1.T7 fixes the values. Text fields of A.3 are not fixed (B-3). | B-3 |

Sandbox commands (each run in `/home/claude/audit`):

| Id | Command | Result |
|---|---|---|
| C1 | `python3 c1_constants.py` | `cells=24 diffs=0`. statsmodels N=2 c 1 % b2 -33.527, N=3 c 5 % b1 -8.5632. S2.T4 order holds for n = 20 to 1e6 (tau_c and tau_ct). Max abs(c(1e9) - b_inf): 6.539e-09 (tau_c), 9.053e-09 (tau_ct). |
| C2 | `python3 c2_adf.py` | Max relative tau difference against `adfuller(autolag=None)`: 8.13e-14 over 25 cases (lags 0, 2, 4, 12; c and ct). Trend origin p+1 against 0: equal tau. S4.T6 arithmetic n = 18. |
| C3 | `python3 c3_iid.py` | IID path of `rank_gate`, `coefficient_covariance`, `fit_core` transcribed: max relative difference 7.07e-14 against `sqrt(rss/(n-k) * inv(X'X)_jj)`. |
| C4 | `python3 c4_powers.py` | Power defects in `c(n)`: max abs error at n=20 / n=100: b2<->b3 swap 0.2511 / 0.0105; b3 omitted 0.0168 / 0.0001; b2 on n^1.5 0.2468 / 0.0256. Each passes S2.T3 (residual 9.05e-09 < 1e-8). |
| C5 | `python3 c5_mc.py` | 100000 random walks, n = 100, lag 0, seed 20261019: abs difference to `c(100)`: tau_c 0.0089, 0.0004, 0.0005; tau_ct 0.0020, 0.0014, 0.0030 (1 %, 5 %, 10 %). 3.8 s. |
| W1 | web fetch of the A.2 URL, text extraction | Table 2 and Table 3 N = 1 rows read on printed pages 13 and 14. Values in C1. |
| G1 | `curl https://ageconsearch.umn.edu/...pdf` | `Could not resolve host`. No page image. |

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | `contracts.py:ModelSpec`, `_check_float_param`, `errors.py:ModelSpecError` give the pattern. | MATCH | R-5 |
| C.2 | `contracts.py:AlignedLevels` holds `names`, `units`, `levels` float64 `(n_series, m)` C-order, `date_ord` int32. | MATCH | none |
| C.3 | `errors.py:CoreDomainError` is keyed by a transform code. A scale needs a new core class. `test_errors.py` fixes the core class set. | MATCH WITH REQUIRED CHANGE | B-1, R-2 |
| C.4 | `design.py:rank_gate` accepts `rank_rtol` in (0, 1). | MATCH | R-4 |
| C.5 | `fit.py:fit_core(design, y, *, intercept, rank_rtol, covariance_kind, covariance_lag)`; IID se confirmed (B, C3). | MATCH | none |
| C.6 | `codes.SAMPLE_SIZE_ERROR_CODES` holds 2 reasons. No reason exists for n below 20. | MATCH WITH REQUIRED CHANGE | B-2, R-1 |
| C.7 | Constants equal the PDF text and statsmodels (C1). No test detects a wrong power (C4). | MATCH WITH REQUIRED CHANGE | B-4 |
| C.8 | `fit_core` gives `coef`, `se`, `gate.condition_number`. "column labels" have no fixed text. | MATCH WITH REQUIRED CHANGE | B-3, R-6 |
| C.9 | `digest.py` precedent fixes every text field. A.3 does not (`column_names`, elided source text). | MATCH WITH REQUIRED CHANGE | B-3 |
| C.10 | `api.py:_map_core_error` needs `column_names` for rank and degenerate errors and a label for every sample-size reason. | MATCH WITH REQUIRED CHANGE | B-2, B-3, R-3 |
| C.11 | `test_architecture.py:_CORE_ALLOWED_IMPORT_PREFIXES` allows numpy, scipy.stats, riskquant.exceptions, own package. C.11 is stricter. `test_factor_regression_core_holds_no_str_type` rejects str and Enum bases. | MATCH | R-9 |
| C.12 | `digest.py` functions and `OlsResult` fields are untouched by the Files lists. `test_api.py:test_t9_result_holds_every_field_and_the_digests` guards `OlsResult`. | MATCH | R-11 |
| C.13 | `contracts.py:_frozen_array_copy` and `fit_core` read-only outputs give the pattern. | MATCH | none |
| C.14 | Covers new tuples. Does not cover a new member of an existing tuple (B-2). | MATCH WITH REQUIRED CHANGE | B-2 |
| C.15 | `test_api.py:_EXPECTED_PUBLIC` and `api.__all__` hold the same 22 names today. Adding 8 names to both keeps S7.T7. | MATCH | none |
| P-001 | Consistent with D104.1 to D104.6 and O-1 to O-8. | MATCH | R-10 |
| S1 | Files list omits `tests/factor_regression/test_errors.py`; labels scope omits an existing group. | MATCH WITH REQUIRED CHANGE | B-1, B-2 |
| S2 | S2.T2 and S2.T3 do not test `c(n)` at small n. | MATCH WITH REQUIRED CHANGE | B-4, R-13 |
| S3 | S3.T4 needs the B-2 reason code. | MATCH WITH REQUIRED CHANGE | B-2 |
| S4 | Exit cites a rule that the Attach list omits. | MATCH WITH REQUIRED CHANGE | B-3, B-5, R-8 |
| A.1 | C2 and C3. `n = m - 1 - p` equals `adfuller` nobs. Trend origin does not change tau. | MATCH | none |
| A.2 | C1, W1. Six rows equal the text layer. Page images not compared. Second source text elided. | MATCH WITH REQUIRED CHANGE | B-3 |
| A.3 | Field order and dtypes fixed. `column_names` text and `window` source not fixed. | MATCH WITH REQUIRED CHANGE | B-3, R-6 |
| A.4 | Fixed texts. No decision word. | MATCH | none |
| A.5 | C1, C2, C5 reproduce the constants, statistic and MC rows. Power, size, HAC and GN rows not repeated. | NOT VERIFIED | none (evidence only) |
| A.6 | Statements, no clause. | MATCH | none |
| A.7 | Placement agrees with D104.1, D104.2 and the hashing rule of S6.T14. | MATCH | none |
| A.8 | Values unique per group; no name clash with the 50 existing constants. Reason code names open. | MATCH WITH REQUIRED CHANGE | B-1, B-2, R-14 |

## D. Blast radius of shared interfaces

Search commands: `awk '/^=== FILE/{file=$3} /from riskquant\.factor_regression|from riskquant\.solutions\.factor_regression/{print file": "$0}'`
over `context_riskquant_factor_regression.txt`, `context_riskquant_solutions_factor_regression.txt` and
`context_riskquant_tests.txt`; `grep -n "factor_regression" context_riskquant_index.txt | grep -- "->"` over the index
import graph. Result: no importer outside the two packages and `tests/`.

| Interface | Change | Callers (`path:symbol`) |
|---|---|---|
| `riskquant/factor_regression/codes.py` | additive constants and tuples; B-2 adds one member to an existing tuple | `align.py`, `covariance.py`, `design.py`, `diagnostics.py`, `fit.py`, `rows.py`, `wald.py` (`codes`); `solutions/factor_regression/contracts.py:Transform`, `CovarianceKind`; `labels.py:ALL_CODE_GROUPS`; `api.py:_map_core_error`; tests: `test_align.py`, `test_covariance.py`, `test_design.py`, `test_diagnostics.py`, `test_errors.py`, `test_fit.py`, `test_rows.py`, `test_wald.py`, `test_api.py`, `test_digest.py` |
| `riskquant/factor_regression/errors.py` | one new class | `align.py`, `covariance.py`, `design.py`, `diagnostics.py`, `fit.py`, `rows.py`, `wald.py`; `api.py:_CORE_ERRORS`, `_map_core_error`; tests: `test_errors.py:_EXPECTED_CLASSES` (closed set, B-1) and 8 other core test modules plus `test_api.py` |
| `solutions/factor_regression/labels.py` | new groups; B-2 adds one entry to `SAMPLE_SIZE_ERROR_NAMES` | `api.py:_map_core_error`; `test_digest.py:test_t10_*` |
| `solutions/factor_regression/errors.py` | one new class | `contracts.py`, `api.py`; `test_contracts.py:_FACADE_ERRORS` (fixed list, no break) |
| `solutions/factor_regression/contracts.py` | new types | `api.py`; `test_api.py`, `test_contracts.py`, `test_digest.py` (`ENGINE_VERSION`) |
| `solutions/factor_regression/digest.py` | two new functions | `api.py`; `test_digest.py:test_t14_scan_sees_the_hashing_import_of_digest` (hashing users must stay `[digest.py]`) |
| `solutions/factor_regression/api.py:__all__` | 8 names added | `test_api.py:_EXPECTED_PUBLIC` (C.15) |

## E. Hidden dependency findings

Closed-set inventory (VERIFY FIRST 1). Every test below enumerates or fixes a set that this workstream changes.

| File | Test | Closed set | Covered by | Status |
|---|---|---|---|---|
| `tests/factor_regression/test_errors.py` | `test_the_module_defines_exactly_the_core_exception_set` | core exception classes `== _EXPECTED_CLASSES` (8) and `_FACTORIES` | nothing | NOT COVERED: B-1 |
| `tests/factor_regression/test_errors.py` | `test_no_core_exception_declares_a_str_field_or_parameter`, `test_the_message_is_the_only_text_on_the_instance`, `test_every_core_exception_subclasses_riskquant_error` | parametrized over `_EXPECTED_CLASSES` | nothing; the new class is not scanned | NOT COVERED: B-1 |
| `tests/factor_regression/test_errors.py` | `test_every_code_group_is_int8_and_unique` | fixed list of 11 tuples | S1.T7 for new tuples (F-16) | COVERED |
| `tests/solutions/factor_regression/test_digest.py` | `test_t10_all_codes_labeled`, `test_t10_no_labels_group_without_code_tuple`, `test_t10_codes_module_declares_code_tuples` | `dir(codes)` `*_CODES` and `labels.ALL_CODE_GROUPS` | C.14, S1 `labels.py`, S1.T6 for new tuples | COVERED for new tuples; NOT COVERED for a new member of `SAMPLE_SIZE_ERROR_CODES`: B-2 |
| `tests/solutions/factor_regression/test_digest.py` | `test_t14_facade_imports_obey_c21` | facade import allow-list (rglob) | C.11, S4.T9 | COVERED |
| `tests/solutions/factor_regression/test_digest.py` | `test_t14_scan_sees_the_hashing_import_of_digest` | hashing importers `== [digest.py]` | A.7 (`digest.py` only importer) | COVERED |
| `tests/solutions/factor_regression/test_digest.py` | `test_t7_*`, `test_t8_*`, `test_t9_oracle_*` | existing digest bytes | C.12, S4 prohibition, S4.T10 | COVERED |
| `tests/solutions/factor_regression/test_api.py` | `test_t7_all_lists_the_public_surface_and_resolves`, `test_t7_helpers_are_private_and_no_other_module_is_public` | `api.__all__ == _EXPECTED_PUBLIC`; top-level defs of `api.py` | C.15, S4 Files, S4.T11 | COVERED |
| `tests/solutions/factor_regression/test_api.py` | `test_t7_api_does_not_import_hashing` | imports of `api.py` | A.7 | COVERED |
| `tests/solutions/factor_regression/test_api.py` | `test_t9_result_holds_every_field_and_the_digests` | `OlsResult` fields `== _C15_FIELDS` | C.12 | COVERED |
| `tests/solutions/factor_regression/test_contracts.py` | `test_t11_subclasses_riskquant_error` (S6.T11) | fixed list `_FACADE_ERRORS` (8) | no break; new class not checked | REFINEMENT R-8 |
| `tests/solutions/factor_regression/test_contracts.py` | `test_t12_*` (S6.T12) | `OlsResult`, `WaldResult` fields | C.12 | COVERED |
| `tests/factor_regression/test_fit.py` | `test_the_result_holds_exactly_the_declared_fields` | `CoreFit` fields | `fit.py` unchanged | COVERED |
| `tests/factor_regression/test_covariance.py` | `test_the_only_loop_of_the_covariance_runs_over_the_lags`, `test_the_fit_has_no_loop_over_rows` | AST of `covariance.py`, `fit.py` | files unchanged | COVERED |
| `tests/test_architecture.py` | `test_factor_regression_import_direction` | rglob core and facade; all of `src/` and `scripts/` must not import the core | C.11, S1.T5 | COVERED |
| `tests/test_architecture.py` | `test_factor_regression_core_holds_no_str_type` | rglob core; no str annotation, no Enum base | C.11, S1.T5 | COVERED |
| `tests/test_architecture.py` | `test_core_packages_obey_the_mechanical_rules` | `factor_regression` modules: no print, no `type: ignore`, no NaN-skipping reduction | pytest full only | COVERED, not named: R-9 |
| `tests/test_architecture.py`, `tests/test_no_stale_path_comments.py` | `test_every_init_file_is_empty`, `test_every_module_declares_its_repository_relative_path` | all modules | D002, pytest full | COVERED |
| `scripts/` runners | none found | `scripts/*.py` absent from the snapshot manifest | not searchable | Unresolved |

Other hidden dependencies:

- E-1. `api.py:_CORE_ERRORS` is a fixed tuple. An unlisted core class leaves `_facade_errors` unmapped. S4 `api.py` covers
  the edit. C.10 does not name it. [Observed]
- E-2. `api.py:_map_core_error` reads `labels.SAMPLE_SIZE_ERROR_NAMES[core_error.reason]`. A reason without a label raises
  `KeyError` inside the mapping. [Observed] This is the runtime consequence of B-2.
- E-3. S4 Gates and Exit cite "WS_FACTOR_REGRESSION section 7 (I-001 rule)". S4 Attach and the WORKFLOW.md section 4
  reviewer attach list do not hold `docs/workstreams/WS_FACTOR_REGRESSION.md`. [Observed] This is B-5.
- E-4. `pyproject.toml` `[tool.mypy] strict = true`, `python_version = "3.11"`. The spec gate "mypy --strict" is
  redundant, not wrong. [Observed]

## F. Findings and required changes

- Carried from earlier rounds: none (round 1).

- BLOCKING:
  - B-1. Core exception closed set. Evidence: `tests/factor_regression/test_errors.py:test_the_module_defines_exactly_the_core_exception_set`
    asserts `set(_defined_classes()) == _EXPECTED_CLASSES` and `set(_FACTORIES) == _EXPECTED_CLASSES` (8 classes). S1 adds
    a domain exception to `factor_regression/errors.py`. S1 Prohibited forbids "a change to any existing test file".
    Consequence: pytest full fails at the S1 exit. The str-payload and message-only tests do not scan the new class.
    Required change: (1) add `tests/factor_regression/test_errors.py` to S1 Files, modify, additive only: one name in
    `_EXPECTED_CLASSES` and one factory in `_FACTORIES`; (2) narrow the S1 prohibition to allow that edit only; (3) name
    the new core class and its payload in A.8 (int and float fields only: term index, aligned index, level, reason);
    (4) add one S1 test row that cites the extended `test_errors.py` tests. The drafter may instead reuse
    `CoreDomainError`. That reuse needs a new transform code (F-3) and changes `TRANSFORM_CODES`, `Transform` and labels.
  - B-2. Reason code for the rule n >= 20. Evidence: `codes.SAMPLE_SIZE_ERROR_CODES` holds `SAMPLE_STEP_NOT_BELOW_COUNT`
    and `SAMPLE_ROWS_NOT_ABOVE_COLUMNS` only. `facade errors.py:InsufficientObservationsError.reason` is a label of that
    group. A.8 names no code for the rule. S1 `labels.py` scope is "one labels group for each new `*_CODES` tuple". S3
    Files modify none. Consequence: S3 cannot implement C.6 and S3.T4 inside its Files list, or the code enters without a
    label and `test_t10_all_codes_labeled[SAMPLE_SIZE_ERROR_CODES]` fails (E-2). Required change: name the code, its group
    and its value in A.8. Proposed: append `SAMPLE_ROWS_BELOW_TABLE_MINIMUM = np.int8(2)` to `SAMPLE_SIZE_ERROR_CODES`.
    Extend the S1 `labels.py` scope and the S1 prohibition to allow that one append and its label. Add the value to
    S1.T7.
  - B-3. Digest and label text not fixed. Evidence: A.3 gives `column_names` uint8 for `unit_root_sample_digest` with no
    line grammar; the existing `digest.py:sample_digest` fixes `Y\x1f{name}`, `C\x1f`, `X\x1f{name}` joined by `\x1e`.
    A.2 writes the second `critical_value_source` text as "... Table 3 tau_ct N=1" (elided). C.8 "column labels" has no
    text. `api.py:_map_core_error` names columns of `CoreRankError` and `CoreDegenerateFitError` from
    `_ErrorContext.column_names`. Consequence: a hash identity (WS_TEMPLATE.md 1.5 production contract) depends on the
    implementer. S4.T7 and S4.T8 cannot detect a different byte format. No oracle can be written. C.10 cannot name a
    rank-deficient ADF column by contract. Required change: in A.3 and A.4, fix (1) the label text of each ADF column:
    ones, trend, `z_{t-1}`, `dz_{t-1}` to `dz_{t-p}`; (2) the full `column_names` grammar, response line included; (3) both
    `critical_value_source` strings in full; (4) the `window` source (R-6). Proposed: add an S4 oracle test that rebuilds
    both new digests with `hashlib` and numpy only, as `test_digest.py:test_t9_oracle_spec_digest_no_threshold` does.
  - B-4. C.7 has no test that detects a wrong power. Evidence: sandbox C4. A b2 and b3 power swap gives an error of
    0.2511 at n = 20 and 0.0105 at n = 100. S2.T2 tolerance at n = 100 is 0.03 (5 %, 10 %) and 0.05 (1 %). S2.T3 residual
    is 9.05e-09 for every defect, below its 1e-8 tolerance. S2.T1 checks constants only. Consequence: a wrong critical
    value passes S2. Required change: add an S2 test that evaluates `c(n)` for the six rows at n in {20, 25, 100, 1000}
    and compares with an exact `fractions.Fraction` evaluation of the constants typed in the test. State the tolerance
    (Proposed: 1e-15 relative). Change the "Defect caught" text of S2.T3 (R-13).
  - B-5. S4 exit depends on a rule that S4 does not attach. Evidence: S4 Gates and Exit cite "WS_FACTOR_REGRESSION
    section 7 (I-001 rule)"; S4 Attach omits `docs/workstreams/WS_FACTOR_REGRESSION.md` (E-3). Consequence: the
    implementer and the reviewer cannot decide whether a mypy exit other than 0 passes. The acceptance is unverifiable.
    Required change: state the rule in S4 (baseline path and the allowed error set), or add the live path to S4 Attach.

- REFINEMENT:
  - R-1. C.6: state the check order when n < 20 and n < k + 1 both hold, and which reason the error carries.
  - R-2. C.3, S1.T3: state the scan order for "first offending date" (term-major as `CoreDomainError`, or date-major).
  - R-3. C.10 says "every core exception". `api.py:_map_core_error` returns `None` for internal `CoreInputError` reasons
    and lets them propagate. List the reachable set: `CoreSampleSizeError`, `CoreRankError`, `CoreDegenerateFitError`,
    `CoreInputError` `INPUT_NON_FINITE` (overflow of `z`), `CoreParameterError`, the new scale class.
  - R-4. C.4: a lag above the int64 range raises `ModelSpecError` before the digest, as `contracts.py:CovarianceSpec` does.
  - R-5. C.1: state the rule for `bool` and `int` weights (`contracts.py:_check_float_param` rejects `bool` with `TypeError`).
  - R-6. C.8, A.3: state that `window` is `AlignedLevels.start_ord, end_ord`. State whether dates `d_0 .. d_p` enter
    `exclusions` as `NOT_USED_BY_ROWS`.
  - R-7. Section 6 cites WORKFLOW.md section 6 for 7 gates. That section lists 4. Cite the 7-gate order as the spec's own.
  - R-8. S6.T11 (`_FACADE_ERRORS`) does not hold `ScaleDomainError`. Check `issubclass(ScaleDomainError, RiskQuantError)`
    in `test_unit_root.py`.
  - R-9. Name `test_architecture.py:test_core_packages_obey_the_mechanical_rules` in S1.T5 and in S2 and S3 gates.
  - R-10. P-001 Evidence: add S4.T2 for clause 5. Clause 6 has no test; add a gate that reads `pyproject.toml`.
  - R-11. S4.T10: "before and after" needs no old-code run. The unchanged S6.T9 and S7 tests, green after S4, are the check.
  - R-12. Value case: ground 2 fails, because `adfuller(autolag=None)` reproduces A.1 (F-17). Ground 3 fails for v1, because
    the F-7 cells are N = 2 and N = 3. The smallest alternative fails on sample binding (F-6, D001, D104.2) and O-3 only.
    Restate the value case on those grounds.
  - R-13. S2.T3: the margin is 9.5e-10 under the tolerance, and no power defect changes the result. Keep it as an
    asymptote check only.
  - R-14. A.8: the reason group of the scale exception shares the prefix `SCALE_` with `SCALE_CODES`. No test conflicts.
    Proposed group name: `SCALE_DOMAIN_ERROR_CODES`.
  - R-15. Run the gates on Python 3.11 (`pyproject.toml` mypy target), not only on 3.12.

- Owner approval only:
  - O-1 to O-8: closed on 2026-10-19. Approvals only.
  - D105: number assignment for P-FACTOR_REGRESSION_COINTEGRATION-001 and the D104 "Amended by D105" line (O-2).
  - Tripwire: the revision that applies B-1 to B-5 is Revision 3 with `Code merged: no`. WS_TEMPLATE.md 1.8 then
    stops further revision and audit. The owner chooses one: build v1 from Revision 3, descope, or stop. Round 2 needs
    that choice.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no (B-2, B-3, B-5) |
| Consistent with `DECISIONS.md` | yes |
| Implementable with existing architecture | yes |
| Blast radius bounded | yes |
| Existing behavior protected | conditionally (C.12; B-1 and B-2 edits additive only) |
| Value case holds: the smallest alternative fails, with evidence | yes (one ground: sample binding and O-3; R-12) |
| Remaining issues are approvals, not technical gaps | no |

Verdict: NOT READY
