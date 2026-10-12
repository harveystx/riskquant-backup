# WS_FACTOR_REGRESSION: Freeze reconciliation pre-audit

Path: docs/workstreams/WS_FACTOR_REGRESSION/R_FREEZE_RECONCILIATION_PREAUDIT.md
Date: 2026-10-12
Spec: WS_FACTOR_REGRESSION.md, revision 2 (A_NUMERICS.md and A_TESTS.md revision 2)
Round: 0 of 2
Clauses covered: C.1 to C.23, P-FACTOR_REGRESSION-001, O-1 to O-7, O-1a, gate blocks S1 to S7
Clauses not covered: none
Snapshot: SNAPSHOT commit=d4ba609 dirty=true content_hash=60ed68e10bd358a7
Mode: read-only
Evidence base: WS_FACTOR_REGRESSION.md (uploaded copy, equal to revision 2 except `Status: REVIEW`), A_NUMERICS.md,
A_TESTS.md, context_riskquant_index.txt, context_riskquant_core.txt, context_riskquant_observations.txt,
context_riskquant_dependence.txt, context_riskquant_solutions_run_identity.txt, context_riskquant_tests.txt
(source of tests/test_architecture.py and tests/test_no_stale_path_comments.py), pyproject.toml, WORKFLOW.md,
WS_TEMPLATE.md, CLAUDE.md, DECISIONS.md, owner baseline in the prompt.
Evidence limits:
- INDEPENDENCE: this audit ran in the chat that drafted revisions 1 and 2. WS_TEMPLATE.md section 1.2 states that the chat
  that wrote the draft cannot audit it. This report is not an independent freeze audit. The owner decides whether it counts as round 1.
- No gate, test, ruff or mypy ran. Every statement about the spec comes from its text. Sandbox numbers are labelled.
- The two repository test files were not attached as files. Their text comes from context_riskquant_tests.txt.
- numpy 1.26 is not available in the sandbox.

## A. Verdict

NOT READY

Five BLOCKING findings: two self-contradictions (C.15 slice ownership, C.21 facade import list) and three unverifiable
acceptance conditions (C.2 duplicate names, C.12 sample size, S4.T2 noise level). Each has exact replacement text in section F.
Next action: owner decision on the independence limit, then revision 3 with the BLOCKING changes only (tripwire, section 1.8).

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| O-1 B2 location | VERIFIED | Spec section 5 CLOSED 2026-10-12, owner. No `src/riskquant/factor_regression` or `solutions/factor_regression` exists (index MANIFEST). | none |
| O-1a frozen dataclasses, not Pydantic | VERIFIED | D031 scopes Pydantic to the scenario-model config seam. D042 `RiskRun` is a frozen dataclass in the solution tier. No locked decision requires Pydantic for every solution contract. | none |
| O-2 own numpy/scipy | VERIFIED | `pyproject.toml` declares numpy and scipy. No new dependency in any slice. | none |
| O-3 HAC Bartlett, Student t | VERIFIED | C.13, A.3. | none |
| O-4 no D088 link | VERIFIED | P-FACTOR_REGRESSION-001 clause 5; C.21 forbids `riskquant.config` in the core. | none |
| O-5 no stationarity tests | VERIFIED | Backlog only. | none |
| O-6 arrays and `str` at the facade | VERIFIED | C.1, C.23. | none |
| O-7 caller `rank_rtol` | VERIFIED | C.5, C.9, A.2. | none |
| D001 tiers and facade | VERIFIED | `api.py` is the only public surface (C.23). Core imports no solution module (C.21). F-15: no non-solution module imports `riskquant.solutions` today. | none |
| D002 empty init, path comment | VERIFIED | `test_every_init_file_is_empty` and `test_every_module_declares_its_repository_relative_path` use `SOURCE_ROOT.rglob` on `src/riskquant`; both new packages fall under them. | none |
| D004 sentinels over NaN in core | VERIFIED | C.14: the result never holds NaN. The `-1` lag code in A.6 is facade digest bytes, not core data. | none |
| D023 lint scope | VERIFIED | Every gate block runs `ruff check $f` and `ruff format --check $f` on the slice files only. | none |
| D025 level domains | VERIFIED | C.7 uses D025 as a convention and states that D025 does not lock it here. | none |
| D088 proxy model | VERIFIED | No feed (P-001 clause 5). Intercept and calibration notes in F-6 match D088 sections 2 and 3. | none |
| P-FACTOR_REGRESSION-001 | OWNER APPROVAL REQUIRED | No contradiction with a locked decision found (D001, D031, D042, D064 read). | owner assigns the D number at closure |

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | `utils.dates` gives int32 ordinals (F-10). New facade type. | MATCH | none |
| C.2 | New code. Duplicate-name rule has no test in S1, S6 or S7. | MATCH WITH REQUIRED CHANGE | B-3 |
| C.3 | New code. S1.T6, S7.T1. | MATCH | none |
| C.4 | D088 section 4 endpoint read. S2.T1. | MATCH | none |
| C.5 | D038 strict typing is scoped to scenario configs. | MATCH | R-6 |
| C.6 | S2 tests cover rows and the row-count raise. | MATCH | none |
| C.7 | D025 rules; `shift_domain` reports no date (F-5). | MATCH | none |
| C.8 | Sandbox A.7 scaling evidence. | MATCH | none |
| C.9 | numpy 2.4.4 `matrix_rank(A, tol, hermitian, *, rtol)` (sandbox). Spec uses `svd` and `tol`. | MATCH | none |
| C.10 | Sandbox kappa bands (A.7). | MATCH | none |
| C.11 | Closed code set; NEAR_PERFECT_FIT removed in revision 2. | MATCH | none |
| C.12 | `R >= k + 1` raise has no test. S4.T2 has no noise level. | MATCH WITH REQUIRED CHANGE | B-4, B-5 |
| C.13 | Lag-to-`ModelSpecError` mapping has no facade test. | MATCH | R-5 |
| C.14 | Sandbox: rule raises on exact fits; uncentred base is the correct scale. | MATCH | R-8 (owner point 1) |
| C.15 | S4 lists C.15, but C.15 fields hold names, units and digests, which C.22 forbids in the core. | CONTRADICTION | B-1 |
| C.16 | `digest_named_arrays` hashes in caller order (hashing.py docstring and body). | MATCH | R-9 (owner point 3) |
| C.17 | S7.T4. | MATCH | none |
| C.18 | S5, S7.T5. | MATCH | none |
| C.19 | S6.T4, S4.T8, S7.T5. | MATCH | none |
| C.20 | `riskquant.exceptions.RiskQuantError` exists and imports nothing. | MATCH | none |
| C.21 | Facade list omits the standard library; `contracts.py` needs `dataclasses` (C.23). | CONTRADICTION | B-2 |
| C.22 | S1.T13 AST scan; A.8 placement. | MATCH | none |
| C.23 | D001; `api.py` only. | MATCH | R-2 |

## D. Blast radius of shared interfaces

`tests/test_architecture.py` (modified in S1). Search: `grep -n "test_architecture"` over context_riskquant_tests.txt and
context_riskquant_scripts.txt. No module imports the file. One reference: a docstring in
`tests/test_no_stale_path_comments.py` (module docstring). Effects:
- `test_core_packages_obey_the_mechanical_rules`: the package tuple gains `factor_regression`. Existing packages are unchanged.
- `test_every_init_file_is_empty`, `test_every_module_declares_its_repository_relative_path`: no edit; they now also scan both new packages.
- Two new tests (S1.T11, S1.T13). No existing test changes.

`riskquant.solutions.run_identity.hashing.digest_named_arrays` (reused, not changed). Search: AWK over every attached profile
for `digest_named_arrays(` outside its definition, with the enclosing `def`. Callers:
- `src/riskquant/solutions/run_identity/hashing.py`: `hash_compiled_book`, `hash_compiled_portfolio`, `hash_parametric_risk_data`,
  `hash_proxy_model`, `hash_risk_factor_matrix`, `hash_universe_mapping`
- `src/riskquant/solutions/risk_run/schemas.py`: `_result_digest`, `digest`
- `src/riskquant/solutions/risk_run/api.py`: one call in a nested function
- `src/riskquant/solutions/sensitivity_pnl/taylor_contract.py`: `base_levels_digest`, `base_settlement_values_digest`
- `tests/solutions/run_identity/test_run_identity.py`: `_composition`, `test_named_bundle_is_order_and_name_sensitive`,
  `test_shock_identity_is_exactly_the_proxy_model_digest`
The spec adds one caller (`solutions/factor_regression/digest.py`). Existing callers are unaffected.

## E. Hidden dependency findings

- `tests/test_architecture.py:_imported_modules` ignores relative imports (`node.level == 0` only). If S1.T11 reuses it, a core module with
  `from . import x` or `from ..solutions import y` escapes the direction test. Consequence: C.21 "any import" is not enforced. See R-1.
- `mypy --strict` on the facade follows the `TYPE_CHECKING` edge of `hashing.py` into `riskquant.config.proxy_model.compiled` (F-16).
  The S6 gate output therefore depends on that module's strict typing. Existing strict code; listed in spec section 7.
- `riskquant.solutions.run_identity.api:engine_identity` records riskquant, python, numpy and numba versions. It is not reusable here:
  `api.py` imports numba and market-data types, which C.21 does not allow in the facade. See R-10.
- `pyproject.toml` sets `pythonpath = ["src", "."]`. A script under `scripts/` is importable by tests inside the checkout. The value case
  claim "fails on reuse" holds for use outside the checkout only. See R-7.

## F. Findings and required changes

- Carried from earlier rounds: none

### BLOCKING

B-1 C.15 is assigned to S4, but it requires names, units and digests that C.22 forbids in the core. The spec contradicts itself
(WS_TEMPLATE.md section 1.6). Decision relied on: D001 (core never holds solution-tier content). Replacement text:
- `WS_FACTOR_REGRESSION.md`, S4 `Contract:` line. Old: `Contract: C.12, C.13, C.14, C.15, C.19`. New: `Contract: C.12, C.13, C.14, C.19`
- `WS_FACTOR_REGRESSION.md`, C.15 test line. Old: `[Proposed] Test: S4.T8, S6.T12`. New: `[Proposed] Test: S6.T12, S7.T10`
- `WS_FACTOR_REGRESSION.md`, S7 `Contract:` line. New: `Contract: C.2, C.3, C.5, C.7, C.9, C.11, C.15, C.17, C.18, C.19, C.20, C.23`
- `WS_FACTOR_REGRESSION.md`, S7 `Tests:` line. New: `Tests: S7.T1 to S7.T10. The table is in `A_TESTS.md`, section S7.`
- `A_TESTS.md`, section S7, new row:
  `| S7.T10 | `fit_ols` on the motivating case | every C.15 field is present with its documented dtype; every array is read-only; no field named accepted, approved or valid | incomplete result contract |`

B-2 C.21 lists the facade imports as "the core, `riskquant.exceptions`, numpy and ... hashing". It omits the standard library.
C.23 requires frozen dataclasses, so `contracts.py` must import `dataclasses`. The spec contradicts itself. Decision: D001. Replacement text:
- `WS_FACTOR_REGRESSION.md`, C.21. Old: `The facade imports the core, `riskquant.exceptions`, numpy and, in `digest.py` only, `riskquant.solutions.run_identity.hashing`.`
  New: `The facade imports only the standard library, numpy, the core, `riskquant.exceptions` and, in `digest.py` only, `riskquant.solutions.run_identity.hashing`.`
- `A_TESTS.md`, S6.T14 Expected cell. New: `every facade import is in the C.21 facade list; only `digest.py` imports `riskquant.solutions.run_identity.hashing`; no direct `riskquant.config` import`

B-3 C.2 requires `AlignmentError` for duplicate series names. No test in any slice checks it. Revision 2 removed the old case from S1.T7
and added none. The acceptance condition is unverifiable. Decision: none (spec-internal, section 1.7 item 4). Replacement text:
- `WS_FACTOR_REGRESSION.md`, C.2 test line. Old: `Test: S1.T5, S1.T7, S7.T1`. New: `Test: S1.T5, S1.T7, S7.T1, S7.T9`
- `A_TESTS.md`, section S7, new row:
  `| S7.T9 | `align_levels` with two series that share one name | `AlignmentError` naming the duplicate name, raised before any core call | ambiguous series mapping |`

B-4 C.12 requires `InsufficientObservationsError` naming `R` and `k` when `R < k + 1`. No test checks it. S2.T8 checks the row count, not the
degrees of freedom. The acceptance condition is unverifiable. Decision: none (section 1.7 item 4). S4.T9 to S4.T12 are removed IDs,
so the new ID is S4.T13. Replacement text:
- `WS_FACTOR_REGRESSION.md`, C.12 test line. Old: `[Proposed] Test: S4.T1, S4.T2`. New: `[Proposed] Test: S4.T1, S4.T2, S4.T13`
- `WS_FACTOR_REGRESSION.md`, S4 `Tests:` line. New: `Tests: S4.T1 to S4.T8, S4.T13. The table is in `A_TESTS.md`, section S4.`
- `A_TESTS.md`, section S4, new row:
  `| S4.T13 | `R = k`: intercept and one regressor on 2 rows | `CoreSampleSizeError` carrying `R` and `k`, raised before the SVD | fit with zero degrees of freedom |`

B-5 S4.T2 runs at mean to sd 1e7 and states no noise level. Under the C.14 rule, a residual sd of 1e-6 raises `DegenerateFitError`
(sandbox: rss 2.0e-10 against threshold 3.1e-10 at R = 250), so the test can fail for a reason outside its intent. The sandbox
setup behind the stated tolerances used noise 0.1 (rss 2.0, threshold 1.9e-11, no raise). The acceptance condition is ambiguous.
Decision: none. Replacement text:
- `A_TESTS.md`, S4.T2 Case cell. Old: `mean to sd 1e7, 250 rows, intercept`.
  New: `x = 1e7 + N(0,1), y = 3 + 0.5x + 0.1 N(0,1), numpy.random.default_rng(1), 250 rows, intercept`
- `A_NUMERICS.md`, A.7 sandbox table, Setup cell of "Centring, ratio 1e6". Old: `with intercept, `n=250`, sd 1`.
  New: `with intercept, `n=250`, x sd 1, y = 3 + 0.5x + 0.1 N(0,1), seeds 1 to 3`

### Rulings on the owner review points

1. Degenerate-fit base. Ruling: keep `sum(y^2)`. Reject the centred sum.
   Evidence (sandbox, 20 seeds per cell, exact `y = 1 + 2x`):
   - Centred rule: with mean to sd at 1e3 or above, it raises on 0 to 14 of 20 exact fits. That breaks the C.14 property "an exact fit raises whether rounding gives 0.0 or not".
   - Uncentred rule: it raises on 20 of 20 in every cell.
   - Computed exact-fit `rss / (eps^2 * sum(y^2))` peaks at 217 (R = 2500). Rounding scales with `|y|`, not with the centred spread.
   The owner's observation is confirmed: at mean to sd 1e7 with residual sd 1e-6, the current rule raises at R = 250 and R = 2500. The factor change is R-8.
   Ruling on a fixed S4.T2 noise level: required. See B-5.
2. Duplicate names. Confirmed. B-3.
3. S6 dependency. Ruling: S6 keeps its dependency on S1 only. S6.T8 uses hand-built row arrays, because the A.6 digests take arrays. R-9.
4. C.15 owner. Ruling: S6 owns the type (S6.T12); S7 owns the populated result (new S7.T10); S4 does not list C.15. B-1.
5. Provenance. Ruling on fields:
   - The spec already records model (terms, step, stride), parameters (covariance, `rank_rtol`, `condition_threshold`), digests and `ENGINE_VERSION`.
   - Add `environment` (riskquant, python, numpy and scipy versions, read with `importlib.metadata` and `platform`) and `created_at_utc` (ISO-8601 `str`) to `OlsResult` and `WaldResult`.
   - Both fields stay outside every digest. This follows the D042 identity-versus-record split, where the identity carries no timestamp.
   - No locked decision requires these fields, so the finding is R-10, not BLOCKING. The project rule in the owner's system instructions is not a `DECISIONS.md` entry. The owner can make it a requirement.

### REFINEMENT (backlog or implementation)

- R-1 S1.T11 must resolve relative imports (`ImportFrom.level > 0`) or forbid them in both packages. Section E.
- R-2 Array-bearing contracts (`ObservedSeries`, `AlignedLevels`, `OlsResult`, `WaldResult`, the core fit result) should declare
  `eq=False`. Precedent: D064 (MD-13) and `tests/credit/test_dataclass_equality_contract.py`. No estate-wide locked rule; the
  dependence package uses default equality.
- R-3 Exit lines: add "pytest full: 0 failed, 0 errors, skipped 36 or fewer" (owner baseline 5746 passed, 36 skipped).
- R-4 The S1 gate runs `ruff format --check` and `mypy --strict` on the existing `tests/test_architecture.py`. Its current state under
  these two tools is not known. Record a baseline before S1.
- R-5 Add a facade test for the lag-to-`ModelSpecError` mapping of C.13.
- R-6 `ModelSpec` should reject `bool` and `str` for numeric fields (D038 method, scoped to scenario configs).
- R-7 Value case: state that the script alternative is importable inside the checkout (`pythonpath = ["src", "."]`). It fails only for use outside the checkout and for the architecture gates.
- R-8 Change the C.14 factor from `(R * eps)^2` to `4 * R * eps^2`. Sandbox: exact fits still raise in every case tried (rss at most 0.087 R eps^2 sum(y^2) at R = 2500). The owner's noisy cases do not raise (ratio 4.1e4 and 5.1e4 against 1.0e3 and 1.0e4). The present rule is conservative and fails loud, so this is not BLOCKING.
- R-9 S6.T8 Case cell: add "row arrays are hand-built; no S2 code".
- R-10 Provenance fields of ruling 5.

### Owner approval only

- The independence limit in the evidence limits.
- The D number for P-FACTOR_REGRESSION-001 at closure.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no (B-1, B-2) |
| Consistent with `DECISIONS.md` | yes |
| Implementable with existing architecture | yes |
| Blast radius bounded | yes |
| Existing behavior protected | yes |
| Value case holds: the smallest alternative fails, with evidence | yes (with R-7 qualification) |
| Remaining issues are approvals, not technical gaps | no (B-1 to B-5) |

Verdict: NOT READY
