# WS_FACTOR_REGRESSION: Freeze reconciliation

Path: docs/workstreams/WS_FACTOR_REGRESSION/R_FREEZE_RECONCILIATION_R1.md
Date: 2026-10-12
Spec: WS_FACTOR_REGRESSION.md, revision 2 (A_NUMERICS.md and A_TESTS.md revision 2)
Round: 1 of 2. Tripwire: a BLOCKING finding forces revision 3, after which section 1.8 permits no audit.
Clauses covered: C.1 to C.23, P-FACTOR_REGRESSION-001, O-1 to O-7, O-1a, gate blocks S1 to S7
Clauses not covered: none (depth per clause is in section C)
Snapshot: SNAPSHOT commit=05314f8 dirty=false content_hash=358d3f657c81a072
Mode: read-only
Evidence base: WS_FACTOR_REGRESSION.md, A_NUMERICS.md, A_TESTS.md, WS_TEMPLATE.md, CLAUDE.md, DECISIONS.md,
WORKFLOW.md, REVIEWER.md, INVESTIGATOR.md, R_FREEZE_RECONCILIATION_PREAUDIT.md (claims only),
context_riskquant_index.txt, context_riskquant_core.txt, context_riskquant_solutions_run_identity.txt,
context_riskquant_tests.txt (source of tests/test_architecture.py and tests/test_no_stale_path_comments.py),
all other context profiles (caller search only), pyproject.toml.
Evidence limits:
- INDEPENDENCE: this chat did not draft, revise or pre-audit the spec. Claims K1 to K13 were verified by own runs.
- Sandbox: a partial tree rebuilt from the context profiles (555 .py files; tests profile holds 230 of the test files).
  Python 3.12.3, numpy 2.4.4, scipy 1.17.1, ruff 0.17.0, mypy 2.4.0, pytest 9.1.1. Owner tool versions are not known.
- The full suite did not run. No scripts profile exists, so scripts/ was not searched.
- The spec header names snapshot d4ba609 dirty=true. This audit used 05314f8 dirty=false (owner baseline).
- numpy 1.26 is not available in the sandbox.

## A. Verdict

NOT READY

Seven BLOCKING findings: two impossible tolerances (B-1, B-3), one false expected result (B-2), one gate that cannot
exit 0 (B-4), three self-contradictions (B-5, B-6, B-7). Section F gives exact replacement text for each.
Next action: the owner applies section F as revision 3, then chooses per section 1.8.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| D001 | VERIFIED | Index DEPENDENCIES: 0 edges into `riskquant.solutions` from a module outside it (F-15). C.21 forbids core-to-solution imports. | none |
| D002 | VERIFIED | `tests/test_architecture.py:test_every_init_file_is_empty` and `test_every_module_declares_its_repository_relative_path` use `SOURCE_ROOT.rglob`; `tests/test_no_stale_path_comments.py` uses `SRC_ROOT.rglob`. Both new packages fall under them. | none |
| D004 | VERIFIED | C.14: the result never holds NaN. | none |
| D023 | VERIFIED | Every gate block scopes ruff to `$f`. | none |
| D025 | VERIFIED | Level domains LOG > 0, SIMPLE != 0, DIFFERENCE finite match C.7. | none |
| D031 | VERIFIED | Pydantic is scoped to the scenario-model seam. No locked decision makes Pydantic mandatory for all solution contracts. | none |
| D040, D042 | VERIFIED | Identity carries no timestamp (D040 "Identity vs record", D042). D040 records an environment section for risk runs only. | owner question K11 |
| D064 | VERIFIED | `eq=False` is locked for `UixMapper`, `MarketDataMatrix`, `ValuationState`, `UniverseMapping` only. | none |
| D088 | VERIFIED | Section 2 has no intercept term. Section 3 gives provenance to governance. Authoring format OPEN. F-6 holds. | none |
| O-1 to O-7, O-1a | VERIFIED | Spec section 5: CLOSED 2026-10-12. pydantic is absent from `pyproject.toml` dependencies (F-9). | none |
| P-FACTOR_REGRESSION-001 | OWNER APPROVAL REQUIRED | No contradiction with a locked decision found (D001, D002, D025, D031, D040, D042, D064, D088 read). Clause 3 cites "K8", which is not a spec ID (R-13). | owner assigns the D number |

STOP AND ASK condition (contradiction between P-FACTOR_REGRESSION-001 and a locked decision): not met.

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | New facade type. S6.T1 to S6.T4 do not test unequal lengths or length 0. | MATCH | R-15 |
| C.2 | New code. The duplicate-name rule has no test. | MATCH | R-1, R-12 |
| C.3 | S1.T6, S7.T1. | MATCH | none |
| C.4 | D088 section 4 endpoint read. S2.T1. | MATCH | none |
| C.5 | S6.T5, S6.T6, S7.T8. | MATCH | none |
| C.6 | LOG 1e-15 holds (sandbox max 5.55e-16). A.1 states a false consequence of its own rule. | MATCH WITH REQUIRED CHANGE | B-7, R-7 |
| C.7 | D025. | MATCH | none |
| C.8 | S3.T1 states no tolerance. | MATCH | R-10 |
| C.9 | S3.T4 expected result is false under the A.1 return form. | MATCH WITH REQUIRED CHANGE | B-2 |
| C.10 | S3.T5 holds (sandbox max relative error 8.9e-16 against 1e-12). S3.T7 band fails. | MATCH WITH REQUIRED CHANGE | B-3 |
| C.11 | A.4 rules agree with S3.T8 to S3.T10. Source of `q` for S3.T10 not stated. | MATCH | R-11 |
| C.12 | Intercept tolerance cannot be met. `R < k + 1` has no test. | MATCH WITH REQUIRED CHANGE | B-1, R-2, R-6 |
| C.13 | Facade `ModelSpecError` mapping for the lag has no test. | MATCH | R-14, R-6 |
| C.14 | Rule fires on every sampled exact fit. | MATCH | R-8 |
| C.15 | S4 lists C.15. C.15 fields hold names, units and digests. C.22 and S4 Prohibited forbid them in the core. | CONTRADICTION | B-5 |
| C.16 | A.6 tables agree with all five S6.T8 mutations. `hashing.py:digest_named_arrays(**arrays)` accepts the A.6 names. | MATCH | none |
| C.17 | `row_digest` depends on row dates only. | MATCH | none |
| C.18 | Closed forms correct: t(2) `1 - abs(t)/sqrt(2 + t^2)`, chi2(2) `exp(-W/2)`, F(2,2) `1/(1+F)`. | MATCH | R-9 |
| C.19 | S6.T4, S4.T8, S7.T5. | MATCH | none |
| C.20 | "integer indices and codes only" conflicts with the float payloads of C.9 and S2.T5. | CONTRADICTION | B-6 |
| C.21 | S6 and S7 mypy gates cannot exit 0. | MATCH WITH REQUIRED CHANGE | B-4, R-3, R-4 |
| C.22 | S1.T9, S1.T13 and mypy --strict are compatible if the message goes to `super().__init__` and no `str` annotation exists (Inferred). | MATCH | none |
| C.23 | D001. | MATCH | R-5 |

## D. Blast radius of shared interfaces

`tests/test_architecture.py` (modified in S1).
- Search: AST scan of every `.py` under the rebuilt `src/` and `tests/` for an import of a name that contains
  `test_architecture`, plus text search of every context profile for `test_architecture`.
- Result: no module imports the file. One text reference: the module docstring of `tests/test_no_stale_path_comments.py`.
- Effects: `test_core_packages_obey_the_mechanical_rules` gains `factor_regression` in its tuple. Existing packages are
  unchanged. Two new tests. No existing test changes.
- Baseline (sandbox): `ruff check` exit=0, `ruff format --check` exit=0, `mypy --strict` exit=0 on the unmodified file.

`riskquant.solutions.run_identity.hashing.digest_named_arrays` (read-only reuse, not changed).
- Search: text search of every context profile for `digest_named_arrays(`; AST scan of the rebuilt `src/` and `tests/`
  for calls, with the enclosing symbol. scripts/ not searched (no profile).
- Callers:
  - `src/riskquant/solutions/run_identity/hashing.py`: `hash_risk_factor_matrix`, `hash_parametric_risk_data`,
    `hash_compiled_book`, `hash_compiled_portfolio`, `hash_universe_mapping`, `hash_proxy_model`
  - `src/riskquant/solutions/risk_run/api.py`: module level, `_NO_TAYLOR_REPRESENTATION_DIGEST`
  - `src/riskquant/solutions/risk_run/schemas.py`: `MethodologyDeclaration.digest`, `_result_digest`
  - `src/riskquant/solutions/sensitivity_pnl/taylor_contract.py`: `base_levels_digest`, `base_settlement_values_digest`
  - `tests/solutions/run_identity/test_run_identity.py`: `_composition`, `test_named_bundle_is_order_and_name_sensitive`,
    `test_shock_identity_is_exactly_the_proxy_model_digest`
- The spec adds one caller (`solutions/factor_regression/digest.py`) and changes no existing caller.

## E. Hidden dependency findings

- `hashing.py` has 13 `mypy --strict` errors: 7 bare `npt.NDArray` [type-arg] and 6 untyped parameters [no-untyped-def].
  mypy reports them for every facade file that imports `hashing`. Consequence: B-4. [Observed]
- `pyproject.toml` maps `scipy.*` to `ignore_missing_imports`. `scipy.stats` resolves to `Any` in the core, so no
  `type: ignore` is needed and the FORBIDDEN rule `type:\s*ignore` does not conflict with the core. [Observed]
- S3 diagnostics need the overlap order `q` of A.1, which S2 builds. S2 and S3 run in parallel. Consequence: duplicate
  logic or an undeclared dependency (R-11). [Inferred]
- No FORBIDDEN regex conflicts with code that a slice must write. `print\s*\(` also matches identifiers that end in
  `print(`, for example `fingerprint(`. [Observed]
- `tests/test_no_stale_path_comments.py` fails `ruff format --check` and `mypy --strict` in the sandbox. It is in `$t`
  only, never in `$f`. Keep it out of `$f`. [Observed]

## F. Findings and required changes

- Carried from earlier rounds: none. The preaudit is not a round report (it ran in the drafting chat).

### BLOCKING

B-1. C.12 and S4.T2 intercept tolerance. Clause C.12. Decision: none. Section 1.6: unverifiable result.
Evidence (sandbox, A.3 algorithm against an exact Fraction oracle, `x = 1e7 + N(0,1)`, `y = 3 + 0.5x + 0.1 N(0,1)`,
`numpy.random.default_rng(seed)`, seeds 1 to 20):
- Intercept absolute error 2.0e-4 to 3.5e-3 at 250 rows and 2.1e-5 to 2.6e-3 at 2500 rows. 1e-6 holds in 0 of 20 at both.
- `abs(da) / (mean(x) * abs(db))` is 0.9999 to 1.0001: the intercept error is the slope error times `mean(x)`.
- Intercept relative error: 1e-6 fails in 1 of 20 at 2500 rows (1.84e-6). The relative form is also not robust.
- Slope relative error at most 6.95e-10. Centred intercept `a + b*mean(x)` relative error at most 1.93e-15.
- S4.T2 states no noise. Residual sd 1e-6 raises under C.14 for `y = 1 + 2x` at 250 and 2500 rows (20 of 20) and for
  `y = 3 + 0.5x` at 2500 rows only. With noise 0.1, C.14 never fires.
Consequence: a correct implementation fails S4.T2.
Replacement text:
- `WS_FACTOR_REGRESSION.md`, C.12.
  Old: `Coefficients match an exact Fraction solution to 1e-12 relative on a small case and to 1e-8 (slope) and 1e-6 (intercept) at a mean to sd ratio of 1e7.`
  New: `Coefficients match an exact Fraction solution to 1e-12 relative on a small case. At a mean to sd ratio of 1e7 the slope matches to 1e-8 relative and the centred intercept a + b * mean(x) matches to 1e-12 relative. The raw intercept is not compared, because its error equals the slope error times mean(x).`
- `A_TESTS.md`, S4.T2.
  Old Case: `mean to sd 1e7, 250 rows, intercept`
  New Case: `x = 1e7 + N(0,1), y = 3 + 0.5x + 0.1 N(0,1), numpy.random.default_rng(seed), seeds 1 to 20, 250 and 2500 rows, intercept`
  Old Expected: `slope within 1e-8, intercept within 1e-6 of the Fraction value`
  New Expected: `slope within 1e-8 relative; a + b * mean(x) within 1e-12 relative of the Fraction value, with mean(x) as a Fraction`

B-2. S3.T4 expected result is false under the A.1 return form. F-12 and A.7 are wrong. Clause C.9. Decision: none.
Section 1.6: wrong result.
Evidence (sandbox, 200 seeds, 250 rows, `rank_rtol = 250*eps`):
- A.1 ratio form `ln(v[t]/v[t-1])`, vol 0.1%: the gate raises in 200 of 200 for level pairs (1, 1), (100, 1.1),
  (1e4, 0.01), (3e7, 0.9).
- Difference-of-logs form `diff(log(v))`, levels (100, 1.1), vol 0.1%: 0 of 200 raise, kappa 4.7e12 to 6.7e12. This
  matches the A.7 numbers. [Inferred: A.7 used this form, not A.1.]
- Ratio form, vol 0.02% and 0.01%: 0 of 200 raise for three level pairs; kappa 1.87e12 to 6.43e12; `matrix_rank` 3;
  smallest scaled singular value at least 2.8 times the tolerance.
- Ratio form, vol 1% and 0.5%, `rank_rtol = 1e-10`: 200 of 200 raise; `matrix_rank` 2. S3.T3 holds.
Consequence: a correct implementation that follows A.1 fails S3.T4.
Replacement text:
- `A_TESTS.md`, S3.T4.
  Old Case: `same at vol 0.1% and 0.05% with rank_rtol = 250*eps`
  New Case: `log-return design r_Z, r_S, r_Q from Q = Z*S, returns ln(v[t] / v[t-1]) (A.1 form), vol 0.02% and 0.01%, 250 rows, seeds 0 to 199, rank_rtol = 250*eps`
- `WS_FACTOR_REGRESSION.md`, F-12 Fact cell.
  Old: `The numpy default rank tolerance misses the exact log identity at low volatility (0 of 200). The scaled simple-return design has rank 3 and kappa about 4.6e2 to 4.6e3.`
  New: `With A.1 ratio-form log returns, 250 rows and tolerance 250*eps, the scaled-SVD gate detects the exact log identity at vol 1%, 0.5% and 0.1% (200 of 200) and misses it at vol 0.05%, 0.02% and 0.01% (0 of 200, kappa 1.9e12 to 1.7e13). The scaled simple-return design has rank 3 and kappa * vol between 2.4 and 3.6.`
- `A_NUMERICS.md`, A.7, row "Rank gate, log identity".
  New Setup: `200 seeds, R=250, tol = R*eps, returns ln(v[t]/v[t-1])`
  New Result: `vol 1%, 0.5% and 0.1%: detected 200/200. vol 0.05%: kappa 9.7e12 to 1.7e13, detected 0/200. vol 0.02% and 0.01%: kappa 1.9e12 to 6.4e12, detected 0/200.`
- `A_NUMERICS.md`, A.7, row "Simple-return design".
  New Result: `kappa * vol 2.40 to 3.59 at vol 1%, 0.5%, 0.1% and 0.05%. kappa(0.5%) / kappa(1%) 1.996 to 2.004 per seed. Rank 3 in all runs.`

B-3. S3.T7 kappa band fails for most seeds. Clause C.10. Decision: none. Section 1.6: wrong result.
Evidence (sandbox, 200 seeds, 250 rows): kappa is in [3e2, 7e2] at vol 1% for 57 seeds, at vol 0.5% for 198 seeds, at
both for 55 seeds. `kappa * vol` is 2.40 to 3.58 at both vols and does not depend on the levels.
Consequence: a correct implementation fails S3.T7 for 145 of 200 seeds.
Replacement text:
- `A_TESTS.md`, S3.T7.
  Old Case: `simple-return sr_Z, sr_S, sr_Q, vol 1% and 0.5%, 250 rows, rank_rtol=1e-10`
  New Case: `simple-return sr_Z, sr_S, sr_Q, vol 1% and 0.5%, 250 rows, same seed for both vols, rank_rtol=1e-10`
  Old Expected: `rank 3; kappa between 3e2 and 7e2; threshold 100 flags`
  New Expected: `rank 3; kappa * vol between 1.5 and 5 at both vols; kappa(0.5%) / kappa(1%) between 1.9 and 2.1; threshold 100 flags`

B-4. S6 and S7 mypy gates cannot exit 0. Clause C.21. Decision: D023. Section 1.7 item 3: gate against prohibition.
Evidence (sandbox):
- `mypy --strict` on a facade module that imports `digest_named_arrays`: exit=1, "Found 13 errors in 1 file",
  every error in `src/riskquant/solutions/run_identity/hashing.py`.
- `mypy --strict src/riskquant/solutions/run_identity/hashing.py`: exit=1, the same error lines.
- With `--follow-imports=silent`: exit=0. WORKFLOW.md section 6 excludes the follow-import skip variant.
- S6 Prohibited: "changes to `riskquant.solutions.run_identity`". Section 1 out of scope: "Any change to existing `src/` files".
- `digest.py` (S6), its test (S6) and `api.py` (S7) import `hashing`.
Consequence: the S6 and S7 exit condition "Every gate line prints exit=0" cannot hold.
Replacement text:
- `WS_FACTOR_REGRESSION.md`, S6.
  Old: `Exit: Every gate line prints exit=0. The cross-process digest test (S6.T7) passes.`
  New: `Exit: Every gate line prints exit=0, except the mypy --strict line. That line passes when its ": error:" lines equal the ": error:" lines of the mypy baseline line. The cross-process digest test (S6.T7) passes.`
- `WS_FACTOR_REGRESSION.md`, S7 (the Exit line in section S7 only).
  Old: `Exit: Every gate line prints exit=0.`
  New: `Exit: Every gate line prints exit=0, except the mypy --strict line. That line passes when its ": error:" lines equal the ": error:" lines of the mypy baseline line.`
- `A_TESTS.md`, S6 gate block and S7 gate block. Insert before the `"== mypy --strict"` line:
  `  "== mypy baseline";       mypy --strict src/riskquant/solutions/run_identity/hashing.py; "exit=$LASTEXITCODE"`
- Owner alternative [Proposed]: a separate LIGHT change that types `hashing.py` before S6. The spec does not allow it now.

B-5. C.15 is assigned to the core slice S4. Clauses C.15, C.22. Decision: D001. Section 1.6: self-contradiction.
Evidence: C.15 lists column names, units, three digests and `ENGINE_VERSION`. C.22: the core holds no names, units or
digests. S4 Prohibited: "any digest in the core". A.8: `fit.py` is core. No test checks a populated `OlsResult`
(S6.T12 builds the type directly).
Consequence: S4 cannot meet C.15 and its own prohibition. The populated result has no owner.
Replacement text:
- `WS_FACTOR_REGRESSION.md`, S4. Old: `Contract: C.12, C.13, C.14, C.15, C.19`. New: `Contract: C.12, C.13, C.14, C.19`
- `WS_FACTOR_REGRESSION.md`, C.15. Old: `[Proposed] Test: S4.T8, S6.T12`. New: `[Proposed] Test: S6.T12, S7.T9`
- `WS_FACTOR_REGRESSION.md`, S7.
  Old: `Contract: C.2, C.3, C.5, C.7, C.9, C.11, C.17, C.18, C.19, C.20, C.23`
  New: `Contract: C.2, C.3, C.5, C.7, C.9, C.11, C.15, C.17, C.18, C.19, C.20, C.23`
- `WS_FACTOR_REGRESSION.md`, S7. Old: `Tests: S7.T1 to S7.T8.` New: `Tests: S7.T1 to S7.T9.`
- `A_TESTS.md`, section S7, new row after S7.T8:
  `| S7.T9 | fit_ols on the motivating log-return case | the result holds every C.15 field; every array is read-only; the three digests equal the digest.py values for the same inputs | incomplete result contract |`
- Unresolved: whether an earlier revision retired the ID S7.T9. Git history shows this. This audit cannot.

B-6. C.20 quantifier conflicts with C.9 and S2.T5. Clauses C.9, C.20. Decision: D001. Section 1.6: self-contradiction.
Evidence: C.20 "Core exceptions carry integer indices and codes only". C.9: the core error carries singular values,
participation scores and null relations (float64). S2.T5: `CoreDomainError` carries the level (float). A.8 repeats C.20.
Replacement text:
- `WS_FACTOR_REGRESSION.md`, C.20.
  Old: `Core exceptions carry integer indices and codes only.`
  New: `Core exceptions carry no str data: integer indices, integer codes and numeric values only.`
- `A_NUMERICS.md`, A.8, row `errors.py`, Reason cell.
  Old: `exceptions carry integer indices and codes only`
  New: `exceptions carry integer indices, integer codes and numeric values; no str data`

B-7. A.1 states a false consequence of its own rule. Clause C.6. Decision: none. Section 1.6: self-contradiction.
Evidence (sandbox brute force, `m` 1 to 14, `h` 1 to 9, `s = 1`): 32 cases have unused aligned dates, all with
`m < 2h`. Example: `m = 5, h = 3`: rows `(0, 3)` and `(1, 4)`, index 2 unused.
Consequence: a test written from the sentence asserts an empty set and fails against a correct implementation.
Replacement text:
- `A_NUMERICS.md`, A.1.
  Old: `For h >= 1 and s = 1 this set is empty.`
  New: `For h >= 1, s = 1 and m >= 2h this set is empty. For m < 2h the aligned indices m - h to h - 1 are unused.`

### REFINEMENT

- R-1 (K3): C.2 duplicate-name rule has no test. Add a facade test: two series with one name raise `AlignmentError`.
  Not BLOCKING: the clause states the rule; REVIEWER.md section 4 step 5 checks failure-mode tests.
- R-2 (K4): C.12 `R >= k + 1` has no test. Add a core test: `R = k` raises `CoreSampleSizeError` with `R` and `k`.
- R-3 (K2): The C.21 facade sentence has no "only". It does not forbid `dataclasses` or `__future__`. To make it a
  closed list, write "The facade imports only the standard library, numpy, ...".
- R-4 (K8): `tests/test_architecture.py:_imported_modules` keeps `ImportFrom` with `level == 0` only. S1.T11 must
  resolve relative imports or forbid them in both packages. Dynamic imports are not covered.
- R-5 (K10): Declare `eq=False` on array-bearing contracts. Default `==` raises ValueError for arrays of size above 1
  and returns a bool silently for size 1. `hash` raises TypeError.
- R-6: S4.T1, S4.T4, S4.T5 use elementwise 1e-12 relative. It fails on exact zero entries: 49 of 1998 random 5-row
  integer cases (39 with an exact zero residual; worst nonzero 6.7e-12). Use `max abs error <= 1e-12 * max abs exact`
  per array.
- R-7: S2.T9 SIMPLE identity to 1e-15 absolute holds up to vol 10% (max 5.83e-16) and fails at vol 50% (3.55e-15 to
  7.11e-15). State the vol (at most 10%). LOG holds to vol 50% (max 5.55e-16).
- R-8 (K7): Centred base misses exact fits (0 to 14 of 20 at mean to sd 1e3 or above). Factor `4*R*eps^2`: sampled
  exact fits reached at most `0.295 * R * eps^2 * sum(y^2)`; sampled noisy fits were at least 4.97 times the 4R
  threshold. A sample does not prove "every" or "no". The current rule is conservative and fails loud.
- R-9: `wald_test(result, R, r)` with parameter `R` fails ruff N803. Uppercase locals of A.3 and A.5 fail N806.
  The repository has 0 `noqa: N8` lines. Name the parameters, or permit `# noqa: N803`.
- R-10: State tolerances for S3.T1 ("unchanged"), S4.T3 (p-value "equals") and S5.T3 ("equals").
- R-11: State whether the S3 diagnostic function takes `q` as input or computes it from `h` and `s`.
- R-12: "AlignedLevels" names a core type (A.1) and a facade type with names (C.17, S7.T8). Use two names.
- R-13: P-FACTOR_REGRESSION-001 clause 3 cites "K8". Replace it with a spec ID (section 1.7 item 5).
- R-14: C.13 facade mapping of a lag above `R - 1` to `ModelSpecError` has no test. Add one to S7.
- R-15: C.1 "lengths are equal and at least 1" has no test. Add cases to S6.T2.
- R-16: The rule "pytest full: exit=0, and 36 skipped or fewer" is not in the spec text. Add it to each Exit line.
- R-17 (K12): Value case. With `pythonpath = ["src", "."]`, a script under `scripts/` imports in pytest and in
  `python -c` from the repository root. The script alternative fails outside the checkout and under the architecture
  gates only.
- R-18: `numpy.linalg.matrix_rank(tol=)` exists in numpy 1.26; `rtol` came in numpy 2.0. [Inferred from the numpy API
  history. Not executed.]

### Owner approval only

- K11 provenance. No locked decision requires environment versions or a timestamp in `OlsResult`.
  - Option 1: keep `ENGINE_VERSION` only. Consequence: no record of numpy and scipy versions; cross-version
    differences are not traceable.
  - Option 2: add an environment record (riskquant, python, numpy, scipy versions), outside every digest, as in D040.
    Consequence: one facade field, stdlib `importlib.metadata` and `platform` imports, one test.
  - Option 3: option 2 plus a UTC timestamp, outside every digest (D042 identity-versus-record split). Consequence: a
    non-deterministic field; S7.T6 bitwise comparison must exclude it.
- D number for P-FACTOR_REGRESSION-001 at closure.

## Claims K1 to K13

| Claim | Status | Evidence |
|---|---|---|
| K1 | CONFIRMED | B-5 |
| K2 | REFUTED as a contradiction | C.21 facade sentence has no "only". R-3. |
| K3 | CONFIRMED | No test row in S1, S6 or S7. REFINEMENT R-1. |
| K4 | CONFIRMED | S2.T8 tests the row count, not `R >= k + 1`. REFINEMENT R-2. |
| K5 | CONFIRMED | B-1: 0 of 20 pass at 250 and 2500 rows; error ratio 0.9999 to 1.0001. |
| K6 | CONFIRMED IN PART | Raise depends on `y` scale and `R` (B-1 evidence). Folded into B-1. |
| K7 | CONFIRMED for the centred base; factor claim not falsified on the sample | R-8 |
| K8 | CONFIRMED | `tests/test_architecture.py:_imported_modules`. R-4. |
| K9 | CONFIRMED | Sandbox pytest 9.1.1: importlib mode, 2 passed; prepend mode, collection error. Existing basenames that the spec reuses: `test_errors.py`, `test_diagnostics.py`, `test_contracts.py`, `test_api.py`. |
| K10 | CONFIRMED | Sandbox probe. R-5. |
| K11 | CONFIRMED | D040, D042. Owner question. |
| K12 | CONFIRMED | Sandbox probe. R-17. |
| K13 | F-8, F-13, F-14, F-15, F-16 CONFIRMED. F-17 CONFIRMED for the tests profile; scripts not searched. | F-13: index DEPENDENCIES parse gives no edge into `riskquant.dependence` from outside it; edges into `riskquant.observations` from outside `dependence` are `dependence_stress.engine`, `dependence_stress.permutations`, `market_state.indices`. F-14: `riskquant.dependence` occurs 0 times in DECISIONS.md. F-15: 0 edges. F-16: hashing.py source and index line `riskquant.solutions.run_identity.hashing -> riskquant.config.proxy_model.compiled`. F-17: only `pkgutil.iter_modules` in `tests/config/test_enum_design_integrity.py` and one docstring mention of `py_compile`. |

## Gate blocks against WORKFLOW.md section 6

- Order agrees with WORKFLOW.md section 6. The blocks add `py_compile`, `import` and `pytest full`. No block adds a
  follow-import skip. Every `ruff format` call has `--check`.
- `mypy --strict` agrees with `pyproject.toml` (`strict = true`).
- S6 and S7 blocks fail on the mypy line (B-4).
- S1 `$f` holds `tests/test_architecture.py`; its baseline is clean (section D).

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no (B-5, B-6, B-7) |
| Consistent with `DECISIONS.md` | yes |
| Implementable with existing architecture | yes, after B-4 |
| Blast radius bounded | yes (scripts/ not searched) |
| Existing behavior protected | yes (no existing `src/` file changes) |
| Value case holds: the smallest alternative fails, with evidence | yes, with the R-17 qualification |
| Remaining issues are approvals, not technical gaps | no (B-1 to B-7) |

Verdict: NOT READY
