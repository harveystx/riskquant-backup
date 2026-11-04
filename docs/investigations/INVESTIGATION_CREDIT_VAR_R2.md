# INVESTIGATION: CREDIT VAR

## RONUD 2

### PROMPT

Investigate per INVESTIGATOR.md.

MODE: PLAN CHANGE
OUTPUT: WORKSTREAM DRAFT
ROUND: 2 of the credit-VaR readiness investigation.
SNAPSHOT: round 1 used commit 90e2498. Round 2 uses commit <SNAP>.
snapshot_delta.txt lists the changes between the two commits.
investigation_credit_var_r1.md holds the round-1 prompt and report.
Round-1 findings I-001 to I-023 are leads, not evidence. Verify each one at
<SNAP> before the draft uses it. Round-1 answers to Q1, Q2, Q3a and Q3b
stand unless verification rejects them. Round 1 framed the goal around one
web app; [G] below replaces that framing.

[Q] QUESTION
Q3d. Read STABLE_API.md as evidence, not as authority. Report:
     - its stated scope and each symbol that it names;
     - for each symbol, whether it exists at <SNAP> with the stated
       signature;
     - each site that cites the file, from STABLE_API_history.txt and the
       profiles;
     - its age and change history, from STABLE_API_history.txt.
     State whether it covers riskquant.solutions.credit_var.api.
     Do not use it as the basis of the credit_var.api stability contract.
Q4a. List each test that calls run_credit_var or simulate_portfolio_losses.
Q4b. For each of the 13 MODEL_CATALOG keys, state whether a public-path test
     with an independent oracle exists. Cite the test and the oracle.
Q5.  Rank each of I-001 to I-023 as BLOCKING or NON-BLOCKING for a first
     stable release of riskquant.solutions.credit_var.api. Separate engine
     gaps from owner decisions.
Q3c. Do not measure. Carry the benchmark specification of round-1 open
     item 1 into the draft as a slice.

[G] CHANGE GOAL
riskquant.solutions.credit_var.api is the single public entry point for
credit VaR models in riskquant. It must be correct and usable on its own
terms, for any consumer: a notebook, a terminal session, a long-running
service, or another project that installs riskquant. Required properties:
  - a typed request and result, with a lossless JSON form;
  - fail-loud validation, with errors that let a caller separate a bad
    request from an engine fault;
  - a declared, versioned stability contract;
  - reproducible runs under D-C15;
  - safe concurrent calls in one process.
This investigation defines the engine-side changes only.

[O] OWNER DECISIONS
O-1 CreditVarResult alpha keys: B. Replace each result dict keyed by
    _alpha_key or _percentile_key with a list of records that carry the
    level as a float, for example {alpha, var, es}. Sort records by
    ascending level. Request validation rejects duplicate alphas; the
    engine does not deduplicate. A lookup method by level raises an error
    for a level that the run did not compute; it does not return NaN.
    A display label such as "99.90%" is not part of the data contract.
O-2 API version: the package version, under semantic versioning over the
    surface that a new D entry declares stable. The draft proposes the
    text of that D entry for riskquant.solutions.credit_var.api. The owner
    assigns its number. The draft does not depend on STABLE_API.md.
    Until a 1.0.0 release, a stated 0.x rule applies: a minor bump marks a
    breaking change to a stable surface.
    Result schema version: CreditVarResult carries a schema version field
    for its serialized form. Major on a breaking change, minor on an added
    optional field.
    Build identity: RunProvenance.engine_version identifies the exact
    build. The spec chooses the mechanism.
    Pins: riskquant keeps NumPy and SciPy ranges, with an upper bound on
    the next major version. Each consuming environment locks exact
    versions. RunProvenance records the NumPy, SciPy and Numba versions of
    each run. Numba is an additional field; D-C15 text does not change.
O-3 Audit IDs C-04, C-07, RQ-AUD-009, P2-07: DEFER. Out of scope for this
    workstream. Tracked as a separate triage item.
O-4 STABLE_API.md disposition: DEFER until round 2 reports its content.
    The draft lists it as an O-n gate with three options: keep it as an
    internal contract of riskquant.credit, merge it into the new D entry,
    or retire it.

[P] PRIORITY CONTEXT
Priority: <high | normal>
Target date: <date>
Related work: <list, or none>

DRAFT CONSTRAINTS:
  1. One slice holds all CreditVarResult and RunProvenance changes, under
     one result schema version. That slice comes first.
  2. The spec states the target shape of CreditVarRequest and
     CreditVarResult, as Python types and as JSON, field by field.
  3. The contract slice acceptance includes a model_dump_json ->
     model_validate_json round-trip test with no loss.
  4. The benchmark is its own slice: script only, no src/ change. No
     performance number appears anywhere without its executed output.
  5. Each owner decision without an answer becomes an O-n gate. Do not
     assume an answer.
  6. Each slice states its review depth: FULL for a src/ contract change or
     an independent oracle; QUICK only for the benchmark script.

VERIFY FIRST:
  b4926a7 adds pydantic to pyproject.toml. The commit is outside a
  workstream and has no review record. Verify I-001 at <SNAP>: the section
  that holds the dependency, its version range, and agreement with the O-2
  pin rule.
  scripts/tests/run_all_tests_solutions_credit_var.py exists, but the tests
  profile has no tests/solutions/credit_var/ path. [Observed, path listing]
  context_riskquant_tests_credit.txt is a filtered extract of the 90e2498
  tests profile: files under tests/credit/, or files whose text contains
  "credit_var" or "simulate_portfolio_losses". snapshot_delta.txt shows
  whether tests/ changed since 90e2498. Scope each negative claim to this
  filter.

OUT OF SCOPE:
  The design of any consumer application: web framework, task queue,
  persistence, authentication, UI.
  solutions.credit_exposure, except where credit_var imports it.
  Any code change. Analysis and draft only.

### RESPONSE

MODE: PLAN CHANGE
SNAPSHOT: `SNAPSHOT commit=c060f06 dirty=false content_hash=fa836966c1293f33`
STATUS: PARTIAL

## Answer

Round 2 verified the round-1 leads at c060f06 and answered Q3d, Q4a, Q4b and Q5. **The workstream draft file is not delivered.** The tool budget for this response ended before the file write. The findings below are the evidence base for the draft. `<SNAP>` was not filled in the prompt, so this answer uses c060f06, which is the commit on the first line of the index.

SELF-CHECK: not run. No draft file exists.

## Evidence map

```
riskquant.solutions.credit_var.api                        [SOLUTION, public]
 ├─ catalog.MODEL_CATALOG      plain dict, mutable (I-012); recipes.SUPPORTED_RECIPES plain dict
 ├─ schemas.CreditVarRequest   pydantic v2, extra="forbid", NOT frozen, no validate_assignment (I-032)
 │    checked:     model/params, unique uids, loadings coverage, groups, MO PD >= systemic_pd, rho use
 │    not checked: duplicate alphas, sum(loading^2) <= 1, duplicate loadings, seed >= 0, uid range (I-033)
 └─ run.run_credit_var(CreditVarRequest | dict)   an instance is not re-validated
     ├─ recipes.assemble_inputs              pd.DataFrame; uid -> int64
     ├─ credit.jtd.ingest.compile.compile_jtd_portfolio
     ├─ MixedCreditFactorGenerator -> build_credit_factor_matrix      (D-C15, D-C04)
     ├─ credit.jtd.engine.run_jtd_for_obligor_ead   9 kernels, all @parallel_njit (I-038)
     ├─ credit.aggregate.summary.summarize_jtd_loss / model_tail_var_es
     ├─ _attribution.obligor_tail_attribution       LOO chunk budget 50_000_000 cells
     └─ CreditVarResult
          summary.var / es / economic_capital_*   keys from _alpha_key (I-004)
          loss_percentiles                        keys from _percentile_key
          obligor_contributions[].ead             float("nan") default (I-006)
          provenance                              9 fields; no construction id, NumPy, SciPy, Numba (I-002)
pyproject.toml [project].dependencies: pydantic>=2.0, numpy>=1.26, scipy>=1.15, numba>=0.59 (no upper bounds)
```

## Findings

**Verification of round-1 leads at c060f06.** Basis "identity" means that the code was not re-read in round 2. The content hash equals the round-1 hash, and `snapshot_delta.txt` lists only `docs/investigations/INVESTIGATION_CREDIT_VAR_R1.md` and `pyproject.toml`. The hash did not change although `pyproject.toml` changed. The hash scope therefore excludes `pyproject.toml` [Inferred].

| ID | Result at c060f06 | Basis |
|---|---|---|
| I-001 | Closed. `"pydantic>=2.0"` is in `[project].dependencies` (runtime). It has no upper bound. The claim that b4926a7 added it is **Unresolved**: the delta shows endpoints only. | re-read |
| I-002 | Holds. The 9 fields are unchanged. | re-read |
| I-003 | Holds. Also: `_engine_version` catches every `Exception` and returns `"0+unknown"`, which is a silent fallback. | re-read |
| I-004 | Holds. The code is unchanged, so the round-1 sandbox output applies. | re-read |
| I-005 | Holds. `float("99.90%")` raises `ValueError`. | re-read |
| I-006 | Holds. | re-read |
| I-007 | **Corrected.** `RiskSummary.es_to_var_ratio` divides Python floats, so a zero VaR raises `ZeroDivisionError` [Inferred], not NaN. `ModelTail.es_to_var_ratio` returns NaN. In `get_var`, `get_es` and `get_es_to_var_ratio`, the NaN default is unreachable because each method raises `ValueError` first. | re-read |
| I-008 | Holds in `src`. The tests extract was not searched for the name, so the negative claim is **Unresolved**. | re-read |
| I-009 | Holds. | re-read |
| I-010 | Holds. A static AST closure from `api` over 5 profiles reaches 95 modules, including package `__init__` files. The closure loads numpy, pandas, scipy, numba and pydantic. 18 of these modules import pandas. | re-read |
| I-011 | Holds. `engine.py:JTD_EVALUATORS = evaluator_table()`; `registry.resolve_evaluator` calls `importlib.import_module`. | re-read |
| I-012 | Holds. `recipes.SUPPORTED_RECIPES` is also a plain `dict[str, object]`. | re-read |
| I-013 to I-016 | Not re-read. Carry these under VERIFY FIRST. | identity |
| I-017 | Holds. Extended by I-039. | re-read (tests) |
| I-018 | Superseded by I-024 to I-028. | re-read |
| I-019 | The recipe to `JtdModel` column holds (`recipes.py`). All 9 kernels use `@parallel_njit`. The evaluator column was not re-read. | partial |
| I-020 | Holds. | re-read |
| I-021 | Still Inferred. `run.py` and `_attribution.py` were re-read; `base.py` and `engine.py` were not. | partial |
| I-022 | Holds in substance. Recount over credit (excluding exposure except `obligor_ead`): `ValueError` 116, `RuntimeError` 9, `TypeError` 2, `ConfigurationError` 1. credit_var: `ValueError` 47, `RuntimeError` 1, `KeyError` 1. scenario_metrics: `ValueError` 11, `TypeError` 3. `_arrays`: `ValueError` 7, `RuntimeError` 2. `pydantic.ValidationError` subclasses `ValueError` [Inferred]. | re-read |
| I-023 | Holds: `nogil=True`, `cache=True`. Extended by I-038. | re-read |

**I-024 | Observed | src/riskquant/credit/STABLE_API.md (Q3d scope and symbols)**
Fact: Scope: "frozen public surface of the credit modelling core". The file lists 28 symbols. 27 exist at the stated path and are in their module `__all__`. The file states `masters.eums.credit_factor_kind.CreditFactorKind`, but the class is at `masters.enums.credit_factor_kind`, where it is in `__all__`.
Evidence: AST check over the extracted `context_riskquant_credit.txt`. The file states names only, not signatures. Current functions: `compile_obligor_master(obligor_df)`, `compile_netting_set_master(netting_set_df)`, `compile_credit_factor_universe(factor_df)`, `build_credit_factor_matrix(batches, *, num_factors, num_scenarios)`, `compile_jtd_portfolio(*, bundle, masters)`, `run_jtd(*, jtd_portfolio, ead_by_obligor, credit_factors)`, `build_portfolio_loss_distribution(result)`, `summarize_jtd_loss(*, result, jtd_portfolio, alphas)`, `model_tail_var_es(*, result, jtd_portfolio, alpha)`, `expected_loss(portfolio_loss)`, `loss_var_es / value_at_risk / expected_shortfall(portfolio_loss, alpha)`, `tail_var_es_by_group(loss_by_group, alpha)`, `loss_quantiles(portfolio_loss, alphas)`. The 13 listed classes exist. `JtdIngestBundle.from_instrument_frame` (`bundle.py:68`) and `MixedCreditFactorGenerator.from_universe` (`mixed.py:169`) exist.
Consequence: Signature stability cannot be checked against the file, because the file states no signature.

**I-025 | Observed | STABLE_API.md policy against `__all__`**
Fact: The policy states "Public = listed = enforced by `__all__`". 29 `__all__` names under `riskquant.credit` are not listed. 16 are outside `exposure`, for example `jtd.engine:run_jtd_for_obligor_ead`, `aggregate.loss_metrics:loss_var_es_contribution`, 4 factor generators and 3 `_scenario_coverage` functions. 13 are in `credit.exposure`.
Evidence: AST scan of all 99 non-init credit modules. credit_var calls 2 unlisted names: `run.py` calls `run_jtd_for_obligor_ead`, and `_attribution.py` calls `loss_var_es_contribution`.
Consequence: The file and the machine-checkable record disagree. The file is not a reliable contract.

**I-026 | Observed | STABLE_API.md: D-C18**
Fact: The file cites D-C18. `DECISIONS.md` has no D-C18 entry (grep returns 0 lines). `tests/credit/jtd/test_beta_binomial_contract.py` also cites D-C18. The "Resolved decisions" section is empty.
Consequence: The file and one test cite a decision that does not exist.

**I-027 | Observed | STABLE_API.md: coverage of credit_var.api**
Fact: The file does not cover `riskquant.solutions.credit_var.api`. Its only reference to `solutions.credit_var` gives the `beta_binomial_grouped` recipe ownership of the per-group factor guarantee. The file declares a core package public. `CLAUDE.md` section 3 states that `riskquant.solutions.*` is the public API.
Consequence: There is a conflict with `CLAUDE.md` section 3. This is evidence for O-4.

**I-028 | Observed | STABLE_API.md citations and history**
Fact: Citation sites: `src/riskquant/credit/factors/simulate/mixed.py:44` (with C-07); `INVESTIGATION_CREDIT_VAR_R1.md` lines 51, 222, 284, 285, 318, 323, 338, 373, 400, 406; `docs/workstreams/WS_TILED_RISK_RUN/R_PROBES_2026-10-08.md:83` (text not available); `context_riskquant_index.txt:48`; `project_structure.txt:482`. No test cites it. History: commit 31200ab on 2026-06-18 and commit dbbfe43 on 2026-09-21, both by Michael Harvey, both with the message ".".
Consequence: The file is 133 days old, and its last change was 38 days ago. No change rationale is recorded.

**I-029 | Observed | Q4a**
Fact: 7 test functions call `run_credit_var`. No test calls `simulate_portfolio_losses`. All 7 import from `credit_var.run` and `credit_var.schemas`, not from `api`.
Evidence: `grep -rn "run_credit_var\|simulate_portfolio_losses"` over the 18-file extract. The tests are: `tests/credit/factors/test_batch_size_reproducibility.py::test_public_credit_var_is_invariant_to_the_internal_factor_batch`, and in `tests/credit/jtd/test_beta_binomial_contract.py`: `test_contradictory_master_pd_is_accepted_not_rejected`, `test_both_obligors_receive_the_same_marginal_despite_60x_pd_spread`, `test_public_expected_loss_matches_the_beta_mean_contract`, `test_provenance_records_the_parameters_that_define_the_marginal`, `test_grouped_pools_carry_different_marginals_and_preserve_membership`, `test_grouped_portfolio_expected_loss_is_the_pool_weighted_sum`.
Consequence: Under O-1, the first test must change, because it reads `dict(summary.var)`. Scope: the filtered 90e2498 extract. `tests/` did not change to c060f06 (`snapshot_delta.txt`).

**I-030 | Observed | tests/solutions/credit_var/**
Fact: The directory exists in `project_structure.txt` with no file listed. `scripts/tests/run_all_tests_solutions_credit_var.py` exists, but its text is in no profile. `scripts/tests/credit/jtd/test_ingest_bundle_validation.py` is a test file outside `tests/`.
Consequence: The runner probably runs zero tests [Inferred]. Its target is **Unresolved**.

**I-031 | Observed | Q4b: independent oracles on the public path**

| Key | run_credit_var test | Independent oracle | Other levels |
|---|---|---|---|
| forced_default | none | none | none |
| independent_bernoulli | `test_public_credit_var_is_invariant_to_the_internal_factor_batch` | none. The test checks self-equality across `_FACTOR_BATCH` values (D-C15 invariance), not a value. | none |
| beta_binomial | 4 tests | EL only: `sum(EAD*LGD)*a/(a+b)`, rel 0.02 | kernel: Beta-mixture marginal, joint and correlation (same file) |
| beta_binomial_grouped | 2 tests | EL only: per obligor (rel 0.05/0.08) and pool-weighted (rel 0.03) | none |
| stress_regime_bernoulli, _grouped | none | none | none |
| sf_gaussian | none | none | kernel compared with the MF kernel (production against production) |
| sf_gaussian_basel, sf_students_t, sf_students_t_basel | none | none | none |
| mf_gaussian, mf_students_t | none | none | kernel: scipy closed forms, `test_mf_factor_independence.py` |
| marshall_olkin | none | none | request validation only |

Consequence: No key has a VaR or ES oracle on the public path. 11 of the 13 keys have no public-path oracle at all.

**I-032 | Observed | schemas.py:CreditVarRequest**
Fact: `_STRICT` sets no `frozen` and no `validate_assignment`. `run_credit_var` accepts an instance without re-validation.
Consequence: A caller can mutate a validated request, for example by appending to `alphas`, and bypass every validator. A request object shared across threads has the same problem.

**I-033 | Observed | recipes.py:_mf_universe_and_bundle, schemas.py**
Fact: The recipe docstring assigns the `sum beta^2 <= 1`, no-duplicate and coverage checks to the engine compile. The request checks only per-row loading range, stray uids and uncovered uids. `seed` has no `ge=0`, and `obligor_uid` has no bound, while the recipes cast uids to `int64`.
Consequence: Some bad requests fail late, after validation, as an engine `ValueError` [Inferred: the raise sites were not re-read]. The exception type alone cannot separate a caller fault from an engine fault.

**I-034 | Inferred | pydantic float and coercion behavior**
Fact: `ObligorRow.ead` has `ge=0.0` only, so `+inf` passes validation. Pydantic serializes inf and NaN as `null` by default, and a `null` float fails `model_validate_json`. Lax mode coerces `"100"` to a number.
Consequence: These behaviors were not executed. S1 and S2 tests must prove or reject them. Coercion conflicts with `CLAUDE.md` section 5.

**I-035 | Observed | run.py:simulate_portfolio_losses**
Fact: The function is in `api.__all__`. It returns `np.ndarray`, not a model, and subsamples with `default_rng(req.seed).choice`.
Consequence: One public function has no JSON form (O-6).

**I-036 | Observed | existing machinery to reuse**
Fact: `run_identity.api.engine_identity` returns the riskquant, python, numpy and numba versions, but not scipy, and uses the same static metadata version as I-003. `run_identity.hashing` implements the D040 canonical float-hex serialization. Other patterns: the `MANIFEST_SCHEMA_VERSION` literal; the `factor_regression` `ENGINE_VERSION` literal; `riskquant.exceptions.RiskQuantError`; `sensitivity_pnl.errors.SensitivityPnlInputError` with `InputDefect`.
Consequence: The draft reuses these patterns. It does not invent new mechanisms.

**I-037 | Observed | pyproject.toml pins against O-2**
Fact: numpy, scipy, numba and pydantic have lower bounds only.
Consequence: The NumPy and SciPy upper bounds are an engine gap under the O-2 rule. Rules for pydantic and Numba are not stated (O-11).

**I-038 | Observed | Numba threading layer**
Fact: All 9 JTD kernels use `parallel_njit`, and 2 tail kernels use `stable_math_parallel_njit`. No profile and no `DECISIONS.md` entry sets a threading layer (grep for `threading_layer` returns 0).
Consequence: Concurrent calls from several threads enter parallel regions at the same time. Safety depends on the threading layer selected at runtime, and some layers are documented as not thread-safe [Inferred]. Unproven.

**I-039 | Observed | further cited IDs with no decision entry**
Fact: The tests extract cites C-15, C-18, C-07A, P2-09 and D-C18. Each has 0 lines in `DECISIONS.md`.
Consequence: These IDs are outside the four that O-3 lists (O-13).

**I-040 | Observed | run.py:run_credit_var provenance**
Fact: `generated_at` is a wall-clock timestamp in the result. `_input_hash` truncates SHA-256 to 16 hex characters and serializes floats with `json` repr.
Consequence: The spec must define reproducibility equality without `generated_at`. The D040 hashing is the reuse candidate.

### Q5 ranking (first stable release of `credit_var.api`)

| ID | Rank | Kind | Reason |
|---|---|---|---|
| I-001 | NON-BLOCKING | engine (closed) | Declared. The pin residue is I-037. |
| I-002 | BLOCKING | engine | D-C15 release condition. O-2 adds Numba. |
| I-003 | BLOCKING | engine | O-2 requires the exact build identity. The fallback is silent. |
| I-004, I-005 | BLOCKING | engine (O-1 decided) | Silent collision; property that always raises. |
| I-006 | BLOCKING | engine | NaN breaks lossless JSON. |
| I-007 | BLOCKING | engine + owner (O-5) | `ZeroDivisionError` on zero VaR. |
| I-008 | NON-BLOCKING | engine | Dead class; backlog. |
| I-009, I-010 | NON-BLOCKING | engine + owner (O-7) | Depends on whether `from_dataframe` is in the surface. |
| I-011 | NON-BLOCKING | engine | Cost; the benchmark measures it. |
| I-012 | BLOCKING | engine | Process-global mutable state. |
| I-013 to I-016 | NON-BLOCKING | none | Compliance; VERIFY FIRST. |
| I-017 | NON-BLOCKING | owner (O-3 DEFER) | Out of scope. |
| I-018 | NON-BLOCKING | owner (O-4) | The draft does not depend on it. |
| I-019, I-020 | NON-BLOCKING | engine | Map; one-way guard. |
| I-021 | NON-BLOCKING | engine + owner (O-12) | Measure, then set the limits. |
| I-022 | BLOCKING | engine + owner (O-9) | G requires the request/engine split. |
| I-023 | BLOCKING | engine + owner (O-10) | Concurrency is unproven (I-012, I-032, I-038). |

### Adjacent (not in scope)

* The `schemas.py` docstring shows keys such as `{"0.99": ...}`, but the code emits `"99.00%"`.
* `loss_percentiles` uses `np.percentile` with linear interpolation, while `summary.var` uses `loss_var_es`. The two estimators can differ at the same level [Inferred].
* D104.3 gives "pydantic is not declared" as a reason. That reason is stale at c060f06.
* The `api.py` docstring states that the internals "may change shape freely". This is a stability claim in a docstring.
* `recipes._universe` still writes `np.nan`.

## Options

**Option A: six-slice HEAVY spec `WS_CREDIT_VAR_API` [Proposed].**
Slices: S1 result contract (all `CreditVarResult` and `RunProvenance` changes, schema version, round-trip test); S2 request validation and error taxonomy; S3 concurrency (immutable maps, frozen request, concurrent-run test); S4 per-key oracle tests through `api`; S5 stability declaration (P-CREDIT_VAR_API-001 text, surface test, O-2 pins); S6 benchmark script (QUICK).
Consequence: All five G properties get a test. Touches D-C15, D040, D041, D039 and D104, plus one new D entry.

**Option B: S1 and S6 only.**
Consequence: Fails G on the error split (I-022, I-033), on concurrency (I-012, I-032, I-038) and on the declared contract.

**Option C: do nothing.**
Consequence: The I-002, I-004, I-006 and I-007 defects stay in place, and every consumer works around them.

## Value assessment

```
VALUE
Benefit:        one typed, versioned, fail-loud entry point; D-C15 release condition met.
Cost proxies:   6 slices; schemas.py, run.py, catalog.py, recipes.py, api.py, new errors
                module, pyproject.toml, tests; 1 wire contract; 1 new D entry; 9 new gates.
Alternative:    Option B. Fails G: I-022, I-033, I-012, I-032, I-038.
Not doing it:   collisions, NaN, ZeroDivisionError, mixed error types, unproven concurrency.
Case against:   11 of 13 keys have no public-path oracle (I-031). A stable declaration
                before S4 would freeze unverified numbers. Mitigation: S5 after S4.
Reversibility:  costly after a consumer pins a release.
Parity:         n/a
Owner inputs:   [P] priority, target date, related work (placeholders).
Class:          HEAVY (public signature, serialization identity, validation, new D entry)
Recommendation: PROCEED [Proposed]
```

Evidence that would change the recommendation: if O-9 and O-10 are descoped, PROCEED SMALLER applies.

## Open items

1. **Draft not delivered.** Input that resolves it: the next prompt below.
2. **Proposed new gates.** Each is OPEN; the draft will list them as O-n.
   * **O-5:** ES/VaR ratio when VaR = 0: raise, omit the value, or remove the ratio.
   * **O-6:** whether `simulate_portfolio_losses` is part of the stable surface.
   * **O-7:** whether the pandas-typed `from_dataframe` is part of the stable surface.
   * **O-8:** request coercion mode: strict or lax.
   * **O-9:** error taxonomy shape: keep `ValidationError` for request faults, or wrap it.
   * **O-10:** threading-layer policy for concurrent calls.
   * **O-11:** upper-bound pin rule for pydantic and Numba.
   * **O-12:** request limits on `n_scenarios` and obligor count.
   * **O-13:** whether C-15, C-18, C-07A, P2-09 and D-C18 join the O-3 triage item.
3. **O-4 stays OPEN**, with three options. Evidence: I-024 to I-028.
4. **Unresolved points and the read-only inputs that resolve them:**
   * The b4926a7 claim: `git log --oneline 90e2498..c060f06 -- pyproject.toml`.
   * The I-008 test scope: `Get-ChildItem tests -Recurse -Filter *.py | Select-String KernelDistribution`.
   * The runner target: `Get-Content scripts\tests\run_all_tests_solutions_credit_var.py`.
5. **[P]** placeholders are not supplied.

## Next prompt

Owner action: save this answer as `docs/investigations/INVESTIGATION_CREDIT_VAR_R2.md`. Then reply "CONTINUE DRAFT" in this chat, or send the prompt below to a fresh chat.

```
MODE: investigation
TASK: Write the HEAVY spec docs/workstreams/WS_CREDIT_VAR_API.md per WS_TEMPLATE.md
      Part 2 from round-2 findings. Do not re-investigate.
ATTACH: INVESTIGATOR.md ; WS_TEMPLATE.md ; CLAUDE.md ; DECISIONS.md ;
        INVESTIGATION_CREDIT_VAR_R2.md ; context_riskquant_index.txt ;
        context_riskquant_solutions_credit_var.txt ; context_riskquant_solutions_run_identity.txt ;
        context_riskquant_tests_credit.txt ; pyproject.toml
DECISIONS: D-C15, D-C04, D-C03, D-C01, D040, D041, D039, D104
WS: none (creates the spec, Status DRAFT)
FACTS: I-001 to I-040 of INVESTIGATION_CREDIT_VAR_R2.md, with their labels.
VERIFY FIRST: I-013, I-014, I-015, I-016, I-019 evaluator column, I-021, I-033 raise
       sites, I-034, I-038.
SCOPE: slices S1 to S6 of R2 Option A. O-1 to O-3 in force as given; O-4 to O-13 OPEN.
       Constraints: S1 holds all CreditVarResult and RunProvenance changes under one
       schema version and comes first; target shapes of CreditVarRequest and
       CreditVarResult as Python types and JSON, field by field; S1 acceptance has a
       model_dump_json -> model_validate_json lossless round-trip test; S6 is script
       only, no src change, no unmeasured number; FULL review for src or oracle
       slices, QUICK for S6 only.
OUT OF SCOPE: code; consumer application design; solutions.credit_exposure; O-3 IDs.
ACCEPTANCE: template 1.7 self-check reported; file presented; no D number assigned.
GATES: none (analysis only).
REPORT: SELF-CHECK line, then INVESTIGATOR.md section 8 order.
STOP AND ASK: any O-n gate that the spec cannot avoid deciding.
```

Queue after the draft: freeze audit (MODE: audit, fresh chat), then S1 to S6.

MODE: PLAN CHANGE
SNAPSHOT: `SNAPSHOT commit=c060f06 dirty=false content_hash=fa836966c1293f33`
STATUS: ANSWERED

## Answer

The HEAVY spec draft `WS_CREDIT_VAR_API.md` is delivered, with two appendices: `A_CONTRACT.md` (target shapes of the request and result, field by field, as Python types and JSON) and `A_ORACLES.md` (one independent oracle per catalog key). The draft has 6 slices and 32 contract clauses. It proposes the D-entry text P-CREDIT_VAR_API-001 and keeps 10 owner gates OPEN.

SELF-CHECK: corrections made.
- S1.T15 had no clause. It now belongs to C.13.
- The quantifier sets of C.1 (float fields, including float values inside dicts) and C.20 (the public entry points) are now stated.
- One invented status marker was removed from S6.
- Deviation: the core spec has 437 lines, which is above the 300-line guide. The facts table (33 rows) and the slice test tables cause the excess. Each slice stays under 40 lines.

## Evidence map

```
WS_CREDIT_VAR_API.md
 S1 result contract  -> schemas.py (result models), run.py, new errors.py, new provenance.py
                        C.1-C.13; A_CONTRACT §3, §4; depends O-5, O-9
 S2 request + errors -> schemas.py (request models), run.py, errors.py, [recipes.py]
                        C.14-C.21; depends S1, O-7, O-8, O-9, O-12
 S3 concurrency      -> catalog.py, recipes.py, [run.py]; C.22-C.25; depends S2, O-10
 S4 oracles          -> tests only; C.26-C.27; A_ORACLES; depends S1
 S5 stability        -> api.py, pyproject.toml, credit/STABLE_API.md (O-4); C.28-C.31
 S6 benchmark        -> scripts/solutions/credit_var/bench_credit_var_api.py; C.32; QUICK
Prohibited in all slices: src/riskquant/credit/** (except STABLE_API.md in S5), jit.py, DECISIONS.md
```

## Findings

Findings I-001 to I-040 are in the round-2 answer. The draft carries them as F-1 to F-33. These are the new findings of this turn.

**I-041 | Observed | run.py:_build_loss_matrix, credit/aggregate/summary.py:model_tail_var_es**
Fact: The keys of `per_model_tail` are JtdModel names, for example "FORCED_DEFAULT". The force_default overlay adds "forced_default", which is a second key for the same concept.
Consequence: The draft adds clause C.7: one key vocabulary, with the overlay merged into "FORCED_DEFAULT" [Proposed].

**I-042 | Observed | credit/jtd/ingest/validate.py**
Fact: The engine rejects a duplicate (obligor_uid, factor_name) at about line 600. It rejects a per-obligor sum of loading^2 > 1 + 1e-9 at lines 611 to 614.
Consequence: The raise sites of round-2 I-033 are now confirmed. C.17 copies the 1e-9 tolerance, so the request rule and the engine rule cannot drift.

**I-043 | Observed (sandbox: pydantic 2.14.1, numpy 2.4.4) | library behavior**
Fact:
- `ge=0` accepts +inf.
- NaN dumps to JSON null, and the round-trip then fails with `float_type`.
- Lax mode coerces "100" and "7".
- An int field accepts 2**70.
- `frozen=True` leaves nested lists and dicts mutable.
- `allow_inf_nan=False` rejects NaN.
- `ValidationError` subclasses ValueError.
- `default_rng(-1)` raises ValueError.
- `0.0/0.0` raises ZeroDivisionError.

Consequence: I-034 and parts of I-007 and I-033 are now Observed, with the sandbox scope stated.

**I-044 | Observed | credit/factors/simulate/base.py, credit/exposure/obligor_ead/align.py, mixed.py**
Fact: The D-C04 coverage calls, the D-C03 `searchsorted_or_raise` alignment and the D-C15 single-consumer construction were re-read at c060f06. I-015 (`jtd/bridge.py`) is not on the credit_var path and was not re-read.
Consequence: Round-1 leads I-013, I-014 and I-016 hold.

**I-045 | Inferred | credit/aggregate/loss_metrics.py docstring**
Fact: The docstring defines VaR as the order statistic at `floor((1 - alpha) * S)`. This is a claim, not read behavior.
Consequence: A_ORACLES uses the population definitions with an atom-gap rule, so the engine's index convention cannot decide a test.

## Options

Option A, as drafted, is the recommendation. Options B and C stay as stated in round 2.

## Value assessment

```
VALUE
Benefit:        one typed, versioned, fail-loud entry point; D-C15 release condition met.
Cost proxies:   6 slices; schemas.py, run.py, catalog.py, recipes.py, api.py, pyproject.toml,
                2 new modules; 1 wire contract (schema 1.0); 1 P- entry; 10 OPEN gates.
Alternative:    S1 + S6 only. Fails G on the error split, concurrency and the declared contract.
Not doing it:   level collisions, NaN, ZeroDivisionError, mixed error types, unproven threads.
Case against:   11 of 13 keys lack a public-path oracle (F-25); S5 is gated on S4 VALIDATED.
Reversibility:  costly after a consumer pins 0.2.0.
Parity:         n/a
Owner inputs:   [P] priority, target date, related work (placeholders).
Class:          HEAVY (public signature, serialization identity, validation, new D entry)
Recommendation: PROCEED [Proposed]
```

## Open items

1. **Owner gates O-4 to O-13 are OPEN.** The questions and options are in spec section 5. The owner decides each gate before the freeze. The audit can run while they are open.
2. **Owner action:** commit the three files under `docs/workstreams/`. Save the round-2 answer as `docs/investigations/INVESTIGATION_CREDIT_VAR_R2.md`.
3. **Unresolved points and the read-only inputs that resolve them:**
   - The b4926a7 claim: `git log --oneline 90e2498..c060f06 -- pyproject.toml`.
   - The KernelDistribution scope: `Get-ChildItem tests -Recurse -Filter *.py | Select-String KernelDistribution`.
   - The runner target: `Get-Content scripts\tests\run_all_tests_solutions_credit_var.py`.
4. **[P] placeholders** are not supplied.

## Next prompt

```
MODE: audit
TASK: Freeze audit, round 1 of 2, of docs/workstreams/WS_CREDIT_VAR_API.md revision 1,
      per WS_TEMPLATE.md Part 3 and section 1.6. Write
      docs/workstreams/WS_CREDIT_VAR_API/R_FREEZE_RECONCILIATION_R1.md.
ATTACH: docs/workstreams/WS_CREDIT_VAR_API.md ;
        docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ;
        docs/workstreams/WS_CREDIT_VAR_API/A_ORACLES.md ;
        WS_TEMPLATE.md ; CLAUDE.md ; DECISIONS.md ; pyproject.toml ;
        context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ;
        context_riskquant_credit.txt ; context_riskquant_solutions_run_identity.txt ;
        context_riskquant_tests_credit.txt
DECISIONS: D-C15, D-C04, D-C03, D039, D040, D041, D104
WS: docs/workstreams/WS_CREDIT_VAR_API.md (Status DRAFT; audit may read it)
VERIFY FIRST: the Student-t scaling and default direction in each JTD kernel (A_ORACLES
       K9, K12); F-18 and F-30 are Inferred.
SCOPE: clauses C.1 to C.32, P-CREDIT_VAR_API-001, gates O-1 to O-13, slices S1 to S6.
       Check each clause against the c060f06 profiles and DECISIONS.md. Check P-001
       clause 5 against D-C15 and O-2. Check the A_ORACLES formulas against the kernels.
OUT OF SCOPE: any code change; consumer application design; solutions.credit_exposure;
       audit IDs under O-3.
ACCEPTANCE: Part 3 sections A to G complete; each finding BLOCKING or REFINEMENT;
       Clauses covered and Clauses not covered stated; every negative claim scoped.
GATES: none (analysis only).
REPORT: final report per CLAUDE.md section 11
STOP AND ASK: a clause that cannot be judged without an OPEN owner gate.
```

Queue after the audit: owner closes O-4 to O-13 and sets the status. Then S1, S2, S3 and S5 run in order, S4 runs after S1, and S6 runs at any time.
