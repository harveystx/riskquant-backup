# WS_CREDIT_VAR_API: Stable public entry point for credit VaR

Path: docs/workstreams/WS_CREDIT_VAR_API.md
Class: HEAVY
Status: REVIEW
Revision: 1
Revision note: First draft from the credit VaR readiness investigation, rounds 1 and 2.
Date: 2026-10-29
Branch: ws/credit-var-api
Snapshot: SNAPSHOT commit=c060f06 dirty=false content_hash=fa836966c1293f33
Drafted by: investigator
Independent review: PENDING
Audit rounds used: 0 of 2
Code merged: no
DECISIONS.md reconciled: NO
Evidence limits: No gate, test or benchmark ran. Code facts come from the c060f06 profiles. Test facts come from a filtered extract of the 90e2498 tests profile (18 files, all under tests/credit/). tests/ did not change to c060f06 (snapshot_delta.txt). Facts marked "sandbox" ran in the investigator sandbox with pydantic 2.14.1 and numpy 2.4.4, not on the owner machine. Not read: credit/jtd/evaluators/*, credit/jtd/bridge.py, scripts/tests/run_all_tests_solutions_credit_var.py, scripts/solutions/credit_var/demo_e2e_*.py, and the Emissary and riskquant-excel repositories.

## 1. Goal and scope

Goal: Make `riskquant.solutions.credit_var.api` the single public entry point for credit VaR models, correct and usable by any consumer in one Python process.

Required properties (owner goal, round 2):
- a typed request and result, with a lossless JSON form;
- fail-loud validation, with errors that separate a bad request from an engine fault;
- a declared, versioned stability contract;
- reproducible runs under D-C15;
- safe concurrent calls in one process.

Value case:
- Benefit: One contract that a notebook, a terminal session, a service or another project can pin. The D-C15 release condition becomes true (F-8, F-9). Silent level collisions (F-4), NaN and ZeroDivisionError on the result (F-6, F-7), and mixed error types (F-14, F-15) stop.
- Cost proxies: 6 slices. Changed src files: schemas.py, run.py, catalog.py, recipes.py, api.py, pyproject.toml, plus 2 new modules. 1 wire contract (CreditVarResult, schema 1.0). 1 new D entry (P-CREDIT_VAR_API-001). 10 open owner gates. At most 2 audit rounds.
- Smallest alternative and why it fails: S1 and S6 only. It leaves the request/engine error split (F-12, F-14), the mutable process state (F-2, F-16) and the declared contract open. 3 of the 5 required properties fail.
- Case against: 11 of 13 catalog keys have no public-path oracle, and no key has a VaR or ES oracle (F-25). A stable declaration before S4 freezes unverified numbers. Mitigation: S5 depends on S4 VALIDATED.
- Recommendation at drafting: PROCEED [Proposed]

In scope:
- CreditVarResult, RunProvenance and the result schema version (S1).
- CreditVarRequest admission rules and the public error types (S2).
- Process-global state and concurrent calls (S3).
- One independent-oracle test per catalog key through `api` (S4).
- The stability declaration, the surface test and the dependency pins (S5).
- A benchmark script (S6).

Out of scope:
- The design of any consumer application: web framework, task queue, persistence, authentication, UI.
- `solutions.credit_exposure`. `credit_var` does not import it.
- The audit IDs C-04, C-07, RQ-AUD-009 and P2-07 (O-3).
- Any change in `src/riskquant/credit/**`, except `src/riskquant/credit/STABLE_API.md` under O-4.
- The text of D-C15 (O-2).
- Removal of pandas from the import path of `api` (F-23), except as O-7 requires.

Owner decisions in force:
- 2026-10-29 (O-1): Each result dict keyed by `_alpha_key` or `_percentile_key` becomes a list of records that carry the level as a float, sorted by ascending level. Request validation rejects duplicate alphas. The engine does not deduplicate. A lookup by level raises for a level that the run did not compute. A display label is not data.
- 2026-10-29 (O-2): API version is the package version, SemVer over the surface that a new D entry declares stable. Before 1.0.0, a minor bump marks a breaking change. CreditVarResult carries a result schema version: major on a breaking change, minor on an added optional field. `RunProvenance.engine_version` identifies the exact build; the spec chooses the mechanism. NumPy and SciPy keep ranges with an upper bound on the next major version. Consuming environments lock exact versions. RunProvenance records the NumPy, SciPy and Numba versions. D-C15 text does not change.
- 2026-10-29 (O-3): C-04, C-07, RQ-AUD-009 and P2-07 are deferred to a separate triage item.

Depends on: none.
Blocks: consumer releases that pin `credit_var.api`. `api.py` names the notebook, Emissary and the xlwings tool as consumers (F-31).

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | `api.__all__` holds 19 names: MODEL_CATALOG, CreditVarRequest, CreditVarResult, GroupRow, LoadingRow, LossHistogram, ModelEntry, ModelTail, ObligorContribution, ObligorRow, ParamSpec, RiskSummary, RunProvenance, StressTestObligor, StressTestResult, model_entry, run_credit_var, simulate_portfolio_losses, supported_model_keys. | `solutions/credit_var/api.py:__all__` | Observed |
| F-2 | CreditVarRequest uses `ConfigDict(extra="forbid", protected_namespaces=())`: not frozen, no `validate_assignment`. `run_credit_var` uses an instance as given and validates only a dict. | `schemas.py:_STRICT`; `run.py:run_credit_var` | Observed |
| F-3 | `frozen=True` does not freeze a nested list or dict. `allow_inf_nan=False` rejects NaN (`finite_number`). Tuple fields round-trip through JSON. | sandbox, pydantic 2.14.1 | Observed |
| F-4 | `summary.var`, `es`, `economic_capital_var`, `economic_capital_es` use `_alpha_key` keys (basis points). 0.9999 and 0.99991 give one key. alpha >= 0.99995 gives "100.00%". `loss_percentiles` uses `_percentile_key`. | `run.py:_alpha_key`, `_percentile_key`; round-1 sandbox run | Observed |
| F-5 | `RiskSummary.max_alpha` applies `float()` to keys such as "99.90%" and raises ValueError on every result. | `schemas.py:RiskSummary.max_alpha` | Observed |
| F-6 | `RiskSummary.es_to_var_ratio` divides Python floats: a zero VaR raises ZeroDivisionError. `ModelTail.es_to_var_ratio` returns NaN for `var == 0`. `get_var` and `get_es` raise ValueError first, so their NaN default is unreachable. | `schemas.py`; sandbox `0.0/0.0` | Observed |
| F-7 | `ObligorContribution.ead` defaults to `float("nan")` for a uid absent from `ead_by_uid`. | `run.py:run_credit_var` | Observed |
| F-8 | RunProvenance fields: engine_version, model, params, n_obligors, n_scenarios, seed, alphas, input_hash, generated_at. No construction identifier and no NumPy, SciPy or Numba version. | `schemas.py:RunProvenance`; DECISIONS.md D-C15 | Observed |
| F-9 | `_engine_version` returns `importlib.metadata.version("riskquant")` (static "0.1.0") and returns "0+unknown" on any Exception. | `run.py:_engine_version`; `pyproject.toml` | Observed |
| F-10 | `input_hash` is the first 16 hex characters of SHA-256 over `json.dumps(model_dump(mode="json"), sort_keys=True)`. `generated_at` is a wall-clock UTC timestamp. | `run.py:_input_hash`, `run_credit_var` | Observed |
| F-11 | `per_model_tail` and `expected_loss_by_model` use JtdModel names, for example "FORCED_DEFAULT". The force_default overlay adds a second key, "forced_default", for the same concept. | `credit/aggregate/summary.py:model_tail_var_es`; `run.py:_build_loss_matrix` | Observed |
| F-12 | These request faults pass validation and fail later in the engine as ValueError: duplicate (uid, factor) loadings and a per-obligor sum of loading^2 > 1 + 1e-9 (`credit/jtd/ingest/validate.py`); seed < 0 (`default_rng` in `mixed.py:iter_batches`, sandbox); a uid outside int64 (`recipes._uids` casts to int64; the int field accepts 2**70, sandbox). | as stated | Observed |
| F-13 | `ObligorRow.ead` (`ge=0.0`) accepts +inf. A NaN float dumps to JSON null and then fails `model_validate_json`. Lax mode turns "100" into 100.0 and "7" into 7. | sandbox, pydantic 2.14.1 | Observed |
| F-14 | `pydantic.ValidationError` subclasses ValueError. | sandbox | Observed |
| F-15 | On the path, ValueError carries both caller faults and engine invariant faults: 116 raise sites in credit, 47 in credit_var, 11 in scenario_metrics, 7 in `_arrays`. No credit_var error type exists. | AST count of raise statements | Observed |
| F-16 | `MODEL_CATALOG` (re-exported by `api`) and `recipes.SUPPORTED_RECIPES` are plain module-level dicts. | `catalog.py`; `recipes.py` | Observed |
| F-17 | All 9 JTD kernels use `parallel_njit`. The 2 tail kernels use `stable_math_parallel_njit`. All set `nogil=True`, `cache=True`. No profile and no decision sets a Numba threading layer. | `credit/jtd/kernels/*`; `scenario_metrics/_kernels/_tail_metrics.py`; `jit.py`; grep `threading_layer` = 0 hits | Observed |
| F-18 | Concurrent calls from several threads enter parallel Numba regions at the same time. Safety depends on the threading layer that Numba selects at runtime. | F-17; Numba documentation | Inferred |
| F-19 | The D-C15 construction holds: one `standard_normal` fill per batch into a C-order buffer, `default_rng(seed)` per `iter_batches` call, PIT through `ndtr`, `gammaincinv`, `betaincinv`. | `credit/factors/simulate/mixed.py:iter_batches` | Observed |
| F-20 | D-C04 holds on the path: `build_credit_factor_matrix` calls `new_scenario_coverage`, `record_scenario_span` per batch, and `assert_scenario_coverage_complete`. | `credit/factors/simulate/base.py` | Observed |
| F-21 | D-C03 holds on the path: EAD alignment uses `align_obligor_ead_to_portfolio` with `searchsorted_or_raise`. | `credit/exposure/obligor_ead/align.py` | Observed |
| F-22 | `[project].dependencies` holds pydantic>=2.0, numpy>=1.26, scipy>=1.15, numba>=0.59, with no upper bound. `version = "0.1.0"`. D104.3 still gives "pydantic is not declared" as a reason. | `pyproject.toml`; DECISIONS.md D104 | Observed |
| F-23 | Import of `api` loads numpy, pandas, scipy, numba and pydantic at module level. 18 modules on the closure import pandas. `engine.py:JTD_EVALUATORS` imports all 9 evaluators and kernels at import. | AST closure; `credit/jtd/engine.py`; `credit/jtd/registry.py` | Observed |
| F-24 | 7 test functions in 2 files call `run_credit_var`, all through `credit_var.run`, none through `api`. No test calls `simulate_portfolio_losses`. `tests/credit/factors/test_batch_size_reproducibility.py` reads `dict(summary.var)`. | tests extract | Observed |
| F-25 | A public-path independent oracle exists only for beta_binomial and beta_binomial_grouped, and only for expected loss. No key has a public-path VaR or ES oracle. | `tests/credit/jtd/test_beta_binomial_contract.py`; tests extract | Observed |
| F-26 | `tests/solutions/credit_var/` exists and holds no file. `scripts/tests/run_all_tests_solutions_credit_var.py` exists; its text is in no profile. | `project_structure.txt` | Observed |
| F-27 | Reusable machinery: `RunManifest.canonical_bytes` and `_canonicalize` (float.hex, sorted keys, compact separators, D040); the `MANIFEST_SCHEMA_VERSION` literal; `riskquant.exceptions.RiskQuantError`; `factor_regression/errors.py` (classes first, mapping later). | `solutions/run_identity/schemas.py`; `exceptions.py`; `solutions/factor_regression/errors.py` | Observed |
| F-28 | `STABLE_API.md` covers `riskquant.credit`, not `credit_var.api`. 27 of 28 listed names exist at the stated path; CreditFactorKind is listed under the wrong path (`eums`). 29 `__all__` names are not listed. It cites D-C18, which DECISIONS.md lacks. | R2 findings I-024 to I-028 | Observed |
| F-29 | `simulate_portfolio_losses` returns `np.ndarray`. Above `max_samples` it subsamples with `default_rng(req.seed).choice`. | `run.py:simulate_portfolio_losses` | Observed |
| F-30 | VaR is the order statistic at `floor((1 - alpha) * S)`. ES is the mean beyond it, with a fractional weight. | docstrings of `credit/aggregate/loss_metrics.py` and `_attribution.py` | Inferred |
| F-31 | `api.py` names the notebook, Emissary and the xlwings tool as consumers. | `api.py` docstring | Observed |
| F-32 | 15 demo scripts exist in `scripts/solutions/credit_var/`. Their text is in no profile. | `project_structure.txt` | Observed |
| F-33 | `from_dataframe` requires `obligor_uid`; its docstring says `uid`. Its default n_scenarios is 100_000; the request default is 250_000. | `schemas.py:CreditVarRequest.from_dataframe` | Observed |

## 3. Contract

Field-level target shapes are in `A_CONTRACT.md`. Oracle definitions are in `A_ORACLES.md`.

Result (S1):
- C.1 The result models (CreditVarResult, RiskSummary, AlphaRisk, PercentileLoss, ModelTail, ObligorContribution, LossHistogram, StressTestObligor, StressTestResult, RunProvenance) set `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`. Their sequence fields are tuples. A NaN or infinite value in a float field, including a float value inside a dict field, raises at construction. Int, bool and str fields are outside this clause. [Proposed] Test: S1.T1
- C.2 `CreditVarResult.result_schema_version` is `Literal["1.0"]`. Validation of a payload with another value raises. [Proposed] Test: S1.T2
- C.3 No result field is a dict keyed by a level string. `RiskSummary.by_alpha` holds one AlphaRisk per requested alpha, ascending. Each `alpha` equals the request float bit for bit. `loss_percentiles` holds one PercentileLoss per grid point, ascending. [Proposed] Test: S1.T3, S1.T4
- C.4 `RiskSummary.at(alpha)` returns the record with `record.alpha == alpha`. For any other value it raises the lookup error that O-9 names. No result method returns NaN. `max_alpha`, `get_var` and `get_es` are removed. [Proposed] Test: S1.T5
- C.5 The ES/VaR ratio follows O-5. [Proposed] Test: S1.T6
- C.6 `obligor_contributions` holds one record per request obligor, ascending by obligor_uid. `ead` equals the request ead. A uid without an ead raises the engine error type (O-9). `stress_test.obligors` is ascending by obligor_uid. [Proposed] Test: S1.T7
- C.7 The keys of `per_model_tail` and `summary.expected_loss_by_model` are JtdModel member names. The forced-default overlay uses "FORCED_DEFAULT". [Proposed] Test: S1.T8
- C.8 RunProvenance holds the fields of A_CONTRACT section 3.10. `numpy_version`, `scipy_version` and `numba_version` equal the runtime `__version__` values. [Proposed] Test: S1.T9
- C.9 `engine_version` follows P-CREDIT_VAR_API-001 clause 5. Missing metadata, or a distribution version with a local segment, raises the engine error type. [Proposed] Test: S1.T10
- C.10 `input_hash` is the 64-character lowercase SHA-256 hex of the canonical request bytes (A_CONTRACT section 4.2). Two requests that validate to equal models give one hash. A 1-ulp change of any float gives another hash. [Proposed] Test: S1.T11
- C.11 For one result per catalog key, `model_validate_json(r.model_dump_json()) == r`, and the second dump is byte-equal to the first. [Proposed] Test: S1.T12
- C.12 The SHA-256 of the LF-normalized source of `credit/factors/simulate/mixed.py` equals the value recorded beside `SCENARIO_CONSTRUCTION_ID`. [Proposed] Test: S1.T13
- C.13 (D-C15) The same request, run twice in one process and once in a fresh subprocess, gives byte-equal `model_dump_json` output after removal of `provenance.generated_at`. The result stays bitwise invariant to `_FACTOR_BATCH`. [Proposed] Test: S1.T14, S1.T15

Request and errors (S2):
- C.14 The request models set `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`; their sequence fields are tuples. A request round-trips through JSON with no loss. [Proposed] Test: S2.T1
- C.15 `run_credit_var` validates the request state at call time. A request that in-place mutation made invalid raises the request error type before any factor generation. The same applies to `simulate_portfolio_losses` if O-6 keeps it. [Proposed] Test: S2.T2
- C.16 `alphas` is non-empty, each value is in (0, 1), and no two values are equal under `==`. [Proposed] Test: S2.T3
- C.17 Request validation rejects: seed < 0 (D039 precedent); obligor_uid outside the int64 range; a duplicate (obligor_uid, factor) loading; a per-obligor sum of loading^2 > 1 + 1e-9, the engine tolerance. A user factor name equal to a recipe-generated factor name is rejected, or isolated so that the result equals the result with a renamed factor. [Proposed] Test: S2.T4, S2.T5, S2.T6, S2.T7
- C.18 Coercion follows O-8. [Proposed] Test: S2.T8
- C.19 Size limits follow O-12. [Proposed] Test: S2.T9
- C.20 A public entry point is `run_credit_var` and, if O-6 keeps it, `simulate_portfolio_losses`. Every exception of type Exception that a public entry point raises is the request error type R or the engine error type E (O-9). Direct construction of a request model raises `pydantic.ValidationError` under both O-9 options. R is raised before any engine call and carries a cell-addressable loc. E chains the original exception as `__cause__` and does not subclass ValueError. [Proposed] Test: S2.T10, S2.T11
- C.21 If O-7 keeps `from_dataframe`, its docstring names the columns that the code requires, and its default n_scenarios equals the request default. [Proposed] Test: S2.T12

Concurrency (S3):
- C.22 `MODEL_CATALOG` and `SUPPORTED_RECIPES` are read-only mappings. Item assignment raises TypeError. [Proposed] Test: S3.T1
- C.23 Concurrent calls on 4 threads give results byte-equal to serial runs (without `generated_at`), and the process does not abort. [Proposed] Test: S3.T2
- C.24 Results with `NUMBA_NUM_THREADS=1` and with the default are byte-equal. [Proposed] Test: S3.T3
- C.25 The threading policy of O-10 holds. [Proposed] Test: S3.T4

Oracles (S4):
- C.26 For each of the 13 MODEL_CATALOG keys, one test through `api` compares EL, VaR and ES at two or more levels with the independent oracle of A_ORACLES.md. [Proposed] Test: S4.T1 to S4.T13
- C.27 The oracle tests import only from `riskquant.solutions.credit_var.api`, numpy, scipy, pytest and the standard library. [Proposed] Test: S4.T14

Stability (S5):
- C.28 `api.__all__` equals the frozen surface list after O-6 and O-7. [Proposed] Test: S5.T1
- C.29 `model_json_schema()` of CreditVarRequest and CreditVarResult equals the golden files. A change needs a new golden file and a version decision under P-CREDIT_VAR_API-001. [Proposed] Test: S5.T2
- C.30 `pyproject.toml`: version "0.2.0"; numpy>=1.26,<3; scipy>=1.15,<2; pydantic and numba per O-11. [Proposed] Test: S5.T3
- C.31 `STABLE_API.md` follows O-4. [Proposed] Test: S5.T4

Benchmark (S6):
- C.32 The benchmark script produces every measurement of S6 with its environment record. No `src/` file changes. [Proposed] Test: S6 gate (script run)

## 4. Decisions

Decisions relied on: D-C15, D-C04, D-C03, D039, D040, D041, D104. Each checked against DECISIONS.md: yes.

D104.3 gives "pydantic is not declared in pyproject.toml" as one reason. That reason is stale at c060f06 (F-22). This workstream does not change D104. The owner decides whether D104 needs a note.

### P-CREDIT_VAR_API-001: Stable public surface of riskquant.solutions.credit_var.api (PROPOSED)

```text
Related: D001, D002, D-C15, D039, D040, D041, D104
Scope: the public surface, the versioning, the build identity and the error and
concurrency contract of riskquant.solutions.credit_var.api.
1. Surface. The stable surface is the name list of api.__all__ that S5.T1 freezes.
   A consumer imports only from riskquant.solutions.credit_var.api. All other
   modules of riskquant.solutions.credit_var and riskquant.credit are internal.
2. Version. The package version follows SemVer over the surface of clause 1.
   Before 1.0.0, a minor bump marks a breaking change; a patch bump marks a
   compatible change.
3. Breaking change. Each item is breaking: removal or rename of a surface name;
   an incompatible signature change; a request schema change that rejects a
   payload that was valid; a result schema change other than an added optional
   field; a change of a published error type; a change of a result value at a
   fixed request and a fixed numerical environment, unless a D entry records it
   as a defect fix.
4. Result schema version. CreditVarResult.result_schema_version is "MAJOR.MINOR".
   MAJOR increments on a breaking change of the serialized form. MINOR increments
   on an added optional field. A reader validates only the version it declares.
5. Build identity. RunProvenance.engine_version is "<V>+src.<H>". V is the
   distribution version from package metadata and has no local segment. H is
   the lowercase SHA-256 hex over each *.py file under the directory of the
   imported riskquant package, ordered by POSIX relative path. Each file
   contributes its UTF-8 path, one NUL byte, its content with CRLF replaced by
   LF, and one NUL byte. Missing metadata raises. No fallback string exists.
6. Numerical environment. RunProvenance records scenario_construction_id,
   numpy_version, scipy_version and numba_version. Reproducibility stays
   environment-relative as D-C15 states. The text of D-C15 does not change.
7. Pins. pyproject.toml keeps NumPy and SciPy ranges with an upper bound on the
   next major version. Each consuming environment locks exact versions.
8. Errors. The public entry points raise only the request error type and the
   engine error type that this decision names (O-9).
9. Concurrency. Calls on separate threads of one process give the results of
   serial calls under the threading policy that this decision names (O-10).
Evidence: S1.T1 to S1.T15, S2.T1 to S2.T12, S3.T1 to S3.T4, S4.T1 to S4.T14,
S5.T1 to S5.T4.
```

Clause 5 alternatives that the draft rejects:
- hatch-vcs dynamic version: the owner tags are `prod_YYYYMMDD[_HHMM]`, which are not PEP 440 version tags, and an editable install fixes the version at install time. [Inferred]
- Package metadata version only: static "0.1.0" identifies no build (F-9).

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Result level keys (option B) | CLOSED | 2026-10-29 | Owner, round-2 prompt |
| O-2 | API version, schema version, build identity, pins | CLOSED | 2026-10-29 | Owner, round-2 prompt; mechanism in P-001 clause 5 |
| O-3 | Audit IDs C-04, C-07, RQ-AUD-009, P2-07: DEFER | CLOSED | 2026-10-29 | Owner, round-2 prompt |
| O-4 | STABLE_API.md: (a) keep as an internal contract of riskquant.credit and correct it to match `__all__`; (b) merge its model-parameter contracts into the new D entry and delete it; (c) delete it. | OPEN | | F-28 |
| O-5 | ES/VaR ratio at VaR = 0: (a) remove both ratio members; (b) keep a method that raises the lookup error at VaR = 0; (c) add an optional field that is absent at VaR = 0. | OPEN | | F-6 |
| O-6 | `simulate_portfolio_losses`: (a) remove from `api`; (b) keep as a Python-only member, exempt from the JSON property; (c) return a typed model. | OPEN | | F-29 |
| O-7 | `from_dataframe` (pandas-typed signature): (a) keep in the stable surface; (b) keep but outside the stable surface; (c) remove. | OPEN | | F-23, F-33 |
| O-8 | Request coercion: (a) strict, no string-to-number conversion; (b) lax, as today. | OPEN | | F-13 |
| O-9 | Error taxonomy: (a) R = `pydantic.ValidationError`, E = `CreditVarEngineError(RiskQuantError)`; (b) R = `CreditVarRequestError(RiskQuantError)` that carries the ValidationError details, E = `CreditVarEngineError`, both under `CreditVarError`. Also: the type of the `at()` lookup error. | OPEN | | F-14, F-15 |
| O-10 | Threading policy: (a) require a thread-safe Numba threading layer and fail loud without one; (b) serialize parallel-kernel entry in credit_var with a process lock; (c) declare one call at a time per process. | OPEN | | F-17, F-18 |
| O-11 | Upper-bound rule for the pydantic and numba pins. | OPEN | | F-22 |
| O-12 | Request size limits: (a) none; (b) a maximum for n_scenarios and for n_obligors x n_scenarios, with values from S6. | OPEN | | F-23; R1 I-021 |
| O-13 | Do C-15, C-18, C-07A, P2-09 and D-C18 join the O-3 triage item? | OPEN | | R2 I-039 |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1, then S2, then S3, then S5. S4 starts after S1 and may run in parallel with S2 and S3. S6 may run in parallel with any slice. S5 starts after S1 to S4 are VALIDATED.

Review depth: FULL for S1 to S5 (src contract changes or independent oracles). QUICK for S6 only.

Gate block (PowerShell, repository root, `WORKFLOW.md` section 6). Each slice sets `$f` to its Files and `$t` to its test paths:

```powershell
& {
  "== ruff check";          ruff check $f;          "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f; "exit=$LASTEXITCODE"
  "== mypy";                mypy $f;                "exit=$LASTEXITCODE"
  "== pytest slice";        pytest $t -q;           "exit=$LASTEXITCODE"
  "== pytest full";         pytest -q;              "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

### S1: Result contract and provenance

Objective: Apply every CreditVarResult and RunProvenance change under result schema version 1.0.
Contract: C.1 to C.13
Depends on: O-5, O-9 (type names only; S1 defines the classes, S2 applies the mapping)
Files:
- create: src/riskquant/solutions/credit_var/errors.py; src/riskquant/solutions/credit_var/provenance.py; tests/solutions/credit_var/test_result_contract.py; tests/solutions/credit_var/test_reproducibility.py
- modify: src/riskquant/solutions/credit_var/schemas.py; src/riskquant/solutions/credit_var/run.py; tests/credit/factors/test_batch_size_reproducibility.py
- modify, if the FIRST step finds a reader of a removed field: scripts/solutions/credit_var/demo_e2e_*.py
FIRST: Search `src/`, `scripts/` and `tests/` for `summary.var`, `summary.es`, `economic_capital_`, `loss_percentiles`, `get_var`, `get_es`, `max_alpha`, `es_to_var_ratio`, `per_model_tail`. Report every hit. STOP AND ASK for a hit outside the Files list.
Prohibited: src/riskquant/credit/**; src/riskquant/scenario_metrics/**; src/riskquant/jit.py; src/riskquant/solutions/run_identity/**; catalog.py; recipes.py; request models; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_run_identity.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/solutions/credit_var", "tests/credit/factors/test_batch_size_reproducibility.py", "tests/credit/jtd/test_beta_binomial_contract.py"`
Exit: every S1 test passes; full suite passes; no NaN, no level-keyed dict and no "0+unknown" path remains in credit_var.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | NaN, +inf, -inf in each float field of each result model | ValidationError (finite_number) | NaN or inf reaching JSON as null (F-13) |
| S1.T2 | Result payload with result_schema_version "0.9" and "1.1" | ValidationError | silent acceptance of a foreign version |
| S1.T3 | alphas (0.999, 0.95, 0.9999, 0.99991) | 4 records, ascending, alpha.hex() equal to input | collision, reorder, rounding (F-4) |
| S1.T4 | alphas (0.99995, 0.999999) | 2 records; neither equals 1.0 | "100.00%" label (F-4) |
| S1.T5 | at(0.999), at(nextafter(0.999, 1)), at(0.5) on a run with 0.999 | record; lookup error; lookup error | rounded lookup, NaN return |
| S1.T6 | Run with VaR = 0 at 0.95 (independent_bernoulli, low PD, N = 3) | O-5 behavior; no ZeroDivisionError; no NaN | F-6 |
| S1.T7 | uids (30, 10, 20), force_default on 20; patched missing ead | uids (10, 20, 30); ead equal to request; E raised | F-7, order drift |
| S1.T8 | model forced_default; sf_gaussian with one force_default obligor | key "FORCED_DEFAULT" in both; no "forced_default" | F-11 |
| S1.T9 | Provenance versions and construction id | equal to runtime `__version__` values and the constant | F-8 |
| S1.T10 | engine_version format; patched PackageNotFoundError; patched version "0.2.0+x" | matches `^[^+]+\+src\.[0-9a-f]{64}$`; E; E | silent fallback (F-9) |
| S1.T11 | Equal requests from dict and from JSON; one ead changed by 1 ulp | equal hashes; different hashes; 64 hex | truncation, repr drift (F-10) |
| S1.T12 | Each of 13 keys (N <= 5, S = 2_000): dump, validate, dump | equal model; byte-equal JSON | lossy JSON |
| S1.T13 | LF-normalized SHA-256 of mixed.py | equals recorded value | unrecorded construction change |
| S1.T14 | One request: twice in process, once in a fresh subprocess | byte-equal JSON without generated_at | D-C15 reproducibility |
| S1.T15 | Batch-invariance test rewritten on by_alpha | bitwise equality across `_FACTOR_BATCH` values | test drift (F-24) |

Appendix: A_CONTRACT.md

### S2: Request admission and error taxonomy

Objective: Move every request-content check before the engine, freeze the request, and publish the two error types.
Contract: C.14 to C.21
Depends on: S1; O-7, O-8, O-9, O-12 (O-12 option (b) also needs S6 output)
Files:
- create: tests/solutions/credit_var/test_request_contract.py; tests/solutions/credit_var/test_errors.py
- modify: src/riskquant/solutions/credit_var/schemas.py; src/riskquant/solutions/credit_var/run.py; src/riskquant/solutions/credit_var/errors.py
- modify, only for factor-name isolation under C.17: src/riskquant/solutions/credit_var/recipes.py
Prohibited: src/riskquant/credit/**; result models; catalog.py; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/solutions/credit_var", "tests/credit/solutions"`
Exit: every S2 test passes; full suite passes; no request-content fault reaches the engine.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | Assignment to each field; request JSON round-trip | frozen_instance error; equal model | mutable request (F-2) |
| S2.T2 | Mutate `params` in place to an out-of-range value; generator patched to fail if called | R; generator not called | validation bypass (F-2, F-3) |
| S2.T3 | alphas (0.99, 0.99); (0.99, nextafter(0.99, 1)); (); (1.0,) | R; accepted; R; R | duplicate collapse (O-1) |
| S2.T4 | seed -1; seed 0 | R; accepted | late failure (F-12) |
| S2.T5 | obligor_uid 2**63; -2**63 - 1; 2**63 - 1 | R; R; accepted | overflow in the engine (F-12) |
| S2.T6 | Duplicate (uid, factor); sum loading^2 = 1 + 2e-9; = 1 + 0.5e-9 | R; R; accepted | predicate drift from the engine (F-12) |
| S2.T7 | mf_gaussian factor named "IDIO_001" and "CHI_GLOBAL" | R, or a result equal to the renamed-factor run | name collision (F-12) |
| S2.T8 | "ead": "100" in a dict and in JSON | O-8 behavior | silent coercion (F-13) |
| S2.T9 | n_scenarios and cell count above the O-12 limit | R (if a limit exists) | memory exhaustion |
| S2.T10 | Catalogue of bad requests: each rule of C.16, C.17 and each existing validator | R only; no engine call | mixed error types (F-15) |
| S2.T11 | Engine function patched to raise ValueError, TypeError, RuntimeError | E with `__cause__`; `not issubclass(E, ValueError)` | engine fault reported as a bad request |
| S2.T12 | from_dataframe docstring columns and default n_scenarios | equal to code and request default (if O-7 keeps it) | F-33 |

### S3: Process state and concurrent calls

Objective: Remove mutable process-global state and prove concurrent calls safe under the O-10 policy.
Contract: C.22 to C.25
Depends on: S2; O-10
Files:
- create: tests/solutions/credit_var/test_concurrency.py
- modify: src/riskquant/solutions/credit_var/catalog.py; src/riskquant/solutions/credit_var/recipes.py
- modify, only if O-10 chooses (a) or (b): src/riskquant/solutions/credit_var/run.py
Prohibited: src/riskquant/credit/**; src/riskquant/jit.py; schemas.py; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/solutions/credit_var", "tests/credit/solutions"`. The final report records `numba.threading_layer()`, `NUMBA_NUM_THREADS` and `os.cpu_count()` from the owner machine.
Exit: every S3 test passes on a multi-core owner machine; full suite passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | Item assignment on MODEL_CATALOG and SUPPORTED_RECIPES | TypeError | process-global mutation (F-16) |
| S3.T2 | Child process: ThreadPoolExecutor(4), 12 calls over forced_default, sf_gaussian, mf_students_t | each result byte-equal to its serial run; child exit code 0 | race or abort (F-18) |
| S3.T3 | Fresh subprocesses with NUMBA_NUM_THREADS=1 and default | byte-equal results | thread-count dependence |
| S3.T4 | O-10 policy check | O-10 behavior | unmanaged threading layer |

### S4: Independent oracles per catalog key

Objective: Verify EL, VaR and ES of each catalog key through `api` against an independent oracle.
Contract: C.26, C.27
Depends on: S1
Files:
- create: tests/solutions/credit_var/test_oracles.py; tests/solutions/credit_var/_oracle_pmf.py
Prohibited: every src/ file; existing tests.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_ORACLES.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt
VERIFY FIRST: the default direction and the Student-t scaling in each kernel. Read the kernel. Do not take them from STABLE_API.md text.
Gates: gate block; `$t = "tests/solutions/credit_var/test_oracles.py"`
Exit: 14 tests pass; a deliberate change of one model parameter in the oracle makes the matching test fail (report the output).
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 to S4.T13 | One key each, as A_ORACLES.md section 2 | VaR equal; EL and ES within the A_ORACLES bounds | wrong model math on the public path (F-25) |
| S4.T14 | AST import check of test_oracles.py and _oracle_pmf.py | imports limited per C.27 | oracle that shares production logic |

Appendix: A_ORACLES.md

### S5: Stability declaration, surface test and pins

Objective: Freeze the surface and the schemas, set the pins, and settle STABLE_API.md.
Contract: C.28 to C.31
Depends on: S1, S2, S3, S4 VALIDATED; O-4, O-6, O-7, O-11
Files:
- create: tests/solutions/credit_var/test_public_surface.py; tests/solutions/credit_var/golden/credit_var_request.schema.json; tests/solutions/credit_var/golden/credit_var_result.schema.json
- modify: src/riskquant/solutions/credit_var/api.py; pyproject.toml
- modify or delete, per O-4: src/riskquant/credit/STABLE_API.md
Prohibited: every other src/ file; DECISIONS.md. The owner promotes P-CREDIT_VAR_API-001 and assigns the D number.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt ; pyproject.toml ; src/riskquant/credit/STABLE_API.md
Gates: gate block; `$t = "tests/solutions/credit_var"`; then `pip install -e .` and `python -c "import importlib.metadata as m; print(m.version('riskquant'))"`
Exit: every S5 test passes; full suite passes; installed version prints 0.2.0.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | `api.__all__` | equals the frozen list | surface drift |
| S5.T2 | `model_json_schema()` of request and result | equals golden files | unversioned schema change |
| S5.T3 | `tomllib` parse of pyproject.toml | version, ranges per C.30 | unbounded majors (F-22) |
| S5.T4 | STABLE_API.md per O-4 | (a) listed set equals credit `__all__` union, paths exist; (b), (c) file absent | F-28 drift |

### S6: Benchmark script

Objective: Measure import cost, JIT warm-up, run time and peak memory of `api`, with no src change.
Contract: C.32
Depends on: none. Measurement 5 needs S1; before S1, omit it and state the reason in the output.
Files:
- create: scripts/solutions/credit_var/bench_credit_var_api.py
Prohibited: every src/ and tests/ file; any new dependency; any deletion of files. Numba cache isolation uses `NUMBA_CACHE_DIR` set to a fresh directory under `./.bench_credit_var/`, created by the script.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/CLAUDE.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt
Measurements (each case in a fresh child process; 5 repetitions; report min, median, max):
1. Wall time of `import riskquant.solutions.credit_var.api`: empty Numba cache, then warm cache.
2. Wall time of the first and the second `run_credit_var` call in one process (JIT warm-up).
3. Wall time and peak RSS of `run_credit_var` at n_scenarios = 250_000 for N in {10, 100, 1000} and model in {forced_default, sf_gaussian, mf_students_t}.
4. The same grid for `simulate_portfolio_losses`.
5. After S1: wall time of the first `engine_version` computation in a process.
Peak RSS: Windows `GetProcessMemoryInfo().PeakWorkingSetSize` through ctypes; POSIX `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
Book: deterministic synthetic book from a fixed seed, PD in [0.005, 0.05], LGD 0.45, lognormal EAD; mf_students_t uses 3 systemic factors with loading 0.3 each.
Record: exact command, CPU model, core count, RAM, OS, Python, NumPy, SciPy, Numba versions, `numba.threading_layer()`, NUMBA_NUM_THREADS, riskquant commit.
Gates: `$f = "scripts/solutions/credit_var/bench_credit_var_api.py"`; ruff check and ruff format --check on `$f`; then `python scripts/solutions/credit_var/bench_credit_var_api.py *>&1 | Tee-Object -FilePath bench_output.txt`
Exit: owner machine output saved as docs/workstreams/WS_CREDIT_VAR_API/R_BENCHMARK_R1.md. A sandbox run is sandbox evidence only.
Status: NOT_STARTED
Evidence: none

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| External consumers read removed keys (F-31) | Emissary, the xlwings tool and notebooks break after S1 | Release S1 to S5 together as 0.2.0; consumer repositories are out of scope |
| No installed distribution in an environment | engine_version raises where "0+unknown" was returned | Editable install in each environment; S1.T10 pins the behavior |
| Other files read removed fields (F-32) | Runtime failure outside tests | S1 FIRST step; STOP AND ASK outside the Files list |
| Guard tests fail on cosmetic edits (S1.T13, S5.T2) | Extra review steps | Intended: each failure forces a version decision |
| Threading layer differs between hosts (F-18) | S3 passes on one host and aborts on another | O-10; S3 report records the layer |
| Oracle tolerances (A_ORACLES) | A wrong model inside 6 sigma passes EL or ES | The atom-gap rule makes VaR an exact comparison |
| A caller mutates a nested dict during a call on another thread | Undefined input | Outside the contract: C.15 covers state at call time |

Deferred:
- Removal of pandas from the import path (F-23): opens when the pandas-to-Polars migration reaches credit ingest.
- Oracles for attribution fields other than forced_default: opens after S4.
- `python_version` in RunProvenance (present in D041 EngineIdentity, absent from O-2): opens on owner request.

Backlog (REFINEMENT findings and items that v1 omits):
- `KernelDistribution` removal (R1 I-008) after a search of the full tests tree.
- One-way catalog guard (R1 I-020).
- `_validate_groups` ignores a defined group that no obligor uses.
- `loss_percentiles` (np.percentile, linear) and `by_alpha.var` use different estimators at the same level.
- `run_identity.schemas._canonicalize` is private; a public helper would remove a copy.
- JavaScript consumers lose integer precision above 2**53 for seed and obligor_uid.
- `recipes._universe` writes np.nan into a frame (R1 adjacent).
- The `schemas.py` docstring shows keys such as `{"0.99": ...}`.

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.

| Item | Value |
|---|---|
| Final commit | <hash, dirty=false> |
| Release tag | <tag> |
| Decisions locked | <D-ids> |
| Independent validation | <date and snapshot> |

Delivered:
- <item>

Deferred:
- <item>

Open work: none
