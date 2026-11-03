# WS_CREDIT_VAR_API: Stable public entry point for credit VaR

Path: docs/workstreams/WS_CREDIT_VAR_API.md
Class: HEAVY
Status: FROZEN
Revision: 5
Revision note: Owner decisions of 2026-11-02 (OD-S2-1, OD-S2-2): S2 scope and A_CONTRACT 2.1 to 2.3 NUL rule. No audit round.
Date: 2026-10-29
Branch: ws/credit-var-api
Snapshot: SNAPSHOT commit=925533b dirty=false content_hash=046ed5dddb6305d3
Drafted by: investigator
Independent review: PASSED (SNAPSHOT commit=925533b dirty=false content_hash=046ed5dddb6305d3)
Audit rounds used: 2 of 2
Code merged: no
DECISIONS.md reconciled: YES (2026-10-29)
Evidence limits: No repository gate, test or benchmark ran. Code facts were read at c060f06. snapshot_delta_c060f06_925533b.txt lists only _context_gen/ and docs/ paths, so each F-n holds at 925533b. F-12, F-30, F-34, F-35 and F-36 were read or run at 925533b. Test facts come from context_riskquant_tests_credit.txt, the 925533b tests bucket (PROFILE tests_credit, TEST DIRECTORY tests/credit/, 18 files), not from the filtered extract. Facts marked "sandbox" ran in an investigator sandbox (Python 3.12.3, pydantic 2.14.1, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, numba 0.68.1, threading layer omp), not on the owner machine. The A_ORACLES revision 2 check ran the engine from a scratch copy of the 925533b profile source in that sandbox; it is sandbox evidence only. The text of src/riskquant/credit/STABLE_API.md was not attached; F-28 and R1 A-06 are the evidence for its content. Not read: credit/jtd/evaluators/*, credit/jtd/bridge.py, scripts/tests/run_all_tests_solutions_credit_var.py, scripts/solutions/credit_var/demo_e2e_*.py, tests/e2e/test_demo_scripts_are_runnable.py, and the Emissary and riskquant-excel repositories.

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
- Cost proxies: 6 slices. Changed src files: schemas.py, run.py, catalog.py, recipes.py, api.py, pyproject.toml, credit/STABLE_API.md, plus 2 new modules. 1 wire contract (CreditVarResult, schema 1.0). 1 new D entry (P-CREDIT_VAR_API-001). 0 open owner gates. At most 2 audit rounds; 1 used.
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
- The audit IDs C-04, C-07, RQ-AUD-009, P2-07, C-15, C-18, C-07A, P2-09 and D-C18 (O-3, O-13).
- Any change in `src/riskquant/credit/**`, except `src/riskquant/credit/STABLE_API.md` under O-4.
- The text of D-C15 (O-2).
- Removal of pandas from the import path of `api` (F-23).

Owner decisions in force:
- 2026-10-29 (O-1): Each result dict keyed by `_alpha_key` or `_percentile_key` becomes a list of records that carry the level as a float, sorted by ascending level. Request validation rejects duplicate alphas. The engine does not deduplicate. A lookup by level raises for a level that the run did not compute. A display label is not data.
- 2026-10-29 (O-2): API version is the package version, SemVer over the surface that a new D entry declares stable. Before 1.0.0, a minor bump marks a breaking change. CreditVarResult carries a result schema version: major on a breaking change, minor on an added optional field. `RunProvenance.engine_version` identifies the exact build; the spec chooses the mechanism. NumPy and SciPy keep ranges with an upper bound on the next major version. Consuming environments lock exact versions. RunProvenance records the NumPy, SciPy and Numba versions. D-C15 text does not change.
- 2026-10-29 (O-3): C-04, C-07, RQ-AUD-009 and P2-07 are deferred to a separate triage item.
- 2026-10-29 (O-4, option (a)): `STABLE_API.md` stays an internal contract of riskquant.credit, corrected to match `__all__`. It states internal status and makes no public, stable or versioned claim (C.31).
- 2026-10-29 (O-5, option (a)): RiskSummary and ModelTail have no ES/VaR ratio member (C.5).
- 2026-10-29 (O-6, option (b)): `simulate_portfolio_losses` stays in `api` as a Python-only member, outside the JSON property (P-CREDIT_VAR_API-001 clause 1).
- 2026-10-29 (O-7, option (b)): `CreditVarRequest.from_dataframe` stays. P-CREDIT_VAR_API-001 clause 1 exempts the member from the stable surface (C.21).
- 2026-10-29 (O-8, option (a)): Number, bool and string fields are strict. Sequence fields accept a list or a tuple and store a tuple (C.18).
- 2026-10-29 (O-9, option (a)): R = `pydantic.ValidationError`. E = `CreditVarEngineError(RiskQuantError)`. The `at()` lookup error is KeyError (C.4, C.20).
- 2026-10-29 (O-10, option (b)): One process lock serializes entry into the parallel kernels of credit_var. The lock covers parallel-kernel entry in credit_var only (C.25).
- 2026-10-29 (O-11): pydantic>=2.0,<3. numba: upper bound at the next minor above the version that S5 tests (C.30).
- 2026-10-29 (O-12, option (a)): No request size limit (C.19).
- 2026-10-29 (O-13): C-15, C-18, C-07A, P2-09 and D-C18 join the O-3 triage item.
- 2026-10-29 (R1 A-08): `input_hash` keeps the D040 canonical form. C.10 states equality as field-wise bit-equal models.
- 2026-10-29 (R1 R-05): A factor-name collision is a request error (C.17).
- 2026-10-29: Audit round 1 ran in two sessions and counts as one round.
- 2026-10-31 (TRIPWIRE, option 1): Build v1 now from the current text. Revision 3 closes S1 with no carried item. No audit round 3: the S1 closure FULL review checks the changed text against the code.
- 2026-10-31 (OD-1, option (b)): S1 enforces the constraint columns of A_CONTRACT sections 3.2 and 3.8 by validation (S1.T16).
- 2026-10-31 (OD-2): Result record order is strictly ascending (A_CONTRACT sections 3.1, 3.2, 3.7).
- 2026-10-31 (ZV): When loss_min == loss_max, loss_std_dev, loss_skewness and loss_excess_kurtosis are 0.0 (A_CONTRACT section 3.2, S1.T17).
- 2026-10-31 (U-a): demo_e2e_all_models.py gets marshall_olkin params: systemic_pd = 0.5 x min book PD, floored to 4 decimals.
- 2026-10-31 (U-b): In demo_e2e_stress_regime_bernoulli_grouped.py, the " " key is replaced by the missing accepted key.
- 2026-10-31 (U-c): Closed by decision. The credit_var demos are not regression gates. The S1 gate runs them.
- 2026-10-31 (U-d): The 15 demos `scripts/solutions/credit_var/demo_e2e_*.py` pass ruff check and mypy.
- 2026-11-02 (U-a amended): demo_e2e_all_models.py uses systemic_pd = 5e-05 for marshall_olkin. The 2026-10-31 formula gives 0.0 for this book (min PD 7.344e-05), and with 0.0 the common shock never fires. Demo value only; no contract clause changes.
- 2026-11-02 (OD-B): S1 code implements the FROZEN S1.T16 (cases 4 to 6) and S1.T17 (cases 1 and 2) rules (S1 closure review R-004 to R-007).
- 2026-11-02 (OD-A): A_CONTRACT 3.8 bin_edges is strictly ascending, as the code enforces. Revision 4 by explicit owner decision under the tripwire; editorial only.
- 2026-11-02 (OD-C): U-a uses systemic_pd = 5e-05 (confirmed).
- 2026-11-02 (ZV comparison, S1 closure review R-011): the ZV rule compares with ==, so -0.0 satisfies it. run.py emits +0.0. "0.0" in A_CONTRACT 3.2 means numeric equality, not bit equality.
- 2026-11-02 (OD-S2-1 (b)): the np.diff int64 wrap in credit/masters/obligors/validate.py and credit/masters/netting_set/validate.py is fixed in S2 (S2 review R-001). The request contract does not change. Revision 5 by explicit owner decision under the tripwire.
- 2026-11-02 (OD-S2-2 (a)): LoadingRow.factor, ObligorRow.group and GroupRow.name contain no NUL character (S2 review R-002). The fault loc is the field path (rule 1).

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
| F-12 | These request faults pass validation and fail later or pass silently. Duplicate (uid, factor) loadings and a per-obligor sum of loading^2 > 1 + 1e-9 raise ValueError (`credit/jtd/ingest/validate.py`). seed < 0 raises ValueError (`default_rng` in `mixed.py:iter_batches`, sandbox). The int field accepts any uid (2**70, sandbox). In `recipes._uids`, a uid in [2**63, 2**64 - 1] wraps silently to a negative int64 (2**63 to -2**63, 2**64 - 1 to -1); a uid above 2**64 - 1 or below -2**63 raises OverflowError, which is not a ValueError. | as stated; sandbox, pandas 3.0.2, numpy 2.4.4 | Observed |
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
| F-30 | Let m = (1 - alpha) S, snapped to the nearest integer when within 16 eps max(1, m). VaR is the loss at 0-based index k = max(0, ceil(m) - 1) of the losses in descending order. ES = (sum of the floor(m) largest losses + (m - floor(m)) VaR) / m. The `loss_metrics.py` docstring "floor((1 - alpha) * S)" is wrong for an integral m. | `scenario_metrics/_computer.py:_compute_lower_tail_parameters`; `scenario_metrics/_kernels/_tail_metrics.py:_kernel_lower_tail_var_es`; `credit/aggregate/loss_metrics.py:_loss_tail_var_es` (negation) | Observed |
| F-31 | `api.py` names the notebook, Emissary and the xlwings tool as consumers. | `api.py` docstring | Observed |
| F-32 | 15 demo scripts exist in `scripts/solutions/credit_var/`. Their text is in no profile. | `project_structure.txt` | Observed |
| F-33 | `from_dataframe` requires `obligor_uid`; its docstring says `uid`. Its default n_scenarios is 100_000; the request default is 250_000. | `schemas.py:CreditVarRequest.from_dataframe` | Observed |
| F-34 | With `strict=True`, pydantic rejects "100" for a float field; 7.0, True, "7" and numpy.int64 for an int field; 1, "true" and numpy.bool_ for a bool field; 5 for a str field. A float field accepts an int and stores a float. A tuple field with `strict=False` accepts a list, a tuple, a set and a generator, and adds a `too_short` entry when the only element fails. A list-or-tuple `BeforeValidator` on a strict tuple field rejects a set and gives one entry per element fault. `from_dataframe` casts each cell with `int`, `float` or `str`. | sandbox, pydantic 2.14.1; `schemas.py:CreditVarRequest.from_dataframe` | Observed |
| F-35 | A `ValidationError` built with `ValidationError.from_exception_data` and raised in a `model_validator(mode="after")` reaches the caller with its own loc, for example ("rows", 2, "uid"). A ValueError raised there has loc (). | sandbox, pydantic 2.14.1; R1 A-09 | Observed |
| F-36 | For mf_gaussian and mf_students_t the recipe adds one factor `IDIO_{uid:03d}` per modelled obligor; mf_students_t adds `CHI_GLOBAL`. A loadings factor with one of these names raises ValueError (duplicate factor_name) in the engine and passes request validation. | `recipes.py:_mf_universe_and_bundle`; `credit/masters/factor_universe/validate.py:validate_factor_names`; sandbox run | Observed |

## 3. Contract

Field-level target shapes are in `A_CONTRACT.md`. Oracle definitions are in `A_ORACLES.md`.

Result (S1):
- C.1 The result models (CreditVarResult, RiskSummary, AlphaRisk, PercentileLoss, ModelTail, ObligorContribution, LossHistogram, StressTestObligor, StressTestResult, RunProvenance) set `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`. Their sequence fields are tuples. A NaN or infinite value in a float field, including a float value inside a dict field, raises at construction. Int, bool and str fields are outside this clause. [Proposed] Test: S1.T1
- C.2 `CreditVarResult.result_schema_version` is `Literal["1.0"]`. Validation of a payload with another value raises. [Proposed] Test: S1.T2
- C.3 No result field is a dict keyed by a level string. `RiskSummary.by_alpha` holds one AlphaRisk per requested alpha, ascending. Each `alpha` equals the request float bit for bit. `loss_percentiles` holds one PercentileLoss per grid point, ascending. [Proposed] Test: S1.T3, S1.T4
- C.4 `RiskSummary.at(alpha)` returns the record with `record.alpha == alpha`. For any other value it raises KeyError (O-9). No result method returns NaN. `max_alpha`, `get_var` and `get_es` are removed. [Proposed] Test: S1.T5
- C.5 RiskSummary and ModelTail have no ES/VaR ratio member: `RiskSummary.es_to_var_ratio`, `RiskSummary.get_es_to_var_ratio` and `ModelTail.es_to_var_ratio` are removed (O-5). A run with VaR = 0 raises no exception and gives no NaN. [Proposed] Test: S1.T6
- C.6 `obligor_contributions` holds one record per request obligor, ascending by obligor_uid. `ead` equals the request ead. A uid without an ead raises CreditVarEngineError. `stress_test.obligors` is ascending by obligor_uid. [Proposed] Test: S1.T7
- C.7 The keys of `per_model_tail` and `summary.expected_loss_by_model` are JtdModel member names. The forced-default overlay uses "FORCED_DEFAULT". [Proposed] Test: S1.T8
- C.8 RunProvenance holds the fields of A_CONTRACT section 3.10. `numpy_version`, `scipy_version` and `numba_version` equal the runtime `__version__` values. [Proposed] Test: S1.T9
- C.9 `engine_version` follows P-CREDIT_VAR_API-001 clause 5. Missing metadata, or a distribution version with a local segment, raises CreditVarEngineError. [Proposed] Test: S1.T10
- C.10 `input_hash` is the 64-character lowercase SHA-256 hex of the canonical request bytes (A_CONTRACT section 4.2, the D040 form). Two requests that validate to field-wise bit-equal models give one hash. Field-wise bit-equal means: each float value, including each float in a dict or sequence field, is equal under `float.hex()`; each int, bool and str value, and each dict key, is equal under `==`. A 1-ulp change of any float gives another hash. -0.0 and 0.0 give different hashes. [Proposed] Test: S1.T11
- C.11 For one result per catalog key, `model_validate_json(r.model_dump_json()) == r`, and the second dump is byte-equal to the first. [Proposed] Test: S1.T12
- C.12 The SHA-256 of the LF-normalized source of `credit/factors/simulate/mixed.py` equals the value recorded beside `SCENARIO_CONSTRUCTION_ID`. [Proposed] Test: S1.T13
- C.13 (D-C15) The same request, run twice in one process and once in a fresh subprocess, gives byte-equal `model_dump_json` output after removal of `provenance.generated_at`. The result stays bitwise invariant to `_FACTOR_BATCH`. [Proposed] Test: S1.T14, S1.T15

Request and errors (S2):
- C.14 The request models set `extra="forbid"`, `frozen=True`, `allow_inf_nan=False`; their sequence fields are tuples. A request round-trips through JSON with no loss. [Proposed] Test: S2.T1
- C.15 `run_credit_var` and `simulate_portfolio_losses` validate the request state at call time. A request that in-place mutation made invalid raises R before any factor generation. [Proposed] Test: S2.T2
- C.16 `alphas` is non-empty, each value is in (0, 1), and no two values are equal under `==`. [Proposed] Test: S2.T3
- C.17 Request validation rejects: seed < 0 (D039 precedent); obligor_uid outside the int64 range; a duplicate (obligor_uid, factor) loading; a per-obligor sum of loading^2 > 1 + 1e-9, the engine tolerance; a loadings factor name in the reserved set of A_CONTRACT section 2.5 (R-05, F-36). [Proposed] Test: S2.T4, S2.T5, S2.T6, S2.T7
- C.18 Request coercion is strict (O-8). An int field accepts an int only. A float field accepts a float or an int and stores a float. A bool field accepts a bool only. A str field accepts a str only. These rules also apply to dict keys, dict values and sequence elements. A sequence field accepts a list or a tuple and stores a tuple; any other type raises R (F-34). [Proposed] Test: S2.T8
- C.19 The request has no size limit other than the constraints of A_CONTRACT section 2 (O-12). [Proposed] Test: S2.T9
- C.20 The public entry points are `run_credit_var` and `simulate_portfolio_losses`. Every exception of type Exception that a public entry point raises is R = `pydantic.ValidationError` or E = `CreditVarEngineError` (O-9). R is raised only before the first engine call. A ValidationError from the construction of a result model is an engine fault and raises E. Direct construction of a request model with a fault raises R. Each R entry carries the loc that A_CONTRACT section 2.5 defines for its rule. E chains the original exception as `__cause__` and does not subclass ValueError. [Proposed] Test: S2.T10, S2.T11
- C.21 The `from_dataframe` docstring names the columns that the code requires. Its default n_scenarios equals the request default (O-7). [Proposed] Test: S2.T12

Concurrency (S3):
- C.22 `MODEL_CATALOG` and `SUPPORTED_RECIPES` are read-only mappings. Item assignment raises TypeError. [Proposed] Test: S3.T1
- C.23 Concurrent calls on 4 threads give results byte-equal to serial runs (without `generated_at`), and the process does not abort. [Proposed] Test: S3.T2
- C.24 Results with `NUMBA_NUM_THREADS=1` and with the default are byte-equal. [Proposed] Test: S3.T3
- C.25 Each entry of `run_credit_var` or `simulate_portfolio_losses` into a parallel kernel (the 9 JTD kernels and the 2 tail kernels of F-17) happens while the thread holds one process-wide lock of credit_var (O-10). At most one thread is inside a credit_var parallel kernel at a time. The lock does not serialize parallel kernels that other riskquant modules enter. [Proposed] Test: S3.T4

Oracles (S4):
- C.26 For each of the 13 MODEL_CATALOG keys, one test through `api` compares EL, VaR and ES at two or more levels with the independent oracle of A_ORACLES.md. [Proposed] Test: S4.T1 to S4.T13
- C.27 `test_oracles.py` imports only from `riskquant.solutions.credit_var.api`, `tests.solutions.credit_var._oracle_pmf`, numpy, scipy, pytest and the standard library. `_oracle_pmf.py` imports only from numpy, scipy and the standard library. [Proposed] Test: S4.T14

Stability (S5):
- C.28 `sorted(api.__all__)` equals the 22-name list of P-CREDIT_VAR_API-001 clause 1. [Proposed] Test: S5.T1
- C.29 `model_json_schema()` of CreditVarRequest and CreditVarResult equals the golden files. A change needs a new golden file and a version decision under P-CREDIT_VAR_API-001. [Proposed] Test: S5.T2
- C.30 `pyproject.toml`: version "0.2.0"; numpy>=1.26,<3; scipy>=1.15,<2; pydantic>=2.0,<3; numba>=0.59,<0.N, where 0.(N - 1) is the numba minor version in the S5 gate environment (O-11). [Proposed] Test: S5.T3
- C.31 `STABLE_API.md` stays an internal contract of riskquant.credit (O-4, D001). The names that it lists equal the union of the riskquant.credit `__all__` lists, each at its stated path. Its first content line states internal status. It makes no public, stable or versioned claim. [Proposed] Test: S5.T4

Benchmark (S6):
- C.32 The benchmark script produces every measurement of S6 with its environment record. No `src/` file changes. [Proposed] Test: S6 gate (script run)

## 4. Decisions

Decisions relied on: D001, D-C15, D-C04, D-C03, D039, D040, D041, D104. Each checked against DECISIONS.md: yes.

D104.3 gives "pydantic is not declared in pyproject.toml" as one reason. That reason is stale at c060f06 (F-22). This workstream does not change D104. The owner decides whether D104 needs a note.

### P-CREDIT_VAR_API-001: Stable public surface of riskquant.solutions.credit_var.api (PROPOSED)

```text
Related: D001, D002, D-C15, D039, D040, D041, D104
Scope: the public surface, the versioning, the build identity and the error and
concurrency contract of riskquant.solutions.credit_var.api.
1. Surface. The stable surface is the 22 names of api.__all__ that S5.T1
   freezes: AlphaRisk, CreditVarEngineError, CreditVarRequest, CreditVarResult,
   GroupRow, LoadingRow, LossHistogram, MODEL_CATALOG, ModelEntry, ModelTail,
   ObligorContribution, ObligorRow, ParamSpec, PercentileLoss, RiskSummary,
   RunProvenance, StressTestObligor, StressTestResult, model_entry,
   run_credit_var, simulate_portfolio_losses, supported_model_keys.
   The member CreditVarRequest.from_dataframe is outside the stable surface;
   clause 3 does not apply to it. simulate_portfolio_losses is a Python-only
   member: its np.ndarray return has no JSON form, no result schema version and
   no golden file. A consumer imports riskquant names only from
   riskquant.solutions.credit_var.api. The request error type comes from
   pydantic; api does not re-export it. All other modules of
   riskquant.solutions.credit_var and riskquant.credit are internal.
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
8. Errors. The public entry points raise only pydantic.ValidationError for a
   request fault and CreditVarEngineError(RiskQuantError) for an engine fault.
   RiskSummary.at raises KeyError for a level that the run did not compute.
9. Concurrency. credit_var serializes entry into its parallel kernels with one
   process lock. Calls on separate threads of one process give the results of
   serial calls. The lock does not serialize parallel kernels that other
   riskquant modules enter.
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
| O-4 | STABLE_API.md: keep as an internal contract of riskquant.credit and correct it to match `__all__`; state internal status; drop the public, stable and versioned claims. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (a); F-28; R1 A-06 |
| O-5 | ES/VaR ratio at VaR = 0: remove both ratio members. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (a); F-6 |
| O-6 | `simulate_portfolio_losses`: keep as a Python-only member, exempt from the JSON property. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (b); F-29 |
| O-7 | `from_dataframe`: keep, outside the stable surface; P-CREDIT_VAR_API-001 clause 1 exempts the member. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (b); F-23, F-33; R1 A-05 |
| O-8 | Request coercion: strict for number, bool and string fields; sequence fields accept a list or a tuple and store a tuple. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (a); F-13, F-34; R1 R-03 |
| O-9 | Error taxonomy: R = `pydantic.ValidationError`, E = `CreditVarEngineError(RiskQuantError)`; the `at()` lookup error is KeyError. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (a); F-14, F-15 |
| O-10 | Threading policy: serialize parallel-kernel entry in credit_var with a process lock. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (b); F-17, F-18; R1 A-03, R-04 |
| O-11 | Upper-bound rule for the pydantic and numba pins: pydantic <3; numba at the next minor above the version that S5 tests. | CLOSED | 2026-10-29 | Owner, revision-2 prompt; F-22 |
| O-12 | Request size limits: none. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, option (a) |
| O-13 | C-15, C-18, C-07A, P2-09 and D-C18 join the O-3 triage item. | CLOSED | 2026-10-29 | Owner, revision-2 prompt, yes; R2 I-039 |

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
Depends on: none. S1 creates CreditVarEngineError; S2 applies the error mapping.
Files:
- create: src/riskquant/solutions/credit_var/errors.py; src/riskquant/solutions/credit_var/provenance.py; tests/solutions/credit_var/test_result_contract.py; tests/solutions/credit_var/test_reproducibility.py
- modify: src/riskquant/solutions/credit_var/schemas.py; src/riskquant/solutions/credit_var/run.py; tests/credit/factors/test_batch_size_reproducibility.py
- modify: scripts/solutions/credit_var/demo_e2e_*.py (15 files): each reader of a removed field that the FIRST step finds; U-a; U-b; U-d
FIRST: Search `src/`, `scripts/` and `tests/` for `summary.var`, `summary.es`, `economic_capital_`, `loss_percentiles`, `get_var`, `get_es`, `max_alpha`, `es_to_var_ratio`, `per_model_tail`. Report every hit. STOP AND ASK for a hit outside the Files list.
Prohibited: src/riskquant/credit/**; src/riskquant/scenario_metrics/**; src/riskquant/jit.py; src/riskquant/solutions/run_identity/**; catalog.py; recipes.py; request models; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt ; context_riskquant_solutions_run_identity.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/solutions/credit_var", "tests/credit/factors/test_batch_size_reproducibility.py", "tests/credit/jtd/test_beta_binomial_contract.py"`; then the 15 demos (U-c): `Get-ChildItem scripts/solutions/credit_var/demo_e2e_*.py | ForEach-Object { python $_.FullName; "$($_.Name) exit=$LASTEXITCODE" }`, each with exit=0
Exit: every S1 test passes; full suite passes; no NaN, no level-keyed dict and no "0+unknown" path remains in credit_var.
Status: VALIDATED
Evidence: S1 closure review ACCEPT 2026-11-02: R-001 to R-010 closed; R-003 closed (15 of 15 demos exit=0); R-011 closed by owner decision (ZV comparison). Owner gate at bf64906 (clean): ruff, format, mypy exit=0; pytest slice 213 passed; full 7690 passed, 36 skipped; 15 of 15 demos exit=0.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | Each result model: NaN, +inf and -inf in each float field, including a float value in a dict field; assignment to each field; an extra key in the payload; each sequence field built from a list | ValidationError (finite_number); ValidationError (frozen_instance); ValidationError (extra_forbidden); the field holds a tuple | NaN or inf reaching JSON as null (F-13); mutable result (F-3); silent extra key; in-place mutation of a list field (S1 review R-002) |
| S1.T2 | Result payload with result_schema_version "0.9" and "1.1" | ValidationError | silent acceptance of a foreign version |
| S1.T3 | alphas (0.999, 0.95, 0.9999, 0.99991) | 4 records, ascending, alpha.hex() equal to input | collision, reorder, rounding (F-4) |
| S1.T4 | alphas (0.99995, 0.999999) | 2 records; neither equals 1.0 | "100.00%" label (F-4) |
| S1.T5 | at(0.999), at(nextafter(0.999, 1)), at(0.5) on a run with 0.999 | record; KeyError; KeyError | rounded lookup, NaN return |
| S1.T6 | Run with VaR = 0 at 0.95 (independent_bernoulli, low PD, N = 3) | no ES/VaR ratio member on RiskSummary or ModelTail; no exception; no NaN | F-6 |
| S1.T7 | uids (30, 10, 20), force_default on 20; patched missing ead | uids (10, 20, 30); ead equal to request; CreditVarEngineError | F-7, order drift |
| S1.T8 | model forced_default; sf_gaussian with one force_default obligor | key "FORCED_DEFAULT" in both; no "forced_default" | F-11 |
| S1.T9 | Provenance versions and construction id | equal to runtime `__version__` values and the constant | F-8 |
| S1.T10 | engine_version format; patched PackageNotFoundError; patched version "0.2.0+x" | matches `^[^+]+\+src\.[0-9a-f]{64}$`; CreditVarEngineError; CreditVarEngineError | silent fallback (F-9) |
| S1.T11 | Equal requests from dict and from JSON; one ead changed by 1 ulp; ead -0.0 against 0.0 | equal hashes; different hashes; different hashes; 64 hex | truncation, repr drift (F-10), signed-zero claim |
| S1.T12 | Each of 13 keys (N <= 5, S = 2_000): dump, validate, dump | equal model; byte-equal JSON | lossy JSON |
| S1.T13 | LF-normalized SHA-256 of mixed.py | equals recorded value | unrecorded construction change |
| S1.T14 | One request: twice in process, once in a fresh subprocess | byte-equal JSON without generated_at | D-C15 reproducibility |
| S1.T15 | Batch-invariance test rewritten on by_alpha | bitwise equality across `_FACTOR_BATCH` values | test drift (F-24) |
| S1.T16 | One result payload per bound of A_CONTRACT 3.2 and 3.8, each with exactly one fault: loss_std_dev -1.0; loss_probability -0.1; loss_probability 1.1; by_alpha not strictly ascending; by_alpha alpha set not equal to provenance.alphas; an expected_loss_by_model key that is not a JtdModel name; bin_edges not ascending; len(bin_edges) != len(counts) + 1; sum(counts) != provenance.n_scenarios | ValidationError in each case | a result that breaks a declared constraint and still serializes (OD-1) |
| S1.T17 | Run: forced_default, one obligor, ead 450000.3, lgd 1.0, S = 2_000 (constant loss); payload with loss_min == loss_max and loss_skewness 0.5 | loss_std_dev, loss_skewness and loss_excess_kurtosis equal 0.0; ValidationError | non-zero moments from rounding of the mean at zero variance (sandbox, numpy 2.4.4: a constant 450000.3 vector of length 2_000 gives std 1.16e-10, skewness 0.999) (ZV) |

Appendix: A_CONTRACT.md

### S2: Request admission and error taxonomy

Objective: Move every request-content check before the engine, freeze the request, and publish the two error types.
Contract: C.14 to C.21
Depends on: S1
Files:
- create: tests/solutions/credit_var/test_request_contract.py; tests/solutions/credit_var/test_errors.py
- modify: src/riskquant/solutions/credit_var/schemas.py; src/riskquant/solutions/credit_var/run.py; src/riskquant/solutions/credit_var/errors.py
- modify, only to give the C.17 reserved-name rule the recipe factor-name format: src/riskquant/solutions/credit_var/recipes.py
- src/riskquant/credit/masters/obligors/validate.py (modify: R-001 overflow-free ascending check)
- src/riskquant/credit/masters/netting_set/validate.py (modify: same)
Prohibited: src/riskquant/credit/**; except the two validate.py files above; result models; catalog.py; pyproject.toml; DECISIONS.md.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt ; context_riskquant_tests_credit.txt
Gates: gate block; `$t = "tests/solutions/credit_var", "tests/credit/solutions"`
Exit: every S2 test passes; full suite passes; no request-content fault reaches the engine.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | Assignment to each field; request JSON round-trip | frozen_instance error; equal model | mutable request (F-2) |
| S2.T2 | Mutate `params` in place to an out-of-range value; call `run_credit_var` and `simulate_portfolio_losses`; generator patched to fail if called | R from each; generator not called | validation bypass (F-2, F-3) |
| S2.T3 | alphas (0.99, 0.99); (0.99, nextafter(0.99, 1)); (); (1.0,) | R; accepted; R; R | duplicate collapse (O-1) |
| S2.T4 | seed -1; seed 0 | R; accepted | late failure (F-12) |
| S2.T5 | obligor_uid 2**63; -2**63 - 1; 2**63 - 1 | R; R; accepted | overflow in the engine (F-12) |
| S2.T6 | Duplicate (uid, factor); sum loading^2 = 1 + 2e-9; = 1 + 0.5e-9 | R; R; accepted | predicate drift from the engine (F-12) |
| S2.T7 | mf_gaussian with obligor_uid 1 and factor "IDIO_001"; mf_students_t with factor "CHI_GLOBAL" | R with the section 2.5 loc; no engine call | name collision (F-36) |
| S2.T8 | Dict input, and JSON input where JSON can express the case: "ead": "100"; "obligor_uid": 7.0; "in_stress_test": 1; "seed": "7"; "group": 5; "ead": 100; alphas and obligors as a list and as a tuple; alphas as a set | R; R; R; R; R; accepted, stored as 100.0; accepted, stored as tuples; R | silent coercion (F-13, F-34) |
| S2.T9 | Validation only, no run: n_scenarios = 10**12 with one obligor | accepted | a size limit added without a version decision |
| S2.T10 | One request per rule of A_CONTRACT section 2.5, each with exactly one fault | R only; `errors()` has one entry; its loc equals the section 2.5 value; no engine call | mixed error types (F-15); loc () on a cross-row rule (F-35) |
| S2.T11 | Engine function patched to raise ValueError, TypeError, RuntimeError; engine output patched to give a NaN to a result model | E with `__cause__` in each case; `not issubclass(E, ValueError)` | engine fault reported as a bad request |
| S2.T12 | from_dataframe docstring columns and default n_scenarios | equal to code and request default | F-33 |

### S3: Process state and concurrent calls

Objective: Remove mutable process-global state and prove concurrent calls safe under the O-10 policy.
Contract: C.22 to C.25
Depends on: S2
Files:
- create: tests/solutions/credit_var/test_concurrency.py
- modify: src/riskquant/solutions/credit_var/catalog.py; src/riskquant/solutions/credit_var/recipes.py; src/riskquant/solutions/credit_var/run.py
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
| S3.T4 | Probes on each parallel-kernel entry that `run.py` reaches; 4 threads, 8 calls over sf_gaussian and mf_students_t | each probe sees the lock held; at most 1 thread inside the probes at a time | unmanaged concurrent kernel entry (F-18) |

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
Status: VALIDATED
Evidence: S4 review ACCEPT 2026-11-02 at 6b98935 (gate_output_S4.txt, RERUN section: ruff, format, mypy exit=0; 14 passed against the real engine; Exit mutation fails S4.T5). OD-S4-1 (a): pytest full waived to the S2 gate, which runs on a tree that contains 6b98935.

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 to S4.T13 | One key each, as A_ORACLES.md section 2 | VaR equal; EL and ES within the A_ORACLES bounds | wrong model math on the public path (F-25) |
| S4.T14 | AST import check of test_oracles.py and _oracle_pmf.py | imports limited per C.27 | oracle that shares production logic |

Appendix: A_ORACLES.md

### S5: Stability declaration, surface test and pins

Objective: Freeze the surface and the schemas, set the pins, and settle STABLE_API.md.
Contract: C.28 to C.31
Depends on: S1, S2, S3, S4 VALIDATED
Files:
- create: tests/solutions/credit_var/test_public_surface.py; tests/solutions/credit_var/golden/credit_var_request.schema.json; tests/solutions/credit_var/golden/credit_var_result.schema.json
- modify: src/riskquant/solutions/credit_var/api.py; pyproject.toml; src/riskquant/credit/STABLE_API.md (O-4)
Prohibited: every other src/ file; DECISIONS.md. The owner promotes P-CREDIT_VAR_API-001 and assigns the D number.
Attach: docs/workstreams/WS_CREDIT_VAR_API.md ; docs/workstreams/WS_CREDIT_VAR_API/A_CONTRACT.md ; docs/CLAUDE.md ; docs/DECISIONS.md ; context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ; context_riskquant_credit.txt ; pyproject.toml ; src/riskquant/credit/STABLE_API.md
Gates: gate block; `$t = "tests/solutions/credit_var"`; then `pip install -e .` and `python -c "import importlib.metadata as m; print(m.version('riskquant'))"`
Exit: every S5 test passes; full suite passes; installed version prints 0.2.0; the final report records `numba.__version__`, and N in C.30 equals its minor version + 1.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | `sorted(api.__all__)` | equals the 22 names of P-CREDIT_VAR_API-001 clause 1 | surface drift |
| S5.T2 | `model_json_schema()` of request and result | equals golden files | unversioned schema change |
| S5.T3 | `tomllib` parse of pyproject.toml | version 0.2.0; numpy, scipy and pydantic specs per C.30; numba spec `>=0.59,<0.N` with integer N; installed numba satisfies it | unbounded majors (F-22) |
| S5.T4 | STABLE_API.md | listed set equals the credit `__all__` union; each stated path exists; first content line starts with "Status: internal"; no line matches `(?i)\b(public\|stable\|versioned)\b` | F-28 drift; claims against D001 (R1 A-06) |

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
Status: IN_PROGRESS
Evidence: Script committed at db92afc; gates and smoke run pass on the owner machine; full run and QUICK review pending.

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
- P-CREDIT_VAR_API-001 clause 5: state the root of "relative path" and the sort key; V is not bound to the imported code under pytest `pythonpath`; H is computed at the first call (R1 R-06).
- P-CREDIT_VAR_API-001 clause 3: define "fixed numerical environment" (R1 R-07).
- C.12 hashes `mixed.py` only; factor order from `recipes._universe`, the universe compile and `base.py` also define the stream (R1 R-08).
- C.29: `model_json_schema()` output can change between pydantic 2.x minor versions (R1 R-09).
- pandas has no upper bound and is on the `api` import path (R1 R-11).
- A_CONTRACT 4.3 compares JSON bytes; the pydantic-core serializer version is not recorded (R1 R-12).
- S1 FIRST: search the whole repository, including `src/riskquant_lab/plotting/credit_var.py` and `tests/e2e/` (R1 R-13).
- C.21: the `loadings_table` docstring also says `uid` (R1 R-19).
- C.13, C.24: state whether BLAS thread settings are inside the contract (`_attribution.obligor_ul_contribution`) (R1 R-20).
- R2-02: the audit did not verify K12 rule 6 at 60x60x300 nodes; S4 runs the rule 6 self-check before any K12 assertion.
- riskquant-excel: not checked against the S1 result contract. Owner decision 2026-10-30: out of scope, no current users. Check it before any Excel tool work.
- A_CONTRACT 3.1 headline_alpha "max of request alphas" has no validator (S1 closure review, adjacent).
- A_CONTRACT 3.1 obligor_contributions "one per request obligor" has no check against provenance.n_obligors (S1 closure review, adjacent).
- A_CONTRACT 3.8 has no counts >= 0 rule (S1 closure review, adjacent).
- _attribution.py:loss_histogram: xmax = max(p99.9, losses.max(), 1.0), so the overflow branch and the display-percentile clip never act; the docstring is wrong (S1 closure review, adjacent).
- src/riskquant_lab/__init__.py is not empty, against D002 (S1 closure review, adjacent).
- Tooling: upgrade ruff (preview = true) after WS_CREDIT_VAR_API merges; one repository-wide format commit on its own branch (S4 review R-002).

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
