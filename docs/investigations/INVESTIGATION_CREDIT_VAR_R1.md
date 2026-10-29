# INVESTIGATION: CREDIT VAR

## RONUD 1

### PROMPT

Investigate per INVESTIGATOR.md.

MODE: PLAN CHANGE

[Q] QUESTION
Assess the readiness of riskquant.credit and riskquant.solutions.credit_var to
serve as the engine dependency of a credit-VaR web app. The web app lives in a
separate repository. It imports only riskquant.solutions.credit_var.api.

Answer these sub-questions. Cite path:symbol for each claim.

Q1. Core state (riskquant.credit).
    a. Map each MODEL_CATALOG key to its recipe, JTD block, evaluator and
       kernel. State which keys share a kernel.
    b. For D-C01, D-C03, D-C04 and D-C15, state whether the code complies.
       Give the enforcing symbol, or state the gap.
    c. List each audit ID that the code cites (for example C-04, C-07, C-07A,
       C-18, RQ-AUD-009, P2-07). For each, state whether DECISIONS.md records
       it. A cited ID with no decision entry is a finding.

Q2. Solution state (riskquant.solutions.credit_var).
    a. Trace run_credit_var from request to result. Record tier, shape, dtype
       and layout at each step.
    b. Confirm that every CreditVarResult field survives
       model_dump_json -> model_validate_json with no loss.
    c. List every NaN, None or float("nan") that a result can carry. Name the
       field and the trigger.
    d. List every module-level import that `import ...credit_var.api` pulls
       in. State whether pandas loads at import time.
    e. Compare RunProvenance with the D-C15 required release condition.
    f. List dead or orphaned symbols. Scope each negative claim with the
       search command and the profiles searched.
    g. Check from_dataframe column names against its docstring.

Q3. External-consumer readiness.
    a. State the exception types that run_credit_var can raise, so a caller
       can map each to a client error or a server error.
    b. State whether run_credit_var holds module-level mutable state, or any
       state that makes concurrent calls in one process unsafe.
    c. State the JIT warm-up cost and the run time and peak memory at the
       default n_scenarios. If no measurement exists in the snapshot, write
       the benchmark that the next prompt must require. Do not estimate.
    d. State which riskquant surface is versioned and stable for an external
       repository. Identify any documented stability contract
       (for example src/riskquant/credit/STABLE_API.md). If its text is not
       in the snapshot, mark it Unresolved.

Q4. Test coverage.
    a. List each test that calls run_credit_var or simulate_portfolio_losses.
    b. For each catalog key, state whether a public-path test with an
       independent oracle exists.

Q5. Gap list. Rank each gap as BLOCKING or NON-BLOCKING for a first web app
    release. Separate engine gaps from owner decisions.

LEADS (claims from an earlier chat; not evidence; verify or reject each):
  L-01 schemas.py:KernelDistribution has no reference in src or tests, and no
       model_config. [Inferred]
  L-02 schemas.py:RunProvenance lacks the scenario construction identifier,
       the NumPy version and the SciPy version that D-C15 requires. [Inferred]
  L-03 schemas.py imports pandas at module level, so importing api loads
       pandas. [Inferred]
  L-04 run.py:run_credit_var sets ObligorContribution.ead to float("nan")
       when a uid is absent from ead_by_uid. [Inferred]
  L-05 schemas.py:RiskSummary.es_to_var_ratio and ModelTail.es_to_var_ratio
       return NaN on a zero or missing VaR. [Inferred]
  L-06 schemas.py:CreditVarRequest.from_dataframe requires column
       "obligor_uid"; its docstring states "uid". [Inferred]
  L-07 No dedicated public-path golden test covers all 13 catalog keys.
       The earlier chat did not read the full tests profile. [Unresolved]
  L-08 run.py lines 3362-3384 and most of recipes.py were not read by the
       earlier chat. [Unresolved]

[G] CHANGE GOAL
riskquant.solutions.credit_var.api is a stable, versioned, fail-loud engine
dependency for an external web-app repository: JSON in, JSON out, reproducible
under D-C15, safe to call from a server process. This investigation defines
the engine-side changes only. The web-app design is out of scope.

[P] PRIORITY CONTEXT
<owner: priority, deadline, related work; omit to have them listed under
Owner inputs>

ATTACH:
  INVESTIGATOR.md ; WS_TEMPLATE.md ; CLAUDE.md ; DECISIONS.md
  context_riskquant_index.txt
  context_riskquant_credit.txt
  context_riskquant_solutions_credit_var.txt
  context_riskquant_scenario_metrics.txt
  context_riskquant_basel.txt
  context_riskquant_tests.txt
  pyproject.toml

OUT OF SCOPE:
  Web framework, task queue, persistence, authentication, UI.
  solutions.credit_exposure, except where credit_var imports it.
  Any code change. Analysis mode only.

### RESPONSE

Now schemas.py.

MODE: PLAN CHANGE
SNAPSHOT: `SNAPSHOT commit=90e2498 dirty=false content_hash=fa836966c1293f33`
STATUS: PARTIAL

## Answer

`riskquant.solutions.credit_var.api` is a correct and fail-loud engine today, but it is not yet a safe external dependency. Three gaps block a first web-app release: `pydantic` is imported but not declared in `pyproject.toml`; `RunProvenance` does not carry the D-C15 required release fields; and the result can carry a non-JSON `NaN` plus a silent confidence-level key collision. Q3c and Q4 are **Unresolved**: no benchmark exists in the snapshot, and `context_riskquant_tests.txt` was named in ATTACH but is not in the Project files.

## Evidence map

```
riskquant.solutions.credit_var.api                      [SOLUTION]
 ├─ catalog.MODEL_CATALOG      13 keys  -> recipes.SUPPORTED_RECIPES 13 assemblers
 ├─ schemas.CreditVarRequest   pydantic v2, extra="forbid"
 └─ run.run_credit_var
     ├─ recipes.assemble_inputs        pd.DataFrame (EDGE, strings alive)
     ├─ jtd.ingest.compile.compile_jtd_portfolio -> CompiledJtdPortfolio (band order)
     ├─ factors.simulate.mixed.MixedCreditFactorGenerator  (D-C15 one RNG consumer)
     ├─ factors.simulate.base.build_credit_factor_matrix
     │     np.empty (F, S) float64 order="C"; D-C04 coverage proof
     ├─ jtd.engine.run_jtd_for_obligor_ead
     │     align_obligor_ead_to_portfolio -> searchsorted_or_raise  (D-C03 strict)
     │     run_jtd: np.empty (N, S) float64 order="C"; evaluator per JTD_MODEL_ID
     ├─ aggregate.summary.summarize_jtd_loss / model_tail_var_es
     ├─ _attribution.obligor_tail_attribution  (chunked LOO, 50M-cell budget)
     └─ schemas.CreditVarResult   dict keys = _alpha_key(alpha) strings
```

## Findings

**I-001 | Observed | pyproject.toml**
Fact: `pydantic` is absent from `[project].dependencies`. `schemas.py` imports it at module level.
Evidence: `grep -n "pydantic" /mnt/project/pyproject.toml` returns nothing; `schemas.py:from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator`.
Consequence: an external repository that installs `riskquant` gets `ImportError` on `import ...credit_var.api` unless it declares pydantic itself. BLOCKING.

**I-002 | Observed | schemas.py:RunProvenance**
Fact: the fields are `engine_version, model, params, n_obligors, n_scenarios, seed, alphas, input_hash, generated_at`. There is no scenario-construction identifier, NumPy version or SciPy version.
Evidence: `schemas.py:RunProvenance`; D-C15 "Required release condition" lists scenario construction identifier, seed, scenario count, NumPy version, SciPy version.
Consequence: L-02 is confirmed. The release condition of D-C15 is not met. BLOCKING.

**I-003 | Observed | pyproject.toml, run.py:_engine_version**
Fact: `version = "0.1.0"` is static, and `_engine_version()` returns `importlib.metadata.version("riskquant")`. `numpy>=1.26` and `scipy>=1.15` are lower bounds only.
Evidence: `pyproject.toml:[project]`; `run.py:_engine_version`.
Consequence: `engine_version` cannot distinguish pre-D-C15 from post-D-C15 builds, which is exactly the distinction D-C15 requires.

**I-004 | Observed | run.py:_alpha_key**
Fact: `_alpha_key` quantises alpha to basis points. Distinct alphas collide, and alpha >= 0.99995 is labelled `"100.00%"`.
Evidence: sandbox run of the function body: `0.9999 -> '99.99%'`, `0.99991 -> '99.99%'` (collision True); `0.99995 -> '100.00%'`, `0.999999 -> '100.00%'` (collision True).
Consequence: `summary.var` / `summary.es` silently lose a requested level, and a 99.995% level is reported as 100%. This is a silent repair, against CLAUDE.md section 5. BLOCKING for a JSON API.

**I-005 | Observed | schemas.py:RiskSummary.max_alpha**
Fact: `max_alpha` runs `max(float(a) for a in self.var)`, but `self.var` keys are `_alpha_key` strings such as `"99.90%"`.
Evidence: `run.py:run_credit_var` builds `var={_alpha_key(a): ...}`; `schemas.py:RiskSummary.max_alpha`.
Consequence: `max_alpha` raises `ValueError` on every result. The property is unusable.

**I-006 | Observed | run.py:run_credit_var**
Fact: `ObligorContribution.ead` is `ead_by_uid.get(int(uid), float("nan"))`.
Evidence: `run.py:run_credit_var`, contributions list comprehension.
Consequence: L-04 is confirmed as code. `float("nan")` is not valid JSON. The uid set comes from `req.obligors` in both branches of `_build_loss_matrix`, so the default looks unreachable today [Inferred]; the NaN default itself still violates the no-NaN rule.

**I-007 | Observed | schemas.py:RiskSummary.es_to_var_ratio, ModelTail.es_to_var_ratio**
Fact: both return `float("nan")` on a zero or missing VaR. Both are `@property`, not model fields.
Evidence: `schemas.py:RiskSummary.es_to_var_ratio`; `schemas.py:ModelTail.es_to_var_ratio`; `get_var` / `get_es` / `get_es_to_var_ratio` also end in `float("nan")` fallbacks.
Consequence: L-05 is confirmed for the return value, and corrected on scope: properties are not serialised, so these NaNs do not cross the wire. They do reach a Python consumer.

**I-008 | Observed | schemas.py:KernelDistribution**
Fact: the class has no `model_config`, is not in `__all__`, is not re-exported by `api.py`, and has no other reference in the attached profiles.
Evidence: `grep -rn "KernelDistribution" /mnt/project` returns only `schemas.py:552` (class definition) and its index entry. Profiles searched: all 31 files in `/mnt/project`. The `tests` profile was not attached.
Consequence: L-01 is confirmed within this scope. The negative claim is **Unresolved** for the tests profile.

**I-009 | Observed | schemas.py:CreditVarRequest.from_dataframe**
Fact: the code requires columns `("obligor_uid", "ead", "pd", "lgd")` and `("obligor_uid", "factor", "loading")`. The docstring states `uid, ead, pd, lgd` and `uid, factor, loading`.
Evidence: `schemas.py:from_dataframe`, `required = ("obligor_uid", "ead", "pd", "lgd")` against its own docstring.
Consequence: L-06 is confirmed, and it also applies to the loadings table. `from_dataframe` also defaults `n_scenarios=100_000` while `CreditVarRequest.n_scenarios` defaults to `250_000`.

**I-010 | Observed | pandas at import**
Fact: `import riskquant.solutions.credit_var.api` loads pandas, scipy and numba at module level.
Evidence: static closure of 71 modules computed from `context_riskquant_index.txt` dependency edges. Within it: `schemas.py`, `recipes.py`, `run.py`, `credit/jtd/ingest/*`, `credit/masters/*/ingest.py` import pandas; `credit/factors/simulate/mixed.py` and `credit/jtd/ingest/_shared.py` import scipy; `riskquant/jit.py` imports numba.
Consequence: L-03 is confirmed and is wider than the lead states. Removing pandas from `schemas.py` alone would not remove pandas from the import path.

**I-011 | Observed | credit/jtd/engine.py:JTD_EVALUATORS**
Fact: `JTD_EVALUATORS: Final[...] = evaluator_table()` runs at import. `registry.JtdModelSpec.resolve_evaluator` calls `importlib.import_module`, so all 9 evaluators and all 9 Numba kernels load at import of `api`.
Evidence: `engine.py:JTD_EVALUATORS`; `registry.py:resolve_evaluator`; `registry.py:evaluator_table`.
Consequence: the static dependency graph understates the import cost. Numba decoration of 9 kernels happens at import. Compilation cost is **Unresolved** (no measurement).

**I-012 | Observed | catalog.py:MODEL_CATALOG**
Fact: `MODEL_CATALOG` is a plain `dict[str, ModelEntry]` at module level, re-exported by `api.py`.
Evidence: `catalog.py:MODEL_CATALOG`; `api.py:__all__`.
Consequence: this is module-level mutable state on the public surface. A consumer can mutate it and change validation for every later call in the process.

**I-013 | Observed | D-C04 compliance**
Fact: the code complies. `build_credit_factor_matrix` calls `new_scenario_coverage`, `record_scenario_span` per batch and `assert_scenario_coverage_complete` before returning.
Evidence: `credit/factors/simulate/base.py:build_credit_factor_matrix`; `credit/_scenario_coverage.py`. The same three calls appear in `credit/jtd/streaming.py:run_jtd_streaming`.
Consequence: no gap. The `credit_var` path uses the non-streaming `run_jtd_for_obligor_ead`, so only the factor-matrix guard is active on this path.

**I-014 | Observed | D-C03 compliance**
Fact: the code complies on both paths. `jtd/bridge.py:aggregate_ead_to_obligor` raises on a target obligor with no netting set, citing D-C03. On the `credit_var` path the enforcing symbol is `exposure/obligor_ead/align.py:align_obligor_ead_to_portfolio` through `riskquant._arrays.searchsorted_or_raise`.
Consequence: no gap.

**I-015 | Observed | D-C01 compliance**
Fact: the code complies. `jtd/bridge.py:_sorted_target_view` makes the sorted copy and keeps the permutation, so target order is free and duplicates raise.
Consequence: no gap. `credit_var` does not call the bridge; it uses the obligor-grain `align` path, whose `universe` is the sorted-unique `CompiledObligorEad.uid`.

**I-016 | Observed | D-C15 construction compliance**
Fact: the construction complies. `MixedCreditFactorGenerator.iter_batches` documents and implements one RNG consumer, the whole-buffer `standard_normal` fill, with PIT-derived non-normal columns and a C-order buffer.
Evidence: `credit/factors/simulate/mixed.py` module docstring section "REPRODUCIBILITY (D-C15)" and `iter_batches`.
Consequence: the construction half of D-C15 holds. The release-condition half fails (I-002, I-003).

**I-017 | Observed | cited audit IDs**
Fact: the code cites `C-04`, `C-07`, `RQ-AUD-009`, `RQ-AUD-012`, `P2-03`, `P2-07`, `P2-11A`, `D-C01`, `D-C03`, `D-C15`, `D080`. Of these, `C-04`, `C-07`, `RQ-AUD-009` and `P2-07` have no entry in `DECISIONS.md`.
Evidence: `grep -rhoE` over the four attached credit profiles for the ID pattern; then `grep -n` per ID against `DECISIONS.md`. `C-04` and `C-07` return zero lines (note: `D-C04` does not contain the substring `C-04`). `RQ-AUD-012` appears once at `DECISIONS.md:45`; `P2-03` four times; `P2-11A` under D-C16.
Consequence: four cited IDs are findings. Sites: `credit/_scenario_coverage.py` docstring (C-04), `credit/factors/simulate/mixed.py` docstring (C-07), `credit/jtd/ingest/validate.py` (C-07), `solutions/credit_var/_attribution.py:obligor_tail_attribution` (RQ-AUD-009), `solutions/credit_var/schemas.py:_validate_marshall_olkin_systemic_pd` (P2-07).

**I-018 | Unresolved | src/riskquant/credit/STABLE_API.md**
Fact: the file exists in the repository and is cited by `credit/factors/simulate/mixed.py`, but its text is not in any attached profile.
Evidence: `project_structure.txt:480`; `context_riskquant_index.txt:48`; `context_riskquant_credit.txt:3548`. Only `.py` files appear in the profile bodies.
Consequence: Q3d cannot be answered. No versioned stability contract can be confirmed for an external repository.

**I-019 | Observed | catalog-to-kernel map**
Fact: 13 catalog keys map to 9 `JtdModel` values, so four kernels are shared by two keys each.

| Catalog key | Recipe assembler | JtdModel / block | Evaluator | Kernel |
|---|---|---|---|---|
| `forced_default` | `_assemble_forced_default` | `FORCED_DEFAULT` / `ForcedDefaultJtdBlock` | `evaluate_forced_default_jtd` | `kernel_forced_default_jtd` |
| `independent_bernoulli` | `_assemble_independent_bernoulli` | `INDEPENDENT_BERNOULLI` | `evaluate_independent_bernoulli_jtd` | `kernel_independent_bernoulli_jtd` |
| `beta_binomial` | `_assemble_beta_binomial` | `BETA_BINOMIAL` | `evaluate_beta_binomial_jtd` | `kernel_beta_binomial_jtd` |
| `beta_binomial_grouped` | `_assemble_beta_binomial_grouped` | `BETA_BINOMIAL` | same | same (shared) |
| `stress_regime_bernoulli` | `_assemble_stress_regime_bernoulli` | `STRESS_REGIME_BERNOULLI` | `evaluate_stress_regime_bernoulli_jtd` | `kernel_stress_regime_bernoulli_jtd` |
| `stress_regime_bernoulli_grouped` | `_assemble_stress_regime_bernoulli_grouped` | `STRESS_REGIME_BERNOULLI` | same | same (shared) |
| `sf_gaussian` | `_assemble_sf_gaussian` | `COPULA_SF_GAUSSIAN` | `evaluate_sf_gaussian_copula_jtd` | `kernel_sf_gaussian_copula_jtd` |
| `sf_gaussian_basel` | `_assemble_sf_gaussian_basel` | `COPULA_SF_GAUSSIAN` | same | same (shared) |
| `sf_students_t` | `_assemble_sf_students_t` | `COPULA_SF_STUDENTS_T` | `evaluate_sf_students_t_copula_jtd` | `kernel_sf_students_t_copula_jtd` |
| `sf_students_t_basel` | `_assemble_sf_students_t_basel` | `COPULA_SF_STUDENTS_T` | same | same (shared) |
| `mf_gaussian` | `_assemble_mf_gaussian` | `COPULA_MF_GAUSSIAN` | `evaluate_mf_gaussian_copula_jtd` | `kernel_mf_gaussian_copula_jtd` |
| `mf_students_t` | `_assemble_mf_students_t` | `COPULA_MF_STUDENTS_T` | `evaluate_mf_students_t_copula_jtd` | `kernel_mf_students_t_copula_jtd` |
| `marshall_olkin` | `_assemble_marshall_olkin` | `MARSHALL_OLKIN` | `evaluate_marshall_olkin_jtd` | `kernel_marshall_olkin_jtd` |

Evidence: `catalog.py:MODEL_CATALOG`; `recipes.py:SUPPORTED_RECIPES`; `registry.py:JTD_MODEL_SPECS`; evaluator-to-kernel imports extracted by `awk` over `context_riskquant_credit.txt`. The `basel` keys differ only by `_per_obligor_rho(regulatory=True)`; the `grouped` keys differ only by one factor per group.
Consequence: a per-key golden test is still needed, because the shared kernel does not test the recipe that feeds it.

**I-020 | Observed | catalog.py no-drift guard**
Fact: `catalog.py` raises `RuntimeError` at import if a catalog key has no recipe. The reverse direction is not guarded.
Evidence: `catalog.py:_missing = set(MODEL_CATALOG) - set(SUPPORTED_RECIPES)`.
Consequence: a recipe with no catalog entry is unreachable and silent. Minor.

**I-021 | Inferred | memory at the default scenario count**
Fact: at `n_scenarios=250_000` the path allocates at least three `(rows, 250_000)` float64 arrays: the factor matrix `(F, S)`, the broadcast EAD `(N, S)`, and the loss `(N, S)`. For `sf_gaussian`, `F = 1 + N`.
Evidence: `base.py:build_credit_factor_matrix` `np.empty((num_factors, num_scenarios))`; `engine.py:run_jtd_for_obligor_ead` `np.ascontiguousarray(np.broadcast_to(...))`; `engine.py:run_jtd` `np.empty((n_obligors, n_scenarios))`; `_attribution.py:_LOO_CELL_BUDGET = 50_000_000`.
Consequence: memory grows as `O(N * S)` with a factor of roughly three, plus a leave-one-out chunk capped near 400 MB. This is arithmetic from the shapes, not a measurement. A benchmark is required.

**I-022 | Observed | exception types**
Fact: the reachable types are `pydantic.ValidationError` (request parsing and all `model_validator` failures), `ValueError` (recipes, ingest, masters, coverage, alignment, engine shape checks), `TypeError` (dtype guards in compiled containers), `KeyError` (`catalog.model_entry`), `RuntimeError` (import-time registry and builder guards) and `riskquant.exceptions.ConfigurationError` (`masters/factor_universe/ingest.py`).
Evidence: `grep -oE "raise [A-Za-z_]+"` over the credit core region (148 `ValueError`, 16 `TypeError`, 10 `RuntimeError`, 1 `ConfigurationError`) and over the credit_var region (47 `ValueError`, 1 `RuntimeError`, 1 `KeyError`).
Consequence: `ValueError` carries both caller errors and engine invariant failures, so a caller cannot map `ValueError` to one HTTP class. There is no `CreditVarError` base type. Note the request validators raise bare `ValueError` inside `model_validator`, which pydantic wraps as `ValidationError`.

**I-023 | Observed | concurrency state**
Fact: `run_credit_var` holds no module-level mutable state of its own. `MixedCreditFactorGenerator` buffers are per instance. `riskquant.jit` sets `nogil=True` and `cache=True` on every decorator.
Evidence: `run.py` module body (only `_PERCENTILE_GRID` and `_FACTOR_BATCH`, both immutable); `jit.py:njit`, `parallel_njit`, `stable_math_parallel_njit`, `inline_njit`.
Consequence: concurrent calls in one process are safe for correctness, except for the mutable `MODEL_CATALOG` (I-012). `cache=True` writes `.nbi` / `.nbc` files beside the installed package, which a read-only container may refuse [Inferred].

### Adjacent defects (one line each, not in scope)

* `recipes.py:_universe` writes `np.nan` into the universe frame for absent `nu` / `alpha` / `beta`, against the no-NaN rule; the comment says ingest maps it to sentinels.
* `run.py:_simulate` and `_obligor_frame` build pandas frames per call inside a solution-tier function.
* `catalog.py:MODEL_CATALOG` is typed `dict[str, ModelEntry]` where `MappingProxyType` is used elsewhere in the estate (`registry.py:JTD_MODEL_REGISTRY`).
* `run.py:_PERCENTILE_GRID` carries a `# NEW (16 points: ...)` comment, which is changelog text in source.
* `_alpha_key` and `_percentile_key` docstrings say "S-tier", which states no contract.

## Options

**Option A (smallest correct change): declare pydantic, extend `RunProvenance`, remove the NaN and the key collision.**
Action: add `pydantic>=2` to `pyproject.toml`; add the D-C15 fields to `RunProvenance`; make the alpha key lossless or key by the raw float string; make a missing `ead` raise rather than return NaN.
Consequence: `api` becomes installable and reproducible under D-C15. `CreditVarResult` gains fields and changes dict keys, so it is a wire-contract change. Touches D-C15.

**Option B: Option A plus a declared stability contract.**
Action: Option A, plus a `STABLE_API` statement for `solutions.credit_var.api` with its own surface version, plus a round-trip test and a per-key golden test.
Consequence: an external repository gets a named contract to pin against. Needs the `tests` profile and the `STABLE_API.md` text first. Touches D-C15 and needs one new decision for the surface version policy.

**Option C: do nothing on the engine; let the web app absorb the gaps.**
Consequence: the web app must declare pydantic itself, re-key the alpha dicts, strip NaN before serialising, and stamp its own environment identity. The D-C15 release condition stays unmet in the engine, which is where the decision places it. Rejected as a recommendation; it moves engine obligations into a repository that cannot see the generator.

Recommendation: **Option B**, run as Option A first. [Proposed]

## Value assessment

```
VALUE
Benefit:        api becomes installable, reproducible under D-C15, and JSON-safe
                for an external repository; the web app stops carrying engine debt.
Cost proxies:   3 slices; files: pyproject.toml, schemas.py, run.py, plus tests;
                shared interface: CreditVarResult wire contract changes (keys +
                fields); 1 new decision (surface version policy); 1 audit round.
Alternative:    Option A alone: pyproject + provenance + NaN/key fix, no contract.
Not doing it:   ImportError on install (I-001); D-C15 release condition unmet
                (I-002); silently dropped confidence levels (I-004).
Case against:   The wire contract is not yet frozen, so fixing the alpha key and
                adding provenance fields twice is cheaper than freezing a
                contract before the web app's field needs are known (I-004,
                I-002). A frozen surface version (Option B) may be premature.
Reversibility:  costly. Changing dict keys and adding provenance fields after an
                external consumer exists is a breaking change in that consumer.
Parity:         n/a. G names no reference artifact.
Owner inputs:   priority, deadline, related work; whether the web app release
                predates or follows a riskquant version bump off 0.1.0.
Class:          HEAVY (contract impact: CreditVarResult and RunProvenance are
                published contracts; D-C15 release condition is touched)
Recommendation: PROCEED SMALLER [Proposed]
```

The smallest alternative, Option A, does not fail: it clears all three blocking gaps. Option B's extra value is the stability contract, which cannot be specified until `STABLE_API.md` and the `tests` profile are read. That is why the recommendation is `PROCEED SMALLER`, not `PROCEED`.

## Open items

1. **Q3c Unresolved.** No JIT warm-up, run-time or peak-memory measurement exists in the snapshot. The next prompt must require a benchmark, not an estimate. Required shape: cold-process import time of `riskquant.solutions.credit_var.api` with an empty Numba cache and with a warm cache; wall time and peak RSS of `run_credit_var` at `n_scenarios=250_000` for `N` in {10, 100, 1000} and `model` in {`forced_default`, `sf_gaussian`, `mf_students_t`}; the same for `simulate_portfolio_losses`. Report the exact command, the machine, and the NumPy, SciPy and Numba versions.
2. **Q3d Unresolved.** `src/riskquant/credit/STABLE_API.md` exists but its text is not in the snapshot. Input that resolves it: the file itself, or a profile that includes non-`.py` files.
3. **Q4 Unresolved.** `context_riskquant_tests.txt` is named in ATTACH but is not in the Project files. Q4a and Q4b cannot be answered, and I-008's negative claim is not fully scoped. Input that resolves it: the `tests` profile (`tokens~=796679` per the index; a filtered `tests/credit` and `tests/solutions/credit_var` extract would be enough).
4. **Owner decision.** Whether `CreditVarResult`'s alpha dict keys may change. Any fix to I-004 changes the keys a consumer reads.
5. **Owner decision.** Whether a `D` entry is opened for the four cited audit IDs with no record (`C-04`, `C-07`, `RQ-AUD-009`, `P2-07`), or whether the citations are removed from the code. The investigator does not assign a `D` number.
6. **Owner decision.** Whether `solutions.credit_var.api` gets a surface version separate from the package version, and what pins NumPy and SciPy for D-C15 reproducibility.
7. **Owner input.** `[P]` was left as a placeholder. Priority, deadline and related work are not supplied.

## Next prompt

```
MODE: investigation
TASK: Answer Q3c, Q3d and Q4 of the credit-VaR web-app readiness review, which
      the first round could not answer because the inputs were absent.
ATTACH: context_riskquant_index.txt ; context_riskquant_solutions_credit_var.txt ;
        context_riskquant_credit.txt ; context_riskquant_tests.txt ;
        src/riskquant/credit/STABLE_API.md ; pyproject.toml ; DECISIONS.md
DECISIONS: D-C15, D-C04, D-C03, D-C01
WS: none
FACTS:
  I-019 src/riskquant/solutions/credit_var/catalog.py:MODEL_CATALOG holds 13 keys
        that map to 9 riskquant.credit.jtd.models:JtdModel values; four kernels
        are shared by two keys each (beta_binomial/+_grouped,
        stress_regime_bernoulli/+_grouped, sf_gaussian/+_basel,
        sf_students_t/+_basel). [Observed]
  I-011 src/riskquant/credit/jtd/engine.py:JTD_EVALUATORS calls
        registry.evaluator_table() at module import, which importlib-imports all
        9 evaluators and all 9 Numba kernels. Import of
        riskquant.solutions.credit_var.api therefore loads numba, pandas and
        scipy. [Observed]
  I-001 pyproject.toml declares no pydantic dependency, although
        src/riskquant/solutions/credit_var/schemas.py imports it. [Observed]
  I-002 src/riskquant/solutions/credit_var/schemas.py:RunProvenance lacks the
        scenario-construction identifier, the NumPy version and the SciPy
        version that D-C15 requires as a release condition. [Observed]
  I-004 src/riskquant/solutions/credit_var/run.py:_alpha_key collides distinct
        confidence levels (0.9999 and 0.99991 both give "99.99%") and labels
        alpha >= 0.99995 as "100.00%". [Observed]
  I-006 src/riskquant/solutions/credit_var/run.py:run_credit_var sets
        ObligorContribution.ead to float("nan") when a uid is absent from
        ead_by_uid. [Observed]
VERIFY FIRST:
  The float("nan") default at I-006 is unreachable on the current path, because
  _build_loss_matrix derives every uid from request.obligors. [Inferred]
  Peak memory grows as about 3 * N * S * 8 bytes plus a leave-one-out chunk
  capped at 50M cells. [Inferred]
SCOPE: Answer only Q3c, Q3d and Q4. Produce the benchmark specification for
       Q3c and run it if the sandbox allows; state the sandbox limit if not.
       For Q4, list each test that calls run_credit_var or
       simulate_portfolio_losses, and state for each of the 13 catalog keys
       whether a public-path test with an independent oracle exists.
       For Q3d, quote the stability guarantee that STABLE_API.md states, and
       say whether it covers riskquant.solutions.credit_var.api.
OUT OF SCOPE: any code change; the web-app design; solutions.credit_exposure.
ACCEPTANCE: every claim carries path:symbol and a label; every negative claim
       carries its search command and the profiles searched; no performance
       number without an executed measurement.
GATES: none (analysis only).
REPORT: final report per CLAUDE.md section 11
STOP AND ASK: whether the alpha dict keys of CreditVarResult may change; whether
       a D entry is opened for C-04, C-07, RQ-AUD-009 and P2-07.
```

Queue after this: S1 declare pydantic and extend `RunProvenance` (D-C15 release condition); S2 remove the NaN and the alpha-key collision; S3 per-key golden tests and a JSON round-trip test.

MODE: PLAN CHANGE
SNAPSHOT: `SNAPSHOT commit=90e2498 dirty=false content_hash=fa836966c1293f33`
STATUS: BLOCKED

## Answer

The investigation is complete for the attached inputs. Q1, Q2, Q3a, Q3b and Q5 are answered in the previous response. Q3c, Q3d and Q4 stay **Unresolved**, and no further work on them is possible in this chat without new input. I do not start a new task on "Continue", because the mode and the question define the scope.

## Missing inputs

| Sub-question | Missing input | Why this chat cannot proceed |
|---|---|---|
| Q3c | An executed benchmark | The sandbox holds context text files only. It has no installed `riskquant` package, no NumPy/SciPy/Numba environment for the engine, and no production machine. A performance claim from reading is forbidden (INVESTIGATOR.md section 4, rule 5). |
| Q3d | `src/riskquant/credit/STABLE_API.md` | The file exists at `context_riskquant_index.txt:48` and `project_structure.txt:480`. The attached profiles carry `.py` bodies only, so its text is absent. |
| Q4 | `context_riskquant_tests.txt` | The ATTACH list names it. It is not in the Project files. The index lists it as `tests status=COMPLETE tokens~=796679`. |

## Options

**Option 1: attach the three inputs and send the round-2 prompt.**
Action: add `context_riskquant_tests.txt` and the `STABLE_API.md` text to the Project, run the benchmark on your own machine, then send the `MODE: investigation` prompt from the previous response with the raw benchmark output under a `[X]` block.
Consequence: Q3c, Q3d and Q4 close. Option B of the plan becomes specifiable.

**Option 2: start the engine work now and leave Q3c, Q3d and Q4 open.**
Action: run slice S1 of the queue (declare `pydantic`; extend `RunProvenance` for the D-C15 release condition). Neither change depends on the missing inputs.
Consequence: the two blocking gaps I-001 and I-002 close first. The alpha-key decision (I-004) and the test coverage question stay open.

**Option 3: request the workstream draft.**
Action: repeat the `PLAN CHANGE` invocation with `OUTPUT: WORKSTREAM DRAFT`.
Consequence: the class is `HEAVY`, so the draft is a spec under `WS_TEMPLATE.md` Part 2, and the open owner decisions become `O-n` gates. The recommendation is `PROCEED SMALLER`, so a draft is allowed without an override.

Recommendation: **Option 2 first, Option 1 in parallel.** I-001 blocks installation in the external repository, and it depends on no missing input. [Proposed]

## Open items

The open items from the previous response stand unchanged: items 1 to 3 are the missing inputs above; items 4 to 6 are owner decisions (alpha dict keys, `D` entries for `C-04`, `C-07`, `RQ-AUD-009` and `P2-07`, and the surface version policy); item 7 is the `[P]` priority context, still a placeholder.

State which option you want. If you want Option 2, say so and I will write the S1 implementer prompt.


