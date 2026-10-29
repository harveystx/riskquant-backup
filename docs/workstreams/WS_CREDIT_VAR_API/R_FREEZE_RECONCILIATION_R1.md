# WS_CREDIT_VAR_API / R_FREEZE_RECONCILIATION_R1

Path: docs/workstreams/WS_CREDIT_VAR_API/R_FREEZE_RECONCILIATION_R1.md
Audited: docs/workstreams/WS_CREDIT_VAR_API.md Revision 1 (Status REVIEW); A_CONTRACT.md Revision 1; A_ORACLES.md Revision 1
Round: 1 of 2. One round in two sessions; the first session stopped early. This file is the complete round 1 output.
Snapshot used: SNAPSHOT commit=925533b dirty=false content_hash=046ed5dddb6305d3
Date: 2026-10-29
Auditor role: freeze audit, analysis only. No repository file changed.

VERDICT: NOT READY. 11 BLOCKING findings. 10 owner gates open.

## A. Inputs and method

### A.1 Inputs

| Input | State | Effect |
|---|---|---|
| docs/WS_TEMPLATE.md | MISSING. project_structure.txt lists it. No attached file holds its text. | Section titles A to G and the BLOCKING / REFINEMENT rule are inferred from the prompt. [Unresolved] |
| snapshot_delta_c060f06_925533b.txt | MISSING. snapshot_delta_c060f06_29ece2c.txt is attached instead. | The c060f06 to 925533b delta is not confirmed. [Unresolved] |
| snapshot_delta_c060f06_29ece2c.txt | 9 paths, all under `_context_gen/`. No path under src/riskquant/, tests/credit/ or tests/solutions/. No pyproject.toml. | The owner claim holds for c060f06 to 29ece2c. [Observed] |
| context_riskquant_index.txt | 925533b. | Every clause check below uses the 925533b profiles directly. |
| context_riskquant_tests_credit.txt | No SNAPSHOT line. 18 files. | project_structure.txt lists the same 18 files under tests/credit/. The commit of the content is unknown. |
| Prompt token "<tests bucket>" | Placeholder, not filled. | Test facts are checked against context_riskquant_tests_credit.txt. |
| context_riskquant_lab.txt | No SNAPSHOT line. | Used only for the consumer search (R-13). |
| project_structure.txt | Filtered: `_context_gen/` is in the delta but not in the listing. | Directory-scoped negative claims are limited (R-13). [Observed] |

### A.2 Method

1. Static inspection of the 925533b profiles, DECISIONS.md, A_CONTRACT.md, A_ORACLES.md, STABLE_API.md and pyproject.toml.
2. Sandbox probes, labelled "sandbox". Environment: Python 3.12, pydantic 2.14.1, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2. Numba is not installed. riskquant is not installed. This is not the owner machine.
3. The oracle probes use numpy and scipy only. They follow A_ORACLES section 2. They call no riskquant function.
4. The ES estimator in the Monte Carlo probe is a re-implementation of the formula in `scenario_metrics/_computer.py` and `scenario_metrics/_kernels/_tail_metrics.py`. It is not the engine.
5. No gate, test or benchmark of the repository ran.

### A.3 Probes executed (sandbox)

| Probe | Purpose | Result |
|---|---|---|
| P-PYD (`p_pydantic.py`) | Strict mode, error loc, signed zero, frozen, re-validation | See R-03, A-08, A-09, C.15 |
| P-JSON | JSON round trip of tuple, dict and float fields | C.11 holds |
| P-ORA (`oracles.py`) | pmf per key, rules 4 to 7 | See A-01, A-10, A-11, R-15 |
| P-MC (`mc_es.py`) | Standard error of the ES estimator, 600 replications, S = 100_000, seed 7 | See A-01 |
| P-QUAD | Gauss-Laguerre convergence for K9, K10, K12 | See A-10 |
| P-DEG (`degenerate.py`) | s_ES^2 at a single-atom tail (K13) | See A-02 |
| P-UID | `to_numpy(dtype=np.int64)` on out-of-range uids | See R-02 |

## B. Clause reconciliation

Status values: CONSISTENT, DEFECT, PARTIAL (evidence incomplete), OWNER APPROVAL REQUIRED.

| Clause | Status | Evidence (925533b unless stated) | Finding |
|---|---|---|---|
| C.1 | CONSISTENT | Result models use `ConfigDict(extra="forbid")` only today (`schemas.py`). `run.py` casts every value with `float()`. | none |
| C.2 | CONSISTENT | New field. P-001 clause 4. | none |
| C.3 | CONSISTENT | `summarize_jtd_loss` keys `var` and `es` by the alpha float (`credit/aggregate/summary.py`). Bit-equal levels survive. | none |
| C.4 | OWNER APPROVAL REQUIRED (O-9) | Implementable under both O-9 options. | none |
| C.5 | OWNER APPROVAL REQUIRED (O-5) | Losses are >= 0, so VaR >= 0. All 3 options are implementable. | none |
| C.6 | CONSISTENT | `ObligorAttribution` arrays follow the loss-matrix row order (`_attribution.py`). `loss_histogram`: edges span [0, max(p99.9, max loss, 1)] and losses are >= 0, so the counts sum to S. | none |
| C.7 | CONSISTENT | Overlay key "forced_default" in `run._build_loss_matrix`. Model keys come from `JtdModel(...).name` (`summary.py`). The overlay is empty when the model is forced_default, so no collision exists. | none |
| C.8 | CONSISTENT | The D-C15 release-condition fields are all in A_CONTRACT 3.10. | none |
| C.9 | CONSISTENT | O-2; D-C15. | R-06 |
| C.10 | DEFECT | `run_identity.schemas._canonicalize` keeps the sign of zero. | A-08 |
| C.11 | CONSISTENT | sandbox P-JSON: model equal, dump bytes equal, every float bit equal (including 5e-324 and -0.0), int 2**53 + 1 exact, dict key order kept. | none |
| C.12 | CONSISTENT | `mixed.py` exists at the stated path. | R-08 |
| C.13 | CONSISTENT | `mixed.iter_batches`: one `standard_normal` fill per batch; PIT for every other marginal. `run._simulate` batches by `_FACTOR_BATCH`. | none |
| C.14 | OWNER APPROVAL REQUIRED (O-8) | Tuple fields interact with strict mode. | R-03 |
| C.15 | CONSISTENT | sandbox P-PYD: a frozen model still allows `r.params["rho"] = 7.0`; `model_validate(instance)` returns the same object without validation. The implementer must re-validate from a dump. | none |
| C.16 | CONSISTENT | O-1. | none |
| C.17 | CONSISTENT | Engine rules in `credit/jtd/ingest/validate.py:_semantic_mf_gaussian`: duplicate `(obligor_uid, factor_name)` rejected; `by > 1.0 + 1e-9` rejected. Request rule equals engine rule. | R-02, R-05 |
| C.18 | OWNER APPROVAL REQUIRED (O-8) | | R-03 |
| C.19 | OWNER APPROVAL REQUIRED (O-12) | Both options are implementable. | none |
| C.20 | DEFECT; OWNER APPROVAL REQUIRED (O-9) | `riskquant.exceptions.RiskQuantError(Exception)`: no ValueError base, so E can meet "not ValueError" under both options. Loc requirement fails. | A-09 |
| C.21 | OWNER APPROVAL REQUIRED (O-7) | | A-05, A-12, R-18, R-19 |
| C.22 | CONSISTENT | `catalog.py` guard uses `set(MODEL_CATALOG)` and works with a read-only mapping. No `monkeypatch`, `setitem`, `setattr` or catalog item assignment in tests/credit/solutions (G.3). | none |
| C.23 | DEFECT; OWNER APPROVAL REQUIRED (O-10) | | A-03 |
| C.24 | CONSISTENT | 9 JTD kernels: `prange` over obligors, disjoint output rows. Tail kernels: `prange` over groups or components, serial sums. No cross-thread reduction. [Observed; invariance Inferred] | R-20 |
| C.25 | OWNER APPROVAL REQUIRED (O-10) | | R-04 |
| C.26 | DEFECT | A_ORACLES rules 5, 6, 7. Kernel formulas match (section E). | A-01, A-10, A-11 |
| C.27 | CONSISTENT with gap | | R-14 |
| C.28 | DEFECT | No frozen name list exists in the document or the appendices. | A-04 |
| C.29 | CONSISTENT with gap | | R-09 |
| C.30 | CONSISTENT (O-2); OWNER APPROVAL REQUIRED (O-11) | `pyproject.toml`: numpy>=1.26, scipy>=1.15, pydantic>=2.0, numba>=0.59, version "0.1.0". | R-10, R-11 |
| C.31 | DEFECT under O-4(a) and O-4(b); OWNER APPROVAL REQUIRED (O-4) | | A-06, A-07, R-16 |
| C.32 | CONSISTENT | | R-17 |
| P-001.1 | DEFECT | | A-04, A-05 |
| P-001.2 | CONSISTENT | O-2. | none |
| P-001.3 | CONSISTENT with gap | | R-07 |
| P-001.4 | CONSISTENT | O-2. | R-17 |
| P-001.5 | CONSISTENT with D-C15 and O-2 | Section D. | R-06 |
| P-001.6 | CONSISTENT | D-C15 text unchanged. | R-12 |
| P-001.7 | CONSISTENT | numpy<3 and scipy<2 are the next majors. | none |
| P-001.8 | OWNER APPROVAL REQUIRED (O-9) | Base class OK (C.20 row). | A-09 |
| P-001.9 | DEFECT under O-10(c) | | A-03 |

Owner gates (each is OWNER APPROVAL REQUIRED):

| Gate | Option that triggers a finding |
|---|---|
| O-4 | (a) A-06; (b) A-07; (b) and (c) R-16 |
| O-5 | none |
| O-6 | (a) and (c) R-17 |
| O-7 | (a) R-18; (b) A-05; (c) A-12 |
| O-8 | (a) R-03 |
| O-9 | (a) and (b) A-09 |
| O-10 | (a) A-03 (host condition), R-04; (b) R-04; (c) A-03 |
| O-11 | R-09, R-10 |
| O-12 | none |
| O-13 | "join", combined with O-4(b): A-07 |

## C. Findings

### C.1 BLOCKING

A-01 | BLOCKING | A_ORACLES section 1 rule 5 (ES bound)
Fact: s_ES is the standard deviation of L under the tail measure Q. It is not the standard error of the engine ES estimator. [Observed: sandbox P-MC]
Evidence:
- Engine ES = (sum of the floor(m) worst losses + fractional_weight x VaR) / m, m = (1 - a) S (`_tail_metrics.py:_kernel_lower_tail_var_es`). [Observed]
- With VaR fixed at the atom v by rule 4: ES_hat = v + (1/m) sum (L_i - v)^+. Then Var(ES_hat) = [s_ES^2 + a (ES - v)^2] / ((1 - a) S). Rule 5 omits a (ES - v)^2. [Inferred, analytic]
- P-MC, sample sd of ES_hat over 600 replications: K2 a=0.975: 0.01843 (corrected formula 0.01900; rule 5 0.00926). K3 a=0.95: 0.02143 (0.02176; 0.01356). K13 a=0.95: 0.15625 (0.15954; 0.11916). [Observed: sandbox]
- P-ORA, rule 5 bound in true standard errors at levels that pass rule 4: K2 a=0.975: 2.92. K4 a=0.99: 3.71. K3 a=0.95: 3.74. K5 a=0.975: 3.79. All others 3.83 to 5.69. [Observed: sandbox]
Consequence: The bound is about 2.9 standard errors, not 6, at its weakest point. Two-sided exceedance is about 0.35 % for that assertion [Inferred, normal approximation]. The seed is fixed, so a correct engine can fail S4.T*. Rule 7 then forces a book change.
Action: Rule 5 must bound the standard error of the engine ES estimator, including the a (ES - VaR)^2 term.

A-03 | BLOCKING | C.23, S3.T2, P-001 clause 9 under O-10(c) and O-10(a)
Fact: C.23 requires byte-equal results from 4 concurrent threads, without condition. O-10(c) declares one call at a time per process. Under O-10(a), a host without a thread-safe layer must fail loud, so C.23 gives an error there. [Observed: clause text]
Evidence: F-17 [Observed]: `parallel_njit` and `stable_math_parallel_njit`, `nogil=True`; 0 hits for `threading_layer` or `THREADING_LAYER` in all 30 profiles. F-18 [Inferred, Numba documentation]: the workqueue layer terminates the process on concurrent entry.
Consequence: Under (c), C.23 and S3.T2 test a behavior that the policy excludes, and S3.T2 can abort the child process. Option (c) also fails the owner property "safe concurrent calls in one process" (section 1).
Action: Make C.23, S3.T2 and clause 9 conditional per O-10 option, or the owner removes option (c).

A-04 | BLOCKING | C.28, S5.T1, P-001 clause 1
Fact: "the frozen surface list" is not enumerated anywhere. A_CONTRACT adds AlphaRisk and PercentileLoss. O-9 adds R and E, and CreditVarError under (b). No clause states which of these `api.__all__` exports. [Observed]
Evidence: WS sections 3, 4 and 6; A_CONTRACT sections 1 to 4.
Consequence: The S5 implementer defines the public SemVer surface. That is an owner decision.
Action: State the frozen name list for each outcome of O-6, O-7 and O-9.

A-05 | BLOCKING | P-001 clause 1 under O-7(b)
Fact: Clause 1 defines the surface as the names in `api.__all__`. `from_dataframe` is a member of `CreditVarRequest`, which is in the list. Clause 3 makes an incompatible signature change breaking. [Observed]
Consequence: Under O-7(b), no clause can say "class stable, one member outside". A later change of `from_dataframe` is breaking under clause 3, against O-7(b).
Action: Under O-7(b), add a member-level exemption to clause 1, or the owner removes option (b).

A-06 | BLOCKING | C.31, S5.T4 under O-4(a)
Fact: STABLE_API.md calls itself "the frozen public surface of the credit modelling core", "supported and versioned". [Observed: STABLE_API.md, Policy]
Evidence: D001: "core packages are private". P-001 clause 1: `riskquant.credit` is internal. S5.T4 option (a) checks only the name list and the paths.
Consequence: Under (a), the file passes S5.T4 and still contradicts D001 and P-001 clause 1 after freeze.
Action: Under (a), C.31 must also require the file to state internal status and to drop the public, stable and versioned claims.

A-07 | BLOCKING | P-001 Scope under O-4(b); interaction with O-13
Fact: O-4(b) merges the model-parameter contracts (beta-binomial, C-07 copulas) into P-001. The P-001 Scope line lists surface, versioning, build identity, errors and concurrency only. STABLE_API.md cites D-C18. DECISIONS.md has no D-C18 heading (G.3). [Observed]
Consequence: Under (b), the P-001 Scope line is wrong. If O-13 sends D-C18 to the O-3 triage item, O-4(b) and O-13 conflict.
Action: Under (b), widen the Scope line and state the D-C18 disposition. The owner closes O-4 and O-13 together.

A-08 | BLOCKING | C.10, second sentence
Fact: C.10 states that two requests that validate to equal models give one hash. The A_CONTRACT 4.2 form uses `float.hex()`, which keeps the sign of zero. Pydantic model equality uses float `==`. [Observed: sandbox P-PYD: `Z(ead=-0.0) == Z(ead=0.0)` is True; hex `-0x0.0p+0` against `0x0.0p+0`]
Evidence: -0.0 passes `ead` (ge=0.0), `pd`, `lgd`, `rho`, `loading`, and params with an inclusive lower bound of 0: rho, stress_prob, systemic_pd (`ParamSpec.contains`). [Observed]
Consequence: The sentence is false for signed zero. A property test of the sentence fails.
Action: Owner chooses: normalize -0.0 in the canonical form (this diverges from D040 `_canonicalize`), or restate C.10 as "field-wise bit-equal models".

A-09 | BLOCKING | C.20, "carries a cell-addressable loc"
Fact: Every cross-field rule is a `@model_validator(mode="after")` that raises ValueError (`schemas.py`). [Observed] Such an error has loc `()`. [Observed: sandbox P-PYD: `[('value_error', ())]`] The new C.17 rules are cross-row rules. No test checks loc (S2.T10, S2.T11).
Consequence: Current code does not meet the clause. The clause is unverifiable as written. It does not define the cell for a fault that spans two rows.
Action: Define the loc for each rule. Add a loc assertion to S2.T10.

A-10 | BLOCKING | A_ORACLES rule 6 for K9, K10, K12
Fact: At the node counts that A_ORACLES section 2 specifies, the oracle EL misses N x PD by more than the rule 6 tolerance of 1e-8. [Observed: sandbox P-ORA, P-QUAD]
Evidence:
- K9 and K10 (120 Gauss-Laguerre nodes): EL - 0.4 = -3.30e-08. With 160 nodes: -1.05e-08. With 200: -4.33e-09. With 300: -8.60e-10.
- K12 (60 Gauss-Laguerre nodes): EL - 0.4 = -5.14e-07. With 200 nodes: -4.33e-09.
- Convergence with the node count shows that the implementation is correct and the specified counts are too low. The integrand depends on sqrt(w), so Gauss-Laguerre converges slowly.
- K7, K8, K11, K13 pass rule 6 (errors 1.7e-16 or less in magnitude, up to 6.7e-16).
Consequence: S4.T9, S4.T10 and S4.T12 fail their own oracle self-check on a correct implementation. The implementer must change the quadrature, which A_ORACLES fixes.
Action: Set node counts (or a change of variable) that meet rule 6, or set a rule 6 tolerance that the specified quadrature meets.

A-11 | BLOCKING | A_ORACLES rule 7 for K12
Fact: Only the level 0.95 passes rule 4 for the K12 book. [Observed: sandbox P-ORA]
Evidence: a=0.975: VaR 4, F(3) = 0.974759 > a - d = 0.972038. a=0.99: F(5) = 0.991304 < a + d = 0.991888. a=0.995: F(6) = 0.994792 > a - d = 0.993662. a=0.999: fails. The F values are stable to 6 decimals with 200 Gauss-Laguerre nodes.
Consequence: Rule 7 makes S4.T12 fail ("fewer than 2 pass"). Rule 7 then requires a book change, which is a change of a frozen appendix.
Action: Change the K12 book before freeze, so that at least 2 candidate levels pass.

A-12 | BLOCKING | O-7(c): no slice owns the removal of `from_dataframe`
Fact: C.21 covers only the "keep" options. No slice lists the removal under O-7(c). `from_dataframe` is called by 5 test files. [Observed]
Evidence: tests_credit profile lines 1770 (`tests/credit/factors/test_batch_size_reproducibility.py`), 2783 to 2880 (`tests/credit/jtd/test_beta_binomial_contract.py`), 222, 394, 498 and 585 (the three files under `tests/credit/solutions/`). Of these, only `test_batch_size_reproducibility.py` is in a Files list (S1).
Consequence: Under O-7(c), the slice that removes the method must leave its Files list. The implementer must stop.
Action: Under O-7(c), name the slice and add the 4 test files to its Files list.

### C.2 REFINEMENT

A-02 | REFINEMENT (reclassified from BLOCKING in the first session) | A_ORACLES rule 5 at a single-atom tail
Fact: K13 passes rule 4 at 0.99, 0.995 and 0.999 with VaR = N = 20 and no mass above VaR, so s_ES = 0 in exact arithmetic. [Observed: sandbox P-ORA]
Evidence: P-DEG, 6 evaluation variants: s_ES^2 is exactly 0 or between +4.4e-12 and +1.3e-10. The oracle ES is exactly 20 or 20 minus up to 6.7e-12. For K2 at a=0.999, the same expression gives s_ES^2 = -5.30e-12; that level fails rule 4. Engine ES at K13 is exactly 20.0: m snaps to an integer at S = 100_000 for all 5 candidates, and the losses are integers. [Inferred]
Consequence: No false failure is shown for K13 at the specified levels. The rule has no clamp at 0 and no floating floor, so a negative s_ES^2 gives NaN and a failed comparison.
Action: Add max(s_ES^2, 0) and an absolute floating floor to rule 5.

R-01 | F-30 and A_ORACLES rule 4. The VaR 0-based index is ceil(m) - 1, m = (1 - a) S after a 16-eps snap (`_computer._compute_lower_tail_parameters`). [Observed] The `loss_metrics.py` docstring "floor((1 - alpha) * S)" is wrong for an integral m. In P-MC, the re-implemented estimator VaR equals the oracle VaR in 600 of 600 replications for each of 3 cases. Raise F-30 to Observed with the index. Replace "whatever order-statistic index" in rule 4 with the observed index.
R-02 | F-12, uid case. sandbox P-UID (pandas 3.0.2, numpy 2.4.4), with the `recipes._uids` expression: uid 2**63 becomes -2**63 and uid 2**64 - 1 becomes -1, silently. uid 2**70 and -2**63 - 1 raise OverflowError, which is not a ValueError. F-12 says ValueError. Correct F-12. C.17 removes both failure modes.
R-03 | O-8(a). sandbox P-PYD: strict mode rejects a Python list for a tuple field (`tuple_type`). It accepts a JSON array through `model_validate_json`. It rejects `json.loads(...)` passed to `model_validate`. The `run.py` docstring advertises a plain dict. `from_dataframe` builds lists internally. State the accepted Python sequence types under O-8(a) and add an S2 test row.
R-04 | C.25, S3.T4. "O-10 behavior" defines no observable outcome. Under (a), the layer is fixed at the first parallel launch; state when the check runs. Under (b), the lock covers credit_var calls only.
R-05 | C.17 says "rejected, or isolated", so the implementer chooses the contract. Today a collision fails loud: `validate_factor_names` rejects a duplicate `factor_name` in the factor universe with ValueError. [Observed] Reachable collisions for the mf keys: `IDIO_{uid:03d}` and `CHI_GLOBAL` (`recipes.py`). The owner picks one behavior.
R-06 | P-001 clause 5. State the root of "relative path" and the sort key. V is not bound to the imported code: pytest `pythonpath = ["src", "."]` can import src/ beside a different installed distribution. H is computed at the first call, not at import.
R-07 | P-001 clause 3. "fixed numerical environment" is not defined. If it includes engine_version, the value-change item never applies.
R-08 | C.12 hashes `mixed.py` only. Factor order from `recipes._universe`, the factor-universe compile and `base.py` also define the stream. State that run identity relies on H.
R-09 | C.29. `model_json_schema()` output depends on the pydantic version. Link C.29 to O-11.
R-10 | O-11. "Next major" is not defined for numba 0.x. Numba minor releases change codegen.
R-11 | pandas has no upper bound but is on the api import path (F-23; `recipes.py` and `run.py` build frames). [Observed]
R-12 | A_CONTRACT 4.3 compares JSON bytes. The pydantic-core serializer version is not recorded.
R-13 | S1 FIRST searches src/, scripts/ and tests/ only. `src/riskquant_lab/plotting/credit_var.py` takes `var_levels: dict[str, float]` [Observed]; the token search does not find it. The text of `tests/e2e/test_demo_scripts_are_runnable.py` is in no profile; it can run the credit_var demos [Unresolved]. Search the whole repository.
R-14 | C.27. The import allow-list omits `tests/solutions/credit_var/_oracle_pmf.py`, which the S4 Files create.
R-15 | A_ORACLES rule 4 pass sets (sandbox P-ORA). K2: 0.95, 0.975, 0.99, 0.995. K3: 0.95, 0.975, 0.99. K4: 0.95, 0.99. K5: 0.95, 0.975, 0.99, 0.995. K6: 0.95, 0.995. K7: 0.95, 0.99, 0.995. K8: 0.95, 0.995. K9: 0.95, 0.99. K10: 0.95, 0.99. K11: 0.95, 0.975. K12: 0.95 only (A-11). K13: 0.95, 0.99, 0.995, 0.999. Record the sets in A_ORACLES, so that S4 does not re-derive them.
R-16 | O-4(b), O-4(c). The `mixed.py` docstring says "see STABLE_API.md, C-07". [Observed] Deletion leaves a dangling reference in a prohibited file that C.12 hashes.
R-17 | O-6(c): a new wire model has no schema version (clause 4) and no golden file (C.29). O-6(a): S6 measurement 4 has no public target.
R-18 | O-7(a) puts a pandas type in the SemVer surface. A later removal of pandas is breaking under clause 3. CLAUDE.md section 4 treats pandas as migration debt.
R-19 | C.21. The `loadings_table` docstring also says `uid`; the code requires `obligor_uid`. [Observed] S2.T12 must cover both docstrings.
R-20 | C.24 and S3.T3 change `NUMBA_NUM_THREADS` only. `_attribution.obligor_ul_contribution` uses BLAS through `@` (matrix-vector and dot). [Observed] D041 records that a BLAS matrix-vector reduction moved 1.4e-13 with blocking width. State whether BLAS thread settings are inside the C.13 and C.24 contract.

### C.3 Adjacent (not in scope, not blockers)

- `simulate_portfolio_losses` subsamples with `default_rng(req.seed)`, the seed of the factor generator. [Observed; statistical dependence Inferred]
- `model_tail_var_es` keeps one entry per model name if two blocks share a JtdModel; `_expected_loss_by_model` sums them. Not reachable from credit_var. [Inferred]
- The `loss_histogram` docstring describes an overflow fold. The branch is unreachable because xmax >= the maximum loss. [Observed]

## D. Decision reconciliation

| ID | Result |
|---|---|
| D001 | P-001 clause 1 is consistent. STABLE_API.md conflicts with it (A-06). |
| D002 | `api.py` is a named module; the `__init__.py` files in the profile are empty. Consistent. |
| D-C15 | P-001 clauses 5 and 6 and A_CONTRACT 3.10 carry the construction id, seed, scenario count, NumPy and SciPy versions. "+src.<H>" separates pre- and post-D-C15 builds. The release condition is met by design. |
| D-C04 | F-20 Observed: `build_credit_factor_matrix` calls `new_scenario_coverage`, `record_scenario_span` per batch and `assert_scenario_coverage_complete` (`base.py`). |
| D-C03 | F-21 Observed: `run_jtd_for_obligor_ead` calls `align_obligor_ead_to_portfolio`, which uses `searchsorted_or_raise`. |
| D039 | C.17 seed >= 0 follows the precedent. |
| D040 | A_CONTRACT 4.2 equals `_canonicalize` with the `json.dumps` settings of `RunManifest.canonical_bytes`. Signed zero: A-08. |
| D041 | C.24 is consistent with the kernel structure. The S3 exit on a multi-core owner machine meets the D041 open item. BLAS: R-20. |
| D104 | The D104.3 reason "pydantic is not declared" is stale: `pyproject.toml` declares pydantic>=2.0. Owner note only. |
| D-C18 | No heading in DECISIONS.md. STABLE_API.md cites it. |

## E. A_ORACLES and VERIFY FIRST

VERIFY FIRST result: the default direction is the lower tail in every kernel. The Student-t scaling matches K9 and K12. F-18 stays Inferred. F-30 becomes Observed with the R-01 correction. [Observed]

| Key | Engine evidence | Match |
|---|---|---|
| K1 | `kernel_forced_default_jtd`: `ead * lgd` | yes |
| K2 | `u < pd` | yes |
| K3, K4 | `u < P`, P = betaincinv(alpha, beta, Phi(Z)); one BETA_PD factor per pool | yes |
| K5, K6 | indicator = 1{Z < Phi^-1(q)}; `min(pd * m, 1)` when indicator > 0.5 | yes |
| K7, K8 | `b z + sqrt(1 - b^2) eps < Phi^-1(clip(PD))`, b = sqrt(rho); `basel_corporate_correlation` equals the K8 R(PD) | yes |
| K9, K10 | chi column = 1/sqrt(W), W = 2 gammaincinv(nu/2, Phi(Z)); `sqrt(nu) x (1/sqrt(W)) x A < t.ppf(clip(PD), nu)` | yes |
| K11 | idio weight sqrt(max(0, 1 - sum beta^2)); Gaussian threshold | yes |
| K12 | K11 with the K9 scaling and one shared W | yes |
| K13 | default when u < p_idio OR u_sys < s (bitwise OR in the kernel), p_idio = 1 - (1 - PD) / (1 - s) | yes |

PD_DUST_TOLERANCE = 1e-10 clipping does not act at PD = 0.02 or 0.05.

Rule results (sandbox P-ORA):

| Rule | Result |
|---|---|
| 1, 2, 3, 8 | Consistent. |
| 4 | Holds for the engine index (R-01). Pass sets: R-15. |
| 5 | Bound too narrow (A-01). No clamp or floor (A-02). |
| 6 | pmf sums within 1e-10 for all keys. EL check fails for K9, K10, K12 (A-10). |
| 7 | Fails for K12 (A-11). All other keys have 2 or more levels. |

## F. Freeze readiness

Verdict: NOT READY.

Blocking: A-01, A-03, A-04, A-05, A-06, A-07, A-08, A-09, A-10, A-11, A-12.

Owner approval only:
- O-4, O-5, O-6, O-7, O-8, O-9, O-10, O-11, O-12, O-13.
- The owner choices inside A-03 (O-10 option set), A-05 and A-12 (O-7), A-06 and A-07 (O-4, O-13), A-08 (canonical form) and R-05 (collision behavior).

Refinements for the revision or the backlog: A-02, R-01 to R-20.

## G. Coverage

### G.1 Clauses covered

C.1 to C.32, P-001 clauses 1 to 9, O-4 to O-13, slices S1 to S6, A_ORACLES K1 to K13 and rules 1 to 8.

### G.2 Clauses not covered

None at clause level. Remaining evidence limits:
- The c060f06 to 925533b delta (A.1).
- The text of WS_TEMPLATE.md (A.1).
- The text of `tests/e2e/test_demo_scripts_are_runnable.py` and of the 15 demo scripts (R-13).
- Numba behavior was not executed: Numba is not in the sandbox (A-03, C.24).
- F-24 function count: call sites seen are 1 in `test_batch_size_reproducibility.py` and 6 in `test_beta_binomial_contract.py`; functions not counted.
- Pydantic serializer change across versions (R-12) not probed.

### G.3 Negative claims and their search scope

Each item gives the claim, the exact command, the files searched and the hits. Commands ran in `/mnt/project` unless stated.

1. No Numba threading-layer setting.
   Command: grep -n "threading_layer\|THREADING_LAYER" *.txt
   Files: all 30 project profiles. Hits: 0.
2. References to STABLE_API.md.
   Commands: grep -n "STABLE_API" *.txt ; grep -rn "STABLE_API" /mnt/user-data/uploads/context_riskquant_tests_credit.txt
   Files: all 30 profiles; tests_credit. Hits: credit profile line 3548 (`mixed.py` docstring), index line 48 (manifest), project_structure.txt line 487.
3. No removed-field reader in riskquant_lab.
   Command: grep -n "summary\.\|loss_percentiles\|per_model_tail\|\.var\b\|get_var\|get_es\|max_alpha\|es_to_var\|economic_capital\|obligor_contributions\|loss_histogram\|provenance" context_riskquant_lab.txt
   Files: context_riskquant_lab.txt. Hits: 1, a docstring (line 943).
4. Test callers of run_credit_var.
   Command: grep -n "run_credit_var\|simulate_portfolio_losses\|summary\.\|loss_percentiles\|per_model_tail\|credit_var\.\(run\|api\|schemas\|catalog\|recipes\)\|from riskquant.solutions" context_riskquant_tests_credit.txt
   Files: tests_credit. Hits: `dict(summary.var)` at line 1782; the other calls read `summary.loss_mean` only; no `simulate_portfolio_losses`.
5. No catalog mutation in tests/credit/solutions.
   Command: grep -n "monkeypatch\|MODEL_CATALOG\[\|SUPPORTED_RECIPES\[\|setitem\|setattr\|alphas=\|alphas\"\|\[$\|obligors=\[\|\"obligors\": \[\|loadings=\[\|model_validate\|CreditVarRequest(" /tmp/sol_tests.txt
   Files: the 3 files under tests/credit/solutions, extracted from tests_credit. Hits: 4, all `alphas=[0.99]` arguments to `from_dataframe`; 0 mutation hits.
6. Callers of from_dataframe.
   Command: grep -n "from_dataframe" context_riskquant_tests_credit.txt
   Files: tests_credit. Hits: 5 CreditVarRequest test files (A-12) and 1 CreditExposureRequest test (out of scope).
7. No D-C18 decision.
   Command: grep -n "^#\+ \|^## D\|^### D" DECISIONS.md | grep -n "D001\|D002\|D-C15\|D-C04\|D-C03\|D039\|D040\|D041\|D104\|D-C18\|D-C1\|D105"
   Files: DECISIONS.md. Hits for D-C1x: D-C15 and D-C16 only.
8. Text of WS_TEMPLATE.md absent.
   Command: grep -l "WS_TEMPLATE" *.txt /mnt/user-data/uploads/*
   Files: all profiles and uploads. Hits: the name only, in project_structure.txt, REVIEWER.md and WORKFLOW.md.
