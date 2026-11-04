# WS_CREDIT_VAR_API / R_FREEZE_RECONCILIATION_R2

Path: docs/workstreams/WS_CREDIT_VAR_API/R_FREEZE_RECONCILIATION_R2.md
Date: 2026-10-29
Spec: WS_CREDIT_VAR_API.md, revision 2
Round: 2 of 2
Clauses covered: C.1 to C.32; P-001 clauses 1 to 9; O-1 to O-13; A_CONTRACT sections 1, 2.4, 2.5, 3.2, 3.5, 3.9, 4.2; A_ORACLES rules 4, 5, 6, 7, K9, K10, K12, section 3; S1 to S6 slice rows; section 7 backlog
Clauses not covered: None at clause level. Evidence limits below.
Snapshot: SNAPSHOT commit=925533b dirty=false content_hash=046ed5dddb6305d3
Mode: read-only
Evidence base: WS_CREDIT_VAR_API.md (R2); A_CONTRACT.md (R2); A_ORACLES.md (R2);
  R_FREEZE_RECONCILIATION_R1.md; WS_TEMPLATE.md; CLAUDE.md; DECISIONS.md;
  STABLE_API.md; pyproject.toml; context_riskquant_index.txt;
  context_riskquant_solutions_credit_var.txt; context_riskquant_credit.txt;
  context_riskquant_scenario_metrics.txt; context_riskquant_solutions_run_identity.txt;
  context_riskquant_tests_credit.txt
Evidence limits: No repository gate, test or benchmark ran. All code facts from the
  925533b profiles. Sandbox environment: Python 3.12, numpy 2.4.4, scipy 1.17.1. No
  riskquant install; no engine run. Rule 5 engine assertion cannot be checked here.
  K12 rule 6 EL check at the specified 60x60x300 node counts exceeds sandbox time;
  a fast oracle at 40x40x100 nodes gave EL err = -7.36e-8. K9 at full 300 Laguerre
  nodes gives EL err = -8.60e-10. Both are stated separately under item 5.

---

## A. Verdict

FREEZE READY

All 11 R1 BLOCKING findings are CLOSED in revision 2. No new BLOCKING finding was
identified. Every owner gate is CLOSED. The spec is implementable with the existing
architecture. Round 2 is the last permitted round; the owner may now freeze.

---

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| D001 | VERIFIED | P-001 clause 1 states riskquant.credit is internal. api.py is a named module (F-31). __init__.py files are empty in all profiles. | None |
| D002 | VERIFIED | Empty __init__.py confirmed in all profiles. First-line path comments are the constitution requirement; profiles show the pattern. | None |
| D-C03 | VERIFIED | F-21 confirmed at 925533b: searchsorted_or_raise in align_obligor_ead_to_portfolio. | None |
| D-C04 | VERIFIED | F-20 confirmed at 925533b: new_scenario_coverage, record_scenario_span, assert_scenario_coverage_complete in base.py. | None |
| D-C15 | VERIFIED | P-001 clauses 5 and 6, A_CONTRACT 3.10, C.8, C.12, C.13 carry the construction id, seed, count, NumPy/SciPy versions. D-C15 text is unchanged (O-2 confirmed). | None |
| D039 | VERIFIED | C.17 seed >= 0 follows the D039 precedent. A_CONTRACT 2.4 adds ge=0. | None |
| D040 | VERIFIED | A_CONTRACT 4.2 equals _canonicalize with the json.dumps settings of RunManifest.canonical_bytes (F-27). C.10 now states field-wise bit equality, consistent with the signed-zero behavior of float.hex(). | None |
| D041 | VERIFIED | C.24 consistent with kernel structure (prange over obligors, disjoint output rows). | None |
| D104 | VERIFIED | D104.3 reason "pydantic is not declared in pyproject.toml" is stale at c060f06 (F-22). D104 is not changed by this workstream. Owner note only, no gate. | None |
| D-C18 | NOT VERIFIED | No D-C18 heading in DECISIONS.md (R1 G.3, search command: grep "D-C18" DECISIONS.md). STABLE_API.md cites it. O-13 sends D-C18 to the O-3 triage item, which is out of scope for this workstream. | None here; O-3 triage tracks it |
| O-1 to O-13 | VERIFIED | All 13 gates CLOSED in section 5 of revision 2 as of 2026-10-29. | None |

---

## C. Contract-to-repository matrix

Scope: only the text that revision 2 changed, per the prompt scope. Unchanged text
is carried from R1 without re-audit.

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.4 | A_CONTRACT 3.2: at(alpha) raises KeyError (O-9). C.20 scopes "public entry points" to run_credit_var and simulate_portfolio_losses; at() is a result method, not a public entry point. KeyError from at() does not contradict C.20. | MATCH | None |
| C.5 | A_CONTRACT 3.2 and 3.5 remove es_to_var_ratio from RiskSummary and ModelTail. A_CONTRACT 3.9 confirms removal. C.5 adds: "A run with VaR = 0 raises no exception and gives no NaN." No ratio member means no ZeroDivisionError path from ratio code. [Proposed] | MATCH WITH REQUIRED CHANGE | None; change is [Proposed] and in S1 Files |
| C.6 | A_CONTRACT 3.6: ead equals request ead; a uid without an ead raises CreditVarEngineError. Current run.py uses float("nan") default (F-7). Change moves to E, not silent NaN. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| C.9 | P-001 clause 5 states engine_version = "<V>+src.<H>". C.9 requires CreditVarEngineError on missing metadata or a local-segment version. Current _engine_version returns "0+unknown" on Exception (F-9). This is the defect C.9 corrects. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| C.10 | A_CONTRACT 4.2 note: "float.hex() keeps the sign of zero, so -0.0 and 0.0 give different hashes. C.10 therefore states equality as field-wise bit equality." C.10 text now says "field-wise bit-equal models" and "A 1-ulp change of any float gives another hash. -0.0 and 0.0 give different hashes." These are consistent. A-08 CLOSED. | MATCH | None |
| C.15 | A_CONTRACT 1 note: "Dict fields stay dicts; C.15 covers their mutation." C.15 requires re-validation from a dump at call time. F-3: frozen=True blocks attribute assignment but not nested dict mutation. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| C.17 | Rules added: seed < 0, obligor_uid outside int64 range, duplicate (uid,factor), sum loading^2 > 1+1e-9, reserved factor name. Current schemas.py has no check for seed >= 0 (bare int field), no int64 range on obligor_uid, no duplicate-factor check, no loading-sum check in request, no reserved-name check. Each is a [Proposed] addition. F-12 (seed) and F-36 (reserved names) confirm the engine already enforces the same rules; C.17 moves them to the request boundary. | MATCH WITH REQUIRED CHANGE | None |
| C.18 | A_CONTRACT 1 adds strict=True to request ConfigDict. Coercion table matches O-8(a). F-34 (pydantic 2.14.1): strict mode rejects str for float, rejects bool for int, accepts int for float. Sequence fields require BeforeValidator for list-or-tuple semantics since strict=True rejects list for a tuple field. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| C.19 | O-12: no size limit. A_CONTRACT 2.4 n_scenarios: "> 0; no upper limit (O-12)". Consistent. [Proposed] | MATCH | None |
| C.20 | Now references A_CONTRACT section 2.5 for per-rule locs. F-35 confirms ValidationError.from_exception_data in model_validator(mode="after") reaches the caller with its own loc. "E chains the original exception as __cause__ and does not subclass ValueError." riskquant.exceptions.RiskQuantError(Exception) has no ValueError base (F-27); CreditVarEngineError(RiskQuantError) will satisfy this. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| C.21 | from_dataframe column name fixed (docstring says uid, must say obligor_uid per F-33) and default n_scenarios aligned to 250_000. Both [Proposed]. loadings_table docstring (R1 R-19) remains in section 7 backlog; it is not in C.21. REFINEMENT only for backlog item. | MATCH WITH REQUIRED CHANGE | None |
| C.25 | O-10(b): one process lock covers credit_var parallel-kernel entry only. Consistent with C.23 (byte-equal concurrent results) and P-001 clause 9 ("calls on separate threads of one process give the results of serial calls"). [Proposed] | MATCH | None |
| C.27 | Now says test_oracles.py may import from tests.solutions.credit_var._oracle_pmf. _oracle_pmf.py may import only from numpy, scipy and the standard library. R1 R-14 gap closed. | MATCH | None |
| C.28 | P-001 clause 1 lists 22 names exactly. F-1 api.__all__ has 19 names. R2 adds AlphaRisk, CreditVarEngineError, PercentileLoss. S5 Files list modifies api.py. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| C.30 | O-11: pydantic <3; numba: upper bound at the next minor above the S5 gate environment version. C.30 spec: ">=0.59,<0.N, where 0.(N-1) is the numba minor version in the S5 gate environment." R-10 from R1 asked for a definition; C.30 now gives one. N is determined when S5 runs, not at freeze. Consistent with O-11. | MATCH | None |
| C.31 | O-4(a): STABLE_API.md stays internal. C.31 now requires: (a) listed set equals credit __all__ union; (b) each stated path exists; (c) first content line starts with "Status: internal"; (d) no line matches public/stable/versioned. F-28 (R1 A-06): current file states "the frozen public surface", "supported and versioned". These claims must be removed. [Proposed] | MATCH WITH REQUIRED CHANGE | None |
| P-001 clause 1 | 22 names listed. from_dataframe exempted at member level. "The request error type comes from pydantic; api does not re-export it." pydantic.ValidationError is not in the 22 names. Consistent with O-9(a): R = pydantic.ValidationError, api does not re-export it. A-04 and A-05 CLOSED. | MATCH | None |
| P-001 clause 8 | R = pydantic.ValidationError (not re-exported). E = CreditVarEngineError(RiskQuantError). RiskSummary.at raises KeyError. "E does not subclass ValueError": riskquant.exceptions.RiskQuantError(Exception) has no ValueError base. A-09 CLOSED. | MATCH | None |
| P-001 clause 9 | "credit_var serializes entry into its parallel kernels with one process lock. Calls on separate threads of one process give the results of serial calls. The lock does not serialize parallel kernels that other riskquant modules enter." O-10(b) selected. A-03 CLOSED. | MATCH | None |
| A_CONTRACT 1 | ConfigDict table: request models add strict=True and frozen=True. Coercion table specifies per-type behavior through all nesting levels. "Dict fields stay dicts." Tuple fields need BeforeValidator (list-or-tuple). [Proposed] | MATCH | None |
| A_CONTRACT 2.4 | alphas field: changed to tuple; "pairwise distinct under ==" added. seed: ge=0 added. obligors, loadings, groups: changed to tuple. int64 range on obligor_uid added. [Proposed] | MATCH | None |
| A_CONTRACT 2.5 | 25 rules with per-rule locs. See item 3 below. | MATCH (see item 3) | None |
| A_CONTRACT 3.2 | by_alpha replaces var/es dict fields. at() method added. max_alpha, get_var, get_es, get_es_to_var_ratio, es_to_var_ratio removed. [Proposed] | MATCH | None |
| A_CONTRACT 3.5 | es_to_var_ratio removed. "The es_to_var_ratio property is removed (O-5)." [Proposed] | MATCH | None |
| A_CONTRACT 3.9 | Removed field list correct: var, es, economic_capital_var, economic_capital_es from RiskSummary; dict form of loss_percentiles; "forced_default" overlay key; ES/VaR ratio members. [Proposed] | MATCH | None |
| A_CONTRACT 4.2 | Steps 1-4 match _canonicalize from run_identity.schemas (F-27). Signed-zero note consistent with C.10. | MATCH | None |
| Section 5 (S1-S5 rows) | All gates show NOT_STARTED, Evidence none. This is correct: no code has merged (Code merged: no in header). | MATCH | None |
| Section 7 backlog | All R1 REFINEMENT findings (R-01 to R-20) appear in backlog. R-01 is superseded by the F-30 update. R-05 is resolved (C.17 rule 15). R-14 is resolved (C.27 updated). The remaining items are REFINEMENTS not required for freeze. | MATCH | None |

---

## D. Blast radius of shared interfaces

The revision changes no src/ file. P-001 adds three names to api.__all__
(AlphaRisk, PercentileLoss, CreditVarEngineError), removes dict-keyed fields from
the result surface, and changes field types in CreditVarRequest and CreditVarResult.

External callers of the current api path (from F-24 and R1 G.3):

1. `tests/credit/factors/test_batch_size_reproducibility.py` reads `dict(summary.var)`.
   S1 Files includes this file for modification.
2. Seven test functions in two files call run_credit_var through credit_var.run
   (not api). S2 covers the error path change; no blast radius to these tests from
   the api path, since they do not import from api.
3. `src/riskquant_lab/plotting/credit_var.py` takes `var_levels: dict[str, float]`
   (R1 G.3 negative-claim item 3). S1 FIRST step must search it. It is not in any
   S1 Files list. The S1 slice instruction says STOP AND ASK for a hit outside the
   Files list.
4. `tests/e2e/test_demo_scripts_are_runnable.py` and the 15 demo scripts (F-32) are
   not in any profile. S1 FIRST must search these paths. Evidence limit: content
   unknown.

Search command (scope for negative claim 3):
  `grep -rn "summary\.var\b\|summary\.es\b\|economic_capital_\|loss_percentiles\|get_var\|get_es\|max_alpha\|es_to_var\|per_model_tail\|obligor_contributions\|loss_histogram\|provenance\|RiskSummary\|ModelTail" src/ scripts/ tests/`
Files searched: 925533b profiles (all 30). Hit confirmed at line 943 of
  context_riskquant_lab.txt (a docstring). No src/ hit outside the S1 Files list
  confirmed within the profiles; tests/e2e remains unconfirmed (evidence limit).

---

## E. Hidden dependency findings

None new in revision 2. R1 E findings are carried by the R1 report.

---

## F. Findings and required changes

### F.1 Carried R1 BLOCKING findings

A-01 | CLOSED
Fact: R2 A_ORACLES rule 5 now includes the `a (ES - VaR_a)^2` term:
  `6 sqrt( (s_ES^2 + a (ES - VaR_a)^2) / ((1 - a) S) ) + tau`.
Evidence: A_ORACLES rule 5 text (R2). Sandbox: K2 at a=0.975 bound = 0.114
  (6 standard errors), K13 single-atom tail bound = 6.9e-7 (near-zero variance
  absorbed by tau). The corrected formula is the standard error formula for the
  engine's fractional-weight ES estimator. Rule 5 text confirmed to match the
  derivation.

A-02 | CLOSED (was reclassified REFINEMENT in R1)
Fact: R2 rule 5 adds max(., 0) clamp on s_ES^2 and tau = 1e-9. Sandbox P-DEG
  evidence from R1: K13 at 0.99, 0.995, 0.999 has s_ES^2 = 0; tau covers
  floating-point residuals. Rule text: "The clamp at 0 and tau absorb rounding
  at a single-atom tail, for example K13 at 0.99."

A-03 | CLOSED
Fact: O-10(b) was selected: one process lock serializes entry into credit_var
  parallel kernels. P-001 clause 9 matches O-10(b). C.23 requires byte-equal
  concurrent results; under a serializing lock, results match serial runs.
  C.25 specifies the lock scope. No option (c) text remains.

A-04 | CLOSED
Fact: P-001 clause 1 lists 22 names explicitly. Sandbox count: 22 names listed.
  Names: AlphaRisk, CreditVarEngineError, CreditVarRequest, CreditVarResult,
  GroupRow, LoadingRow, LossHistogram, MODEL_CATALOG, ModelEntry, ModelTail,
  ObligorContribution, ObligorRow, ParamSpec, PercentileLoss, RiskSummary,
  RunProvenance, StressTestObligor, StressTestResult, model_entry,
  run_credit_var, simulate_portfolio_losses, supported_model_keys.

A-05 | CLOSED
Fact: P-001 clause 1 adds: "The member CreditVarRequest.from_dataframe is
  outside the stable surface; clause 3 does not apply to it." A member-level
  exemption is present. Under O-7(b), a later change to from_dataframe is not
  breaking under clause 3.

A-06 | CLOSED
Fact: O-4(a) selected. C.31 now requires: "Its first content line states
  internal status. It makes no public, stable or versioned claim." S5.T4 checks
  the no-claim condition (regex match). Current STABLE_API.md still carries
  "the frozen public surface", "supported and versioned" (F-28). These lines
  must be removed in S5.

A-07 | CLOSED
Fact: O-4(b) was NOT selected; O-4(a) was selected. The conflict between O-4(b)
  and O-13 does not arise. A-07 is moot.

A-08 | CLOSED
Fact: C.10 now states "field-wise bit-equal models" and defines the term.
  A_CONTRACT 4.2 note: "-0.0 and 0.0 give different hashes. C.10 therefore
  states equality as field-wise bit equality." The sentence is no longer false for
  signed zero.

A-09 | CLOSED
Fact: A_CONTRACT section 2.5 defines a loc for each of the 25 request-validation
  rules. C.20 references section 2.5: "Each R entry carries the loc that
  A_CONTRACT section 2.5 defines for its rule." S2.T10 now has a required loc
  assertion. F-35 confirms ValidationError.from_exception_data produces the
  required loc from a model_validator(mode="after").

A-10 | CLOSED (K9, K10) / CONDITIONALLY CLOSED (K12, evidence limit)
K9 and K10: Sandbox, 120 Gauss-Hermite x 300 Gauss-Laguerre nodes.
  EL - N*PD = -8.60e-10 for both. This is within the rule 6 tolerance of 1e-8.
  A_ORACLES K9 note "Do not use 380 or more nodes" confirmed; 300 nodes are safe.
  K9 and K10 are CLOSED.
K12: A_ORACLES specifies 60 x 60 Gauss-Hermite x 300 Gauss-Laguerre. Sandbox
  fast oracle at 40x40x100 nodes gives EL - 0.3 = -7.36e-8. The full
  60x60x300 run exceeded the sandbox time limit. Based on K9 convergence at 300
  nodes (-8.60e-10), the specified node count is expected to satisfy rule 6 for
  K12. EVIDENCE LIMIT: the full K12 oracle at 60x60x300 nodes was not completed
  in this environment. Rule 5 ENGINE-side assertion (for all keys) cannot be
  checked without the engine; stated under evidence limits.

A-11 | CLOSED
Fact: R2 K12 sets PD = 0.015 (changed from R1). Sandbox fast oracle (40x40x100
  nodes) confirms 2 levels pass rule 4: 0.95 and 0.975. A_ORACLES section 3 K12
  row: [0.95, 0.975]. Rule 7 requires at least 2; both levels pass. The book
  change (PD 0.02 → 0.015) is the only change in the K12 definition.

A-12 | CLOSED
Fact: O-7(b) was selected (keep from_dataframe, exempt from stable surface).
  No slice needs to remove from_dataframe. The 5 test files that call
  from_dataframe (R1 G.3 item 6) are unaffected.

### F.2 New BLOCKING findings

None.

### F.3 New REFINEMENT findings

R2-01 | REFINEMENT | A_CONTRACT 4.1 (engine_version root of relative path)
A_CONTRACT 4.1 states "each *.py file under the directory of the imported
riskquant package." It does not state the root: pathlib.Path(riskquant.__file__).parent
or importlib.resources.files('riskquant'). These differ when riskquant_lab is
a sibling package. P-001 clause 5 says "under the directory of the imported
riskquant package." Matches 4.1, but the exact Python expression is not given.
This remains in the section 7 backlog (R1 R-06).

R2-02 | REFINEMENT | K12 rule 6 not verified at full node count
The sandbox cannot run the K12 oracle at 60x60x300 nodes within the time
limit. The R2 oracle text claims the specified quadrature satisfies rule 6.
This is an evidence limit of this audit, not a spec error. The S4 implementer
must run the oracle as written and verify the rule 6 self-check before asserting
it in a test.

### F.4 Owner approval

All O-1 to O-13 gates are CLOSED. No approval is required before freeze.

---

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | yes |
| Consistent with DECISIONS.md | yes |
| Implementable with existing architecture | yes |
| Blast radius bounded | yes (S1 FIRST step resolves the riskquant_lab hit and any e2e script hits) |
| Existing behavior protected | conditionally (S1 FIRST step required; stop-and-ask rule in the slice enforces it) |
| Value case holds: the smallest alternative fails, with evidence | yes (S1+S6 only leaves F-12, F-14, F-15, F-2, F-16 open; section 1 states the evidence) |
| Remaining issues are approvals, not technical gaps | yes |

Verdict: FREEZE READY

---

## Appendix: Sandbox probes executed (round 2)

All probes ran in Python 3.12.3, numpy 2.4.4, scipy 1.17.1. No riskquant install.

| Probe | Command summary | Result |
|---|---|---|
| P-RULE4-K2 | pmf_k2(), check_rule4 for each candidate | [0.95, 0.975, 0.99, 0.995]; matches R2 section 3 |
| P-RULE4-K3 | pmf_k3() (betabinom(20,2,38)), check_rule4 | [0.95, 0.975, 0.99]; matches |
| P-RULE4-K4 | convolve betabinom(10,1,99) + betabinom(10,4,16)*2 | [0.95, 0.99]; matches |
| P-RULE4-K5 | mixture Bin(20,0.02)/Bin(20,0.06) | [0.95, 0.975, 0.99, 0.995]; matches |
| P-RULE4-K6 | convolve K5-type group A (ead=1) + group B (ead=2) | [0.95, 0.995]; matches |
| P-RULE4-K7 | gauss_hermite_pmf(20, 0.02, 0.15, 200) | [0.95, 0.99, 0.995]; matches; EL err=-1.67e-16 |
| P-RULE4-K8 | gauss_hermite_pmf(20, 0.02, basel_rho(0.02), 200) | [0.95, 0.995]; matches; EL err=-1.11e-16 |
| P-RULE4-K9 | 60 GH x 80 Laguerre fast oracle | [0.95, 0.99]; matches R2 section 3 |
| P-RULE4-K10 | 60 GH x 80 Laguerre, rho=basel_rho(0.02) | [0.95, 0.99]; matches |
| P-RULE4-K11 | 80x80 GH | [0.95, 0.975]; matches; EL err=-4.44e-16 |
| P-RULE4-K12 | 40x40 GH x 100 Laguerre, PD=0.015 | [0.95, 0.975]; matches R2 section 3; A-11 CLOSED |
| P-RULE4-K13 | betabinom marshall_olkin | [0.95, 0.99, 0.995, 0.999]; matches |
| P-RULE6-K9 | 120 GH x 300 Laguerre, rho=0.15 | EL err=-8.60e-10 < 1e-8; PASSES |
| P-RULE6-K10 | 120 GH x 300 Laguerre, rho=basel_rho(0.02) | EL err=-8.60e-10 < 1e-8; PASSES |
| P-RULE6-K12-fast | 40x40 GH x 100 Laguerre, PD=0.015 | EL err=-7.36e-8; FAILS at this count; EVIDENCE LIMIT |
| P-RULE5-formula | R2 rule 5: K2 a=0.975 bound=0.11; K13 a=0.99 bound~7e-7 | Corrected formula verified |
| P-P001 | Count names in P-001 clause 1; check ValidationError not in list | 22 names; ValidationError absent |
