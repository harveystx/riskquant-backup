# WS_RISK_RUN_POSITION_CHUNKING: Freeze reconciliation

Path: docs/workstreams/WS_RISK_RUN_POSITION_CHUNKING/R_FREEZE_RECONCILIATION_2026-10-06.md
Date: 2026-10-06
Spec: WS_RISK_RUN_POSITION_CHUNKING.md revision 1
Snapshot: SNAPSHOT commit=043a7f4 dirty=false content_hash=12995a03dacaa471
Mode: read-only
Evidence base: CLAUDE.md; WS_TEMPLATE.md; DECISIONS.md (D041, D042, D055, D081, D083 to D087, D102, D103); the spec; context profiles index, core, solutions_risk_run, solutions_run_identity, reporting, scenario_metrics, sensitivity_pnl, portfolio, valuation, market_data, tests. The audit rebuilt a 555-file source tree from the `=== FILE: <path> ===` blocks of all attached profiles and searched that tree.
Evidence limits: No gate, test or benchmark ran. Every test statement comes from source. No pass claim. Three items in the prompt ATTACH list were not supplied: the `scripts` profile, `A_PROBES.md`, and the credit / solutions_credit_* profiles. 155 manifest entries are absent from the tree, including all of `scripts/`, `src/riskquant/credit/`, `src/riskquant/solutions/credit_*` and `src/riskquant_lab/`. `spot_production_scale_certifcate.json` was not read. `pyproject.toml` was not re-read. One non-gate NumPy probe ran in the audit sandbox (NumPy 2.4.4, Python 3.12.3, x86_64). It is not project-environment evidence.

## A. Verdict

NOT READY

Reason: C.9 and the Goal cannot hold as written, because K, the mapper effective universes and the shocked-level buffer scale with the book, not with the chunk size. C.4 is a placeholder.
Blockers: the C.9 restatement, a missing V_0 clause (D081 §3), and the open O-1 textual conflict with D081 §8, D042 and D087 clause 5.
Next action: the drafter issues revision 2 with section F applied. The owner then closes O-1 and O-2.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| D041 batch_size operational; bitwise portfolio P&L in a matching environment | VERIFIED | DECISIONS.md D041 "Numerical contract". `api.py:run_scenario_var_es` docstring for `batch_size`. | D041 does not name a position chunk size. P-001.1 needs a new decision. |
| D042 one canonical `compute_var_es` over `[positions; portfolio]` | OWNER APPROVAL REQUIRED | D042 "The entry point". `api.py:run_scenario_var_es` builds `groups` and calls `compute_var_es` once. | P-001.2 contradicts the literal text. The owner amends D042 or records an interpretation (O-1). |
| D055 multi-row UID, PV = sum of rows | VERIFIED | `api.py:_row_aggregation`, `_value_positions` (stable argsort, `np.add.reduceat`). | None. C.5 is consistent. |
| D081 §1 one row per economic UID | VERIFIED | `api.py:_row_aggregation` Gate A, `_grid_to_production_rows`. | None. |
| D081 §3 "The base valuation is computed once" | NOT VERIFIED | The spec does not state the V_0 source in the chunk loop. | Add a clause (F, item 2). |
| D081 §8 "one PnlGrid -> one D081 reporting transformation -> VaR / ES" | OWNER APPROVAL REQUIRED | DECISIONS.md D081 §8. | P-001.2 contradicts the literal text (O-1). |
| D083 result_digest separate from identity | VERIFIED | `schemas.py:_result_digest`, D083 clauses 1 and 2. | R2-b changes stored portfolio bits and `result_digest`. The owner decides this at O-2. |
| D084 inverted-FX SPOT Taylor refusal | VERIFIED | `api.py:_require_taylor_capable_spot` reads the full `context.portfolio`. | Precedence clause needed (F, item 3). |
| D085 run axis derived once | VERIFIED | `api.py:_reporting_route`, `_require_factor_axis`. | None. |
| D087 clause 3 batch tiles `[0, S)` | VERIFIED | `api.py:run_scenario_var_es` tiling check. | Analogue for C.6. No conflict. |
| D087 clause 5 memory reads D081 §8 as whole-run arrays | OWNER APPROVAL REQUIRED | D087 clause 5 text: "one shocked-level buffer, one PnlGrid, the reporting rows". | Textual conflict with P-001.2 (O-1). |
| D102 D102.3 on every partition before `_row_aggregation` | VERIFIED | `api.py:_require_forward_capability` reads full `ingest_batch` and `portfolio`. | Precedence clause needed (F, item 3). |
| Spec header consistency | NOT VERIFIED | Header: "DECISIONS.md reconciled: NO". Section 4: "Each checked against DECISIONS.md: yes". | Make both fields agree. |

P-001.2 conflict statement (Observed, text comparison):

- P-001.2 conflicts with the literal text of D081 §8. §8 names one PnlGrid and one reporting transformation. P-001.2 makes one grid and one conversion per chunk.
- P-001.2 conflicts with the literal text of D042. D042 names one canonical `compute_var_es` over `[positions; portfolio]`.
- P-001.2 conflicts with D087 clause 5. Clause 5 reads D081 §8 as whole-run arrays.
- Inferred: per-position rows can be equal under P-001.2. The conversion is element-wise and the VaR kernel is per-row (section C, C.3).
- Inferred: the portfolio row is equal only under R2-a with S >= 2 (F-17).
- The audit does not decide whether the mathematical equality satisfies the decisions. That is O-1.

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | `api.py:run_scenario_var_es` single path. 38 monkeypatch sites call through `api` globals (section D). | MATCH | A new keyword with default unset keeps current statements. S5.T1 reuses existing tests. |
| C.2 | `api.py:run_scenario_var_es` builds the manifest before numerics. `schemas.py:_result_digest` members. | MATCH WITH REQUIRED CHANGE | "Inputs other than var/es" is implicit. Name `identity`, manifest JSON, `position_uids`, `reproducible`. Extend S5.T2 to `position_uids` and `reproducible`. |
| C.3 | `_kernel_lower_tail_var_es` per-row; `fx_conversion.py` element-wise; pricing kernels per-row (`pricer_kernels/spot.py`, `vanilla_forward.py`). | MATCH | Inferred feasible. Per-row valuation under a row subset is not evidenced (E-3). |
| C.4 | none | NOT VERIFIED | Unresolved placeholder. WS_TEMPLATE 1.3 forbids a placeholder at FROZEN. O-2 must close first. |
| C.5 | `engine.py:compile_portfolio_from_ingest_batch` stable sort; block compilers slice and gather per row. | MATCH | A UID mask subset keeps relative row order (Inferred). |
| C.6 | `api.py:run_scenario_var_es` scenario tiling check is the existing pattern. | MATCH | None. |
| C.7 | `api.py:_require_block_wiring`, `_require_taylor_capable_spot`, `_require_forward_capability`. | MATCH WITH REQUIRED CHANGE | Refusal precedence against existing checks is not stated. All observed SPOT certification books have non-empty U_D (E-1). |
| C.8 | `run_identity/api.py:build_run_manifest` has no operational parameter. `_context_identity` hashes `context.portfolio`. `digest_reporting_plan` uses the full plan. | MATCH | F-15 confirmed. |
| C.9 | `market_data/compile.py:assemble_valuation_state`; `api.py` shocked levels; `ValuationContext.__post_init__`. | CONTRADICTION | Peak RSS keeps `16 K S` (F-7) and O(E_eff x b) valuation state. K and E_eff are sized by the whole book (E-2). "No term in M x S" is not provable. |
| C.10 | `api.py:_write_full_reval_block` writes `vs - v0_full[:, None]` into the float32 block. The scatter widens to float64. | MATCH | None. |
| C.11 | `api.py:run_scenario_var_es` `batch_size <= 0` check pattern. | MATCH | None. |

Spec fact verification (Observed unless stated):

| Fact | Result |
|---|---|
| F-1, F-2, F-4, F-5, F-7, F-8, F-9, F-10 (18 arrays counted), F-11, F-14, F-15, F-16, F-19, F-20 | MATCH. |
| F-3 | MATCH, understated. Also the `rates - base_rates[:, None]` temporary, the `gathered` and `np.where` temporaries, and `np.isfinite(settlement_pnl)` bool (M, S). `compute_var_es` adds an `np.isfinite` pass, and `vstack` copies `groups`. |
| F-6 | Inferred, supported. `stable_math_parallel_njit` omits `reassoc`. Row g reads only `group_value_matrix[g]` and writes only `scratchpad[g]`, `var_out[g]`, `es_out[g]`. Tail parameters depend on S and confidence only. |
| F-12 | Inferred, supported. Compilers gather per row with mapper lookups. `CompiledPortfolio.uix_fingerprint` is the mapper fingerprint. `unique_dates_ord32` comes from the mapper. `run_valuation` checks only the fingerprint. |
| F-13 | Observed for the dependency: both builders take mapper, universe map and levels, no portfolio. "Rebuilds per batch and per chunk" is Inferred from the proposed loop order. |
| F-17 | Corroborated in the audit sandbox, same NumPy 2.4.4. S = 1: 2 of 20 equal. S in {2,3,7,8,16,520}: 20 of 20 equal each. Block partial sums differ. A running sequential accumulator across chunks equals the reference (4 of 4). Not project-environment evidence. |
| F-18 | MATCH, incomplete. Also patched: `build_run_manifest`, `build_scenario_generator`, `apply_linear_terms_to_base_levels`, `build_market_data_matrix_from_flat_levels`, `_row_aggregation` (section D). |

## D. Blast radius of shared interfaces

Search tree: 555 files rebuilt from the attached profiles. Not searched: `scripts/`, `src/riskquant/credit/`, `src/riskquant/solutions/credit_*`, `src/riskquant_lab/`.

Commands (run in the rebuilt tree root):

- `grep -rnw --include=*.py "<symbol>" src tests | awk -F: '{print $1}' | sort | uniq -c`
- `grep -rn --include=*.py -E "setattr\(" tests src | grep -E "_value_positions|_write_full_reval_block|convert_scenario_pnl_to_reporting_currency|compute_var_es|allocate_pnl_grid|run_scenario_var_es|PnlGrid|IngestBatch|FxConversionPlan|api,"`
- Loop-driven patches were resolved by reading the stage tuples at each site.

| Interface | src callers | tests callers | Monkeypatch sites (setattr on `api`) |
|---|---|---|---|
| `run_scenario_var_es` | `solutions/risk_run/api.py` (def); docstring refs in `risk_run/schemas.py`, `sensitivity_pnl/taylor_contract.py` | `e2e/test_demo_e2e_sensi_based_var.py`; `risk_run/_fwdfix.py`, `_resultfix.py`; 17 `risk_run/test_*.py` files | none |
| `_value_positions` | `api.py:run_scenario_var_es`, `_write_full_reval_block` | patch sites only | `test_composition_negative_controls.py:52,170`; `test_preflight_fail_fast.py:62`; `test_reporting_axis.py:256`; `test_axis_input_contract.py:121`; `test_inverted_fx_taylor_admission.py:322`; `test_vanilla_forward_taylor_admission.py:390` |
| `_write_full_reval_block` | `api.py:run_scenario_var_es` | `test_vanilla_forward_node_production.py` (1 ref, no setattr) | `test_spot_production_certification.py:185`; `test_methodology_composition.py:112` |
| `convert_scenario_pnl_to_reporting_currency` | `api.py:run_scenario_var_es` | `reporting/test_fx_conversion.py`; `e2e/test_demo_e2e_sensi_based_var.py` | `test_spot_production_certification.py:186`; `test_composition_negative_controls.py:133`; `test_axis_input_contract.py:121,386,456`; `test_reporting_axis.py:246`; `test_taylor_order_semantics.py:421`; `test_vanilla_forward_node_production.py:117`; `test_vanilla_forward_taylor_admission.py:390`; `test_methodology_composition.py:112` |
| `compute_var_es` | `api.py:run_scenario_var_es`; doc refs in `scenario_metrics/api/contribution.py`; credit loss path per D042 (NOT VERIFIED) | `scenario_metrics/test_*.py`; `sensitivities/test_scenario_risk_measures.py`; `run_identity/test_*.py`; `dependence_stress/test_phase0_benchmark_identities.py`; `test_architecture.py:139` (existence only); `risk_run/_fwdoracle.py` | `test_preflight_fail_fast.py:62`; `test_reporting_axis.py:256`; `test_axis_input_contract.py:121`; `test_methodology_composition.py:112` (stores `seen["groups"]`) |
| `IngestBatch` | `portfolio/engine.py` (sole `reorder` caller, `compile_portfolio_from_ingest_batch`); block compilers spot, vanilla_forward, vanilla_european_option; `compilers/registry.py`; `ingest/compile.py`, `pipeline.py`; `portfolio/validation.py`; `market_data/builder.py`, `compile.py`; `universe.py`; `risk_run/schemas.py` | `dataclasses.replace` at `test_vanilla_forward_taylor_admission.py:182,809`, `test_multi_row_trade_identity.py:51`, 3 market_data remediation tests; field-set assertion `test_multi_row_trade_identity.py:283` | none |
| `FxConversionPlan` | `reporting/fx_conversion.py`; `api.py:_conversion_plan` | none by name | none |
| `PnlGrid` | `sensitivity_pnl/grid.py`, `engine.py`, `scheduling.py`, `kernels/taylor_validity.py`; `solutions/sensitivity_pnl/api.py`, `calculator.py`, `handoff.py`, `run.py`; `risk_run/api.py`, `schemas.py` | `sensitivity_pnl/unit/test_grid.py`; 4 risk_run tests | none |
| `allocate_pnl_grid` | `sensitivity_pnl/engine.py`, `grid.py`; `solutions/sensitivity_pnl/api.py`, `calculator.py`, `handoff.py`, `_profile/profile_phases.py`, 3 `_verify/*.py`; `risk_run/api.py` | sensitivity_pnl golden and unit tests; 6 risk_run tests | `test_preflight_fail_fast.py:62,101`; `test_reporting_axis.py:256`; `test_axis_input_contract.py:121`; `test_inverted_fx_taylor_admission.py:322`; `test_vanilla_forward_taylor_admission.py:390`; `test_parametric_run_world.py:119`; `test_methodology_composition.py:112` |

Inferred: an additive `IngestBatch` method and an additive `FxConversionPlan` slice method have a blast radius limited to the defining files. `test_multi_row_trade_identity.py:283` blocks only named new fields.

Unresolved: `scripts/` callers and monkeypatches (K7 of H), and the credit callers of `compute_var_es` and `IngestBatch`.

## E. Hidden dependency findings

E-1. SPOT production path, U_D and U_S.

- Observed: U_S is empty in every admitted run. `_require_block_wiring` refuses a non-empty U_S and any `taylor_supplied`.
- Observed: the SPOT certification test book is split. `test_spot_production_certification.py` sets `DERIVED = set(mf.slots(UIDS, mf.DERIVED_SLOTS))`, and `_mixedfix.DERIVED_SLOTS = (0, 4, 5)`.
- Observed: `_certfix.inputs` always passes `taylor_derived=TaylorBlock(...)`.
- Inferred: U_D is non-empty in the release certification, because `_require_block_wiring` refuses a block with an empty U_D.
- Unresolved: the partition of `certify_spot_production_scale.py` and `bench_spot_scaling.py` (not supplied).
- Observed: `tests/e2e/test_demo_scripts_are_runnable.py:EXCLUDED` records `bench_spot_scaling.py` as "up to ~14 GiB".
- Consequence: C.7 refuses every observed SPOT certification book. S7.T1 cannot run unchanged on a split book.
- Observed: `_mixedfix.inputs(derived=())` gives a FULL-only book. S5 has a fixture.

E-2. C.9 residual terms.

- Observed: `assemble_valuation_state` sizes every effective matrix by the mapper tables (`spot_effective_pairs`, `edf_effective_triples`, `efp_effective_quads`, `vol_effective_uix`) times the scenario batch width.
- Observed: `build_uix_mapper(ingest_batches)` builds the mapper from the full batch.
- Observed: K is the run axis of the whole book. The shocked levels hold `16 K S` bytes (F-7).
- Consequence: if K or E_eff grows with M, peak RSS has terms that scale with M times S or M times b, at any chunk size.
- Consequence: S6.T1 cannot discriminate unless the fixture holds K and E_eff fixed while M grows.

E-3. Per-row valuation invariance under a row subset.

- Observed: the pricing kernels use `parallel_njit` with `reassoc` and `contract`. The inner loop runs over the scenario axis.
- Observed: D041 and `test_release_certification.py` prove invariance over batch size and thread count. They do not prove it over row subsets.
- Inferred: low risk, because each row's arithmetic is independent.
- Consequence: S1.T2 compares compile arrays only. It does not compare valued rows.

E-4. Missing decision sections.

- Observed: `api.py` and `schemas.py` docstrings cite "D081 §11". D082, D084 and D085 cite D081 §10 and §11.
- Observed: D081 in DECISIONS.md has §1 to §9 only.
- Consequence: the methodology-partition authority that C.7 and P-001.4 extend has no locked text. This is an adjacent issue. The audit did not change it.

E-5. V_0 source.

- Observed: D081 §3 states "The base valuation is computed once".
- Observed: `api.py` computes `v0_economic` once at S = 1 over the full portfolio.
- Consequence: per-chunk recomputation would violate D081 §3. The spec is silent.

E-6. Composition precedence.

- Observed: D102.3, D103, D084, Gate A (`_row_aggregation`) and `_settlement_ccy_ids` read the full context.
- Consequence: C.7 must not run before these checks or replace them.

E-7. Environment.

- Observed: D042 `verify_environment` requires strict NumPy and Numba version equality on reproduction.
- Inferred: an R2-a guarantee "per environment" matches the existing D041/D042 contract. O-4 affects test stability, not the reproduction contract.

## F. Required changes before freeze

- Mandatory (technical):
  1. Restate the Goal and C.9 with explicit residual terms: `16 K S` (F-7, O-6), O(E_eff x b) valuation state, and the O(M) full compile in `ValuationContext`. Require the S6 fixture to fix K and E_eff while M grows.
  2. Add a clause: V_0 and `base_settlement_values` come from the one run base valuation (D081 §3), sliced per chunk, never recomputed. Name a test.
  3. Extend C.7: every existing composition and admission check runs unchanged on the full context before the chunk refusal and before chunk planning. Add a stage-order test.
  4. C.2: name the members `identity`, manifest JSON, `position_uids`, `reproducible`. Extend S5.T2.
  5. S1.T2: add bitwise comparison of `run_valuation` rows (sub-compile against full compile) at several batch sizes and thread counts (E-3).
  6. Correct F-3 to list every (M, S) temporary, or mark the count as a lower bound. Complete F-18.
  7. Relabel F-6 and F-12 as Inferred and supported by source, citing section C. Keep S1 as the empirical proof.
  8. S7: state the partition of the SPOT production script. If it is split, define a FULL-only certificate variant or move S7.T1 out of scope (E-1).
  9. Make the header field "DECISIONS.md reconciled" agree with section 4.
  10. Add to Evidence limits that `A_PROBES.md` and the scripts profile were not supplied to the audit.
- Owner approval only: O-1, including an amendment or recorded interpretation of D081 §8, D042 and D087 clause 5. O-2, which replaces the C.4 placeholder and the S = 1 rule. O-3 to O-6. D numbers for P-001 and P-002. Locked text for D081 §10 and §11 (E-4, adjacent).
- Implementation detail (not in the contract): accumulator kernel module and `riskquant.jit` decorator choice; names of the subset method and planner; `FxConversionPlan` slice placement.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no |
| Consistent with `DECISIONS.md` | no |
| Implementable with existing architecture | yes |
| Blast radius bounded | yes (src/ and tests/ only; scripts/ and credit not verified) |
| Existing behavior protected | conditionally |
| Remaining issues are approvals, not technical gaps | no |

Verdict: NOT READY
