# WS_SOLUTIONS_SPOT_INPUTS: Freeze reconciliation

Path: docs/workstreams/WS_SOLUTIONS_SPOT_INPUTS/R_FREEZE_RECONCILIATION_2026-10-06.md
Date: 2026-10-06
Spec: WS_SOLUTIONS_SPOT_INPUTS.md revision 1, with A_GAP_MAP.md
Snapshot: SNAPSHOT commit=043a7f4 dirty=false content_hash=12995a03dacaa471
Mode: read-only
Evidence base: WS_SOLUTIONS_SPOT_INPUTS.md; A_GAP_MAP.md; WS_TEMPLATE.md; CLAUDE.md; DECISIONS.md; context_riskquant_index.txt; profiles solutions_risk_run, solutions_scenario_models, solutions_sensitivity_pnl, solutions_run_identity, portfolio, config, market_data, simlab, sensitivities, valuation, utils, tests, scripts; pyproject.toml (project file).
Evidence limits: No test, gate or script was executed. Every statement comes from source. No pass claim. The source tree was reconstructed from the profiles (876 files). Credit profiles, `rq_logging.py` and `synthetic_level_generators.py` were not available. The scripts profile has no SNAPSHOT line; its identity rests on a length match (B.1). `pyproject.toml` is a project file with no snapshot binding.

## A. Verdict

NOT READY

Reason: four Observed facts are wrong (F-2, F-13, F-15, F-16), two clauses have no test that detects their defect (C.11, C.13 in S8), and one input path has no slice (H-1).
Blockers: the mandatory changes in F, then owner closure of O-1 and of every gate that a slice depends on.
Next action: the drafter issues revision 2 that applies section F; a fresh chat audits revision 2.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| B.1 Scripts profile identity with 043a7f4 | VERIFIED (length match only) | Index estimator `round(len(text)/3.7)` reproduces all 24 available profile counts exactly. Scripts profile: 1721364 characters, 465234 tokens; index row `scripts status=COMPLETE tokens~=465234`. | Replace the spec risk row "Scripts profile identity unproven" with this evidence. A content hash per profile closes the residual risk. |
| B.2 Snapshot line in spec | VERIFIED | Spec header equals index line 1. | none |
| D001 | VERIFIED | `DECISIONS.md:13`, LOCKED. Tier model supports a solutions facade over core. | none |
| D002 | NOT VERIFIED (relevance) | `DECISIONS.md:21`, LOCKED. It covers empty `__init__.py` and path comments. P-SOLUTIONS_SPOT_INPUTS-002 cites it as related to an import rule. D002 is not in the spec list "Decisions relied on". | Remove D002 from P-002, or state the relation and add it to section 4. |
| D006 | VERIFIED | `DECISIONS.md:55`, LOCKED. Config ingest is Polars. Supports O-7. | none |
| D019 | VERIFIED | `DECISIONS.md:232`, LOCKED. Golden numbers change only with their checks and `tests/e2e/test_demo_e2e_sensi_based_var.py`. C.14 and S9 agree. | none |
| D022 | CONTRADICTION WITH SPEC CLAIM | `DECISIONS.md:279`, Status OPEN (RQ-AUD-008). Spec section 4 lists it as relied on and "checked: yes". An OPEN decision gives no rule to rely on. | State D022 as OPEN context for O-7, not as a decision relied on. |
| D023 | VERIFIED | `DECISIONS.md:286`, LOCKED. Common gates are scoped to changed files. | none |
| D042 | NOT VERIFIED (status) | `DECISIONS.md:1524`. No `**Status:**` field in the decision block. The text names `run_scenario_var_es` as the public production entry. | Owner confirms D042 status, or the spec cites the field that carries it. |
| D067 | VERIFIED, with required change | `DECISIONS.md:3110`, LOCKED. Clause: `build_market_data_matrix_from_snapshot` requires `snapshot.as_of_date == as_of_date`. `extract_levels_from_snapshot` (`market_data/assemble.py:566`) has no as-of check. See C.6 and E-3. | Apply F.M-6. |
| D076 | VERIFIED | `DECISIONS.md:3424`, LOCKED. `build_risk_factor_matrix_from_df` normalises once. S4 prohibition agrees. | none |
| D081, D082, D085, D086, D088 | VERIFIED | LOCKED at `DECISIONS.md:4178`, `4509`, `4619`, `4654`, `4704`. D086 is cited in S8 but missing from section 4. | Add D086 to section 4. |
| D093 | VERIFIED | `DECISIONS.md:4868`, LOCKED. Clause 3 (row order, duplicates kept) and clause 7 (API stays typed, frame is an adapter) agree with C.9 and O-9. | none |
| D094 | VERIFIED | `DECISIONS.md:4888`, LOCKED. Message at `portfolio/ingest/pipeline.py:85` contains `(D094)`. C.5 agrees. | none |
| Status DRAFT, independent review | VERIFIED | Header `Status: DRAFT`, `Independent review: NONE`. This audit runs in a fresh chat. | Owner sets REVIEW. |
| Placeholders | OWNER APPROVAL REQUIRED | S1 to S5 paths contain `<O-1>`. S9 and S10 files are "per O-10", "per O-11". WS_TEMPLATE 1.3 forbids placeholders at FROZEN. | Close O-1, O-10, O-11, or freeze only the slices without placeholders. |

## C. Contract-to-repository matrix

Facts:

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| F-1 | `solutions/risk_run/schemas.py:ValuationContext` (lines 205 to 242) | MATCH | none |
| F-2 | `solutions/sensitivity_pnl/schemas.py:63` imports `BaseModel, ConfigDict`; `solutions/scenario_models/api.py:19` imports `TypeAdapter` | CONTRADICTION | Pydantic is not limited to two modules. O-14 premise is incomplete. |
| F-3 | `solutions/risk_run/api.py:run_scenario_var_es` (line 880) | MATCH | The run also takes `required_factor_ids` (int32) and `required_shift_type_id` (int8). Both come from `RunFactorAxis`. The six-input list omits them. |
| F-4 | `portfolio/ingest/pipeline.py:compile_raw_df_into_ingest_batch` | MATCH | none |
| F-5 | `portfolio/ingest/schema.py:ALL_ALLOWED_COLS_SET`; `canonicalize.py:95` | MATCH | Also: an empty frame raises `ValueError` (`canonicalize.py:77`). |
| F-6 | `portfolio/engine.py:build_uix_mapper`; `utils/dates.py:date_to_ord32` | MATCH | A_GAP_MAP "Dates" row names `ord_to_date`. It lives in `utils/business_days.py:131`, not `utils/dates.py`. |
| F-7 | `config/static_config/compile.py:compile_risk_static_config_from_dict_of_dfs`; `loader.py:load_risk_static_config_xlsx_into_dict` | MATCH | none |
| F-8 | `market_data/risk_factors/builder_snapshot.py:28`; `market_data/assemble.py:566` | MATCH | none |
| F-9 | `market_data/risk_factors/builder_matrix.py:21` | MATCH | Unknown column names raise `KeyError`. |
| F-10 | `config/proxy_model/ingest.py:59` | MATCH | none |
| F-11 | `solutions/scenario_models/api.py:59`; `solutions/risk_run/schemas.py:129`, `:260`; `solutions/risk_run/api.py:620` | MATCH | none |
| F-12 | `scripts/demos/e2e/07_spot_production_example.py` docstring | MATCH | none |
| F-13 | AST scan, command in D.1 | CONTRADICTION | 33 non-solutions `riskquant` modules, not 25. One more top-level package: `riskquant_lab.polars_display`. A_GAP_MAP is complete by symbol except the Display row, which omits `riskquant_lab`. |
| F-14 | `tests/test_architecture.py:23`; `pyproject.toml [tool.ruff.lint]` | MATCH WITH REQUIRED CHANGE | `tests/test_proxy_v1_exclusivity.py:19` already AST-scans `("src", "scripts")`. The fact omits this precedent for O-11. |
| F-15 | D.2 | CONTRADICTION | User list incomplete. `test_composition_negative_controls.py` and `test_spot_production_certification.py` also import `_mixedfix`. Core-symbol reads through the `_fixtures` module occur in exactly six files (D.2). |
| F-16 | `_forward_world.py` lines 8 and 364 | CONTRADICTION | `_forward_world.py` does not import `_spot_world`. It names it in docstrings only. `10` and `11` import it and use only presentation helpers (D.3). |
| F-17 | `00` docstring lines 19 to 20; `_spot_world.py:975`, `:1021`, `:1047`; `demo_e2e_sensi_based_var.py:166`, `:304` | MATCH | none |
| F-18 | `solutions/risk_run/api.py:1143` (`TaylorBaseBinding` built in the run) | MATCH | none |
| F-19 | `05_spot_reproducibility_and_identity.py:162` | MATCH | none |
| F-20 | `solutions/risk_run/api.py:1024`; `config/proxy_model/compile.py:127`; run docstring lines 910 to 915 | MATCH (source only, not executed) | The run passes the terms to `compile_proxy_model`, which iterates them with no non-empty guard. Every V1 axis node without a term gets its identity row. `()` is admissible for any methodology. The cited site `tests/solutions/risk_run/_fixture.py:173` is all TAYLOR/DERIVED (`full_reval_uids=[]`), so it does not show the SPOT FULL_REVAL case. Label stays Inferred. |
| F-21 | D.2 | MATCH (source only, not executed) | Six files read core names as attributes of the `_fixtures` module. Removing the import removes the attribute; each read raises `AttributeError`. |

Clauses:

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 | `solutions/sensitivity_pnl/taylor_contract.py:131:static_config_content_digest` | MATCH | S1.T1 is testable. |
| C.2 | `config/static_config/compile.py:347` (`validate_static_config_dfs`) | MATCH WITH REQUIRED CHANGE | Name the exception type that S1.T2 expects. |
| C.3 | `solutions/run_identity/hashing.py:232:hash_compiled_portfolio` | MATCH WITH REQUIRED CHANGE | `_spot_world.TINY_POSITIONS` is `pd.DataFrame` (`_spot_world.py:1196`). S2.T1 must state `pl.from_pandas`. |
| C.4 | `portfolio/ingest/compile.py:556`; `portfolio/engine.py:489` | MATCH | Chapter 07 passes no `as_of_date` to ingest. C.4 is stricter than the current path and agrees with D067. |
| C.5 | `pipeline.py:85` | MATCH | none |
| C.6 | `assemble.py:566` | MATCH WITH REQUIRED CHANGE | The builder takes an as-of and a `RunFactorAxis` but no `ValuationContext`. Nothing ties the snapshot as-of to `uix_mapper.as_of_ord32`. See E-3. |
| C.7 | `assemble.py:578` (`searchsorted_or_raise`) | MATCH WITH REQUIRED CHANGE | "existing snapshot error" is vague. The raise comes from `searchsorted_or_raise`. Name the exception type. |
| C.8 | F-9 | NOT VERIFIED | Label Unresolved; depends on O-7. Cannot freeze. |
| C.9 | F-10 | MATCH | none |
| C.10 | `RiskRun` fields `identity`, `var_hex`, `es_hex` (`schemas.py:391`) | MATCH | S6.T1 is testable. |
| C.11 | S6.T2 | CONTRADICTION | S6.T2 scans the imports of the new test file. It does not detect a change to `run_scenario_var_es` or to a core module. |
| C.12 | D.2 | MATCH WITH REQUIRED CHANGE | S7.T1 needs the fixed list of the 9 core names in D.2. Pattern `FX\.` also matches `mf.FX.`; `_FIXTURES\.` covers `tests/e2e/test_demo_e2e_sensi_based_var.py:77`. |
| C.13 | S8, S10 | CONTRADICTION | Label Unresolved. S8 claims "C.13 (Taylor part)" but S8.T1 tests run identity, not imports. H-1 imports in chapter 07 have no slice. |
| C.14 | `DECISIONS.md:232` | MATCH | none |

## D. Blast radius of shared interfaces

D.1 Demo import surface (F-13). Command: Python `ast.walk` over `scripts/demos/e2e/{_fixtures,_spot_world,00..07,bench_spot_scaling,demo_e2e_sensi_based_var}.py`; count `ImportFrom` modules with first segment `riskquant` and not `riskquant.solutions`. Result: 33 modules. Solutions modules used outside `api.py`: `risk_run.schemas`, `scenario_models.schemas`, `scenario_models.world_columns`, `sensitivity_pnl.taylor_contract`.

D.2 `scripts/demos/e2e/_fixtures.py`. Commands: `grep -rln "_fixtures.py\|import _fixtures\|_fixtures\"" tests scripts` (three `ingest_fixtures` false positives removed); `grep -rln "import _mixedfix\|from _mixedfix\|_mixedfix\b" tests`; regex `\b(?:mf\.FX|FX|_FIXTURES|fx)\.(\w+)` over the files found, classified against the `_fixtures.py` import table. Files searched: `tests/**`, `scripts/**`.

| Caller | Mechanism | Reads core names | Reads private fixture names |
|---|---|---|---|
| `scripts/demos/e2e/demo_e2e_sensi_based_var.py` | `import _fixtures as fx` | via own imports | yes |
| `tests/solutions/risk_run/_mixedfix.py:57` | `spec_from_file_location` as `FX` | yes | `_run_risk_static_config`, `_RISK_STATIC_CONFIG_DICT_OF_DFS`, `_NODE_META_DF`, `_run_proxy_terms`, `_run_risk_factor_snapshot`, `_build_historical_matrix`, `_run_axis_inputs`, `_run_market_data`, `_run_pricing`, `_SNAPSHOT_AS_OF_DATE` |
| `tests/solutions/risk_run/_fixture.py` | `mf.FX` | `DiscountingMode`, `run_sensitivities` | none |
| `tests/solutions/risk_run/_certfix.py` | `FX = mf.FX` | yes | `_RISK_STATIC_CONFIG_DICT_OF_DFS`, `_NODE_META_DF`, `_SNAPSHOT_AS_OF_DATE` |
| `tests/solutions/risk_run/test_run_owned_identity.py` | `mf.FX` | yes | `_RISK_STATIC_CONFIG_DICT_OF_DFS`, `_NODE_META_DF` |
| `tests/solutions/risk_run/test_methodology_composition.py` | `mf.FX` | `DiscountingMode` | none |
| `tests/solutions/risk_run/test_spot_production_certification.py:44` | `FX = mf.FX` | none | `_HISTORY_LEVELS`, `_RISK_FACTOR_LEVELS`, `_SCENARIO_SOURCE_NAMES` |
| `tests/solutions/risk_run/test_composition_negative_controls.py:20` | `import _mixedfix` | none (transitive) | none (transitive) |
| `tests/e2e/test_demo_e2e_sensi_based_var.py:77` | `_FIXTURES = _load_module(...)` | `build_risk_factor_level_snapshot_from_dict`, `date_to_ord32` | `_POSITIONS_DF`, `_RISK_FACTOR_LEVELS`, `_SNAPSHOT_AS_OF_DATE`, `_run_market_data`, `_run_portfolio`, `_run_pricing`, `_run_risk_static_config`, `_run_sensitivities`, `build_fixture` |

Core names read through the module (9): `DiscountingMode`, `build_risk_factor_level_snapshot_from_dict`, `build_uix_mapper`, `build_universe_mapping`, `compile_portfolio_from_ingest_batch`, `compile_raw_df_into_ingest_batch`, `compile_risk_static_config_from_dict_of_dfs`, `date_to_ord32`, `run_sensitivities`. Solutions names read through the module: `SensitivityPnlCalculator`, `TaylorBaseBinding`, `TaylorOrder`, `base_levels_digest`, `base_settlement_values_digest`, `build_sensitivity_book_spec`, `derive_taylor_declaration`, `static_config_content_digest`, `valuation_contract_fingerprint`.


D.3 `scripts/demos/e2e/_spot_world.py`. Commands: `grep -rln "_spot_world" tests scripts src`; regex `\bsw\.(\w+)` per importer. No file in `tests/` or `src/` imports it.

| Caller | Domain names used (presentation helpers excluded) |
|---|---|
| `00_spot_getting_started.py` | `TINY_*` constants, `author_proxy_terms`, `compile_static_config`, `history_matrix` |
| `01_spot_taylor_derived.py` | `anatomy_matches_run`, `certified_taylor_block`, `methodology`, `realistic_fixture`, `reporting_spec`, `restrict_sensitivities`, `run_base_binding`, `scenario_pnl_anatomy`, `tiny_fixture`, `valuation_context` |
| `02_spot_mixed_book.py` | `SpotWorld`, `anatomy_matches_run`, `certified_taylor_block`, `methodology`, `realistic_fixture`, `reporting_spec`, `scenario_pnl_anatomy`, `tiny_fixture`, `valuation_context` |
| `03_spot_fx_and_reporting_currency.py` | `ScenarioAnatomy`, `SpotWorld`, `TINY_LEVELS`, `TINY_POSITIONS`, `anatomy_matches_run`, `certified_taylor_block`, `methodology`, `realistic_fixture`, `reporting_spec`, `scenario_pnl_anatomy`, `tiny_fixture`, `valuation_context` |
| `04_spot_scenario_models.py` | `anatomy_matches_run`, `methodology`, `realistic_fixture`, `reporting_spec`, `scenario_pnl_anatomy`, `valuation_context` |
| `05_spot_reproducibility_and_identity.py` | `RealisticFixture`, `build_world`, `methodology`, `realistic_fixture`, `reporting_spec`, `valuation_context` |
| `06_spot_batching_and_scale.py` | `certified_taylor_block`, `methodology`, `realistic_fixture`, `reporting_spec`, `valuation_context` |
| `07_spot_production_example.py` | `realistic_fixture` |
| `bench_spot_scaling.py` | `RealisticFixture`, `methodology`, `realistic_fixture`, `realistic_positions`, `reporting_spec`, `valuation_context` |
| `10_forward_node_full_reval.py`, `11_forward_node_taylor_derived.py` | presentation helpers only (`Checks`, `formula`, `kv`, `num`, `para`, `step`, `table`, `title`, `write_evidence`) |

Indirect caller: `tests/e2e/test_demo_scripts_are_runnable.py:41` runs every `scripts/demos/e2e` entry without a leading `_` as a subprocess. Every chapter above is in its scope.

## E. Hidden dependency findings

- E-1 (H-1). Chapter 07 step 3 imports `compile_proxy_model`, `lower_axis_operator`, `world_request` (`07_spot_production_example.py:67`, `:69`, `:181` to `:202`). It checks that the history covers every proxy driver column. S5 wraps `proxy_terms_from_df` only. S9 excludes 07. Consequence: after S6 and S8, 07 still violates C.13 and S10.T1 fails. A consumer also has no solutions path to learn which history columns the run needs (O-6).
- E-2. If O-12 selects "`api.py` only", every demo already violates the rule (D.1). O-12 must be decided against this fact.
- E-3. C.6 plus P-SOLUTIONS_SPOT_INPUTS-001 clause 1. The base level builder composes `build_risk_factor_level_snapshot_from_dict` and `extract_levels_from_snapshot`. The second has no as-of check. A builder with an independent as-of argument permits a snapshot as-of that differs from the mapper as-of. D067 forbids a split as-of. Clause 1 forbids added validation. The two rules conflict unless the builder takes the as-of from the `ValuationContext`.
- E-4. `bench_spot_scaling.py:483` to `:516` replaces `risk_run.api._value_positions`, `convert_scenario_pnl_to_reporting_currency` and `compute_var_es` at run time. A missing name is skipped (`getattr(..., None)`). A rename inside `risk_run/api.py` silently removes a benchmark stage.
- E-5. Tests read 17 fixture-owned `_fixtures.py` names: 16 private names and `build_fixture` (D.2). S8 modifies `_fixtures.py` Taylor steps. S7 decouples core names only. A change of a private name or its return type in S8 or S9 breaks tests that S7 does not touch.
- E-6. An empty positions frame raises `ValueError` at ingest (`canonicalize.py:77`). A consumer day with no SPOT positions cannot build a `ValuationContext`. The spec does not state this behaviour.
- E-7. The slice order sentence is self-contradictory: both branches put S7 before S6. The dependency of S7 on O-16 is the only real constraint.

## F. Required changes before freeze

Mandatory (technical):
- M-1 (P0). Correct F-13 to 33 modules plus `riskquant_lab`. Add `riskquant_lab.polars_display` to the A_GAP_MAP Display row.
- M-2 (P0). Correct F-15: add `test_composition_negative_controls.py` and `test_spot_production_certification.py` as transitive users. Fix the S7.T1 name list to the 9 core names in D.2.
- M-3 (P0). Correct F-16: remove `_forward_world.py`; state that `10` and `11` use presentation helpers only. Update the S10 exit and the risk row.
- M-4 (P0). Give C.11 a test that detects its defect, for example a gate that `git diff --name-only <base>` lists no path under `src/riskquant/` outside the O-1 package.
- M-5 (P0). Cover E-1: add 07 step 3 to a slice, or list it as an O-10 or O-13 exemption candidate in C.13.
- M-6 (P1). Resolve E-3: the base level builder takes the `ValuationContext` and reads the as-of from `uix_mapper.as_of_ord32`, or C.6 states the D067 check as a named exception to P-001 clause 1.
- M-7 (P1). Correct F-2. Re-state the O-14 premise.
- M-8 (P1). Move D022 from "relied on" to OPEN context. Add D086. Remove or justify D002 in P-002.
- M-9 (P1). Re-map S8: either S8 implements no part of C.13, or add S8.T2 that scans the imports of the files S8 modifies.
- M-10 (P2). Name the exception types in C.2 and C.7. State `pl.from_pandas` in S2.T1. Fix the `ord_to_date` module in A_GAP_MAP. Add E-6 to section 7. Rewrite the slice order sentence (E-7). Add E-2 to O-12 and E-5 to the S8 and S9 prohibitions.
- M-11 (P2). Update the header `Evidence limits` and the risk table with B.1. Re-label F-20 with the source evidence in C.

Owner approval only:
- O-1 (blocks S1 to S5 file paths), O-16 (blocks S7), O-7 (blocks C.8), O-10 and O-11 (block S9, S10 file lists), O-12, O-13, O-17 (define C.13). The other gates block their named slices only.
- Confirm the D042 status field.
- Decide whether to freeze S1 to S7 first and keep S8 to S10 in a later revision.

Implementation detail (not in the contract):
- Test placement under `tests/solutions/<O-1>/`.
- Reuse of `tests/test_proxy_v1_exclusivity.py` style for the O-11 AST rule.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | no (E-3, E-7, C.13 in S8) |
| Consistent with `DECISIONS.md` | no (D022 OPEN relied on; D067 gap in C.6) |
| Implementable with existing architecture | yes |
| Blast radius bounded | no (F-15 and F-16 wrong; E-1 and E-5 not covered) |
| Existing behavior protected | conditionally (C.11 has no detecting test) |
| Remaining issues are approvals, not technical gaps | no |

Verdict: NOT READY
