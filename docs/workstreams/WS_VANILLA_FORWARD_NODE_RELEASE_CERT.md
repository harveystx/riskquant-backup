# WS_VANILLA_FORWARD_NODE_RELEASE_CERT: release-scale certification world and release script for VANILLA_FORWARD_NODE

Path: docs/workstreams/WS_VANILLA_FORWARD_NODE_RELEASE_CERT.md
Status: DRAFT
Revision: 1
Revision note: First draft from an investigator PLAN CHANGE task.
Date: 2026-10-06
Branch: ws/vanilla-forward-node-release-cert
Snapshot: SNAPSHOT commit=043a7f4 dirty=false content_hash=12995a03dacaa471
Drafted by: investigator
Independent review: NONE
DECISIONS.md reconciled: NO
Evidence limits: No test, script or gate in the repository ran. All facts come from
static reading of the context profiles. The sha256 values in F-1 come from text
extracted from `context_riskquant_scripts.txt`, not from repository bytes. Line
numbers of `_forward_world.py` are not verified; this spec cites symbols only.
Ingest admission of HENRY_HUB_GAS_FWD and of the tokens [M+2], [M+3], [M+6],
[M+12] is not evidenced. Forward run-level thread invariance is not measured.
Peak RSS is an estimate (F-14), not a measurement. Validation of 77 synthetic
LABEL curves by the static-config compiler is not executed. D041, D043, D065,
D081, D082, D084 and D085 were not read in full in the drafting chat.

## 1. Goal and scope

Goal: Certify VANILLA_FORWARD_NODE on the production run path at the SPOT release
dimensions, with a world in `tests/solutions/risk_run/` and a release script in
`scripts/release/` that writes a certificate with the SPOT certificate sections.

In scope:
- A certification world module in `tests/solutions/risk_run/` (C.1 to C.9, C.11).
- Structural tests of the world and static tests of the script.
- A single-file release script and its certificate (C.10, C.12 to C.22).
- One release execution on the owner host and its report (S6).

Out of scope:
- Any change in `src/`, `DECISIONS.md`, SPOT, Proxy V1, the scenario engine and
  the core valuation kernel.
- CURVE numerics. TAYLOR/DERIVED in DiscountingMode.CURVE (D103.6, D104).
- TAYLOR/SUPPLIED (D103.5). VANILLA_FORWARD_CURVE.
- Changes to `scripts/demos/e2e/10_forward_node_full_reval.py`,
  `11_forward_node_taylor_derived.py`, `_forward_world.py`,
  `bench_forward_scaling.py`, `tests/solutions/risk_run/_fwdfix.py`,
  `_fwdoracle.py`, `_certfix.py`, and the SPOT release script and certificate.
- Currencies other than USD and EUR. Repeated UIDs. SPOT rows in the book.

Owner decisions in force:
- 2026-10-06: The world lives in `tests/solutions/risk_run/`, as `_certfix.py` does (SPOT precedent).
- 2026-10-06: The private hooks `api._value_positions` and `api._write_full_reval_block` are allowed, only inside the Capture copy in the release script.
- 2026-10-06: Dimensions are the SPOT certificate dimensions: M=200, K_axis near 140, K_nodes 250, T=600, S=2000, B=256, confidence 0.99, stress (20000, 2048) and (100000, 4096), the 5 models of CFGS, USD and EUR reporting, DiscountingMode.NONE.
- 2026-10-06: CURVE is a negative control only. Expected result: D103.4 CompositionContractError when U_D holds a forward UID. No CURVE numerics.

Depends on: D101, D102, D103 (LOCKED).
Blocks: none.

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | Extracted text hashes: `_forward_world.py` 0bd15934..., `certify_spot_production_scale.py` eddd4ce6.... Neither equals the earlier cited f1281fe6.... | `sha256sum` on profile extraction | Unresolved |
| F-2 | `LEVELS` and `VOLS` hold 7 names. `build_world` and `history` raise KeyError for any other axis row. `static_config` takes no parameter. | `scripts/demos/e2e/_forward_world.py:LEVELS`, `build_world`, `history`, `static_config` | Observed |
| F-3 | `value_by_uid`, `production_shocked_levels`, `scenario_pnl_full_and_taylor` call production. `fx_leg` and `DISCOUNT_CURVE` resolve USD and EUR only. `_family` maps HENRY_HUB to a key absent from `UNDERLYING_CCY`. | `_forward_world.py:value_by_uid`, `fx_leg`, `_family`, `UNDERLYING_CCY` | Observed |
| F-4 | The SPOT script reads cf.K_TOTAL, inputs, FIDS, POSITIONS, UIDS, NAMES, VALS, FX, _BATCH, AS_OF, RSC, NONE, _static_config, MAPPER, K, partition, M, T, S, CFGS. It binds `FX_ID = cf.K_TOTAL`. | `certify_spot_production_scale.py` (grep `cf\.`) | Observed |
| F-5 | The SPOT oracle indexes `VALS[:, cf.FIDS - 1]`, uses `cf.NAMES.index(name) + 1`, and assumes LOG_RETURN for every level. | `certify_spot_production_scale.py:oracle_historical_levels`, `book` | Observed |
| F-6 | The SPOT script uses fixed `ATOL_PNL`, `ATOL_V0`, `ATOL_LEVEL`. | `certify_spot_production_scale.py` module constants | Observed |
| F-7 | `_fwdoracle` imports only math, decimal, numpy. It defines `forward_value`, `effective_rate`, `tol_v0`, `tol_dv_full`, `reporting_dw`, `tol_dw`, `tail_geometry`, `var_es`, `tol_es`, `portfolio_eps`, `level_oracle`, `tol_level`, `forward_taylor_pnl`, `tol_dv_derived`. | `tests/solutions/risk_run/_fwdoracle.py`; `test_vanilla_forward_node_production.py:test_oracle_imports_no_production_code` | Observed |
| F-8 | `fx_effective_pairs[:, 0]` indexes `market_id_universe`. | `src/riskquant/universe.py:UixMapper` | Observed |
| F-9 | Canonical ids: nodes 0 to 9, USDEUR is 0. Curves 250_000_000 + k*1_000, k = 0..12, rf blocks 1_000 + k*1_000. BRENT, WTI, HENRY_HUB hold [M+0], [M+1], [M+2], [M+3], [M+6], [M+12] at rf 9_000, 10_000, 11_000 + j. Surface rf ids start at 100_000. | `ingest_fixtures/config/risk_static_config/_parts/node_meta.py:CANONICAL_NODE_NAMES`, `curve_point_meta.py:_CURVE_BLOCKS`, `surface_point_meta.py:_SURFACE_BLOCKS` | Observed |
| F-10 | Test-owned ids in use elsewhere: 250_013_000 with rf 14_003, 14_009; 250_014_000 with rf 15_011. | `_forward_world.py:DEMO_CURVE`; `_fwdfix.py:_EXACT_CURVES` | Observed |
| F-11 | rf-id uniqueness across nodes, curve points and surface points, and namespace bounds, are enforced at compile. | `config/static_config/validate.py:_validate_rf_id_uniqueness`, `_validate_global_id_namespace` | Observed |
| F-12 | BRENT, WTI, HENRY_HUB use OFFSET_M + CALENDAR_PERIOD, anchor 1. CO2 uses OFFSET_Y + ROLL_ON_MMDD. POWER uses OFFSET_HY + HALF_YEAR_BY_LABEL. | `_parts/curve_meta.py` roll columns | Observed |
| F-13 | `_require_forward_capability` order: D102.3, D103.3, D103.4(b), D103.4(a), D102.4. It runs after `_require_taylor_capable_spot` and before `build_universe_mapping`. Precedent match: `^D103\.4: .*DiscountingMode\.NONE only`. | `solutions/risk_run/api.py:_require_forward_capability`, `run_scenario_var_es`; `test_vanilla_forward_taylor_admission.py:test_d103_4b_curve_mode_is_refused` | Observed |
| F-14 | Run buffers at S=100_000, K=140, M=200: chunks plus hstack 213.6 MiB, grid 76.3, settlement 152.6, converted 152.6, groups 153.4; total near 748 MiB, plus 106.8 MiB for the Capture shocked copy. | `api.py:run_scenario_var_es` allocation sites | Inferred |
| F-15 | `_write_full_reval_block` receives `shocked_levels[:k_val]` and runs only when `u_full.size > 0`. | `api.py:run_scenario_var_es` | Observed |
| F-16 | SPOT stress HWM is process-lifetime: before 1382.4 MiB, after 1488.8 MiB at S=100_000. `bench_forward_scaling.py` uses one subprocess per cell. | `spot_production_scale_certifcate.json:stress`; `bench_forward_scaling.py:run_cell_subprocess` | Observed |
| F-17 | No forward test sets `NUMBA_NUM_THREADS`. SPOT run-level thread evidence uses child processes and a `numba.get_thread_id` probe. The SPOT certificate verdict `multi_core_determinism` is "see release gate". | `grep -rnE "NUMBA_NUM_THREADS\|set_num_threads" tests scripts`; `test_release_certification.py` | Observed |
| F-18 | The release runner executes every `*.py` under `scripts/release/` recursively, except `_scratch*`, `audit_*`, `run_all_*`. | `scripts/run_all_scripts_release.py:main` | Observed |
| F-19 | `_fwdfix` imports pandas. `_certfix` loads `scripts/demos/e2e/_fixtures.py` through `_mixedfix.FX`. `_certfix.AS_OF` is an int ord32; `_forward_world.AS_OF` is a dt.date. | `_fwdfix.py` imports; `_mixedfix.py:FX`; `_certfix.py:AS_OF` | Observed |
| F-20 | ParametricRiskData accepts LOG_RETURN, SIMPLE_RETURN and DIFFERENCE driver charts. | `simlab/scenarios/parametric_data.py:PARAMETRIC_DRIVER_CHART_IDS` | Observed |
| F-21 | Top-level SPOT certificate keys: determinism, dimensions, identity, models, performance, reporting, stress, verdict. | `spot_production_scale_certifcate.json` | Observed |

## 3. Contract

World module: `tests/solutions/risk_run/_fwdcertfix.py` (Proposed name, O-4).
Release script: `scripts/release/forward_node/certify_forward_node_production_scale.py` (Proposed, O-4).

- C.1 The world exposes exactly these names, with these types. [Proposed] Test: S1.T1

| Name | Type and contract |
|---|---|
| `K_TOTAL`, `M`, `T`, `S` | int; 250, 200, 600, 2000; no environment override |
| `HIST_IDS` | int32 (K_TOTAL,), ascending, unique, C-contiguous; history column ids |
| `NAMES` | dict[int, str]; rf id to factor name for every `HIST_IDS` entry; built from `STATIC_DFS` |
| `STATIC_DFS` | dict[str, pl.DataFrame]; canonical sheets plus synthetic rows of C.5 |
| `VALS` | float64 (T, K_TOTAL), F-order; column j holds `HIST_IDS[j]` |
| `RSC` | RiskStaticConfig compiled from `STATIC_DFS` |
| `_static_config(first_exact_shift)` | changes only the shift type of synthetic curve k = 0 |
| `POSITIONS` | pl.DataFrame raw rows of C.6; uid order not ascending |
| `UIDS`, `INVERTED_UIDS` | int64 ascending; `INVERTED_UIDS` is a subset of `UIDS` |
| `AS_OF`, `AS_OF_DATE` | int ord32; dt.date; equal dates |
| `NONE` | `DiscountingMode.NONE` |
| `_BATCH`, `MAPPER`, `PORTFOLIO` | ingest batch (as_of_date=AS_OF_DATE), mapper at AS_OF, compiled portfolio |
| `FIDS`, `K`, `BASE` | run axis (int32), its length, float64 base levels on the axis from `VALS[-1]` |
| `FX_ID` | int; C.4 |
| `partition(mixed)` | (U_F, U_D) int64; C.8 |
| `calculator(base, variant, uids)`, `v0(base)` | producer-side Taylor book and base values; never read by the oracle |
| `matrix()` | RiskFactorMatrix over `HIST_IDS`, `VALS`, T dates ending on AS_OF |
| `build_parametric_data`, `freeze_parametric_data`, `load_parametric_data`, `parametric_data`, `PARAMETRIC_DATA_PATH_ENV` | as `_certfix`, over `HIST_IDS`, chart per column from `STATIC_DFS` |
| `CFGS` | the 5 SPOT model configurations, same keys and values |
| `inputs(model, batch_size, seed, calc, mixed, total_scenarios)` | same signature and validation as `_certfix.inputs` |

- C.2 The world imports no module under `scripts/`, no `_forward_world`, `_fwdfix`, `_mixedfix`, `_certfix`, and no pandas. [Proposed] Test: S1.T2
- C.3 `K_TOTAL == 250`, `M == 200`, `T == 600`, `S == 2000`, `K == 140` (O-1). [Proposed] Test: S1.T3
- C.4 `FX_ID` is the single element of `unique(MAPPER.market_id_universe[MAPPER.fx_effective_pairs[:, 0]])`. It equals the raw `node_meta.node_id` of "USDEUR" in `STATIC_DFS`; a mismatch raises at import. [Proposed] Test: S1.T4
- C.5 Id allocation follows this table. The node ids, curve-point rf ids and surface-point rf ids of `STATIC_DFS` are pairwise distinct. The curve ids are distinct. The synthetic ranges intersect no canonical id and no id in F-10. [Proposed] Test: S1.T5

| Item | Ids | Count | Use |
|---|---|---|---|
| Canonical node USDEUR | 0 | 1 | FX leg and reporting factor |
| Other canonical nodes | 1 to 9 | 0 used | not in history |
| BRENT_CRUDE_FWD points | 9_000 to 9_005 | 6 | relative forwards |
| WTI_CRUDE_FWD points | 10_000 to 10_005 | 6 | relative forwards |
| HENRY_HUB_GAS_FWD points | 11_000 to 11_005 | 6 | relative forwards |
| Other canonical curve and surface points | 1_000 to 8_008, 12_000 to 13_003, from 100_000 | 0 used | not in history |
| Synthetic curves `CERT_FWD_<kkk>` | curve id 250_100_000 + k*1_000, k = 0..76 | 77 | LABEL, PRICE, quote USD, chart LOG, SIMPLE, DIFFERENCE for k mod 3 = 0, 1, 2 |
| Synthetic points | rf 20_000 + k*1_000 + j, j = 0..2, tokens [1M], [3M], [6M] | 231 | exact forwards |
| Synthetic nodes | none | 0 | |
| History universe | 1 + 18 + 231 | 250 | `HIST_IDS` |

- C.6 The book has M rows, one row per UID, all VANILLA_FORWARD_NODE. UIDs are sparse in [1, 50_000). 18 rows are relative, one per token of C.11, with `delivery_date` in the resolved contract period (D065, D101.4). 182 rows are exact on 121 synthetic points; each point has at least one row; at least 40 exact rows omit `delivery_date` (D101.3). Settlement alternates USD and EUR. At least 10 rows have strike EUR and settlement USD (`INVERTED_UIDS`). [Proposed] Test: S1.T6
- C.7 Settlement, strike and underlying quote currencies are in {USD, EUR}. The script oracle raises ValueError for any other route. Reporting currencies are USD and EUR. [Proposed] Test: S1.T7, S2.T5
- C.8 `partition(True)`: U_F = sorted(UIDS[::2] union INVERTED_UIDS); U_D = UIDS minus U_F; both non-empty; U_D holds no element of `INVERTED_UIDS` (D103.3). `partition(False)`: U_F = UIDS, U_D empty. [Proposed] Test: S1.T8
- C.9 For reporting USD and EUR, `derive_run_factor_axis` gives K_rep = 0, and its factor ids and shift type ids equal `FIDS` and the `inputs` shift types row for row. [Inferred] Test: S1.T9
- C.10 Tolerance policy: every reconciliation is elementwise `|production - oracle| <= tol` with tol from `_fwdoracle`: `tol_v0`; `tol_dv_full` for U_F rows; `tol_dv_derived` with `forward_taylor_pnl` for U_D rows; `tol_dw`; `portfolio_eps`; VaR eps = max of `portfolio_eps`; ES `tol_es`; levels `tol_level` through `level_oracle`, per chart, historical model only. The script defines no fixed absolute tolerance. [Proposed] Test: S2.T1, S2.T2
- C.11 The axis holds these 18 relative names: BRENT_CRUDE_FWD, WTI_CRUDE_FWD and HENRY_HUB_GAS_FWD, each with [M+0], [M+1], [M+2], [M+3], [M+6], [M+12]. CO2_EUA_ANNUAL_FWD and POWER_CH_BASELOAD_HY are not used (F-12). [Proposed] Test: S1.T10
- C.12 The script oracle reads only `cf.POSITIONS`, `cf.STATIC_DFS`, `cf.VALS`, `cf.HIST_IDS`, `cf.UIDS`, `cf.FIDS`, `cf.INVERTED_UIDS`, run inputs and captured arrays. It calls only numpy, math and `_fwdoracle`. It never calls `cf.v0`, `cf._price`, `cf.calculator`, `cf.MAPPER`, `cf.RSC`, `cf.PORTFOLIO`, any riskquant symbol, `_forward_world` or `_fwdfix` (I-031). [Proposed] Test: S2.T3
- C.13 Only the script's `Capture` context manager replaces `api._value_positions` and `api._write_full_reval_block`. It restores both on exit, also on exception. A reconciliation without a captured shocked array raises. [Proposed] Test: S2.T4
- C.14 The certificate top-level keys equal F-21. The `dimensions` keys equal the SPOT keys. The certificate is one JSON file beside the script. The script is one file (F-18). [Proposed] Test: S2.T6, S6.T1
- C.15 `models` holds one record per CFGS model with `ok = true`. Records keep the SPOT key names. [Proposed] Test: S2.T7
- C.16 `reporting` holds `axis_USD`, `axis_EUR` (K_rep = 0, bound to `FIDS`), `eur_reporting` with `ok = true`, and `unconfigured_currency` = "refused:CompositionContractError" with no call of `build_run_manifest`. [Proposed] Test: S2.T8
- C.17 `identity` holds the SPOT keys. `identity.discounting_mode` equals "refused:CompositionContractError:D103.4" when the message matches `^D103\.4: .*DiscountingMode\.NONE only` on a mixed partition. No run in DiscountingMode.CURVE reaches `_row_aggregation`. [Proposed] Test: S3.T1, S3.T2
- C.18 `determinism` holds the SPOT batch keys (batch 64, 256, 2000; repeat; no cross-run leakage). Every value is a bool and true. [Proposed] Test: S3.T3
- C.19 `stress` holds one record per stress case with the SPOT keys. Each large-batch run executes in a fresh child process. `process_peak_rss_hwm_before_mib` is read after world build and warm-up; `process_peak_rss_hwm_after_mib` after run, capture and oracle. The batch-256 reference runs in another child. [Proposed] Test: S4.T1, S4.T2
- C.20 A failed RSS measurement fails the script. No estimate replaces a measurement. An RSS acceptance limit applies only if O-2 sets one. [Proposed] Test: S4.T3
- C.21 Run-level threads: for each n in {1, 2, 4, 8} with n <= cores, a child process with `NUMBA_NUM_THREADS = n` runs sbb and historical at batch 256. The observed thread count equals n. VaR and ES (`float.hex`), `identity` and `result_digest` equal the n = 1 child. `verdict.multi_core_determinism` is CERTIFIED, NOT CERTIFIED, or NOT TESTED when cores < 2 (O-3). [Proposed] Test: S5.T1, S5.T2
- C.22 Exit code is 0 only if every section is ok. [Proposed] Test: S5.T3
- C.23 No file in `src/` and no file listed out of scope in section 1 changes. [Proposed] Test: every slice gate `git diff --name-only`

## 4. Decisions

Decisions relied on: D041 (no), D043 (no), D065 (no), D081 (no), D082 (no), D084 (no), D085 (no), D101 (yes), D102 (yes), D103 (yes). Each checked against `DECISIONS.md`: yes for D101 to D103; no for the others, which the audit must check.

### P-VANILLA_FORWARD_NODE_RELEASE_CERT-001: VANILLA_FORWARD_NODE release certification record (PROPOSED)

```text
Related: D041, D043, D101, D102, D103
Scope: Record the release certification of VANILLA_FORWARD_NODE in DiscountingMode.NONE at the SPOT release dimensions.
1. The certificate of S6 is the release evidence. Its verdicts are as in C.21 and C.22.
2. The certification covers FULL_REVAL and TAYLOR/DERIVED on one mixed book, USD and EUR reporting, 5 models.
3. The certification excludes CURVE numerics, TAYLOR/SUPPLIED, VANILLA_FORWARD_CURVE and currencies other than USD and EUR.
Evidence: S6.T1 to S6.T3 and the report R_RELEASE_CERTIFICATE_<YYYY-MM-DD>.md.
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | Is K_nodes the number of history columns (rf ids, not node_meta rows), and must K_axis equal 140 exactly? | OPEN | | |
| O-2 | Is the RSS result a record only, or must it stay under a stated limit? | OPEN | | |
| O-3 | May the forward script execute the thread section (C.21) and change the value of `verdict.multi_core_determinism` from "see release gate"? May sections hold extra keys (tolerance ratios, `discounting_mode` suffix)? | OPEN | | |
| O-4 | Confirm the world name `_fwdcertfix.py` and the script path `scripts/release/forward_node/`. | OPEN | | |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1, S2, S3, S4, S5, S6. No slices run in parallel: S2 to S5 change one file.

### S1: Certification world

Objective: Create the world and prove its structure.
Contract: C.1 to C.9, C.11, C.23
Depends on: O-1, O-4
Files:
- create: tests/solutions/risk_run/_fwdcertfix.py
- create: tests/solutions/risk_run/test_vanilla_forward_node_release_world.py
Prohibited: `src/`, `DECISIONS.md`, `scripts/`, `_certfix.py`, `_fwdfix.py`, `_fwdoracle.py`, `_mixedfix.py`; pandas; imports from `scripts/`; private api symbols.
Gates: `python -m py_compile tests/solutions/risk_run/_fwdcertfix.py tests/solutions/risk_run/test_vanilla_forward_node_release_world.py` ; `ruff check` and `ruff format --check` on both files ; `mypy --strict tests/solutions/risk_run/_fwdcertfix.py` ; `pytest tests/solutions/risk_run/test_vanilla_forward_node_release_world.py -q` ; `pytest tests/solutions/risk_run -q` ; `git diff --name-only`
Exit: All gates pass. The test file runs one sbb run on `inputs(mixed=True)` and the run is admitted.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | Names and types of C.1 | all present, exact dtypes and layouts | interface drift against the mirrored script |
| S1.T2 | AST imports of the world | no forbidden module | hidden dependency on demos or pandas |
| S1.T3 | Dimensions | C.3 values | wrong scale |
| S1.T4 | FX_ID from mapper vs raw id | equal | 1-based id assumption (F-4) |
| S1.T5 | Id table and collisions | C.5 holds; compiler accepts `STATIC_DFS` | rf or curve id collision |
| S1.T6 | Book composition from `POSITIONS` | C.6 counts | missing exact, relative or inverted coverage |
| S1.T7 | Currency sets | subset of {USD, EUR} | unsupported FX route |
| S1.T8 | Partition rules | C.8 holds | D103.3 refusal of the mixed run |
| S1.T9 | Axis for USD and EUR | K_rep = 0, equal to FIDS | oracle FX row outside capture |
| S1.T10 | Relative names on the axis | the 18 names of C.11 | token not admitted at ingest |

Appendix: none

### S2: Release script, models and reporting

Objective: Create the script with Capture, the oracle, `models`, `reporting`, `dimensions`.
Contract: C.7, C.10, C.12 to C.16
Depends on: S1, O-3
Files:
- create: scripts/release/forward_node/certify_forward_node_production_scale.py
- create: tests/solutions/risk_run/test_vanilla_forward_node_release_script.py
Prohibited: as S1; a second `.py` file under `scripts/release/`; private hooks outside `Capture`; fixed absolute tolerances.
Gates: `python -m py_compile` on both files ; `ruff check` and `ruff format --check` on both files ; `pytest tests/solutions/risk_run/test_vanilla_forward_node_release_script.py -q` ; `python scripts/release/forward_node/certify_forward_node_production_scale.py --json-out /tmp/fwd_s2.json` ; `git diff --name-only`
Exit: `models` and `reporting` records have `ok = true` in `/tmp/fwd_s2.json`.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S2.T1 | AST search for numeric tolerance constants | none | fixed SPOT ATOL reuse |
| S2.T2 | Detector: scale one captured dW row by (1 + 1e-6) | reconciliation fails | tolerance too wide to detect a defect |
| S2.T3 | AST of oracle functions | only C.12 reads and calls | oracle that calls production (I-031) |
| S2.T4 | AST of module; Capture exit after a raised run | hooks only in Capture; both restored | leaked monkeypatch |
| S2.T5 | Oracle route for GBP | ValueError | silent FX route |
| S2.T6 | Script files under `scripts/release/forward_node/` | one `.py` | helper run as a release script (F-18) |
| S2.T7 | Five model records | `ok = true` | model-specific defect |
| S2.T8 | EUR reporting and "ZZZ" | `ok = true`; refusal before identity | reporting route defect |

Appendix: none

### S3: Identity mutations, batch determinism, CURVE control

Objective: Add `identity` and `determinism` (batch) sections.
Contract: C.17, C.18
Depends on: S2
Files:
- modify: scripts/release/forward_node/certify_forward_node_production_scale.py
- modify: tests/solutions/risk_run/test_vanilla_forward_node_release_script.py
Prohibited: as S2; any run in DiscountingMode.CURVE that reaches valuation.
Gates: as S2, with `--json-out /tmp/fwd_s3.json`
Exit: `identity` and `determinism` satisfy C.17 and C.18.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S3.T1 | CURVE context, mixed partition | "refused:CompositionContractError:D103.4" | CURVE admitted for U_D |
| S3.T2 | Spy on `_row_aggregation` in the CURVE control | zero calls | refusal after numerics |
| S3.T3 | Batch 64, 256, 2000; repeat; other model between runs | all true | batch or state leakage |

Appendix: none

### S4: Stress and RSS record

Objective: Add the `stress` section with fresh-child RSS.
Contract: C.19, C.20
Depends on: S3, O-2
Files:
- modify: scripts/release/forward_node/certify_forward_node_production_scale.py
Prohibited: as S2; full U x S oracle matrices at S = 100_000.
Gates: as S2, with `--json-out /tmp/fwd_s4.json`
Exit: Both stress records have `ok = true` and measured RSS values.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S4.T1 | Stress S=20_000 and S=100_000 | oracle within C.10; same-S bitwise equal | large-batch defect |
| S4.T2 | Child process ids of stress runs | differ from the parent | HWM polluted by earlier sections (F-16) |
| S4.T3 | RSS reader raises | script exits non-zero | substituted memory value |

Appendix: none

### S5: Run-level thread section and verdict

Objective: Add the thread checks and the final verdict.
Contract: C.21, C.22
Depends on: S4, O-3
Files:
- modify: scripts/release/forward_node/certify_forward_node_production_scale.py
Prohibited: as S2; thread claims from configuration only.
Gates: as S2, with `--json-out /tmp/fwd_s5.json`
Exit: `verdict` fields satisfy C.21 and C.22 on a host with at least 2 cores.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S5.T1 | Thread probe per n | observed equals n | clamped thread ceiling |
| S5.T2 | VaR, ES, identity, result_digest per n | equal to n = 1 | thread-dependent forward result |
| S5.T3 | One section `ok = false` | exit code 1 | false release pass |

Appendix: none

### S6: Owner release execution

Objective: Run the script on the owner host and record the evidence.
Contract: C.14, C.22
Depends on: S5, O-1 to O-4
Files:
- create: scripts/release/forward_node/forward_node_production_scale_certificate.json
- create: docs/workstreams/WS_VANILLA_FORWARD_NODE_RELEASE_CERT/R_RELEASE_CERTIFICATE_<YYYY-MM-DD>.md
Prohibited: any source change; any edit of the JSON by hand.
Gates: `sha256sum scripts/demos/e2e/_forward_world.py scripts/release/spot/certify_spot_production_scale.py` ; `python scripts/release/forward_node/certify_forward_node_production_scale.py` ; `python scripts/run_all_scripts_release.py` ; `pytest -q`
Exit: Exit code 0. Verdicts CERTIFIED. The full suite passes.
Status: NOT_STARTED
Evidence: none

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S6.T1 | Certificate keys vs SPOT certificate | F-21 sets equal | section drift |
| S6.T2 | Release runner | both release scripts pass | runner conflict |
| S6.T3 | Repository hashes in the report | recorded with `dirty=false` | unbound evidence (F-1) |

Appendix: none

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| Relative tokens [M+2] to [M+12] or HENRY_HUB fail ingest | C.11 cannot hold | S1.T10 runs first; stop and report; no production change |
| Forward run is not thread invariant | C.21 fails | stop at S5; report as a production finding |
| RSS at S=100_000 exceeds the owner host | S4 cannot run | record the failure; O-2 decides |
| `_fwdoracle` forms do not bound production at M=200 | false failure | report the ratio; do not widen a tolerance (CLAUDE.md section 8) |
| 77 synthetic curves fail static-config validation | S1 blocked | report the validator message |

Deferred:
- TAYLOR/DERIVED in DiscountingMode.CURVE: opens with D104.
- CURVE FULL_REVAL numerics: opens with a CURVE workstream.
- Currencies other than USD and EUR: opens when an owner gate requests cross routes.
- Repeated UIDs and SPOT rows under forward UIDs at release scale: opens on owner request.
- Production-data readiness: opens with the spot production data workstream.

## 8. Closure record

This section stays empty until the owner sets CLOSED.
