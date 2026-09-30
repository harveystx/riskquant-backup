# RiskQuant Proxy Model V1: Design Specification

## 0. Document status

| Field | Value |
|---|---|
| Title | RiskQuant Proxy V1: Design Specification |
| Proxy version | V1 |
| Revision | V01 with closure pass of 2026-09-28: identity domains, D088 §13, Slice 2 numerical contract, NO_SHIFT rule, binary64 wording, legacy status. |
| Design status | Frozen. D088 is recorded as "LOCKED" with "contract and design FROZEN 2026-09-25". D088 §12 adds five precision clarifications and states "no semantic change". |
| Implementation status (as recorded in D088, 2026-09-29) | "D088 STAGE D COMPLETE" and "D088 STAGE E COMPLETE": production executes Proxy V1 over the full run axis; `ShockIdentity` is `proxy_model_digest`; D013 is removed from the repository. The authoring file format remains OPEN (§13.2). Status rows and statements elsewhere in this document that say "not implemented", "Slice 3" or "Slice 4" describe the 2026-09-28 snapshot and are superseded by this row. |
| Slice 1 (semantic compilation and identity) | Closed. The compiler, canonical representation and digest function exist in the current snapshot, and their tests pass (Section 36). |
| Slice 2 (numerical seam and binding function) | Implemented and independently validated; not wired into the production run. `src/riskquant/simlab/shocks/{linear_term_kernel,linear_term_apply,column_binding}.py` implement the closed contract (D088 §13.4 to §13.6; Section 26). |
| Slice 3 and later | Implemented 2026-09-29: world-side `(d, τ, h)` columns (`bind_operator_to_world`, `build_world_column_plan`), the production run (`run_scenario_var_es`), the §13.3 manifest rule (v5), and the removal of D013 (Stage E). |
| Production proxy behaviour today | The legacy D013 input contract of `run_scenario_var_es` (`required_shift_type_id`, `source_factor_ids`, `shock_scale`). The legacy resolver still exists but has no caller under `src/`. `SimlabRunner` has been removed from the code base (`DECISIONS.md` update notes, 2026-09-28). |
| Authoritative decisions | D088 including §12 and §13; D089 (parametric two-sided horizon contract). D088 supersedes D013 and amends, by recorded status notes, D026 F7, D034, D036, D040 (`ShockIdentity`), D082 clause 11, D085 clause 8 and D087 clause 2. D025, D041, D060, D067, D069, D070, D075, D078.4, D081 and D085 are referenced and remain in force. |
| Nature of this document | Normative and explanatory. Content labelled DECISION and the rules in Section 41 restate D088. All other content explains D088, the repository and their consequences. If this document and D088 differ, D088 governs. |
| Repository snapshot | Project context dumps as supplied on 2026-09-28; `DECISIONS.md` containing D088 §12 and §13. |

---

## 1. Evidence conventions

Every substantive statement in this document carries one of the following labels, either inline or in the heading of the block it belongs to.

| Label | Meaning |
|---|---|
| **FACT** | Verified from repository code, repository tests, or the text of a decision record. |
| **DECISION** | Adopted by the project. Where a decision is recorded in `DECISIONS.md`, the record is cited. Where a decision is adopted in Slice 1 code and tests but is not recorded in `DECISIONS.md`, this is stated. |
| **RATIONALE** | The reason for a decision. Unless D088 states the reason, the rationale is an interpretation by this document. |
| **LEGACY** | Behaviour of the pre-V1 (D013) implementation, reconstructed from repository code. |
| **EXECUTED** | A result obtained by running repository code or tests in a scratch copy of the snapshot. Such results are evidence, not repository artifacts. |
| **OPEN** | Not decided. No decision is invented for it. |
| **Not yet specified** | A detail that later implementation must fix, with no decision today. |

Citations use `D088 §n`, `D088 §12.n` (the n-th clarification), `src/...` and `tests/...`.

---

## 2. Purpose and scope

### 2.1 Problem

**FACT.** A RiskQuant risk run produces one scenario level for every run-axis factor in every scenario (D081 §1; D085 decision 4). Some run-axis factors do not have their own scenario data, or their scenario movement is to be defined from the movement of other factors.

**DECISION (D088 title and §2).** Proxy V1 defines the target factor's scenario shock as a model over explicit driver shocks. It does not substitute one factor for another.

### 2.2 Definition

Proxy V1 is the contract recorded in D088. For every V1 target node, the scenario shock of the target is a declared linear combination of world shock columns, and the target's own reconstruction chart turns that shock into a scenario level.

### 2.3 Layers and scope

| Layer | Part of Proxy V1? | Owner | Status (current snapshot) |
|---|---|---|---|
| Proxy specification (authoring) | Semantic content yes (D088 §12.3). File format no: "a separate edge concern". | Proxy configuration layer | Semantic records implemented (`src/riskquant/config/proxy_model/schema_objects.py`). No file loader exists. File format **OPEN**. |
| Proxy compilation | Yes | Proxy compiler | Implemented (Slice 1) |
| Proxy identity | Yes | Canonical model and `hash_proxy_model` | Function implemented (Slice 1). Not yet used by the run manifest. |
| World shock generation | No. The world supplies columns (D088 §4). | Scenario world and generators | Legacy generators exist. No V1 column request. |
| World-column binding | The seam is defined by D088 §12.5. | Run and world seam | Integer-key binding function implemented (Slice 2: `bind_term_columns`). World-side column production is Slice 3. |
| Numerical shock aggregation | Yes, as operator semantics (D088 §2). | Numerical proxy application | Implemented (Slice 2); not wired into the production run |
| Target reconstruction | V1 fixes its order and its owner. The inverse charts already exist. | Numerical application, using the target-owned chart | V1 path implemented (Slice 2); not wired into the production run. The legacy inverse kernel is unchanged. |
| Valuation | No | Existing valuation | Unchanged |
| P&L | No | Existing risk-run machinery (D081) | Unchanged |
| Risk aggregation (VaR, ES) | No | Existing risk-run machinery | Unchanged |
| Reporting FX | No (D088 §9: "Reporting FX (D081, D085) is outside the proxy model") | Risk run reporting route (D085) | Unchanged |

---

## 3. Executive definition

**DECISION (D088 §2).** For target $A$:

$$
X_A=\sum_{j\in J_A}\beta_{Aj}\,Y_j,
\qquad
Y_j=Y_{(d_j,\tau_j,h)},
$$

followed by target-owned reconstruction:

$$
A_s=T^{-1}_{\kappa_A}\!\left(A_0,\;X_A(\omega_s)\right).
$$

| Symbol | Meaning |
|---|---|
| $A$ | Target factor-node. Its identity is the run-resolved `rf_id` (D088 §3). |
| $J_A$ | The set of terms of target $A$. It is non-empty after canonicalisation. |
| $d_j$ | Driver factor-node of term $j$. |
| $\tau_j$ | Driver chart of term $j$. It fixes how the driver's movement is represented. |
| $\beta_{Aj}$ | Loading of term $j$: signed, finite, float64 (D088 §2). |
| $h$ | Horizon declared by the scenario world, in the world's declared step unit (D088 §2, §3). |
| $Y_{(d,\tau,h)}$ | World shock column: the $h$-horizon movement of $d$ in chart $\tau$ (D088 §4). |
| $X_A$ | Aggregated target shock, in the space of the target's chart $\kappa_A$. |
| $\kappa_A$ | Target reconstruction chart: the configured `shift_type` of the target factor (D088 §3). |
| $A_0$ | Base level of $A$, bound to the run (D088 §3; D081 §1). |
| $\omega_s$ | Scenario $s$ of the world. |
| $A_s$ | Reconstructed scenario level of $A$ in scenario $s$. |
| $T^{-1}_{\kappa}$ | Inverse of chart $\kappa$: the map from a base level and a shock to a level. |

A Proxy V1 model defines a relationship between a target risk factor and one or more explicitly identified driver factor/chart terms.

It does not define "choose one source factor and scale its shock". That was the legacy definition (D013), and D088 supersedes it (Sections 7 and 8).

---

## 4. Core ontology

| Object | Meaning | Owner | Present in compiled model? |
|---|---|---|---|
| Target factor ($A$) | A run-axis factor-node whose shock the model defines | Factor definition and run axis | Yes, as `row_target_rf_ids` |
| Driver factor ($d$) | A factor-node whose world column appears in a term | Proxy model (by reference) | Yes, as `term_driver_rf_ids` |
| Driver chart ($\tau$) | The chart in which the driver's movement is represented | Proxy model | Yes, as `term_chart_ids` |
| Target chart ($\kappa_A$) | The chart that turns $X_A$ into a level | Target factor definition (D088 §3) | No. It appears only as the $\tau$ of a materialised identity term, where $\tau=\kappa_A$ by definition. |
| Loading ($\beta$) | The coefficient $\partial X_A/\partial Y_j$ (D088 §5) | Proxy model | Yes, as `term_loadings` |
| Term | One summand $\beta\,Y_{(d,\tau,h)}$ of $X_A$ | Proxy model | Yes, in canonical form |
| Authored term | A term as written by the author: names, chart, loading. Either a node term or a container term. | Proxy configuration layer | No |
| Resolved term | An authored term after name resolution and container expansion: (target `rf_id`, driver `rf_id`, chart id, loading) | Proxy compiler (internal) | No |
| Canonical term | A resolved term after grouping, merging and zero removal (D088 §6) | Proxy compiler | Yes |
| Target row | All canonical terms of one target | Proxy compiler | Yes, one per V1 axis node |
| Run axis | The ordered factor axis of one run (D082; D085 decision 4) | Production run | Rows follow its order. The axis itself is an input, not stored. |
| Compiled proxy model | The canonical run-bound operator (D088 §12.1) | Proxy compiler | It is this object. |
| World shock column | The random variable $Y_{(d,\tau,h)}$ | Scenario world | No |
| Horizon ($h$) | The world's declared elapsed horizon | Scenario world (D088 §3) | No (D088 §12.5) |
| Concrete column identity | The triple $(d,\tau,h)$ | World, bound by the run | No. The model holds $(d,\tau)$ only. |
| Native factor | An axis node whose configured `shift_type` is `ABSOLUTE_BASE_RETURN` or `NO_SHIFT` | D025 native path | No (D088 §1, §12.4) |
| Identity term | The term $(A,\kappa_A,1)$ that totality gives to a V1 node with no authored terms | Proxy compiler | Yes, as an ordinary term |
| Proxy digest (`proxy_model_digest`) | `hash_proxy_model(model)`. It identifies the proxy operator only. | Run identity layer (function) | No. It is a function result (D088 §12.1). |
| Target-definition identity | Identity of $(A,\kappa_A)$ and the rest of the target configuration | `RunContextIdentity` (`static_config_digest`, `universe_map_digest`) | No |
| World identity | Identity of the scenario world: law, seed, count, horizon, data | `scenario_config` and `DataIdentity` in the manifest | No |
| Run identity ($ID_{\text{run}}$) | `RunManifest.identity()`: every input capable of changing the result | Run identity layer | No |

The central distinction:

$$
\underbrace{(d,\tau)}_{\text{declared by the proxy model}}
\qquad\text{versus}\qquad
\underbrace{(d,\tau,h)}_{\text{identity of a concrete world column}}
$$

---

## 5. Mathematical notation

| Symbol | Definition |
|---|---|
| $A$, $A_n$ | Target factor-node. $A_n$ is the target of row $n$. |
| $d$, $d_t$ | Driver factor-node. $d_t$ is the driver of term $t$. |
| $\tau$, $\tau_t$ | Driver chart. $\tau\in\{\text{LOG},\text{SIMPLE},\text{DIFF}\}$ in V1. |
| $\kappa_A$, $\kappa_n$ | Target reconstruction chart of $A$, or of row $n$. |
| $\beta_{Aj}$, $\beta_t$ | Loading of term $j$ of $A$, or of canonical term $t$. |
| $Y_j$, $Y_{(d,\tau,h)}$ | World shock column. |
| $X_A$, $X_n$ | Aggregated target shock. |
| $A_0$, $A_{0,n}$ | Base level. |
| $A_s$, $A_n^{(b)}$ | Reconstructed level in scenario $s$, or in batch position $b$. |
| $h$ | World horizon. |
| $s$ | Scenario index in the world. |
| $b$ | Scenario position inside one batch, $b\in[0,B)$. |
| $c$, $c_t$ | World column index. $c_t$ is the column bound to term $t$. |
| $t$ | Canonical term index. |
| $n$ | Row index: position of the target among the V1 axis nodes, in axis order. |
| $o_n$ | Row offset. Row $n$ owns terms $t\in[o_n,o_{n+1})$. |
| $K$ | Number of world columns in a batch. |
| $N$ | Number of rows. |
| $\mathcal B$ | The canonical operator: the sparse $N\times K$ matrix with $\mathcal B_{n,c_t}=\beta_t$ (Section 10). |
| $T_\tau$, $T^{-1}_\kappa$ | Forward chart (levels to shock) and inverse chart (base and shock to level). |

$\tau$ and $\kappa_A$ are different concepts:

* **Driver chart $\tau$** determines the representation of the driver shock: what number $Y_{(d,\tau,h)}$ is, for a given movement of $d$.
* **Target chart $\kappa_A$** determines how the aggregated shock $X_A$ is applied to the base level $A_0$.

A term reads a column in chart $\tau$. The target applies the sum in chart $\kappa_A$. D088 §3 states that the configured `shift_type` of a driver node does not imply $\tau$, and that $\kappa_A$ is never owned by the proxy model.

---

## 6. Chart semantics

### 6.1 The chart enumeration

**FACT.** The repository enumeration is `ShiftType` in `src/riskquant/market_data/risk_factors/shift_types.py`: `NO_SHIFT = -1`, `LOG_RETURN = 1`, `SIMPLE_RETURN = 2`, `DIFFERENCE = 3`, `ABSOLUTE_BASE_RETURN = 4`. The value `0` is intentionally unused (D025). D088 uses "chart" for a member of this enumeration used in a role: a driver chart $\tau$, or a target reconstruction chart $\kappa$. Legacy code calls the same values "shift types".

### 6.2 Definitions

| Chart | Forward (construct a column from levels) | Inverse (reconstruct a level) | Source |
|---|---|---|---|
| LOG (`LOG_RETURN`) | $\ln(v_1/v_0)$, requires $v>0$ | $A_0\,e^{x}$ | D088 §2; `simlab/scenarios/historical.py`; `simlab/shocks/shock_kernel.py` |
| SIMPLE (`SIMPLE_RETURN`) | $v_1/v_0-1$, requires $v_0\neq0$ | $A_0(1+x)$ | D088 §2; same files |
| DIFF (`DIFFERENCE`) | $v_1-v_0$, requires finite levels | $A_0+x$ | D088 §2; same files |
| ABS (`ABSOLUTE_BASE_RETURN`) | $(v_1-v_0)/\lvert v_0\rvert$, requires $v_0\neq0$ | $A_0+x\,\lvert A_0\rvert$ | D025; `shock_kernel.py` |
| NO_SHIFT | No forward map is needed, because reconstruction ignores the shock. The historical-family kernels write a zero column without reading levels. The parametric generator does not special-case NO_SHIFT. No world is required to supply a zero column. | $A_0$ for every shock | D025; D088 §1, §13.4 |

**FACT (D088 §4, D025).** Multi-step composition, which applies only when a column is constructed from levels:

* LOG and DIFF compose by sum.
* SIMPLE composes by product-minus-one.
* ABS: D025 records the Historical endpoint read $(V_t-V_{t-h})/\lvert V_{t-h}\rvert$ and a daily product in the daily-composing engines. D025 accepts that these differ on negative paths. D088 §10 records the example: on $-10\to-9\to-8.1$ the endpoint statistic is $0.19$ and the sign-blind product is $0.21$.

**FACT (D088 §4).** A world column may be supplied directly in shock space or constructed from levels. A directly supplied column carries no level path and no level-domain obligation. The level-domain rules above apply only to level-backed construction (for the historical family: `src/riskquant/simlab/builder/shift_domain.py`, D025).

**FACT (D088 §2; D070; D075).** The reconstruction output must be finite. Whether the reconstructed level is admissible for the factor (finite, and $>0$ for `FX_RATE` and `VOL`) is decided by the existing factor-state guard in `build_market_data_matrix` (`src/riskquant/market_data/builder.py`), not by the chart.

### 6.3 Roles permitted in V1

| Chart | As driver chart $\tau$ | As target chart $\kappa_A$ | Status |
|---|---|---|---|
| LOG | Yes | Yes | V1 |
| SIMPLE | Yes | Yes | V1 |
| DIFF | Yes | Yes | V1 |
| ABS | No | No | Native only (D025). Deferred as driver and target (D088 §10). |
| NO_SHIFT | No | No | Native execution behaviour. "It is not a chart and not a node status" (D088 §1). |

**FACT (`src/riskquant/config/proxy_model/compiled.py`).** `V1_TARGET_CHART_IDS = {1, 2, 3}` and `V1_DRIVER_CHART_IDS = V1_TARGET_CHART_IDS`.

A native node may still appear as a driver under a V1 chart. D088 §3: "The configured `shift_type` of a driver node does not imply `τ_j`." The Slice 1 test `test_native_nodes_may_drive_under_a_v1_chart` verifies this for an ABS-configured node and a NO_SHIFT-configured node.

---

## 7. The legacy mechanism (D013)

All statements in this section are **LEGACY** unless marked otherwise. They describe code that is still present in the current snapshot. Section 7.5 states which part production still executes.

### 7.1 Legacy configuration

**FACT.** Legacy proxy configuration is a table with the required columns `proxying_level`, `required_name`, `proxy_name` and `vol_scaling_factor` (`src/riskquant/config/proxy_config/validate.py`). Legacy validation rejects `vol_scaling_factor` outside $(0,10]$ (`_VOL_SCALE_MIN_EXCLUSIVE = 0.0`, `_VOL_SCALE_MAX_INCLUSIVE = 10.0`). It also rejects foreign-key failures, proxy chains and cross-level conflicts.

The canonical fixture (`src/riskquant/ingest_fixtures/config/risk_proxy_config/canonical_risk_proxy_config.py`) contains these rows:

| `proxying_level` | `required_name` | `proxy_name` | `vol_scaling_factor` |
|---|---|---|---|
| CONTAINER | WTI_CRUDE_FWD | BRENT_CRUDE_FWD | 0.95 |
| CONTAINER | HENRY_HUB_GAS_FWD | BRENT_CRUDE_FWD | 1.40 |
| CONTAINER | POWER_CH_BASELOAD_HY | CO2_EUA_ANNUAL_FWD | 1.20 |
| RISK_FACTOR | USDGBP | USDEUR | 0.85 |
| RISK_FACTOR | EURCHF | USDEUR | 0.60 |
| RISK_FACTOR | USD_LIBOR_3M[10Y] | USD_SOFR_OIS[10Y] | 1.00 |

`compile_proxy_topology_from_df` (`src/riskquant/config/proxy_config/compile.py`) produces `ProxyTopology` (`proxy_topology.py`). It holds dense arrays: `rf_id_to_proxy_rf_id`, `rf_id_to_proxy_vol_scale`, `container_proxy_id`, `container_proxy_vol_scale`.

### 7.2 Legacy source resolution

**FACT (`src/riskquant/proxy_resolver/proxy.py`).** The resolver has this signature:

```text
resolve_required_factor_ids_to_market_data_factor_ids(
    static_config, proxy_topology, required_factor_ids, market_data_factor_ids)
    -> ProxyResolution
```

`market_data_factor_ids` is the set of factors for which scenario data exists. For each required factor, the resolver selects one source by the first tier that applies:

| Code | Tier | Rule | Scale |
|---|---|---|---|
| `RES_DIRECT = 0` | (1) DIRECT | The required factor is in the market data. Source = itself. | 1.0 |
| `RES_ID_PROXY = 2` | (2) RF-level proxy | A risk-factor proxy is registered, and the proxy must exist in the market data | `vol_scaling_factor` |
| `RES_CONTAINER_PROXY = 3` | (3) Container proxy | The parent container is proxied. The matching tenor is chosen on the proxy container, restricted to the market data. Curves only. | container scale |
| `RES_TENOR_FALLBACK = 1` | (4) Same-container tenor fallback | No proxy registered. The nearest tenor present on the same container is used. | 1.0 |
| `RES_NOT_SIMULATED = -1` | (5) | Nothing resolves. A `PrettyKeyError` is raised. | |

`ProxyResolution` carries `required_factor_ids` (int32), `source_factor_ids` (int32), `required_to_source_index`, `shock_scale` (**float32**, "multiplicative scale applied to the source shock"), `unique_source_factor_ids` and `resolution_code`.

### 7.3 Legacy shock generation

**FACT.**

* `build_shock_universe` (`src/riskquant/simlab/builder/shock_universe.py`) keys world columns by (source `rf_id`, **required** shift type id). The chart of a column is therefore the target's configured shift type, not a property of the term.
* The historical generator (`src/riskquant/simlab/scenarios/historical.py`) computes each column with the forward chart given by that key, applied to the source's levels. For example, it computes `math.log(v1 / v0)` for a LOG key.
* The parametric generator (`src/riskquant/simlab/scenarios/parametric.py`) draws $\sigma_S(Lz)_S$ per source factor and does not use the shift id in generation. `ParametricRiskData` (`parametric_data.py`) carries `factor_ids`, `cholesky_matrix` and `volatilities`, with no chart and no horizon.

### 7.4 Legacy application

**FACT (`src/riskquant/simlab/shocks/shock_kernel.py`, `_kernel_apply_shocks`).** For each required row $n$, the kernel reads one column, `shocks[:, required_to_shock_uix[n]]`, and one scale, `shock_scale[n]` (float64 inside the kernel; float32 in `ProxyResolution`). It then applies:

| Target shift type | Legacy level |
|---|---|
| LOG | `base * exp(shock * scale)` |
| SIMPLE | `base * (1 + shock * scale)` |
| DIFF | `base + shock * scale` |
| ABS | `base + shock * scale * abs(base)` |
| NO_SHIFT | `base` |

### 7.5 Legacy consumers

**FACT.**

* `run_scenario_var_es` (`src/riskquant/solutions/risk_run/api.py`) is the production consumer of the legacy mechanism. It takes `required_shift_type_id`, `source_factor_ids` (int32), `shock_scale` (float32) and `base_levels` from the caller.
* `_require_shock_sources` in the same file enforces that every source is a data column, and that "resolution tier 1 (DIRECT) wins first: a row whose own factor is in the scenario data reads its own column".
* `ShockIdentity` (`src/riskquant/solutions/run_identity/schemas.py`) digests `required_shift_type_ids`, `source_factor_ids` and `shock_scale`.
* The resolver `resolve_required_factor_ids_to_market_data_factor_ids` has no caller under `src/`. Its callers in the snapshot are `tests/_support/spot_scenario.py` and `tests/market_data/remediation/test_relative_spot_as_of_identity.py`. Demo scripts are not in the snapshot.
* `SimlabRunner` (`src/riskquant/simlab/runner.py`) was legacy and out of V1 scope. It has been removed from the code base (`DECISIONS.md` update notes, 2026-09-28).

The legacy production behaviour has not been replaced. It is reached through the `run_scenario_var_es` input contract, with arrays built by callers.

### 7.6 Legacy formula

Write $S=S(A)$ for the selected source and $c_A$ for the selected scale.

Historical family:

$$
A_s=T^{-1}_{\kappa_A}\!\left(A_0,\;c_A\cdot T_{\kappa_A}\!\left(S_{t_s-h},S_{t_s}\right)\right).
$$

Parametric:

$$
A_s=T^{-1}_{\kappa_A}\!\left(A_0,\;c_A\,\sigma_S\,\varepsilon_{S,s}\right).
$$

In both cases the legacy mechanism is $A_s=T^{-1}_{\kappa_A}(A_0,\;Y_{\text{source},s}\times c_A)$, where $Y_{\text{source}}$ is computed in the target's chart.

### 7.7 Legacy data flow

```text
legacy table (proxying_level, required_name, proxy_name, vol_scaling_factor)
    |
    v
ProxyTopology (dense proxy map + positive scale)
    |
    v   + market_data_factor_ids (which factors have data)
resolver: DIRECT > ID_PROXY > CONTAINER_PROXY > TENOR_FALLBACK
    |
    v
source_factor_ids[n], shock_scale[n] (float32), resolution_code[n]
    |
    v
shock universe keyed (source, TARGET shift type)
    |
    v
source shock computed in the TARGET's chart
    |
    x   scalar shock_scale[n]
    v
T^-1 of the TARGET's chart  ->  target level
```

D088 records the supersession of D013: "The definition of proxying above ... is no longer authoritative."

---

## 8. Why the legacy model was inadequate

### 8.1 Source selection versus relationship modelling

**LEGACY.** The resolver answers one question: which existing factor's column represents this target? Its output is one index and one scale per row.

**DECISION (D088 §2, §9).** V1 answers a different question: what mathematical relationship defines the target shock in terms of explicit driver shocks? D088 §9: "The model determines the source columns; data availability validates those references and never chooses between models; there is no fallback tier."

**RATIONALE.** These are different problems.

* A selection is a function from (configuration, data availability) to one column. Its result can change when data availability changes.
* A relationship is a function from driver shocks to the target shock. It is fixed by the model and independent of availability.

Risk methodology needs the second, because the meaning of a target's scenario must not depend on which columns happen to be present.

### 8.2 One source versus multiple terms

**LEGACY.** Each row has exactly one `source_factor_ids[n]` and one `shock_scale[n]`.

$$
X_A=0.7\,Y_1+0.3\,Y_2
$$

therefore has no legacy representation: there is no second index and no second coefficient.

**DECISION (D088 §2).** $J_A$ has one or more terms. V1 can express the following, which the legacy mechanism cannot:

* weighted blends of two or more drivers;
* exact multi-factor identities, such as FX triangulation with loadings $(+1,-1)$ (D088 §11);
* the same driver under two charts (Section 21, Example 3);
* authored shock-space interpolation weights (D088 §8).

### 8.3 Scalar scaling versus signed loadings

**LEGACY.** `vol_scaling_factor` lies in $(0,10]$. `shock_scale` is float32 and "multiplicative scale applied to the source shock". The name and the bound describe a positive rescaling of a volatility.

Consequences:

* A negative relationship cannot be written. FX triangulation, for example, needs $-1$ on the reciprocal pair.
* The float32 storage changes authored values: `0.95` becomes `0.949999988079071` (**EXECUTED**, relative difference $1.25\times10^{-8}$).
* The coefficient's meaning is presented as a volatility adjustment, although it also fixes the dependence between target and driver.

**DECISION (D088 §2, §5).** $\beta$ is a signed, finite, float64 modelling coefficient, $\beta_{Aj}=\partial X_A/\partial Y_j$. "No magnitude bound is part of the contract."

### 8.4 Driver chart versus target reconstruction chart

**LEGACY.** The legacy mechanism has no driver chart. The column chart is the target's configured shift type (Section 7.3), and the source's own configured shift type is not read.

**DECISION (D088 §2, §3).** $\tau_j$ is explicit per term. The configured `shift_type` of a driver does not imply $\tau_j$, and $\kappa_A$ is owned by the target.

Discriminating example (D088 §11):

* LOG target, $A_0=70$.
* One term: loading $0.04$ on a spread driver $S$ read under DIFF.
* Spread path $2.00\to2.50$.

| Computation | Formula | Result |
|---|---|---|
| V1 (correct) | $70\,e^{0.04\,(2.50-2.00)}=70\,e^{0.02}$ | $71.41409380$ |
| Target-chart routing, same loading 0.04 | $70\,e^{0.04\ln(2.50/2.00)}$ | $70.62759867$ |
| Target-chart routing, legacy loading 0.95 | $70\,e^{0.95\ln(2.50/2.00)}$ | $86.52917288$ |

All three values are **EXECUTED** arithmetic.

**Correction.** The request for this document, and the earlier Slice 1 gate audit, paired $86.5292$ with the $0.04$ example. That pairing is not a same-loading comparison. $86.5292$ uses the loading $0.95$, which is the legacy WTI scale. With the same loading $0.04$, target-chart routing gives $70.6276$. The Slice 1 test `test_driver_chart_is_independent_of_target_chart` asserts the same-loading difference: it checks that the misrouted value differs from $71.4141$ by more than $0.5$.

**RATIONALE.** Target-chart routing measures the driver's movement in the target's chart. For a spread this means a relative change (ratio $1.25$) instead of an absolute change ($0.50$). For a spread near zero the relative change is unbounded. For a spread that crosses zero, the LOG forward chart is undefined, and the legacy domain gate stops the run.

The conceptual error is that one symbol stands for two different facts: how the driver moved, and how the target is rebuilt. In general $\tau\neq\kappa_A$.

### 8.5 Fallback and source-selection semantics

**LEGACY.** TENOR_FALLBACK substitutes the nearest available tenor of the same container, with scale 1.0, when no proxy is registered and the required tenor has no data. DIRECT substitutes the factor's own column whenever that column exists, and `_require_shock_sources` enforces this.

**DECISION (D088 §9, §12.5; D085 clause 8 status note).** The model declares its terms. The world supplies or validates the required columns. A missing column is a failure.

```text
V1:      model declares relationship -> world supplies/validates column -> missing column: failure
legacy:  model declares relationship -> resolver picks something available
```

**RATIONALE.** A fallback makes the scenario meaning of a target depend on which data happened to be loaded. Two runs with the same configuration could then value different models without any change of identity inputs other than data coverage. It also hides an ingest gap as a model change.

### 8.6 Hidden availability dependence

**LEGACY.** `market_data_factor_ids` is an input of the resolver, so the resolved source depends on data availability.

**DECISION (D088 §12.1, §12.3).** The compiled model is `M = f(authored terms, static configuration, as-of, V1 run axis)`. The compiler "does not own ... scenario generation, world construction".

**RATIONALE.** Whether a factor has historical data, parametric data, a particular horizon, a scenario count or a batch size is a property of the world or of the execution. None of these may change what the model is. Availability is checked at the run and world seam (Section 23).

### 8.7 Other legacy defects supported by evidence

| # | Defect | Evidence |
|---|---|---|
| 1 | The configured shift type of the source is never read on a proxied row. A DIFF-configured spread used as source for a LOG target is read with `ln`. If any level of that spread is $\le0$, the run stops at the domain gate. If all levels are positive, the target moves by a power of the spread ratio. | Code, Section 7.3. **EXECUTED** in a proxy audit on an earlier snapshot; the code involved is unchanged in the current snapshot. |
| 2 | The parametric scale has no representation. One $\sigma$ per factor is read under whatever chart the target uses, so one draw can be reconstructed under two different charts. | D034 status note; `parametric.py` ignores the shift id. **EXECUTED** in the same audit. |
| 3 | Container matching uses integer token offsets. `[HY+3]` and `[Y+3]` match although they are different maturities. | `_parse_offset_from_token` and `_nearest_offset_match_batch` in `proxy.py`; D088 §8 cites this example. **EXECUTED** in the same audit on the canonical fixture row POWER_CH_BASELOAD_HY ← CO2_EUA_ANNUAL_FWD. |
| 4 | Implicit tenor fallback (Section 8.5). | `proxy.py` tier (4) |
| 5 | Data-availability precedence (Section 8.5). | `proxy.py` tier (1); `_require_shock_sources`; D085 clause 8 status note |
| 6 | Positive-only scale and float32 storage (Section 8.3). | `validate.py`; `ProxyResolution` |
| 7 | A DIFF target reads a DIFF column of any driver with a dimensionless scale. A rate target driven by a price receives the price change in price units. | **EXECUTED** in the same audit (rate $0.025\to1.525$ for Brent $80\to81.5$) |
| 8 | When one source is read under two shift ids on a single-column history, the historical generator has no matching Numba overload for $K_{req}>K_{total}$, and the run fails with a `TypeError`. | **EXECUTED** in the same audit. D024 assumed $K_{total}=1\Rightarrow K_{req}=1$. |

---

## 9. Proxy V1 design

### 9.1 Core relationship

$$
X_A=\sum_j\beta_{Aj}\,Y_{(d_j,\tau_j,h)},\qquad A_s=T^{-1}_{\kappa_A}(A_0,X_A).
$$

### 9.2 Topology

```text
authored proxy specification            (names, driver charts, loadings)
    |
    v
resolution at the run as-of             (names -> rf_id; container EXACT expansion)
    |
    v
resolved semantic terms                 (target rf_id, driver rf_id, chart id, loading)
    |
    v
admissibility                           (V1 target universe, V1 driver charts, finite loadings)
    |
    v
totality over the V1 run axis           (one row per V1 axis node; identity where nothing is authored)
    |
    v
canonicalisation                        (D088 §6)
    |
    v
compiled proxy operator                 (CompiledProxyModel; digest by hash_proxy_model)
    |
    v
world-column binding                    ((d, tau) x h -> column index)
    |
    v
sparse shock aggregation                (X = sum beta * Y)
    |
    v
target reconstruction                   (T^-1 of kappa_A)
    |
    v
levels -> existing valuation, P&L and risk machinery
```

### 9.3 Stages

| Stage | Input | Output | Owner | Status |
|---|---|---|---|---|
| Authoring | Human-authored terms | `NodeProxyTerm`, `ContainerProxyTerm` records | Proxy configuration layer | Records implemented. No file loader (**OPEN**). |
| Resolution | Records, static configuration, as-of | Resolved terms | Proxy compiler (`compile.py`), calling `build_risk_factor_name_mapping_for_as_of_date` | Implemented |
| Admissibility | Resolved terms | Accepted terms, or `ConfigurationError` | Proxy compiler | Implemented |
| Totality | Accepted terms, run axis | One term set per V1 axis node | Proxy compiler | Implemented |
| Canonicalisation | Term sets | Canonical CSR arrays | `canonical.py` | Implemented |
| Compiled operator | Canonical arrays | `CompiledProxyModel` | `compiled.py` | Implemented |
| Identity | Compiled operator | `proxy_model_digest` | `hash_proxy_model` in `run_identity/hashing.py` | Function implemented. Not in the manifest. |
| World-column binding | $(d,\tau)$ pairs, world $h$, world column set | Column index per term | Run and world seam | Binding function implemented (Slice 2); worlds do not yet supply $(d,\tau,h)$ columns (Slice 3) |
| Sparse aggregation | Shock batch, column indices, loadings | $X$ per row and scenario | Numerical application | Implemented (Slice 2); not wired |
| Reconstruction | $X$, $A_0$, $\kappa$ per row | Levels | Numerical application | Implemented (Slice 2); not wired. Legacy inverse unchanged. |
| Downstream | Levels | Valuation, P&L, VaR/ES | Existing risk run | Unchanged |

---

## 10. Why V1 is an operator

**DECISION (D088 §2).** Proxy V1 is a sparse linear operator in shock space.

Single term:

$$
X_A=\beta\,Y.
$$

Multi-term:

$$
X_A=\beta_1Y_1+\beta_2Y_2+\cdots+\beta_mY_m.
$$

All targets together:

$$
X=\mathcal B\,Y,\qquad \mathcal B\in\mathbb R^{N\times K}\ \text{sparse},\qquad \mathcal B_{n,c}=\sum_{t\in[o_n,o_{n+1}):\,c_t=c}\beta_t .
$$

* Row $n$ of $\mathcal B$ is target row $n$.
* Column $c$ is one bound world column $(d,\tau,h)$.
* An identity row is a unit vector: $\mathcal B_{n,c}=1$ on the column $(A_n,\kappa_n,h)$.

**FACT.** The compiled model stores the rows of $\mathcal B$ in compressed-row form over $(d,\tau)$ pairs (Section 20). The column index $c_t$ does not exist until binding, because it depends on $h$ and on the world's column set.

**RATIONALE.**

* The relationship to be modelled is between movements, not levels. A linear map on movements keeps the target's own base level and its own reconstruction chart (D088 §2, §3).
* Aggregation in shock space before reconstruction is the definition. Reconstructing each driver separately and then combining levels is a different model (D088 §8: level-space weighting is not shock-space weighting).
* The operator is compiled once per run and applied to every scenario and batch without change. It is the correct object to identify (Sections 19 and 30).
* The compressed-row arrays are one representation of $\mathcal B$. $\mathcal B$ itself is the conceptual model.

---

## 11. Driver identity

**DECISION (D088 §12.5).**

```text
Proxy model:   declares (d, tau) for each term
Run / world:   owns h (D088 §3: "Horizon h, step unit, scenario law | scenario world")
Binding:       (d, tau, h) -> column index c
Generator:     validates the requested h against the world declaration
```

A driver term is identified by $(d,\tau)$. A concrete world column is identified by $(d,\tau,h)$.

**FACT.** `CompiledProxyModel` has no horizon field (`compiled.py`). `compile_proxy_model` has exactly four parameters: `authored_terms`, `static_config`, `as_of_date` and `run_axis_rf_ids`. The test `test_compiler_inputs_are_exactly_the_four_semantic_inputs` pins them.

**RATIONALE.** The relationship between target and driver movements does not depend on the world's temporal representation. A concrete column does depend on it. Putting $h$ into the model would make one relationship into several models, and would let a model select a horizon that the world does not declare.

D088 §12.5 forbids inference: "No layer infers `h` from the target, the driver, a resolver, a fallback, source selection or the target chart." D088 §4 adds that "horizon-invariance of `β` is a V1 modelling assumption, not a theorem".

**DECISION (D088 §4, §10, §13.2).**

* $\beta$ is horizon-invariant by the V1 modelling assumption. $\beta=\beta(h)$ is not supported in V1; horizon-dependent loadings are deferred.
* The horizon is not part of a compiled proxy term. It is part of the concrete world-column identity $(d,\tau,h)$.
* `proxy_model_digest` does not identify a world or a run. Two runs at different horizons may legitimately share one proxy digest: they apply the same operator to two world instantiations.
* The complete run identity distinguishes the horizons through the world identity. Today the matrix-backed configurations carry `horizon`, and `RunManifest.scenario_config` records it. The parametric world declares no horizon (Section 32).

**FACT (current world representation).**

* The four matrix-backed scenario configurations carry `horizon` (`src/riskquant/solutions/scenario_models/schemas.py`).
* `ParametricRiskData` carries no horizon.
* No world representation declares a step unit.

The request of $(d,\tau,h)$ from a world is therefore not yet possible for parametric data (Section 32).

---

## 12. Target reconstruction

**DECISION (D088 §3).** "Reconstruction chart `κ_A` | target factor definition (configured `shift_type`); never the proxy model."

**FACT.**

* The configured `shift_type` of each factor is compiled into `rf_id_to_shift_type_id` (`src/riskquant/config/static_config/static_config.py`).
* The run axis carries it per row as `required_shift_type_id` in `run_scenario_var_es`, and as `RunFactorAxis` shift type ids in `src/riskquant/solutions/risk_run/schemas.py`.
* `CompiledProxyModel` contains no target chart.
* The compiler reads $\kappa_A$ for two purposes only:
  1. admissibility: is $\kappa_A$ in the V1 set?
  2. the driver chart of a materialised identity term, where $\tau=\kappa_A$.

$$
\text{driver representation}=\tau
\qquad\text{versus}\qquad
\text{target reconstruction}=\kappa_A .
$$

**RATIONALE.** Because $\kappa_A$ is carried by the factor definition and the run axis, the proxy operator can be a pure relationship over $(d,\tau,\beta)$. It needs no copy of reconstruction semantics, and a factor definition cannot disagree with a proxy model about how the factor is rebuilt.

The Slice 1 test `test_target_chart_is_not_part_of_the_proxy_identity` changes WTI's configured chart among LOG, SIMPLE and DIFF. It shows that the digest of an authored model on WTI does not change.

---

## 13. Authoring forms

### 13.1 Implemented semantic records

**FACT (`src/riskquant/config/proxy_model/schema_objects.py`).**

| Record | Fields | Meaning |
|---|---|---|
| `NodeProxyTerm` | `target_name`, `driver_name`, `driver_chart: ShiftType`, `loading: float` | One term on exact market points. The author selects the pairing (D088 §8: `EXPLICIT`). |
| `ContainerProxyTerm` | `target_container_name`, `driver_container_name`, `driver_chart`, `loading` | Authoring shorthand. It expands at compile time to one node term per target pillar, paired by `EXACT` (D088 §8). |

Neither record carries a target chart or a horizon.

### 13.2 File format

**OPEN.** No authoring file format exists. D088 §12.3: "The authoring file format is a separate edge concern." No loader and no canonical Proxy V1 ingest fixture exist. The legacy D013 loader and its canonical ingest fixture were removed in Stage E (2026-09-29); they could not express $\tau$.

### 13.3 Illustrative example

The following YAML is illustrative only. No repository parser reads it. The factor names are not in the repository fixtures.

```yaml
proxies:
  - target: VER_EUR
    terms:
      - driver: EUA_EUR
        driver_chart: LOG
        loading: 0.7
      - driver: CER_EUR
        driver_chart: LOG
        loading: 0.3
```

The example requested for this document also contained `target_chart: LOG`. V1 authoring does not carry a target chart. $\kappa_{VER}$ is the configured `shift_type` of `VER_EUR` (D088 §3). An authored copy would be a second owner of the same fact, and D088 does not provide for a target-chart assertion.

As implemented records:

```text
NodeProxyTerm(target_name="VER_EUR", driver_name="EUA_EUR", driver_chart=LOG_RETURN, loading=0.7)
NodeProxyTerm(target_name="VER_EUR", driver_name="CER_EUR", driver_chart=LOG_RETURN, loading=0.3)
```

Canonical semantics:

$$
X_{VER}=0.7\,Y_{(EUA,\text{LOG},h)}+0.3\,Y_{(CER,\text{LOG},h)},
\qquad
VER_s=T^{-1}_{\kappa_{VER}}(VER_0,X_{VER}).
$$

| Part | Authoring syntax | Canonical semantics |
|---|---|---|
| Names `VER_EUR`, `EUA_EUR`, `CER_EUR` | Yes | Replaced by resolved `rf_id`s |
| Term order in the file | Yes | Replaced by the sort on (driver `rf_id`, chart id) |
| Two entries for one $(d,\tau)$ | Yes | Merged (D088 §6) |
| `driver_chart` | Yes | Kept as chart id |
| `loading` | Yes | Kept as binary64, after merging |
| Container shorthand | Yes | Expanded to node terms; not present after compilation |

---

## 14. Resolution and as-of semantics

### 14.1 Pipeline

**DECISION (D088 §12.2).**

```text
authoring
    |
    v
as-of name resolution          build_risk_factor_name_mapping_for_as_of_date(static_config, as_of)
    |
    v
resolved rf_id                 relative and absolute names accepted
    |
    v
EXACT coordinate matching      container terms only
    |
    v
semantic terms
```

### 14.2 Facts

**FACT (`compile.py`; `src/riskquant/config/static_config/rf_resolution.py`).**

* Node and driver names resolve through `RiskFactorAbsToRelNameIdMap.get_rf_id`, which accepts relative names (for example `WTI_CRUDE_FWD[M+0]`) and absolute names (for example `WTI_CRUDE_FWD_202610` at 2026-09-25).
* Aliases that resolve to the same `rf_id` are the same factor-node. The test `test_aliases_resolving_to_one_node_are_identical` covers this.
* EXACT compares resolved coordinates, not tokens. The coordinate of a pillar is the absolute-name remainder after `"<container>_"` at the as-of: `YYYYMM`, `YYYY`, `YYYY_FH` or `YYYY_BH`, `YYYYMMDD`, or a bare label. For a surface point, the remainder includes the `[y_label]` suffix.
* Coordinate kind:
  * curves: (`x_axis_kind`);
  * surfaces: (`x_axis_kind`, `y_axis_kind`).
* A container term whose target and driver differ in topology or in coordinate kind is rejected with `ConfigurationError`, because no pillar can ever match (test `test_exact_rejects_unequal_coordinate_kinds`, using the legacy fixture row POWER_CH_BASELOAD_HY ← CO2_EUA_ANNUAL_FWD).
* Two driver pillars with the same resolved coordinate are rejected.
* The as-of is a compiler input. An EXACT pairing can change with the as-of (Section 21, Example 10).

### 14.3 Rationale

**RATIONALE.** Tokens such as `[M+3]` are relative to a curve's own anchor and roll rules. Equal tokens on two curves can denote different delivery periods, and D088 §8 records this. The economic coordinate exists only after resolution at a date, so the as-of must be an input to compilation.

### 14.4 Coverage note

**FACT.** Surface EXACT expansion is implemented in `compile.py` (`_pillar_coordinates` handles surface points). No Slice 1 test exercises a successful surface expansion.

---

## 15. Totality

**DECISION (D088 §2; §12.1).** Every V1 axis node, meaning every run-axis node with $\kappa\in\{\text{LOG},\text{SIMPLE},\text{DIFF}\}$, produces exactly one compiled row, in run-axis order. A V1 node with no authored terms receives the identity row:

$$
J_A=\{(A,\kappa_A)\},\qquad \beta=1\qquad\Longrightarrow\qquad X_A=Y_{(A,\kappa_A,h)} .
$$

**FACT (`compile.py`).** A node counts as authored when any node term names it as target, or when a container term covers it (the node is a pillar of the target container and is on the V1 axis). This is decided before zero removal.

| Case | Result |
|---|---|
| No authored terms | Identity row |
| Authored terms that survive canonicalisation | The canonical row |
| Authored terms that vanish (all zero, or exact cancellation) | `ConfigurationError` ("no term after canonicalisation") |
| Covered container pillar with no EXACT match, and no other surviving term | `ConfigurationError` |

**DECISION (D088 §6 step 7; §9 "there is no fallback tier").** An authored target does not fall back to identity.

**RATIONALE.** An authored model is a statement by the author that the target's shock is not its own column. If that statement cancels to nothing, identity would silently replace it with a different model. That would be a fallback, and it would hide a configuration error.

**FACT.** Authored terms whose target is not a V1 row of this run are validated (names, charts, loading, target admissibility) but produce no row (test `test_off_axis_terms_are_validated_but_produce_no_row`).

---

## 16. Native rows

**DECISION (D088 §1; §12.4).** Axis nodes whose configured `shift_type` is `ABSOLUTE_BASE_RETURN` or `NO_SHIFT` are outside the V1 target universe. They keep their D025 representation and numerical path. They are not rows of the compiled proxy model and not inputs to `proxy_model_digest`.

**FACT.**

* `compile_proxy_model` drops such axis nodes from the rows.
* An authored term that targets such a node is rejected with "outside the V1 target universe".
* Tests: `test_native_axis_nodes_are_not_rows` and `test_native_targets_are_rejected`.

**D025 native path (FACT).**

* ABS: forward $(v_1-v_0)/\lvert v_0\rvert$; inverse $A_0+x\lvert A_0\rvert$.
* NO_SHIFT: reconstruction returns the base for every shock, and `validate_shift_domain` skips the column. The historical-family kernels write a zero column; the parametric generator does not zero it. The content of the column is irrelevant.

**DECISION (D088 §12.4; refined for NO_SHIFT by D088 §13.4 below).** "A later numerical kernel may lower them to its seam as one-term, unit-loading rows; that is a numerical representation only and does not make them part of the authored model, the canonical proxy identity or the proxy digest."

**DECISION (D088 §13.4).**

* A NO_SHIFT row needs no world column. At the numerical seam it is a zero-term row, or an equivalent explicit no-shock path, and its output is $A_0$. A fake NO_SHIFT column must not be requested.
* An ABS row keeps its native world column $(A,\text{ABS},h)$ with unit loading.

**Not yet specified.** The request form by which the run asks the world for a native ABS column belongs to Slice 3.

---

## 17. Canonicalisation

**DECISION (D088 §6).** In order:

| Step | Rule | Implementation (FACT) |
|---|---|---|
| 1 | Resolve every target and driver to node identity. | `compile.py`: name resolution and EXACT expansion before any grouping |
| 2 | Remove authored coefficients that compare equal to zero (`== 0.0`; this includes `-0.0`). | `canonical.py`: `keep = term_loadings != 0.0` |
| 3 | Group terms by identical $(d,\tau)$ within a row. | `canonical.py`: sort keys (row, driver, chart) |
| 4 | Order the coefficients of a group by their IEEE-754 binary64 bit pattern. | `np.lexsort((bits, charts, drivers, rows))` with `bits = loadings.view(np.uint64)`, which is the unsigned bit-string order |
| 5 | Sum in that order with plain float64 left-to-right addition. | An explicit Python loop. NumPy reductions are not used, because they may use pairwise summation. |
| 6 | Remove merged coefficients that compare equal to zero. | `if acc == 0.0: continue` |
| 7 | Reject a row with no remaining term. | `CanonicalTerms.empty_rows`, then `ConfigurationError` in `compile.py` |
| 8 | Sort terms by (`rf_id`, chart id). Order rows by the run axis. | Sort order from step 3. Rows follow the V1 subset of the axis. |
| 9 | Digest the canonical representation. | `hash_proxy_model` |

**FACT (additional implemented rule).** A merged coefficient that overflows to $\pm\infty$ is rejected (`CanonicalTerms.non_finite_rows`). D088 §2 requires finite loadings.

**DECISION (D088 §6).** "No epsilon tolerance, no Decimal, no compensated summation." Canonicalisation "is not symbolic algebraic normalisation".

**RATIONALE for prohibiting approximate equality.** A tolerance creates classes of loadings that compare equal but give different numbers in execution. Identity would then fail to separate two models that produce different results. Tolerance comparison is also not transitive.

**Verified binary64 examples (EXECUTED; asserted in Slice 1 tests):**

| Authored group on one $(d,\tau)$ | Canonical loading | Note |
|---|---|---|
| $\{0.1,0.2\}$ | `0x1.3333333333334p-2` ($0.30000000000000004$) | Not equal to authored $0.3$, which is `0x1.3333333333333p-2` |
| $\{0.1,0.2,0.3\}$, any order | `0x1.3333333333334p-1` ($0.6000000000000001$) | Bit order is numeric order for positive values |
| $\{0.1,0.2,-0.3\}$, any order | $2^{-54}=5.551115123125783\times10^{-17}$ | Inexact cancellation survives. Unsigned bit order puts positive values first: $(0.1+0.2)+(-0.3)$. $2^{-54}$ is the ordered binary64 result; the rational sum of the three doubles is $2^{-55}$. |
| $\{0.3,-0.3\}$ | term removed | Exact cancellation |
| $\{0.0,-0.0\}$ | term removed | Authored zeros |
| $\{0.3,0.2\}$ | $0.5$ (`0x1.0000000000000p-1`) | Exact: the rational sum of the two doubles is representable. |
| $\{0.5,0.45\}$ | `0x1.e666666666666p-1`, the binary64 value of the literal 0.95 | Not exact. The rational sum of the two doubles lies half an ULP from two neighbours; round-half-to-even selects the value of 0.95. |

**FACT (EXECUTED mutation run).** Removing only $+0.0$ at step 2 while keeping $-0.0$ does not change any output: an authored $\pm0.0$ is an additive identity, and step 6 removes a group that merges to zero. Step 2 therefore has no separate observable effect. The tests cover it through the combined behaviour "authored zeros disappear".

---

## 18. Floating-point identity

**DECISION (D088 §2, §6, §7).** Loadings are binary64. Identity is over the exact binary64 content of canonical loadings.

| Topic | Rule |
|---|---|
| Representation | Canonical loadings are float64 and are hashed as exact bytes. |
| Exact cancellation | A group that merges to $\pm0.0$ is removed. |
| Inexact cancellation | A group that merges to a non-zero residue is kept with that residue. |
| Signed zero | Authored $\pm0.0$ is removed at step 2. No canonical loading is $\pm0.0$: construction rejects zero loadings. |
| Summation order | Fixed by bit pattern. It is independent of authoring order. |
| One-ULP rule | A one-ULP change of an authored coefficient affects identity when, and only when, it survives into different canonical binary64 content. |

**FACT (tests in `test_proxy_model_identity.py`).**

* A sole loading $0.95$ moved by one ULP changes the digest.
* $1.0+10^{-17}$ and $1.0+\text{nextafter}(10^{-17})$ both merge to $1.0$, so the digest is unchanged.
* An exactly cancelling pair perturbed by one ULP still cancels, so the digest is unchanged.
* `test_digest_equality_is_exactly_canonical_binary_equality` checks, over 15 authored variants in every pairing, that digest equality holds exactly when the canonical arrays are bit-identical.

**RATIONALE.** The run executes with the binary64 values. Two models whose canonical loadings differ by one ULP can produce different numbers, so they must have different identities. Two authored forms that produce identical canonical bits cannot produce different numbers, so they must share an identity.

---

## 19. Identity domains and digests

### 19.1 Core rule

**DECISION (D088 §13.1).** A digest identifies only the object whose canonical content it hashes. It must not be treated as the identity of a larger executable object.

$$
\text{proxy\_model\_digest}\neq ID_{\text{run}} .
$$

$ID_{\text{run}}=$ `RunManifest.identity()` is the identity that must change whenever an input capable of changing the result changes.

### 19.2 Identity domains

| Domain | Meaning | Current carrier (FACT) |
|---|---|---|
| Proxy model identity | The canonical run-bound operator | `hash_proxy_model(model)`. Not yet in the manifest. |
| Target-definition identity | $(A,\kappa_A)$ and the rest of the static target configuration and axis | `RunContextIdentity.static_config_digest`, `RunContextIdentity.universe_map_digest`; the as-of is `RunContextIdentity.as_of_ord32` |
| World identity | Scenario law, seed, count, horizon and data | `RunManifest.scenario_config`, `DataIdentity` |
| Run identity | The complete run | `RunManifest.identity()` (SHA-256 of the canonical manifest bytes) |
| Result identity | Not defined as a separate object. Results are identified through the run identity. | None |
| Identity row | Not an identity at all. It is the row $(A,\kappa_A,1)$ produced by totality. | Compiled model |

### 19.3 Definitive identity table

| Identity | Identifies | Includes $\kappa_A$? | Includes $h$? | Includes world data? |
|---|---|---|---|---|
| `proxy_model_digest` | Canonical proxy operator | Only through identity rows, where $\tau=\kappa_A$. Not for authored rows. | No | No |
| `static_config_digest` | Compiled static configuration, including `rf_id_to_shift_type_id` for every factor | Yes, for every factor | No | No |
| `universe_map_digest` | Ordered valuation axis, its shift types and the level layout | Yes, for the valuation prefix | No | No |
| `scenario_config` (manifest) | World law, seed, count; horizon for matrix-backed worlds | No | Yes for matrix-backed worlds; no for parametric today | No (law only) |
| `DataIdentity` | World data | No | No | Yes |
| `ShockIdentity` (today) | Legacy mapping: shift types, sources, scales | Yes, per row | No | No |
| `ShockIdentity` (D088, implemented: manifest v5) | Proxy operator only: `proxy_model_digest` | As `proxy_model_digest` | No | No |
| `RunContextIdentity.as_of_ord32` | Run as-of | No | No | No |
| `PnlIdentity.base_levels_digest` | Run-bound base levels $A_0$ | No | No | No |
| `RunManifest.identity()` | Complete run | Through the constituent identities | Through the world identity | Through the world identity |

No cache or certificate in `src/` is keyed on any of these digests (FACT, search of the current snapshot).

### 19.4 `hash_proxy_model`

**FACT (`src/riskquant/solutions/run_identity/hashing.py`).** `hash_proxy_model(model)` returns `digest_named_arrays` over the five canonical arrays, in this order: `row_target_rf_ids`, `row_term_offsets`, `term_driver_rf_ids`, `term_chart_ids`, `term_loadings`.

* `digest_named_arrays` folds, for each array, the name, `=`, then `digest_array(array)`, then `;`.
* `digest_array` folds the dtype string, `|`, the shape, `|`, and the C-ordered native-endian bytes.
* The hash is SHA-256.
* In the current snapshot the function is defined at the end of `hashing.py` and is exported in `__all__`.
* The test `test_digest_encoding_is_pinned` re-derives these bytes independently.

**DECISION (D088 §12.1).** The digest is a function result. It is not a stored field of the model.

Two proxy models share a digest if and only if their canonical arrays are bit-identical.

**Same proxy identity (FACT, tested):**

* permutation of authored terms and rows;
* order of coefficients inside a duplicate group;
* aliases resolving to the same `rf_id`;
* container shorthand versus the equivalent node terms at the same as-of;
* omitted identity versus an authored identity term $(A,\kappa_A,1.0)$;
* splits whose ordered binary64 sum equals the authored value: the exact split $0.3+0.2$ versus $0.5$, and the rounded split $0.5+0.45$ versus $0.95$;
* the target's configured chart $\kappa_A$ for an authored row;
* the world horizon.

**Different proxy identity (FACT, tested):**

* a different canonical loading, including one ULP;
* a different driver;
* a different driver chart;
* a different target;
* a different row order (run-axis order);
* an additional V1 axis node, which adds an identity row;
* a different EXACT result;
* a different as-of that changes EXACT resolution;
* $0.1+0.2$ versus $0.3$, which are different binary64 values.

### 19.5 Authored $\kappa$ versus identity-row $\kappa$

* For an authored row $A\leftarrow\beta\,\tau(B)$, $\kappa_A$ comes from the target definition. It is not part of the canonical operator, and it enters the run identity only through the target-definition identity.
* For an identity row the canonical relationship itself is $A\leftarrow1\times\kappa_A(A)$: the row's driver chart is $\kappa_A$. A change of $\kappa_A$ therefore changes `proxy_model_digest` for identity rows only.

This split is intended. $\kappa_A$ must not be added to every proxy row.

### 19.6 `ShockIdentity` and the A1 requirement

**DECISION (D088 §7, §13.1, §13.3; D040 status note; not implemented).** The manifest section named `ShockIdentity` will carry only `proxy_model_digest`. Despite its historical name, it will identify the proxy operator, not the world shocks. The schema field is not renamed.

* `required_shift_type_ids` becomes derived operational data identified through the target-definition identity.
* `source_factor_ids` and `shock_scale` are replaced by the proxy digest.

**FACT.** The current `ShockIdentity` still has the three legacy fields. `PnlIdentity.composition: CompositionIdentity | None = None`, and its docstring says `None` occurs "only for a manifest built outside the production entry point".

**DECISION (D088 §13.3, Slice 4 requirement).** A manifest whose `ShockIdentity` carries `proxy_model_digest` must carry `RunContextIdentity`. A manifest without it must be refused. Without this rule, two composition-less V1 manifests that differ only in $\kappa_A$ of an authored row would have the same identity.

The schema change is implemented (manifest schema v5, 2026-09-29): `proxy_model_digest` is a lowercase SHA-256 hex digest, and a v5 manifest without `RunContextIdentity` is refused.

### 19.7 Relationship between proxy identity and executable run identity

```
                         PROXY MODEL
                              │
                              │ is identified by
                              ▼
                    proxy_model_digest
                              │
                              │ is a hash of
                              ▼
              canonical proxy operator
                 (target, driver, τ, β)
                              │
                              │ instantiated by
                              ▼
                         RUN / WORLD
                    ┌─────────┴─────────┐
                    │                   │
                    h (world)          κ_A (target definition)
                    │                   │
                    ▼                   ▼
              (d, τ, h)            reconstruction
                    │                   │
                    ▼                   │
              world column              │
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     executable result

                 ID_run identifies the whole
                 executable run, not the
                 proxy operator alone. It also
                 covers A_0, the as-of, the world
                 data and the target definitions
                 (Section 19.3).
```

## 20. Canonical representation

**FACT (`src/riskquant/config/proxy_model/compiled.py`).**

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `row_target_rf_ids` | $(N,)$ | int32 | Target `rf_id` of row $n$, in run-axis order |
| `row_term_offsets` | $(N+1,)$ | int32 | Row $n$ owns terms $[o_n,o_{n+1})$; $o_0=0$, $o_N=T$ |
| `term_driver_rf_ids` | $(T,)$ | int32 | Driver `rf_id` $d_t$ |
| `term_chart_ids` | $(T,)$ | int8 | Driver chart $\tau_t\in\{1,2,3\}$ |
| `term_loadings` | $(T,)$ | float64 | Canonical loading $\beta_t$ |

In operator form:

$$
X_n=\sum_{t=o_n}^{o_{n+1}-1}\beta_t\,Y_{(d_t,\tau_t,h)} .
$$

**Construction invariants (FACT).** `CompiledProxyModel.__post_init__` enforces all of the following. Every array is copied into an owned, read-only, C-contiguous array. The dataclass is `eq=False`.

* exact dtype and rank;
* offsets start at 0, end at $T$, and strictly increase, so no row is empty;
* row targets are unique and non-negative;
* charts lie in $\{1,2,3\}$;
* loadings are finite and non-zero;
* $(d_t,\tau_t)$ strictly increases within a row.

**Contract status.**

* D088 §7 names these five arrays, their dtypes and their order as the input of `ID_proxy`. The identity encoding is therefore fixed by D088. Changing a name, a dtype or the order changes every digest and requires a decision.
* The Python class `CompiledProxyModel`, its module location and its properties (`num_rows`, `num_terms`) are the current implementation representation. D088 does not freeze them as public API.

---

## 21. Worked examples

All expected values are **EXECUTED** binary64 arithmetic, or are asserted in the cited tests. $h$ denotes the world horizon. The level paths are illustrative inputs.

The equalities shown are real-number identities printed to the digits of the binary64 result. Decimal equality does not imply bitwise equality. A test that compares a future kernel with these values uses the tolerance policy of Section 26.6, unless it explicitly asserts bitwise equality. For example, $73.6\,e^{\ln(82/80)+\ln(0.93/0.92)}$ evaluates to 76.25999999999999, while $82\times0.93$ evaluates to 76.26 (D088 §11 FX identity).

### Example 1: single-term LOG proxy (D088 §11)

| Item | Content |
|---|---|
| Authoring | `NodeProxyTerm(WTI_CRUDE_FWD[M+0] <- BRENT_CRUDE_FWD[M+0], LOG, 0.95)` |
| Resolved | row WTI: $\{(\text{BRENT},1,0.95)\}$ |
| Equation | $X=0.95\,\ln(82/80)=0.02345798196$ |
| World columns | $(\text{BRENT},\text{LOG},h)$ |
| Reconstruction | $\kappa_{WTI}=\text{LOG}$: $A=70\,e^{X}$ |
| Expected | $71.66146991455523$ |
| Why | This is the legacy D013 case written as V1. It is the special case $J_A=\{(\text{BRENT},\text{LOG})\}$, $\beta=0.95$ (D013 status note). |

Canonical arrays for axis [WTI, BRENT] (test `test_single_explicit_term_produces_exact_canonical_arrays`):

* rows [WTI, BRENT];
* offsets [0, 1, 2];
* drivers [BRENT, BRENT];
* charts [1, 1];
* loadings [0.95, 1.0].

The second row is BRENT's identity row.

### Example 2: two-term proxy (D088 §11)

| Item | Content |
|---|---|
| Authoring | WTI ← 0.7 LOG(BRENT) + 0.3 LOG(HENRY_HUB) |
| Resolved | $\{(\text{BRENT},1,0.7),(\text{HENRY},1,0.3)\}$, sorted by driver `rf_id` |
| Equation | $X=0.7\ln(1.025)+0.3\ln(0.98)=0.01122401662$ |
| World columns | $(\text{BRENT},\text{LOG},h)$, $(\text{HENRY},\text{LOG},h)$ |
| Reconstruction | $70\,e^{X}$ |
| Expected | $70.79010695536556$ |
| Why | The legacy mechanism has no representation for this. Test: `test_multiple_terms_are_sorted_by_driver_then_chart`. |

### Example 3: same driver, two charts

| Item | Content |
|---|---|
| Authoring | WTI ← 0.5 LOG(BRENT) + 0.02 DIFF(BRENT) |
| Resolved | $\{(\text{BRENT},1,0.5),(\text{BRENT},3,0.02)\}$ |
| Equation | For BRENT $80\to82$: $X=0.5\ln(1.025)+0.02\times2=0.05234631$ |
| World columns | $(\text{BRENT},\text{LOG},h)$ and $(\text{BRENT},\text{DIFF},h)$: two distinct columns |
| Reconstruction | $70\,e^{X}$ |
| Expected | $73.76184174352942$ |
| Why | $(d,\tau)$ is the term key. Two charts on one driver are two random variables and two columns (D088 §4, §6 step 3). Test: `test_one_driver_under_two_charts_is_two_terms`. |

### Example 4: target chart differs from driver chart (D088 §11)

| Item | Content |
|---|---|
| Authoring | WTI (LOG) ← 0.04 DIFF(spread) |
| Resolved | $\{(\text{SPREAD},3,0.04)\}$; $\kappa_{WTI}=1$ |
| Equation | Spread $2.00\to2.50$: $X=0.04\times0.50=0.02$ |
| World columns | $(\text{SPREAD},\text{DIFF},h)$ |
| Reconstruction | $70\,e^{0.02}$ |
| Expected | $71.41409380187291$ |
| Incorrect (target-chart routing, same loading) | $70\,e^{0.04\ln1.25}=70.62759866892789$ |
| Why | $\tau$ and $\kappa$ are independent. See Section 8.4 for the value $86.5292$, which uses the loading 0.95. Test: `test_driver_chart_is_independent_of_target_chart`, using the test-owned curve `TEST_BRENT_WTI_SPREAD_FWD`. |

### Example 5: signed loading

| Item | Content |
|---|---|
| Authoring | WTI ← 0.7 LOG(BRENT) − 0.3 LOG(HENRY_HUB) |
| Equation | $X=0.7\ln(1.025)-0.3\ln(0.98)=0.02334564101$ |
| Reconstruction | $70\,e^{X}$ |
| Expected | $71.65341984896365$ |
| Exact identity case (D088 §11) | $\text{EURCHF}=\text{USDCHF}/\text{USDEUR}$, loadings $(+1,-1)$, $A_0=0.90/0.92$: $A_s=0.978261\,e^{0.029853}=1.00791=0.918/0.9108$ |
| Why | Negative loadings are needed for exact identities and for inverse relationships. The legacy $(0,10]$ bound excluded them. The test `test_authoring_permutation_is_invariant` includes the $(+1,-1)$ EURCHF terms. |

### Example 6: omitted identity

| Item | Content |
|---|---|
| Authoring | Nothing for `USD_LIBOR_3M[10Y]` (configured DIFF). The node is on the axis. |
| Resolved | Identity row $\{(\text{LIBOR10Y},3,1.0)\}$ |
| Equation | $X=Y_{(\text{LIBOR10Y},\text{DIFF},h)}$ |
| Reconstruction | $A_0+X$ |
| Why | Totality. An authored `NodeProxyTerm(LIBOR10Y <- LIBOR10Y, DIFF, 1.0)` compiles to the same arrays. Authored under LOG, it is a different operator. Test: `test_identity_closure_equals_explicit_identity`. |

### Example 7: authored terms that become empty

| Item | Content |
|---|---|
| Authoring | WTI ← 0.3 LOG(BRENT) + (−0.3) LOG(BRENT) |
| Canonicalisation | The group merges to $0.0$ and is removed. The row is empty. |
| Result | `ConfigurationError`: "no term after canonicalisation" |
| Why | An authored target never falls back to identity (Section 15). Test: `test_authored_target_never_falls_back_to_identity`. |

### Example 8: alias equivalence

| Item | Content |
|---|---|
| Authoring A | `WTI_CRUDE_FWD[M+0] <- BRENT_CRUDE_FWD[M+1]` |
| Authoring B | `WTI_CRUDE_FWD_202610 <- BRENT_CRUDE_FWD_202611` |
| As-of | 2026-09-25 |
| Result | Identical canonical arrays and digest |
| Why | Names resolve to `rf_id` before canonicalisation. Test: `test_aliases_resolving_to_one_node_are_identical`. |

### Example 9: EXACT by resolved coordinate

| Item | Content |
|---|---|
| Authoring | `ContainerProxyTerm(TEST_ANCHOR2_FWD <- BRENT_CRUDE_FWD, LOG, 0.95)` (test-owned curve, `anchor_offset = 2`; BRENT `anchor_offset = 1`) |
| Resolution at 2026-09-25 | `TEST_ANCHOR2[M+0]` → `202611` ↔ `BRENT[M+1]`; `[M+1]` → `202612` ↔ `BRENT[M+2]`; `[M+2]` → `202701` ↔ `BRENT[M+3]` |
| Unmatched | `TEST_ANCHOR2[M+3]` → `202702`: BRENT has no such pillar. On the axis, this gives `ConfigurationError`. |
| Why | Equal tokens are not equal coordinates. Tests: `test_exact_pairs_by_resolved_coordinate_not_token`, `test_container_shorthand_equals_explicit_expansion`. |

### Example 10: different as-of

| Item | Content |
|---|---|
| Authoring | `ContainerProxyTerm(TEST_ANNUAL_CAL_FWD <- CO2_EUA_ANNUAL_FWD, LOG, 1.2)`. TEST is calendar-period with anchor 0. CO2 is `ROLL_ON_MMDD` with roll date 04-01. |
| 2026-02-15 | TEST `[Y+k]` → year $2026+k$ pairs with `CO2[Y+k+1]` |
| 2026-09-25 | TEST `[Y+k]` pairs with `CO2[Y+k]` |
| Result | Different driver `rf_id`s and different digests. A node-term pairing gives the same digest on both dates. |
| Why | As-of resolution is upstream of identity (D088 §12.2). Tests: `test_exact_resolution_follows_the_as_of`, `test_as_of_is_upstream_of_identity`. |

### Example 11: native ABS and NO_SHIFT

| Item | Content |
|---|---|
| Setup | Test-owned configuration: `AAPL_EQ` configured ABS, `AAPL_ATM_VOL` configured NO_SHIFT. Axis [AAPL_EQ, USDEUR, AAPL_ATM_VOL]. |
| Result | One row (USDEUR identity). The digest equals the digest for axis [USDEUR]. |
| Authored target AAPL_EQ | `ConfigurationError` ("outside the V1 target universe") |
| As a driver | `USDEUR <- 0.5 LOG(AAPL_ATM_VOL)` compiles. $\tau$ does not follow the driver's configured shift type. |
| Later numerical seam | The ABS row may be lowered to a one-term, unit-loading row on its native column $(A,\text{ABS},h)$. The NO_SHIFT row is a zero-term row with output $A_0$ (D088 §12.4, §13.4). Numerical seam implemented (Slice 2); native-row lowering is not implemented. |
| Tests | `test_native_axis_nodes_are_not_rows`, `test_native_targets_are_rejected`, `test_native_nodes_may_drive_under_a_v1_chart` |

### Example 12: missing world column

| Item | Content |
|---|---|
| Model | WTI ← 0.95 LOG(C). The world has no column $(C,\text{LOG},h)$. |
| Required behaviour | Rejection before numerics. D088 §11: "missing column rejected before numerics". D088 §4: "no silent missing values". D088 §9: no fallback tier. |
| Status | The Slice 2 function `bind_term_columns` rejects a missing $(d,\tau,h)$ (`ValueError`). It is not wired into the production run, and worlds do not yet supply $(d,\tau,h)$ columns (Slice 3). The legacy analogue is `_require_shock_sources` (`CompositionContractError`). |
| Why | Replacing a missing column with another column would reintroduce source selection. |

---

## 22. Key invariants

Each invariant is **DECISION** unless marked otherwise. The final column gives the current enforcement.

| # | Invariant | Source | Enforced today |
|---|---|---|---|
| 1 | No source selection. The model determines its driver columns. | D088 §9 | Compiler: yes. No availability input exists (signature test). Run: not integrated. |
| 2 | No fallback source. | D088 §9 ("there is no fallback tier") | Compiler: an authored row never becomes identity (tested). Run: not integrated. |
| 3 | No horizon inference. | D088 §12.5 | Compiler: no horizon field or input (tested). Binding: requested $h$ must equal the world $h$ (Slice 2, tested). |
| 4 | No target-chart-as-driver-chart substitution. | D088 §2, §3 | Compiler: $\tau$ is taken from the term (tested; mutation-killed) |
| 5 | Signed loadings are supported. | D088 §2 | Compiler (tested) |
| 6 | Multiple driver terms are supported. | D088 §2 | Compiler (tested) |
| 7 | The same driver under two charts gives two distinct terms and two distinct columns. | D088 §4, §6 step 3 | Compiler (tested). Binding and numerical seam (Slice 2, tested). |
| 8 | Aggregation occurs in shock space. | D088 §2, §8 | Numerical seam (Slice 2, tested); not wired |
| 9 | Reconstruction happens after aggregation. | D088 §2, §4 | Numerical seam (Slice 2, tested); not wired |
| 10 | Target reconstruction is target-owned. | D088 §3 | Compiler stores no $\kappa$ (tested) |
| 11 | The world owns the horizon. | D088 §3, §12.5 | Binding compares the requested and the world $h$ (Slice 2). Worlds declaring $h$ for $(d,\tau,h)$ columns: Slice 3. |
| 12 | $(d,\tau,h)$ identifies a world column. | D088 §4 | Binding keys (Slice 2, tested). World-side columns: Slice 3. |
| 13 | Canonicalisation is deterministic. | D088 §6 | Tested (permutations; mutation-killed) |
| 14 | The digest is over canonical semantics. | D088 §7, §12.1 | Tested |
| 15 | Native ABS and NO_SHIFT rows remain outside the proxy model. | D088 §1, §12.4 | Tested |
| 16 | Missing world columns fail fast. | D088 §4, §11 | Binding (Slice 2, tested); not wired |
| 17 | Proxy compilation is independent of scenario count, batch size, valuation and P&L. | D088 §12.1, §12.3 | Tested (signature; import-graph subprocess test) |
| 18 | As-of resolution precedes canonicalisation. | D088 §6 step 1, §12.2 | Tested (alias and as-of tests) |
| 19 | EXACT uses coordinates, not tokens. | D088 §8 | Tested (mutation-killed) |
| 20 | Exact zero is the only zero-elimination criterion. | D088 §6 | Tested (inexact residue kept; mutation-killed) |
| 21 | `proxy_model_digest` identifies the operator only; it is not $ID_{\text{run}}$. | D088 §13.1 | Identity table (Section 19.3); manifest not yet migrated |
| 22 | A V1 manifest carrying the proxy digest carries `RunContextIdentity`. | D088 §13.3 | Implemented (manifest v5, 2026-09-29) |
| 23 | The term sum follows ascending canonical order in float64 without reassociation. | D088 §13.5 | Numerical seam (Slice 2, tested) |
| 24 | A NO_SHIFT row outputs $A_0$ and requires no world column. | D088 §13.4 | Numerical seam and binding (Slice 2, tested) |

---

## 23. Ownership model

| Concern | Owner | Evidence |
|---|---|---|
| Authoring (semantic content) | Proxy configuration layer: `NodeProxyTerm`, `ContainerProxyTerm` | `schema_objects.py`; D088 §12.3 |
| Authoring file format | Edge. **OPEN.** | D088 §12.3 |
| Alias resolution | Static as-of resolution: `build_risk_factor_name_mapping_for_as_of_date`, called by the compiler | `rf_resolution.py`; `compile.py` |
| EXACT coordinate matching | Proxy compiler | `compile.py` (`_expand_container_term`) |
| Admissibility | Proxy compiler | `compile.py` |
| Totality | Proxy compiler | `compile.py` |
| Canonicalisation | Proxy compiler | `canonical.py` |
| Proxy identity | Canonical model plus `hash_proxy_model` (solution tier) | `hashing.py`; D088 §7, §12.1 |
| Driver $(d,\tau)$ and loading $\beta$ | Proxy model | D088 §3 |
| Target $\kappa_A$ | Target factor definition (static configuration), carried per row by the run axis | D088 §3; `static_config.py`; `risk_run/schemas.py` |
| Base level $A_0$ | Run-bound base state (`base_levels`) | D088 §3; D081 §1 |
| Horizon $h$ and step unit | Scenario world | D088 §3 |
| Concrete $(d,\tau,h)$ column | World | D088 §4 |
| Term-to-column binding | Run and world seam | D088 §12.5. Binding function implemented (Slice 2); not wired. |
| Shock values | World generators | D088 §4 |
| Level-domain rule for level-backed columns | Generator edge | D025; `shift_domain.py` |
| Sparse shock aggregation | Numerical proxy application | D088 §2. Implemented (Slice 2); not wired. |
| Target reconstruction | Numerical application, using $\kappa$ from the run axis | D088 §2, §3. Implemented (Slice 2); not wired. |
| Factor-state admissibility of levels | Market-data producer (`build_market_data_matrix`) | D075; D070 |
| Valuation | Existing RiskQuant valuation | Unchanged |
| P&L | Existing risk-run machinery | D081 |
| Risk aggregation | Existing risk-run machinery | Unchanged |
| Reporting FX | Risk-run reporting route | D085; D088 §9 |
| Provenance (calibration method, window, rationale) | Governance; never numerical | D088 §3 |

---

## 24. RiskQuant implementation architecture

### 24.1 Implemented (Slice 1)

| Path | Owns |
|---|---|
| `src/riskquant/config/proxy_model/__init__.py` | Empty package marker |
| `src/riskquant/config/proxy_model/schema_objects.py` | Authored record types `NodeProxyTerm` and `ContainerProxyTerm`. No validation; the compiler owns validation. |
| `src/riskquant/config/proxy_model/compile.py` | `compile_proxy_model(*, authored_terms, static_config, as_of_date, run_axis_rf_ids)`: axis validation, as-of resolution, admissibility, EXACT expansion, totality, error reporting |
| `src/riskquant/config/proxy_model/canonical.py` | `canonicalise_resolved_terms`: D088 §6 steps 2 to 8 over resolved arrays. It knows no names. |
| `src/riskquant/config/proxy_model/compiled.py` | `CompiledProxyModel` with its construction invariants; `V1_TARGET_CHART_IDS`; `V1_DRIVER_CHART_IDS` |
| `src/riskquant/solutions/run_identity/hashing.py` | `hash_proxy_model(model)`. The function is in the solution tier; the model type is imported for type checking only. |

**FACT.** No module outside these files and their tests imports `riskquant.config.proxy_model` in the current snapshot. The compiler imports no module from `simlab`, `valuation`, `sensitivity_pnl`, `pnl`, `reporting`, `scenario_metrics`, `portfolio`, `proxy_resolver`, `config.proxy_config`, `solutions.risk_run` or `solutions.scenario_models`. This is checked by the subprocess test `test_compiler_imports_no_world_runtime_or_legacy_module`.

### 24.1a Implemented (Slice 2; not wired into the production run)

| Path | Owns |
|---|---|
| `src/riskquant/simlab/shocks/column_binding.py` | `bind_term_columns`: integer-key binding $(d,\tau)$ plus the world horizon to column indices; horizon equality; missing and duplicate keys rejected; NO_SHIFT never requested |
| `src/riskquant/simlab/shocks/linear_term_apply.py` | `apply_linear_terms_to_base_levels`: edge validation of the Section 26 contract, then the kernel call with zero-copy read-only views |
| `src/riskquant/simlab/shocks/linear_term_kernel.py` | `_kernel_apply_linear_terms`: term aggregation and reconstruction under `stable_math_parallel_njit` |

### 24.2 Conceptual architecture and mapping

```text
configuration (authored records)                     implemented: schema_objects.py
    |
    v
proxy compiler                                        implemented: compile.py, canonical.py
    |
    v
CompiledProxyModel  -> hash_proxy_model              implemented: compiled.py, hashing.py
    |
    v
world-column binding  (d, tau) x h -> c              IMPLEMENTED (Slice 2): column_binding.py; not wired
    |
    v
numerical sparse operator + reconstruction           IMPLEMENTED (Slice 2): linear_term_apply.py, linear_term_kernel.py; not wired
    |                                                 (legacy: simlab/shocks/apply.py, shock_kernel.py)
    v
existing level path: build_market_data_matrix_from_flat_levels (market_data/assemble.py)
                     -> build_market_data_matrix (market_data/builder.py, D075 guard)
    |
    v
risk run: run_scenario_var_es (solutions/risk_run/api.py)
          FULL_REVAL / Taylor -> reporting FX (reporting/fx_conversion.py) -> VaR/ES
```

**Not yet specified.** The module names of the binding and of the V1 numerical application. This document does not name them.

---

## 25. End-to-end data flow

The factor names and column numbers are illustrative.

```text
authoring
    VER_EUR <- 0.7 x LOG(EUA_EUR) + 0.3 x LOG(CER_EUR)
        |
        v   compile_proxy_model(authored_terms, static_config, as_of_date, run_axis_rf_ids)
resolved rf_ids
    VER_EUR -> rf a,  EUA_EUR -> rf e,  CER_EUR -> rf c          (as-of name map)
        |
        v
canonical terms (sorted by driver rf_id, then chart id; assume e < c)
    row a: (e, LOG, 0.7), (c, LOG, 0.3)
        |
        v
CompiledProxyModel
    row_target_rf_ids = [..., a, ...]         (run-axis order; other V1 nodes get identity rows)
    term_driver_rf_ids / term_chart_ids / term_loadings for row a = [e, c] / [1, 1] / [0.7, 0.3]
        |                                     proxy_model_digest = hash_proxy_model(model)
        v
world h                                       (declared by the scenario world)
    (EUA_EUR, LOG, h) -> column 17            (binding: Slice 2 function; worlds supply (d, tau, h) in Slice 3)
    (CER_EUR, LOG, h) -> column 28
        |
        v
aggregation per scenario b                    (numerical seam: Slice 2; not wired)
    X_VER[b] = 0.7 * Y[b, 17] + 0.3 * Y[b, 28]
        |
        v
reconstruction with kappa_VER from the run axis
    kappa_VER = LOG:  VER[b] = VER_0 * exp(X_VER[b])
        |
        v
VER scenario levels
```

**FACT (existing pipeline after levels).** In `run_scenario_var_es`, scenario levels feed two paths:

* FULL_REVAL: through `build_market_data_matrix_from_flat_levels`, where the D075 level guard applies, then valuation;
* Taylor: through the Taylor axis levels.

P&L is formed per D081. Reporting FX conversion is applied per D085. VaR and ES are computed on the resulting P&L. Proxy V1 changes none of these steps.

---

## 26. Numerical application

The V1 numerical seam is implemented by Slice 2 (`linear_term_apply.py`, `linear_term_kernel.py`) and is not wired into the production run. Its contract is closed (**DECISION**, D088 §13.4 to §13.6).

### 26.1 Aggregation

For row $n$ and scenario position $b$:

$$
X_n^{(b)}=\sum_{t=o_n}^{o_{n+1}-1}\beta_t\,Y_{b,c_t},
\qquad
A_n^{(b)}=T^{-1}_{\kappa_n}\!\left(A_{0,n},\,X_n^{(b)}\right).
$$

The sum is evaluated in ascending canonical term index $t$. Reconstruction applies after the complete sum.

### 26.2 Precision

| Quantity | dtype |
|---|---|
| Shock storage $Y$ | float32, `(B, K)`, F-contiguous (retained part of D026 F7) |
| Shock value used in the product | Converted to float64 before multiplication |
| $\beta$ | float64 (D088 §2) |
| Product, accumulation | float64 |
| Output | float64 |

### 26.3 Reassociation and parallelism

* The reduction kernel uses `riskquant.jit.stable_math_parallel_njit`. This decorator omits `"reassoc"`, so the compiler cannot reorder the specified summation. `parallel_njit` and `njit` enable `"reassoc"` and must not be used for this reduction (FACT: `src/riskquant/jit.py`).
* Parallelism is over independent rows only, never over the term reduction of one row.
* No `fsum`, pairwise or compensated summation is used.
* `"contract"` (FMA) remains enabled in every canonical decorator. The canonical term order is still the mathematical evaluation order of the contract.

### 26.4 Native rows at the seam

* $\kappa_n=\text{NO\_SHIFT}$: $A_n^{(b)}=A_{0,n}$ for every $b$. The row is a zero-term row, or an equivalent explicit no-shock path. A zero-term row is admissible only for NO_SHIFT.
* $\kappa_n=\text{ABS}$: one term with loading 1.0 on the native column $(A,\text{ABS},h)$; $A_n^{(b)}=A_{0,n}+X\lvert A_{0,n}\rvert$.
* Neither row is part of the compiled proxy model or of `proxy_model_digest` (D088 §12.4).

### 26.5 Inputs and edge validation

| Input | Owner |
|---|---|
| Base levels $A_0$, float64 `(N,)` | Run-bound base state (D088 §3) |
| Row offsets, term column indices $c_t$, term loadings $\beta_t$ | Compiled model plus binding (`bind_term_columns`, Slice 2); world-side column production is Slice 3 |
| Reconstruction ids $\kappa_n$ | Run axis (D088 §3) |
| Shock matrix $Y$ | World (Slice 3) |
| Output buffer `(N, B)` float64 | Allocated at the edge |

Offsets, column indices, dtypes and layouts are validated at the numerical edge, before the kernel. The kernel performs no semantic resolution. D041 batch invariance applies: the result is bitwise invariant to `batch_size` within one execution environment (Section 26.6).

**FACT (Slice 2).** The implemented output is `(N, B)` float64, C-contiguous, the same layout as the legacy apply contract (`shock_kernel.py`).

### 26.6 Oracle tolerance policy

**DECISION (D088 §13.6).**

* A test that compares the numerical kernel with a Python float64 reference uses a relative tolerance of $10^{-12}$ and an absolute tolerance of $10^{-15}$. FMA contraction makes bitwise comparison with a Python expression inappropriate.
* Determinism tests (batch size, thread count, repetition) compare kernel output with kernel output bitwise.
* Scope of the tolerance. It is the Slice 2 policy for kernel-versus-Python-reference tests only. It is not a certification tolerance; the D019 golden tolerance remains open (Open Item 5).
* Scope of bitwise determinism. It means kernel-to-kernel equality within one execution environment: one platform, one set of library and compiled-kernel versions (the versions are recorded in `EngineIdentity`). It does not claim bitwise equality across platforms, because FMA contraction and the `exp` and `log` implementations may differ between CPUs and library versions.

## 27. Fail-fast behaviour

| Condition | Where | Error | Status |
|---|---|---|---|
| Target name does not resolve, or names a container | Compiler | `ConfigurationError` | Implemented, tested |
| Target is a native node (ABS, NO_SHIFT) | Compiler | `ConfigurationError` | Implemented, tested |
| Driver name does not resolve | Compiler | `ConfigurationError` | Implemented, tested |
| Driver chart is not a `ShiftType`, or is not in {LOG, SIMPLE, DIFF} | Compiler | `ConfigurationError` | Implemented, tested |
| Loading is not a float (int, bool and `np.float32` are rejected) or is not finite | Compiler | `ConfigurationError` | Implemented, tested |
| Merged loading overflows to $\pm\infty$ | Compiler | `ConfigurationError` | Implemented, tested |
| Container name is not a curve or surface | Compiler | `ConfigurationError` | Implemented, tested |
| EXACT between unequal topology or coordinate kinds | Compiler | `ConfigurationError` | Implemented, tested |
| Two driver pillars with the same resolved coordinate | Compiler | `ConfigurationError` | Implemented, not tested |
| Covered axis pillar with no EXACT match, and no other surviving term | Compiler | `ConfigurationError` | Implemented, tested |
| Authored row empty after canonicalisation | Compiler | `ConfigurationError` | Implemented, tested |
| Run axis not a 1-D int32 array, has duplicates, is out of range, or contains unoccupied `rf_id`s | Compiler | `TypeError`, `ValueError`, `ConfigurationError` | Implemented, tested |
| `as_of_date` not a `date` (`datetime` and `str` are rejected) | Compiler | `TypeError` | Implemented, tested |
| Non-canonical compiled arrays | `CompiledProxyModel` construction | `TypeError`, `ValueError` | Implemented, tested |
| Missing world column $(d,\tau,h)$ | Run and world seam | Rejected before numerics (D088 §11) | Binding function implemented and tested (Slice 2); not wired |
| Requested $h$ differs from the world's declared $h$ | Generator | Rejected (D088 §12.5) | Binding function rejects it (Slice 2); generator-side validation is Slice 3 |
| Invalid column index; malformed numerical input | Numerical application edge | Rejected before the kernel (CLAUDE.md §4; D088 §13.5) | Implemented and tested (Slice 2) |
| Zero-term row with $\kappa\neq\text{NO\_SHIFT}$ | Numerical application edge | Rejected (D088 §13.5) | Implemented and tested (Slice 2) |
| Level-backed column outside its chart's level domain | Generator edge | Existing `validate_shift_domain` (D025) | Exists (legacy path) |
| Reconstructed level inadmissible for its factor | Market-data producer | Existing D075 guard | Exists |

**RATIONALE.** Any silent repair would change which model is valued without changing the configuration or the model identity. Examples of such repairs: a substitute column, a default chart, an identity fallback, a clipped level. D088 §4 lists "no silent missing values; no clipping; no scenario dropping". D088 §9 excludes any fallback tier.

---

## 28. Adopted V1 decisions

"Record" gives the location of the decision. Decisions 14 and 15 are implemented and tested in Slice 1 and are recorded as clarifications in D088 §13.7.

| # | Decision | Record | Why | Consequence | What it prevents |
|---|---|---|---|---|---|
| 1 | Proxy V1 is a multi-term relationship model, not a selected-source model. | D088 title, §2; D013 status note | The quantity to model is the target's movement as a function of driver movements. | $X_A=\sum\beta Y$ with $\lvert J_A\rvert\ge1$ | Single-source limits; availability-dependent meaning |
| 2 | A term is semantically $(d,\tau,\beta)$. | D088 §2, §3 | These three fields fully determine a summand, given $h$. | The compiled model stores exactly driver, chart and loading per term. | Implicit charts; hidden scaling |
| 3 | $\kappa_A$ is separate from $\tau$. | D088 §2, §3 | Measuring a driver and rebuilding a target are different operations. | $\tau$ per term; $\kappa$ per target | The target-chart routing error (Section 8.4) |
| 4 | The horizon belongs to the world and run, not to the proxy model. | D088 §3, §12.5 | The relationship does not depend on the world's temporal representation. | No $h$ in the model | Horizon inference; one relationship becoming several models |
| 5 | Concrete world-column identity is $(d,\tau,h)$. | D088 §4 | A column is one random variable. | Binding keys include $h$. | Reading a column of another horizon |
| 6 | Resolution happens before canonicalisation. | D088 §6 step 1, §12.2 | Grouping must be on `rf_id`s, not on names. | Aliases merge; EXACT feeds canonicalisation. | Duplicate terms surviving under different names |
| 7 | Canonicalisation is deterministic and bit-exact. | D088 §6 | Identity must not depend on authoring order. | Bit-pattern order; left-to-right sum | Order-dependent identities |
| 8 | Identity uses exact binary64 semantics, not tolerances. | D088 §6, §7 | Execution uses exact values. | One ULP that survives changes identity. | Two models that give different numbers but share one identity |
| 9 | The digest is computed from the canonical arrays. | D088 §7, §12.1 | The canonical operator is what executes. | `hash_proxy_model` over five arrays | Identity drifting with syntax |
| 10 | An as-of change can change proxy identity when it changes EXACT resolution. | D088 §12.2 | EXACT coordinates exist only at a date. | The as-of is a compiler input. | Treating different operators as the same |
| 11 | Native ABS and NO_SHIFT rows are excluded from the compiled model. | D088 §1, §12.4 | They have no V1 target contract; D025 governs them. | No native rows in the model or digest | Inventing ABS semantics; mixing native identity with proxy identity |
| 12 | A target with no authored terms receives identity through totality closure. | D088 §2 | Every V1 node needs exactly one model. | Identity rows are materialised. | Missing rows; a separate "direct" concept |
| 13 | An authored target does not fall back to identity when its terms disappear. | D088 §6 step 7, §9 | An authored model is an explicit statement. | `ConfigurationError` | Silent model substitution |
| 14 | EXACT between incompatible coordinate kinds is rejected. | D088 §13.7(a); Slice 1 (`compile.py`; tests for curves and surfaces). Basis: D088 §8. | Such a term can never match any pillar. | Early `ConfigurationError`. The legacy row POWER_CH_BASELOAD_HY ← CO2_EUA_ANNUAL_FWD is invalid under V1. | Vacuous rules; token-based matching |
| 15 | Container and node terms that resolve to the same pillar and $(d,\tau)$ are summed. | D088 §13.7(b); Slice 1 (`compile.py`; test). Basis: D088 §6 step 3. | Canonicalisation merges by $(d,\tau)$ after resolution; D088 defines no override. | $0.5+0.25=0.75$ on one term | Order-dependent override rules |
| 16 | The world validates availability; it does not choose a replacement. | D088 §9; D085 clause 8 status note | Availability is not model semantics. | Missing column means failure. | Fallback; tier-1 precedence |
| 17 | The numerical kernel does not know what a proxy is. | D088 §12.4 (lowering is "a numerical representation only"); CLAUDE.md §4 (kernels do not resolve identities or apply business policy) | Semantics are resolved before the kernel. | The kernel receives offsets, column indices, loadings, $\kappa$ ids and base levels only. | Semantic resolution inside kernels |
| 18 | `proxy_model_digest` identifies the proxy operator only. | D088 §13.1 | A smaller digest cannot identify a larger object. | $ID_{\text{run}}$ is the only run identity. | Using the proxy digest as a run, result or cache key |
| 19 | A V1 manifest carrying `proxy_model_digest` must carry `RunContextIdentity`. | D088 §13.3 (Slice 4 requirement) | $\kappa_A$ of authored rows is identified only there. | Composition-less V1 manifests are refused. | Two runs differing in $\kappa_A$ sharing one identity |
| 20 | Slice 2 numerical contract: ascending canonical order, float64 accumulation, `stable_math_parallel_njit`, parallel over rows. | D088 §13.5 | Canonical order must also be the executed order. | Deterministic, batch-invariant term sums | Compiler reassociation; order drift |
| 21 | NO_SHIFT rows need no world column; ABS rows keep their native column. | D088 §13.4 | $T^{-1}_{\text{NO\_SHIFT}}$ ignores the shock. | Zero-term NO_SHIFT rows | Fake columns; a false world-column invariant |
| 22 | A parametric world has one horizon, declared as requested (`ParametricGeneratorConfig`) and effective (`ParametricRiskData`); the two must match exactly before identity; the generator performs no temporal/horizon conversion. | D089 | The arrays cannot reveal their period; the run and the data are produced independently. | Mismatch fails before the manifest; n-period parameters give an n-period shock. | Mislabelled pairings; hidden √h or dependence scaling; confusing lookback with horizon |

---

## 29. Rejected alternatives

| Alternative | Why it is tempting | Why rejected | Problem it would recreate | Record |
|---|---|---|---|---|
| Selected-source proxy | Simple; one lookup per target | Cannot express relationships; its meaning depends on availability | Sections 8.1, 8.5 | D013 superseded by D088 |
| One scalar `shock_scale` | One number per target | Positive only; float32; one term; described as volatility scaling | Section 8.3 | D040 and D082 clause 11 status notes |
| Target chart determining driver chart | No driver-chart field is needed | Different operations: measurement versus reconstruction | Section 8.4 | D088 §2, §3 |
| Runtime fallback source resolution | Runs complete when data is missing | Silent change of model | Sections 8.5, 15 | D088 §9 |
| Horizon in the proxy model | The column identity contains $h$ | $h$ is world-owned; inference risk | Section 11 | D088 §12.5 |
| Approximate coefficient equality | Treats $0.1+0.2$ as $0.3$ | Not transitive; identity would not follow execution | Section 17 | D088 §6 |
| Hashing raw authoring syntax | No compilation needed before hashing | Equal operators would get different digests, and the reverse | Section 30 | D088 §12.1 |
| Storing the target chart in the proxy operator | Self-contained rows | Two owners of $\kappa$ | Section 12 | D088 §3 |
| Mixing native rows into proxy identity | One uniform row set | ABS and NO_SHIFT have no V1 contract | Section 16 | D088 §12.4 |
| World availability determining proxy semantics | Automatic behaviour | Availability-dependent meaning | Section 8.6 | D088 §9, §12.3 |
| Reconstructing each driver separately and combining levels | Level-space intuition | Level-space weighting is not shock-space weighting | Section 10 | D088 §8 |
| Building proxy semantics around `SimlabRunner` | It called the resolver (it has since been removed) | Legacy and out of V1 scope | Legacy interfaces constraining V1 | Project instruction for Slice 1 (not a `DECISIONS.md` record) |

---

## 30. Why canonicalisation matters mathematically

One operator $\mathcal B$ has many authoring forms. The following all describe the same row:

* term order changes;
* a duplicate split, such as $0.5+0.45$;
* aliases;
* container shorthand versus node terms;
* an omitted identity versus an authored identity.

Authoring syntax therefore does not identify a model.

The converse also holds. Real-number equivalence does not imply execution equivalence: $0.1+0.2$ and $0.3$ are different binary64 values. The object that executes is the canonical operator: resolved, merged in a fixed order, with zero terms removed and rows in axis order.

**RATIONALE.** Identity must satisfy both directions:

$$
\text{same canonical bits}\Rightarrow\text{same digest},
\qquad
\text{different canonical bits}\Rightarrow\text{different digest}.
$$

Only a deterministic, bit-exact canonical form hashed directly satisfies both. Tests enforce this over 15 variants in every pairing (Section 18).

---

## 31. Why the model is independent of the world

**FACT.** The compiler needs exactly four inputs:

* authored terms;
* the static configuration;
* the as-of;
* the run axis.

It does not receive:

* a scenario count, or scenario values;
* a batch size;
* horizon data;
* historical observations or parametric draws;
* valuation inputs;
* P&L inputs.

**DECISION (D088 §12.1, §12.3).** The compiled model is a function of those four inputs, and the compiler does not own world construction.

**RATIONALE.**

* **Testing.** The model can be compiled and identified with no world. The Slice 1 tests do exactly this.
* **Reproducibility.** A model does not change when data coverage changes.
* **Architecture.** Availability is checked where the world is known: at binding.

---

## 32. Relationship to historical and parametric worlds

A proxy model declares $(d,\tau)$. A world must supply $(d,\tau,h)$.

**DECISION (D088 §4).** "The proxy contract does not prescribe how any engine constructs a column; each engine's own contract shows that its columns satisfy this definition." The same compiled operator therefore applies to any world that supplies compatible columns.

**FACT (current state).**

* The historical family computes columns from levels. Its horizon comes from its scenario configuration.
* `ParametricRiskData` is keyed by factor only, with no chart and no horizon.

The D034 and D087 status notes require parametric columns keyed $(d,\tau,h)$, a declared horizon and step unit, and a Case-1 or Case-2 declaration where a world supplies several charts of one node. These are **not implemented**, and their representation is **not yet specified**.

**DECISION (D089).** A parametric world has one forecast horizon $H_W$ with two declarations: the requested horizon in `ParametricGeneratorConfig` (entering `scenario_config` identity) and the effective period of the supplied law in `ParametricRiskData` (entering `DataIdentity`). The run's composition contract checks $H_{\text{requested}}=H_{\text{effective}}$ exactly, before identity, RNG, generation and P&L; a mismatch fails and nothing is converted. The generator performs no temporal/horizon conversion, so weekly parameters give a one-week shock and five-week parameters a five-week shock. The calibration lookback is not the horizon; producing horizon-matched parameters is a calibration concern outside the generator. VaR and ES are both calculated from the P&L distribution generated for $H_W$. The representation of the horizon remains a subsequent decision (D088 §13.8(b)).

---

## 33. Relationship to FX and currency

| Topic | Rule | Source |
|---|---|---|
| Price-formation currency | Each factor has a configured quote currency, compiled to `rf_id_to_ccy_id` | FACT: `static_config.py` |
| Reporting currency | The caller owns only the reporting currency. The run derives settlement currencies and the reporting FX route. | D085 decisions 1 to 3 |
| Reporting FX rows | The reporting FX factors that the valuation prefix lacks are appended to the run axis. "Reporting FX is scenario state, not a Taylor factor." | D085 decisions 4, 5 |
| Reporting FX and the proxy | "Reporting FX (D081, D085) is outside the proxy model." | D088 §9 |
| FX as a risk factor | FX pairs are factor-nodes (quote type `FX_RATE`) with one orientation per pair. `BASEQUOTE` means quote-currency units per one base-currency unit. | D060; D078.4 |
| Exact cross-currency LOG identity | $A_t=B_tP_t\Rightarrow X_A=Y_{(B,\text{LOG})}+s\,Y_{(P,\text{LOG})}$. $s=+1$ when the pair is quoted as target currency per driver currency, and $s=-1$ for the reciprocal. | D088 §9 |
| Base-state condition | Reconstruction reproduces the level identity only if $A_0=B_0P_0$. Otherwise the same terms are a reduced-form model. | D088 §9 |
| Omitted FX term | "Attributes no explicit FX dynamics" | D088 §9 |
| DIFFERENCE-space identity | Bilinear; deferred. Reduced-form DIFFERENCE terms remain admissible. | D088 §9, §10 |
| Units of loadings | $[\beta]=u_{\kappa_A}/u_{\tau}$. Physical-unit metadata is not a prerequisite. | D088 §5 |

**RATIONALE.** A currency effect in a proxy must be an explicit term with an explicit sign. If it were hidden in a source choice or a scale, the sign and the base condition could not be checked or identified.

**CLOSED (D088 §13.8(a), LOCKED 2026-09-29).** Reporting-only FX suffix rows are rows of the proxy compiler axis on the same terms as valuation-prefix rows; ABSOLUTE_BASE_RETURN and NO_SHIFT suffix nodes keep their native path; "reporting FX" in §9 is the reporting conversion. The analysis below records the question as it stood before the decision.

* D088 §1 defines the V1 universe as the run-axis nodes with a V1 chart. D085 decision 4 places the reporting-only FX rows on the run axis. D088 §9 says reporting FX is outside the proxy model.
* If they are included: totality gives them identity rows, authors may write terms for them, and `proxy_model_digest` depends on the reporting currency.
* If they are excluded: the compiler's axis input is the valuation prefix, and the suffix rows obtain their own columns outside the proxy model.

This document does not select either reading. The compiler accepts whatever axis it is given.

---

## 34. Testing strategy

| Category | Purpose | Most discriminating cases | Status |
|---|---|---|---|
| Compiler | Resolution, admissibility, totality, errors | Native targets rejected; native drivers accepted; authored row never falls back; off-axis terms validated; surface EXACT | 61 tests in `tests/config/proxy_model/test_proxy_model_compile.py` |
| Canonicalisation | D088 §6 | All permutations of $\{0.1,0.2,0.3\}$ and of $\{0.1,0.2,-0.3\}$; exact cancellation; signed zero; merged overflow | Included in the 61 |
| Identity and digest | D088 §7 | Digest equality if and only if canonical bit equality (15 variants); the one-ULP rule; pinned encoding; $\kappa$ not in the digest; as-of sensitivity | 11 tests in `tests/config/proxy_model/test_proxy_model_identity.py` |
| Target chart versus driver chart | D088 §2, §3 | LOG target ← DIFF spread: a closed-form oracle on the compiled term ($71.4141$ against $70.6276$) | Tested |
| Two charts on one driver | D088 §4 | Two distinct terms | Tested |
| Numerical application | Aggregation before reconstruction; canonical order; negative loadings; precision; NO_SHIFT and ABS rows; batch and thread invariance (D041) | D088 §11 oracles under the Section 26.6 tolerance; bitwise kernel-to-kernel determinism | Written (Slice 2): `tests/simlab/shocks/test_linear_term_application.py`. The thread-count test skips when only one Numba thread is available. |
| World binding | Missing column; horizon mismatch; key $(d,\tau,h)$ | D088 §11: "missing column rejected before numerics"; parametric keying | Binding function: written (Slice 2), `tests/simlab/shocks/test_column_binding.py`. World-side and parametric keying: Slice 3. |
| Integration | Run identity uses `proxy_model_digest`; D019 goldens as the LOG/LOG special case | Release certification | Not written (Slice 4) |

**EXECUTED (mutation run, Slice 1 delivery, scratch copy).** 14 deliberate defects were introduced. The suite detected 11. The 3 undetected mutants are output-equivalent:

1. a zero-seeded accumulator;
2. keeping authored $-0.0$ at step 2;
3. expanding container pillars that are not on the axis, whose terms are then discarded.

The detected defects included:

* authoring order instead of bit order;
* a tolerance for cancellation;
* $\tau$ taken from $\kappa$;
* identity hard-coded to LOG;
* identity fallback for an emptied authored row;
* the as-of ignored;
* token matching;
* no kind check;
* the chart omitted from the digest;
* native rows included.

**Surface EXACT (FACT, closure pass).** Three tests cover the surface branch:

* `test_surface_self_exact_maps_every_point_to_itself`;
* `test_surface_exact_pairs_by_resolved_coordinate_and_label`, which uses the test-owned `TEST_ANCHOR2_VOL_SURF` (anchor 2 against BRENT's anchor 1);
* `test_surface_exact_rejects_unequal_surface_kinds`, which includes a pair that differs only in `y_axis_kind`.

**EXECUTED (targeted mutation check, closure pass).** Two mutants were introduced: surface token matching, and a surface kind that ignores `y_axis_kind`. Both are detected. The earlier campaign (14 mutants, 11 detected, 3 equivalent) was not re-run.

**Duplicate-coordinate guard (FACT).** Static-configuration validation rejects duplicate `(curve_id, tenor_token)` pairs ("Composite uniqueness violation"). Within one container, distinct tokens resolve to distinct coordinates, so the compiler's duplicate-coordinate check is a defensive guard that valid static configurations do not reach. No test is added.

---

## 35. Implementation status

| Component | Status | Evidence |
|---|---|---|
| Authored semantic records | Implemented | `schema_objects.py` |
| Authoring file format and loader | Not implemented. **OPEN.** | D088 §12.3; no loader in the snapshot |
| Canonical compiler | Implemented | `compile.py`, `canonical.py`, `compiled.py`; 61 tests |
| Digest function | Implemented | `hashing.py::hash_proxy_model`; 11 tests |
| Digest in the run manifest (`ShockIdentity`) | Not implemented | `run_identity/schemas.py` keeps the three legacy fields |
| Slice 2 numerical contract | Closed | D088 §13.4 to §13.6; Section 26 |
| Numerical seam | Implemented (Slice 2), independently validated, not wired | `linear_term_apply.py`, `linear_term_kernel.py`. Legacy `apply.py` and `shock_kernel.py` are unchanged. |
| World binding | Binding function implemented (Slice 2), not wired | `column_binding.py`. Worlds do not yet supply $(d,\tau,h)$ columns (Slice 3). |
| Generator integration (column requests) | Not implemented | `iter_batches` keeps the legacy signature. `ScenarioSetBatch` keeps the legacy fields. |
| Parametric keyed columns | Not implemented | `ParametricRiskData` is factor-keyed |
| Production risk-run integration | Not implemented | `run_scenario_var_es` takes `source_factor_ids` and `shock_scale` |
| Certification | Not started | |
| Atomic switch and legacy retirement | Not started | Legacy resolver and `config/proxy_config` are present; the resolver has no caller under `src/`; production uses the legacy input contract of `run_scenario_var_es` |

---

## 36. Current Slice 1 implementation

**Added:** the five `src/riskquant/config/proxy_model/` modules; `hash_proxy_model` in `hashing.py`; `tests/config/proxy_model/conftest.py` and the two test modules; D088 §12.

**Deliberately untouched:**

* the legacy resolver and `config/proxy_config`;
* `simlab`, including the generators, `ScenarioSetBatch`, `apply.py` and the kernels (`SimlabRunner` has since been removed from the code base);
* `solutions/risk_run`;
* the run-identity schemas and API;
* fixtures and demos.

**Test fixture.** `tests/config/proxy_model/conftest.py` extends the canonical static tables in memory. It adds:

* `TEST_BRENT_WTI_SPREAD_FWD`, a DIFF-configured spread curve;
* `TEST_ANNUAL_CAL_FWD`, an annual calendar curve;
* `TEST_ANCHOR2_FWD`, a monthly curve with anchor 2;
* `TEST_ANCHOR2_VOL_SURF`, a copy of `BRENT_CRUDE_VOL_SURF` with anchor 2 (added in the closure pass);
* ABS and NO_SHIFT overrides on `AAPL_EQ` and `AAPL_ATM_VOL`.

The canonical fixture objects are not mutated.

**EXECUTED on the current snapshot (scratch copy, Python 3.12.3):**

| Command | Result |
|---|---|
| `PYTHONPATH=src:. python3 -m pytest tests/config/proxy_model -q -p no:cacheprovider` | 72 passed (closure pass; 67 before the three surface tests) |
| `PYTHONPATH=src:. python3 -m pytest tests/config tests/solutions/run_identity -q -p no:cacheprovider` | 359 passed |

**EXECUTED at Slice 1 delivery (earlier snapshot; the file contents are unchanged):**

* ruff 0.16.9 check and format: passed, using ruff's defaults, because `pyproject.toml` is not in the snapshot.
* mypy `--strict` on the four new production modules: 0 issues.

**Known limitations of the evidence.**

* The run of `tests/solutions/risk_run` in the scratch environment shows 43 failures and 5 collection errors. The failing set was identical before and after the `hashing.py` change.
* The collection errors are caused by `scripts/demos/e2e/_fixtures.py` being absent from the snapshot.
* The failures are in `test_release_certification.py` (batch-size and thread-count invariance, BLAS determinism, fresh-process agreement). They were not investigated.
* In the current snapshot, `hash_proxy_model` is defined at the end of `hashing.py`. The delivered patch placed it after `hash_compiled_book`. The function text is identical.

---

## 37. Future implementation slices

Structure only. Function names are not specified.

| Slice | Input | Output | Ownership | Acceptance criterion | Must not modify |
|---|---|---|---|---|---|
| 1 (closed) | Authored records, static configuration, as-of, run axis | `CompiledProxyModel`; `hash_proxy_model` | Proxy compiler; run identity (function) | Sections 22, 34 | Legacy path |
| 2 (implemented, validated) | Canonical rows with bound column indices; $\kappa$ per row; base levels; shock batch; output buffer | Scenario levels; integer-key binding function | Numerical application (kernel plus edge validation); `bind_term_columns` | Section 26 contract (D088 §13.4 to §13.6); D088 §11 oracles under the Section 26.6 tolerance; D041 batch invariance | The legacy `apply_shocks_to_base_levels` path, which production uses until the switch |
| 3 | Model $(d,\tau)$ pairs; world declaration ($h$, step unit); native D025 keys | World-column requests $(d,\tau,h)$; column batches keyed $(d,\tau,h)$; parametric representation and step unit; use of the Slice 2 binding with real worlds | Run and world seam; generators | D026 F7, D034, D036 and D087 status notes: missing column rejected; $h$ mismatch rejected; parametric keying | Proxy compilation; valuation |
| 4 | Compiled model; world; axis | `run_scenario_var_es` on V1 inputs; the `ShockIdentity` section carries only `proxy_model_digest` and `RunContextIdentity` is mandatory (D088 §13.3); tier-1 precedence removed | Production run | D040, D082 clause 11 and D085 clause 8 status notes; D088 §13.3 (manifest must carry `RunContextIdentity`); decision on reporting-only FX rows (D088 §13.8(a)) | Valuation, P&L and reporting semantics |
| Certification | Complete V1 path | Regression, determinism and release evidence | Release process | D041; D019 goldens as the LOG/LOG special case (tolerance **OPEN**); D088 §11 | |
| Atomic switch | Certified V1 path | Legacy production behaviour replaced | Production run | Section 38 | |

**FACT relevant to Slice 3.** When one source column is read under two charts on a single-column history, the historical generator has no matching overload for $K_{req}>K_{total}$ (Section 8.7, item 8). V1 makes this layout reachable by design, because the same driver can appear under two charts.

---

## 38. Migration and coexistence

* **Untouched now.** The legacy configuration, resolver, shock universe keying, batch fields, apply kernel, `run_scenario_var_es` inputs and legacy `ShockIdentity`. Production runs use them.
* **Legacy code does not define V1.** D088 supersedes D013. The status notes in D026 F7, D034, D036, D040, D082 clause 11, D085 clause 8 and D087 clause 2 mark the conflicting legacy requirements as no longer authoritative. The legacy code remains as the executing implementation only.
* **Coexistence.** The Slice 1 compiler and the Slice 2 numerical seam and binding are additive and unwired. The generator seam and batch contract are shared by the legacy production paths, so changing them (Slice 3) and switching the run (Slice 4) break the legacy path at the same time. **RATIONALE:** they belong in one atomic switch.
* **Atomic switch.** After the switch, production reads only the V1 path:
  * the resolver, `ProxyTopology`, `ProxyResolution`, the legacy batch fields and the legacy `ShockIdentity` are no longer used;
  * the legacy canonical proxy fixture rows are migrated (explicit $\tau$; the POWER_CH_BASELOAD_HY ← CO2_EUA_ANNUAL_FWD row is not expressible by EXACT);
  * demos and other callers are migrated.
* **Before retirement.** The Section 37 acceptance criteria, D041 invariance, and certification against D019 must hold.
* **Evidence gaps.** Demo scripts and the `riskquant-excel` terminal are not in the snapshot. Their dependence on the legacy interfaces cannot be assessed here.
* **Not yet happened.** No legacy proxy code has been removed. `SimlabRunner` was removed separately as a legacy path (`DECISIONS.md` update notes, 2026-09-28); it is not part of the V1 switch.

---

## 39. Architectural failure modes

These are ways in which future implementation could break the model.

| Failure mode | Violates | Detection |
|---|---|---|
| Reintroducing source selection, including under another name | D088 §9 | Review: any code that chooses a column from availability |
| Using availability as semantics | D088 §9, §12.3 | Compiler signature and import tests (existing); binding tests (future) |
| Using the target chart for driver column lookup | D088 §2, §3 | Two-charts and LOG←DIFF oracles |
| Putting the horizon into proxy identity | D088 §12.5 | Model fields; digest-encoding test |
| Hashing authoring syntax | D088 §12.1 | Digest-equality tests |
| Falling back silently to identity | D088 §6 step 7, §9 | `test_authored_target_never_falls_back_to_identity` |
| Mixing native rows into proxy identity | D088 §12.4 | Native-row tests |
| Aggregating reconstructed levels instead of shocks | D088 §2, §8 | Numerical oracles (future) |
| Letting the kernel perform semantic resolution | CLAUDE.md §4; D088 §12.4 | Kernel signature review |
| Approximate equality in canonicalisation | D088 §6 | Inexact-residue tests |
| Summation that depends on authoring order or on a NumPy reduction's internal order | D088 §6 | Permutation tests |
| Using `proxy_model_digest` as a run, result or cache key | D088 §13.1 | Review; identity table (Section 19.3) |
| Accepting a V1 manifest without `RunContextIdentity` | D088 §13.3 | Manifest schema tests (Slice 4) |
| Compiling the term reduction with a decorator that allows `reassoc` | D088 §13.5 | Kernel decorator review; determinism tests |
| Requesting a fake NO_SHIFT world column | D088 §13.4 | Seam tests (Slices 2, 3) |

---

## 40. Glossary

| Term | Definition |
|---|---|
| Proxy | In V1: the model that defines a target's scenario shock as $\sum\beta_jY_j$. It does not mean a substitute factor. |
| Target | A V1 run-axis factor-node whose shock the model defines |
| Driver | A factor-node whose world column appears in a term |
| Term | One summand $\beta\,Y_{(d,\tau,h)}$, stored as $(d,\tau,\beta)$ |
| Loading | The signed binary64 coefficient $\beta=\partial X_A/\partial Y$ |
| Chart | A member of the `ShiftType` enumeration used in a role: a forward and inverse pair |
| Driver chart ($\tau$) | The chart of a term's world column |
| Target chart ($\kappa_A$) | The configured `shift_type` of the target, used for reconstruction |
| World column | The random variable $Y_{(d,\tau,h)}$ supplied by a scenario world |
| Horizon ($h$) | The world-declared elapsed horizon in the world's step unit |
| EXACT | Container pairing by equal coordinate kind and equal resolved coordinate at the as-of |
| NODE term | An authored term on exact market points, with the pairing chosen by the author (D088 "EXPLICIT") |
| Canonical | The form after D088 §6. It is the identity representation. |
| Totality | The rule that every V1 axis node has exactly one compiled row |
| Identity row | The row $(A,\kappa_A,1)$ given by totality to a node with no authored terms |
| Native row | An axis node configured ABS or NO_SHIFT. It keeps the D025 path and is not a compiled row. |
| Proxy digest (`proxy_model_digest`) | `hash_proxy_model(model)` over the five canonical arrays. It identifies the proxy operator only. |
| Target-definition identity | The identity of target configuration, including $\kappa_A$, carried by `RunContextIdentity` |
| World identity | The identity of the scenario world: `scenario_config` and `DataIdentity` |
| Run identity | `RunManifest.identity()` |
| `ShockIdentity` | Historical name of the manifest section that will carry only the proxy digest (D088 §7, §13.1). It does not identify world shocks. |
| Source selection | Legacy: choosing one column per target from configuration and availability. Excluded in V1. |
| Reconstruction | Applying $T^{-1}_{\kappa_A}(A_0,X_A)$ to obtain a scenario level |
| Binding | The mapping of $(d,\tau)$ and the world's $h$ to a column index |

---

## 41. Normative requirements

Each rule restates the cited record.

1. A Proxy V1 term MUST identify a driver factor-node, a driver chart and a loading (D088 §2).
2. The loading MUST be a signed, finite binary64 value (D088 §2).
3. The driver chart MUST be one of LOG, SIMPLE or DIFFERENCE (D088 §1).
4. A proxy target MUST have configured `shift_type` LOG, SIMPLE or DIFFERENCE (D088 §1).
5. The driver chart MUST NOT be inferred from the target reconstruction chart or from the driver's configured shift type (D088 §2, §3).
6. The target reconstruction chart MUST be owned by the target factor definition and MUST NOT be owned by the proxy model (D088 §3).
7. The proxy model MUST NOT select a source based on world availability, and MUST NOT contain a fallback tier (D088 §9).
8. Every V1 run-axis node MUST have exactly one compiled row. A node with no authored terms MUST receive the identity row (D088 §2).
9. An authored row that is empty after canonicalisation MUST be rejected (D088 §6 step 7).
10. Resolution MUST precede canonicalisation (D088 §6 step 1, §12.2).
11. Canonicalisation MUST be deterministic and bit-exact, and MUST NOT use tolerances, Decimal or compensated summation (D088 §6).
12. Zero elimination MUST use exact comparison with zero only (D088 §6 steps 2 and 6).
13. The proxy digest MUST be computed from the canonical representation, and MUST NOT be computed from authoring syntax (D088 §7, §12.1).
14. ABS and NO_SHIFT axis nodes MUST NOT be rows of the compiled model or inputs to the proxy digest (D088 §12.4).
15. The compiled proxy model MUST NOT contain a horizon. No layer may infer $h$ from the target, the driver, a resolver, a fallback, source selection or the target chart (D088 §12.5).
16. A concrete world column MUST be identified by (driver, driver chart, horizon) (D088 §4).
17. The generator MUST validate the requested horizon against the world declaration (D088 §12.5).
18. Missing required world columns MUST be rejected before numerics (D088 §11; D088 §4).
19. Target reconstruction MUST occur after shock aggregation (D088 §2).
20. The numerical kernel MUST NOT perform semantic resolution (CLAUDE.md §4; D088 §12.4).
21. The final risk result MUST be bitwise invariant to `batch_size` (D041).
22. Reporting FX MUST remain outside the proxy model (D088 §9).
23. A later numerical kernel MAY lower native ABS rows to one-term, unit-loading rows on their native column; NO_SHIFT rows are zero-term rows (rule 29). This MUST NOT make them part of the authored model or the proxy digest (D088 §12.4, §13.4).
24. An authoring file format MAY be introduced. It is an edge concern (D088 §12.3).
25. `proxy_model_digest` MUST NOT be used alone as a run, result or cache key (D088 §13.1).
26. A manifest whose `ShockIdentity` carries `proxy_model_digest` MUST carry `RunContextIdentity` (D088 §13.3).
27. The term sum MUST be evaluated in ascending canonical term order, in float64, with shocks converted from float32 before multiplication (D088 §13.5).
28. The reduction kernel MUST use `stable_math_parallel_njit`, and MUST NOT parallelise the reduction within one row (D088 §13.5).
29. A NO_SHIFT row MUST output $A_0$ and MUST NOT require a world column (D088 §13.4).
30. Kernel-to-Python oracle tests MUST use relative tolerance $10^{-12}$ and absolute tolerance $10^{-15}$. Determinism tests MUST compare bitwise within one execution environment (D088 §13.6).

---

## 42. What Proxy V1 is

Proxy V1 is a rule for computing the scenario movement of one factor from the scenario movements of other factors. The rule is written as a short list of terms. Each term names a driver factor, the chart in which that driver's movement is measured, and a signed number. The target's movement is the sum of each term's driver movement multiplied by its number. The target then turns that summed movement into a level using its own chart, starting from its own base level.

The rule is compiled once per run. Compilation reads:

* the authored terms;
* the static configuration;
* the run date;
* the run's factor axis.

Compilation resolves every name to one factor-node, pairs container pillars by their real economic coordinates at the run date, and checks that every target and chart is allowed. It gives every eligible axis factor exactly one row: a factor that nobody configured gets the identity row, which reads its own column. It then puts every row into one exact form, and hashes that form to identify the model. That hash identifies the model only: the run identity also covers the world (including the horizon), the target definitions (including each target's chart) and the base levels. Nothing about scenario data, horizons, batches or valuation enters this step.

At run time, which is not yet implemented:

1. the run attaches the world's horizon to each (driver, chart) pair and finds the matching world column;
2. a missing column stops the run;
3. a sparse sum forms each target's movement in every scenario;
4. the target's own chart rebuilds the level;
5. from that point the existing level validation, valuation, P&L, reporting FX and VaR/ES steps run unchanged.

In compact form:

$$
\text{canonical sparse operator over explicit }(d,\tau,\beta)\text{ terms}
\;\rightarrow\;
\text{world-column binding }(d,\tau,h)\mapsto c
\;\rightarrow\;
\text{shock-space aggregation}
\;\rightarrow\;
\text{target-owned reconstruction.}
$$

The legacy mechanism chose one existing column per target, based on configuration and data availability, and scaled it by a positive number in the target's chart. Proxy V1 does not choose. It states a relationship, and the world must supply exactly the columns that the relationship names.

---

## Source Coverage

**Decision records (`DECISIONS.md`, as attached):**

* D088 including §12 and §13;
* D089;
* D013 and its status note;
* D025;
* D026 F7 status note;
* D034 status note;
* D036 status note;
* D040 status note;
* D041;
* D060;
* D069;
* D070;
* D075;
* D078.4;
* D081 §1;
* D082 clause 11 status note;
* D085 decisions 1 to 8 and the clause 8 status note;
* D087 clause 2 status note.

**Slice 1 source:**

* `src/riskquant/config/proxy_model/__init__.py`
* `src/riskquant/config/proxy_model/schema_objects.py`
* `src/riskquant/config/proxy_model/compiled.py`
* `src/riskquant/config/proxy_model/canonical.py`
* `src/riskquant/config/proxy_model/compile.py`
* `src/riskquant/solutions/run_identity/hashing.py`

**Slice 1 tests:**

* `tests/config/proxy_model/conftest.py`
* `tests/config/proxy_model/test_proxy_model_compile.py`
* `tests/config/proxy_model/test_proxy_model_identity.py`

**Static configuration and resolution:**

* `src/riskquant/config/static_config/static_config.py`
* `src/riskquant/config/static_config/rf_resolution.py`
* `src/riskquant/market_data/risk_factors/shift_types.py`

**Legacy mechanism:**

* `src/riskquant/config/proxy_config/validate.py`, `compile.py`, `proxy_topology.py`, `loader.py`
* `src/riskquant/proxy_resolver/proxy.py`
* `src/riskquant/ingest_fixtures/config/risk_proxy_config/canonical_risk_proxy_config.py`
* `src/riskquant/simlab/builder/shock_universe.py`
* `src/riskquant/simlab/builder/shift_domain.py`
* `src/riskquant/simlab/scenarios/historical.py`, `parametric.py`, `parametric_data.py`, `base.py`
* `src/riskquant/simlab/shocks/apply.py`, `shock_kernel.py`
* `src/riskquant/simlab/runner.py`: removed from the code base (`DECISIONS.md` update notes, 2026-09-28)
* `src/riskquant/jit.py` (decorator flags)

**Run, identity and levels:**

* `src/riskquant/solutions/risk_run/api.py` (`run_scenario_var_es`, `_require_shock_sources`, `derive_run_factor_axis`)
* `src/riskquant/solutions/risk_run/schemas.py` (`RunFactorAxis`)
* `src/riskquant/solutions/run_identity/schemas.py` (`ShockIdentity`)
* `src/riskquant/solutions/scenario_models/schemas.py`
* `src/riskquant/market_data/assemble.py`
* `src/riskquant/market_data/builder.py`
* `src/riskquant/reporting/fx_conversion.py`

## Open Items

1. **Reporting-only FX suffix rows and the proxy compiler axis (D088 §13.8(a)).** This must be decided before Slice 4 (Section 33).
2. **Parametric world representation of $(d,\tau,h)$ (D088 §13.8(b)).** It covers the declared horizon, step unit, chart identity, chart-specific parameter interpretation and the D034 Case declaration. It must be decided before Slice 3. `ParametricRiskData` is unchanged.
3. **Native ABS column request form at the world seam.** This is a Slice 3 item. The NO_SHIFT rule is closed (D088 §13.4).
4. **Authoring file format and loader (D088 §12.3).** `NodeProxyTerm` and `ContainerProxyTerm` are semantic schema objects, not a file format. A loader is an edge tooling concern and is needed before the atomic switch.
5. **Certification tolerance for the D019 goldens.** Float64 loadings versus the legacy float32 `shock_scale` differ by $1.25\times10^{-8}$ relative.
6. **Migration of the legacy canonical proxy fixture rows.** They need an explicit $\tau$. The row POWER_CH_BASELOAD_HY ← CO2_EUA_ANNUAL_FWD is not expressible by EXACT.
7. **Callers outside the snapshot.** Demo scripts and the `riskquant-excel` terminal are absent from the snapshot. Their dependence on legacy interfaces must be verified before the switch.
