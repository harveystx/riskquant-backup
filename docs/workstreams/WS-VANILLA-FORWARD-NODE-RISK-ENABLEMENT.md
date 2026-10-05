# WS-VANILLA-FORWARD-NODE-RISK-ENABLEMENT

Status: SPECIFICATION FROZEN (revision 8, final). NODE FORWARD SENSITIZER SPECIFICATION: LOCKED. Implementation may begin.
Source of truth: snapshot `0c70f76`; `DECISIONS.md` with D101 LOCKED.
Supersedes revisions 1 to 7.

Owner decisions in force:
- Same production treatment as SPOT for FULL_REVAL and TAYLOR/DERIVED. TAYLOR/SUPPLIED out of scope.
- Continue NODE first. `VANILLA_FORWARD_CURVE` valuation is not a prerequisite.

Labels: OBSERVED (read in the snapshot), INFERRED (derived from observed facts), ASSUMED (not verified; stated as an assumption).

Corrections to revision 4:
- (C1) Revision 4 stated that UID-keyed provenance "holds for multi-row UIDs" but did not state the ambiguity rule. Section 3 now states it: the repository does not guarantee one forward source per UID at the run boundary, and D102.3 fails closed.
- (C2) Revision 4 used a generic level bound. Section 5.3 derives it per chart from the actual kernel operations.
- (C3) Revision 4 controlled F4/F4' by "same seed". Section 7 WS-B now asserts identical captured driver shocks.
- (C4) Revision 4 did not give F4 an FX leg. A coefficient that depends on the forward level is needed to detect driver substitution in a coefficient. F4 is now EUR-settled.

Corrections to revision 5:
- (C5) Revision 5 test C8 ("delivery present versus absent, identical output") did not restrict the case to exact NODE. Delivery presence/absence invariance is an exact-NODE property only, because only exact NODE admits an absent delivery. Section 3.1 and tests C8a to C8d now state the class-specific contract.
- (C6) Revision 5 test C7 ("one code path; rf = `rel_name_to_rf_id[name]`") compressed two identity contracts into one name lookup. Section 3.1 and tests C7a to C7c now separate exact and relative identity.
- (C7) Revision 5 listed inverted FX under "not analytically exact". Inverted FX is mathematically defined; it is outside the current sensitizer admission contract. Section 4 now classifies it as unsupported, not as a category of Taylor non-exactness.

Corrections to revision 6:
- (C8) Revision 6 referenced the inverted-FX refusal as both D103.3 and D103.4(c). The authoritative clause is D103.3. Condition (c) is removed from D103.4, which now requires D103.3 in addition, and every reference uses D103.3. The contract is unchanged.
- (C9) Revision 6 section 5.9 listed generic batch invariance together with the F4 Run A/Run B equalities. The generic determinism contract and the Proxy V1 target-coordinate / β-separation invariant are now separate subsections (§5.10 and §5.11 in revision 8).

Corrections to revision 7 (numerical closure):
- (C10) Section 5 is rewritten from the traced operation sequences: absolute subtraction model, computed versus exact T, operation counts including the inverted-FX reciprocal (V0), float32 floor φ, and explicit preconditions (§5.0).
- (C11) The historical level bounds are re-derived per chart. The revision-7 LOG formula dominated the rigorous bound only for |β·Y64| up to about 0.35. ΔX is now the actual non-negative difference and Δ̄X its proven bound. A-L1 is restated against a `decimal` reference; A-L2 is added.
- (C12) The symbol Y is split into Y64, Y64_p, Y32 and Y32_o. The F4 driver factor is renamed D (rf_D, x_D,0), so Y denotes only chart returns.
- (C13) The ES bound includes the oracle's own rounding (2γ_{f+2}). The portfolio bound includes the (1 + 2γ_{M−1}) term and defines M = |U|. The oracle reproduces the tail geometry, including the snap rule.
- (C14) DERIVED: n is defined on the compiled book (n_m = d_m + c_m); production and oracle depth is γ_{2n+1}; P-n is asserted on the compiled book, with no inference from UID cardinality.
- (C15) Oracle independence is separated into structural independence (A) and input-binding assurance (B).
- (C16) The numerical certification scope is NODE + `DiscountingMode.NONE`. The CURVE-discounting calculation is labelled informational.
The acceptance formulas for V0, FULL_REVAL dV, DERIVED dV, dW and VaR are unchanged by C10 to C16.

---

## 1. Concept separation

```text
instrument source ≠ RF namespace ≠ dependency topology ≠ block type ≠ Proxy V1 target/driver topology
```

| Concept | Representation (OBSERVED) | Responsibility |
|---|---|---|
| Instrument source | `IngestBatch.instrument_ids`, bound to block rows by trade UID | Source-specific forward admission (D102, D103), applied in addition to global run and methodology rules |
| RF namespace | `GlobalIdNamespace`, enforced by `_validate_global_id_namespace` (`config/static_config/validate.py`) | rf identity and topology validation |
| Dependency topology | `UixMapper.efp_effective_quads`, `edf_effective_triples`, `fx_effective_pairs`, `market_id_universe` | Sensitivity expansion |
| Block type | `BlockType.VANILLA_FORWARD` | Kernel and sensitizer dispatch |
| Proxy V1 | compiled operator; `apply_linear_terms_to_base_levels` | Target-factor shock construction and target/driver topology |

`GlobalIdNamespace.NODES` is NOT a NODE-source discriminator. `_validate_global_id_namespace` places node ids AND curve-point rf ids in `NODES`, so a future price-curve pillar is also in `NODES`. No source-kind field is introduced.

Admission model: global run and methodology rules (D081 composition, P-1, D084, D085, D086) + source-specific forward admission (D102) + methodology-specific forward conditions (D103). Instrument source is one input, not the whole predicate.

---

## 2. Proxy V1 coordinate (OBSERVED at every arrow)

```text
Proxy model: compile_proxy_model → lower_axis_operator → bind_operator_to_world
             (one compiled row per target, D088 §1; equal (d, τ) terms merged by summation, D088 §6)
  → target level: _kernel_apply_linear_terms (simlab/shocks/linear_term_kernel.py)
        X_A = Σ_t β_t·float64(Y32_t)        Y32_t: float32 driver shock column consumed by the kernel
        LOG: x_A,s = x_A,0·exp(X_A); SIMPLE: x_A,0·(1 + X_A); DIFFERENCE: x_A,0 + X_A
  → scenario level: shocked_levels (float64), assembled in run_scenario_var_es
  → Taylor axis level: taylor_axis.levels(base_levels=base_levels[:k_val],
        shocked_levels=shocked_levels[:k_val]); the same slice goes to _write_full_reval_block
  → book factor: build_sensitivity_book_spec(required_factor_ids=<run axis>) keys the book on axis
        TARGET rf ids; ABSOLUTE is the default convention (book_adapter)
  → displacement: kernel_build_displacement, float32(x_s − x0) per book factor row
  → Taylor P&L: kernel_taylor_pnl
```

For an authored term A ← β·D (target factor A, driver factor D), the sensitivity coordinate is ∂PV/∂x_A. It is not β·∂PV/∂x_A and not ∂PV/∂x_D. β multiplies the consumed driver shock Y32 (section 5.1). The `Sensitizer` protocol `(block, uix_mapper, market_data, scenario)` (`sensitizer_registry.py`) receives no loading and no driver identity, so a sensitizer cannot apply β. Proxy V1 governs target-factor shock construction and target/driver topology. It does not change the sensitivity coordinate.

Scenario charts define the reconstruction T⁻¹_κ only. They do not define the Taylor coordinate (D081 §4).

---

## 3. Provenance contract

OBSERVED facts:
1. `dispatch_ingest_validation` (`portfolio/ingest/dispatcher.py`) calls `validate_unique_uids(df)`, which raises `ValueError("Duplicate UIDs")`. A batch from `compile_raw_df_into_ingest_batch` therefore has unique UIDs.
2. `IngestBatch.__post_init__` checks dtypes, contiguity and lengths only. It does not check UID uniqueness.
3. D055 (LOCKED): "one economic trade → one or many valuation rows"; `trade_uids` are not required to be unique (D045/VAL-05).
4. The run accepts multi-row UIDs: `_settlement_ccy_ids` groups ingest rows by UID ("Rows of one UID must share one settlement currency"); `_row_aggregation` aggregates rows.
5. `compile_portfolio_from_ingest_batch` stable-sorts by `instrument_id` and compiles each contiguous run; `compile_vanilla_forward_block` sets `trade_uids = batch.uids[start:end]`. Only `VANILLA_FORWARD_NODE_ID` has a forward compiler (D050).
6. `INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY` (`portfolio/instruments.py`) maps `VANILLA_FORWARD_NODE` and `VANILLA_FORWARD_CURVE` to `InstrumentTypeFamily.FORWARD`.

Result: **Case B.** The repository does not guarantee one FORWARD instrument per UID at the run boundary. `ValuationContext` accepts any valid `IngestBatch`, and D055 permits repeated UIDs. Production raw-frame ingest yields unique UIDs, but that is not a run-boundary invariant.

Fail-closed rule (D102.3): for every UID u that has a row in `portfolio.vanilla_forward`, every ingest row with `batch.uids == u` and a FORWARD-family instrument must have `instrument_id == VANILLA_FORWARD_NODE_ID`. Otherwise the run raises `CompositionContractError` (the run-boundary convention of `_require_taylor_capable_spot` and `_settlement_ccy_ids`).

| Case for UID u | Outcome |
|---|---|
| NODE + NODE | Homogeneous. D102.3 passes. The sensitizer sums both rows. |
| NODE + CURVE | Ambiguous source. D102.3 refuses. |
| FORWARD + non-FORWARD (for example an option row, D055) | D102.3 checks the forward rows only. FULL_REVAL is not affected. TAYLOR/DERIVED is refused by D103.4(a) unless every non-forward row is a SPOT row admitted under D084. |

### 3.1 NODE identity and delivery (D101, D077.4; OBSERVED)

Code path:
- rf lookup: `_resolve_underlying` (`portfolio/ingest/compile.py`) maps the underlying name through the static `rel_name_to_rf_id` table, with no as-of input. This lookup is the same for exact and relative names. It is a name-to-rf table, not the identity contract.
- Identity contract: `_validate_contract_identity` (`portfolio/ingest/identity.py`) classifies the parent curve by `x_axis_kind`, then applies the class rule. `_resolve_curve_or_surface_suffix` (`config/static_config/rf_resolution.py`) returns the bare label for LABEL and resolves an OFFSET token through the roll method at the as-of. `build_risk_factor_name_mapping_for_as_of_date` gives the as-of mapping `abs_to_rf_id`.
- D077.4: a relative name, together with the run as-of, resolves to exactly one rf_id, fixed for the run; delivery never re-resolves that rf.

```text
EXACT NODE (LABEL x-axis)
  → exact identity: rel_name_to_rf_id[label name]; absolute name = bare label for every as-of
  → delivery optional; delivery never selects the rf
  → delivery present → asset-leg DF date = delivery; absent → settlement
  → in DiscountingMode.NONE (DF ≡ 1.0) presence and absence give identical sensitivities

RELATIVE NODE (OFFSET_M, OFFSET_Y, OFFSET_HY; supported only for (x_axis_kind, roll_method) pairs in
               _SUPPORTED_PERIOD_LABELS, i.e. OFFSET_M or OFFSET_Y with CALENDAR_PERIOD; every other pair fails closed)
  → rf: rel_name_to_rf_id[relative name], fixed for the run (D077.4)
  → contract identity: resolved at the run as-of (abs_to_rf_id; OFFSET resolver)
  → delivery REQUIRED; absent → PrettyKeyError at ingest
  → delivery must fall in the resolved period (D065); otherwise PrettyKeyError at ingest
```

Delivery presence/absence invariance in `DiscountingMode.NONE` is an exact-NODE property only: a relative NODE has no admissible absent-delivery case. The statement "delivery is irrelevant whenever discounting_mode == NONE" is false. In every mode, delivery:
- is required and period-checked for relative NODE (D065);
- is refused before the as-of (D079, `build_market_data_matrix`);
- enters the run identity (`unique_dates_ord32` in `hash_compiled_portfolio`).

After exact identification or relative resolution, the forward block holds the canonical scalar rf that it values (`market_id_universe[efp_effective_quads[·, 0]]` equals the ingest-resolved `underlying_ids` for that row). The sensitizer emits F rows keyed by that rf.

---

## 4. Mathematical contract (verified; frozen)

```text
PV = w_fp·E_fp + w_df·E_df,   w_fp = Q,   w_df = −Q·K     (_kernel_price_vanilla_forward:
                                                          qty_i·(pv_leg_u − strike_i·eff_df))
E_fp = F·DF_a·X_a  (_fold_effective_forward_prices),   E_df = DF_k·X_k  (_fold_effective_discount_factors)
E_e = Π_{a∈A_e} p_a(x);   P^{¬a} = Π_{c≠a} p_c;   P^{¬ab} = Π_{c∉{a,b}} p_c    (explicit products)

∂PV/∂x_i      = Σ_e w_e Σ_a P^{¬a} ∂_i p_a
∂²PV/∂x_i∂x_j = Σ_e w_e [ Σ_a P^{¬a} ∂_ij p_a + Σ_a Σ_{b≠a} P^{¬ab} ∂_i p_a ∂_j p_b ]
```

Verification:
- PV is affine in (E_fp, E_df), so there is no effective-input cross term.
- ∂_j P^{¬a} = Σ_{b≠a} P^{¬ab}·∂_j p_b gives the double sum over ordered pairs.
- If i = j and two primitives depend on x_i, the diagonal receives 2·P^{¬ab}·∂_i p_a·∂_i p_b.
- If i ≠ j, H is symmetric.

Emission:
- `first_order[i] = ∂PV/∂x_i`.
- `second_order[i] = ∂²PV/∂x_i²`. `kernel_taylor_pnl` applies 0.5 exactly once (Horner form (d1 + 0.5·d2·x)·x).
- `cross_second_order{i,j} = ∂²PV/∂x_i∂x_j`, once per unordered pair, coefficient 1 (D081 §5).

Uniqueness: the book compiler canonicalises cross pairs to (lo, hi), then `_collapse_duplicates` raises under `DuplicatePolicy.RAISE` on a repeated (position, factor) or (position, lo, hi) key, and rejects i = j (OBSERVED). The sensitizer emits:
- at most one diagonal row (first_order and second_order together) per (position, global rf);
- at most one cross row per (position, min rf, max rf).

It sums every path, leg, effective-input and row contribution before emission. `DuplicatePolicy.RAISE` is a guard only.

NODE + NONE primitives: F = x_F (`forward_nodes`); direct X = x_X (`fx_spots_effective`); identity X ≡ 1 (`fx_eff_uix == INVALID_UIX`, D062); DF ≡ 1.0. Every ∂_ij p = 0. DF ≡ 1.0 follows from the D061 decision text ("`discounting_mode = NONE`, `DF(as_of, target_date) = 1.0`") and the `build_market_data_matrix` NONE branch (`np.ones`, comment "(D061)"). D080 amends D061 for CURVE only.

| Topology | first_order | second_order | cross |
|---|---|---|---|
| {F} | F: Q·DF_a | 0 | none |
| {F, X_a} | F: Q·DF_a·X_a; X_a: Q·F·DF_a | 0, 0 | (F,X_a) = Q·DF_a |
| {F, X_k} | F: Q·DF_a; X_k: −Q·K·DF_k | 0, 0 | (F,X_k) = 0 |
| {F, X_a, X_k} | as above | 0, 0, 0 | (F,X_a) = Q·DF_a; (F,X_k) = 0; (X_a,X_k) = 0 |
| {F, X}, X_a = X_k | F: Q·DF_a·X; X: Q·(F·DF_a − K·DF_k) | 0, 0 | (F,X) = Q·DF_a |

Structure:
- F(p) is the rf row set by topology, never by value (`derive_taylor_declaration` docstring).
- SECOND_ORDER_FULL requires every unordered pair of F(p), with zero pairs stored explicitly (`verify_declared_order_against_book`, D086).
- Q = 0 keeps the full topology with zero coefficients. This is a low-level sensitizer and book contract test, not a production-ingest fixture: `_validate_non_zero_quantity(df, mask, "Forward")` rejects zero quantity at ingest.

Exactness classification:
- ANALYTICALLY EXACT: NODE + `DiscountingMode.NONE`, level space, ABSOLUTE or RELATIVE book. SECOND_ORDER_FULL equals the bilinear PV change. TRUNCATED and FIRST_ORDER omit exactly Q·DF_a·ΔF·ΔX_a.
- NOT ANALYTICALLY EXACT (mathematically defined; the current Taylor representation does not equal full valuation): a LOG book; `DiscountingMode.CURVE` (DF = exp(Σ c_p z_p)).
- UNSUPPORTED (mathematically defined; outside the current NODE sensitizer admission contract): inverted FX on either leg. The sensitizer detects `fx_effective_pairs[·, 1] == 1`, raises `NotImplementedError`, and performs no kernel arithmetic. The run refuses the route under D103.3. The valuation of an inverted leg is supported in FULL_REVAL (`_build_fx_spots_effective` writes 1.0/level). Admission requires a separate decision with the D084 evidence set.
- NUMERICALLY APPROXIMATE: every production comparison (section 5).

---

## 5. Numerical validation contract

Scope: this contract certifies the admitted NODE + `DiscountingMode.NONE` paths only. A calculation marked "informational" is non-admitted, non-production and not part of this workstream's certification.

### 5.0 Arithmetic model and preconditions

- A-F (platform): every float64 operation is IEEE-754 binary64, round-to-nearest-even, with no extended-precision intermediates. The float32 store is round-to-nearest-even. An FMA rounds once.
- Under A-F and P-N, for floating-point operands a and b: fl(a·b) = ab(1+δ), fl(a/b) = (a/b)(1+δ), and |fl(a ± b) − (a ± b)| ≤ ε·|a ± b| ≤ ε(|a| + |b|), with |δ| ≤ ε. No other float64 rounding statement is used. Every subtraction is bounded in this absolute form, so the bounds stay valid under cancellation.
- Float32 store: |fl32(v) − v| ≤ u·|v| + 2⁻¹⁵⁰ for |v| ≤ 2¹²⁷.

| ID | Precondition | Checked by |
|---|---|---|
| P-N | Every nonzero float64 input, intermediate and result lies in [2⁻¹⁰⁰⁰, 2¹⁰⁰⁰]. This excludes nonzero subnormal float64 values. | fixture magnitude assertions |
| P-O | Every float32-stored value v satisfies \|v\| ≤ 2¹²⁷ | B-tests and D-tests (\|dV_o\| + tol ≤ 2¹²⁷) |
| P-Δ | Every displacement Δ_i is 0 or \|Δ_i\| ≥ 2⁻¹²⁶ | D-tests |
| P-Y | Every Y64 is 0 or 2⁻¹²⁶ ≤ \|Y64\| ≤ 2¹²⁶ | level tests |
| P-L1 | Every fixture factor has `rf_id_to_unit_scale_factor == 1.0`, so D069 normalisation is the identity | fixture |
| P-L2 | Every axis row has exactly one term (`row_term_offsets` differences are 1) | fixture |
| P-L3 | Horizon 1 | fixture |
| P-C | m_c ≤ 4: each emitted coefficient is formed from its inputs with at most 4 float64 roundings, in the sensitizer and in the oracle | satisfied by the section 4 formulas on the fixture positions |
| P-n | n ≤ 32 on the compiled book (§5.6) | C-tests and D-tests, from `position_offsets` and `cross_offsets` |
| P-T | Tail mass m > 0 after the snap. The fixtures use c = 0.75 and S ≥ 4, so m ≥ 1. | VaR/ES tests |

### 5.1 Symbols

| Symbol | Definition |
|---|---|
| u | 2⁻²⁴: float32 unit roundoff |
| ε | 2⁻⁵³: float64 unit roundoff |
| eps | 2⁻⁵²: NumPy machine epsilon; used only in the tail snap rule (§5.8) |
| γ_k | k·ε/(1 − k·ε) |
| φ | 2⁻¹⁴⁹: float32 floor |
| s | scenario index |
| x0_f, x_s,f | base and shocked level of axis factor f (float64) |
| A, D | the F4 proxy target factor (`WTI_CRUDE_FWD[M+0]`) and its driver factor (`BRENT_CRUDE_FWD[M+0]`); the authored term is A ← β·D |
| Y64 | the oracle's float64 chart return from raw history, horizon 1: LOG `math.log(fl(v1/v0))` (CPython); SIMPLE `fl(fl(v1/v0) − 1.0)`; DIFFERENCE `fl(v1 − v0)` |
| Y64_p | the production float64 value inside `_kernel_fused_h_day_shock` before narrowing (not observable) |
| Y32 | fl32(Y64_p): the production float32 shock that `apply_linear_terms_to_base_levels` consumes; captured by the B-β0 spy |
| Y32_o | fl32(Y64): the oracle's reproduction of the narrowing |
| β | loading of the row's single term (1.0 for an identity row) |
| X_p, X_o | fl(β·Y32) and fl(β·Y32_o) |
| ΔX | \|X_p − X_o\| ≥ 0 (actual difference) |
| Δ̄X | proven, oracle-computable upper bound of ΔX (§5.3) |
| x̂ | oracle level T⁻¹_κ(x0, X_o) in float64, evaluated with the A-L2 forms |
| Q, K | signed quantity and strike of a forward row |
| X_a, X_k | exact effective rates: the stored level x (direct), the exact real 1/x (inverted), or 1 (identity) |
| F, DF_a, DF_k | node level and discount factors (DF ≡ 1.0 in NONE) |
| T0_exact | \|Q·F·DF_a·X_a\| + \|Q·K·DF_k·X_k\| in exact arithmetic at the base |
| T0_o | fl(fl(\|Q\|·\|a_o\|) + fl(\|Q\|·\|b_o\|)), the oracle-computed value, with a_o, b_o the oracle leg values (§5.4) |
| T_s,exact, T_s,o | the same at scenario s |
| V0_o, V_s,o, dV_o | oracle PV at the base, oracle PV at s, and dV_o = fl(V_s,o − V0_o) |
| Ā_i, Ā_ij | sum of the absolute values of the contributions that form coefficient i or pair ij (for V_X = Q·(F·DF_a − K·DF_k): \|Q·F·DF_a\| + \|Q·K·DF_k\|) |
| Δ_i | fl64(x_s,i − x0_i) from the captured levels; production and oracle values are bit-identical |
| τ̄_k | per declared book entry of the position, zero pairs included: Ā_i·\|Δ_i\| + ½·\|b_i\|·Δ_i² (diagonal row); Ā_ij·\|Δ_i·Δ_j\| (cross pair). Σ τ̄ runs over the position's entries for the declared order; Σ τ̄_o is the oracle-computed value |
| d_m, c_m | `position_offsets[m+1] − position_offsets[m]` and `cross_offsets[m+1] − cross_offsets[m]` of the compiled book that `kernel_taylor_pnl` receives |
| n_m, n | n_m = d_m + c_m; n = max_m n_m |
| μ_m | 2d_m + c_m ≤ 2n_m: monomials of position m |
| m_c | float64 roundings that form one coefficient (P-C) |
| dV_T,o | the oracle's declared Taylor polynomial value (float64) |
| r_s, r_0 | reporting rates at s and at the base: the captured FX level, or fl(1.0/level) when inverted; identity 1.0 |
| d_r | fl(r_s − r_0) |
| dW_o | fl(fl(dV_o·r_s) + fl(V0_o·d_r)) (oracle) |
| M | \|U\|: rows summed by `converted.sum(axis=0)`; M ≥ 1 |
| P_s | portfolio P&L at s |
| α, m, f, w, var_index | tail geometry of §5.8 |
| ε_in | sup-norm bound of the input P&L vector difference for a VaR/ES comparison |
| ∥·∥_∞ | maximum over scenarios |

### 5.2 Comparison design and oracle independence

- The level test compares oracle levels from raw history with captured production levels, for the historical model only.
- P&L tests feed the CAPTURED production shocked levels and the run base levels to the oracle (pattern: SPOT `cap` fixture wrapping `_write_full_reval_block` and `convert_scenario_pnl_to_reporting_currency`). The oracle recomputes valuation, Taylor, conversion and VaR/ES itself. Level-generation error does not enter P&L comparisons. Scope: P&L tests certify downstream arithmetic for given levels; level generation is certified by the historical level test only.

A. Structural independence. The oracle imports only `numpy`, `math`, `decimal` and static-config name tables. It never imports production valuation, the folds, the sensitizer, `run_valuation`, `_compute_lower_tail_parameters` or `compute_var_es`. It recomputes rates, d_r, tail geometry, values, Taylor polynomials, VaR and ES itself. Shared inputs are proof inputs, not the arithmetic under test: captured shocked levels, run base levels, the captured F4 Y32 (B-β0 only), raw frame, raw snapshot, raw history and the authored proxy rule.

B. Input-binding assurance. The historical level-by-name test and the axis test verify that each axis row the oracle reads binds to the intended factor name. This is test evidence about indexing. It is not used as proof of A.

Prescribed oracle forms:
- V = fl(Q·fl(fl(F·X_a) − fl(K·X_k))), with X = fl(1.0/x) for an inverted leg, x for a direct leg, and no multiplication for an identity leg.
- dW_o = fl(fl(dV_o·r_s) + fl(V0_o·d_r)), with r and d_r recomputed from the captured levels by the same correctly rounded operations as production, so they are bit-identical.
- Y64 forms as in §5.1 (A-L2).
- Tail geometry as in §5.8; ES = (sum of the f smallest values + w·s_(f)) / m.
- Taylor polynomial: each monomial with at most 2 multiplications (½·b is exact), summed in any order.

### 5.3 Historical levels (approximate; CONDITIONAL on A-L1)

Production sequence (OBSERVED):
1. History is float64. `build_risk_factor_matrix_from_df` defaults to float64; `_kernel_fused_h_day_shock` is dtype-pinned to float64 ("A float32 matrix raises TypeError").
2. `_kernel_fused_h_day_shock` (`simlab/scenarios/historical.py`, `stable_math_parallel_njit`: no `reassoc`) computes Y64_p (LOG `math.log(v1 / v0)`; SIMPLE `v1 / v0 − 1.0`; DIFFERENCE `v1 − v0`) and stores Y32 = fl32(Y64_p).
3. `_kernel_apply_linear_terms` (`parallel_njit`: `contract`, `reassoc`, `nsz`) widens Y32 exactly, forms X_p = fl(β·Y32) (added to 0.0, or fma(β, Y32, 0.0); both equal fl(β·Y32)), then reconstructs: LOG x0·exp(X); SIMPLE x0·(1 + X); DIFFERENCE x0 + X.

A-L1 (fixture-scoped, conditional contract input): each of Numba `math.log` (compiled with `stable_math_parallel_njit`), Numba `math.exp` (compiled with `parallel_njit`), CPython `math.log` and CPython `math.exp` returns, on the fixture's arguments, a value within 4 binary64 ulps of the `decimal` reference computed by the test-owned helper at precision 40. The float argument is converted to `Decimal` exactly. The helper also asserts that log(1.0) returns exactly 0. A-L1 is an empirically verified implementation fact for the fixture arguments, not an analytical theorem. The contract relies on the documented correct rounding of `Decimal.exp` and `Decimal.ln` at the context precision to read the reference as within 0.5·10⁻³⁹ relative of the exact value. Each implementation error used below is therefore κ ≤ 8ε + 2⁻¹²⁹ (relative).

A-L2: the oracle uses the Y64 forms of §5.1 and the reconstruction forms x0·exp(X), x0·(1 + X), x0 + X.

LOG derivation:
- |Y64_p − Y64| ≤ (16ε + 2⁻¹²⁸)·|Y64|/(1 − 8ε). The rounding of the shared ratio fl(v1/v0) cancels.
- Under P-Y both values narrow in the normal float32 range: |Y32 − Y32_o| ≤ u·|Y64_p| + u·|Y64| + |Y64_p − Y64| ≤ (2u + 2⁻⁴⁸)·|Y64|.
- ΔX ≤ |β|·|Y32 − Y32_o| + ε(|β·Y32| + |β·Y32_o|) ≤ |β|·|Y64|·(2u + 2⁻⁴⁷) =: Δ̄X.
- For every real d: |e^d − 1| ≤ e^{|d|} − 1 ≤ |d|·e^{|d|}. So |e^{X_p} − e^{X_o}| ≤ e^{X_o}·ΔX·e^{ΔX}, for either sign of X_p − X_o. The map t ↦ t·e^t is increasing for t ≥ 0, so ΔX·e^{ΔX} ≤ Δ̄X·e^{Δ̄X}.
- With exp results e^{X}(1+κ), |κ| ≤ 8ε + 2⁻¹²⁹, one multiplication by x0 on each side, and |x̂| ≥ |x0|·e^{X_o}·(1 − 8ε − 2⁻¹²⁹)(1 − ε): |x_p − x̂| ≤ |x̂|·[(1 + 2⁻⁴⁸)·Δ̄X·e^{Δ̄X} + 19ε].

SIMPLE derivation: the division and subtraction are correctly rounded on identical operands, and the shock kernel has no `reassoc`, so Y64_p = Y64, Y32 = Y32_o and X_p = X_o bit for bit. Production evaluates x0·(1 + X), x0 + x0·X or fma(x0, X, x0); each is within γ₂·|x0|·(1 + |X|) of the exact value. The oracle form x0·(1 + X) has the same bound.

DIFFERENCE derivation: X_p = X_o. Production evaluates fl(x0 + X) or fma(β, Y32, x0); each is within γ₂·(|x0| + |X_o|) of x0 + X_o. The oracle form has the same bound.

Acceptance (|x_p − x̂| ≤ tol_level):
- LOG: tol_level = |x̂|·[(1 + 2⁻⁴⁸)·Δ̄X·e^{Δ̄X} + 19ε], with Δ̄X = |β|·|Y64|·(2u + 2⁻⁴⁷).
- SIMPLE: tol_level = 2γ₂·|x0|·(1 + |X_o|).
- DIFFERENCE: tol_level = 2γ₂·(|x0| + |X_o|).

Non-historical models: no level comparison (SPOT pattern).

### 5.4 Base value V0 (approximate)

Exact quantities: a = F·X_a, b = K·X_k, V = Q·(a − b), T0_exact = |Q|·(|a| + |b|).

Production sequence, `DiscountingMode.NONE`, single-row segment (OBSERVED: `_build_fx_spots_effective` writes `1.0 / raw_fx_spots` for inverted rows; `_fold_effective_forward_prices` out = F, out *= DF, out *= X for an FX row; `_fold_effective_discount_factors` out = DF, out *= X for an FX row; `_kernel_price_vanilla_forward` pv_k = strike·edf, pv_u − pv_k, qty·(…); `np.add.reduceat` returns a one-row segment unchanged):
- A_p = a + α with |α| ≤ η_a·|a|; B_p = b + β′ with |β′| ≤ η_b·|b|.

| Leg | η (NONE) | Roundings in the leg |
|---|---|---|
| Identity FX | 0 | 0 (DF ≡ 1.0 and multiplication by an exact 1.0 is exact) |
| Direct FX | ε | 1 (FX product) |
| Inverted FX (F5) | (1+ε)² − 1 = 2ε + ε² | 2 (reciprocal, then product) |

- d_p = fl(A_p − B_p) = (A_p − B_p) + σ, with |σ| ≤ ε·(|A_p| + |B_p|) ≤ ε·(1 + η)·(|a| + |b|).
- V_p = fl(Q·d_p) = Q·d_p + ρ, with |ρ| ≤ ε·|Q|·|d_p| ≤ ε·|Q|·(|a − b| + |α| + |β′| + |σ|).
- Per side: |V_p − V| ≤ k(η)·T0_exact, with k(η) = η + 2ε + 2εη + ε²(1 + η) and η = max(η_a, η_b). Identity: 2ε + ε². Direct: 3ε + 3ε² + ε³. Inverted: 4ε + 6ε² + 4ε³ + ε⁴ ≤ 4ε(1 + 2ε). An FMA contraction of `pv_u − strike·edf` removes one rounding, so the bound still holds.

Oracle: the prescribed form (§5.2) has the same per-leg η and the same k(η). The bound uses the larger of the two counts and does not require them to be equal.

Combined: |V0_p − V0_o| ≤ 2k·T0_exact ≤ 8ε(1 + 2ε)·T0_exact.

Computed scale: T0_o is a sum of non-negative terms, so no cancellation occurs. With |a_o| ≥ (1 − η)|a|: T0_o ≥ (1 − ε)²·(1 − η)·T0_exact ≥ (1 − 4ε)·T0_exact.

Acceptance: |V0_p − V0_o| ≤ 8ε(1 + 2ε)/(1 − 4ε)·T0_o ≤ 8ε(1 + 7ε)·T0_o < 32ε·T0_o = 2⁻⁴⁸·T0_o.

tol(V0) = 2⁻⁴⁸·T0_o. VALID for all admitted NODE + NONE FX paths (identity, direct, inverted). Complete cancellation (V = 0, T0 > 0) is covered because the bound depends only on T0. P-N excludes nonzero subnormal float64 intermediates. When T0 = 0, both sides compute exactly 0.

Informational robustness calculation (non-admitted, non-production, not part of this workstream's certification): with one additional DF product per leg (`DiscountingMode.CURVE`), η ≤ (1+ε)³ − 1, k ≤ 5ε(1 + 3ε), T0_o ≥ (1 − 5ε)·T0_exact and the combined error ≤ 10ε(1 + 9ε)·T0_o, which is also below 2⁻⁴⁸·T0_o. This calculation does not admit `DiscountingMode.CURVE`. It does not certify CURVE valuation, CURVE discounting or CURVE price-source risk, and it does not close any part of the deferred CURVE workstream.

### 5.5 FULL_REVAL dV (approximate)

Sequence (OBSERVED, `_write_full_reval_block`): dV64 = fl(V_s,p − V0_p); the assignment into the float32 `PnlGrid` rounds once: dV_grid = fl32(dV64).

Derivation, with D = V_s − V0 exact:
- |dV_grid − dV_o| ≤ u·|dV64| + 2⁻¹⁵⁰ + ε·|V_s,p − V0_p| + 2k·(T_s + T0)_exact + ε·|V_s,o − V0_o|.
- |V_s,o − V0_o| ≤ |dV_o|/(1 − ε); |V_s,p − V0_p| ≤ |dV_o|/(1 − ε) + 2k·T_exact; |dV64| ≤ (1 + ε)·|V_s,p − V0_p|; T_exact ≤ T_o/(1 − 4ε).
- Result: |dV_grid − dV_o| ≤ (u + 3ε + 2uε)·|dV_o| + 8ε(1 + u + 9ε)·(T_s,o + T0_o) + 2⁻¹⁵⁰.

Acceptance: tol_dV_F = 2u·|dV_o| + 2⁻⁴⁸·(T_s,o + T0_o) + φ. VALID for all admitted NODE + NONE FX paths. The T term is absolute, so the bound holds under cancellation. The compared quantity is the production grid value, widened exactly to float64, against the oracle dV_o. P-O excludes float32 overflow.

### 5.6 TAYLOR/DERIVED dV (approximate)

Definition of n, measured on the compiled book that `kernel_taylor_pnl` receives (OBSERVED loop structure):
- Diagonal entries e ∈ [`position_offsets[m]`, `position_offsets[m+1]`): each holds `first_order[e]`, `second_order[e]`, `factor_uix[e]` and evaluates (a + b·x)·x with b = 0.5·`second_order[e]`, formed in the kernel (exact under P-N). Atoms {a, b, x}; exactly two monomials: a·x and b·x·x.
- Cross entries q ∈ [`cross_offsets[m]`, `cross_offsets[m+1]`): each evaluates c·x_i·x_j. Atoms {c, x_i, x_j}; exactly one monomial.
- The accumulator is seeded per position ("Seed: WRITE the accumulator"), so no sum crosses a position boundary.
- d_m, c_m, n_m = d_m + c_m, n = max_m n_m and μ_m = 2d_m + c_m ≤ 2n_m, as defined in §5.1. Explicit zero pairs and zero b are counted, because the kernel evaluates them.

Multiple valuation rows under one UID are permitted (D055; the run groups ingest rows by UID). Before the kernel, the sensitizer pre-aggregates all contributions per (position, rf) and (position, min rf, max rf). The book compiler canonicalises cross pairs, and `_collapse_duplicates` (`DuplicatePolicy.RAISE`) rejects any repeated key. Each compiled position therefore has unique diagonal factors and unique pairs, so its μ_m monomials are distinct and each holds exactly one coefficient atom. UID cardinality does not affect the validity of the bound. The F(p) topology table (section 4) bounds the entries contributed by one admitted NODE + NONE valuation row: at most 3 diagonal factors (F, X_a, X_k) and at most 3 pairs. The contract does not infer the compiled-position bound from UID cardinality: P-n is asserted directly on the compiled book.

Depth: an expression tree of + and × whose expansion is a sum of μ_m distinct monomials, each with one coefficient atom, contains exactly μ_m − 1 addition nodes (a product of two sums would create monomials with two coefficient atoms; none exist). Each monomial has at most 3 atoms, so at most 2 multiplications on its path. Depth ≤ μ_m + 1 ≤ 2n_m + 1 for every evaluation that the declared `parallel_njit` fast-math contract (`contract`, `reassoc`, `nsz`) permits: re-bracketing, factoring a shared displacement and FMA contraction. The written order (OBSERVED) has depth n_m + 2 (diagonal) and n_m + 1 (cross), both ≤ 2n_m + 1. The oracle evaluates the same monomials and is also bounded by depth 2n_m + 1.

Error terms per monomial:
- displacement narrowing: (1 + u)² − 1 = 2u + u² (at most two displacement atoms; P-Δ);
- coefficient perturbation: 2γ₄·Ā (P-C, both sides, absolute through Ā);
- production arithmetic: γ_{2n+1}; oracle arithmetic: γ_{2n+1};
- float32 output: u·|acc| + 2⁻¹⁵⁰.

Derivation: |dV_T,grid − dV_T,o| ≤ [(2u + u²) + u(1 + 3u) + (2γ₄ + 2γ_{2n+1})(1 + 3u)]·Σ τ̄_exact + 2⁻¹⁵⁰. For n ≤ 32: (2γ₄ + 2γ_{2n+1})(1 + 3u) + 4u² ≤ 138ε(1 + 3u) + 2⁻⁴⁶ ≤ 2⁻⁴⁴, so the error is ≤ (3u + 2⁻⁴⁴)·Σ τ̄_exact + 2⁻¹⁵⁰. The oracle computes τ̄ with at most 6 roundings, so Σ τ̄_exact ≤ Σ τ̄_o/(1 − γ₆), and (3u + 2⁻⁴⁴)/(1 − γ₆) ≤ 6u + 2⁻⁴⁷.

Acceptance: tol_dV_T = (6u + 2⁻⁴⁷)·Σ τ̄_o + φ. Preconditions: P-C, P-n, P-Δ.

### 5.7 Reporting dW (approximate)

Sequence (OBSERVED, `convert_scenario_pnl_to_reporting_currency`, NumPy, no FMA): r = `np.where(inv, 1.0/x, x)`; d_r = fl(r_s − r_0); dW = fl(fl(dV·r_s) + fl(V0·d_r)). Rates are strictly positive (`_require`); the identity rate is exactly 1.0.

Shared inputs: the oracle recomputes r_s, r_0 and d_r from the captured shocked FX levels and the base levels with the same correctly rounded operations, so they are bit-identical. dV_o and V0_o stay independent.

Derivation, with e_dV and e_V0 the actual dV and V0 discrepancies:
- |dW_p − dW_o| ≤ (1 + ε)(1 + ε(1 + ε))·(|r_s|·e_dV + |d_r|·e_V0) + (4ε + 2ε²)·(|dV_o·r_s| + |V0_o·d_r|).
- |d_r| ≤ (1 + ε)·|r_s − r_0|, where r_s − r_0 is the exact real difference of the two float64 rates.
- (1 + 3ε)·e_dV ≤ tol_dV and (1 + 4ε)·e_V0 ≤ tol_V0 (§5.4 to §5.6: each actual bound is at most half of its tolerance in every term); (4ε + 2ε²)(1 + ε) ≤ 2⁻⁴⁸.

Acceptance: tol_dW = |r_s|·tol_dV + |r_s − r_0|·tol_V0 + 2⁻⁴⁸·(|dV_o·r_s| + |V0_o|·(|r_s| + |r_0|)), where tol_dV is tol_dV_F or tol_dV_T according to the UID's methodology. r_s ≈ r_0 is covered because d_r is shared.

### 5.8 VaR and ES (approximate)

Tail geometry (OBSERVED, `_compute_lower_tail_parameters`; the oracle reimplements this rule and does not import the helper):
- α = 1.0 − c; m = α·float(S);
- if |m − rint(m)| ≤ 16·eps·max(1, |m|), then m = rint(m);
- f = floor(m); w = m − float(f); var_index = f − 1 if w == 0, else f.

The convention is lower tail, with no interpolation. P-T requires m > 0.

Exactness of w = m − f:
- m integral (including a snapped m): w = m − m = 0 exactly; Sterbenz is not involved.
- f = 0: w = m − 0 = m exactly; Sterbenz is not involved.
- f ≥ 1 and m non-integral: f ≤ m < f + 1 ≤ 2f, so f/2 ≤ m ≤ 2f, and Sterbenz's lemma makes the subtraction exact.
- float(f) is exact for integers below 2⁵³. In every case f + w = m exactly, so the ES weights sum to exactly 1.

VaR (OBSERVED, `_kernel_lower_tail_var_es`, `stable_math_parallel_njit`): quickselect, then `row[var_index]`, with no arithmetic. The partition invariant (elements left ≤ the selected value ≤ elements right) is INFERRED from the kernel; a failure would make the test fail. The order statistic is 1-Lipschitz in the sup norm, and ties do not change its value. tol_VaR = ε_in.

ES (production): `tail_sum = 0.0`, f sequential additions (the first is exact), `+= w·var` when w > 0 (one product and one addition; `contract` may fuse them), `/ tail_mass` (one division).

| Case | Maximum roundings on any path |
|---|---|
| f ≥ 1, w > 0 | f + 1; boundary term 3 |
| f ≥ 1, w = 0 | f |
| f = 0 | 2 |

All cases are ≤ f + 2. The oracle form (§5.2), with any summation order, is also ≤ f + 2 per path. Each side's rounding is ≤ γ_{f+2}·(Σ_{i<f}|p_(i)| + w·|p_(f)|)/m ≤ γ_{f+2}·∥p∥_∞.

Acceptance: tol_ES = ε_in + 2γ_{f+2}·(∥P_o∥_∞ + ε_in).

Per UID: ε_in = ε_uid = ∥tol_dW(uid, ·)∥_∞ and ∥P_o∥_∞ = ∥dW_o(uid, ·)∥_∞.

### 5.9 Portfolio (approximate)

Sequence (OBSERVED): `converted.sum(axis=0)`, a float64 NumPy reduction over M = |U| rows, M ≥ 1 (`MethodologyDeclaration` rejects an empty U). The order is implementation-defined. Any binary tree over M terms has M − 1 additions; applying the absolute model at each node gives |fl(Σ) − Σ| ≤ γ_{M−1}·Σ|x|. The oracle's independent sum obeys the same bound. For M = 1, γ₀ = 0.

Per scenario s:
- |P_p,s − P_o,s| ≤ Σ_uid tol_dW + γ_{M−1}·(Σ|dW_o| + Σ tol_dW) + γ_{M−1}·Σ|dW_o| ≤ ε_P,s, with
- ε_P,s = (1 + 2γ_{M−1})·Σ_uid tol_dW(uid, s) + 2γ_{M−1}·Σ_uid |dW_o(uid, s)|.

Acceptance: tol_VaR,P = ∥ε_P,·∥_∞; tol_ES,P = tol_VaR,P + 2γ_{f+2}·(∥P_o∥_∞ + tol_VaR,P).

### 5.10 Generic determinism (exact comparisons)

Contract: the same semantic inputs give the same result.

| Quantity | Status | Basis |
|---|---|---|
| UID sets, axes, partition, rf keys, uix bindings | exact | integer identity |
| Batch-size and repetition invariance | bitwise | level generation: `apply_linear_terms_to_base_levels` notes, bitwise reproducible across batch partitions within one environment (D088 §13.6; documented); FULL_REVAL: "each column is priced independently" (`_write_full_reval_block`; OBSERVED); Taylor path: per-column computation (INFERRED); evidence: existing SPOT test `test_batch_size_and_repetition_do_not_move_the_certified_numbers` |

### 5.11 Proxy V1 target-coordinate / β-separation invariant (F4, Run A and Run B)

This is a mathematical test of the sensitivity coordinate. It is not a determinism or batch-invariance test.

Setup: Run A authors A ← 0.95·D; Run B authors A ← 1.00·D (A = rf of `WTI_CRUDE_FWD[M+0]`, D = driver `BRENT_CRUDE_FWD[M+0]`, LOG_RETURN). Base state, target rf, economic position and dependency graph are identical.

| Element | Status | Requirement | Basis |
|---|---|---|---|
| Control: driver shock vector Y32 | bitwise | The captured `shocks` column bound to the A row is equal in Run A and Run B (B-β0). This captured equality, not the seed, is the logical basis of the β proof. A configured seed is a setup detail only. | `_kernel_fused_h_day_shock` reads float64 history and uses no random state (OBSERVED); asserted directly |
| Sensitizer coefficients | bitwise | All eight `BlockSensitivities` arrays are equal between Run A and Run B: `position_uids`, `factor_rf_ids` (keys and order), `first_order`, `second_order`, `cross_position_uids`, `cross_factor_rf_ids_i`, `cross_factor_rf_ids_j`, `cross_second_order` (C18a) | The `Sensitizer` protocol takes `(block, uix_mapper, market_data, scenario)` and no proxy model (OBSERVED). Both runs give it identical inputs. The computation is deterministic per row, and lexsort/reduceat is deterministic (INFERRED). Asserted directly. |
| Target shocked levels | approximate (tol_level LOG, §5.3) | x_A,s(Run A) versus T⁻¹(x_A,0, fl64(0.95·Y32_o,s)); x_A,s(Run B) versus T⁻¹(x_A,0, fl64(1.00·Y32_o,s)); the two runs differ on the asserted non-degenerate scenario s* (B-β1, B-β2) | `_kernel_apply_linear_terms` consumes Y32 and forms X = fl(β·Y32) (OBSERVED) |

Consequence: β enters target shock construction exactly once, applied to the production shock Y32. The emitted coefficient is ∂PV/∂x_A in both runs and is never multiplied by β. Bug coverage:
- β omitted: B-β1.
- β squared: B-β1.
- β inserted into a sensitivity coefficient: C18a, D-F4.
- driver level used instead of target level: B-β1 (level), C18b (coefficient), C18c (key), D-F4 (displacement).
- target's own history used instead of the proxy driver: B-β3.
- different driver shocks compared: B-β0.

### 5.12 Sensitivity-level tolerances (C-tests, float64 only)

- Coefficients versus formula: 2⁻⁴⁹·Ā_i (at most 4 roundings on each side).
- C11 numeric cross-check, F·DF·X recomputed versus `ValuationState.effective_forward_prices`: 2⁻⁵⁰·|F·DF·X| (at most 2 roundings on each side).
- C12 float64 Taylor-from-rows versus float64 repricing: 2⁻⁴⁸·(Σ τ̄ + T_s + T0).
- C17 convention transforms: 2⁻⁵⁰·|value|.

No comparison that passes a float32 boundary is described as ulp agreement.

### 5.13 Final tolerance table

| # | Quantity | Class | Acceptance bound | Preconditions and assumptions |
|---|---|---|---|---|
| 1a | Historical target level, LOG | approximate (CONDITIONAL) | \|x̂\|·[(1 + 2⁻⁴⁸)·Δ̄X·e^{Δ̄X} + 19ε], Δ̄X = \|β\|·\|Y64\|·(2u + 2⁻⁴⁷) | A-L1, A-L2, P-Y, P-L1 to P-L3 |
| 1b | Historical target level, SIMPLE | approximate | 2γ₂·\|x0\|·(1 + \|X_o\|) | A-L2, P-Y, P-L1 to P-L3 |
| 1c | Historical target level, DIFFERENCE | approximate | 2γ₂·(\|x0\| + \|X_o\|) | A-L2, P-Y, P-L1 to P-L3 |
| 2 | V0 | approximate | 2⁻⁴⁸·T0_o | P-N; admitted NODE + NONE FX paths |
| 3 | FULL_REVAL dV | approximate | 2u·\|dV_o\| + 2⁻⁴⁸·(T_s,o + T0_o) + φ | P-N, P-O |
| 4 | TAYLOR/DERIVED dV | approximate | (6u + 2⁻⁴⁷)·Σ τ̄_o + φ | P-N, P-O, P-Δ, P-C, P-n |
| 5 | D081 dW | approximate | \|r_s\|·tol_dV + \|r_s − r_0\|·tol_V0 + 2⁻⁴⁸·(\|dV_o·r_s\| + \|V0_o\|·(\|r_s\| + \|r_0\|)) | shared r_s, r_0, d_r; rates > 0 |
| 6 | Per-UID VaR | approximate | ε_uid = ∥tol_dW(uid, ·)∥_∞ | identical tail geometry; P-T |
| 7 | Per-UID ES | approximate | ε_uid + 2γ_{f+2}·(∥dW_o(uid, ·)∥_∞ + ε_uid) | identical tail geometry; P-T |
| 8 | Portfolio dW, per scenario | approximate | ε_P,s = (1 + 2γ_{M−1})·Σ_uid tol_dW + 2γ_{M−1}·Σ_uid \|dW_o\| | M = \|U\| ≥ 1 |
| 9 | Portfolio VaR | approximate | ∥ε_P,·∥_∞ | identical tail geometry; P-T |
| 10 | Portfolio ES | approximate | ∥ε_P,·∥_∞ + 2γ_{f+2}·(∥P_o∥_∞ + ∥ε_P,·∥_∞) | identical tail geometry; P-T |

Exact by structural contract: integer ids, factor keys, axis partitions, captured Y32 equality between the F4 runs, sensitizer coefficient equality between the F4 runs (§5.10, §5.11). Analytically exact: NODE + NONE SECOND_ORDER_FULL with an ABSOLUTE or RELATIVE book (section 4). Every production-versus-oracle comparison above is floating-point approximate.

---


## 6. Proposed decisions (not yet in `DECISIONS.md`)

### D102 - VANILLA_FORWARD_NODE risk admission boundary (WS-A; OPEN; LOCKED after WS-B)

```text
Related: D045, D050, D055, D061, D063, D081, D084, D085, D086, D088, D101.
Scope: source-specific admission for UIDs with rows in the vanilla_forward block. All global run
and methodology contracts (D081 composition, P-1 SUPPLIED refusal, D084, D085, D086) apply in
addition and unchanged.
1. BlockType.VANILLA_FORWARD confers no risk admission. GlobalIdNamespace is never a source
   criterion.
2. Source binding: the forward source rows of UID u are the ingest rows with batch.uids == u
   and family FORWARD (INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY). IngestBatch and D055 permit
   repeated UIDs; no single-row assumption is made.
3. Provenance (fail closed): if any forward source row of such a UID has an instrument other
   than VANILLA_FORWARD_NODE_ID, including a UID that mixes forward instruments, the run raises
   CompositionContractError (message prefix "D102.3"). This applies to every methodology
   partition, before _row_aggregation.
4. Methodology: a UID with a vanilla_forward row may participate in the NODE forward path only
   through full_reval_uids, in addition to all global contracts. In taylor_supplied_uids or
   taylor_derived_uids it is refused (message prefix "D102.4") until D103 is LOCKED.
5. VANILLA_FORWARD_CURVE is outside this decision (D050, D061).
6. VANILLA_FORWARD_NODE is not production complete while clause 4 refuses TAYLOR/DERIVED.
Evidence: WS-A tests; WS-B production-run evidence.
```

### D103 - VANILLA_FORWARD_NODE TAYLOR/DERIVED capability (WS-C drafts; LOCKED after WS-D)

```text
Amends: D102 clause 4. All global contracts and D102.3 apply in addition.
1. sensitize_vanilla_forward_block, registered for BlockType.VANILLA_FORWARD, implements the
   dependency and mathematical contracts of this document. Registration is not admission.
2. Coordinate: target-factor LEVELS. Proxy V1 governs target shock construction and
   target/driver topology; the sensitizer never applies a loading.
3. Inverted FX (invert = 1 on either FX leg of a forward row) is mathematically defined but outside
   this admission contract (D084 analogue). The sensitizer raises NotImplementedError before any
   kernel arithmetic. The run refuses a UID in taylor_derived_uids with such a row by raising
   CompositionContractError (message prefix "D103.3") before _row_aggregation.
4. A UID in taylor_derived_uids with a vanilla_forward row may participate only when, in addition
   to D102.3, D103.3 and the global contracts:
   (a) every non-forward row of the UID is a SPOT row admitted under D084;
   (b) context.discounting_mode is DiscountingMode.NONE.
   A violation raises CompositionContractError (message prefix "D103.4") before _row_aggregation.
5. taylor_supplied_uids: unchanged global refusal (P-1).
6. After LOCK: FULL_REVAL in the admitted modes; TAYLOR/DERIVED in DiscountingMode.NONE.
   Deferred: TAYLOR/DERIVED in DiscountingMode.CURVE (D104); every VANILLA_FORWARD_CURVE capability.
```

---

## 7. Workstreams

### WS-A: admission boundary

- Execution boundary (OBSERVED):
  - `run_scenario_var_es` is the only public run entry (`__all__ = ["derive_run_factor_axis", "run_scenario_var_es"]`).
  - Before the composition checks there are only input validations that raise (`batch_size`, `_validate_confidence_level`, parametric binding, reproducibility).
  - The body then runs straight-line: `_owned_axis_inputs`, `methodology.blocks()`, `_require_block_wiring(...)`, `_require_taylor_capable_spot(...)`, `_row_aggregation(...)`. There is no methodology branch and no early return.
  - `_require_block_wiring` and `_require_taylor_capable_spot` may raise. A statement placed after `_require_taylor_capable_spot` executes on every run that passes them, FULL_REVAL-only or mixed.
- Production change: `_require_forward_capability(*, context, methodology)` in `src/riskquant/solutions/risk_run/api.py`, an unconditional statement directly after `_require_taylor_capable_spot`.
  - It returns when `context.portfolio.vanilla_forward is None`.
  - It does not return early when no TAYLOR UID exists, because D102.3 applies to every partition.
  - Order: D102.3, then D102.4.
  - UID binding as in section 3. `_settlement_ccy_ids` already uses the same UID-keyed binding pattern.
- Prohibited: changes to SPOT rules, `_require_block_wiring`, P-1, D085, sensitizers, NODE FULL_REVAL behaviour.
- Files: the above; new `tests/solutions/risk_run/test_vanilla_forward_taylor_admission.py`; `DECISIONS.md` (D102 OPEN).
- Exit: A-tests and P-tests green; no new failure against the base; D102 OPEN with WS-A evidence.

### WS-B: NODE FULL_REVAL evidence

- Objective: raw input → ingest → compile → market-data assembly → valuation state → `VanillaForwardBlock` → `_write_full_reval_block` → `PnlGrid` → D081 conversion → `compute_var_es` → `RiskRun`, against the independent oracle.
- Production change: none. A defect stops the stream for a decision.
- Files: new `tests/solutions/risk_run/_fwdfix.py` (built on `_mixedfix.py`); new `tests/solutions/risk_run/test_vanilla_forward_node_production.py`.
- Book (raw frame; hostile UIDs and rf ids; SPOT slots present; every forward ingest row `VANILLA_FORWARD_NODE_ID`, asserted; the preconditions of §5.0 asserted):
  - F1: exact NODE (LABEL; test-owned extension, precedent `tests/config/proxy_model/conftest.py`), delivery D1, long, identity FX.
  - F2: exact NODE (LABEL), delivery absent (admissible for exact NODE only), short.
  - F3: relative `BRENT_CRUDE_FWD[M+0]` (rf_D), EUR settlement (direct USDEUR on both legs).
  - F4: relative `WTI_CRUDE_FWD[M+0]` (rf_A), EUR settlement (direct USDEUR on both legs, so V_X = Q·(x_A·DF_a − K·DF_k) depends on the forward level). It is the target of the proxy term described below. The fixture asserts |x_A,0 − x_D,0| > 10·tol_level.
  - F5: USD underlying, EUR strike, USD settlement (inverted strike-leg FX).
- Proxy runs:
  - Authored terms: exactly one CONTAINER term, copied from the canonical frame (`ingest_fixtures/config/risk_proxy_model/canonical_risk_proxy_model.py`): `WTI_CRUDE_FWD ← β·D`, with D the `BRENT_CRUDE_FWD` pillar and driver chart LOG_RETURN, expanded by EXACT pairing.
  - Run A: β = 0.95. Run B: β = 1.00. Everything else is identical. The logical control is the same captured driver shock vector Y32 (B-β0, §5.11), not the seed.
  - Separate runs are required: D088 §1 gives one compiled row per target, and D088 §6 merges equal (d, τ) terms by summation. The canonical frame itself demonstrates the merge: AAPL_EQ 0.3 + 0.2 = 0.5.
  - The world holds WTI's own history column with returns that differ from β·Y64 of the driver on the asserted scenario.
- Scenario model for β tests: historical, horizon 1.
- Mode: `DiscountingMode.NONE`. A CURVE sub-run only under OD-1.
- Exit: B-tests green; D102 Evidence filled; owner LOCKS D102.

### WS-C: NODE forward sensitizer

- Files:
  - new `src/riskquant/sensitivities/sensitizers/vanilla_forward.py` (layer 2: unpack index tables, inverted-FX refusal, kernel call, rf expansion, pre-aggregation, emission);
  - new `src/riskquant/sensitivities/sensitizer_kernels/vanilla_forward.py` (layer 1: per-row primitive values and ∂PV/∂primitive, `riskquant.jit`);
  - registration in `src/riskquant/sensitivities/sensitizer_registry.py`;
  - new `tests/sensitivities/test_vanilla_forward_sensitizer.py` and `tests/sensitivities/test_vanilla_forward_full_revaluation_reconciliation.py`;
  - update the VANILLA_FORWARD assertion in `tests/sensitivities/test_spot_sensitizer.py::test_unauthored_block_types_fail_fast_and_are_named` and keep the option assertion;
  - `DECISIONS.md` (D103 OPEN).
- Ordering: unpack index tables → read `fx_effective_pairs[:, 1]` for every used leg → raise `NotImplementedError` if any equals 1 → only then gather level arrays and call the kernel.
- Pre-aggregation: vectorised lexsort over (uid, rf) and (uid, lo_rf, hi_rf), then `np.add.reduceat`. No Python loop over trades.
- Numerical preconditions: P-C and P-n (§5.0) hold for the certified positions; P-n is asserted on the compiled book.
- Refusal set: inverted FX only (D103.3). D084 requires a separate decision, analytic partials, a finite-difference oracle and a remainder test before inverted-FX TAYLOR admission.
- Not added, with reasons:
  - F namespace check: not provenance; a non-axis id fails in `_require_declared_axis`.
  - DF ≠ 1.0 check: not a mode discriminator; the methodology boundary decides.
  - Non-finite check: NaN/inf propagates without raising, and `BlockSensitivities.__post_init__` rejects non-finite coefficients.
- Prohibited: pillar rows, a source-kind column, price-curve logic, Sensitizer protocol changes, `BlockSensitivities` changes, SPOT changes.
- Exit: C-tests green; D103 OPEN with WS-C evidence.

### WS-D: TAYLOR/DERIVED admission (NONE)

- Production change: the D103.3 refusal and the D103.4 admitted set inside `_require_forward_capability`.
- Files: `api.py`; extend the WS-A and WS-B test files; `DECISIONS.md` (D103 Evidence).
- Prohibited: CURVE-mode admission, SUPPLIED admission, SPOT changes.
- Exit: D-tests green; owner LOCKS D102 and D103.

WS-B and WS-C may be developed in parallel. Neither changes this contract.

---

## 8. Test matrix

### A-tests and P-tests (`test_vanilla_forward_taylor_admission.py`)

Spy set S: the `spy` pattern of `test_methodology_composition.py` (recording `build_scenario_generator`, `allocate_pnl_grid`, `build_market_data_matrix_from_flat_levels`) plus a recorder on `api._row_aggregation`. "No downstream work" means zero recorded calls in S.

| ID | Case | Expected |
|---|---|---|
| A1 | Forward UID in taylor_derived_uids with a caller-built certified book | "D102.4"; S empty |
| A2 | A1 with the check disabled (D059) | Book priced; VaR moves |
| A3 | Valid NODE FULL_REVAL-only run | Admitted; `_row_aggregation` recorded; run completes |
| A4 | Forward in full_reval_uids beside SPOT in taylor_derived_uids; SPOT book from a SPOT-only sub-portfolio (D084.4) | Admitted; run completes |
| A5 | SPOT-only run | Admitted; SPOT suites unchanged |
| P1-F | Inconsistent-provenance rejection: valid NODE context; one forward UID's ingest row relabelled `VANILLA_FORWARD_CURVE_ID` (`dataclasses.replace` on the batch; `object.__setattr__` on the frozen context); all UIDs in full_reval_uids | "D102.3"; S empty; the rf is in `GlobalIdNamespace.NODES`; block type unchanged |
| P1-D | As P1-F with the UID in taylor_derived_uids | "D102.3" (checked before D102.4); S empty |
| P2 | Mixed forward sources under one UID: append to a valid NODE batch one extra ingest row with the same UID and `VANILLA_FORWARD_CURVE_ID` (all SoA fields concatenated; set on the frozen context); FULL_REVAL | "D102.3"; S empty |
| P3 | Homogeneous multi-row: a context compiled from a hand-built batch with two `VANILLA_FORWARD_NODE_ID` rows of one UID; FULL_REVAL | Not refused by D102; `_row_aggregation` recorded |

P1 and P2 are inconsistent-provenance rejection tests. They prove that block type and RF namespace do not establish source, and that inconsistent or mixed source metadata is refused. They do not certify CURVE compiler construction, CURVE valuation, price-curve production, interpolation or CURVE sensitivities.

### B-tests (`test_vanilla_forward_node_production.py`)

- B-L0: the test-owned A-L1 helper (§5.3) runs before any level comparison; the §5.0 preconditions are asserted.
- Axes authoritative and hostile; levels by factor name (historical; tol_level of §5.3).
- Per-UID V0, dV, dW; per-UID and portfolio VaR and ES; every admitted scenario model; hostile versus sorted worlds; batch-size and repetition invariance (bitwise); manifest context; `reproduce=`.
- Detectors: forward row excluded or zeroed; node not shocked (a scenario where only F moves); two forward rfs swapped (distinct shocks); reporting conversion lost; aggregation omission.

β tests, with s* a historical scenario where the driver return satisfies |Y64_D,s*| ≥ 1e-3 (asserted). The fixture asserts non-degeneracy, with Y32_o the driver's reproduced float32 shock at s*:
- |T⁻¹(x_A,0, fl64(0.95·Y32_o)) − T⁻¹(x_A,0, Y32_o)| > 2·tol_level;
- |T⁻¹(x_A,0, fl64(0.95·Y32_o)) − T⁻¹(x_A,0, fl64(0.9025·Y32_o))| > 2·tol_level;
- |T⁻¹(x_A,0, fl64(0.95·Y32_o)) − T⁻¹(x_D,0, fl64(0.95·Y32_o))| > 2·tol_level.

| ID | Assertion | Bug caught |
|---|---|---|
| B-β0 | The captured `shocks` column bound to the WTI row (spy on `api.apply_linear_terms_to_base_levels`, kwargs `shocks` and the term column from `term_column_uix`) is bitwise equal in Run A and Run B | Establishes the §5.11 control: identical driver shock vector Y32 |
| B-β1 | Run A: x_A,s* versus T⁻¹(x_A,0, fl64(0.95·Y32_o)) within tol_level (LOG) | Bug 1 (β omitted gives T⁻¹(x_A,0, Y32_o)); Bug 2 (β² gives T⁻¹(x_A,0, fl64(0.9025·Y32_o))); Bug 4 in level generation (T⁻¹(x_D,0, ·)) |
| B-β2 | Run B: x_A,s* versus T⁻¹(x_A,0, Y32_o) within tol_level (LOG); Run A and Run B rf_A levels differ; rf_D levels are bitwise equal | β affects only the target |
| B-β3 | x_A,s* differs from T⁻¹(x_A,0, Y32_o of WTI's own history) by more than 2·tol_level | Target shocked from its own history instead of the proxy topology |

### C-tests

| ID | Case | Expected |
|---|---|---|
| C1-C5 | Each section 4 topology, long and short | Coefficients equal the formula within 2⁻⁴⁹·Ā |
| C6 | Q = 0 (low-level structural; not an ingest fixture) | Full topology, zero coefficients; compiles at SECOND_ORDER_FULL |
| C7a | Exact NODE (LABEL) row | The F row key equals the ingest-resolved `underlying_ids` value of the row and `market_id_universe[efp_effective_quads[·, 0]]`. The same row ingested with delivery present, absent, and at two different delivery dates gives the same F key |
| C7b | Relative NODE (OFFSET_M, CALENDAR_PERIOD) row | The F row key equals the ingest-resolved `underlying_ids` value of the row and `market_id_universe[efp_effective_quads[·, 0]]` |
| C7c | Relative NODE at two as-of dates whose resolved absolute contracts differ (asserted), same relative name, delivery inside the first contract's period | Same rf_id at both as-of dates (D077.4: fixed by name); `build_risk_factor_name_mapping_for_as_of_date` maps it to two different absolute contracts; the delivery is accepted at the first as-of and refused (`PrettyKeyError`, D065) at the second |
| C8a | Exact NODE, delivery present versus absent, `DiscountingMode.NONE` | Identical sensitizer output (exact-NODE property only) |
| C8b | Relative NODE, valid delivery | Admitted at ingest; sensitizer output as for any NODE row |
| C8c | Relative NODE, delivery absent | Reuse `test_relative_node_without_delivery_fails_at_ingest` and `test_relative_forward_missing_delivery_fails_at_ingest`: `PrettyKeyError` at ingest; no new test |
| C8d | Relative NODE, delivery outside the resolved period; unsupported roll method | Reuse `test_forward_delivery_outside_resolved_contract_is_rejected` and `test_unsupported_relative_roll_fails_closed`: `PrettyKeyError` at ingest; no new test |
| C9 | Hostile rf ids | Global rf ids, never uixs |
| C10 | invert = 1 on the asset leg, then the strike leg (D103.3) | `NotImplementedError`; a spy replacing the kernel symbol in the sensitizer module records zero calls |
| C11 | Index binding (semantic) | Per row: `price_uix`, `curve_uix`, `date_uix` and FX index equal those of the production fold tables; rf_F equals the rf of the raw underlying name; rf_X equals the configured pair rf (integer equality). Numeric cross-check within 2⁻⁵⁰·\|F·DF·X\|; not bitwise |
| C12 | Float64 reconciliation, ABSOLUTE book; levels from LOG_RETURN, SIMPLE_RETURN and DIFFERENCE charts | Taylor from emitted rows versus repricing within 2⁻⁴⁸·(Σ τ̄ + T_s + T0); without the (F, X_a) pair, the residual equals Q·DF_a·ΔF·ΔX_a within the same tolerance |
| C13 | X_a = X_k | One X row; second_order 0; no i = j row; one (F, X) pair |
| C14 | Synthetic two-primitive same-rf product (hand-built tables) | second_order = 2·P^{¬ab} |
| C15 | Empty block | Empty valid `BlockSensitivities` |
| C16 | Emission uniqueness | No repeated (uid, rf) or (uid, lo, hi) |
| C17 | RELATIVE and LOG conventions through `build_sensitivity_book_spec` | RELATIVE: V_y = V_x·x0, cross V_{xi xj}·x0_i·x0_j, reproduces repricing within the C12 tolerance. LOG: V_yy = V_x·x0; the residual equals the analytic third-order remainder within 2⁻⁴⁸·scale; not exact |
| C18 | Sensitizer output for the Run A and Run B contexts (§5.11) | (a) all eight `BlockSensitivities` arrays bitwise equal, including factor keys and order and cross keys (catches Bug 3: β in coefficient); (b) the F4 X-row coefficient equals Q·(x_A,0·DF_a − K·DF_k), not the same expression with x_D,0 (catches Bug 4 in coefficients); (c) F4 rows are keyed by rf_A; no F4 row is keyed by rf_D |

### D-tests (extend the WS-A and WS-B files)

- P-n and P-Δ are asserted on the compiled book before any DERIVED comparison (§5.0).
- Mixed FULL_REVAL / TAYLOR/DERIVED book at SECOND_ORDER_TRUNCATED against the declared-polynomial oracle (pattern `_mixedfix.l4_taylor`); tolerances tol_dV_T and tol_dW.
- SECOND_ORDER_FULL forward rows versus FULL_REVAL rows: tol_dV_T + tol_dV_F.
- D-F4 (Runs A and B): the DERIVED P&L of F4 equals the oracle polynomial built from ∂PV/∂x_A (no β) and the run's own captured Δx_A. It does not equal the polynomial built from Δx_D (non-degeneracy asserted). This catches Bug 3 and Bug 4 in the Taylor path.
- Refusals with D059 proof: inverted leg ("D103.3"); `DiscountingMode.CURVE` ("D103.4"); P1-D and P2 after D103 ("D102.3"); a UID with a forward row and an option row in taylor_derived_uids ("D103.4").
- D086 for forward books: FIRST_ORDER with a cross refused; TRUNCATED with a cross refused; FULL missing a zero pair refused.

Upstream cases (existing tests; reused, not duplicated):
- Exact NODE: `test_exact_node_ingests_with_and_without_delivery`, `test_delivery_never_selects_the_exact_node`, `test_asset_leg_date_is_delivery_or_settlement`, `test_omitted_delivery_values_exactly_as_delivery_equal_to_settlement`, `test_absent_delivery_is_admitted_only_for_label_price_points` (`tests/market_data/remediation/test_exact_node_forward_delivery.py`).
- Relative NODE: `test_relative_node_without_delivery_fails_at_ingest`, `test_unsupported_relative_roll_fails_closed` (same file); `test_relative_forward_missing_delivery_fails_at_ingest`, `test_forward_delivery_outside_resolved_contract_is_rejected`, `test_forward_delivery_inside_resolved_contract_is_accepted` (`tests/market_data/remediation/test_contract_failure_paths.py`); `test_resolve_offset_m_calendar_period` (as-of-dependent OFFSET resolution).
- A target date before the as-of raises `ValueError` (D079) in `build_market_data_matrix`.

---

## 9. Status matrix

| Capability | Implemented | Registered | Admitted | Evidenced | Governing | Deferred |
|---|---|---|---|---|---|---|
| NODE valuation | yes | yes | yes | yes | D050, D052, D101 | none |
| NODE FULL_REVAL, NONE | yes (generic) | n/a | implicit today; explicit after WS-A | after WS-B | D085.2, D102 | none |
| NODE FULL_REVAL, CURVE | yes (generic) | n/a | after WS-A | only under OD-1 | D080, D102 | per OD-1 |
| NODE sensitizer | after WS-C | after WS-C | n/a | after WS-C | D103 | pillar rows (D104) |
| NODE TAYLOR/DERIVED, NONE | no | no | unsafe today (caller-built book priced); refused after WS-A; admitted after WS-D | after WS-D | D102, D103 | none |
| NODE TAYLOR/DERIVED, CURVE | no | no | refused | n/a | D103.4(b) | D104, OD-1 |
| TAYLOR/SUPPLIED | n/a | n/a | refused (P-1) | n/a | `_require_block_wiring` | outside scope |
| CURVE valuation | no | unregistered | no | no | D050 | CURVE workstream |
| CURVE risk | no | no | refused (D085.2 incidental; D102.3 explicit) | P1, P2 (provenance only) | D102 | CURVE workstream |
| Price-curve production | no | n/a | refused | builder raise | D061 | CURVE workstream |

---

## 10. Deferred CURVE boundary

Prohibited in this workstream:
- `VANILLA_FORWARD_CURVE` valuation;
- price-curve production or interpolation;
- a source-kind column;
- curve-pillar sensitivity expansion;
- generic curve dependency machinery;
- `BlockSensitivities` changes; SPOT changes;
- admission keyed on `BlockType` or `GlobalIdNamespace`;
- compatibility layers; speculative abstractions.

Before any CURVE risk admission, future work must establish:
- a source-provenance contract at the run boundary;
- the D061 price-curve point and date contract;
- CURVE valuation;
- the CURVE dependency expansion;
- the treatment of an rf that is both a scalar node and a price pillar on one run axis;
- its own admission decision.

A NODE/CURVE equality check is at most a later validation, after those contracts exist.

## 11. Open policy question (deferred, not blocking)

OD-1: is `DiscountingMode.CURVE` in the target envelope? It decides the WS-B CURVE sub-run and whether D104 (NODE TAYLOR/DERIVED under CURVE discounting) opens. WS-A, and WS-B to WS-D in `DiscountingMode.NONE`, do not depend on it.

## 12. Implementation order

1. WS-A
2. WS-B
3. WS-C (may be developed in parallel with WS-B)
4. WS-D
5. Independent validation and audit; owner LOCKS D102 and D103
