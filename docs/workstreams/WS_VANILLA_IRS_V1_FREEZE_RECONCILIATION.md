# Workstream: VANILLA_INTEREST_RATE_SWAP V1 — Repository-Truth Freeze Reconciliation

Path: `docs/workstreams/WS_VANILLA_IRS_V1_FREEZE_RECONCILIATION.md`
Date: 2026-10-04
Scope: valuation engine only (FULL_REVAL). No risk-engine, ES, Taylor, Proxy, PNIM work.
Mode: read-only reconciliation. No file in `src/`, `tests/` or `docs/DECISIONS.md` was modified.
Evidence base: generated repository context (`context_riskquant_*.txt`, `project_structure.txt`, `CLAUDE.md`). `docs/DECISIONS.md` is listed in `project_structure.txt` but its content was not available to this reconciliation. Test execution was not available: every test statement below is OBSERVED from source, never a PASS claim.

---

## A. Executive Verdict

**FREEZE READY PENDING OWNER APPROVAL.**

No repository fact contradicts the candidate contract C.1–C.18. Three repository facts require a stated implementation rule (Sections C, D) and are recorded as MATCH WITH REQUIRED CHANGE; none widens V1 or changes the economics. O-5 is NOT VERIFIED because `docs/DECISIONS.md` content was not available; it is an authority gap, not a contradiction. Implementation remains blocked until O-1 to O-5 are closed.

---

## B. Authority Reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| O-1 Floating kernel: forward rate → period factor | OWNER APPROVAL REQUIRED | `valuation/pricer_kernels/floating_cashflow.py` computes `N·α·ε·(f+m)·EDF` and documents `forward_rates` as "annualised SIMPLE on the same basis as accrual_factor". `valuation/blocks/discounted_cashflow_block.py::FloatingCashflowBlock.forward_rate_uix`. `market_data/state.py::ValuationState.forward_rates` default `(0,0)`, no producer (`market_data/compile.py`), pinned by `tests/valuation/oracle/test_multi_row_trade_identity.py::test_forward_rate_state_field_has_no_producer`. Callers of `forward_rate_uix` outside the block: one contract test (`tests/.../test_block_and_dispatch_contracts`-style constructor at tests context line ~62518). No production consumer. | Owner approves kernel formula `N·ε·(R + m·α)·EDF`, field rename `forward_rate_uix → forward_period_factor_uix`, state rename `forward_rates → forward_period_factors`, state default `(0, batch_size)`. |
| O-2 Economic contract C.1–C.18 | OWNER APPROVAL REQUIRED | Section C matrix: no CONTRADICTION. | Owner records C.1–C.18 (with the three required-change notes) as a numbered D-series decision. |
| O-3 Shared `DayCount` enum | OWNER APPROVAL REQUIRED | `config/enums/financial_conventions.py::DayCount = {ACT_365F, ACT_360, DC_30_360="30_360"}`. Consumers: `VALID_DAY_COUNTS` (frozenset for validation, no observed caller outside the module's `__main__` review printer), the `__main__` printer. No instrument, curve-meta row, hash, or serializer references `DayCount` or `"30_360"` except the defective IRS fixture `ingest_fixtures/portfolio/vanilla_irs.py`. `THIRTY_360_BOND` does not exist under any name. | Owner approves: add `THIRTY_360_BOND`, `THIRTY_E_360`; deprecate `DC_30_360` (keep the member until the fixture is rewritten; IRS validator rejects the string regardless). Blast radius: zero production callers. |
| O-4 Shared compiler/portfolio ABI | OWNER APPROVAL REQUIRED | Section D. Callers of `CompilerFn`/`FIELD_NAMES`: `portfolio/compilers/registry.py` (definition, `_register`), `portfolio/engine.py::compile_portfolio_from_ingest_batch` (sole consumer). Tests touch `COMPILERS[...]` only (3 sites), never `FIELD_NAMES`. | Owner approves the ABI change with acceptance evidence E-1..E-10 (Section G) as lock condition. |
| O-5 `DECISIONS.md` verification | NOT VERIFIED | `project_structure.txt` lists `docs/DECISIONS.md`; content not in the attached context. Decision ids relied on (code citations only): D045/VAL-04/05/06/15/16, D061, D062, D063, D064/AUD-02, D066, D067, D069, D079, D080, D081, D101. | Owner attaches `docs/DECISIONS.md`; confirm each id's text matches the code citation; confirm no decision forbids a `CompilerFn` return-type change, a `ValuationState` field rename, or a `DayCount` extension. |

---

## C. Contract-to-Repository Matrix

| Contract | Repository evidence | Status | Finding |
|---|---|---|---|
| C.1 Instrument scope | `config/enums/instrument_type.py::VANILLA_IRS = 5`, family RATES, topology MTM (`portfolio/instruments.py`). `BlockType` docstring decomposes IRS into FIXED + FLOATING. No stub/EOM/amortisation primitive exists. | MATCH | Scope is additive; nothing in the repository presumes a wider IRS. |
| C.2 Sign | `FixedCashflowBlock`/`FloatingCashflowBlock.pay_receive_mult` int8 pay=-1/receive=+1, notional unsigned (D045/VAL-16). `SwapDirection(PAY_FIXED=-1, RECEIVE_FIXED=+1)`. Kernels apply `ε` exactly once. `ValuationResult`/`_row_aggregation` sum rows without sign. | MATCH | No second sign carrier downstream. |
| C.3 Unadjusted schedule | `utils/business_days.py::add_months_ord`: `total = y*12 + (m-1) + n`; `day = min(d.day, monthrange)`. Computes from the given date, so calling it from `effective` with `k·f` preserves day-of-month across clamps. | MATCH | Contract clamp rule equals the implementation when called from `effective` each time, never cumulatively. Exact-division test is new edge logic. |
| C.4 Adjusted accrual | `adjust_ord(convention: str, ...)` implements FOLLOWING, MOD_FOLLOWING, PRECEDING, MOD_PRECEDING, NONE on ordinals; scalar, string convention. | MATCH WITH REQUIRED CHANGE | A vectorised int-id variant `adjust_ords` is required (strings may not enter the schedule compiler; scalar loops over ~10^5–10^6 dates are an anti-pattern). Equality with `adjust_ord` on a dense sweep is a required test. |
| C.5 Payment date | `business_day_add_ord(n_days=0)` **rolls FOLLOWING onto the next business day** (docstring: spot-lag semantics). Contract requires `bd_add(d,0) = d`. | MATCH WITH REQUIRED CHANGE | Under BDC ≠ NONE, `E_i` is already a business day, so the two agree. Under BDC = NONE (lag forced 0) the existing helper would silently roll a non-business `E_i`. Rule: the IRS schedule compiler never calls `business_day_add_ord`; the new `business_day_add_ords` defines `n = 0` as identity. The existing scalar helper is unchanged for its spot-date callers. |
| C.6 Fixing date | No `business_day_sub_ord` exists. `business_day_add_ord` raises on `n < 0`. | MATCH WITH REQUIRED CHANGE | `business_day_sub_ords` is a new primitive (`n = 0` identity; `n ≥ 1` the n-th business day strictly before). Admission check on `X_0` only; later `X_i` follow from monotonicity (Section F). |
| C.7 Floating convention | `RateFixingConvention`, `CompoundingType` exist, unused. No fixing table, no compounding kernel. | MATCH | New enum `FloatRateConvention{TERM_SIMPLE_IN_ADVANCE}` replaces using the two unused enums. |
| C.8 Spread | `FloatingCashflowBlock.spread` float64 exists; kernel applies `(f + m)` today. | MATCH WITH REQUIRED CHANGE | Becomes `(R + m·α)` under O-1. Block field unchanged. |
| C.9 Day count | `DayCount` has no numeric implementation anywhere (searched `year_frac`, `accrual`, `day_count` across all contexts). `DC_30_360` is semantically undefined in the repository. | MATCH | `year_fraction_ords` is new. No existing instrument depends on any day count. |
| C.10 BDC | `BusinessDayConvention` StrEnum values `FOLLOWING, MOD_FOLLOWING, PRECEDING, MOD_PRECEDING, NONE`; `adjust_ord` NONE returns the input. | MATCH | Int ids are new (edge-only table). Fixture spelling `MODIFIED_FOLLOWING` is a fixture defect. |
| C.11 Calendar | `static_config.holiday_calendars_dict[name].holiday_dates_ord` sorted int32; fixture calendars `US`, `GB`, `EUR`. `CurveMeta.maturity_calendar` is a curve attribute, not a trade default. | MATCH | No currency→calendar default exists in the repository; none will be introduced. |
| C.12 Curves | `ingest/identity.py::_validate_discount_linkage` (D066) applied to forward/option rows only. `market_data/assemble.py::D080_CONSTRUCTIBLE_CURVE_IDS = {250_000_000}`; other curves under CURVE raise `NotImplementedError(D061)`. `container_name_to_id` resolves names. | MATCH | D066 must be extended to IRS rows plus the equality check; no new curve abstraction. Fixture curve names `USD_3M`, `USD_OIS` do not exist in `curve_meta` (`USD_SOFR_OIS`, `USD_LIBOR_3M`). |
| C.13 Model | `curves/zero_curve.py`: continuous zero, log-linear in T, flat-forward right tail, ACT/365F coordinate (D080); `discount_factor_grid (C_df, D, S)`. Any ordinal in the date universe is a valid DF target. | MATCH | `R = DF[c,S,:]/DF[c,E,:] − 1` is computable from the existing grid; no forward infrastructure exists to conflict with it. |
| C.14 Discounting mode | `DiscountingMode{NONE, CURVE}`; builder uses `np.ones` under NONE. No existing gate ties an instrument class to a mode. | MATCH | New gate in `build_universe_mapping`: NONE with non-empty period-factor table raises. Docstrings in `discounting_modes.py`/`builder.py` still say CURVE is unimplemented (drift; not an IRS defect). |
| C.15 FX | `ingest/fx_resolver.resolve_fx_rf_ids_inplace`; `edf_effective_triples (curve, date, fx)`; `_fold_effective_discount_factors` applies `DF·FX` once per row. `reporting/fx_conversion.py` (D081) converts settlement→reporting downstream and requires `V_0, q_0`. | MATCH | One conversion inside EDF; D081 is a separate layer, not a second conversion of the same cashflow. |
| C.16 As-of | `builder.py`/`zero_curve.py` enforce D079 (targets ≥ as_of) on the whole date universe; `IngestBatch.as_of_ord32` (D067). | MATCH | Admission checks `S_0`, `X_0`, `S_i < E_i`, lags ≥ 0 are new edge checks. D079 remains an independent producer guard, not an IRS validator. |
| C.17 Unsupported features | `ingest/dispatcher.py::FAMILY_VALIDATORS` has no RATES entry; `ingest/schema.py::ALL_ALLOWED_COLS_SET` has no IRS column; `canonicalize._validate_schema` rejects unknown columns. | MATCH | Today every IRS row fails closed at ingest. The new validator takes over that role; no fallback path exists to remove. |
| C.18 Derivation policy | `compile_ingest_soa_payload_from_df` derives only fx ids and curve ids by lookup; no defaults observed for dates/conventions. | MATCH | Only leg currency (from curve meta), fx ids, schedule dates and α are derived. |

---

## D. Shared ABI Blast Radius

### D.1 Compiler ABI callers
- `portfolio/compilers/registry.py`: `CompilerFn: TypeAlias = Callable[[IngestBatch, int, int, UixMapper], Block | None]`; `COMPILERS`, `FIELD_NAMES`, `_register`; three registrations (spot, forward NODE, option NODE). `VANILLA_FORWARD_CURVE_ID` and `VANILLA_IRS_ID` unregistered.
- `portfolio/engine.py::compile_portfolio_from_ingest_batch`: the only consumer. Observed logic: `block = compiler(...)`; `if block is not None: compiled_fields[field_name] = block`.
- Tests: `COMPILERS[SPOT_ID]` docstring reference; `assert COMPILERS[VANILLA_FORWARD_CURVE_ID] is None`; `test_unregistered_instrument_fails_loud` parametrised over unregistered ids (includes 5). No test imports `FIELD_NAMES`.
- Solutions tier (`solutions/risk_run/api.py`) calls `compile_portfolio_from_ingest_batch` only; it never calls a compiler or reads `FIELD_NAMES`.

### D.2 Two latent defects in the existing ABI (OBSERVED)
1. **Overwrite path.** `compiled_fields[field_name] = block` assigns, never concatenates. `VANILLA_FORWARD_NODE` and `VANILLA_FORWARD_CURVE` both map to `"vanilla_forward"`. If CURVE were registered, a batch with both would keep only the second run's block. Unreachable today only because CURVE is unregistered (asserted by a test).
2. **Silent drop on `None`.** A compiler returning `None` for a non-empty run removes every trade of that run from the portfolio without error. No existing compiler returns `None` for a non-empty run, and `risk_run` Gate A (declared universe vs `portfolio.trade_uids`) would catch the loss at the solution tier, but the valuation tier itself does not.

Both defects are removed by the proposed ABI (tuple return, per-slot concatenation, non-empty invariant, run-uid set equality). This is evidence for O-4, not a blocker. The blast radius is exactly: `registry.py`, `engine.py`, three compiler bodies (wrap the single block in a one-tuple; no numeric change), `compiled.py` (slot table, two new slots, `BlockT`).

### D.3 CompiledPortfolio callers
`valuation/engine.py::run_valuation` (iterates `portfolio.blocks`), `solutions/run_identity/hashing.py::hash_compiled_portfolio` (iterates blocks in band order), `solutions/risk_run/api.py` (`trade_uids`, `block_offsets`, `total_trades`, Gate A), sensitivities (`sensitizer_registry` dispatch on block type), tests. None indexes slots by name except through `CompiledPortfolio.ordered`/`blocks`; appending two slots after `vanilla_european_option` preserves every existing band index.

### D.4 FIELD_NAMES callers
`registry.py` (definition), `engine.py` (consumer). Nothing else. Deletion is safe.

### D.5 Floating block field callers
`FloatingCashflowBlock.forward_rate_uix`: `discounted_cashflow_block.py` (definition), `valuation/pricers/floating_cashflow.py` (unpack), kernel signature, one contract test constructing a dummy block. No production producer or compiler. Rename touches four files plus the test.

### D.6 DayCount callers
Definition, `VALID_DAY_COUNTS` (no observed caller), module `__main__` printer. No instrument, no hash, no serializer, no config row.

### D.7 Registry dependencies
`pricer_registry.py` import-time completeness over `BlockType` is independent of the compiler registry. The new `SLOT_BY_BLOCK_TYPE_ID` table gets an identical completeness assertion.

### D.8 Serialization / hash dependencies
`hash_compiled_portfolio` hashes every field of every block in band order with a type tag; adding bands changes digests only for portfolios that contain those bands. `UixMapper._compute_fingerprint` hashes named tables; the new `period_factor_effective_triples` enters by name; existing portfolios produce the same bytes only if the new table is empty **and** the fingerprint loop includes the empty table deterministically (same as `fk_effective_pairs` today). Required: the new table is always present (possibly `(0,3)`), never conditionally hashed.

**Conclusion:** the proposed changes are demonstrably bounded to the files listed and remove two latent silent-loss paths. Safety is contingent on the acceptance evidence in Section G (NOT EXECUTED here).

---

## E. Hidden Dependency Findings

1. `ValuationState` source is replicated verbatim into several generated contexts (`core`, `sensitivities`, `sensitivity_pnl`, `risk_run`, `solutions_sensitivity_pnl`). This is a documentation artefact of the context generator, not multiple definitions; the single source is `market_data/state.py`. The rename `forward_rates → forward_period_factors` must be followed by a context regeneration or the generated docs will show a stale field.
2. `business_day_add_ord(n_days=0)` rolls to the next business day (spot-lag semantics). Any IRS code path that reuses it for payment-date derivation under BDC = NONE would silently adjust. Recorded as a required implementation rule (C.5).
3. `tests/.../test_unregistered_instrument_fails_loud[5]`, `test_cashflow_blocks_are_still_not_production_reachable`, `test_forward_rate_state_field_has_no_producer` pin the current absence and must be retired at the phases named in the implementation plan; their docstrings state their own retirement condition.
4. `ingest_fixtures/portfolio/exhaustive.py` concatenates the defective IRS rows. Any test that ingests the exhaustive fixture through `compile_raw_df_into_ingest_batch` fails on unknown columns at the moment the IRS schema lands unless the fixture is rewritten in the same phase.
5. Docstring drift: `discounting_modes.py` and `builder.py` state CURVE is not implemented; `assemble.py` implements D080. `taxonomy.PrimitiveFactorType.FORWARD_RATE` labels the forward as primitive; under C.13 it is derived. Neither is a caller; both are reported for the owner, not changed here.

No other hidden dependency was found in the inspected repository surface (searches: `CompilerFn`, `FIELD_NAMES`, `CompiledPortfolio`, `forward_rate_uix`, `forward_period_factor`, `DayCount`, `30_360`, `BlockType`, `FixedCashflowBlock`, `FloatingCashflowBlock`, `COMPILERS[`, `compile_portfolio_from_ingest_batch`, `.reorder(`, `_row_aggregation`, `forward_rates`, `forward_period_factors`, `DECISIONS.md`).

---

## F. Economic Contract Consistency

| Element | Check | Result |
|---|---|---|
| Schedule construction | `u_k` strictly increasing by construction (later month ⇒ later date under any clamp); exact division and `u_n == maturity` are admission checks. | consistent |
| BDC adjustment | `fwd`, `bwd` monotone non-decreasing; MOD_FOLLOWING/MOD_PRECEDING monotone within and across months (proof in prior review); NONE identity. Property depends on the implementation and is covered by a mandatory dense-sweep/property test, not assumed. | consistent, test-dependent |
| `S_i < E_i` | not derivable (degenerate calendars give equality); enforced as admission check. | consistent |
| Accrual factor | `α = yf(S_i, E_i, dc)`; same `(S_i, E_i)` feed `R_i` and `α_i`; three closed-form day counts. | consistent |
| Fixing date | `X_i = bd_sub(S_i, ℓ_fix) ≤ S_i`; `S_i` strictly increasing and `bd_sub` monotone ⇒ `X_i ≥ X_0 ≥ as_of`. Only `X_0` checked. | consistent |
| Payment date | `T_i = bd_add(E_i, ℓ_pay) ≥ E_i` with `bd_add(d,0) = d` (new primitive semantics, see C.5). | consistent |
| Forward period factor | `R_i = DF(S_i)/DF(E_i) − 1`, key `(curve, S, E)`, no α; `f_i = R_i/α_i` definitional only, never computed. | consistent, no α in factor |
| Fixed cashflow | `ε N K α_i · EDF(T_i)` (unchanged kernel). | consistent |
| Floating cashflow | `−ε N (R_i + m α_i) · EDF(T_i)`. | consistent |
| Discounting | each cashflow at its own `T_i` from the D080 grid; deterministic single-curve model; no timing adjustment claimed. | consistent |
| FX | once, inside `EDF = DF·Q`; identity when ccy equal; D081 is downstream and separate. | consistent, no double FX |
| Sign symmetry | `PV(−ε) = −PV(ε)` row-wise. | holds |
| Zero spread, zero lag | floating leg telescopes to `−ε N Q (DF(S_0) − DF(E_{n−1}))`. | holds |
| Zero spread, lag > 0 | `−ε N Q Σ (DF(S_j) − DF(E_j))·DF(T_j)/DF(E_j)`; exact under the model; no telescoping. | holds |
| Spread linearity | `PV(m) − PV(0) = −ε N Q m Σ α_j DF(T_j)`. | holds |
| Par rate | `K* = Σ_j (R_j + m α_j) DF(T_j) / Σ_i α_i DF(T_i)`; `PV(K*) = 0`. | holds |

The contract is internally consistent. No redesign.

---

## G. Required Changes Before Freeze

### Mandatory (technical, before freeze)
None beyond recording the three MATCH WITH REQUIRED CHANGE rules inside the frozen text:
1. C.4: `adjust_ords` (vectorised, int-id) with dense-sweep equality to `adjust_ord`.
2. C.5: `bd_add(d, 0) = d`; the IRS path never calls `business_day_add_ord`.
3. C.6: `business_day_sub_ords` is a new primitive with `n = 0` identity.

### Owner Approval Only
- O-1 kernel/field/state change (Section B).
- O-2 numbering of C.1–C.18 with the three rules above.
- O-3 `DayCount` extension and deprecation.
- O-4 ABI change, lock condition = acceptance evidence E-1..E-10:
  - E-1 spot fixture: `hash_compiled_portfolio` and `run_valuation` `out` bitwise equal to a pre-refactor baseline.
  - E-2 forward-node fixture: same; `test_forward_production_path.py`, `test_vanilla_forward_full_revaluation_reconciliation.py` unmodified.
  - E-3 option-node fixture: same; `test_option_production_path.py` unmodified.
  - E-4 `UixMapper.fingerprint` equal pre/post on each fixture (table `(0,3)` present, hashed by name).
  - E-5 exhaustive fixture without IRS rows: digest, `out`, `block_offsets`, `trade_uids` equal pre/post.
  - E-6 `_row_aggregation` and `get_trade_pv` equal pre/post on E-5.
  - E-7 absent slots `None`; no empty block in any slot (negative test).
  - E-8 registry: `UNREGISTERED_INSTRUMENT_IDS` unchanged except IRS; import-time guards present; `SLOT_BY_BLOCK_TYPE_ID` completeness assertion; `mypy --strict` on the new `CompilerFn`.
  - E-9 `concat_blocks([b]) == b` field-wise; two-block concatenation preserves order, size, dtype, contiguity.
  - E-10 run-uid set-equality and non-empty invariants raise on a deliberately broken compiler.
  All E-items are NOT EXECUTED in this reconciliation.
- O-5 `docs/DECISIONS.md` attached and reconciled.

### Implementation Detail (not in the frozen contract)
- Enum int-id values; module names (`utils/schedule.py`, `ingest/irs_terms.py`, `ingest/cashflow_ledger.py`); per-calendar business-day array construction; `np.repeat` expansion mechanics; test file layout; context regeneration after renames; retirement timing of the three pinned tests; fixture rewrite (`vanilla_irs.py`, `exhaustive.py`) to V1-admissible values (positive notional, `USD_SOFR_OIS`, enum spellings, explicit calendars, USD rows only).

---

## H. Final Freeze Gate

1. Economic contract internally consistent: **yes** (Section F).
2. Consistent with authoritative repository decisions: **not verifiable** for `DECISIONS.md` text (O-5); consistent with every decision as cited in code (D045, D061/D080, D064, D066, D067, D069, D079, D081). No code citation contradicts the contract.
3. Implementable with existing architecture plus the approved ABI changes: **yes**; required new primitives are edge-only (`adjust_ords`, `business_day_add_ords`, `business_day_sub_ords`, `year_fraction_ords`, terms/ledger containers, mapper table, fold).
4. Shared compiler ABI change fully understood: **yes**; single consumer, three compiler bodies, two latent defects removed.
5. Blast radius bounded: **yes** (Section D).
6. Existing instrument behaviours protected: **conditionally**, by E-1..E-10 (NOT EXECUTED).
7. Remaining issues governance/approval rather than technical ambiguity: **yes**.

**Final verdict: FREEZE READY PENDING OWNER APPROVAL.**

No technical contract ambiguity remains. The remaining blockers are governance/authority approvals only: O-1, O-2, O-3, O-4 (with E-1..E-10 as lock condition) and O-5 (`docs/DECISIONS.md` verification).

**IMPLEMENTATION MUST NOT BEGIN** until O-1 to O-5 are closed in writing.

---

## Appendix 1. Frozen V1 Contract Text (C.1–C.18, as reconciled)

Reproduced from the candidate contract with the three reconciliation rules inserted at C.4, C.5, C.6. All other clauses unchanged.

- C.1 Scope: single-currency fixed-for-floating IRS; constant positive notional; no principal exchange; no amortisation; no stubs; no EOM rule; no optionality; regular periodic schedule. Unsupported features fail closed.
- C.2 Sign: `ε = −1` (PAY_FIXED), `+1` (RECEIVE_FIXED); fixed sign `ε`; floating sign `−ε`; no second sign carrier.
- C.3 Unadjusted schedule: `u_k = add_months(effective, k·freq)` from `effective` each time; day-of-month preserved, clamped to month end; `freq ∈ {1,3,6,12}`; `n` integer; `u_n == maturity`; no stubs.
- C.4 Accrual: `S_i = adjust(u_i)`, `E_i = adjust(u_{i+1})` with the leg BDC and calendar, via a vectorised int-id `adjust_ords` equal to `adjust_ord`; admission `S_i < E_i`; no UNADJUSTED mode.
- C.5 Payment: `T_i = bd_add(E_i, payment_lag)`, `payment_lag ≥ 0` business days, `bd_add(d,0) = d`; the IRS path does not call `business_day_add_ord`; no hard upper bound.
- C.6 Fixing: `X_i = bd_sub(S_i, fixing_lag)`, `fixing_lag ≥ 0`, `bd_sub(d,0) = d`, `bd_sub(d,n)` the n-th business day strictly before `d` (new primitive); admission `X_0 ≥ as_of` only; no fixing date stored; no historical fixing mechanism.
- C.7 Floating convention: `TERM_SIMPLE_IN_ADVANCE` only; one rate per period; same single curve.
- C.8 Spread: annualised decimal; floating cashflow `−ε·N·(R_i + m·α_i)`.
- C.9 Day count: `ACT_360`, `ACT_365F`, `THIRTY_360_BOND` (ISDA 2006 §4.16(f): `D1 = min(d1,30)`; `D2 = 30 if d2 == 31 and D1 == 30 else d2`; no February rule); reject `THIRTY_E_360`, `30_360`, others.
- C.10 BDC: `FOLLOWING, MOD_FOLLOWING, PRECEDING, MOD_PRECEDING, NONE`; NONE ⇒ no adjustment, no calendar, both lags 0; one BDC per leg for accrual, payment and fixing dates.
- C.11 Calendar: explicit iff BDC ≠ NONE; no default calendar.
- C.12 Curves: discount and projection explicit; D066 identity; D080 constructible; `projection == discount`; no multi-curve.
- C.13 Model: deterministic single-curve; `R_i = DF(S_i)/DF(E_i) − 1`; `f_i = R_i/α_i` (definitional); `CF_fixed = ε N K α_i`; `CF_float = −ε N (R_i + m α_i)`; `PV_i = CF_i · DF(T_i)` (times FX inside EDF); no convexity/timing adjustment; no "negligible" approximation.
- C.14 Discounting mode: CURVE only; NONE rejected when the period-factor table is non-empty.
- C.15 FX: single-currency economics; settlement currency explicit; conversion once inside EDF; identity when equal.
- C.16 As-of: explicit checks `S_0 ≥ as_of` (fixed leg), `X_0 ≥ as_of` (floating leg), `S_i < E_i`, lags ≥ 0; no raw `effective ≥ as_of`; all other date validity is derived.
- C.17 Unsupported: stubs, irregular periods, amortisation, principal exchange, EOM, unsupported frequencies/day counts/BDCs, missing calendars, historical fixings, other floating conventions, curve mismatch, non-CURVE discounting, multi-curve, unsupported FX topology, anything not admitted; no fallback, no default.
- C.18 Derivations: leg currency (curve meta), fx ids (resolver), schedule dates, α only.

## Appendix 2. Data, Factor, Block, Compiler Contracts (as reconciled)

- Terms: `IrsTradeTerms` uid-sorted SoA (fields per prior review E.1, without `period_end_adjustment`).
- Ledger: `LegSchedule` × 2 = `offsets`, `accrual_start_ord`, `accrual_end_ord`, `payment_ord`, `accrual_factor`. No uid, period index, notional, rate, spread, sign, curve, fx or day-count columns (repository ABI does not require any; blocks are built by `np.repeat` at compile).
- Factor: `period_factor_effective_triples (K,3) int32` `(curve_uix, S_uix, E_uix)`; `ValuationState.forward_period_factors (K,S)`; default `(0, batch_size)`; shape asserted against the mapper; table always present and fingerprinted.
- Blocks: `FixedCashflowBlock` unchanged; `FloatingCashflowBlock.forward_rate_uix → forward_period_factor_uix`; kernel `N·ε·(R + m·α)·EDF`.
- Compiler ABI: `CompilerFn -> tuple[Block, ...]`; per invocation (one instrument run) ≤ 1 block per `BLOCK_TYPE_ID`, all non-empty, tuple non-empty, every block uid ⊆ run uids, union of block uids == run uids; slot by `CompiledPortfolio.SLOT_BY_BLOCK_TYPE_ID`; per-slot concatenation in ascending `instrument_id`; `None` iff no contribution; `FIELD_NAMES` deleted.
- IRS compiler invariant: exactly one Fixed and one Floating block; per uid, fixed/floating row counts equal CSR period counts; block sizes equal summed counts; rows ascending uid then period; failure raises before return.

## Appendix 3. Decision Register Status

All P-IRS-001..030 remain PROPOSED until O-1..O-5 are closed. LOCK candidates after approval: 001–012, 014–022, 025–030. Implementation-level (no lock): 023, 024.
