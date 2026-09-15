# riskquant.simlab — Production Closure Report

Mode: PRODUCTION CLOSURE. Implementation + tests + benchmarks, not further
architecture. Architecture treated as FROZEN (D036); no new abstractions, no new
ADRs beyond closing D038.

---

## 1. Production-readiness assessment (baseline inspected)

Traced the real executable path, not filenames:

```
parse_scenario_model_config (Pydantic, solution tier)
  -> build_scenario_generator(config, *, data: RiskFactorMatrix | ParametricRiskData)
  -> {Historical, FilteredHistorical, StationaryBlockBootstrap, FilteredBootstrap, Parametric}
  -> ScenarioGenerator protocol -> Iterator[ScenarioSetBatch]  (shocks float32, F-contiguous)
  -> sensitivity P&L -> scenario aggregation -> VaR/ES  (E2E demo, D019 goldens)
```

Baseline on the current dump before any change: **526 passed** (scenario-model
seam + `tests/simlab`), E2E `ALL CHECKS PASSED`.

Verified already-safe (evidence, not assumption):

* **Factor integrity.** Generators are constructed only inside
  `build_scenario_generator`; no production bypass. `ParametricRiskData` enforces
  int32 / 1-D / C-contiguous / non-empty / sorted-ascending / unique factor ids,
  square finite float64 Cholesky, positive finite vols, all K-coupled — rejected,
  never coerced (the generator resolves columns by `searchsorted`, so sorted+unique
  is a correctness precondition, now enforced at the data boundary).
* **Config/data pairing.** Full matrix fails fast: every matrix-backed config
  rejects `ParametricRiskData` and the parametric config rejects a
  `RiskFactorMatrix` (`TypeError` at the solution boundary).
* **Protocol uniformity.** All five generators nominally subclass
  `ScenarioGenerator`, share an identical `iter_batches` signature and yield the
  single `ScenarioSetBatch` type; no model-specific type leaks downstream.
* **Boundary purity.** The builder passes only primitives and typed NumPy arrays
  into generators (`config.<attr>`, `pdata.<array>`); no Pydantic object, dict, or
  model-name string reaches a generator constructor or a kernel.
* **Batch invariance (D029).** Implemented structurally: all randomness is drawn
  upfront at full run size, then sliced; `batch_size` never enters the RNG
  consumption path. Exercised across the stochastic generators.

## 2. Confirmed P0 findings

**None.** No correctness, factor-alignment, reproducibility, batch-invariance,
numerical-contract, or fail-fast blocker was found in this pass. The
previously-identified gaps (D021/D037 RNG independence, D038 typing) are
capability/uniformity gaps, not wrong-number risks.

## 3. Confirmed P1 findings

### P1-1 — Continuous config typing was non-uniform and internally incoherent (FIXED)

*Observed defect (implementation gap).* Two policies coexisted in
`scenario_models/schemas.py`. The lax policy (`expected_block_size`,
`decay_factor`, `max_vol_ratio`) silently coerced representations — `"0.9"`,
`Decimal("0.9")`, `Fraction(9,10)` were accepted as `0.9`. The strict policy
(Parametric) used `isinstance(x, (int, float))`, which **accidentally accepted
`np.float64`** (a `float` subclass) while **rejecting `np.int64` and
`np.float32`** — an indefensible asymmetry across genuine NumPy real scalars.
`numbers.Real` is not a usable criterion (admits `bool` and `Fraction`, rejects
`Decimal`).

*Risk.* A numerical configuration field accepting a string is a fail-fast hole at
the solution boundary — it absorbs an upstream parsing failure that belongs at
EDGE. No invalid *value* passed (domain/bool/non-finite were rejected on both
paths), so this was config-integrity, not a wrong-number defect.

*Files/symbols.* `src/riskquant/solutions/scenario_models/schemas.py`:
`_require_real`, `_reject_bool_block_size`, two `_reject_bool_real`,
`SbbGeneratorConfig.expected_block_size`,
`FilteredHistorical/FilteredBootstrap.decay_factor`/`.max_vol_ratio`,
`StudentTDistribution.df`, `VarianceGammaDistribution.nu`,
`RegimeSwitchingDistribution.prob_crisis`/`.crisis_vol_multipliers`.

*Remediation — COMPLETED (D038).* One canonical `_require_real` defined once at
module top, applied to all continuous parameters:

```
accept : int, float, np.integer, np.floating   (NOT bool)
reject : bool, str, Decimal, Fraction, complex, None, NaN, +inf, -inf
store  : Python float ; domain via existing Field bounds (unchanged)
```

`crisis_vol_multipliers` keeps its legitimate **scalar-or-sequence** semantics;
the typing rule applies to the numerical leaves. Error messages name the
parameter and the offending type. Change confined to `schemas.py`; no
`api.py`/generator/kernel/union/`ParametricRiskData` change; no golden
re-baseline.

*Verification (measured, after fix; uniform across SBB / FHS / FB / Parametric):*

| Input | Before (lax) | Before (strict) | After (all four) |
|---|---|---|---|
| float / int / np.float64 | accept | accept | **accept → float** |
| np.int64 / np.float32 | accept | **REJECT** | **accept → float** |
| bool | reject | reject | **reject** |
| `"2.5"` / Decimal / Fraction | **accept** | reject | **reject** |
| NaN / ±inf (incl. np scalars) | reject | reject | **reject** |

### P1-2 — Peak memory scales O(total_scenarios x K) regardless of batch_size (DOCUMENTED, not a defect)

*Observed behaviour (already-safe, with an operational consequence).* D029's
batch-invariance guarantee is implemented by drawing all randomness upfront.
Streaming therefore bounds the **batch buffer**, not the **RNG draw**. Measured
peak (Parametric, K=250, tracemalloc):

| total_scenarios | bs=100 | bs=1000 | bs=S | theoretical S·K·8 B |
|---|---|---|---|---|
| 5,000 | 10.9 MB | 13.6 MB | 25.6 MB | 10.0 MB |
| 10,000 | 20.9 | 23.6 | 50.6 | 20.0 |
| 20,000 | 40.9 | 43.6 | 100.6 | 40.0 |
| 40,000 | 80.9 | 83.6 | 200.6 | 80.0 |

Peak tracks the upfront float64 normals draw exactly and is flat in `batch_size`
(small batches add only the batch buffer). Bootstrap-family generators draw
integer indices and stay ~1 MB at the same scale.

*Risk.* Not a correctness risk. It is an operational sizing fact: a Parametric run
at S=1,000,000 x K=250 needs ~2 GB for the draw alone. **Operator rule:
budget ≈ total_scenarios × K × 8 bytes for Parametric, independent of batch_size.**

*Why not "fixed".* Reducing it means per-batch reseeding / counter-based RNG,
which changes the D029 mechanism — a locked decision and a new architectural
decision. Out of scope for a closure pass; recorded with a sizing rule instead.

### P1-3 — Co-seeded stream independence not implementable (DEFERRED, no production use case)

Confirmed by archaeology: the three stochastic generators rebuild
`default_rng(self.seed)` per `iter_batches`, so two same-seed instances are
perfectly correlated (identical), and `seed=None` is entropy-backed per call.
Per D037 the resolution is an additive `stream_id` with
`SeedSequence(seed, spawn_key=(stream_id,))` and `stream_id=0` preserving the
exact legacy stream. **No production or demo caller currently constructs
co-seeded independent engines** (the seam builds one generator per config; the
demo uses one deterministic Historical generator). Per the explicit instruction,
D037 was NOT implemented: no production use case requires the machinery today.

## 4. P2 observations (not actioned)

* `crisis_vol_multipliers` accepts scalar or `list`/`tuple` but not `np.ndarray`
  at the config boundary. Pre-existing, legitimate (configs are JSON-shaped), and
  out of D038's scope.
* Credit-tier configs use bare lax `float` (they do not even reject `bool`). D038
  is deliberately scoped to the scenario-model seam; extending the contract is a
  separate future decision.
* `ScenarioGenerator` is not `@runtime_checkable`; dispatch is on config type, so
  no runtime `isinstance` on the protocol is needed. Accepted design choice.

## 5. Completed remediation

| Change | File | Type |
|---|---|---|
| Canonical `_require_real` (float-family predicate) defined once; applied to all 9 continuous parameters; two ad-hoc bool-only validators and a duplicate helper removed | `src/riskquant/solutions/scenario_models/schemas.py` | production (D038) |
| D038 typing suite: accepted/rejected types x every continuous field x all four models, domain boundaries, canonical stored type, vector-leaf semantics | `tests/solutions/scenario_models/test_continuous_typing.py` | new test |

## 6. End-to-end validation status

| Suite | Result |
|---|---|
| D038 targeted (`test_continuous_typing.py`) | **160 passed, 2 skipped** (skips = the (0,1) domains that contain no integer) |
| Scenario-model seam suite | **405 passed, 2 skipped** |
| `tests/simlab` | **281 passed** |
| Focused regression + solutions | **1565 passed, 4 skipped** |
| E2E demo (config → generator → batches → sensitivity P&L → VaR/ES) | **ALL CHECKS PASSED** (exit 0) |
| D019 LOG_RETURN goldens | **preserved** (only the known US/NL calendar-order non-determinism) |
| Changed-file lint (D023) | clean |
| Scope diff vs pristine dump | exactly `schemas.py` + the new test file |

**Pre-existing unrelated failure (not caused by, and not in scope for, this
pass):** `tests/dependence_stress/test_invariants.py` fails *collection* with a
bare `from canonical_measures import ...` ImportError. Status unchanged.

## 7. Performance evidence

Production-representative fixture: K=250 factors, T=1000 observations (~4y daily),
S=10,000 scenarios, `batch_size=1000`, warm (post-JIT), single process.

| model | warm runtime | scenarios | throughput | peak MB |
|---|---|---|---|---|
| Historical | 0.007 s | 999 | 137 k scen/s | 2.0 |
| FilteredHistorical | 0.005 s | 999 | 214 k scen/s | 2.0 |
| StationaryBlockBootstrap | 0.026 s | 10,000 | 388 k scen/s | 1.1 |
| FilteredBootstrap | 0.010 s | 10,000 | 963 k scen/s | 1.1 |
| Parametric (student-t) | 0.064 s | 10,000 | 155 k scen/s | 23.7 |

Batch-size sensitivity (Parametric, S=10,000): bs=1 → 0.211 s (47 k scen/s);
bs=100 → 0.066 s; bs=1000 → 0.062 s; bs=S → 0.075 s. Per-scenario cost at bs=1 is
~3x the batched cost (Python-level per-batch overhead), still 47 k scen/s.

**Conclusion: performance is sufficient; no optimization undertaken.** Every
generator produces a full 10,000-scenario run over 250 factors in ≤64 ms. Per the
benchmark rule, nothing was optimized because nothing was measured to be a
bottleneck.

## 8. Remaining deferred items

| Item | Status | Trigger to implement |
|---|---|---|
| D037 `stream_id` / `SeedSequence` independence | methodology resolved, deferred | first co-seeded / parallel / ensemble consumer |
| Parametric O(S·K) draw memory | documented with sizing rule | a run whose S·K·8 exceeds the host budget |
| D038 contract for credit-tier configs | out of scope | separate decision |
| `dependence_stress` collection error | pre-existing, unrelated | separate fix |

## 9. Final self-grill (evidence-based)

* **Strongest remaining wrong-number path?** Not inside SimLab's seam. The
  residual risk is upstream: a caller supplying a `RiskFactorMatrix` whose
  factor-id ordering disagrees with its own column ordering. The matrix contract
  owns that, and Parametric's equivalent is now enforced by `ParametricRiskData`.
* **Can a configured run differ across executions?** No, for `seed=int`:
  restart-reproducible, batch-invariant, construction-order independent.
  `seed=None` is explicitly entropy-backed with no reproducibility claim.
* **What prevents a shock hitting the wrong factor?** Sorted+unique int32 factor
  ids enforced at construction, `searchsorted` resolution against them, and
  fail-fast on unknown/duplicate ids.
* **Silent dtype/layout change?** Shocks are float32 F-contiguous at every
  generator; the data contracts reject non-float64/non-C-contiguous inputs rather
  than coercing. Verified in the seam-completeness suite.
* **Tiny batches?** Exact concatenated-stream equality at bs=1/7/full, measured.
* **10x scale?** Measured at K=250/S=10k; runtime linear and memory governed by
  the documented S·K rule.
* **Malformed input still accepted?** After D038, no continuous config field
  accepts a string, `Decimal`, `Fraction`, `bool`, or non-finite value.
* **Reproduce a VaR six months later?** Requires: model tag, config parameters,
  seed, factor universe, matrix/calibration window, horizon, shift convention. All
  are explicit, serializable values in the Pydantic config plus the construction
  data — no metadata object needed in the numerical core. Persisting the parsed
  config JSON alongside the result is sufficient and is the recommended operator
  practice.
* **Anything architecturally unnecessary added?** No. One helper corrected, two
  ad-hoc validators removed, one test file added. Net production complexity fell.

## 10. Production-readiness verdict

| Dimension | % | Evidence |
|---|---|---|
| Architecture | 95% | D036 frozen seam; tier boundaries clean; no bypass; protocol uniform; net complexity reduced this pass |
| Correctness | 92% | No P0; factor/layout/dtype contracts enforced and tested; D019 goldens preserved; independent-oracle coverage from earlier passes intact |
| Testing | 90% | 405 seam + 281 simlab + 1565 focused green; D038 matrix covers every continuous field x every model; E2E golden in place |
| Performance | 88% | Measured at production scale; ≤64 ms per 10k-scenario run; memory scaling law quantified; no bottleneck found |
| Operability | 80% | Config is fully serializable and reproducible; provenance achievable by persisting parsed config; no first-class run-record helper exists |
| **Overall** | **90%** | No known P0; all locked contracts hold; remaining items are explicit and justified |

**Verdict: PRODUCTION READY for the supported use cases**, which are: single-engine
scenario generation per configuration, with `seed=int` reproducibility, any
batch size, at scales where `total_scenarios × K × 8 bytes` fits the host budget.

Explicit, justified limitations: (a) co-seeded independent streams are not
supported (D037 deferred — no consumer); (b) Parametric memory scales with
S·K by construction (D029 guarantee); (c) `seed=None` carries no reproducibility
claim.

## 11. Exact next production increment

**Operability closure:** add a minimal solution-tier run record — the parsed
config (already fully serializable), the factor-universe identity, and the
engine/library version — emitted alongside the risk result, so a VaR number is
reconstructible without bespoke operator discipline. Small, additive, no core
contamination. This is the single remaining gap between 90% and production-hardened.
