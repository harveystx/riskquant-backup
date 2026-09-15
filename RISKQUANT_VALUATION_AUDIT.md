# RISKQUANT — SEVENTH AUDIT PASS

## Compiled Sensitivity Book, Taylor P&L, Chunking, VaR / ES, and Final Risk-Number Integrity

**Status:** AUDIT COMPLETE — remediation and assurance phase follows
**Audit status:** No code modified during this pass
**Seam:** `sensitivity_pnl.{compile, engine, contracts, conventions, scheduling, grid}` · `sensitivity_pnl.kernels.{displacement, gather, taylor}` · `scenario_metrics.*` · `solutions.sensitivity_pnl.{pipeline, calculator, market}`
**Date:** 2026-09-15

---

# 1. Executive Verdict

## Final sensitivity-P&L engine

**PROVEN MATHEMATICALLY CORRECT under its admitted input contracts.**

The final transformation

```text
SensitivityBookSpec
    → compiled active-factor / position representation
    → CSR
    → displacement
    → Taylor P&L
    → scenario grid
    → VaR / ES
    → contribution
```

was independently audited at each material mathematical and index boundary.

The evidence establishes:

- DIRECT and SLOT factor identity preservation;
- sparse/high/global identifier safety;
- position identity preservation;
- correct CSR construction and duplicate policy;
- correct cross-term canonicalisation;
- exactly one diagonal factor of `1/2`;
- correct first- and second-order Taylor arithmetic;
- correct displacement conventions and domain gates;
- chunk invariance;
- batch-size invariance;
- scenario-axis preservation;
- deterministic position ordering;
- correct lower-tail VaR definition;
- exact empirical ES tail weighting;
- deterministic tie handling;
- contribution footing under its documented caller precondition.

An independent closed-form oracle produced a maximum relative discrepancy of approximately `6.46e-08`, consistent with the deliberate float32 displacement/output storage. Chunked and threaded execution were bit-identical in the tested configurations.

## Overall production risk number

**NOT TRUSTWORTHY in the current repository state.**

The final engine is not the source of the production risk error. It faithfully computes on the inputs it receives.

The current production risk number remains invalidated by the following upstream defects:

| Root defect | Demonstrated consequence |
|---|---:|
| **X-01** wrong FX effective-factor row | VaR99 `+15.71%`, ES99 `+15.93%`, tail overlap `95.5%` |
| **Z-01 / Z-02** shock-factor permutation | VaR99 `−57.07%`, ES99 `−56.98%`, tail overlap `0.0%` |
| **Y-01** uninitialised FX-node price rows | potentially undefined / garbage-valued valuation and sensitivity inputs |

Z-01/Z-02 is the most dangerous demonstrated scenario-construction defect because it can destroy factor-shock identity and materially change tail membership rather than merely rescale an exposure.

## Assurance verdict

The repository contains genuine downstream independent tests, including a test-owned hand-computed sensitivity-P&L oracle and an independent sorted VaR/ES reference.

However:

> **No existing end-to-end assurance mechanism was found that independently derives the expected risk number from economic factor identity and base state through scenario construction, market-data assembly, valuation/sensitivity, P&L, and final VaR/ES.**

The audit is therefore complete, but the production risk path is **not release-trustworthy**.

---

# 2. Final Production Data-Flow

```text
Static configuration
        ↓
ProxyResolution
        ↓
ShockUniverse / ShockComputePlan
        ↓
Scenario generation
        ↓
flat_levels
        ↓
MarketDataMatrix
        ↓
┌───────────────────────────────┬───────────────────────────────┐
│ valuation path                │ sensitivity path             │
│                               │                               │
│ valuation state               │ sensitize_spot_block          │
│ valuation kernels             │ BlockSensitivities            │
└───────────────┬───────────────┴───────────────┬───────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ↓
                         SensitivityBookSpec
                                ↓
                         FactorAxis / relabel
                                ↓
                    compile_sensitivity_book
                                ↓
                     CompiledSensitivityBook
                                ↓
                      displacement engine
                                ↓
                         Taylor P&L kernel
                                ↓
                         scenario P&L grid
                                ↓
                       scenario_metrics
                         ┌──────┴──────┐
                         ↓             ↓
                        VaR            ES
                         │             │
                         └──────┬──────┘
                                ↓
                    contribution / decomposition
```

Every production arrow in this path has now been classified.

---

# 3. Canonical Mathematical Contract

For position `p`, active factor `i`, and scenario `s`:

$$
\Delta V_p(s)
=
\sum_i
\left(
d^{(1)}_{p,i}x_i(s)
+
\frac{1}{2}d^{(2)}_{p,i}x_i(s)^2
\right)
+
\sum_{i<j} h_{p,ij}x_i(s)x_j(s).
$$

Where:

- `d1` is the first derivative with respect to the factor's declared displacement;
- `d2` is the raw second derivative;
- `h_ij` is the cross derivative for an unordered pair `i < j`;
- `x_i` is reconstructed from `(base_level, shocked_level)` according to `DisplacementConvention`.

The diagonal `1/2` is applied exactly once in the Taylor kernel. Cross terms are stored once for canonical `i < j` pairs and therefore require neither another `1/2` nor a factor of two.

---

# 4. Factor and Position Identity

The compiled book uses one active-factor index space for:

- active factor IDs;
- active level UIX;
- displacement conventions;
- quote bases;
- diagonal factor indices;
- cross factor indices.

Positions use one compiled row index with:

$$
out[m,s] \equiv position\_uids[m].
$$

DIRECT and SLOT representations are mutually exclusive by construction.

---

# 5. Proven Correct: CSR and FactorAxis

Proven properties:

- DIRECT identifiers remain semantically correct;
- SLOT identifiers are relabelled through the sanctioned pipeline;
- sparse and high IDs are safe;
- reordered valid inputs preserve semantics;
- duplicate handling is explicit;
- cross pairs canonicalise to `i < j`;
- diagonal cross-pairs are rejected;
- invalid factor mappings fail at the boundary;
- position rows remain associated with the correct position UID.

## Q-01 — MEDIUM / LATENT

A direct call to `compile_sensitivity_book` can silently misinterpret a SLOT axis whose integer semantic keys happen to be a permutation of `[0, K)` if the required `FactorAxis.relabel` step is bypassed.

This is **not a demonstrated sanctioned-pipeline production defect** because `pipeline.resolve` applies relabelling before compilation.

Canonical description:

> **Direct-compiler bypass hazard.** The remaining question is whether the direct compiler API is intentionally internal/preconditioned or should itself enforce canonical factor-axis representation.

---

# 6. Proven Correct: Displacement Reconstruction

The admitted displacement conventions are:

### ABSOLUTE

$$
x=L-L_0
$$

### RELATIVE

$$
x=\frac{L}{L_0}-1
$$

### LOG

$$
x=\ln\left(\frac{L}{L_0}\right)
$$

Arithmetic is evaluated in float64 and narrowed to float32 on storage.

For RELATIVE and LOG, valid base levels require:

$$
L_0>0.
$$

For LOG, shocked levels additionally require:

$$
L>0.
$$

ABSOLUTE has no positivity requirement.

These domain distinctions were explicitly attacked and are mathematically correct.

---

# 7. Proven Correct: Taylor Kernel

The implementation was independently matched at:

```text
nnz = 0, 1, 2, 3, 4, 5, 8, 9, 12
```

and on a cross-only book.

The kernel computes diagonal terms as:

$$
(d_1 + \frac12 d_2x)x
$$

and cross terms as:

$$
h_{ij}x_ix_j.
$$

The factor of `1/2` is applied exactly once. Cross terms contain no additional factor.

---

# 8. Proven Correct: Chunking, Scheduling, and Scenario Axis

`balanced_chunk_offsets` partitions the position axis into contiguous ranges and cannot reorder positions.

The P&L result was bit-identical across tested chunk counts:

```text
1, 2, 3, 4, 8, 16, 64
```

and across tested thread counts.

Asymmetric scenario markers preserved scenario identity through batch sizes 1, 2, 3 and 5.

The sensitivity-P&L engine handles partial scenario batches through explicit offsets and does not reproduce the full-revaluation runner's partial-column contiguity defect.

---

# 9. Proven Correct: VaR and ES

RiskQuant's D015 convention is:

> **VaR is the signed lower P&L quantile itself.**

The engine does not negate the result.

With:

$$
\alpha=1-C,\qquad m=\alpha S,
$$

and the documented floating-point snap, VaR is the `ceil(m)`-th smallest observation.

ES is the exact empirical lower-tail integral with fractional boundary mass.

The implementation was matched against independent sorted reference calculations across confidence levels, ties, small samples and adversarial orderings.

**D015: NO DEFECT.**

---

# 10. Contribution Contract

Contribution footing is exact under the documented precondition:

$$
total_s=\sum_k component_{k,s}.
$$

## Q-02 — LOW / DOCUMENTATION

The metrics layer does not rederive `total_row` from the component matrix. This is explicitly documented as a caller responsibility because the total may be accumulated independently.

Therefore Q-02 is a **documented contract precondition**, not a demonstrated numerical defect.

---

# 11. Independent Oracle Verdict

The repository contains two genuine downstream independent oracle classes:

1. a test-owned frozen book with hand-computed P&L;
2. an independently implemented sorted VaR/ES reference.

These are legitimate independent tests.

They begin downstream of the known production-state defects and therefore cannot establish end-to-end economic correctness.

The missing assurance artefact is:

$$
\text{factor identity + base state + shock semantics}
\rightarrow
\text{expected shocked state}
\rightarrow
\text{expected P\&L}
\rightarrow
VaR/ES.
$$

The oracle must deliberately include non-ascending and sparse IDs and heterogeneous factor relationships.

---

# 12. Upstream Corruption Propagation

The final engine was deliberately fed already-corrupted inputs to establish whether it masks or amplifies upstream errors.

## X-01

Correct:

```text
VaR99 ≈ -488.56
ES99 ≈ -562.20
```

Corrupted:

```text
VaR99 ≈ -565.32
ES99 ≈ -651.76
```

Impact:

```text
VaR99: +15.71%
ES99:  +15.93%
tail overlap: 95.5%
```

## Z-01 / Z-02

Correct:

```text
VaR99 ≈ -931.39
ES99 ≈ -1066.14
```

3-cycle shock permutation:

```text
VaR99 ≈ -399.85
ES99 ≈ -458.63
```

Impact:

```text
VaR99: -57.07%
ES99:  -56.98%
tail overlap: 0.0%
```

The engine is not responsible for either corruption. It faithfully evaluates the corrupted state.

---

# 13. Important Assurance Finding: Identifiability

A single pure-bilinear payoff:

$$
V=QSF
$$

cannot in general identify a permutation of its two factor shock processes because multiplication is commutative.

Thus a real production engine can execute correctly while a test fixture remains incapable of detecting a real factor-identity defect.

This yields a general assurance requirement:

> **Tests for index-space integrity must be identifiable with respect to the defect being tested.**

A fixture must contain sufficient heterogeneous factors and/or positions that exchanging factor shocks changes the independently calculated economic result.

---

# 14. Final Finding Matrix

| ID | Severity | Category | Status | Canonical disposition |
|---|---|---|---|---|
| **X-01** | **CRITICAL** | B | **CONFIRMED** | Wrong FX effective-factor row; production impact quantified. |
| **Y-01** | **CRITICAL** | B | **CONFIRMED** | FX-node price rows can remain uninitialised. |
| **Z-01** | **CRITICAL** | B | **CONFIRMED** | Parametric shock columns use wrong index space. |
| **Z-02** | **CRITICAL** | B | **CONFIRMED** | Historical-family permutation uses wrong index space. |
| **Y-02** | **CRITICAL** | D | **CONFIRMED** | Designated full-revaluation oracle is non-exercising. |
| **Y-03** | **HIGH** | D | **CONFIRMED** | Fixture topology masks index-space defects. |
| **Y-05** | **HIGH** | D | **CONFIRMED** | Base-state comparison cannot establish independence from shared upstream state. |
| **P-03** | **HIGH** | C | **CONFIRMED / UPSTREAM-ONLY** | Proxy-related arrays cross the generator boundary positionally without semantic-domain verification. |
| **Z-03** | **HIGH** | C | **CONFIRMED / OUTSIDE SENSITIVITY RISK PATH** | Partial full-revaluation batch is non-C-contiguous for the pinned kernel. |
| **Y-06** | **MEDIUM** | D | **CONFIRMED** | Valuation supports inverse FX while sensitivity rejects it. |
| **Y-08** | **MEDIUM** | C | **CONFIRMED** | Coefficient-evaluation scenario index is unchecked. |
| **P-04** | **MEDIUM** | D | **EXPLAINABILITY ONLY** | Proxy provenance is dropped after `ScenarioSetBatch`. |
| **Y-07b** | **MEDIUM** | C | **LATENT** | Caller-settable displacement convention over fixed-coordinate coefficients. |
| **X-12** | **MEDIUM** | C | **CONFIRMED / CARRIED** | Effective-FX row dimension not explicitly anchored to its producer universe. |
| **Z-04** | **MEDIUM** | C | **LATENT** | Unknown shift id can leave output unwritten when the public batch contract is bypassed. |
| **Q-01** | **MEDIUM** | B | **LATENT** | Direct-compiler SLOT relabelling bypass hazard. |
| **P-05** | **LOW** | C | **LATENT / FAILS LOUD LATE** | Non-finite proxy scale is not rejected at configuration boundary; downstream metrics reject non-finite P&L. |
| **V-10** | **LOW** | D | **CONFIRMED** | Writable aliases used where canonical RO aliases exist. |
| **Q-02** | **LOW** | E | **DOC** | Contribution footing precondition documented but not verified. |
| **P-07** | **LOW** | E | **DOC / MAINTENANCE** | Unused source-only proxy dedup representation. |
| **D013** | — | E | **RESOLVED SEMANTICALLY** | Proxy identity/provenance semantics are correct; downstream provenance loss remains P-04. |
| **D015** | — | A | **NO DEFECT** | Signed lower-tail P&L quantile implemented correctly. |
| **Y-07** | — | A | **CLOSED / NO DEFECT** | Shock convention and sensitivity displacement convention are independent. |
| **P-01** | — | B | **CLOSED / NO DEFECT** | Proxy output row binding proven. |
| **P-06** | — | A | **CLOSED / NO DEFECT** | Proxy chains/cycles prevented and structurally impossible. |
| **V-05** | — | — | **CLOSED** | Cashflow reachability treated as buildout state. |
| **W-03 / X-10** | — | — | **CLOSED ON SENSITIVITY PATH** | Sensitivity path has position/factor keying and explicit duplicate policy. |

---

# 15. Cross-Pass Architecture Finding

The recurring architectural failure pattern is:

> **Semantically distinct integer index spaces are represented as interchangeable positional integers across subsystem boundaries without sufficiently explicit binding contracts.**

Confirmed examples:

```text
global market-universe UIX
    ≠
dense FX-subuniverse position
```

```text
shock-universe UIX
    ≠
sorted shock-column position
```

```text
sorted shock position
    ≠
inverse-permutation UIX
```

```text
SLOT semantic key
    ≠
SLOT row position
```

These remain separate concrete defects with separate owners.

They are unified only at the architectural-pattern level.

### Remediation principle

> **Every integer index crossing a subsystem boundary must have an explicit semantic index-space contract, with a boundary invariant proving that the producer's index space equals the consumer's expected index space.**

This does not require object-heavy runtime wrappers in hot paths. The representation can remain DOD/SoA and integer-based.

---

# 16. Assurance Architecture Finding

The repository has strong local tests and genuine downstream independent oracles.

What it lacks is a whole-path oracle starting at economic factor identity.

The required assurance fixture must include:

- non-ascending identifiers;
- sparse/high identifiers;
- interleaved factor classes;
- at least one proxy;
- non-unit proxy scale;
- heterogeneous factor shocks;
- multiple positions and factors;
- enough diversity to make permutations identifiable;
- independent shocked-level calculation;
- independent P&L calculation;
- independent VaR/ES calculation.

It must not reuse the implementation being tested to establish expected semantics.

---

# 17. Remediation Gate

## Gate 1 — restore economic state integrity

Resolve:

- X-01;
- Y-01;
- Z-01;
- Z-02.

## Gate 2 — harden semantic boundaries

Resolve or explicitly contract:

- P-03;
- Y-08;
- X-12;
- Z-04;
- Y-07b;
- Q-01, subject to direct-compiler API classification.

## Gate 3 — preserve provenance where required

Address P-04 as an auditability/reporting requirement, without coupling it unnecessarily to numerical kernels.

## Gate 4 — build end-to-end assurance

Create the independent factor-identity-to-VaR/ES oracle described above and ensure its fixture explicitly defeats the benign ascending-ID layouts that masked earlier defects.

## Gate 5 — protect proven kernel contracts

Any implementation change must preserve the Pass-7 theorem, including:

- CSR identity;
- displacement mathematics;
- Taylor semantics;
- chunk invariance;
- batch-size invariance;
- VaR/ES definitions;
- DOD/SoA architecture;
- canonical `riskquant.jit` policy;
- performance assumptions validated by benchmark.

---

# 18. Final Mathematical Theorem

> **Theorem.** Let `B` be a valid `CompiledSensitivityBook` produced from a semantically valid `SensitivityBookSpec`, with factor identity correctly resolved, displacement domains satisfied, and semantically correct base and shocked levels. Then `SensitivityPnlEngine` computes the contractual Taylor expansion per position and scenario exactly up to its declared floating-point storage and reduction effects, independently of batch size, chunk count, thread count, valid position ordering, valid factor ordering, and valid CSR entry ordering.
>
> For an exactly bilinear payoff, inclusion of the mixed second derivative makes the second-order Taylor representation algebraically exact rather than a truncation approximation.

### Important qualification

The theorem is proven.

Its upstream hypotheses are not currently guaranteed.

The principal violations are:

1. X-01;
2. Y-01;
3. Z-01;
4. Z-02.

Therefore:

> **The final engine is mathematically trustworthy under contract. The current end-to-end production risk number is not.**

---

# 19. Final Production-Path Verdict

> **AUDIT COMPLETE FOR THIS PRODUCTION PATH.**

Every material production arrow has been classified.

The repository should now leave broad audit mode and enter controlled remediation mode.

The first remediation acceptance criterion is not merely that tests pass.

It is:

$$
\boxed{
\text{independent economic oracle}
=
\text{production result}
}
$$

on a deliberately hostile, non-ascending, sparse, proxy-containing, multi-factor universe.

The final engine should not be redesigned to compensate for upstream corruption. The proven kernel contract should be preserved while economic-state integrity and assurance are repaired upstream.
