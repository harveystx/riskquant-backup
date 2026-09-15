# RISKQUANT — ULTRA-TIER FOURTH AUDIT PASS

## Production Sensitivity Path vs Full-Revaluation Oracle

You are conducting the **fourth independent, adversarial audit pass** of RiskQuant.

This remains an **AUDIT ONLY** phase.

---

# ABSOLUTE RULE

**DO NOT MODIFY CODE.**

Do not patch.
Do not refactor.
Do not redesign.
Do not write production code.
Do not alter tests.
Do not alter documentation.

Produce only an audit report.

The purpose of this pass is to establish facts, contracts, mathematical correctness, architectural ownership, and residual risk.

---

# GOVERNING CONSTITUTION

The governing RiskQuant architecture is:

> **FAIL FAST AT BOUNDARIES. KERNELS DO DUMB MATH.**

This is non-negotiable.

Therefore:

* kernels may assume valid canonical inputs when their preconditions are established by the appropriate boundary;
* validation belongs at the correct compiler / ingest / orchestration boundary;
* do not flag missing kernel validation merely because validation is absent from the kernel;
* do not add defensive branches to hot numerical kernels;
* do not turn kernels into policy engines;
* identify exactly which layer owns each invariant;
* distinguish a valid kernel assumption from an unowned precondition;
* distinguish a current defect from a latent buildout risk;
* distinguish a shared-input reconciliation from an independent oracle;
* distinguish model ambiguity from demonstrated current mispricing.

Existing code is not precedent.

---

# CUMULATIVE AUDIT CONTEXT

The previous audits established the following.

## Pass 1

The core product-pricing algebra was broadly sound, but several contract and documentation weaknesses were identified.

## Pass 2

The audit attacked the valuation boundary and established:

* unchecked empty factor universes can reach kernels;
* scenario-axis mismatches can produce plausible finite wrong values;
* block-row versus economic-trade semantics are unresolved;
* several Pass-1 findings were overcalled and were reclassified;
* Black-76 vega remains a genuine CRITICAL mathematical defect.

## Pass 3

The audit traced the market-data producer seam and found the first confirmed, economically material defect on the actual production risk path:

### X-01 — CRITICAL

`fx_spots_effective` is constructed from a dense FX sub-block of the flat levels matrix but indexed using UIX from the full `market_id_universe`.

This creates two incompatible index spaces.

In an ordinary constructible universe containing a traded FX spot alongside a settlement FX pair, this can produce:

> **a finite, plausible, silently incorrect FX rate**

rather than an exception.

Pass 3 also established:

### X-02

Wrong-class rows in shared price arrays can remain `np.empty` and are justified by a validator that does not exist.

### X-03

There is no explicit required-factor manifest binding block requirements to populated `ValuationState` factor universes.

### X-05

Negative TTE is floored to a positive numerical floor rather than rejected, potentially treating expired instruments as live near-zero-expiry instruments.

### X-06

FX sentinel semantics are inconsistent across layers.

### X-07

Cross-currency option volatility basis remains undefined, but actual cross-currency option pricing is currently unreachable because both the necessary market-data surface implementation and option compiler remain unimplemented.

### X-08 / X-09

Additional latent dual-universe and "by construction" coupling risks exist in the volatility surface path.

### X-10

Block-row versus economic-trade result semantics remain unresolved.

### X-11

Multiple instrument runs can conditionally collide on one flat compiled-field name.

---

# PRIMARY QUESTION OF PASS 4

Pass 3 identified that the **production sensitivity engine** is the actual risk path, while full revaluation is used as an oracle / reconciliation path.

Therefore the central question for this pass is:

> **Is the production sensitivity path independently correct, and is its reconciliation to full revaluation genuinely capable of detecting errors such as X-01?**

Do not assume that agreement proves correctness.

Specifically investigate the possibility:

```text
same corrupted factor
        ↓
full revaluation
        ↓
wrong result

same corrupted factor
        ↓
sensitivity engine
        ↓
same wrong result

difference ≈ 0
        ↓
reconciliation passes
```

A mutually consistent wrong result is **not** an oracle.

---

# PRIMARY SCOPE

Concentrate on:

* `riskquant.sensitivities`;
* specifically `sensitize_spot_block`;
* the immediate sensitivity compiler/orchestrator;
* `MarketDataMatrix` consumption by sensitivity;
* `UixMapper` consumption by sensitivity;
* sensitivity scenario construction;
* sensitivity PnL construction;
* sensitivity aggregation;
* reconciliation against full revaluation;
* the path from sensitivity PnL to the firm's VaR / ES calculation.

Inspect `riskquant.valuation` only where necessary to establish the oracle contract.

Inspect `riskquant.market_data` where required to establish shared inputs.

Do not broaden into unrelated packages unless the call graph or mathematical contract requires it.

---

# PHASE 1 — RECONSTRUCT THE ACTUAL PRODUCTION RISK PATH

Trace the complete production sensitivity path:

```text
raw market data
    ↓
market-data assembly
    ↓
MarketDataMatrix
    ↓
UixMapper / UIX
    ↓
portfolio sensitivity compiler
    ↓
sensitize_spot_block
    ↓
scenario PnL
    ↓
portfolio aggregation
    ↓
VaR / ES consumer
```

For each transition identify:

* exact data structure;
* exact index space;
* row semantics;
* scenario semantics;
* currency;
* unit;
* whether discounting is embedded;
* whether FX is embedded;
* exact validation owner;
* whether the same object is consumed by full revaluation;
* whether the data is independently reconstructed or shared.

Build an explicit dependency graph.

---

# PHASE 2 — X-01 IMPACT ON THE PRODUCTION SENSITIVITY PATH

This is the highest-priority investigation.

Reproduce the Pass-3 interleaved-FX configurations.

At minimum test:

### Case A — FX IDs occupy the first rows

Expected:

* correct result;
* no index mismatch.

### Case B — FX IDs are interleaved with non-FX market IDs

Expected:

* correct result;
* no silent row shift.

### Case C — a traded FX spot is itself present in the same universe

This is particularly important because Pass 3 identified that the presence of an FX spot position can create the silent-wrong regime.

Test:

```text
asset / non-FX ID
settlement FX ID
traded FX spot ID
```

and variants.

### Case D — multiple settlement FX pairs

### Case E — direct and inverse FX

For each case compare:

```text
raw expected FX
vs
fx_spots_effective
vs
sensitivity output
vs
full revaluation output
```

Determine exactly where a wrong value enters.

Do not merely state that X-01 exists.

Determine whether the sensitivity path:

1. consumes the corrupted `fx_spots_effective`;
2. independently reconstructs FX;
3. uses a different UIX;
4. corrects or masks the row mismatch;
5. amplifies the error;
6. or remains unaffected.

---

# PHASE 3 — ORACLE INDEPENDENCE AUDIT

This is mandatory.

For the production sensitivity path and full-revaluation path, identify all shared inputs and transformations.

Construct a table:

| Component             | Sensitivity | Full revaluation | Shared? | Independent? | Consequence |
| --------------------- | ----------- | ---------------- | ------- | ------------ | ----------- |
| raw prices            |             |                  |         |              |             |
| raw FX                |             |                  |         |              |             |
| `prices`              |             |                  |         |              |             |
| `fx_spots_effective`  |             |                  |         |              |             |
| effective spot prices |             |                  |         |              |             |
| `UixMapper`           |             |                  |         |              |             |
| scenario shocks       |             |                  |         |              |             |
| portfolio quantities  |             |                  |         |              |             |
| settlement FX         |             |                  |         |              |             |

Classify every reconciliation relationship as one of:

### INDEPENDENT

Separate construction paths with only economic agreement expected.

### SHARED INPUT

Both paths consume the same transformed factor.

### SHARED CALCULATION

Both paths reproduce substantially the same transformation.

### UNKNOWN

Independence cannot be established.

Do not call shared-input agreement "oracle validation."

---

# PHASE 4 — RE-DERIVE THE PRODUCTION SENSITIVITY MATHEMATICS

Derive the exact formula implemented by `sensitize_spot_block`.

Do not infer it from names.

Establish whether it computes something equivalent to:

$$
PV_i = Q_i \, S_{effective,i}
$$

and:

$$
\Delta PV_{i,s}
=
Q_i
\left(
S_{effective,i,s}
-
S_{effective,i,0}
\right)
$$

or another formula.

Then derive:

$$
PortfolioPnL_s
=
\sum_i \Delta PV_{i,s}
$$

or the actual implemented aggregation.

Verify:

* quantity applied once;
* base subtraction exactly once;
* FX applied correctly;
* no accidental discounting;
* no missing FX;
* no double FX;
* sign convention;
* scenario alignment;
* factor linkage.

The result must be expressed in the firm's reporting / settlement currency.

---

# PHASE 5 — SCENARIO CONSTRUCTION AUDIT

Trace exactly how sensitivity scenarios are constructed.

Determine:

* what scenario 0 means;
* whether scenario 0 is the base market;
* how shocks are generated;
* whether shocks are additive, multiplicative, absolute, relative, log-return, or otherwise;
* whether FX and underlying prices are shocked consistently;
* whether scenario ordering is stable;
* whether every factor uses the same scenario axis;
* whether scenario transformations preserve currency semantics.

Pay particular attention to:

```text
underlying price
+
FX transformation
+
settlement conversion
```

A sensitivity calculation can be locally correct while the scenario construction applies the wrong economic shock.

---

# PHASE 6 — INDEX-SPACE AUDIT

Be exceptionally strict about index spaces.

For every integer index encountered, label it explicitly as one of:

* global market ID;
* `market_id_universe` index;
* flat-level index;
* FX-sub-block index;
* price-universe UIX;
* effective-factor UIX;
* block-row index;
* trade-row index;
* scenario index.

For every gather verify:

$$
\text{producer index space}
=
\text{consumer expected index space}
$$

Do not accept:

* same dtype;
* same integer range;
* same numerical value in a fixture

as proof of index-space equivalence.

A wrong-universe index that is numerically valid is a HIGH/CRITICAL finding.

---

# PHASE 7 — `MarketDataMatrix` CONSUMPTION AUDIT

Determine exactly which fields of `MarketDataMatrix` sensitivity consumes.

At minimum investigate:

* `prices`;
* `fx_spots_effective`;
* any sensitivity-specific factor arrays;
* scenario matrices;
* UIX maps.

For every consumed matrix determine:

* axis 0 semantics;
* axis 1 semantics;
* cardinality;
* producer;
* validator;
* whether it shares storage with full valuation;
* whether it shares the same index-space assumptions.

Specifically investigate the Pass-3 X-01 seam:

> Does anything guarantee that `fx_spots_effective.shape[0]` corresponds exactly to the index space used by the sensitivity FX UIX?

Also determine whether the sensitivity path uses:

```text
fx_spots_effective
```

directly or indirectly.

---

# PHASE 8 — FX SEMANTICS

Re-derive FX treatment in the sensitivity engine.

For direct FX:

$$
FX(A\rightarrow B)
$$

For inverse FX:

$$
FX(A\rightarrow B)
=
\frac{1}{FX(B\rightarrow A)}
$$

Determine exactly where inversion occurs.

Then trace:

```text
raw FX
→ fx_spots_effective
→ effective spot
→ sensitivity PV
```

Check:

* direct orientation;
* inverse orientation;
* identity currency;
* settlement currency;
* duplicate pairs;
* sentinel handling;
* wrong pair;
* wrong row;
* wrong UIX.

Specifically test the X-06 sentinel conventions:

```text
NO_FX_RF_REQUIRED = -1
INVALID_UIX       = -999
```

Determine whether sensitivity interprets these consistently with market-data and valuation.

---

# PHASE 9 — SENSITIVITY VS FULL-REVALUATION RECONCILIATION

Locate every sensitivity/revaluation reconciliation.

For each one establish:

1. exact quantity compared;
2. exact scenario set;
3. exact base value;
4. tolerance;
5. whether raw inputs are shared;
6. whether transformed inputs are shared;
7. whether factor construction is shared;
8. whether the test can exercise interleaved FX IDs;
9. whether the test can detect a wrong factor row;
10. whether the test is genuinely independent.

Explicitly test this scenario:

```text
wrong fx_spots_effective
+
same matrix consumed by both paths
→
both paths wrong
→
reconciliation passes
```

If this occurs, state clearly:

> **The reconciliation is self-consistency, not independent validation.**

Do not describe it as an oracle in that case.

---

# PHASE 10 — PORTFOLIO AGGREGATION

Trace sensitivity rows into portfolio PnL.

Audit:

* block-row cardinality;
* trade UID semantics;
* instrument grouping;
* duplicate UIDs;
* multi-row trades;
* block ordering;
* scenario ordering;
* sign convention;
* offsets;
* missing rows;
* aggregation by economic trade.

Cross-reference **X-10**.

Determine whether the production sensitivity path inherits the same row/trade ambiguity found in `ValuationResult`.

Do not assume that because one result object is affected, the production VaR path is affected.

Prove the actual path.

---

# PHASE 11 — PRODUCTION RISK-METRIC CONSEQUENCE

Trace the sensitivity output through the actual risk metrics:

```text
sensitivity PnL
    ↓
scenario portfolio PnL
    ↓
VaR / ES
    ↓
contribution / decomposition where applicable
```

Determine whether the path feeds:

* VaR;
* ES;
* scenario ranking;
* VaR-day identification;
* contribution;
* diversification measures;
* any other production risk measure.

For each identified defect, determine whether it can reach those outputs.

Where possible, quantify the consequence.

Do not merely state "could affect VaR."

Demonstrate the mechanism.

---

# PHASE 12 — FAIL-FAST OWNERSHIP

For every invalid condition encountered, identify the exact owner.

At minimum:

* wrong FX index space;
* invalid UIX;
* empty factor matrix;
* wrong scenario count;
* wrong matrix shape;
* missing factor;
* malformed portfolio;
* duplicate or ambiguous UID;
* multiple instrument runs mapping to one field.

The central question is:

> **Does the boundary reject the invalid state before the numerical kernel sees it?**

Do not move this validation into the kernel.

If an invariant is valid "by construction", show the construction that proves it.

If the invariant is merely assumed, classify it as an unowned or latent contract.

---

# PHASE 13 — SOA / DOD / NUMBA AUDIT

Only after the production mathematics and data contracts are understood, re-check:

* SoA layout;
* primitive dtypes;
* memory locality;
* C/F contracts;
* read-only aliases;
* kernel allocations;
* object crossings;
* string elimination;
* canonical `riskquant.jit`;
* branchlessness;
* UIX gathers;
* unnecessary transformations.

Do not manufacture optimization findings.

The existence of a gather is not itself a problem.

The question is whether the gather operates on the correct index space with the required contract.

---

# PHASE 14 — ADVERSARIAL TEST MATRIX

Run actual executable probes wherever possible.

At minimum:

| Test                          | Required question                         |
| ----------------------------- | ----------------------------------------- |
| FX IDs first                  | Does sensitivity price correctly?         |
| FX IDs interleaved            | Does sensitivity remain correct?          |
| FX spot held as an instrument | Does X-01 manifest?                       |
| Multiple settlement FX pairs  | Correct row mapping?                      |
| Direct FX                     | Correct orientation?                      |
| Inverse FX                    | Correct inversion?                        |
| Identity FX                   | Correct no-conversion semantics?          |
| Invalid UIX `-1`              | Loud failure?                             |
| Invalid UIX `-999`            | Loud failure?                             |
| Empty factor universe         | Loud failure before kernel?               |
| Scenario mismatch             | Loud failure?                             |
| Multi-row trade               | Correct aggregation?                      |
| Mixed instruments             | No silent field overwrite?                |
| Sensitivity vs revaluation    | Independent or shared-input?              |
| Wrong shared factor           | Does reconciliation detect it?            |
| Production VaR                | Can any demonstrated defect reach VaR/ES? |

---

# FINDING DISCIPLINE

Every material finding must contain:

## Claim

One precise sentence.

## Evidence

Exact file, symbol, call path, and executed probe where available.

## Contract

State the invariant that should hold.

## Mathematical consequence

Derive the numerical/economic consequence.

## Architectural consequence

Identify the RiskQuant principle involved, if any.

## Ownership

Exactly one primary owner:

* ingest;
* market-data builder;
* state;
* UIX mapper;
* portfolio compiler;
* valuation engine;
* sensitivity engine;
* result/aggregation;
* kernel;
* model;
* registry.

If responsibility spans layers, identify the primary failed contract and the secondary affected layer.

## Category

One of:

* A — Mathematical defect
* B — Contract defect
* C — Documentation defect
* D — Architectural violation
* E — No defect

## Status

One of:

* CONFIRMED
* CONDITIONAL
* LATENT
* DOCUMENTATION ONLY
* NUMERICAL OBSERVATION
* NO DEFECT

## Severity

* CRITICAL
* HIGH
* MEDIUM
* LOW

## Confidence

* HIGH
* MEDIUM
* LOW

---

# EVIDENCE STANDARD

Do not assert a defect from naming alone.

Use, in descending order of strength:

1. directly executed source;
2. actual call-graph tracing;
3. mathematical derivation;
4. executed numerical probe;
5. source-level invariant proof;
6. documentation-only evidence.

If source cannot be re-opened, state that explicitly.

If a result comes from a faithful reimplementation of previously inspected source, label it as such.

Do not present a faithful reimplementation as though the current repository itself was directly executed.

---

# IMPORTANT NEGATIVE FINDINGS

Actively record important things that were attacked and survived.

Especially determine whether the sensitivity path has:

* correct spot mathematics;
* correct FX direction;
* correct inversion;
* correct identity-FX treatment;
* correct UIX usage;
* correct scenario alignment;
* independent reconciliation;
* correct aggregation;
* no double FX;
* no missing FX;
* no double discounting;
* no missing discounting;
* correct sign;
* no production VaR contamination.

A strong audit is not a list of concerns. It is a list of proven truths and proven failures.

---

# REQUIRED OUTPUT

## 1. Executive verdict

Answer explicitly:

1. Is the production sensitivity mathematics correct?
2. Is the sensitivity path independent of full revaluation?
3. Does X-01 affect the production sensitivity path?
4. Does full-revaluation reconciliation detect X-01?
5. Are sensitivity FX/index contracts correct?
6. Are UIX contracts fail-fast before sensitivity kernels?
7. Does sensitivity inherit X-10 row/trade ambiguity?
8. Can any confirmed defect reach production VaR/ES?
9. What is the single most important risk discovered by this pass?

---

# 2. Production dependency graph

Provide the actual path:

```text
raw market data
→ producer
→ factor matrix
→ UIX
→ sensitivity engine
→ scenario PnL
→ aggregation
→ VaR / ES
```

Mark every node shared with full revaluation.

Use:

* `[SHARED INPUT]`
* `[SHARED CALCULATION]`
* `[INDEPENDENT]`
* `[UNKNOWN]`

Do not hide shared dependencies.

---

# 3. Sensitivity mathematical derivation

Derive the complete formula from raw economic exposure through effective factor to scenario PnL.

Explicitly show:

* units;
* currency;
* FX;
* signs;
* scenario transformation;
* aggregation.

---

# 4. X-01 impact report

This section is mandatory.

Answer explicitly:

### Does X-01 affect sensitivity?

YES / NO / CONDITIONAL

### How?

Show the exact index path.

### Does sensitivity consume the same corrupted `fx_spots_effective`?

YES / NO / UNKNOWN

### Does full revaluation consume the same factor?

YES / NO / UNKNOWN

### Does reconciliation catch the defect?

YES / NO / ONLY IF INDEPENDENT

### Can X-01 reach VaR / ES?

YES / NO / CONDITIONAL

### Quantified consequence

Provide a concrete numerical example where possible.

---

# 5. Oracle-independence report

Create:

| Reconciliation | Shared raw data? | Shared transformed factors? | Shared scenario construction? | Independent? | Can detect X-01? |
| -------------- | ---------------- | --------------------------- | ----------------------------- | ------------ | ---------------- |

Then state whether the current "oracle" is actually an oracle.

---

# 6. Finding matrix

| ID | Severity | Category | Status | File | Symbol | Finding | Evidence | Owner | Confidence |

Only include material findings.

Do not pad the matrix with cosmetic observations.

---

# 7. Detailed findings

For every CONFIRMED or material CONDITIONAL finding use:

### Statement

### Evidence

### Contract

### Mathematical consequence

### Architectural consequence

### Ownership

### Severity

### Confidence

### What would prove this closed?

---

# 8. Production-risk consequence

Map every material finding to:

```text
market data
→ sensitivity
→ scenario PnL
→ portfolio PnL
→ VaR
→ ES
→ contribution/decomposition
```

State exactly where the defect enters and where it propagates.

---

# 9. Important negative findings

Explicitly record major tests that passed and explain why they genuinely establish correctness.

Especially include:

* correct FX orientation;
* correct sensitivity formula;
* independence or non-independence of the oracle;
* no double FX;
* correct scenario construction;
* correct aggregation, if proven.

---

# 10. Architecture verdict

Evaluate exactly:

```text
boundary
→ MarketDataMatrix
→ UIX
→ sensitivity kernel
→ scenario PnL
→ portfolio aggregation
→ production risk metric
```

For each segment classify:

* SAFE
* SAFE BY CONSTRUCTION
* UNSAFE
* CONDITIONAL
* UNOWNED

Explain only material reasons.

---

# 11. Final classification

Separate all material conclusions into:

## Proven correct

## Proven defective

## Conditional

## Latent

## Documentation only

## No defect

---

# 12. Single highest-value next audit target

Identify exactly **one** remaining seam whose inspection would reduce the most material uncertainty after this pass.

Do not provide implementation steps.

Do not provide patches.

Do not write code.

Do not produce a remediation plan.

This remains an audit-only exercise.

---

# FINAL PRINCIPLE

The purpose of this pass is not to find the maximum number of issues.

The purpose is to determine whether the firm's **actual production risk path** is mathematically correct, independently validated, contractually safe, and constitutionally aligned.

Be adversarial.

Be precise.

Do not manufacture defects.

Do not excuse defects as "buildout" when the current path can already execute them.

And never treat agreement between two paths as proof of correctness when the paths share the same potentially corrupted input.
