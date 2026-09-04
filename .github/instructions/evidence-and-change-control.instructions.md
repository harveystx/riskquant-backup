---
applyTo: "**/*"
---
# Evidence and Change-Control Standards

## Evidence states

Use only evidence states that have actually been earned:

`IMPLEMENTED → TESTED → BENCHMARKED → INDEPENDENTLY VALIDATED → AUDITED → RELEASED`

A later state implies its evidence was produced, not merely that a reviewer believes it should pass.

## Change classes

- **A:** local defect/implementation change;
- **B:** architecture/boundary/ownership change;
- **C:** mathematical/statistical methodology change;
- **D:** public/API contract change.

B/C/D changes require explicit decision capture and appropriate independent review.

## Benchmark claims

Never state “faster”, “more efficient”, “scales better”, or similar material claims without measurements. Record the benchmark dimensions and environment sufficient for reproduction.

## Scope

Do not widen a focused change into opportunistic cleanup. Broken legacy/demo code is in scope only when it participates in the active contract or has been explicitly promoted.

## Refactor ledger

Material rejected optimisation hypotheses, deferred architectural clean-ups, and known non-goals should be recorded in `REFACTORS_NEEDED.md` or the repository's equivalent decision ledger.

## Evidence honesty

Never invent test results, benchmark results, audit outcomes, or external approval. State unknowns explicitly.
