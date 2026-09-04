---
applyTo: ".github/methodology/**/*.md,docs/methodology/**/*.md,docs/**/*.md"
---
# Methodology Artefact Standards

Approved methodology documents are the source of truth for mathematical/statistical semantics. Instruction files are compact operational renderings; they are not a substitute for methodology approval.

Every material methodology artefact should contain:

1. status (`DRAFT`, `APPROVED`, `SUPERSEDED`);
2. scope and non-scope;
3. notation and dimensions;
4. exact mathematical definitions;
5. estimator/algorithm;
6. assumptions;
7. finite-sample semantics;
8. boundary and failure cases;
9. numerical/tolerance semantics;
10. stochastic/RNG semantics;
11. units/sign conventions;
12. validation/oracle plan;
13. known limitations;
14. change history/decision record.

Open methodological decisions must be explicitly marked `OPEN`. Production implementation is not authorised by an artefact that remains open unless a separate recorded decision grants provisional implementation authority.
