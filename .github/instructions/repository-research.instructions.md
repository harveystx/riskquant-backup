---
applyTo: "**/*"
---
# Repository Research and Reconnaissance Standards

Repository research establishes facts before design or editing.

## Required reconnaissance

For material work, identify:

- actual repository root and applicable instruction files;
- active package paths;
- callers and downstream consumers;
- tests and test entry points;
- benchmarks and benchmark harnesses;
- demos/scripts;
- generated files;
- legacy/broken/orphaned paths;
- relevant methodology/design artefacts;
- current public contracts.

Never infer a path from a filename mentioned in a discussion. Verify it exists.

## Evidence reliability

Classify repository evidence rather than treating all sources as equally authoritative.

Highest confidence is normally:

1. approved methodology/architecture/API artefacts;
2. active implementation on the contract path;
3. independently constructed tests/oracles;
4. regression/golden tests;
5. benchmarks and profiling evidence;
6. demos/documentation;
7. legacy/orphaned code.

Contradictions must be surfaced and reconciled; tests are evidence, not intrinsically authoritative.

## Deliverable

Research outputs should identify facts, unknowns, conflicts, scope boundaries, and proposed next decisions. Do not edit production code while acting as the research-only role.
