---
applyTo: "tests/**/*.py,**/tests/**/*.py"
---
# Testing and Falsification Standards

Tests are executable specifications and falsification tools, not decorative coverage.

## Required evidence

Use the strongest applicable evidence set:

1. public contract;
2. analytical or independent reference oracle;
3. invariants/property tests;
4. adversarial and boundary cases;
5. regressions for known defects;
6. ownership/aliasing checks where relevant;
7. reproducibility/RNG checks where relevant;
8. integration tests;
9. performance tests where performance is contractual/material.

## Exactness classes

Every material numerical claim should be classified where relevant as:

- `EXACT` — invariant must match exactly;
- `NUMERICALLY APPROXIMATE` — bounded numerical tolerance is intrinsic;
- `STATISTICAL` — judged through a statistical acceptance criterion.

Do not use tolerance merely because exact equality is inconvenient.

## Oracle independence

The test oracle should not simply call the production implementation through another route. Prefer a direct formula, slow reference implementation, structural invariant, or independently constructed expected result.

## Adversarial coverage

Where relevant, test:

- empty inputs;
- singleton dimensions;
- duplicated/tied values;
- constant series;
- extreme but valid values;
- malformed shapes/dtypes;
- non-contiguous arrays;
- read-only inputs;
- deterministic/repeated RNG runs;
- boundary probabilities and finite-sample limits.

## Defect lifecycle

Every discovered defect must either become a regression test or have a documented reason why a test would not be meaningful.
