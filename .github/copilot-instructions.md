# RiskQuant — Global Copilot Engineering Constitution

RiskQuant is a deterministic, numerically rigorous, memory-efficient, auditable quantitative engine.
Human-readable at the edge. Intentionally contiguous at the core.

This file is the global engineering constitution for the repository. It defines the invariant rules that apply to all AI-assisted and human engineering work. Path-specific instruction files add applicable constraints; they do not silently override this constitution. Agent files define role boundaries and decision rights.

## 1. Governing principle

Correctness, mathematical validity, data integrity, reproducibility, and architectural integrity take precedence over local convenience, stylistic preference, and speculative optimisation.

Existing code is evidence, not automatic truth.

Every material change must leave a reconstructable decision trail:

`requirement → evidence → decision → implementation → tests → benchmark → independent validation → audit`

Do not manufacture evidence. Do not infer approval from precedent.

## 2. Authority hierarchy

When sources conflict, resolve them in this order unless an explicit change decision is recorded:

1. Approved mathematical methodology.
2. Approved public/API contract.
3. Approved architecture and design decisions.
4. This global constitution.
5. Applicable path-specific instructions.
6. Existing tests, golden masters, benchmark evidence, and observed behaviour.
7. Established implementation conventions.
8. Generic language/library conventions.
9. Local stylistic preference.

A user request may explicitly authorise a methodology, architecture, or public-contract change. Treat that as a change decision, not as an excuse to rewrite unrelated areas.

## 3. Non-negotiable engineering principles

- Data-Oriented Design (DOD) and Structure-of-Arrays (SoA) are the default numerical architecture.
- Human-readable representations belong at boundaries; compact numerical representations belong in the core.
- Strings die early. They must not cross the JIT boundary.
- Memory locality is a first-class design objective.
- Correctness comes before cleverness.
- Determinism and reproducibility are explicit contracts.
- Contracts must be explicit, typed, validated, and testable.
- Fail fast at boundaries; never silently repair malformed domain inputs.
- Kernels contain numerical work, not policy or orchestration.
- Compilers/solutions own policy resolution, dispatch, mapping, and orchestration.
- Scenario-independent work must be completed before the scenario loop.
- Minimise allocations and preserve favourable memory layout and ownership.
- Polars is the default dataframe technology at dataframe boundaries; pandas is not the numerical-core dataframe substrate.
- NumPy/SciPy/Numba own numerical execution where appropriate.
- Ordinary `__init__.py` files are byte-empty.

## 4. Repository architecture

RiskQuant is organised conceptually as:

`TERMINAL → SOLUTION → CORE`

with `EDGE` providing external data and application boundaries into the controlled solution/core layers.

### EDGE

External schemas, source-system representations, data access, canonicalisation, file/database adapters, and human-facing data preparation.

### SOLUTION

Stable public/API contracts, Pydantic schemas, catalogue/recipe selection, validation, provenance, canonicalisation, orchestration, and controlled dispatch.

### CORE

Compact numerical state and high-performance kernels. Core code should be independent of transport formats, UI objects, strings, dataframe abstractions, and policy configuration.

### TERMINAL

Thin adapters such as Excel/xlwings. No business logic, model methodology, numerical policy, or duplicated orchestration.

## 5. Identity semantics

The identity vocabulary is normative:

- `*_id`: raw external/source-system identifier.
- `*_uid`: forged global business identifier.
- `*_uix`: compact numerical universe index / array position.

Never substitute one identity class for another.

Resolve identity and mapping before entering the numerical hot path.

## 6. Memory, layout, and dtype discipline

Prefer SoA over object-rich arrays and per-record Python objects in numerical execution.

Core numerical arrays should be explicitly shaped and dtype-controlled. Global defaults are:

| Domain | Default dtype |
|---|---|
| prices / PVs / quantities / strikes | `float64` |
| scenario shocks | `float32` |
| UIX / UID / indices / date ordinals | `int32` |
| flags / option-type codes | `int8` |

These are defaults, not prohibitions. An explicit approved domain/API contract may override them; the explicit contract is then normative.

Do not introduce implicit dtype promotion in hot paths without evidence that it is required and acceptable.

Preserve C-contiguity where the kernel contract expects it. Do not copy merely for aesthetic consistency; do not retain hostile layouts in performance-critical kernels.

## 7. Structural missingness

Structural missingness uses canonical sentinels from `riskquant.constants` where such semantics are defined.

Do not use `None` or `NaN` as generic linkage sentinels. Missingness must have explicit domain semantics.

Never silently drop, fill, interpolate, replace, or coerce missing values in a mathematical pipeline unless the governing methodology explicitly authorises it.

## 8. Compile once; price many

Before scenario execution, resolve as much scenario-independent work as possible:

- identities and universe mappings;
- product/model dispatch;
- interpolation setup;
- relabelling/mapping;
- permutation/gather construction;
- band or bucket resolution;
- static policy decisions;
- validation of invariant dimensions and domains.

The scenario loop should contain only scenario-dependent numerical work.

## 9. JIT boundary

The JIT boundary admits only primitive numerical data:

- NumPy arrays;
- scalar numerical values;
- integer indices;
- compact flags/codes;
- approved numerical sentinels.

The JIT boundary does not admit:

- strings;
- dataframe objects;
- Pydantic models;
- dataclasses;
- arbitrary Python objects;
- dictionaries;
- opaque policy/configuration objects.

Canonicalisation from rich objects to numerical state belongs outside the kernel.

## 10. Numba governance

Use the repository's canonical decorators. Do not create local alternatives unless a documented architecture decision explicitly requires one.

```python
from functools import partial
import numba as nb

njit = partial(
    nb.njit,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)

parallel_njit = partial(
    nb.njit,
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)

inline_njit = partial(
    nb.njit,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
    inline="always",
)
```

`parallel=True` requires an algorithmic justification and benchmark evidence. Fastmath is never a generic performance switch; numerical consequences must be acceptable under the governing methodology.

`error_model="numpy"` means mathematical domain validation belongs at the appropriate boundary. Do not rely on exceptions from inside the kernel as domain-policy enforcement.

## 11. Kernel discipline

A numerical kernel should:

- accept only canonical primitive inputs;
- have explicit shape/dtype/ownership expectations;
- perform one coherent numerical task;
- avoid hidden allocation where practical;
- avoid Python containers and object dispatch;
- avoid policy branching;
- have independently testable mathematical behaviour.

Do not place catalogue lookup, identity interpretation, product selection, logging, user-facing formatting, or configuration resolution inside numerical kernels.

## 12. Hot-path branching and allocation

Do not branch on configuration that is invariant across scenarios from inside the scenario loop.

Preallocate where complete overwrite is guaranteed. Understand that advanced indexing may materialise copies. Avoid full temporary matrices when indexed gathers or streaming aggregates satisfy the contract.

A performance-motivated change is not accepted on intuition alone. Material changes require benchmark evidence.

## 13. Dataframe boundary policy

Polars is the default for dataframe-like source and boundary work.

Dataframes do not cross into JIT kernels. Convert once into canonical arrays with explicit ownership, shape, order, and dtype.

Do not use dataframe convenience semantics to hide changes in ordering, missingness, duplicate handling, or type coercion.

## 14. Determinism and RNG

Determinism is a contract unless stochasticity is explicitly part of the methodology.

Randomness must have explicit ownership, state, seed semantics, draw-consumption semantics, and reproducibility expectations.

Do not create hidden global RNGs. Do not reseed inside low-level kernels. Do not change random-number consumption order casually in a stochastic algorithm.

## 15. Public API discipline

Public APIs are contracts, not convenience wrappers.

Use explicit typed parameters and return contracts. Avoid unstructured `dict[str, object]`, arbitrary tuples, magic positional conventions, and hidden defaults where a stable model is appropriate.

Public solution layers should normally expose Pydantic v2 contracts with `ConfigDict(extra="forbid")` where appropriate.

User-facing validation should be deterministic, specific, and cell/address-level where the consuming surface is Excel.

`api.py` should be the sole public surface of a solution package unless an explicit architecture decision states otherwise.

## 16. Python engineering baseline

For ordinary Python modules:

1. repository-relative path comment;
2. `from __future__ import annotations`;
3. explicit concrete typing;
4. no unexplained `Any`;
5. no `# type: ignore`;
6. small cohesive modules;
7. no speculative inheritance;
8. no catch-all `utils.py`/`helpers.py`/`manager.py` modules;
9. immutable slotted dataclasses where a dataclass is appropriate;
10. `Final` and `Literal` where they clarify stable semantics;
11. no import-time business side effects.

The sole exception is `__init__.py`: it must be byte-empty.

## 17. Error handling

Validate malformed external/domain inputs at explicit boundaries and raise precise exceptions.

Do not use `assert` for input validation.

Do not catch broad `Exception` unless the boundary explicitly requires exception translation and preserves the causal context.

Do not silently continue after contract failure.

## 18. Testing and proof

Tests are executable specifications and falsification tools.

Use the strongest applicable combination of:

- contract tests;
- analytical/reference-oracle tests;
- invariant/property tests;
- adversarial and edge-case tests;
- regression tests;
- ownership/aliasing tests;
- stochastic/reproducibility tests;
- integration tests;
- performance tests.

Label correctness claims as `EXACT`, `NUMERICALLY APPROXIMATE`, or `STATISTICAL` where this distinction matters.

Golden masters are evidence, not unquestionable truth. Every important numerical component should have at least one independent oracle where practical.

## 19. Performance governance

The evidence ladder is:

`baseline → profile → hypothesis → candidate → benchmark → numerical equivalence → decision`

Do not claim an optimisation without measurements appropriate to the material claim.

Material benchmark evidence should record, where relevant:

- CPU/model;
- OS;
- Python version;
- NumPy/Numba/SciPy versions;
- thread settings;
- problem dimensions;
- warm/cold state;
- benchmark repetitions;
- allocation/memory observations.

## 20. Methodology governance

Mathematical methodology must be explicit before production implementation.

For any material quantitative model, document at minimum:

- mathematical definition;
- estimator or algorithm;
- assumptions;
- finite-sample semantics;
- boundary cases;
- numerical tolerances;
- exactness classification;
- stochasticity and RNG semantics;
- units/sign conventions;
- validation and reference implementation;
- known limitations.

NumPy/SciPy defaults are implementation choices, not methodology, unless explicitly adopted by the approved methodology.

## 21. Change control

Classify changes before broad implementation:

- **A — defect/local implementation:** no contract or methodology change;
- **B — architecture change:** boundary, ownership, or data-flow change;
- **C — methodology change:** mathematical/statistical contract change;
- **D — public contract change:** API/schema/result semantics change.

B/C/D changes require an explicit decision record and appropriate independent review.

Never combine unrelated architectural, mathematical, and cleanup changes merely because they touch the same files.

## 22. Scope discipline

Search and classify before editing.

Separate:

- active production code;
- active tests;
- benchmarks;
- demos/examples;
- generated code;
- legacy code;
- orphaned/broken code.

Do not “repair” every broken demo or legacy path during a focused production change unless it is on the active contract path or has been explicitly promoted to scope.

## 23. Required engineering chain

For material work, the default sequence is:

`Research → Architecture → Quant Methodology → Implementation → Tests → Performance → Independent Validation → Audit`

Agents must respect role separation. A specialist may flag a blocker owned by another role, but should not silently assume that role's decision authority.

## 24. Release gate

Release status is earned evidence, not prose:

`IMPLEMENTED → TESTED → BENCHMARKED → INDEPENDENTLY VALIDATED → AUDITED → RELEASED`

A stage cannot be claimed solely because an earlier stage passed.

The final auditor is entitled to return `PASS`, `CONDITIONAL PASS`, or `FAIL`.

## 25. Decision hierarchy

When trade-offs are unavoidable:

`mathematical correctness > public contract > data integrity > architecture > determinism > memory safety > CPU efficiency > allocation efficiency > maintainability > local elegance`

## 26. AI response and delivery rules

An agent must:

- state material assumptions;
- identify unresolved ambiguity instead of inventing semantics;
- distinguish facts from hypotheses;
- cite repository evidence when performing reconnaissance;
- make exact changes narrowly;
- report tests actually run;
- report benchmarks actually run;
- identify known residual risks;
- never claim a file, test, benchmark, or validation exists when it was not observed.

For implementation work, deliver the smallest coherent patch that satisfies the approved contract.

For reviews, prefer precise findings with file/symbol references, severity, evidence, and remediation over generic commentary.

## 27. What not to do

Do not:

- add framework layers without need;
- introduce convenience abstractions across core boundaries;
- push strings/dataframes into numerical kernels;
- hide policy in kernels;
- silently change statistical conventions;
- rely on random seeds without defining draw semantics;
- optimise without a benchmark;
- change public result shape incidentally;
- use existing tests as the sole proof of mathematical correctness;
- keep dead compatibility surfaces indefinitely without evidence of need.
