# RiskQuant Claude Instructions

This file is the permanent operating contract for Claude working on RiskQuant.

`context_riskquant_all.txt`, provided in the Claude Project files, is the
**current RiskQuant repository snapshot for the task**. Treat the files and
contents represented in it as the current codebase state and source of truth
for repository inspection.

`DECISIONS.md` is **always attached separately to the task prompt**. It is the
authoritative record of locked architectural and mathematical intent. Do not
assume it is present in `context_riskquant_all.txt`.

The user's current prompt defines the task.

This file contains global operating rules only. It is not a project log,
continuation document, audit ledger, or task-specific manual.

---

## 1. Scope

**Follow the user's scope exactly. Do not expand it.**

A narrow task stays narrow.

* Do not fix adjacent defects merely because you found them.
* Do not turn a targeted task into a refactor.
* Do not redesign architecture unless the task requires it.
* Do not create abstractions, tooling, rules, documentation, or migrations
  unless requested.
* Do not infer additional deliverables from discoveries.
* The absence of an explicit prohibition is not permission to do adjacent work.
* Reuse existing production machinery before introducing anything new.
* Preserve behaviour outside the requested change unless the task or evidence
  establishes that the behaviour is incorrect or obsolete.

**Audit / review / red-team / assessment means analysis only unless the user
explicitly authorises changes.**

In audit mode:

* inspect the relevant project files and `DECISIONS.md`;
* identify concrete findings;
* distinguish observed facts from inference;
* do not modify anything;
* report adjacent issues rather than fixing them.

In implementation mode:

* change only what the task requires;
* modify only directly affected tests/scripts;
* preserve unrelated behaviour.

**Do not make owner-level architectural or mathematical decisions on the
user's behalf.** Present evidence, alternatives, and consequences.

---

## 2. Authority and repository state

Use each source for its proper purpose:

**Current codebase state:** `context_riskquant_all.txt`.
**Locked architectural/mathematical intent:** `DECISIONS.md`, attached to the prompt.
**Requested change:** the user's current prompt.

Do not assume files, APIs, dependencies, tests, configuration, behaviour, or
history that are not represented in the supplied project snapshot.

`DECISIONS.md` is separate from the codebase snapshot and must be consulted
whenever the task touches a locked contract or decision.

A project-file snapshot establishes repository content; it does **not** prove
that a command was executed, a test ran, a file was modified, or a benchmark
was measured.

If a task conflicts with a locked decision:

* identify the decision ID;
* state the conflict;
* get confirmation before changing the locked contract.

If code/tests appear to violate a locked decision, do not reinterpret the
decision to match the implementation. Treat the divergence as a possible
implementation defect and surface it when relevant.

If evidence conflicts or is incomplete:
**surface the conflict or uncertainty. Do not silently choose, repair, or
invent an answer.**

Never use chat history, old handoffs, continuation notes, audit ledgers, or
unstated assumptions as repository authority.

---

## 3. RiskQuant architecture

RiskQuant is a production risk engine built around:

**DOD → SoA → typed NumPy/Numba computation**

Architecture:

**TERMINAL → SOLUTION → CORE**

* `riskquant.solutions.*` is the public API.
* Core never depends on higher tiers.
* Computational state is Data-Oriented, not object-oriented.
* Human-readable data is resolved at the edge.
* Strings, external identifiers, dates, and configuration objects do not enter
  computational kernels.

### DOD / SoA / CSR

* **DOD:** computational state is organised for efficient traversal, not as
  networks of Python objects.
* **SoA:** related fields live in separate flat typed arrays with stable
  indexing and explicit dtype/layout contracts.
* **CSR:** existing sparse relationships retain their established
  Compressed Sparse Row semantics. Do not silently densify or change topology.
* Memory layout is a correctness and performance contract, not cosmetic detail.

---

## 4. Boundaries and kernels

### Edge

Human-readable representations and Polars belong at the boundary.

Prefer Polars `LazyFrame` for ingestion and transformation pipelines wherever
lazy execution is applicable. Materialise to a concrete `pl.DataFrame` only at a
deliberate boundary that requires it.

Pandas is not part of the target architecture. Do not introduce new pandas
usage. Existing pandas code is migration debt to be removed, not a permanent
legacy allowance.

Never use pandas in kernels.

Strings and external identifiers are resolved before core computation.

### Solution

`riskquant.solutions.*` is the public contract/API boundary.

### Kernel

Kernels perform mathematics on already validated primitive arrays.

Kernels do not:

* resolve identities;
* implement business policy;
* interpret sentinels;
* accept strings, DataFrames, Pydantic objects, Python datetimes, or pandas;
* allocate hot-path outputs;
* perform orchestration or business dispatch.

`riskquant.jit` is the canonical source of Numba decorators and type/layout
aliases.

Do not redefine Numba aliases locally or bypass established JIT configuration.

Preserve established dtype, shape, indexing, and C/F-contiguity contracts.

---

## 5. Numerical and data discipline

**FAIL FAST.**

Preserve established:

* dtypes;
* shapes;
* axis orientation;
* C/F contiguity;
* index semantics;
* sentinel semantics;
* sign conventions;
* FX direction;
* scenario conventions.

Do not silently repair incompatible input by:

* broadcasting;
* transposing;
* coercing;
* dropping;
* inventing defaults;
* substituting values.

No NaN/None error state. No invented sentinels.

Use established mathematical conventions. Before changing a locked mathematical
contract, read the relevant `DECISIONS.md` entry.

For numerical work, use an **independent mathematical/economic oracle**.
Do not validate a result only through another implementation sharing the same
logic.

---

## 6. Engineering principles

* **FAIL FAST.**
* **Kernels do dumb math.**
* **Human-readable at the edge, typed arrays at the core.**
* **Smallest correct change.**
* **Reuse existing production machinery.**
* **One concern per patch.**
* **No speculative abstractions.**
* **No unrelated cleanup.**
* **No silent behaviour changes.**
* **Do not preserve obsolete design merely because it already exists.**
* **Do not trade correctness for elegance.**

Do not weaken tests, assertions, tolerances, or contracts merely to make an
implementation pass.

Established behaviour is evidence, not automatically intent. Distinguish
current implementation from locked design intent.

---

## 7. Verification and evidence

Inspect the supplied project snapshot and the attached `DECISIONS.md` before
making claims or changes.

For non-trivial work:

1. identify the exact contract;
2. inspect the relevant implementation, tests, and decisions;
3. make the smallest required change;
4. perform the smallest relevant verification;
5. run broader verification where appropriate;
6. report exactly what was verified.

**Never claim execution unless execution actually occurred.**

When execution is unavailable, distinguish static inspection and analytical
reasoning from executed verification.

Use these distinctions when useful:

* **Observed:** directly established from project files or executed output.
* **Inferred:** conclusion derived from observed evidence.
* **Unresolved:** not established by available evidence.
* **Proposed:** suggested change, not current behaviour.

A broader failure is evidence to report, not permission to expand the task.

---

## 8. Testing

Prefer:

**structural invariants → tiny golden cases → focused tests → broader regression**

Use independent golden/oracle tests for mathematical behaviour.

Test the failure modes that matter to the contract.

Tolerances must reflect the actual computational dtype.

Never loosen a tolerance merely to hide an error.

Never report a test or suite as passing unless it was actually run.

When execution is available, report the exact command and actual result.

---

## 9. Claude-specific discipline

**The user controls scope.**

These are **not automatic permissions**:

* fixing nearby bugs;
* refactoring;
* redesigning architecture;
* changing `DECISIONS.md`;
* creating `.claude` rules;
* adding enforcement tooling;
* rewriting tests wholesale;
* migrating dependencies;
* packaging files;
* creating handoffs or continuation documents;
* resolving unrelated repository defects.

When information is missing, say it is missing.

When evidence conflicts, surface the conflict.

When an architectural or mathematical choice is required, present the choices
and evidence rather than silently making the owner decision.

When a discovered issue is outside the task, report it and leave it unchanged.

The goal is not to maximise code changed.

The goal is the **smallest correct, evidenced change within the user's scope**.
