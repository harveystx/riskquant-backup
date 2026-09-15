# RISKQUANT — CANONICAL ULTRA HANDOFF PROMPT

You are at the end of a long-running, high-complexity RiskQuant software-engineering session.

A new Claude conversation will be used to continue this work.

Your task is to produce a **durable, repository-grade engineering handoff** that allows the next Claude instance to continue with minimal rediscovery and without accidentally reversing established architecture, numerical conventions, implementation decisions, or unresolved issues.

This is **NOT a conversational summary**.

You are reconstructing the **CURRENT ENGINEERING STATE** of the repository and encoding it into three Markdown artifacts.

The objective is continuity of engineering judgment, not brevity.

---

# SOURCE-OF-TRUTH RULE

Use two sources of truth and reconcile them explicitly:

### Repository

The repository is the source of truth for:

* what code currently exists
* actual module and file paths
* actual classes, functions, APIs, signatures, and imports
* current implementation state
* actual tests
* actual test results
* current data structures
* current dtypes
* current numerical implementation
* current dependency configuration
* code that has actually been changed

Do not describe intended or discussed architecture as implemented architecture unless the repository confirms it.

### Conversation

The conversation is the source of truth for:

* architectural rationale
* explicit design decisions
* rejected alternatives
* constraints that may not be obvious from code
* numerical conventions explicitly agreed during discussion
* interpretation of unusual implementation choices
* known limitations
* unresolved questions
* reasons for deliberately non-obvious design choices

### Conflict rule

If repository state and conversation state appear to conflict:

1. Do not silently choose one.
2. Identify the conflict.
3. Treat the affected item as `OPEN` or `UNKNOWN` as appropriate.
4. Record the discrepancy in `CONTINUATION.md`.
5. Do not manufacture a resolution that was not explicitly established.

### Evidence rule

Never invent:

* file paths
* classes
* functions
* signatures
* test counts
* benchmark results
* golden values
* numerical results
* implementation status
* decisions
* rationale
* completed work

Where information is genuinely unavailable, write:

`UNKNOWN`

Where something remains unresolved, write:

`OPEN`

Where something is an explicit durable project rule or architectural decision, label it accordingly.

Where an earlier decision was replaced, mark it:

`SUPERSEDED`

---

# CRITICAL DISTINCTION: IMPLEMENTED VS INTENDED

For every material architectural statement, distinguish between:

* **CURRENT IMPLEMENTATION** — confirmed by repository state
* **INTENDED DESIGN** — explicitly agreed but not yet fully implemented
* **OPEN** — unresolved
* **UNKNOWN** — insufficient evidence

Never convert an intended design into a claim about current implementation.

Never convert an implementation accident into a project decision unless the conversation explicitly established it as intentional.

Never "clean up" an unusual implementation merely because a different implementation appears more elegant.

---

# REQUIRED DELIVERABLES

Create these three complete Markdown artifacts:

1. `CLAUDE.md`
2. `DECISIONS.md`
3. `CONTINUATION.md`

If any of these already exist:

* inspect them first
* preserve useful durable material
* reconcile them against the current repository and conversation
* refine/update them rather than blindly replacing them

Do not merely explain what should go into the files.

Produce the **complete Markdown contents**.

---

# REQUIRED RECONCILIATION WORK

Before writing the artifacts, perform a full reconciliation pass.

Inspect enough of the repository to establish:

* package structure
* major modules
* relevant source files
* tests
* demos/scripts where materially relevant
* project configuration
* existing documentation
* existing `CLAUDE.md`
* existing `DECISIONS.md`
* existing `CONTINUATION.md`
* any repository-level agent instructions that affect engineering work

For the current workstream, inspect the relevant implementation directly rather than relying solely on conversation descriptions.

The repository should be treated as a live engineering artifact, not merely as supporting material.

---

# ARTIFACT 1 — CLAUDE.md

## Purpose

`CLAUDE.md` is the **permanent operating manual** for future Claude instances working on RiskQuant.

It must contain **durable project knowledge only**.

Do NOT put ephemeral task state here.

Do NOT turn unresolved implementation details into permanent policy.

Do NOT duplicate the current workstream merely because it is currently important.

It should survive the current refactor and remain useful months later.

Use the following structure.

---

# 1. PROJECT PURPOSE

Explain what RiskQuant is for at an engineering level.

Be precise and technical.

Describe the purpose of the library, its computational role, and the nature of the problems it is designed to solve.

Only include facts supported by repository/conversation evidence.

---

# 2. ARCHITECTURE

Document the established architecture.

Where applicable, cover:

* package/module boundaries
* major layers
* data flow
* system edges vs computational cores
* compiler/orchestrator responsibilities
* kernel responsibilities
* SoA vs AoS
* ownership
* mutability
* public API vs internal implementation
* dependency boundaries
* representation changes across layers
* important architectural invariants
* separation of validation and computation

For each major layer, make clear:

* what belongs there
* what does not belong there
* what representations enter
* what representations leave

Do not document hypothetical future architecture as current architecture.

---

# 3. CORE ENGINEERING PRINCIPLES

Capture established principles as concrete engineering rules.

Especially verify whether the repository has established principles such as:

## FAIL FAST

Translate this into actual behavior.

For example, establish where invalid:

* shapes
* dtypes
* IDs
* indices
* dates
* configuration
* mappings
* numerical inputs
* sentinel values

must be rejected.

Do not merely write the slogan.

## Kernels do dumb math

State concretely that computational kernels:

* perform numerical computation
* should not own orchestration
* should not interpret business-level representations
* should not perform avoidable validation
* should not carry unnecessary abstractions

Only state restrictions actually established by the project.

Also cover, where evidenced:

* validation at boundaries
* human-readable structures at system edges
* typed numeric arrays at computational cores
* no strings in numerical kernels
* no Python objects in Numba kernels
* no DataFrames in kernels
* no Pydantic models in kernels
* explicit dtypes
* explicit indexing
* no implicit sentinel semantics
* benchmark-first development
* golden-oracle validation

---

# 4. TYPE / DTYPE CONVENTIONS

Document established dtype rules precisely.

Explicitly distinguish:

### Python / container representation

Examples may include:

* Python `int`
* Python `str`
* `dict`
* mappings
* configuration objects

### Boundary representation

Examples may include:

* Polars
* NumPy arrays
* validated records
* normalized IDs
* typed arrays

### Computational representation

Document exact established dtypes for:

* identifiers
* UIDs
* UIX / indices
* dates
* flags
* prices
* levels
* PVs
* PnLs
* scenario values
* shocks
* weights
* probabilities
* permutations
* masks
* structured arrays / SoA fields

Where applicable, establish the distinction between:

* Python/container integer types
* NumPy scalar dtypes
* array dtypes

Do not infer a dtype policy merely because a dtype appears in one file.

Only promote a rule to project-wide policy when supported by evidence.

---

# 5. ID / UID / UIX SEMANTICS

Where established, document separately:

* `id`
* `uid`
* `uix`
* source IDs
* row indexes
* factor indexes
* scenario indexes

Explain what each represents and whether it is:

* externally meaningful
* globally unique
* locally unique
* stable
* an index into another array
* safe to persist

Preserve exact semantics.

Do not collapse distinct concepts merely because they are all integers.

---

# 6. NUMBA POLICY

Document the repository's actual Numba policy.

Include, where established:

* allowed kernel inputs
* prohibited inputs
* validation location
* typing expectations
* allocation policy
* memory-layout requirements
* contiguity requirements
* read-only expectations
* scalar vs array kernel rules
* error handling
* exception expectations
* `fastmath` policy
* bounds-checking policy
* caching
* `nogil`
* `parallel`
* `error_model`
* orchestration location
* compiler-wrapper responsibilities

Only document actual project policy.

Do not invent settings simply because they are common Numba practice.

If a convention is module-specific rather than global, say so.

---

# 7. NUMERICAL CONVENTIONS

Document durable mathematical conventions that future engineers must not reinterpret.

This section is especially important.

Include, where established:

* sign conventions
* units
* axis meanings
* matrix orientation
* scenario orientation
* historical ordering
* return conventions
* shock definitions
* additive vs multiplicative shocks
* log returns
* difference shocks
* quantile conventions
* VaR conventions
* ES conventions
* diversification definitions
* weighted risk conventions
* EWMA conventions
* stress conventions
* FX conventions
* settlement conventions
* date semantics
* price/PV semantics
* portfolio aggregation semantics

Use equations where they make the convention harder to misunderstand.

Do not describe conventions generically.

State the exact RiskQuant semantics.

---

# 8. RISK / VAR / ES CONVENTIONS

Where established, preserve exact definitions.

For example, if applicable, document:

* whether VaR is represented as a negative PnL quantile
* which tail is used
* confidence level / quantile semantics
* horizon conventions
* historical scenario ordering
* diversification definitions
* inclusion/exclusion semantics
* VaRC
* iVaR
* stress or add-on treatment
* ES interpretation

Do not substitute textbook terminology for project-specific semantics.

---

# 9. WEIGHTING / EW CONVENTIONS

Where established, document:

* chronological scenario ordering
* whether oldest or newest scenario occupies a given column
* weight ordering
* λ semantics
* normalization
* fallback behavior
* quantile interpretation
* whether weights are applied to scenarios, observations, or another dimension

State formulas where appropriate.

Make the newest/oldest convention explicit.

---

# 10. TESTING PHILOSOPHY

Document durable test philosophy.

Cover, where established:

* unit tests
* integration tests
* regression tests
* golden oracles
* independent mathematical oracles
* invalid-input tests
* fail-fast tests
* edge cases
* shape tests
* dtype tests
* numerical tolerances
* demos
* property-based or statistical validation if actually used
* performance validation

Do not put ephemeral passing counts here.

Those belong in `CONTINUATION.md`.

---

# 11. PERFORMANCE PHILOSOPHY

Document durable performance requirements.

Where established, include:

* allocation constraints
* memory layout
* contiguous arrays
* SoA preference
* vectorization strategy
* kernel granularity
* benchmark-first development
* warm vs cold compilation considerations
* expected computational complexity
* scalability expectations
* benchmark/regression discipline

Do not put temporary benchmark numbers here unless they are an explicit durable performance contract.

---

# 12. DEPENDENCY POLICY

Document established dependency rules.

Include, where supported:

* Python version requirements
* required dependencies
* optional dependencies
* deliberately avoided dependencies
* boundary-only dependencies
* numerical-core dependencies
* package/import rules
* dependency-layering constraints

Where known, reflect the actual `pyproject.toml` / package metadata.

Do not infer dependency policy from personal preference.

---

# 13. CODE STYLE

Document durable project style:

* naming
* typing
* formatting
* line length
* docstrings
* comments
* module structure
* class/function structure
* error messages
* readability
* abstraction level
* import style
* public/private naming
* constants
* test naming
* demo naming

Preserve established style preferences where they are actually documented or repeatedly enforced.

The code should optimize for maintainability and auditability, not cleverness.

---

# 14. AGENT OPERATING RULES

Tell future Claude instances how to work safely in this repository.

Include rules such as:

* inspect code before changing architecture
* inspect existing documentation before introducing new conventions
* preserve locked decisions
* preserve numerical semantics
* prefer narrow changes
* do not redesign locked components casually
* do not replace narrow architectural solutions with broad abstractions merely because they look more elegant
* run focused tests before broad test suites
* use the repository as the source of truth for implementation state
* use the conversation as the source of truth for rationale not encoded in code
* never claim tests passed without actually running them
* never claim a benchmark was run without actual evidence
* never silently reinterpret conventions
* never silently change signs, axis orientation, dtype, weighting, date semantics, or sentinel behavior
* fail fast when repository state contradicts assumptions
* preserve deliberately unusual behavior when it is intentional
* distinguish implementation from intended design
* classify unresolved questions as `OPEN`
* classify unavailable information as `UNKNOWN`

Do not include conversational history.

---

# ARTIFACT 2 — DECISIONS.md

## Purpose

`DECISIONS.md` is the durable **architectural decision record**.

It explains **WHY** RiskQuant is structured the way it is.

It should contain material decisions that future engineering work must respect.

It is not a duplicate of `CLAUDE.md`.

It is not a task log.

It is not a list of every implementation detail.

Only include decisions that materially constrain future implementation or explain important design choices.

For every decision use exactly this structure:

## D001 — [Decision title]

**Status:** LOCKED / OPEN / SUPERSEDED

**Decision:**
[Precise statement]

**Reason:**
[Why this was chosen]

**Implications:**
[What future code must do differently because of this]

**Rejected alternatives:**
[Only explicitly discussed alternatives]

---

Continue numbering sequentially.

Never reuse an identifier.

If an existing `DECISIONS.md` already contains decision IDs, continue from the highest existing ID.

---

# DECISION CAPTURE REQUIREMENTS

Capture material decisions relating to any applicable area, including:

* architecture
* module boundaries
* API design
* SoA
* AoS
* IDs
* UIDs
* UIX / indexing
* dtypes
* Python vs NumPy scalar representation
* Numba
* validation
* fail-fast behavior
* sentinels
* NaN policy
* memory layout
* ownership
* mutability
* indexing
* scenario orientation
* historical ordering
* EWMA
* weighted risk
* VaR
* ES
* diversification
* reporting currency
* FX
* settlement
* stress engines
* simulation engines
* add-ons
* mapping representations
* dependency choices
* performance decisions
* testing strategy
* naming conventions
* module boundaries
* public API
* compiler/orchestrator responsibilities
* kernel responsibilities
* numerical semantics

Only include decisions actually evidenced.

---

# DECISION STATUS RULES

### LOCKED

Use `LOCKED` only when the project has explicitly established the decision or when the architecture is clearly fixed and treated as such.

### OPEN

Use `OPEN` when:

* the team is still deciding
* alternatives remain viable
* implementation has not been resolved
* repository and conversation evidence conflict
* a design question was raised but not conclusively settled

Do not convert an unresolved discussion into a LOCKED decision.

### SUPERSEDED

Use `SUPERSEDED` when a prior decision was later changed.

Preserve the old decision and explicitly state:

* what replaced it
* why it changed
* what future code should follow now

---

# IMPORTANT DECISION-DISCIPLINE RULE

Do not infer decisions merely because:

* a particular implementation exists
* a particular dtype happens to be present
* a function happens to be written one way
* a dependency is currently imported
* a benchmark happens to look good
* a design would be mathematically cleaner

A design becomes a documented decision only when supported by project evidence.

---

# ARTIFACT 3 — CONTINUATION.md

## Purpose

`CONTINUATION.md` is the **CURRENT WORKING STATE**.

Unlike `CLAUDE.md` and `DECISIONS.md`, this file is expected to become stale as engineering work progresses.

It should allow a new Claude instance to continue immediately without rereading the entire old conversation.

It must be factual and operational.

Use the following exact top-level structure.

---

# CURRENT MISSION

State exactly what the broader engineering effort is trying to accomplish.

Describe the actual current objective, not generic aspirations.

---

# CURRENT WORKSTREAM

Identify the specific subsystem, module, refactor, bugfix, API change, numerical problem, performance problem, or architectural work currently being handled.

---

# CURRENT STATUS

Clearly separate:

## DONE

Only work with actual evidence of completion.

## IN PROGRESS

Only work currently being implemented or actively investigated.

## BLOCKED

Only genuinely blocked items.

## NOT STARTED

Important planned work that has not yet begun.

## KNOWN / ACCEPTED

Known behavior, limitations, or unusual implementation that is currently accepted and should not be mistaken for a bug.

---

# CURRENT OBJECTIVE

State the exact engineering problem currently being solved.

Make it operationally precise.

---

# IMMEDIATE NEXT ACTION

State the first action the next Claude should take.

It must be specific enough to execute immediately.

Bad:

> Continue the refactor.

Good:

> Inspect `riskquant/.../shock_universe.py`, verify the current `(source_factor_id, shift_type_id)` deduplication implementation, then update only the affected builder path while preserving the established `K × S` orientation. Run the focused shock-universe tests before modifying downstream callers.

Do not give a vague next step.

---

# RELEVANT FILES

For every materially relevant file record:

* exact path
* purpose
* current state
* important functions/classes
* recent changes
* remaining changes
* relevant tests

Do not invent paths.

Only include files actually confirmed by repository state or clearly identified in the conversation.

For files whose relevance is uncertain, mark the uncertainty explicitly.

---

# IMPLEMENTATION STATE

Provide this table:

| Component | Status | Current reality | Remaining work |
| --------- | ------ | --------------- | -------------- |

Use statuses such as:

* DONE
* IN PROGRESS
* BLOCKED
* NOT STARTED
* OPEN
* UNKNOWN
* ACCEPTED

The **Current reality** column must describe what actually exists now.

---

# API STATE

Record APIs currently being changed, depended on, or materially constrained.

For each relevant API, include where known:

* exact name
* exact signature
* argument semantics
* return semantics
* shapes
* dtypes
* defaults
* validation
* error behavior
* compatibility requirements
* callers that have already been migrated
* callers that still require migration

Do not invent signatures.

Where the exact signature is not known, write:

`UNKNOWN`

---

# NUMERICAL STATE

Record all numerical conventions relevant to the current workstream.

Be explicit about:

* matrix dimensions
* row/column semantics
* axis meanings
* signs
* units
* dates
* scenario ordering
* quantiles
* confidence levels
* weighting
* EW λ values
* stress conventions
* FX conventions
* settlement conventions
* golden values
* regression values

For matrices, explicitly state notation such as:

`[N × S] = N objects/factors × S scenarios`

and state exactly what each axis means.

Do not rely on context or intuition.

---

# TEST STATE

Record actual evidence only.

Include:

* exact test commands
* exact test paths
* passing counts
* failing counts
* skipped counts when relevant
* known failures
* golden values
* regression values
* demo status
* benchmark results
* unresolved warnings
* compilation behavior
* warm/cold distinctions when relevant

Never write:

> tests pass

unless the evidence supports that exact statement.

Prefer:

> `pytest tests/... -q` → 19 passed, 0 failed

when that is the actual observed result.

If test evidence is unavailable:

`UNKNOWN`

---

# KNOWN ISSUES

Separate:

## MUST FIX

Issues that must be resolved for the current work or correctness.

## SHOULD FIX

Important but non-blocking issues.

## ACCEPTED / KNOWN

Known behavior that is intentionally retained or currently accepted.

Preserve:

* audit IDs
* issue numbers
* regression IDs
* finding IDs
* named exceptions
* warning identifiers

Where the project has audit findings, preserve their exact identifiers.

---

# RECENT CHANGES

Summarize only changes relevant to understanding the current implementation.

Use exactly this conceptual format:

`BEFORE -> AFTER -> WHY`

Focus on changes that affect:

* architecture
* APIs
* numerical semantics
* data structures
* dtypes
* performance
* tests
* downstream compatibility

Do not turn this into a chronological conversation transcript.

---

# DO NOT REGRESS

List concrete behaviors, conventions, APIs, or implementation properties that future Claude must not accidentally break.

Especially protect, where applicable:

* FAIL FAST semantics
* dumb-kernel architecture
* validation boundaries
* SoA layouts
* dtype semantics
* ID / UID / UIX semantics
* scenario orientation
* historical ordering
* VaR sign conventions
* diversification definitions
* EW weighting
* stress/simulation separation
* reporting-currency conventions
* settlement conventions
* golden values
* known regression values
* accepted unusual behavior
* performance contracts
* audit-remediation requirements

Only include items actually relevant to the current work.

---

# OPEN QUESTIONS

List only genuinely unresolved questions.

Do not include settled issues.

Do not disguise decisions as questions.

Do not use this section as a wishlist.

Each item should make clear:

* what remains unresolved
* why it matters
* what evidence is missing, if applicable

---

# CURRENT BLOCKER

If there is no blocker:

`NONE`

If there is one:

* state the exact blocker
* identify the affected component
* state what is required to unblock it

---

# NEXT STEPS

Provide a short ordered sequence beginning with the immediate next action.

The first item must match `IMMEDIATE NEXT ACTION`.

Do not include speculative work unrelated to the current workstream.

---

# RISKQUANT-SPECIFIC ENGINEERING GUARDRAILS

Throughout all three artifacts, pay particular attention to the following **only where verified by repository/conversation evidence**.

The following concepts are known to be important RiskQuant engineering themes and must be preserved when they are part of the established architecture:

* `FAIL FAST`
* kernels perform dumb computational mathematics
* validation and orchestration outside computational kernels
* human-readable representations at system edges
* typed numeric representations at computational cores
* SoA
* explicit dtypes
* explicit indexing
* `id` / `uid` / `uix`
* Python/container types vs computational NumPy types
* no strings in Numba numerical kernels
* no Python objects in numerical kernels
* no DataFrames in computational kernels
* no Pydantic models in numerical kernels
* explicit memory layout
* read-only inputs where established
* no implicit sentinel semantics
* explicit NaN policy
* explicit shape/orientation semantics
* historical scenario ordering
* VaR sign semantics
* diversification-benefit definitions
* EW weighting semantics
* stress/simulation separation
* reporting-currency semantics
* settlement conventions
* golden-oracle validation
* benchmark-first development
* audit findings and remediation state

These are **verification targets, not permission to invent facts**.

---

# AUDIT / REVIEW FINDINGS

If the conversation or repository contains formal audit findings, preserve their identifiers exactly.

For each relevant finding, state:

* identifier
* status
* affected file/module
* observed problem
* remediation status
* remaining action

Do not omit a finding merely because it is inconvenient or old.

Do not assume that a finding is fixed merely because a related code path changed.

---

# PERFORMANCE EVIDENCE

When performance has been materially discussed, preserve actual evidence in `CONTINUATION.md`.

Record:

* workload
* dimensions
* warm/cold state
* machine/environment if known
* observed timing
* benchmark command
* status
* whether it is a durable contract or merely an observation

Do not elevate one-off benchmark results into permanent requirements unless explicitly decided.

---

# DEMOS / EXAMPLES

Where demos are part of the workstream, record:

* exact path
* purpose
* current status
* assumptions
* expected output
* golden/oracle checks
* known caveats

Do not treat demos as authoritative implementation unless they are explicitly designed as golden examples.

---

# RECONCILIATION PASS

Before finalizing the artifacts, check all of the following:

1. `CLAUDE.md` contains durable project knowledge only.

2. `DECISIONS.md` contains decisions and rationale, not an entire project manual.

3. `CONTINUATION.md` contains current state, not permanent policy.

4. No `LOCKED` decision contradicts another `LOCKED` decision.

5. No `OPEN` question is accidentally represented as `LOCKED`.

6. Superseded decisions are clearly marked.

7. All file paths match the repository.

8. Function and class names are real where asserted.

9. API signatures are not invented.

10. Test results are supported by actual evidence.

11. Benchmark results are supported by actual evidence.

12. Numerical conventions are internally consistent.

13. Matrix dimensions and axis semantics are internally consistent.

14. Sign conventions are internally consistent.

15. Historical ordering and weighting conventions are internally consistent.

16. The handoff does not accidentally reverse any previously agreed architectural decision.

17. Temporary implementation details have not been elevated into permanent project policy.

18. Intended architecture has not been represented as implemented architecture.

19. Repository/conversation conflicts have been surfaced rather than silently resolved.

20. Unavailable information is labelled `UNKNOWN`.

21. Unresolved decisions are labelled `OPEN`.

22. Superseded decisions are labelled `SUPERSEDED`.

23. The `IMMEDIATE NEXT ACTION` is consistent with the actual repository state.

24. Another Claude instance could continue the work from these three files without requiring the previous conversation.

25. No important implementation constraint, numerical convention, audit finding, golden value, or accepted unusual behavior has been silently dropped.

---

# FINAL OUTPUT FORMAT

Return the three complete artifacts in exactly this order:

# CLAUDE.md

```markdown
[complete contents]
```

# DECISIONS.md

```markdown
[complete contents]
```

# CONTINUATION.md

```markdown
[complete contents]
```

Then provide exactly:

# HANDOFF CHECK

* Repository state reconciled: YES / NO
* Durable rules captured: YES / NO
* Decisions captured: YES / NO
* Current workstream captured: YES / NO
* Exact next action captured: YES / NO
* Known uncertainties explicitly labelled: YES / NO
* Implemented vs intended state reconciled: YES / NO
* Repository/conversation conflicts surfaced: YES / NO

Do not add a conversational summary.

Do not explain your chain of thought.

Do not apologize.

Do not discuss the handoff process.

Do not omit tedious but materially relevant implementation details merely to make the output shorter.

Accuracy takes precedence over brevity.

The goal is not to summarize the past.

The goal is to preserve enough engineering state that the next Claude can safely continue RiskQuant without rediscovering established decisions or accidentally regressing the architecture.
