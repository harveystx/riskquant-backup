# RiskQuant Agent Handoff Contract

**Status:** APPROVED  
**Scope:** All material multi-agent RiskQuant engineering work  
**Authority:** `.github/copilot-instructions.md` and applicable approved methodology/architecture/API artefacts remain authoritative. This document defines the mandatory handoff protocol between specialist roles.

---

## 1. Purpose

A handoff is an evidence boundary between specialist roles.

Its purpose is to ensure that:

- facts are not confused with assumptions;
- decisions are not confused with observations;
- unresolved questions are not silently resolved downstream;
- evidence is transferred with the responsibility for the next decision;
- scope cannot expand implicitly between stages;
- each specialist operates only within its authority;
- the complete engineering chain remains reconstructable.

The required chain for material work is:

`requirement → research → architecture → methodology → implementation → tests → benchmark → independent validation → audit`

A downstream role must not be required to reconstruct material upstream reasoning from conversation history alone.

---

## 2. When a Handoff Is Mandatory

A formal handoff is required when:

1. work moves between two specialist agents;
2. a material decision is transferred between roles;
3. a role identifies an unresolved issue owned by another role;
4. implementation, testing, benchmarking, validation, or audit status is claimed;
5. a B/C/D change progresses to another stage;
6. the lead records completion of a material workflow stage.

For a trivial A-class local defect, the Lead may determine that a full multi-stage handoff is unnecessary.

The Lead must record that determination.

---

## 3. Handoff Invariant

A stage is **not complete** merely because work was performed.

A stage is complete only when:

```text
work performed
    +
required evidence produced
    +
handoff complete
    +
next owner identified
```

Incomplete handoffs must not be silently completed by the receiving role.

The receiving role must either:

- return the handoff as incomplete; or
- explicitly record the missing information as an unresolved item.

The receiving role must not infer missing material decisions.

---

## 4. Mandatory Handoff Fields

Every material handoff must contain the following fields.

### 4.1 Handoff Identity

```text
handoff_id
date/time
source_role
receiving_role
change_id
change_class
```

`change_class` must be one of:

```text
A — local defect / implementation
B — architecture / boundary / ownership
C — methodology / mathematical / statistical
D — public/API contract
```

---

### 4.2 Scope

State:

```text
objective
in_scope
explicitly_out_of_scope
affected_packages
affected_public_surfaces
```

Scope must be concrete enough that a receiving agent can determine whether a proposed action is inside or outside the authorised change.

No downstream role may expand scope silently.

---

### 4.3 Requirements

Record the requirements that govern the work.

Where applicable identify:

```text
user requirement
approved methodology
approved architecture
approved API/public contract
applicable repository instructions
acceptance criteria
```

When authorities conflict, the conflict must be recorded rather than silently resolved.

---

### 4.4 Repository Facts

Facts discovered during the stage must be separated from interpretation.

Use the form:

```text
FACT
    statement
    evidence/reference
```

Facts must be based on observed repository state or authoritative artefacts.

Do not record hypotheses as facts.

---

### 4.5 Decisions

Every material decision must be classified as either:

```text
INHERITED DECISION
DECISION MADE BY THIS ROLE
```

For each decision record:

```text
decision
authority
evidence
rationale
status
```

A role may make decisions only within its defined authority.

---

### 4.6 Assumptions

All non-trivial assumptions must be explicitly recorded.

Use:

```text
ASSUMPTION
    statement
    reason
    owner
    materiality
```

An assumption must never be promoted to an approved methodology decision merely by repetition in later handoffs.

---

### 4.7 Open Questions

Every unresolved issue must be explicitly classified:

```text
OPEN
    question
    owner
    blocking: YES | NO
    required resolution stage
```

The owner is the role with authority to resolve the question.

A downstream role must not silently close an upstream-owned `OPEN` item.

---

### 4.8 Conflicts

Any contradiction between:

- methodology;
- architecture;
- API contract;
- implementation;
- tests;
- documentation;
- observed behaviour;

must be recorded as:

```text
CONFLICT
    issue
    competing positions
    authoritative source
    current status
    resolution owner
```

A conflict cannot be removed merely because one implementation happens to pass existing tests.

---

## 5. Evidence

Evidence must be distinguished by type.

Permitted classifications include:

```text
IMPLEMENTATION
TEST
ANALYTICAL_ORACLE
INVARIANT
REGRESSION
BENCHMARK
PROFILE
INDEPENDENT_VALIDATION
AUDIT
DOCUMENTATION
```

For each material claim record:

```text
claim
evidence type
evidence location/reference
result
exactness classification
```

Numerical evidence must be labelled where applicable:

```text
EXACT
NUMERICALLY APPROXIMATE
STATISTICAL
```

A claim cannot be promoted to a stronger evidence state without the corresponding evidence.

---

## 6. Files and Repository Changes

Record:

```text
files_inspected
files_added
files_modified
files_deleted
generated_files
```

For each production modification, state its purpose.

Do not report a file as changed unless it was actually changed.

Do not report a file as inspected unless it was actually inspected.

---

## 7. Testing

When tests were performed, record:

```text
test command
test scope
result
failures
known limitations
```

Tests must state their evidentiary nature where material.

Passing tests do not automatically establish:

- mathematical validity;
- methodology approval;
- architectural compliance;
- release readiness.

If no tests were run, state:

```text
TEST STATUS: NOT RUN
REASON: <explicit reason>
```

Never imply execution that did not occur.

---

## 8. Performance

Where performance is material, record:

```text
baseline
hypothesis
candidate
benchmark
numerical-equivalence result
decision
```

Benchmark evidence should record, where relevant:

```text
CPU
OS
Python version
NumPy/Numba/SciPy versions
thread configuration
problem dimensions
warm/cold state
repetitions
memory/allocation observations
```

Do not make material performance claims from intuition.

---

## 9. Ownership and Data Integrity

Where the change involves arrays, state, or mutable data, explicitly state:

```text
input ownership
output ownership
aliasing expectations
copy/view semantics
mutation guarantees
dtype
shape
order/layout
contiguity requirements
```

Any uncertainty about ownership is an unresolved issue and must be escalated before release.

---

## 10. Numerical and Methodological Assumptions

For material quantitative work, the handoff must identify the governing:

```text
mathematical definition
estimator/algorithm
finite-sample semantics
boundary behaviour
numerical domain
tolerance semantics
sign convention
units
stochasticity/RNG semantics
exactness classification
```

The implementation role must inherit these semantics rather than redefine them.

An implementation convenience is never, by itself, a methodology decision.

---

## 11. Residual Risks

Every handoff must identify known residual risks.

Use:

```text
RISK
    description
    severity
    model-risk materiality
    mitigation
    owner
    release blocking: YES | NO
```

Severity:

```text
BLOCKER
HIGH
MEDIUM
LOW
```

Model-risk materiality:

```text
IMMATERIAL
LOW
MODERATE
MATERIAL
CRITICAL
```

---

## 12. Acceptance Criteria

The source role must state exactly what the receiving role is expected to establish.

Acceptance criteria must be testable.

Examples:

```text
Every returned permutation row contains 0…S-1 exactly once.
```

```text
Input observation matrix is not mutated.
```

```text
The implementation agrees with the independent analytical oracle within the approved tolerance.
```

Avoid acceptance criteria expressed solely as:

```text
looks correct
is performant
is robust
is production-ready
```

---

## 13. Receiving-Role Obligations

The receiving role must:

1. verify that the handoff is complete enough to execute its responsibility;
2. respect upstream authority boundaries;
3. preserve inherited decisions unless formally escalated;
4. distinguish inherited facts from its own findings;
5. record newly discovered conflicts;
6. assign ownership for newly discovered unresolved issues;
7. produce the evidence required by its stage;
8. produce a subsequent handoff when the workflow continues.

The receiving role must not silently repair an upstream governance failure.

---

## 14. Escalation Rules

Escalate rather than infer when:

- methodology is ambiguous;
- public contract is ambiguous;
- architecture and implementation disagree;
- two authoritative artefacts conflict;
- ownership/aliasing is unclear;
- a required test oracle is unavailable;
- a material benchmark claim cannot be reproduced;
- release evidence is incomplete;
- scope expansion is proposed.

Routing:

```text
methodology / statistical ambiguity
    → quant-modeler

architecture / ownership / boundary ambiguity
    → riskquant-architect

implementation defect
    → implementation-engineer

test / oracle deficiency
    → test-engineer

performance evidence
    → performance-engineer

model-risk / implementation alignment
    → model-validator

release decision
    → riskquant-auditor

cross-role conflict / orchestration
    → riskquant-lead
```

The Lead may coordinate the conflict but must not override an authority's technical decision without an explicit recorded decision.

---

## 15. Handoff Completion States

A handoff must have exactly one state:

```text
READY
BLOCKED
RETURNED
SUPERSEDED
```

### READY

Required information and evidence are complete for the receiving role.

### BLOCKED

A material unresolved issue prevents downstream work.

### RETURNED

The receiving role has rejected the handoff because required information or evidence is missing.

### SUPERSEDED

A later approved decision replaces the handoff.

A `BLOCKED` or `RETURNED` handoff cannot be treated as complete.

---

## 16. Stage Completion

The Lead may mark a stage complete only when the applicable specialist has produced:

```text
required work
+
required evidence
+
handoff
+
no unowned blocking issue
```

Stage completion does not imply release readiness.

Evidence states remain independent:

```text
IMPLEMENTED
→ TESTED
→ BENCHMARKED
→ INDEPENDENTLY VALIDATED
→ AUDITED
→ RELEASED
```

A later state must never be inferred solely from an earlier state.

---

## 17. Independence Rules

The following separation is mandatory for material work:

```text
implementation-engineer
    does not approve its own methodology

test-engineer
    does not certify production implementation

performance-engineer
    does not certify mathematical validity solely from speed/equivalence

model-validator
    remains independent of implementation approval

riskquant-lead
    orchestrates but does not manufacture evidence

riskquant-auditor
    independently determines release readiness
```

Evidence generated by the implementation author cannot automatically constitute independent validation.

---

## 18. Minimum Handoff Template

The following template is the minimum acceptable structure for a material handoff:

```text
HANDOFF ID:
DATE/TIME:
SOURCE ROLE:
RECEIVING ROLE:
CHANGE ID:
CHANGE CLASS:

OBJECTIVE:

SCOPE
- In scope:
- Out of scope:
- Affected packages:
- Affected public surfaces:

GOVERNING AUTHORITY
- Methodology:
- Architecture:
- API/contract:
- Applicable instructions:

ACCEPTANCE CRITERIA
- ...

FACTS
- FACT:
  Evidence:

INHERITED DECISIONS
- Decision:
  Authority:
  Evidence:

DECISIONS MADE BY THIS ROLE
- Decision:
  Authority:
  Evidence:
  Rationale:

ASSUMPTIONS
- Assumption:
  Owner:
  Materiality:

OPEN QUESTIONS
- Question:
  Owner:
  Blocking:
  Resolution stage:

CONFLICTS
- Issue:
  Competing positions:
  Authority:
  Status:

FILES
- Inspected:
- Added:
- Modified:
- Deleted:

TEST EVIDENCE
- Command:
- Result:
- Exactness:
- Limitations:

BENCHMARK EVIDENCE
- Baseline:
- Candidate:
- Environment:
- Dimensions:
- Result:
- Numerical equivalence:

OWNERSHIP / DATA INTEGRITY
- Input ownership:
- Output ownership:
- Mutation:
- Aliasing:
- Shape/dtype/layout:

NUMERICAL SEMANTICS
- Definition:
- Estimator/algorithm:
- Boundary cases:
- Tolerances:
- Units/sign:
- RNG semantics:
- Exactness:

RESIDUAL RISKS
- Risk:
  Severity:
  Model-risk materiality:
  Owner:
  Release blocking:

NEXT ROLE:

NEXT-ROLE ACCEPTANCE CRITERIA:

HANDOFF STATE:
```

---

## 19. Prohibited Handoff Patterns

The following are not acceptable:

```text
"Looks good."
"Tests pass."
"No issues found."
"Implemented as discussed."
"Performance is improved."
"Validated."
"Production ready."
```

unless the underlying evidence required by this contract is also supplied.

Likewise prohibited:

```text
silent methodology decisions
silent scope expansion
silent conflict resolution
unsupported performance claims
unsupported validation claims
unsupported release claims
```

---

## 20. Governing Principle

The handoff is part of the engineering artefact.

A material RiskQuant change must remain reconstructable without relying on undocumented conversation context.

The required chain is:

`requirement → evidence → decision → implementation → tests → benchmark → independent validation → audit`

A broken evidentiary link is a governance finding, even when the implementation appears correct.