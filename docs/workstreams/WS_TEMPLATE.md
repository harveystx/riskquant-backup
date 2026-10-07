# WS_TEMPLATE

This file is the standard for RiskQuant workstream documents.

* Part 1 gives the rules.
* Part 2 gives the skeleton of a workstream spec. Copy it. Delete Part 1.
* Part 3 gives the skeleton of a freeze reconciliation report (HEAVY class).
* Part 4 gives the skeleton of a brief (LIGHT class).

`CLAUDE.md` stays in force. If this file and `CLAUDE.md` conflict, `CLAUDE.md`
wins.

---

## Part 1. Rules

### 1.1 Document types

| Type | Path | Purpose |
|---|---|---|
| Spec | `docs/workstreams/WS_<NAME>.md` | HEAVY class. One per workstream. Holds the goal, the contract, and the slices. |
| Brief | `docs/workstreams/WS_<NAME>.md` | LIGHT class. One slice, S1. About 80 lines. See Part 4. |
| Appendix | `docs/workstreams/WS_<NAME>/A_<TOPIC>.md` | Detail for named slices. Attach it only to the slices that name it. |
| Report | `docs/workstreams/WS_<NAME>/R_<KIND>_<YYYY-MM-DD>.md` | Dated evidence from an audit, reconciliation, or validation. |

* `<NAME>` is the workstream name in the style of the instrument enum, for
  example `VANILLA_IRS`. An optional goal suffix is allowed, for example
  `VANILLA_FORWARD_NODE_RISK_ENABLEMENT`.
* The git branch is `ws/<name-with-hyphens>`.
* A report is evidence. It is never the contract. The contract lives in the spec
  (HEAVY) or in the acceptance section of the brief (LIGHT).
* A spec and a brief share one path. The `Class:` header field tells them apart.
  A workstream has one of them at a time.
* When the class changes, the new document keeps the path. First move the old
  document to `docs/workstreams/WS_<NAME>/A_PRIOR_REV<n>.md`. Do not delete it.
  The new document cites it in its backlog. The new document continues the
  `Revision` count of the old document. A class change does not reset the
  tripwire (section 1.8).
* Do not rename or rewrite a CLOSED document.

### 1.2 Status lifecycle

| Status | Meaning | Scope authority | Set by |
|---|---|---|---|
| `DRAFT` | Proposed plan. Not reviewed. | No | Any chat that drafts it |
| `REVIEW` | Independent audit in progress. | No | Owner |
| `FROZEN` | Owner approved in writing. | Yes, for the contract and the slices | Owner |
| `CLOSED` | All slices validated. Closure record complete. | Historical | Owner |
| `ABANDONED` | Stopped. Reason recorded. | No | Owner |

Rules:

* A chat never sets `REVIEW`, `FROZEN`, `CLOSED`, or `ABANDONED`. The owner sets
  them.
* After `FROZEN`, a contract change needs a new revision and owner approval.
* A LIGHT brief skips `REVIEW`. The owner moves it from `DRAFT` to `FROZEN`
  without a spec audit.
* `DECISIONS.md` wins over a workstream document. The snapshot wins for code
  state. A workstream document never proves code state.
* The audit that moves a HEAVY spec from `DRAFT` to `REVIEW` must run in a fresh
  chat. The chat that wrote the draft cannot audit it. A review by the model that
  implemented the work is not independent.

### 1.3 Writing rules

* Write in ASD-STE100. Keep identifiers, formulas, dtypes, paths, and decision
  IDs exact. Do not use em dashes. Use LF line endings.
* Use only these labels: Observed, Inferred, Unresolved, Proposed
  (`CLAUDE.md` section 7). A report may also use the verdict words in Part 3.
* Cite code as `path:symbol`.
* Use these IDs: facts `F-n`, contract clauses `C.n`, proposed decisions
  `P-<NAME>-NNN`, owner gates `O-n`, slices `S1` to `Sn`, tests `S<n>.T<m>`.
* Do not assign a `D` number. The owner assigns it. After the owner assigns it,
  replace the `P-` ID in a new revision.
* Do not keep a correction log in the document. The header holds the revision
  number and one revision note. Git holds the history.
* Do not leave a placeholder at `FROZEN` or `CLOSED`.
* Do not restate text from `CLAUDE.md` or `DECISIONS.md`. Cite section numbers
  and decision IDs.
* Make every contract clause testable. Each clause names a test or a gate in a
  slice.
* State what was not executed in the header field `Evidence limits`.
  Never write a pass result that came only from reading source.
* Write the value case in section 1 from the value assessment of the
  investigation. Do not estimate hours or tokens. The freeze audit challenges
  the value case.
* Keep the core spec near 300 lines. Move slice detail of more than 40 lines to
  an appendix. Name the appendix in the slice.

### 1.4 Authority in a prompt

A prompt that contains `WS: <path> S<n>` makes the contract clauses named in
`S<n>`, and the fields of `S<n>`, the scope of that task.

* The implementer reads the header, section 1, the clauses named in `S<n>`, and
  `S<n>`. It reads the rest only when the slice requires it.
* For an implementation task, the status must be `FROZEN`. If it is not, the
  implementer reports this and stops.
* An audit task may read a `DRAFT` or `REVIEW` document.
* A prompt may narrow the slice. A prompt never widens it.

### 1.5 Weight classes

Classify every workstream as HEAVY or LIGHT before drafting. The class depends
on contract impact. Slice count, file count, and owner-gate count never decide
the class.

HEAVY if at least one of these is true:

* The work changes a `src/` production contract: a public signature, a shared
  interface with callers in 2 or more modules, an ABI, a hash or serialization
  identity, or a numerical contract.
* The work needs a new `D` decision.
* The work changes admission or validation behavior.

LIGHT in all other cases. Examples: release scripts, demos, benchmarks,
certification worlds in `tests/`, test-only work, and documentation.

| Class | Document | Audit | Prompts | Review |
|---|---|---|---|---|
| HEAVY | Spec (Part 2) | Freeze audit, at most 2 rounds (section 1.6) | One per slice | Code review per slice |
| LIGHT | Brief (Part 4) | None. The owner freezes the brief. | One implementation prompt | One code review |

Rules:

* A LIGHT brief has one slice, S1. S1 covers sections 2 to 5 of the brief.
* If LIGHT work shows a HEAVY criterion, stop. Report it to the owner. The owner
  reclassifies.
* A reference such as "like X" imports the requirements of X. List what parity
  imports and what v1 omits. Parity is a cost, not a default.
* Put what v1 omits in the backlog. Do not widen v1 to include it.
* Owner decisions after revision 1: before the drafter applies one, it writes
  one line: what the decision imports, with cost proxies. If the decision
  widens the scope, the drafter writes `SCOPE INCREASE` and waits for owner
  confirmation.

### 1.6 Audit budget

A HEAVY spec has at most 2 audit rounds.

* Label each finding BLOCKING or REFINEMENT.
  * BLOCKING: the spec as written would make the implementation violate a
    locked decision, break a production contract, give a wrong or unverifiable
    result, or contradict itself.
  * REFINEMENT: wording, extra detail, an edge case outside the stated scope, or
    a platform detail that does not change an acceptance condition.
* Only BLOCKING findings cause a revision. Put each REFINEMENT finding in the
  backlog (spec section 7), or leave it to implementation.
* After round 1, a revision applies the BLOCKING findings only. Then round 2
  runs.
* After round 2, the owner chooses one option: freeze with named known gaps,
  descope, or stop. Round 3 needs an explicit owner decision.
* Audit scope: attach only the context profiles that the contract clauses name.
  If the audit needs more than 5 profiles, split it by clause range.
* Each report lists `Clauses covered` and `Clauses not covered`. An audit that
  stops early, for example at a tool limit, counts as a round. The next round
  covers only the clauses that were not covered.

### 1.7 Pre-delivery self-check

The drafter runs this check before delivery. It applies to a spec and to a
brief.

1. **Quantifiers.** For each clause and each owner decision that contains "all", "every", "never", or
   "only", list the set that the clause covers. Check each member against the
   facts or the code. State the types that the clause covers. Example: "all
   fields" must say whether it includes boolean and string fields.
2. **Status values.** Every status, verdict, and label in the document appears
   in an enumeration of this template. Do not invent a value.
3. **Contradictions.** Compare every gate and every prohibition with every
   clause and every owner decision. A gate must not forbid what a clause uses.
4. **Tests.** Every clause names a test or a gate that exists in a slice.
5. **References.** Every referenced ID (C, F, O, S, P, D) exists.
6. **Result.** Report `SELF-CHECK: passed`, or list the corrections made.

### 1.8 Effort tripwire

This rule applies to every class. It limits the cost of a wrong early judgment.

* The header fields `Revision` and `Code merged` control the tripwire.
* If `Revision` is 3 and `Code merged` is `no`, stop. Do not revise or audit
  again.
* The next output is a value re-check of at most 6 lines: the deliverable, what
  the revisions changed, and three options. The options are: build v1 now from
  the current text, descope, or stop. The owner chooses.
* Revision 4 without merged code needs an explicit owner decision. Record it in
  the revision note.
* Every chat that drafts, revises, or audits a document checks the tripwire
  first.

---

## Part 2. Spec skeleton

Copy from the next line to the end of this part.

````markdown
# WS_<NAME>: <title>

Path: docs/workstreams/WS_<NAME>.md
Class: HEAVY
Status: DRAFT
Revision: 1
Revision note: <one line>
Date: <YYYY-MM-DD>
Branch: ws/<name>
Snapshot: <first line of context_riskquant_index.txt>
Drafted by: investigator | implementer | owner
Independent review: NONE | PENDING | PASSED (<snapshot>)
Audit rounds used: 0 of 2
Code merged: no | yes (<commit>)
DECISIONS.md reconciled: NO | YES (<date>)
Evidence limits: <what was not executed or not available>

## 1. Goal and scope

Goal: <one sentence>

Value case:
- Benefit: <what changes and what depends on it>
- Cost proxies: <slices, files, shared interfaces, new decisions, audit rounds>
- Smallest alternative and why it fails: <alternative, evidence>
- Case against: <strongest argument against, with evidence>
- Recommendation at drafting: PROCEED | PROCEED SMALLER [Proposed]

In scope:
- <item>

Out of scope:
- <item>

Owner decisions in force:
- <date>: <decision>

Depends on: <workstreams or decisions, or none>
Blocks: <workstreams, or none>

## 2. Verified facts

| Id | Fact | Evidence | Label |
|---|---|---|---|
| F-1 | <fact> | `<path>:<symbol>` | Observed |

## 3. Contract

- C.1 <testable clause>. [<label>] Test: S1.T1
- C.2 <testable clause>. [<label>] Test: S1.T2

## 4. Decisions

Decisions relied on: <D-ids>. Each checked against `DECISIONS.md`: yes | no.

### P-<NAME>-001: <title> (PROPOSED)

```text
Related: <D-ids>
Scope: <one sentence>
1. <clause>
2. <clause>
Evidence: <tests or gates that prove the decision>
```

## 5. Owner gates

| Id | Question | Status | Closed on | Evidence |
|---|---|---|---|---|
| O-1 | <question> | OPEN | | |

Status values: OPEN, CLOSED.

## 6. Slices

Order: S1, then S2. State which slices may run in parallel.

### S1: <title>

Objective: <one sentence>
Contract: <C-ids that this slice implements>
Depends on: <slice ids, owner gates, or none>
Files:
- create: <paths>
- modify: <paths>
Prohibited: <paths, behaviors, or changes>
Gates: <exact commands, in order>
Exit: <observable conditions>
Status: NOT_STARTED | IN_PROGRESS | DONE | VALIDATED
Evidence: <commit or report path, or none>

| Id | Case | Expected | Defect caught |
|---|---|---|---|
| S1.T1 | <case> | <result> | <defect that this test detects> |

Appendix: <A_TOPIC.md, or none>

## 7. Risks and deferred

| Risk | Consequence | Mitigation |
|---|---|---|
| <risk> | <consequence> | <mitigation> |

Deferred:
- <item>: opens when <condition>

Backlog (REFINEMENT findings and items that v1 omits):
- <item>

## 8. Closure record

Fill this section when the status becomes CLOSED. Leave no placeholder.

| Item | Value |
|---|---|
| Final commit | <hash, dirty=false> |
| Release tag | <tag> |
| Decisions locked | <D-ids> |
| Independent validation | <date and snapshot> |

Delivered:
- <item>

Deferred:
- <item>

Open work: none
````

---

## Part 3. Freeze reconciliation report skeleton

Use this skeleton for an audit that decides whether a HEAVY `DRAFT` spec can
move to `REVIEW` and then `FROZEN`. Sections 1.6 and 1.8 limit the audit. The audit runs in a fresh chat, in analysis mode
(`CLAUDE.md` section 1). It changes no file in `src/`, `tests/`, or
`DECISIONS.md`.

Path: `docs/workstreams/WS_<NAME>/R_FREEZE_RECONCILIATION_<YYYY-MM-DD>.md`

````markdown
# WS_<NAME>: Freeze reconciliation

Path: <path>
Date: <YYYY-MM-DD>
Spec: WS_<NAME>.md revision <n>
Round: <1 or 2> of 2
Clauses covered: <C-ids>
Clauses not covered: <C-ids, or none>
Snapshot: <first line of context_riskquant_index.txt>
Mode: read-only
Evidence base: <files read>
Evidence limits: <for example: test execution not available. Every test
statement comes from source. No pass claim.>

## A. Verdict

FREEZE READY | FREEZE READY PENDING OWNER APPROVAL | NOT READY

<three lines at most: the reason, the blockers, and the next action>

Rule: the verdict is NOT READY only if a BLOCKING finding exists.

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|

Status values: VERIFIED, NOT VERIFIED, OWNER APPROVAL REQUIRED.
Include the check of `DECISIONS.md` against each decision that the spec relies
on.

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|

Status values: MATCH, MATCH WITH REQUIRED CHANGE, CONTRADICTION, NOT VERIFIED.

## D. Blast radius of shared interfaces

For each shared interface that the spec changes, list every caller with
`path:symbol`. State the search command and the files searched.

## E. Hidden dependency findings

<findings, each with evidence and consequence>

## F. Findings and required changes

Label each finding BLOCKING or REFINEMENT (section 1.6).

- BLOCKING: <changes to the spec text, with evidence>
- REFINEMENT: <items for the backlog or for implementation>
- Owner approval only: <gates>

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | yes / no / not verifiable |
| Consistent with `DECISIONS.md` | yes / no / not verifiable |
| Implementable with existing architecture | yes / no |
| Blast radius bounded | yes / no |
| Existing behavior protected | yes / conditionally / no |
| Value case holds: the smallest alternative fails, with evidence | yes / no / not verifiable |
| Remaining issues are approvals, not technical gaps | yes / no |

Verdict: <the verdict from section A>
````

After the audit, the owner decides. If the audit has BLOCKING findings, the
drafter issues a new revision that applies the BLOCKING findings of section F
only. The owner then sets the
status. A report never changes the status.

---

## Part 4. Brief skeleton (LIGHT class)

Use this skeleton for LIGHT work (section 1.5). The owner reads the brief and
sets `Status: FROZEN`. No spec audit runs. One implementation prompt follows.
One code review follows the implementation.

The brief has one slice, S1. S1 covers sections 2 to 5.

Keep the brief near 80 lines.

````markdown
# WS_<NAME>: <title>

Path: docs/workstreams/WS_<NAME>.md
Class: LIGHT
Status: DRAFT
Revision: 1
Code merged: no | yes (<commit>)
Date: <YYYY-MM-DD>
Branch: ws/<name>
Snapshot: <first line of context_riskquant_index.txt>
Drafted by: investigator | implementer | owner
Evidence limits: <what was not executed or not available>

## 1. Goal and value

Goal: <one sentence>
Benefit: <one line>
Case against: <one line, with evidence>
Reclassify trigger: Stop and tell the owner if the work needs a `src/`
production contract change or a new `D` decision.

## 2. Scope

In scope (v1):
- <item>

Out of scope (see section 6):
- <item>

Parity: <reference artifact. What v1 copies. What v1 omits.>

## 3. Owner decisions in force

- <date>: <decision>

A frozen brief has no open owner decision.

## 4. Acceptance

- C.1 <testable clause>. [<label>] Verified by: <gate or test>
- C.2 <testable clause>. [<label>] Verified by: <gate or test>

Files:
- create: <paths>
- modify: <paths>
Prohibited: <paths, behaviors, or changes>

## 5. Gates

<exact commands, in order>

Evidence format: raw output of each command, in the final report.

## 6. Backlog (v2)

- <item>: <why v1 omits it>

## 7. Closure

| Item | Value |
|---|---|
| Final commit | <hash, dirty=false> |
| Date | <YYYY-MM-DD> |
| Open work | none |
````
