# WS_TEMPLATE

This file is the standard for RiskQuant workstream documents.

* Part 1 gives the rules.
* Part 2 gives the skeleton of a workstream spec. Copy it. Delete Part 1.
* Part 3 gives the skeleton of a freeze reconciliation report.

`CLAUDE.md` stays in force. If this file and `CLAUDE.md` conflict, `CLAUDE.md`
wins.

---

## Part 1. Rules

### 1.1 Document types

| Type | Path | Purpose |
|---|---|---|
| Spec | `docs/workstreams/WS_<NAME>.md` | One per workstream. Holds the goal, the contract, and the slices. |
| Appendix | `docs/workstreams/WS_<NAME>/A_<TOPIC>.md` | Detail for named slices. Attach it only to the slices that name it. |
| Report | `docs/workstreams/WS_<NAME>/R_<KIND>_<YYYY-MM-DD>.md` | Dated evidence from an audit, reconciliation, or validation. |

* `<NAME>` is the workstream name in the style of the instrument enum, for
  example `VANILLA_IRS`. An optional goal suffix is allowed, for example
  `VANILLA_FORWARD_NODE_RISK_ENABLEMENT`.
* The git branch is `ws/<name-with-hyphens>`.
* A report is evidence. It is never the contract. The contract lives in the spec.
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
* `DECISIONS.md` wins over a workstream document. The snapshot wins for code
  state. A workstream document never proves code state.
* The audit that moves a document from `DRAFT` to `REVIEW` must run in a fresh
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

---

## Part 2. Spec skeleton

Copy from the next line to the end of this part.

````markdown
# WS_<NAME>: <title>

Path: docs/workstreams/WS_<NAME>.md
Status: DRAFT
Revision: 1
Revision note: <one line>
Date: <YYYY-MM-DD>
Branch: ws/<name>
Snapshot: <first line of context_riskquant_index.txt>
Drafted by: investigator | implementer | owner
Independent review: NONE | PENDING | PASSED (<snapshot>)
DECISIONS.md reconciled: NO | YES (<date>)
Evidence limits: <what was not executed or not available>

## 1. Goal and scope

Goal: <one sentence>

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

Use this skeleton for an audit that decides whether a `DRAFT` spec can move to
`REVIEW` and then `FROZEN`. The audit runs in a fresh chat, in analysis mode
(`CLAUDE.md` section 1). It changes no file in `src/`, `tests/`, or
`DECISIONS.md`.

Path: `docs/workstreams/WS_<NAME>/R_FREEZE_RECONCILIATION_<YYYY-MM-DD>.md`

````markdown
# WS_<NAME>: Freeze reconciliation

Path: <path>
Date: <YYYY-MM-DD>
Spec: WS_<NAME>.md revision <n>
Snapshot: <first line of context_riskquant_index.txt>
Mode: read-only
Evidence base: <files read>
Evidence limits: <for example: test execution not available. Every test
statement comes from source. No pass claim.>

## A. Verdict

FREEZE READY | FREEZE READY PENDING OWNER APPROVAL | NOT READY

<three lines at most: the reason, the blockers, and the next action>

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

## F. Required changes before freeze

- Mandatory (technical): <changes to the spec text>
- Owner approval only: <gates>
- Implementation detail (not in the contract): <items>

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | yes / no / not verifiable |
| Consistent with `DECISIONS.md` | yes / no / not verifiable |
| Implementable with existing architecture | yes / no |
| Blast radius bounded | yes / no |
| Existing behavior protected | yes / conditionally / no |
| Remaining issues are approvals, not technical gaps | yes / no |

Verdict: <the verdict from section A>
````

After the audit, the owner decides. If the audit requires spec changes, the
drafter issues a new revision that applies section F. The owner then sets the
status. A report never changes the status.
