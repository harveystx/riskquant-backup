# WS_FACTOR_REGRESSION_GUARDS: Freeze reconciliation

Path: docs/workstreams/WS_FACTOR_REGRESSION_GUARDS/R_FREEZE_RECONCILIATION_R2.md
Date: 2026-10-21
Spec: WS_FACTOR_REGRESSION_GUARDS.md, revision 2
Round: 2 of 2
Clauses covered: C.4, C.13; tests S1.T5, S2.T3; the O-2 owner decision text; the header Snapshot line.
Clauses not covered: none in scope. Round 1 covered C.1 to C.13. This round does not re-audit C.1 to C.3 or C.5
to C.12.
Snapshot: SNAPSHOT commit=3b39de7 dirty=true content_hash=7461aad8accf3519
Mode: read-only
Evidence base: the spec, revision 2; R_FREEZE_RECONCILIATION_R1.md; context_riskquant_index.txt (project copy);
context_riskquant_tests.txt; context_riskquant_solutions_factor_regression.txt (Exclusion, _check_int_param,
wald_test only); DECISIONS.md (headings of D001, D002, D064, D104, D105); WS_TEMPLATE.md; CLAUDE.md; WORKFLOW.md.
Evidence limits: No code ran. No gate ran. No test ran. Every statement comes from reading text or source.
Revision 1 of the spec was not attached (live version rule, WS_TEMPLATE.md section 1.1). No git diff was attached.
For this reason, the claim "revision 2 changes nothing else" is checked only against the revision 1 text that the
round 1 report quotes. No uploaded index file exists. The only index is the project copy. Its first line equals the
round 1 Snapshot line. It is not known whether this index was regenerated after revision 2.

## A. Verdict

NOT READY

B-1 and B-2 are CLOSED. One new BLOCKING finding exists: B-3, the header Snapshot line holds a placeholder and a
dirty flag that the index contradicts. Next action: the owner chooses per WS_TEMPLATE.md section 1.6 (round 2 used).

## B. Authority reconciliation

| Gate | Status | Repository evidence | Remaining action |
|---|---|---|---|
| Tripwire (section 1.8) | VERIFIED | Header: Revision 2, Code merged no. The tripwire is not tripped. A revision 3 with Code merged no trips it | owner: see B-3 |
| Audit budget (section 1.6) | VERIFIED | Header: Audit rounds used 1 of 2 before this round. This report is round 2 | owner: choose an option after round 2 |
| Revision scope (BLOCKING only) | NOT VERIFIED | Each changed line in the table below is in the permitted list (Observed). The revision 1 text that round 1 quotes for R-1, R-2, R-3, R-9 and R-10 stays verbatim, so no refinement entered the body (Observed). The absence of other changes is Inferred: no diff was attached | owner: confirm with `git diff` on the spec before the status change |
| D001, D002, D064, D104, D105 | VERIFIED | Each ID exists in DECISIONS.md (lines 13, 21, 3057, 5066, 5084). No changed line in revision 2 cites or changes a decision | none |
| O-2 owner rewording (O-A-1) | VERIFIED | New first sentence: "SeriesContractError for the value checks of C.9". It agrees with the Scope sentence, the Rule sentence, C.8 (TypeError) and C.9 (SeriesContractError). The round 1 conflict is gone | none |
| O-2 authorship | NOT VERIFIED | The revision note and a parenthetical attribute the rewording to the owner. No file can prove authorship | owner: confirm |
| Other 2026-10-21 owner decisions (O-1, O-3, O-4, O-5) | NOT VERIFIED | Round 1 quotes none of these texts. No attached evidence shows a change (Inferred). The O-2 Imports line still names "message assertions in S2.T2" only | owner: the same `git diff` check |
| Snapshot line equals index line 1 | NOT VERIFIED | Spec: `SNAPSHOT commit=3b39de7 dirty=false content_hash=<copy from the refreshed context_riskquant_index.txt>`. Index line 1: `SNAPSHOT commit=3b39de7 dirty=true content_hash=7461aad8accf3519` | B-3 |
| Status move to REVIEW or FROZEN | OWNER APPROVAL REQUIRED | Header: Status DRAFT | owner |

Changed lines (Observed in revision 2; change from revision 1 Inferred from the round 1 report):

| Location | Change | In the permitted list |
|---|---|---|
| Header: Revision, Revision note, Date, Snapshot, Independent review, Audit rounds used | header update | yes |
| C.4 | names the inputs of S1.T1 and S1.T2 | yes |
| S1.T5 | case-by-case Expected; filter on the wald_core call only | yes |
| C.13 | Test field names S2.T2 and S2.T3 | yes |
| S2.T3 | Expected adds the message rule | yes |
| O-2, first sentence | owner rewording | yes |
| Section 7, first backlog line | records R-1 to R-15 and O-A-1 | yes |

## C. Contract-to-repository matrix

| Clause | Repository evidence | Status | Finding |
|---|---|---|---|
| C.4 | Text: "for the inputs of C.1 and for the inputs of S1.T1 and S1.T2". This closes the B-1 gap: under O-1 = B, S1.T1 and S1.T2 return W and are not C.1 inputs. The clause names wald_test. S1.T5 puts only the wald_core call under the "error" filter. `api.py:wald_test` runs wald_core inside `_facade_errors`, then calls `digest.restriction_digest` and builds WaldResult | MATCH WITH REQUIRED CHANGE | R-16 |
| S1.T5 | Expected: S1.T1 and S1.T2 give W as in C.3; S1.T3 and S1.T4 give CoreRestrictionError(RESTRICTION_NON_FINITE); no RuntimeWarning in any of the four. Only the wald_core call runs under `warnings.simplefilter("error")`. Each item of the round 1 B-1 required change is present. The expected results agree with C.1, C.3 and C.4 | MATCH | none |
| C.13 | Test field: "S2.T2, S2.T3". This closes B-2 | MATCH | none |
| S2.T3 | Expected: "TypeError; the message names the field and, where one exists, the exclusion index". This is the B-2 required text. The case list has no case for 4 C.8 items: a unit that is not str, Exclusion.series not str, Exclusion.reason not str, and end_ord not an integer. `contracts.py:Exclusion` has no `__post_init__`, so a non-str series or reason constructs | MATCH WITH REQUIRED CHANGE | R-17 |
| O-2 text | See section B. The first sentence, the Scope sentence and the Rule sentence agree. C.8 and C.9 agree with O-2 | MATCH | R-19 |

## D. Blast radius of shared interfaces

Revision 2 changes only spec text. It changes no shared interface relative to revision 1. The round 1 section D
stands. No search command ran in this round. Reason: no new symbol, caller or file enters the spec in revision 2.

## E. Hidden dependency findings

* E-5. `contracts.py:Exclusion` is a frozen slots dataclass with no `__post_init__` (line 1497). Its fields
  `series: str`, `date_ord: int` and `reason: str` are annotations only. Consequence: every C.8 type check on an
  Exclusion field is new behavior in AlignedLevels. S2.T3 must exercise each one. (Observed)
* E-6. In `api.py:wald_test`, the only numerical call is wald_core (line 1178). The other calls are a digest and a
  dataclass construction. Consequence: a RuntimeWarning from wald_test without one from wald_core is unlikely.
  (Inferred)

## F. Findings and required changes

Carried from earlier rounds:

* B-1. CLOSED. Evidence: C.4 text and the S1.T5 row in revision 2 (section C). Each item of the required change is
  present: case-by-case results, no RuntimeWarning in the four cases, the filter on the wald_core call only, and the
  C.4 extension to the S1.T1 and S1.T2 inputs.
* B-2. CLOSED. Evidence: the C.13 Test field names S2.T3. The S2.T3 Expected column holds the required message text
  verbatim.

BLOCKING:

* B-3. The header Snapshot line does not equal the first line of the index.
  * Spec: `SNAPSHOT commit=3b39de7 dirty=false content_hash=<copy from the refreshed context_riskquant_index.txt>`.
  * Index line 1: `SNAPSHOT commit=3b39de7 dirty=true content_hash=7461aad8accf3519`. (Observed)
  * Two faults exist. The content_hash is a placeholder. The dirty flag contradicts the only index available.
  * Consequence: WS_TEMPLATE.md section 1.3 forbids a placeholder at FROZEN. The spec states a code identity that no
    attached evidence supports. C.6 and S1.T8 take golden values "at 3b39de7". With dirty=true, that commit is not
    the snapshot state (round 1 R-9).
  * Unresolved: the owner may have committed the tree after the index was generated. Then dirty=false can be true,
    and the index is stale.
  * Required change: regenerate the index from the tree at freeze. Copy its first line verbatim into the Snapshot
    field.
  * Tripwire note: a revision 3 with Code merged no trips section 1.8. Whether a header identity edit counts as a
    revision is an owner decision. [Proposed] Treat it as an owner header edit made with the status change, as for
    Status. The owner decides.

REFINEMENT:

* R-16. C.4 names wald_test. No test runs wald_test under the "error" filter. S1.T6 runs wald_test with no filter.
  Risk is low (E-6). Action: in the S1 prompt, run the S1.T6 wald_test call under the "error" filter, or narrow C.4
  to wald_core in a later revision.
* R-17. S2.T3 does not cover 4 C.8 items (section C, E-5). Action: in the S2 prompt, require one S2.T3 case per C.8
  item, with the C.13 message assertion. The prompt narrows the slice. It does not widen it (section 1.4).
* R-18. The backlog line carries R-4, R-5, R-6, R-7, R-8, R-10, R-13 and R-15 into the prompts "as prompt-level
  owner decisions". R-6, R-7 and R-8 are contract content of C.8 and C.9. R-8 needs owner choices that are not
  recorded. The WORKFLOW.md section 3 prompt form has no field for them. Consequence: the S2 reviewer cannot check
  these items against the frozen spec (input F). Action: the owner records the R-8 choices before the S2 prompt.
* R-19. The O-2 parenthetical "first sentence reworded by the owner on 2026-10-21, O-A-1" and a two-entry revision
  note are correction-log text (section 1.3). Lines 7, 50 and 54 exceed the wrap width of the body. Action:
  backlog. No acceptance condition changes.

Owner approval only:

* O-A-2. Choose one section 1.6 option after round 2: freeze with named known gaps, descope, or stop. If the choice
  is freeze, B-3 must be fixed first, because a placeholder cannot stay at FROZEN.
* O-A-3. Decide whether the B-3 header edit is a revision (tripwire consequence, section 1.8).
* O-A-4. Confirm the O-2 rewording as owner text (section B).
* O-A-5. Set the status. A report never changes the status.

Value case challenge: no change since round 1. S1 holds. S2 holds conditionally on the 2026-10-20 owner decision
that a hand-built AlignedLevels is a supported input.

## G. Final freeze gate

| Check | Result |
|---|---|
| Contract internally consistent | yes (B-1, B-2 closed; C.4, C.13, S1.T5, S2.T3 and O-2 agree) |
| Consistent with `DECISIONS.md` | yes |
| Implementable with existing architecture | yes |
| Blast radius bounded | yes (unchanged from round 1) |
| Existing behavior protected | conditionally (R-15; scripts/ not searched; carried from round 1) |
| Value case holds: the smallest alternative fails, with evidence | yes (S1); conditionally (S2, owner decision) |
| Remaining issues are approvals, not technical gaps | no (B-3) |

Verdict: NOT READY
