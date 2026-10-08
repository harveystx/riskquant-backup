# RiskQuant Workflow

This file is a cheat sheet for the owner. It also serves as input R of
`ORCHESTRATOR.md`. It defines no rule that another file defines. It points to
that file.

---

## 1. Roles

| Role | Files to attach | Use when | Output |
|---|---|---|---|
| Implementer | `CLAUDE.md`, the prompt, `DECISIONS.md`, files named in the prompt | A prompt exists | Code and a final report |
| Reviewer | `REVIEWER.md`, `CLAUDE.md`, `DECISIONS.md` | An implementer report exists | Verdict and next prompt |
| Investigator | `INVESTIGATOR.md`, `WS_TEMPLATE.md`, `CLAUDE.md`, `DECISIONS.md`, context files | A question or a plan is needed | Answer, value assessment, draft |
| Orchestrator | `ORCHESTRATOR.md`, `WORKFLOW.md`, `CLAUDE.md`, the live workstream document | You do not know the next step | Route and message |

Use a fresh chat for each task.

---

## 2. Routing table

| You hold | Next step |
|---|---|
| Investigator answer with a spec draft (HEAVY) | Fresh audit chat. Use the prompt that the investigator wrote. |
| Brief draft (LIGHT) | Read it. Set `Status: FROZEN`. Then run the implementer. |
| Audit report, BLOCKING findings, rounds used below 2 | Fresh revision chat. Apply the BLOCKING findings only. |
| Audit report, BLOCKING findings after round 2 | You decide: freeze with named gaps, descope, or stop. |
| Audit report, no BLOCKING findings | Set `Status: FROZEN`. Move refinements to the backlog. |
| Frozen document, a slice is not done | Implementer chat with the standard prompt (section 3). |
| Implementer report | Reviewer chat (section 4). |
| Reviewer ACCEPT | Set the slice status as the reviewer proposes. Next slice, or close. |
| Reviewer ACCEPT WITH FIXES or REJECT | Send the reviewer's next prompt to a fresh implementer chat. |
| Reviewer BLOCKED | Send the missing inputs in the same reviewer chat. |
| All slices validated | Commit. Set `Code merged`. Set `Status: CLOSED`. No chat. |
| Revision 3 or more, no merged code | Stop. Choose: build v1 now, descope, or stop. |
| Any other case | Orchestrator chat (section 5). |

---

## 3. Standard implementation prompt

Use this for a slice of a `FROZEN` document. The slice holds the files, the
prohibitions, the gates, and the attach list.

```
MODE: implementation
TASK: Implement slice S<n> of the attached workstream document.
WS: <live document path> S<n>
ATTACH: <the Attach field of S<n>> ; DECISIONS.md
FIRST: <only if S<n> is IN_PROGRESS> Compare the snapshot with S<n>. List what exists. Do not redo it.
REPORT: final report per CLAUDE.md section 11
STOP AND ASK: a conflict with DECISIONS.md, or a need to leave the Files list of S<n>.
```

---

## 4. Reviewer message

Attach `REVIEWER.md`, `CLAUDE.md`, `DECISIONS.md`. For `FULL`, also attach the
refreshed context files and the live workstream document. For `QUICK`, attach
no context files.

```
Review this implementation per REVIEWER.md.
DEPTH: FULL | QUICK

[A] ORIGINAL PROMPT
<the exact prompt that the implementer received>

[B] IMPLEMENTER REPORT
<the final report only>

[C] CODE STATE
<FULL: "Refreshed context files are attached. Changed files: <paths>."
 QUICK: the output of git diff --ignore-cr-at-eol for the changed files>

[D] GATE OUTPUT
<raw output of your own gate run, with the exit lines>
```

---

## 5. Orchestrator message

```
Route per ORCHESTRATOR.md.

[W] <attach the live workstream document>
[O] <paste the latest output, or write: none>
[I] <one line: what you want>
```

---

## 6. Gate run (PowerShell, repository root)

Replace `$f` with the changed files. Do not add `--follow-imports=skip`. Do not
run `ruff format` without `--check`.

```powershell
$f = "<path1>", "<path2>"
& {
  "== ruff check";          ruff check $f;          "exit=$LASTEXITCODE"
  "== ruff format --check"; ruff format --check $f; "exit=$LASTEXITCODE"
  "== mypy";                mypy $f;                "exit=$LASTEXITCODE"
  "== pytest";              pytest <test paths> -q; "exit=$LASTEXITCODE"
} *>&1 | Tee-Object -FilePath gate_output.txt
```

---

## 7. Standing rules

* Only the owner sets a workstream status.
* Attach the live version of each file only (`WS_TEMPLATE.md` section 1.1).
* Classify the work as HEAVY or LIGHT by contract impact (`WS_TEMPLATE.md`
  section 1.5).
* The effort tripwire stops revision 3 without merged code (`WS_TEMPLATE.md`
  section 1.8).
* Run your own gates. A report is a claim.
