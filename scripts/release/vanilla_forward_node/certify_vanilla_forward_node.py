# scripts/release/vanilla_forward_node/certify_vanilla_forward_node.py
"""
VANILLA_FORWARD_NODE delivery certification gate (D101).

Certifies the ws/vanilla-forward-node contract on the CURRENT git worktree:

    exact NODE (LABEL)       delivery optional; it never selects the rf; the
                             asset leg is discounted at delivery, or at
                             settlement when delivery is absent

    relative NODE (OFFSET)   delivery required; D065 period check; only
                             OFFSET_M/OFFSET_Y + CALENDAR_PERIOD; every other
                             roll method fails closed

    VANILLA_FORWARD_CURVE    delivery required at ingest

Certification principles
------------------------
    control
        Focused contract/oracle suite must be green before mutation.

    mutations
        MU1-MU8 alter PRODUCTION source only. Every mutation must be KILLED
        by its declared detector(s), with no collection/import/infrastructure
        failure. Every mutated file is restored byte-for-byte and verified
        against HEAD before the next mutation starts.

    targeted
        tests/market_data, tests/portfolio and tests/valuation must be green.

    full
        Full suite is run on HEAD and on --base. Existing baseline failures
        are allowed; NEW failures are not. Both runs must be valid pytest
        executions with no collection/import/infrastructure failure.

    static
        ruff check, ruff format --check and mypy --strict run on every changed
        Python file relevant to the remediation, including files mutated by
        MU4/MU7. New findings relative to --base fail the gate.

    structure
        Required remediation files and contract registrations must be present;
        the deleted legacy test must remain absent.

    tree
        All validation work must leave the pre-existing git state unchanged.
        The certificate itself is a deliberate output and is checked
        separately so it cannot invalidate the validation-tree gate.

    governance
        D101 status is read from DECISIONS.md. CERTIFIED requires LOCKED.

Safety
------
    A mutation is never applied to a file that differs from HEAD.
    PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider are used.
    --base is executed in a temporary detached git worktree which is removed
    and verified.

The machine-readable certificate is written automatically to:

    scripts/release/vanilla_forward_node/vanilla_forward_node_certificate.json

The output path can be overridden with:

    --json-out <path>

Usage:
    python scripts/release/vanilla_forward_node/certify_vanilla_forward_node.py
    python scripts/release/vanilla_forward_node/certify_vanilla_forward_node.py --base <rev>
    python scripts/release/vanilla_forward_node/certify_vanilla_forward_node.py --no-full --no-static
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from importlib import metadata
from typing import Any

# ---------------------------------------------------------------------------
# Repository-root bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

if not (_REPO_ROOT / "pyproject.toml").is_file():
    raise RuntimeError(
        f"Could not locate RiskQuant repository root from {__file__!s}: "
        f"expected {_REPO_ROOT / 'pyproject.toml'}"
    )

PY = sys.executable

DEFAULT_BASE = "5f3381f"
DEFAULT_JSON_NAME = "vanilla_forward_node_certificate.json"

# ---------------------------------------------------------------------------
# Repository contract paths
# ---------------------------------------------------------------------------

VAL = "src/riskquant/portfolio/ingest/validate.py"
IDN = "src/riskquant/portfolio/ingest/identity.py"
DSP = "src/riskquant/portfolio/ingest/dispatcher.py"
CMP = "src/riskquant/portfolio/ingest/compile.py"
FWC = "src/riskquant/portfolio/compilers/block_compilers/vanilla_forward.py"
ENG = "src/riskquant/portfolio/engine.py"
FOLD = "src/riskquant/market_data/compile.py"

NEW_TEST = "tests/market_data/remediation/test_exact_node_forward_delivery.py"
DELETED_TEST = "tests/valuation/oracle/test_node_forward_optional_delivery.py"

REMEDIATION_FILES = (
    VAL,
    IDN,
    DSP,
    CMP,
    FWC,
    ENG,
    FOLD,
)

# Every Python file touched by the remediation or mutation surface.
#
# CMP and FOLD are intentionally included here. MU4 mutates CMP and MU7
# mutates FOLD; excluding either from static analysis would create a blind
# spot in the certification gate.
CHANGED_PY = (
    *REMEDIATION_FILES,
    "tests/market_data/remediation/test_contract_failure_paths.py",
    "tests/market_data/remediation/test_node_option_delivery_contract.py",
    NEW_TEST,
)

FOCUSED = (
    NEW_TEST,
    "tests/market_data/remediation/test_contract_failure_paths.py",
    "tests/market_data/remediation/test_node_option_delivery_contract.py",
    "tests/valuation/oracle",
)

TARGETED = (
    "tests/market_data",
    "tests/portfolio",
    "tests/valuation",
)

Edit = tuple[str, str, str]

# ---------------------------------------------------------------------------
# Mutation matrix
# ---------------------------------------------------------------------------
#
# (id, semantic defect, production edits, expected detector test names)
#
# The mutation source edits are deliberately exact-anchor replacements.
# An anchor appearing zero or more than once is a hard mutation failure.
# ---------------------------------------------------------------------------

MUTATIONS: tuple[tuple[str, str, tuple[Edit, ...], tuple[str, ...]], ...] = (
    (
        "MU1",
        "exact NODE becomes delivery-required (FORWARD family list)",
        (
            (
                VAL,
                '        "settlement_date",\n'
                '        "under_discount_curve",\n'
                '        "strike_discount_curve",\n'
                "    )\n\n"
                "    _require_cols(df, cols)\n"
                '    _validate_non_null_masked(df, mask, cols, "Forward")',
                '        "settlement_date",\n'
                '        "under_discount_curve",\n'
                '        "strike_discount_curve",\n'
                '        "delivery_date",\n'
                "    )\n\n"
                "    _require_cols(df, cols)\n"
                '    _validate_non_null_masked(df, mask, cols, "Forward")',
            ),
        ),
        ("test_exact_node_ingests_with_and_without_delivery",),
    ),
    (
        "MU2",
        "unsupported relative roll becomes exact (original R1)",
        (
            (
                IDN,
                "        if curve.x_axis_kind == XAxisKind.LABEL.name:\n",
                "        if curve.x_axis_kind == XAxisKind.LABEL.name or "
                "(curve.x_axis_kind, str(curve.roll_method)) "
                "not in _SUPPORTED_PERIOD_LABELS:\n",
            ),
        ),
        ("test_unsupported_relative_roll_fails_closed",),
    ),
    (
        "MU3",
        "VANILLA_FORWARD_CURVE delivery refinement removed",
        (
            (
                DSP,
                "INSTRUMENT_VALIDATORS[VANILLA_FORWARD_CURVE_ID] = "
                "validate_forward_curve_instrument\n",
                "",
            ),
        ),
        ("test_curve_forward_without_delivery_fails_at_ingest",),
    ),
    (
        "MU4",
        "delivery selects the exact NODE (next rf on the same curve)",
        (
            (
                CMP,
                "    underlying_ids, under_ccy_ids = _resolve_underlying(\n"
                "        underlying_names, static_config, n\n"
                "    )\n",
                "    underlying_ids, under_ccy_ids = _resolve_underlying(\n"
                "        underlying_names, static_config, n\n"
                "    )\n"
                "    _nxt = underlying_ids.astype(np.int64) + 1\n"
                "    _par = static_config.rf_id_to_parent_id\n"
                "    _ok = (delivery_ord != NO_DATE_ORD32) & (instrument_ids == 1)"
                " & (underlying_ids >= 0) & (_nxt < _par.shape[0])\n"
                "    _idx = np.flatnonzero(_ok)\n"
                "    _ok[_idx] = _par[_nxt[_idx]] == _par[underlying_ids[_idx]]\n"
                "    underlying_ids = np.where(_ok, _nxt, underlying_ids).astype(np.int32)\n",
            ),
        ),
        (
            "test_delivery_never_selects_the_exact_node",
            "test_production_pv_matches_the_independent_oracle",
        ),
    ),
    (
        "MU5",
        "missing-delivery -> settlement fallback removed (mapper and compiler)",
        (
            (
                FWC,
                "        asset_date_ord[no_delivery] = "
                "batch.settle_ord[start:end][no_delivery]\n",
                "        pass\n",
            ),
            (
                ENG,
                "        fwd_mask & (delivery_ord == int(NO_DATE_ORD32)),\n"
                "        settle_ord,\n",
                "        fwd_mask & (delivery_ord == int(NO_DATE_ORD32)),\n"
                "        delivery_ord,\n",
            ),
        ),
        (
            "test_asset_leg_date_is_delivery_or_settlement",
            "test_production_pv_matches_the_independent_oracle",
        ),
    ),
    (
        "MU6",
        "explicit delivery ignored; settlement always (mapper and compiler)",
        (
            (
                FWC,
                "    no_delivery = raw_delivery == int(NO_DATE_ORD32)\n",
                "    no_delivery = np.ones_like(raw_delivery, dtype=np.bool_)\n",
            ),
            (
                ENG,
                "        fwd_mask & (delivery_ord == int(NO_DATE_ORD32)),\n",
                "        fwd_mask,\n",
            ),
        ),
        (
            "test_asset_leg_date_is_delivery_or_settlement",
            "test_production_pv_matches_the_independent_oracle",
        ),
    ),
    (
        "MU7",
        "fold reads price_curve_grid at the asset date",
        (
            (
                FOLD,
                "    out[:] = matrix.forward_nodes[price_uix]\n",
                "    out[:] = matrix.price_curve_grid[price_uix, date_uix]\n",
            ),
        ),
        (
            "test_effective_forward_factor_is_exact_node_times_asset_df_times_fx",
            "test_production_pv_matches_the_independent_oracle",
        ),
    ),
    (
        # Removing only the missing-delivery check is an EQUIVALENT mutation:
        # the D065 period check still rejects the sentinel month (1696-03).
        # Excluding the row from the relative set skips both checks.
        "MU8",
        "supported relative NODE skips the delivery and period checks",
        (
            (
                IDN,
                "        relative |= at\n",
                "        relative |= at & (delivery_ord != NO_DATE_ORD32)\n",
            ),
        ),
        (
            "test_relative_node_without_delivery_fails_at_ingest",
            "test_relative_forward_missing_delivery_fails_at_ingest",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Logging / process helpers
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[D101] {message}", flush=True)


def _run(
    cmd: list[str],
    cwd: pathlib.Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _env(root: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            str(root / "src"),
            str(root),
            env.get("PYTHONPATH", ""),
        )
        if p
    )
    return env


def _combined_output(proc: subprocess.CompletedProcess[str]) -> str:
    parts = [proc.stdout, proc.stderr]
    return "\n".join(part for part in parts if part).strip()


def _tail(text: str, limit: int = 15) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _infrastructure_error(output: str) -> bool:
    patterns = (
        "No module named",
        "ModuleNotFoundError",
        "Traceback (most recent call last)",
        "command not found",
        "No such file or directory",
    )
    return any(pattern in output for pattern in patterns)


def _pytest_summary(output: str) -> str:
    candidates = [
        line.strip()
        for line in output.splitlines()
        if re.search(
            r"\b\d+\s+(?:passed|failed|error|errors|skipped|xfailed|xpassed|deselected)\b",
            line,
        )
    ]
    if candidates:
        return candidates[-1].strip("= ").strip()

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _pytest_failed_nodes(output: str) -> list[str]:
    failed: set[str] = set()

    for line in output.splitlines():
        if line.startswith("ERROR collecting"):
            continue

        match = re.match(r"^(?:FAILED|ERROR)\s+(\S+)", line)
        if match:
            failed.add(match.group(1))

    return sorted(failed)


def _pytest_collection_or_import_error(output: str) -> bool:
    explicit = (
        "ERROR collecting ",
        "ImportError while importing test module",
        "ImportError while loading conftest",
        "ImportError while loading plugin",
    )

    if any(pattern in output for pattern in explicit):
        return True

    # A collection/import failure can also be reported without pytest's
    # standard "ERROR collecting" banner.
    if re.search(r"(^|\n)(?:ImportError|ModuleNotFoundError):", output):
        return True

    return False


def _pytest(
    root: pathlib.Path,
    paths: tuple[str, ...],
    *,
    label: str,
) -> dict[str, Any]:
    _log(f"{label}: START")

    proc = _run(
        [
            PY,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--tb=line",
            "-rfE",
            *paths,
        ],
        root,
        env=_env(root),
    )

    output = _combined_output(proc)
    summary = _pytest_summary(output)
    failed = _pytest_failed_nodes(output)
    collection_errors = _pytest_collection_or_import_error(output)
    infrastructure_error = _infrastructure_error(output)

    # Pytest return codes:
    #   0 = clean
    #   1 = test failures
    #   2+ = interruption/usage/no-tests/infrastructure/etc.
    valid_run = (
        proc.returncode in (0, 1) and not collection_errors and not infrastructure_error
    )

    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "valid_run": valid_run,
        "summary": summary,
        "failed": failed,
        "collection_errors": collection_errors,
        "infrastructure_error": infrastructure_error,
    }

    status = "PASS" if proc.returncode == 0 and valid_run else "FAIL"
    _log(f"{label}: {status}  {summary or f'returncode={proc.returncode}'}")

    if proc.returncode != 0:
        tail = _tail(output)
        if tail:
            for line in tail:
                print(f"    {line}", flush=True)

    return result


def _native(text: str, snippet: str) -> str:
    """Match the file's line endings (Windows checkout may hold CRLF)."""
    return snippet.replace("\n", "\r\n") if "\r\n" in text else snippet


def _same_as_head(root: pathlib.Path, rel: str) -> bool:
    return _run(["git", "diff", "--quiet", "HEAD", "--", rel], root).returncode == 0


def _head_sha(root: pathlib.Path) -> str:
    proc = _run(["git", "rev-parse", "HEAD"], root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git rev-parse HEAD failed")
    return proc.stdout.strip()


def _status_short(root: pathlib.Path) -> str:
    proc = _run(["git", "status", "--short"], root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git status failed")
    return proc.stdout.strip()


def _status_without_path(status: str, rel: str) -> str:
    """Remove the exact certificate path from porcelain-v1 status output."""
    kept: list[str] = []

    for line in status.splitlines():
        if len(line) < 4:
            kept.append(line)
            continue

        path_text = line[3:]
        if " -> " in path_text:
            old_path, new_path = path_text.split(" -> ", 1)
            if old_path == rel or new_path == rel:
                continue
        elif path_text == rel:
            continue

        kept.append(line)

    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Repository state / governance
# ---------------------------------------------------------------------------


def _tree_state(root: pathlib.Path) -> dict[str, Any]:
    status = _status_short(root)

    state: dict[str, Any] = {
        "head": _head_sha(root),
        "branch": _run(["git", "branch", "--show-current"], root).stdout.strip(),
        "status_short": status,
        "diff_stat": _run(["git", "diff", "--stat"], root).stdout.strip(),
    }

    state["remediation_files_present"] = all(
        (root / f).is_file() for f in (*REMEDIATION_FILES, NEW_TEST)
    )

    state["deleted_test_absent"] = not (root / DELETED_TEST).exists()

    idn_text = (root / IDN).read_text(encoding="utf-8")
    dsp_text = (root / DSP).read_text(encoding="utf-8")

    state["label_classifier_present"] = "XAxisKind.LABEL.name" in idn_text

    state["curve_refinement_registered"] = (
        "INSTRUMENT_VALIDATORS[VANILLA_FORWARD_CURVE_ID]" in dsp_text
    )

    state["mutation_surface_clean"] = all(
        _same_as_head(root, f)
        for f in sorted({f for _, _, edits, _ in MUTATIONS for f, _, _ in edits})
    )

    return state


def _d101_status(root: pathlib.Path) -> str:
    path = root / "DECISIONS.md"

    if not path.is_file():
        return "NOT FOUND: DECISIONS.md"

    text = path.read_text(encoding="utf-8")

    match = re.search(
        r"^##\s+D101\b.*?(?=^##\s|\Z)",
        text,
        flags=re.M | re.S,
    )

    if match is None:
        return "NOT FOUND: D101"

    status = re.search(
        r"^\*\*Status:\*\*\s*(.+?)\s*$",
        match.group(0),
        flags=re.M,
    )

    return status.group(1).strip() if status else "NOT FOUND: D101 status line"


def _contract_state_gate(state: dict[str, Any]) -> str:
    required = (
        "remediation_files_present",
        "deleted_test_absent",
        "label_classifier_present",
        "curve_refinement_registered",
        "mutation_surface_clean",
    )

    return "PASS" if all(bool(state.get(key)) for key in required) else "FAIL"


# ---------------------------------------------------------------------------
# Mutation gate
# ---------------------------------------------------------------------------


def _mutation_files() -> list[str]:
    return sorted({rel for _, _, edits, _ in MUTATIONS for rel, _, _ in edits})


def _mutations(root: pathlib.Path) -> list[dict[str, Any]]:
    files = _mutation_files()

    # Hard safety stop. Never overwrite a pre-existing edit.
    dirty = [f for f in files if not _same_as_head(root, f)]

    if dirty:
        _log(f"MUTATIONS: NOT RUN; mutation files differ from HEAD: {dirty}")
        return [
            {
                "id": "ALL",
                "result": "NOT RUN",
                "reason": f"differ from HEAD: {dirty}",
            }
        ]

    results: list[dict[str, Any]] = []

    for mid, defect, edits, detectors in MUTATIONS:
        _log(f"{mid}: START  {defect}")

        originals = {rel: (root / rel).read_bytes() for rel, _, _ in edits}

        rec: dict[str, Any] = {
            "id": mid,
            "defect": defect,
            "files": sorted(originals),
            "detectors": list(detectors),
        }

        try:
            # Re-check before every mutation. This makes the safety invariant
            # local to each mutation rather than dependent only on the first
            # global check.
            dirty_now = [f for f in files if not _same_as_head(root, f)]
            if dirty_now:
                rec.update(
                    result="NOT RUN",
                    reason=f"mutation surface became dirty: {dirty_now}",
                )
                results.append(rec)
                break

            for rel, old, new in edits:
                path = root / rel
                text = path.read_bytes().decode("utf-8")

                old_native = _native(text, old)
                new_native = _native(text, new)

                count = text.count(old_native)

                if count != 1:
                    raise LookupError(
                        f"{rel}: anchor count is {count}, expected exactly 1"
                    )

                updated = text.replace(old_native, new_native)
                path.write_bytes(updated.encode("utf-8"))

            pytest_result = _pytest(
                root,
                FOCUSED,
                label=f"{mid} detector suite",
            )

            expected_hits = [
                detector
                for detector in detectors
                if any(detector in failed for failed in pytest_result["failed"])
            ]

            rec.update(
                summary=pytest_result["summary"],
                failed=pytest_result["failed"],
                detector_hits=expected_hits,
                collection_errors=pytest_result["collection_errors"],
                infrastructure_error=pytest_result["infrastructure_error"],
                pytest_returncode=pytest_result["returncode"],
            )

            if not pytest_result["valid_run"]:
                rec["result"] = "INVALID (collection/import/infrastructure error)"
            elif pytest_result["returncode"] != 0 and expected_hits:
                rec["result"] = "KILLED"
            else:
                rec["result"] = "SURVIVED"

        except Exception as exc:
            rec.update(
                result="INVALID",
                reason=f"{type(exc).__name__}: {exc}",
            )

        finally:
            # Restoration is unconditional.
            for rel, data in originals.items():
                (root / rel).write_bytes(data)

            rec["restored_bytes_equal"] = all(
                (root / rel).read_bytes() == data for rel, data in originals.items()
            )

            rec["same_as_head_after_restore"] = all(
                _same_as_head(root, rel) for rel in originals
            )

        if not (rec["restored_bytes_equal"] and rec["same_as_head_after_restore"]):
            _log(f"{mid}: FATAL RESTORE FAILURE on {sorted(originals)}")
            raise SystemExit(f"{mid}: RESTORE FAILED. Inspect {sorted(originals)}.")

        results.append(rec)

        _log(f"{mid}: {rec['result']}  {rec.get('summary', rec.get('reason', ''))}")

        if rec["result"] != "KILLED":
            # Do not run subsequent mutations after one has failed. The
            # restored tree is still safe, but the certification evidence is
            # already invalid.
            break

    return results


# ---------------------------------------------------------------------------
# Static-analysis gate
# ---------------------------------------------------------------------------


def _ruff_findings(output: str) -> list[str]:
    findings: set[str] = set()

    for line in output.splitlines():
        if re.match(r"\S+:\d+:\d+:", line):
            findings.add(re.sub(r":\d+:\d+:", ":", line.strip()))

    return sorted(findings)


def _ruff_format_findings(output: str) -> list[str]:
    paths: set[str] = set()

    for line in output.splitlines():
        match = re.search(r"Would reformat(?:\s*:\s*|\s+)(.+)$", line)
        if match:
            paths.add(match.group(1).strip())

    return sorted(paths)


def _mypy_findings(output: str) -> list[str]:
    findings: set[str] = set()

    for line in output.splitlines():
        if ": error:" not in line:
            continue

        findings.add(re.sub(r":\d+(:\d+)?:", ":", line.strip()))

    return sorted(findings)


def _tool_valid(proc: subprocess.CompletedProcess[str]) -> bool:
    output = _combined_output(proc)

    # Ruff and mypy use 1 for ordinary findings.  A result with 2+ is treated
    # as an invocation/configuration/infrastructure failure.
    return proc.returncode in (0, 1) and not _infrastructure_error(output)


def _static(root: pathlib.Path, *, label: str) -> dict[str, Any]:
    files = [f for f in CHANGED_PY if (root / f).is_file()]

    src = [f for f in files if f.startswith("src/")]

    _log(f"{label}: START")

    ruff_check = _run(
        [
            PY,
            "-m",
            "ruff",
            "check",
            "--output-format",
            "concise",
            *files,
        ],
        root,
        env=_env(root),
    )

    ruff_format = _run(
        [
            PY,
            "-m",
            "ruff",
            "format",
            "--check",
            *files,
        ],
        root,
        env=_env(root),
    )

    with tempfile.TemporaryDirectory(prefix="rq_d101_mypy_") as cache_dir:
        mypy = _run(
            [
                PY,
                "-m",
                "mypy",
                "--strict",
                "--no-incremental",
                "--cache-dir",
                cache_dir,
                *src,
            ],
            root,
            env=_env(root),
        )

    ruff_check_output = _combined_output(ruff_check)
    ruff_format_output = _combined_output(ruff_format)
    mypy_output = _combined_output(mypy)

    result: dict[str, Any] = {
        "files": files,
        "ruff_check_returncode": ruff_check.returncode,
        "ruff_check_valid_run": _tool_valid(ruff_check),
        "ruff_check": _ruff_findings(ruff_check_output),
        "ruff_format_returncode": ruff_format.returncode,
        "ruff_format_valid_run": _tool_valid(ruff_format),
        "ruff_format_would_reformat": _ruff_format_findings(ruff_format_output),
        "mypy_returncode": mypy.returncode,
        "mypy_valid_run": _tool_valid(mypy),
        "mypy_summary": (
            mypy_output.splitlines()[-1] if mypy_output.splitlines() else ""
        ),
        "mypy_errors": _mypy_findings(mypy_output),
    }

    result["valid_run"] = all(
        bool(result[key])
        for key in (
            "ruff_check_valid_run",
            "ruff_format_valid_run",
            "mypy_valid_run",
        )
    )

    if result["valid_run"]:
        clean = not any((
            result["ruff_check"],
            result["ruff_format_would_reformat"],
            result["mypy_errors"],
        ))
        _log(f"{label}: {'PASS' if clean else 'FINDINGS'}")
    else:
        _log(f"{label}: INVALID TOOL EXECUTION")

    return result


def _static_new_findings(
    head: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    keys = (
        "ruff_check",
        "ruff_format_would_reformat",
        "mypy_errors",
    )

    return {
        key: {
            "new": sorted(set(head[key]) - set(base[key])),
            "resolved": sorted(set(base[key]) - set(head[key])),
        }
        for key in keys
    }


# ---------------------------------------------------------------------------
# Temporary baseline worktree
# ---------------------------------------------------------------------------


def _with_base(
    root: pathlib.Path,
    rev: str,
    fn: Callable[[pathlib.Path], dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    verify = _run(
        [
            "git",
            "rev-parse",
            "--verify",
            f"{rev}^{{commit}}",
        ],
        root,
    )

    if verify.returncode != 0:
        return {
            "not_run": f"revision {rev!r} not found",
        }

    temp_parent = pathlib.Path(tempfile.mkdtemp(prefix="rq_d101_base_"))
    wt = temp_parent / "wt"

    _log(f"{label}: creating detached worktree at {rev}")

    add = _run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(wt),
            rev,
        ],
        root,
    )

    if add.returncode:
        try:
            temp_parent.rmdir()
        except OSError:
            pass

        return {
            "not_run": add.stderr.strip() or "git worktree add failed",
        }

    try:
        return fn(wt)
    finally:
        remove = _run(
            [
                "git",
                "worktree",
                "remove",
                "--force",
                str(wt),
            ],
            root,
        )

        try:
            temp_parent.rmdir()
        except OSError:
            pass

        if remove.returncode:
            raise RuntimeError(
                f"failed to remove temporary base worktree {wt}: "
                f"{remove.stderr.strip()}"
            )

        _log(f"{label}: temporary worktree removed")


# ---------------------------------------------------------------------------
# Environment / certificate helpers
# ---------------------------------------------------------------------------


def _environment() -> dict[str, str]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    for pkg in (
        "numpy",
        "numba",
        "polars",
        "pytest",
        "ruff",
        "mypy",
    ):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = "not installed"

    env["cores"] = str(os.cpu_count() or 1)

    return env


def _resolve_json_path(
    root: pathlib.Path,
    raw: str,
) -> tuple[pathlib.Path, str | None]:
    path = pathlib.Path(raw).expanduser()

    if not path.is_absolute():
        path = root / path

    resolved = path.resolve()

    try:
        relative = resolved.relative_to(root.resolve())
        return resolved, relative.as_posix()
    except ValueError:
        return resolved, None


def _validate_json_path(
    root: pathlib.Path,
    json_path: pathlib.Path,
    json_rel: str | None,
) -> None:
    # Never allow the certificate to overwrite any source/test file involved
    # in certification, or the certification runner itself.
    forbidden = set(CHANGED_PY)
    forbidden.add(pathlib.Path(__file__).resolve().relative_to(root).as_posix())

    if json_rel in forbidden:
        raise RuntimeError(
            f"--json-out resolves to a certification source/test file: {json_rel}"
        )

    if json_path == root / "DECISIONS.md":
        raise RuntimeError("--json-out must not overwrite DECISIONS.md")

    json_path.parent.mkdir(parents=True, exist_ok=True)


def _sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _write_json_atomic(
    path: pathlib.Path,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    temp_path = pathlib.Path(temp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _certificate_output_state(
    root: pathlib.Path,
    json_path: pathlib.Path,
    json_rel: str | None,
    tree_before_output: dict[str, Any],
    tree_after_output: dict[str, Any],
) -> dict[str, Any]:
    before_status = tree_before_output["status_short"]
    after_status = tree_after_output["status_short"]

    if json_rel is None:
        status_only_certificate = before_status == after_status
    else:
        status_only_certificate = _status_without_path(
            before_status, json_rel
        ) == _status_without_path(after_status, json_rel)

    return {
        "path": str(json_path),
        "relative_path": json_rel,
        "inside_repo": json_rel is not None,
        "sha256": _sha256(json_path),
        "status_delta_is_certificate_only": status_only_certificate,
        "tree_status_before_output": before_status,
        "tree_status_after_output": after_status,
    }


# ---------------------------------------------------------------------------
# Main certification flow
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--json-out",
        default=str(pathlib.Path(__file__).resolve().with_name(DEFAULT_JSON_NAME)),
        help=(
            "Path for the machine-readable certificate. Defaults beside this script."
        ),
    )

    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="baseline revision",
    )

    parser.add_argument(
        "--no-base",
        action="store_true",
        help="skip baseline comparison",
    )

    parser.add_argument(
        "--no-full",
        action="store_true",
        help="skip the full suite",
    )

    parser.add_argument(
        "--no-static",
        action="store_true",
        help="skip ruff and mypy",
    )

    args = parser.parse_args()

    root = _REPO_ROOT
    base = None if args.no_base else args.base

    json_path, json_rel = _resolve_json_path(
        root,
        args.json_out,
    )

    _validate_json_path(
        root,
        json_path,
        json_rel,
    )

    report: dict[str, Any] = {
        "schema": "D101-v2",
        "environment": _environment(),
        "base": base or "none",
        "configuration": {
            "full_enabled": not args.no_full,
            "static_enabled": not args.no_static,
            "base_enabled": base is not None,
            "focused_paths": list(FOCUSED),
            "targeted_paths": list(TARGETED),
            "changed_python_files": list(CHANGED_PY),
        },
    }

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    _log("CERTIFICATION START")

    tree_before = _tree_state(root)
    report["tree_before"] = tree_before

    report["governance"] = {
        "D101_status": _d101_status(root),
    }

    contract_state = _contract_state_gate(tree_before)
    report["contract_state"] = {
        "gate": contract_state,
        "remediation_files_present": tree_before["remediation_files_present"],
        "deleted_test_absent": tree_before["deleted_test_absent"],
        "label_classifier_present": tree_before["label_classifier_present"],
        "curve_refinement_registered": tree_before["curve_refinement_registered"],
        "mutation_surface_clean": tree_before["mutation_surface_clean"],
    }

    _log(f"contract state: {contract_state}")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    if contract_state == "PASS":
        control = _pytest(
            root,
            FOCUSED,
            label="CONTROL",
        )
    else:
        control = {
            "returncode": None,
            "valid_run": False,
            "summary": "not run: contract-state preflight failed",
            "failed": [],
            "collection_errors": False,
            "infrastructure_error": False,
            "not_run": "contract-state preflight failed",
        }

    report["control"] = control

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    if (
        contract_state == "PASS"
        and control.get("returncode") == 0
        and control.get("valid_run")
    ):
        mutations = _mutations(root)
    else:
        mutations = [
            {
                "id": "ALL",
                "result": "NOT RUN",
                "reason": (
                    "control run not green"
                    if contract_state == "PASS"
                    else "contract-state preflight failed"
                ),
            }
        ]

    report["mutations"] = mutations

    # ------------------------------------------------------------------
    # Targeted suites
    # ------------------------------------------------------------------

    targeted: dict[str, dict[str, Any]] = {}

    for path in TARGETED:
        targeted[path] = _pytest(
            root,
            (path,),
            label=f"TARGETED {path}",
        )

    report["targeted"] = targeted

    # ------------------------------------------------------------------
    # Full suite + baseline
    # ------------------------------------------------------------------

    if not args.no_full:
        full_head = _pytest(
            root,
            (),
            label="FULL HEAD",
        )
        report["full_head"] = full_head

        if base:
            full_base = _with_base(
                root,
                base,
                lambda wt: _pytest(
                    wt,
                    (),
                    label="FULL BASE",
                ),
                label="FULL BASE",
            )

            report["full_base"] = full_base

            if (
                "failed" in full_base
                and full_head.get("valid_run")
                and full_base.get("valid_run")
            ):
                head_failed = set(full_head["failed"])
                base_failed = set(full_base["failed"])

                report["full_compare"] = {
                    "new": sorted(head_failed - base_failed),
                    "resolved": sorted(base_failed - head_failed),
                    "common": sorted(head_failed & base_failed),
                }
            else:
                report["full_compare"] = {
                    "not_run": ("HEAD or BASE full-suite execution was invalid"),
                }

    # ------------------------------------------------------------------
    # Static analysis + baseline
    # ------------------------------------------------------------------

    if not args.no_static:
        static_head = _static(
            root,
            label="STATIC HEAD",
        )
        report["static_head"] = static_head

        if base:
            static_base = _with_base(
                root,
                base,
                lambda wt: _static(
                    wt,
                    label="STATIC BASE",
                ),
                label="STATIC BASE",
            )

            report["static_base"] = static_base

            if static_head.get("valid_run") and static_base.get("valid_run"):
                report["static_compare"] = _static_new_findings(
                    static_head,
                    static_base,
                )
            else:
                report["static_compare"] = {
                    "not_run": ("HEAD or BASE static execution was invalid"),
                }

    # ------------------------------------------------------------------
    # Validation-tree invariant
    #
    # This is captured BEFORE the deliberate certificate write. The
    # certificate itself is handled separately below.
    # ------------------------------------------------------------------

    tree_after_validation = _tree_state(root)
    report["tree_after_validation"] = tree_after_validation

    tree_unchanged = (
        tree_after_validation["head"] == tree_before["head"]
        and tree_after_validation["branch"] == tree_before["branch"]
        and tree_after_validation["status_short"] == tree_before["status_short"]
    )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    control_pass = control.get("returncode") == 0 and bool(control.get("valid_run"))

    mutation_pass = len(mutations) == len(MUTATIONS) and all(
        mutation.get("result") == "KILLED"
        and mutation.get("restored_bytes_equal") is True
        and mutation.get("same_as_head_after_restore") is True
        for mutation in mutations
    )

    targeted_pass = all(
        result.get("returncode") == 0 and result.get("valid_run")
        for result in targeted.values()
    )

    if args.no_full:
        full_gate = "NOT RUN"
    elif (
        "full_compare" in report
        and "new" in report["full_compare"]
        and report["full_head"].get("valid_run")
        and report["full_base"].get("valid_run")
    ):
        full_gate = "PASS" if not report["full_compare"]["new"] else "FAIL"
    else:
        full_gate = "FAIL"

    if args.no_static:
        static_gate = "NOT RUN"
    elif (
        "static_compare" in report
        and "not_run" not in report["static_compare"]
        and report["static_head"].get("valid_run")
        and report["static_base"].get("valid_run")
    ):
        has_new_static = any(
            result.get("new") for result in report["static_compare"].values()
        )

        static_gate = "FAIL" if has_new_static else "PASS"
    else:
        static_gate = "FAIL"

    gates: dict[str, str] = {
        "contract_state": contract_state,
        "control": "PASS" if control_pass else "FAIL",
        "mutations": "PASS" if mutation_pass else "FAIL",
        "targeted": "PASS" if targeted_pass else "FAIL",
        "full_no_new_failures": full_gate,
        "static_no_new_findings": static_gate,
        "tree_unchanged_by_validation": ("PASS" if tree_unchanged else "FAIL"),
    }

    execution_ok = all(value == "PASS" for value in gates.values())

    d101 = report["governance"]["D101_status"]

    if execution_ok and d101.startswith("LOCKED"):
        overall = "CERTIFIED"
    elif execution_ok and d101.startswith("OPEN"):
        overall = "NOT CERTIFIED - OWNER APPROVAL REQUIRED FOR D101"
    else:
        overall = "NOT CERTIFIED"

    report["verdict"] = {
        "gates": gates,
        "D101_status": d101,
        "overall": overall,
    }

    # ------------------------------------------------------------------
    # Write provisional certificate.
    #
    # This happens only after all validation gates have been calculated.
    # ------------------------------------------------------------------

    report["certificate_output"] = {
        "path": str(json_path),
        "relative_path": json_rel,
        "inside_repo": json_rel is not None,
        "written": False,
    }

    _write_json_atomic(
        json_path,
        report,
    )

    # ------------------------------------------------------------------
    # Verify that the certificate is the only deliberate post-validation
    # tree change, if it lives inside the repository.
    # ------------------------------------------------------------------

    tree_after_output = _tree_state(root)

    output_state = _certificate_output_state(
        root,
        json_path,
        json_rel,
        tree_after_validation,
        tree_after_output,
    )

    report["certificate_output"].update({
        "written": True,
        "status_delta_is_certificate_only": (
            output_state["status_delta_is_certificate_only"]
        ),
        "tree_status_before_output": (output_state["tree_status_before_output"]),
        "tree_status_after_output": (output_state["tree_status_after_output"]),
        "sha256": output_state["sha256"],
    })

    # A certificate path inside the repo is an intentional output. The
    # certification gate itself is still based on the pre-output tree.
    #
    # If anything OTHER than the certificate changed between validation and
    # final output, record that as a hard failure.
    if not output_state["status_delta_is_certificate_only"]:
        report["verdict"]["gates"]["certificate_output_hygiene"] = "FAIL"
        report["verdict"]["overall"] = "NOT CERTIFIED"
    else:
        report["verdict"]["gates"]["certificate_output_hygiene"] = "PASS"

    # Recompute execution_ok after certificate-output hygiene.
    execution_ok = all(value == "PASS" for value in report["verdict"]["gates"].values())

    if execution_ok and d101.startswith("LOCKED"):
        report["verdict"]["overall"] = "CERTIFIED"
    elif execution_ok and d101.startswith("OPEN"):
        report["verdict"]["overall"] = (
            "NOT CERTIFIED - OWNER APPROVAL REQUIRED FOR D101"
        )
    else:
        report["verdict"]["overall"] = "NOT CERTIFIED"

    overall = report["verdict"]["overall"]

    # Final write. Git status is unaffected by rewriting the same certificate
    # path, so the recorded certificate-output status remains valid.
    _write_json_atomic(
        json_path,
        report,
    )

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------

    print()
    print(
        f"HEAD {report['tree_before']['head']}  "
        f"branch {report['tree_before']['branch']}"
    )

    for mutation in mutations:
        print(
            f"  {mutation['id']}: "
            f"{mutation['result']}"
            f"  {mutation.get('summary', mutation.get('reason', ''))}"
        )

    for name, gate in report["verdict"]["gates"].items():
        print(f"  gate {name}: {gate}")

    print(f"  D101: {d101}")
    print(f"  VERDICT: {overall}")
    print(f"\n  JSON certificate written to {json_path}")

    return 0 if overall == "CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
