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

Gates
-----
    control      focused suite green before any mutation
    mutations    MU1-MU8 alter PRODUCTION source; each must be KILLED by an
                 expected detector, without collection or import errors, and
                 each file is restored byte for byte and verified against HEAD
    targeted     tests/market_data, tests/portfolio, tests/valuation green
    full         full suite on HEAD and on --base; zero NEW failures
    static       ruff check, ruff format --check and mypy --strict on the
                 changed files; zero NEW findings against --base
    tree         the runner leaves `git status --short` unchanged
    governance   D101 status read from DECISIONS.md; CERTIFIED requires LOCKED

Safety
------
    A mutation never runs on a file that differs from HEAD, so uncommitted
    edits are never overwritten. pytest runs with `-p no:cacheprovider` and
    PYTHONDONTWRITEBYTECODE=1. --base uses a temporary `git worktree` that is
    removed afterwards.

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
DEFAULT_BASE = "5f3381f"  # last commit before the ws/vanilla-forward-node remediation

VAL = "src/riskquant/portfolio/ingest/validate.py"
IDN = "src/riskquant/portfolio/ingest/identity.py"
DSP = "src/riskquant/portfolio/ingest/dispatcher.py"
CMP = "src/riskquant/portfolio/ingest/compile.py"
FWC = "src/riskquant/portfolio/compilers/block_compilers/vanilla_forward.py"
ENG = "src/riskquant/portfolio/engine.py"
FOLD = "src/riskquant/market_data/compile.py"

NEW_TEST = "tests/market_data/remediation/test_exact_node_forward_delivery.py"
DELETED_TEST = "tests/valuation/oracle/test_node_forward_optional_delivery.py"
REMEDIATION_FILES = (VAL, IDN, DSP, FWC, ENG)
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
TARGETED = ("tests/market_data", "tests/portfolio", "tests/valuation")

Edit = tuple[str, str, str]

# (id, semantic defect, production edits, expected detector test names)
MUTATIONS: tuple[tuple[str, str, tuple[Edit, ...], tuple[str, ...]], ...] = (
    (
        "MU1",
        "exact NODE becomes delivery-required (FORWARD family list)",
        (
            (
                VAL,
                '        "settlement_date",\n        "under_discount_curve",\n'
                '        "strike_discount_curve",\n    )\n\n    _require_cols(df, cols)\n'
                '    _validate_non_null_masked(df, mask, cols, "Forward")',
                '        "settlement_date",\n        "under_discount_curve",\n'
                '        "strike_discount_curve",\n        "delivery_date",\n    )\n\n'
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
                "        underlying_names, static_config, n\n    )\n",
                "    underlying_ids, under_ccy_ids = _resolve_underlying(\n"
                "        underlying_names, static_config, n\n    )\n"
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
# Process helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _env(root: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "src"), str(root), env.get("PYTHONPATH", "")) if p
    )
    return env


def _pytest(root: pathlib.Path, paths: tuple[str, ...]) -> dict[str, Any]:
    proc = subprocess.run(
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
        cwd=root,
        env=_env(root),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = (proc.stdout + proc.stderr).splitlines()
    summary = next(
        (ln for ln in reversed(lines) if re.search(r"\d+ (passed|failed|errors?)", ln)),
        lines[-1] if lines else "",
    )
    failed = sorted({
        ln.split(" ")[1] for ln in lines if ln.startswith(("FAILED ", "ERROR "))
    })
    return {
        "returncode": proc.returncode,
        "summary": summary.strip("= ").strip(),
        "failed": failed,
        "collection_errors": bool(re.search(r"\d+ errors?\b", summary))
        or "Interrupted" in summary,
    }


def _native(text: str, snippet: str) -> str:
    """Match the file's line endings (a Windows checkout may hold CRLF)."""
    return snippet.replace("\n", "\r\n") if "\r\n" in text else snippet


def _same_as_head(root: pathlib.Path, rel: str) -> bool:
    return _run(["git", "diff", "--quiet", "HEAD", "--", rel], root).returncode == 0


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _tree_state(root: pathlib.Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "head": _run(["git", "rev-parse", "HEAD"], root).stdout.strip(),
        "branch": _run(["git", "branch", "--show-current"], root).stdout.strip(),
        "status_short": _run(["git", "status", "--short"], root).stdout.strip(),
        "diff_stat": _run(["git", "diff", "--stat"], root).stdout.strip(),
    }
    state["remediation_files_present"] = all(
        (root / f).is_file() for f in (*REMEDIATION_FILES, NEW_TEST)
    )
    state["deleted_test_absent"] = not (root / DELETED_TEST).exists()
    state["label_classifier_present"] = "XAxisKind.LABEL.name" in (
        root / IDN
    ).read_text(encoding="utf-8")
    state["curve_refinement_registered"] = (
        "INSTRUMENT_VALIDATORS[VANILLA_FORWARD_CURVE_ID]"
        in (root / DSP).read_text(encoding="utf-8")
    )
    return state


def _d101_status(root: pathlib.Path) -> str:
    path = root / "DECISIONS.md"
    if not path.is_file():
        return "NOT FOUND: DECISIONS.md"
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## D101\b.*?$(.*?)^## ", text, flags=re.M | re.S)
    if m is None:
        return "NOT FOUND: D101"
    status = re.search(r"^\*\*Status:\*\*\s*(.+?)\s*$", m.group(1), flags=re.M)
    return status.group(1) if status else "NOT FOUND: D101 status line"


def _mutations(root: pathlib.Path) -> list[dict[str, Any]]:
    files = sorted({f for _, _, edits, _ in MUTATIONS for f, _, _ in edits})
    dirty = [f for f in files if not _same_as_head(root, f)]
    if dirty:
        return [
            {"id": "ALL", "result": "NOT RUN", "reason": f"differ from HEAD: {dirty}"}
        ]
    results: list[dict[str, Any]] = []
    for mid, defect, edits, detectors in MUTATIONS:
        originals = {f: (root / f).read_bytes() for f, _, _ in edits}
        rec: dict[str, Any] = {
            "id": mid,
            "defect": defect,
            "files": sorted(originals),
            "detectors": list(detectors),
        }
        try:
            for rel, old, new in edits:
                text = (root / rel).read_bytes().decode("utf-8")
                o, n = _native(text, old), _native(text, new)
                if text.count(o) != 1:
                    raise LookupError(f"anchor count {text.count(o)} in {rel}")
                (root / rel).write_bytes(text.replace(o, n).encode("utf-8"))
            res = _pytest(root, FOCUSED)
            hits = [t for t in res["failed"] if any(d in t for d in detectors)]
            if res["collection_errors"]:
                rec["result"] = "INVALID (collection or import error)"
            elif res["returncode"] != 0 and hits:
                rec["result"] = "KILLED"
            else:
                rec["result"] = "SURVIVED"
            rec.update(summary=res["summary"], failed=res["failed"], detector_hits=hits)
        except LookupError as exc:
            rec.update(result="NOT RUN", reason=str(exc))
        finally:
            for rel, data in originals.items():
                (root / rel).write_bytes(data)
            rec["restored_bytes_equal"] = all(
                (root / rel).read_bytes() == data for rel, data in originals.items()
            )
            rec["same_as_head_after_restore"] = all(
                _same_as_head(root, f) for f in originals
            )
        results.append(rec)
        if not (rec["restored_bytes_equal"] and rec["same_as_head_after_restore"]):
            raise SystemExit(f"{mid}: RESTORE FAILED. Inspect {sorted(originals)}.")
    return results


def _ruff_findings(root: pathlib.Path, files: list[str]) -> list[str]:
    out = _run([PY, "-m", "ruff", "check", "--output-format", "concise", *files], root)
    return sorted({
        re.sub(r":\d+:\d+:", ":", ln)
        for ln in out.stdout.splitlines()
        if re.match(r"\S+:\d+:\d+:", ln)
    })


def _static(root: pathlib.Path) -> dict[str, Any]:
    files = [f for f in CHANGED_PY if (root / f).is_file()]
    src = [f for f in files if f.startswith("src/")]
    fmt = _run([PY, "-m", "ruff", "format", "--check", *files], root)
    mypy = _run(
        [
            PY,
            "-m",
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            tempfile.mkdtemp(),
            *src,
        ],
        root,
    )
    return {
        "ruff_check": _ruff_findings(root, files),
        "ruff_format_would_reformat": sorted(
            ln.split(" ", 2)[-1]
            for ln in fmt.stdout.splitlines()
            if ln.startswith("Would reformat")
        ),
        "mypy_summary": (mypy.stdout.strip().splitlines() or [""])[-1],
        "mypy_errors": sorted({
            re.sub(r":\d+(:\d+)?:", ":", ln)
            for ln in mypy.stdout.splitlines()
            if ": error:" in ln
        }),
    }


def _with_base(
    root: pathlib.Path, rev: str, fn: Callable[[pathlib.Path], dict[str, Any]]
) -> dict[str, Any]:
    if _run(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], root).returncode:
        return {"not_run": f"revision {rev!r} not found"}
    wt = pathlib.Path(tempfile.mkdtemp(prefix="rq_cert_base_")) / "wt"
    add = _run(["git", "worktree", "add", "--detach", str(wt), rev], root)
    if add.returncode:
        return {"not_run": add.stderr.strip()}
    try:
        return fn(wt)
    finally:
        _run(["git", "worktree", "remove", "--force", str(wt)], root)


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "numba", "polars", "pytest", "ruff", "mypy"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = "not installed"
    env["cores"] = str(os.cpu_count() or 1)
    return env


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        default=str(
            pathlib
            .Path(__file__)
            .resolve()
            .with_name("vanilla_forward_node_certificate.json")
        ),
        help="Path for the machine-readable certificate. Defaults beside this script.",
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="baseline revision")
    parser.add_argument(
        "--no-base", action="store_true", help="skip baseline comparison"
    )
    parser.add_argument("--no-full", action="store_true", help="skip the full suite")
    parser.add_argument("--no-static", action="store_true", help="skip ruff and mypy")
    args = parser.parse_args()
    root = _REPO_ROOT
    base = None if args.no_base else args.base

    report: dict[str, Any] = {"environment": _environment(), "base": base or "none"}
    report["tree_before"] = _tree_state(root)
    report["governance"] = {"D101_status": _d101_status(root)}

    control = _pytest(root, FOCUSED)
    report["control"] = control
    report["mutations"] = (
        _mutations(root)
        if control["returncode"] == 0
        else [{"id": "ALL", "result": "NOT RUN", "reason": "control run not green"}]
    )
    report["targeted"] = {p: _pytest(root, (p,)) for p in TARGETED}

    if not args.no_full:
        report["full_head"] = _pytest(root, ())
        if base:
            report["full_base"] = _with_base(root, base, lambda wt: _pytest(wt, ()))
            if "failed" in report["full_base"]:
                h, b = (
                    set(report["full_head"]["failed"]),
                    set(report["full_base"]["failed"]),
                )
                report["full_compare"] = {
                    "new": sorted(h - b),
                    "resolved": sorted(b - h),
                    "common": sorted(h & b),
                }
    if not args.no_static:
        report["static_head"] = _static(root)
        if base:
            report["static_base"] = _with_base(root, base, _static)
            if "mypy_errors" in report["static_base"]:
                sh, sb = report["static_head"], report["static_base"]
                report["static_compare"] = {
                    key: {
                        "new": sorted(set(sh[key]) - set(sb[key])),
                        "resolved": sorted(set(sb[key]) - set(sh[key])),
                    }
                    for key in (
                        "ruff_check",
                        "ruff_format_would_reformat",
                        "mypy_errors",
                    )
                }

    report["tree_after"] = _tree_state(root)
    tree_ok = (
        report["tree_after"]["status_short"] == report["tree_before"]["status_short"]
    )

    muts = report["mutations"]
    gates: dict[str, str] = {
        "control": "PASS" if control["returncode"] == 0 else "FAIL",
        "mutations": "PASS"
        if len(muts) == len(MUTATIONS) and all(m["result"] == "KILLED" for m in muts)
        else "FAIL",
        "targeted": "PASS"
        if all(r["returncode"] == 0 for r in report["targeted"].values())
        else "FAIL",
        "full_no_new_failures": "NOT RUN"
        if "full_compare" not in report
        else ("PASS" if not report["full_compare"]["new"] else "FAIL"),
        "static_no_new_findings": "NOT RUN"
        if "static_compare" not in report
        else (
            "PASS"
            if not any(v["new"] for v in report["static_compare"].values())
            else "FAIL"
        ),
        "tree_unchanged_by_runner": "PASS" if tree_ok else "FAIL",
    }
    execution_ok = all(v == "PASS" for v in gates.values())
    d101 = report["governance"]["D101_status"]
    if execution_ok and d101.startswith("LOCKED"):
        overall = "CERTIFIED"
    elif execution_ok and d101.startswith("OPEN"):
        overall = "NOT CERTIFIED - OWNER APPROVAL REQUIRED FOR D101"
    else:
        overall = "NOT CERTIFIED"
    report["verdict"] = {"gates": gates, "D101_status": d101, "overall": overall}

    output = json.dumps(report, indent=2, sort_keys=True)
    json_path = pathlib.Path(args.json_out).resolve()
    json_path.write_text(output, encoding="utf-8")

    print(
        f"HEAD {report['tree_before']['head']}  branch {report['tree_before']['branch']}"
    )
    for m in muts:
        print(f"  {m['id']}: {m['result']}  {m.get('summary', m.get('reason', ''))}")
    for name, gate in gates.items():
        print(f"  gate {name}: {gate}")
    print(f"  D101: {d101}")
    print(f"  VERDICT: {overall}")
    print(f"\n  JSON certificate written to {json_path}")
    return 0 if overall == "CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
