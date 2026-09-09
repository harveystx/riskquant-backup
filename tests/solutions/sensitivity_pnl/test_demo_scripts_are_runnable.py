# tests/solutions/sensitivity_pnl/test_demo_scripts_are_runnable.py
"""
Every demo script for `sensitivity_pnl` / `solutions.sensitivity_pnl`, run as
a real subprocess, once per script, in CI.

This test exists because of exactly the kind of defect it would have caught:
`scripts/solutions/sensitivity_pnl/demo_e2e.py` imported
`riskquant.solutions.sensitivity_pnl.api`, and `api.py` imported a module —
`riskquant.solutions.sensitivity_pnl.taylor_validity` — that did not exist on
disk. Every golden master still passed, because nothing in the test suite
ever executed that import chain; the break was only visible by actually
running the demo. A demo script with zero execution coverage is a code path
nothing verifies, and `scripts/` is exactly where visible-but-untested code
accumulates, because nobody's CI fails when a demo rots.

Each script is checked for:
  - exit code 0
  - no traceback on stderr (a script that swallows an exception and exits 0
    would pass the exit-code check alone and still be broken)
  - its own completion marker, for the scripts that print one — this catches
    a script that dies partway through but happens to reach a `print` that
    looks like success before an unhandled path further down

This does not replace a golden master: it does not check that the NUMBERS are
right, only that the script runs to completion the way it is written to. The
golden masters own numerical correctness; this owns "does the code path
between here and a working example still exist."
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class DemoScript:
    path: Path
    #: A string that must appear in stdout on success. None for scripts that
    #: print a report with no fixed final banner (e.g. the HVaR demos, whose
    #: last line is a data-dependent diagnostic table, not a constant).
    completion_marker: str | None
    timeout_seconds: float = 60.0


DEMO_SCRIPTS: tuple[DemoScript, ...] = (
    DemoScript(
        _REPO_ROOT
        / "scripts/demos/sensitivity_pnl/e2e"
        / "demo_e2e_forward_legs_sensitivity_vs_full_reval.py",
        completion_marker="ALL CHECKS PASSED",
    ),
    DemoScript(
        _REPO_ROOT
        / "scripts/demos/sensitivity_pnl/e2e"
        / "demo_e2e_correlated_spots_vs_parametric_normal.py",
        completion_marker="ALL CHECKS PASSED",
    ),
    DemoScript(
        _REPO_ROOT
        / "scripts/demos/sensitivity_pnl/e2e"
        / "demo_e2e_hvar_sensitivity_pnl_and_full_reval.py",
        completion_marker=None,
    ),
    DemoScript(
        _REPO_ROOT
        / "scripts/demos/sensitivity_pnl/e2e"
        / "demo_e2e_hvar_sensitivity_pnl_only.py",
        completion_marker=None,
    ),
    DemoScript(
        _REPO_ROOT
        / "scripts/solutions/sensitivity_pnl"
        / "demo_print_canonical_usage_example.py",
        completion_marker="COPY TO HERE",
    ),
    DemoScript(
        _REPO_ROOT
        / "scripts/demos/sensitivity_pnl/kernels"
        / "taylor_kernel_throughput.py",
        completion_marker=None,
        timeout_seconds=120.0,  # a throughput benchmark, not a fixed workload
    ),
    DemoScript(
        _REPO_ROOT / "scripts/solutions/sensitivity_pnl" / "demo_e2e.py",
        completion_marker="demo complete",
    ),
)


def _run(script: DemoScript) -> subprocess.CompletedProcess[str]:
    assert script.path.is_file(), f"demo script not found: {script.path}"
    return subprocess.run(
        [sys.executable, str(script.path)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=script.timeout_seconds,
        check=False,
    )


@pytest.mark.parametrize(
    "script", DEMO_SCRIPTS, ids=lambda s: s.path.relative_to(_REPO_ROOT).as_posix()
)
def test_demo_script_runs_to_completion(script: DemoScript) -> None:
    result = _run(script)

    assert result.returncode == 0, (
        f"{script.path} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "Traceback (most recent call last)" not in result.stderr, (
        f"{script.path} printed a traceback despite exiting 0:\n{result.stderr}"
    )
    if script.completion_marker is not None:
        assert script.completion_marker in result.stdout, (
            f"{script.path} did not print its completion marker "
            f"{script.completion_marker!r}\n--- stdout tail ---\n"
            + "\n".join(result.stdout.splitlines()[-15:])
        )


def test_every_demo_script_on_disk_is_covered() -> None:
    """
    Guards against the list above silently falling out of date: a new demo
    script added under either directory and never added here would otherwise
    keep accumulating the exact kind of untested rot this file exists to
    catch.
    """
    covered = {script.path for script in DEMO_SCRIPTS}
    on_disk: set[Path] = set()
    for directory in (
        _REPO_ROOT / "scripts/demos/sensitivity_pnl",
        _REPO_ROOT / "scripts/solutions/sensitivity_pnl",
    ):
        if directory.is_dir():
            on_disk.update(directory.rglob("*.py"))

    uncovered = on_disk - covered
    assert not uncovered, (
        "demo script(s) with no runnability coverage in DEMO_SCRIPTS above: "
        f"{sorted(p.relative_to(_REPO_ROOT).as_posix() for p in uncovered)}"
    )
