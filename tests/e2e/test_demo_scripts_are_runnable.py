# tests/e2e/test_demo_scripts_are_runnable.py
"""
`scripts/demos/e2e/demo_e2e_sensi_based_var.py` and
`demo_e2e_sensi_based_var_models.py`, run as real subprocesses, once each.

Mirrors `tests/solutions/sensitivity_pnl/test_demo_scripts_are_runnable.py`:
a demo script with zero execution coverage is a code path nothing verifies,
and `scripts/` is exactly where visible-but-untested code accumulates. This
file did not exist before this refactor -- `test_demo_e2e_sensi_based_var.py`
only ever imported the demo module and called individual `_run_*` functions
directly; nothing ran either script end to end as a user would.

Each script is checked for exit code 0, no traceback on stderr, and its
completion marker. This does not replace a golden master: it does not check
that the numbers are right (see `test_demo_e2e_sensi_based_var.py` for that),
only that the script runs to completion the way it is written to.

`_fixtures.py` is deliberately NOT covered here: it is a shared helper module
with no `main()` / `if __name__ == "__main__":` entry point, not a runnable
demo -- the leading underscore is the same "not a script" convention
`scripts/run_all_scripts.py` already applies to its own exclusions.
`demo_e2e_canonical_base_valuation.py` is a pre-existing, unrelated demo in
the same directory that this refactor did not touch; it is out of scope for
this file (see the current-state audit for a discovered, pre-existing defect
in its own `print_df`/polars-vs-pandas call path).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DIR = _REPO_ROOT / "scripts" / "demos" / "e2e"


@dataclass(frozen=True, slots=True)
class DemoScript:
    path: Path
    completion_marker: str
    timeout_seconds: float = 120.0


DEMO_SCRIPTS: tuple[DemoScript, ...] = (
    DemoScript(
        _DEMO_DIR / "demo_e2e_sensi_based_var.py",
        completion_marker="ALL CHECKS PASSED",
    ),
    DemoScript(
        _DEMO_DIR / "demo_e2e_sensi_based_var_models.py",
        completion_marker="ALL CHECKS PASSED",
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
    assert script.completion_marker in result.stdout, (
        f"{script.path} did not print its completion marker "
        f"{script.completion_marker!r}\n--- stdout tail ---\n"
        + "\n".join(result.stdout.splitlines()[-15:])
    )
