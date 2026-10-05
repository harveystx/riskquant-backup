# tests/e2e/test_demo_scripts_are_runnable.py
"""
The registered e2e demo gate: the explicit NORMATIVE set of
`scripts/demos/e2e/`, each run as a real subprocess.

Registration contract (PR-07)
-----------------------------
* Every entry of `scripts/demos/e2e/` whose name does not start with ``_`` is
  either REGISTERED here (a normative regression demo) or EXCLUDED with a
  stated reason. `test_the_registry_covers_the_demo_directory_exactly`
  refuses an unclassified file, a stale registration and a file in both sets,
  so a new or removed demo can never drop out of the gate silently.
  A ``_``-prefixed file is a shared helper module, not an entry point (the
  `scripts/run_all_scripts.py` convention).
* A registered demo passes only if it exits 0, prints no traceback on
  stderr, and prints its completion marker. The 00-07 chapters print
  ``ALL CHECKS PASSED`` from `Checks.done()`, reached only after every
  `check` and `expect_rejection` succeeded; any failed check raises
  `DemoCheckError` and the process exits non-zero.
  `demo_e2e_sensi_based_var.py` exits non-zero on ``VALIDATION FAILED``.
* The failure detection itself is tested (`test_the_gate_detects_*`) against
  throw-away scripts, so a gate that cannot fail is caught.

This gate proves that each chapter's stated checks hold on the current
engine. It is not the only evidence for any invariant: each demo's
`purpose` names the contract it exercises; the unit, oracle and certification
tests under `tests/` remain the primary evidence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DIR = _REPO_ROOT / "scripts" / "demos" / "e2e"


_MARKER = "ALL CHECKS PASSED"


@dataclass(frozen=True, slots=True)
class DemoScript:
    path: Path
    purpose: str
    completion_marker: str = _MARKER
    timeout_seconds: float = 120.0


def _demo(name: str, purpose: str, timeout_seconds: float = 120.0) -> DemoScript:
    return DemoScript(_DEMO_DIR / name, purpose, timeout_seconds=timeout_seconds)


#: The normative e2e set, in reading order. Each purpose names the contract the
#: demo's checks exercise (see the demo's own docstring for the evidence type).
DEMO_SCRIPTS: tuple[DemoScript, ...] = (
    _demo(
        "00_spot_getting_started.py",
        "SPOT pipeline vs hand q*P*X oracle; RiskRun == building blocks bitwise",
    ),
    _demo(
        "01_spot_taylor_derived.py",
        "TAYLOR/DERIVED orders vs exact increment and FULL_REVAL bound; D086 identity; "
        "stale-base, uncertified and SUPPLIED refusals",
        timeout_seconds=240.0,
    ),
    _demo(
        "02_spot_mixed_book.py",
        "mixed FULL_REVAL + DERIVED composition == pure runs bitwise; wiring refusals",
    ),
    _demo(
        "03_spot_fx_and_reporting_currency.py",
        "D081 vs hand; routes, reciprocal FX, reporting-only suffix (D085); "
        "inverted FX leg FULL_REVAL only beside DERIVED legs (D084)",
        timeout_seconds=180.0,
    ),
    _demo(
        "04_spot_scenario_models.py",
        "five scenario models and a 10-day horizon through one call; strict config",
        timeout_seconds=180.0,
    ),
    _demo(
        "05_spot_reproducibility_and_identity.py",
        "identity, reproduce=, fresh process bitwise, D083 result integrity",
        timeout_seconds=240.0,
    ),
    _demo(
        "06_spot_batching_and_scale.py",
        "batch_size operational (D041, D087): identity and VaR/ES bitwise",
        timeout_seconds=300.0,
    ),
    _demo(
        "07_spot_production_example.py",
        "integration template from edge inputs; persist/reload/reproduce bitwise",
    ),
    _demo(
        "demo_e2e_sensi_based_var.py",
        "D019 golden: hand-verified proxy/FX/Taylor wiring; reproducibility",
    ),
)

#: Entry points of the demo directory that are deliberately NOT regression gates.
EXCLUDED: dict[str, str] = {
    "bench_spot_scaling.py": (
        "benchmark, not a regression gate: minutes to hours and up to ~14 GiB; "
        "run by hand, timings are never asserted"
    ),
}


def _run(script: DemoScript) -> subprocess.CompletedProcess[str]:
    assert script.path.is_file(), f"demo script not found: {script.path}"

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    return subprocess.run(
        [sys.executable, str(script.path)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=script.timeout_seconds,
        check=False,
    )


def _verdict(script: DemoScript, result: subprocess.CompletedProcess[str]) -> list[str]:
    """Every reason the run is not a pass; empty means pass."""
    problems = []
    if result.returncode != 0:
        problems.append(f"exited {result.returncode}")
    if "Traceback (most recent call last)" in result.stderr:
        problems.append("printed a traceback on stderr")
    if script.completion_marker not in result.stdout:
        problems.append(
            f"did not print its completion marker {script.completion_marker!r}"
        )
    return problems


@pytest.mark.parametrize(
    "script",
    DEMO_SCRIPTS,
    ids=lambda s: s.path.relative_to(_REPO_ROOT).as_posix(),
)
def test_demo_script_runs_to_completion(script: DemoScript) -> None:
    result = _run(script)
    problems = _verdict(script, result)
    tail = "\n".join(result.stdout.splitlines()[-25:])
    assert not problems, (
        f"{script.path.name} ({script.purpose}): {'; '.join(problems)}\n"
        f"--- stdout tail ---\n{tail}\n--- stderr ---\n{result.stderr}"
    )


def test_the_registry_covers_the_demo_directory_exactly() -> None:
    entries = {p.name for p in _DEMO_DIR.glob("*.py") if not p.name.startswith("_")}
    registered = [s.path.name for s in DEMO_SCRIPTS]
    assert len(registered) == len(set(registered)), "a demo is registered twice"
    missing = [s.path for s in DEMO_SCRIPTS if not s.path.is_file()]
    assert not missing, f"registered demo(s) not found: {missing}"
    both = set(registered) & set(EXCLUDED)
    assert not both, f"registered AND excluded: {sorted(both)}"
    unclassified = entries - set(registered) - set(EXCLUDED)
    assert not unclassified, (
        f"demo(s) neither registered nor excluded with a reason: {sorted(unclassified)}"
    )
    stale = set(EXCLUDED) - entries
    assert not stale, f"exclusion(s) for file(s) that no longer exist: {sorted(stale)}"
    assert all(s.purpose and s.completion_marker for s in DEMO_SCRIPTS)


# ---- the gate itself must be able to fail -----------------------------------------

_GATE_CASES = {
    "non_zero_exit": (
        "import sys\nprint('ALL CHECKS PASSED')\nsys.exit(3)\n",
        "exited 3",
    ),
    "failed_check": ("raise AssertionError('[FAIL] a check')\n", "exited 1"),
    "no_marker_exit_0": ("print('VALIDATION FAILED')\n", "completion marker"),
    "swallowed_traceback": (
        (
            "import traceback, sys\n"
            "try:\n    1 / 0\nexcept ZeroDivisionError:\n    traceback.print_exc()\n"
            "print('ALL CHECKS PASSED')\n"
        ),
        "traceback",
    ),
}


@pytest.mark.parametrize("case", list(_GATE_CASES))
def test_the_gate_detects_a_failing_demo(tmp_path: Path, case: str) -> None:
    body, expected = _GATE_CASES[case]
    script = DemoScript(tmp_path / f"{case}.py", "gate self-test")
    script.path.write_text(body, encoding="utf-8")
    problems = _verdict(script, _run(script))
    assert any(expected in p for p in problems), problems


def test_the_gate_detects_a_missing_demo(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="demo script not found"):
        _run(DemoScript(tmp_path / "absent.py", "gate self-test"))


def test_a_passing_demo_passes_the_gate(tmp_path: Path) -> None:
    script = DemoScript(tmp_path / "ok.py", "gate self-test")
    script.path.write_text("print('ALL CHECKS PASSED')\n", encoding="utf-8")
    assert _verdict(script, _run(script)) == []
