# tests/solutions/sensitivity_pnl/test_canonical_usage_example_is_runnable.py
"""
The one guarantee `demo_print_canonical_usage_example.py` exists to make: what
it prints is not merely readable, it runs.

`CANONICAL_EXAMPLE_SOURCE` is written to a real temp file and executed as a
real subprocess with a clean environment (nothing inherited from the test
runner's own working directory or fixtures) — the exact scenario described to
the person reading the printed output: paste this into a .py file, run it.

There is exactly one copy of the example text. This test does not re-type any
of it; it imports the same constant the demo prints, so the two can never
silently drift apart.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest


def _load_canonical_example_source() -> str:
    """
    `scripts/` ships demo/example code, not an installed package — there is no
    `scripts/__init__.py` to import through, by design. Load the module
    directly from its file path instead, resolved relative to this test file
    rather than the working directory a test runner happens to be invoked
    from.
    """
    repo_root = Path(__file__).resolve().parents[3]
    demo_path = (
        repo_root
        / "scripts"
        / "solutions"
        / "sensitivity_pnl"
        / "demo_print_canonical_usage_example.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_print_canonical_usage_example", demo_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source: str = module.CANONICAL_EXAMPLE_SOURCE
    return source


CANONICAL_EXAMPLE_SOURCE = _load_canonical_example_source()

_ROW_PATTERN = re.compile(
    r"^\s*(\S+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*$", re.MULTILINE
)


def test_canonical_example_contains_no_triple_single_quote() -> None:
    """
    The demo wraps this text in a raw `r'''...'''` string literal. A stray
    `'''` inside the example would terminate that literal early and corrupt
    everything printed after it — silently, since the wrapping module would
    still be syntactically valid Python, just wrong. Guard the assumption
    directly rather than relying on `py_compile` to notice.
    """
    assert "'''" not in CANONICAL_EXAMPLE_SOURCE


def test_canonical_example_uses_only_the_public_solutions_api() -> None:
    """
    "Canonical inputs expected by the solutions API" means the public facade,
    not a core-tier internal a real user would never be told to import.
    """
    assert (
        "from riskquant.solutions.sensitivity_pnl.api import"
        in CANONICAL_EXAMPLE_SOURCE
    )
    assert "riskquant.sensitivity_pnl.compile" not in CANONICAL_EXAMPLE_SOURCE
    assert "riskquant.sensitivity_pnl.engine" not in CANONICAL_EXAMPLE_SOURCE
    assert "riskquant.sensitivity_pnl.contracts" not in CANONICAL_EXAMPLE_SOURCE


@pytest.mark.parametrize("run_index", [1, 2])
def test_canonical_example_runs_as_a_standalone_file(
    tmp_path: Path, run_index: int
) -> None:
    """
    Runs the printed text, unmodified, as its own subprocess — twice, to also
    confirm the fixed seed makes it exactly reproducible, since a "copy this
    and run it" example that gives a different answer each time would be a
    much worse thing to have shipped than one that merely failed loudly.
    """
    script_path = tmp_path / "sensitivity_pnl_var_example.py"
    script_path.write_text(CANONICAL_EXAMPLE_SOURCE)

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        f"canonical example exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )

    rows = {
        match.group(1): (
            float(match.group(2).replace(",", "")),
            float(match.group(3).replace(",", "")),
        )
        for match in _ROW_PATTERN.finditer(completed.stdout)
    }
    assert set(rows) == {"1001", "1002", "1003", "portfolio"}, completed.stdout

    for label, (var, es) in rows.items():
        assert var > 0.0, f"{label}: VaR must be a positive loss figure, got {var}"
        assert es >= var, f"{label}: ES must be at least VaR, got ES={es} VaR={var}"

    # The known hedge: 1003 nets against part of 1001's EQ_ACME delta, so the
    # portfolio must show real diversification against the naive sum, and in
    # particular must come in below 1001's own standalone VaR.
    naive_sum = sum(var for label, (var, _es) in rows.items() if label != "portfolio")
    portfolio_var, _ = rows["portfolio"]
    assert portfolio_var < naive_sum
    assert portfolio_var < rows["1001"][0]


def test_canonical_example_is_reproducible_across_runs(tmp_path: Path) -> None:
    """Same fixed seed, byte-identical stdout, run twice in separate processes."""
    script_path = tmp_path / "sensitivity_pnl_var_example.py"
    script_path.write_text(CANONICAL_EXAMPLE_SOURCE)

    outputs = [
        subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
