# scripts/tests/dependence/run_all_tests_dependence.py

"""
Canonical test invocation for ``riskquant.dependence``.

Two things this script must do that ``pytest.main`` will not do for you.

First, put ``src`` and the repository root on ``sys.path``. The suite imports
``riskquant`` from ``src`` and the dependence tests import shared helpers as
``tests.dependence.conftest``, which needs the repository root importable.
``python -m pytest`` from the repository root supplies both by accident; a
script does not, which is why the previous version of this runner collected
nothing.

Second, run under ``-W error``. Warning-free execution is part of the accepted
standard: the tail-count and coefficient layers are required never to evaluate
``log(0)`` or divide by zero, and a warning is the only evidence that they did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_DIR = PROJECT_ROOT / "tests" / "dependence"


def main() -> None:
    """Run the dependence suite and exit with pytest's status."""
    for entry in (str(PROJECT_ROOT), str(SOURCE_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    if not TEST_DIR.is_dir():
        print(f"[!] test directory not found: {TEST_DIR}")
        sys.exit(1)

    print("=" * 63)
    print("=== RUNNING TEST SUITE: riskquant.dependence".ljust(59) + "===")
    print(f"=== root: {PROJECT_ROOT}".ljust(59) + "===")
    print("=" * 63)

    arguments = [
        str(TEST_DIR),
        "-v",
        "--tb=short",
        "-rs",
        "--durations=10",
        "-W",
        "error",
        *sys.argv[1:],
    ]

    exit_code = pytest.main(arguments)

    if exit_code == pytest.ExitCode.OK:
        print("\n[OK] riskquant.dependence suite passed.")
    elif exit_code == pytest.ExitCode.NO_TESTS_COLLECTED:
        # Collecting nothing is a failure, not a pass. This is the exact defect
        # the previous runner hid.
        print("\n[!] NO TESTS COLLECTED - treating as failure.")
        sys.exit(1)
    else:
        print(f"\n[!] suite finished with exit code {exit_code}")

    sys.exit(int(exit_code))


if __name__ == "__main__":
    main()
