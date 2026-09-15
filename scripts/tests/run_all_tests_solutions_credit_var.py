# scripts/tests/run_all_tests_solutions_credit_var.py
"""Production-grade master test runner for the RiskQuant test suite.

Responsibilities
----------------
* Resolve the project root independently of the current working directory.
* Execute the selected RiskQuant pytest suite.
* Preserve pytest's native, high-signal terminal output.
* Enforce strict pytest configuration and marker validation.
* Show pytest's short test summary and slowest tests.
* Emit an explicit, perfectly aligned final status.
* Return pytest's real exit code to the operating system.

The banner implementation is deliberately ASCII-only so that its geometry is
stable across Windows terminals, IDE consoles, CI systems, and log files.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from riskquant.utils.printer import print_calling_file

# =============================================================================
# Configuration
# =============================================================================

TEST_SUBFOLDER_PATH: str | None = "solutions/credit_var"

BANNER_WIDTH = 88
BANNER_BORDER = "=" * BANNER_WIDTH

# Exact interior width between the two "=== " / " ===" delimiters.
BANNER_CONTENT_WIDTH = BANNER_WIDTH - len("=== ") - len(" ===")

PYTEST_ARGS: tuple[str, ...] = (
    "-vv",
    "--tb=short",
    "-ra",
    "--strict-config",
    "--strict-markers",
    "--durations=20",
    "--color=yes",
)


# =============================================================================
# Console formatting
# =============================================================================


def _banner_line(text: str = "") -> str:
    """Return one banner line with an invariant total width."""
    text = str(text)

    if len(text) > BANNER_CONTENT_WIDTH:
        text = text[:BANNER_CONTENT_WIDTH]

    return f"=== {text:<{BANNER_CONTENT_WIDTH}} ==="


def _print_banner(title: str) -> None:
    """Print a top-level section banner."""
    print()
    print(BANNER_BORDER)
    print(_banner_line(title))
    print(BANNER_BORDER)


def _print_run_context(project_root: Path, test_dir: Path) -> None:
    """Print deterministic execution context."""
    print()
    print("Execution context")
    print("-----------------")
    print(f"Project root : {project_root}")
    print(f"Test suite   : {test_dir}")
    print(f"Python       : {sys.executable}")
    print(f"Pytest       : {pytest.__version__}")
    print()


def _get_status_message(exit_code: pytest.ExitCode) -> str:
    """Map pytest's exit code to a human-readable status."""
    match exit_code:
        case pytest.ExitCode.OK:
            return "[PASS] TEST SUITE PASSED"

        case pytest.ExitCode.TESTS_FAILED:
            return "[FAIL] TEST SUITE FAILED"

        case pytest.ExitCode.INTERRUPTED:
            return "[WARN] TEST SUITE INTERRUPTED"

        case pytest.ExitCode.NO_TESTS_COLLECTED:
            return "[WARN] NO TESTS COLLECTED"

        case pytest.ExitCode.USAGE_ERROR:
            return "[FAIL] PYTEST USAGE / CONFIGURATION ERROR"

        case pytest.ExitCode.INTERNAL_ERROR:
            return "[FAIL] PYTEST INTERNAL ERROR"

        case _:
            return f"[FAIL] TEST SUITE FAILED - EXIT CODE {int(exit_code)}"


def _print_final_status(
    exit_code: pytest.ExitCode,
    elapsed_seconds: float,
) -> None:
    """Print the final test result with perfectly aligned borders."""
    print()
    print(BANNER_BORDER)
    print(_banner_line(_get_status_message(exit_code)))
    print(_banner_line(f"Elapsed: {elapsed_seconds:,.2f}s"))
    print(BANNER_BORDER)
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Execute the selected RiskQuant test suite."""

    # Resolve the repository root from this script's actual location.
    # scripts/tests/run_all_tests.py
    #                     ^ parents[0] = tests
    #                     ^ parents[1] = scripts
    #                     ^ parents[2] = project root
    project_root = Path(__file__).resolve().parents[2]

    test_path = Path("tests")

    if TEST_SUBFOLDER_PATH is not None:
        test_path /= TEST_SUBFOLDER_PATH

    test_dir = project_root / test_path

    # -------------------------------------------------------------------------
    # Startup
    # -------------------------------------------------------------------------

    _print_banner(f"RUNNING RISKQUANT TEST SUITE: {test_path}")

    print_calling_file()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if not project_root.exists():
        print(f"ERROR: Project root does not exist: {project_root}")
        return int(pytest.ExitCode.USAGE_ERROR)

    if not project_root.is_dir():
        print(f"ERROR: Project root is not a directory: {project_root}")
        return int(pytest.ExitCode.USAGE_ERROR)

    if not test_dir.exists():
        print(f"ERROR: Test directory does not exist: {test_dir}")
        return int(pytest.ExitCode.NO_TESTS_COLLECTED)

    if not test_dir.is_dir():
        print(f"ERROR: Test path is not a directory: {test_dir}")
        return int(pytest.ExitCode.NO_TESTS_COLLECTED)

    # -------------------------------------------------------------------------
    # Context
    # -------------------------------------------------------------------------

    _print_run_context(project_root, test_dir)

    pytest_args = [
        str(test_dir),
        *PYTEST_ARGS,
    ]

    print("Pytest arguments")
    print("----------------")
    print(" ".join(pytest_args))
    print()

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    started_at = time.perf_counter()

    try:
        exit_code = pytest.ExitCode(pytest.main(pytest_args))
    except KeyboardInterrupt:
        exit_code = pytest.ExitCode.INTERRUPTED

    elapsed_seconds = time.perf_counter() - started_at

    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------

    _print_final_status(
        exit_code=exit_code,
        elapsed_seconds=elapsed_seconds,
    )

    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
