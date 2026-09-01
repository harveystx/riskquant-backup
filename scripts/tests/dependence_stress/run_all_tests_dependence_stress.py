# scripts/tests/dependence_stress/run_all_tests_dependence_stress.py
import sys
from pathlib import Path

import pytest

from riskquant.utils.printer import print_calling_file, print_success_message


def main() -> None:
    print("\n===============================================================")
    print("=== RUNNING TEST SUITE: riskquant.dependence_stress         ===")
    print("===============================================================")

    print_calling_file()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    test_dir = str(project_root / "tests" / "dependence_stress")

    # Base arguments for a rigorous test run
    base_args = [
        test_dir,
        "-v",  # Verbose output
        "--tb=short",  # Short tracebacks
        "-rs",  # Show skipped test reasons
        "--durations=10",  # Profile the 10 slowest tests (useful for tracking JIT overhead)
        "-W",
        "error",  # Treat warnings (like numpy deprecations) as errors
        # "-n", "auto",     # Uncomment if using pytest-xdist for parallel execution
    ]

    # Pass through any additional CLI arguments (e.g., -k "test_name" or --pdb)
    cli_args = sys.argv[1:]
    args = base_args + cli_args

    exit_code = pytest.main(args)

    # Only print the success message if all tests passed
    if exit_code == pytest.ExitCode.OK:
        print_success_message()
    else:
        print(f"\n[!] Test suite finished with exit code {exit_code}")

    print("====================================================")
    print("=== END OF DEPENDENCE STRESS TEST SUITE          ===")
    print("====================================================\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
