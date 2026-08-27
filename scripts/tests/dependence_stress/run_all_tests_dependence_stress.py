# scripts/tests/dependence_stress/run_all_tests_dependence_stress.py
import sys
from pathlib import Path

import pytest

from riskquant.utils.printer import print_calling_file, print_success_message


def main():
    print("\n===============================================================")
    print("=== RUNNING TEST SUITE: riskquant.dependence_stress         ===")
    print("===============================================================")

    print_calling_file()

    # Dynamically resolve the absolute path to the tests/dependence_stress/ directory.
    # __file__ is scripts/tests/dependence_stress/run_all_tests_dependence_stress.py
    # .parent is dependence_stress/
    # .parent.parent is tests/
    # .parent.parent.parent is scripts/
    # .parent.parent.parent.parent is riskquant/ (the project root)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    print(project_root)
    test_dir = str(project_root / "tests" / "dependence_stress")

    args = [
        test_dir,
        "-v",
        "--tb=short",
    ]

    exit_code = pytest.main(args)

    print_success_message()

    print("====================================================")
    print("=== END OF DEPENDENCE STRESS TEST SUITE          ===")
    print("====================================================\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
