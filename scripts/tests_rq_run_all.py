# scripts/tests_rq_run_all.py
import sys
from pathlib import Path
import pytest

from riskquant.utils.printer import print_calling_file


def main():
    print("\n====================================================")
    print("=== RUNNING RISKQUANT MASTER TEST SUITE          ===")
    print("====================================================\n")

    print_calling_file()

    # Dynamically resolve the absolute path to the tests/ directory.
    # __file__ is scripts/tests_rq_run_all.py
    # .parent is scripts/
    # .parent.parent is riskquant/ (the project root)
    project_root = Path(__file__).resolve().parent.parent
    test_dir = str(project_root / "tests")

    args = [
        test_dir,  # Now passing the absolute path: C:\...\riskquant\tests
        "-v",
        "--tb=short",
    ]

    exit_code = pytest.main(args)
    sys.exit(exit_code)

    print("\n====================================================")
    print("=== END OF RISKQUANT MASTER TEST SUITE          ===")
    print("====================================================\n")


if __name__ == "__main__":
    main()
