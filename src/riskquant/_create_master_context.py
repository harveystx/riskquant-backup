# src/riskquant/_create_master_context.py

import argparse
import time
from pathlib import Path
from typing import List


def main():
    parser = argparse.ArgumentParser(
        description="Generate RiskQuant master context."
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=[
            "all",
            "ingest",
            "valuation",
            "simlab",
            "config",
        ],
        help="Select the domain profile to generate context for.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    print(f"Generating context for profile '{args.profile}' from: {base_dir}")

    # 1. Define Domain Profiles
    # Map each profile to the exact directories and interface files it needs.
    PROFILES = {
        "all": [
            base_dir / "config",
            base_dir / "market_data",
            base_dir / "portfolio",
            base_dir / "simlab",
            base_dir / "valuation",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
        ],
        "simlab": [
            base_dir / "simlab",
            base_dir / "market_data" / "state.py",  # Needed for matrix shapes
            base_dir / "market_data" / "risk_factors",  # Needed for snapshots
            base_dir / "config",
            base_dir / "constants.py",
        ],
        "valuation": [
            base_dir / "valuation",
            base_dir / "market_data" / "state.py",
            base_dir / "portfolio" / "compiled.py",
            base_dir / "portfolio" / "blocks",
        ],
        "ingest": [
            base_dir / "portfolio" / "ingest",
            base_dir / "portfolio" / "compilers",
            base_dir / "config",
        ],
        "config": [
            base_dir / "config",
            base_dir / "utils" / "tenors.py",
        ],
    }

    target_paths: List[Path] = PROFILES[args.profile]

    # Name the file based on the profile to avoid overwriting
    out_file = base_dir.parent.parent / f"context_{args.profile}.txt"

    exclude_dirs = {
        "__pycache__",
        ".git",
        ".idea",
        ".venv",
        "venv",
        ".pytest_cache",
        "tests",
        "_create_master_context.py",
        ".ruff_cache",
        "ruff_cache",
    }

    # 2. Gather all valid files
    valid_files: List[Path] = []

    for path in target_paths:
        if not path.exists():
            print(f"Warning: Path not found: {path}")
            continue

        paths_to_check = path.rglob("*.py") if path.is_dir() else [path]

        for subpath in paths_to_check:
            if not any(part in exclude_dirs for part in subpath.parts):
                if (
                    subpath not in valid_files
                ):  # Prevent duplicates if paths overlap
                    valid_files.append(subpath)

    valid_files.sort()

    # Delete existing output file if it exists
    if out_file.exists():
        out_file.unlink()
        time.sleep(0.5)

    # 3. Write the optimized context file
    with out_file.open("w", encoding="utf-8") as out:
        out.write(f"# RISKQUANT CONTEXT: {args.profile.upper()} PROFILE\n\n")
        out.write("### DIRECTORY STRUCTURE & FILES INCLUDED ###\n")
        for f in valid_files:
            out.write(f"- {f.relative_to(base_dir)}\n")
        out.write("\n" + "=" * 60 + "\n\n")

        for subpath in valid_files:
            rel_path = subpath.relative_to(base_dir)
            out.write(f"### FILE: {rel_path} ###\n")
            out.write("```python\n")
            out.write(subpath.read_text(encoding="utf-8"))
            out.write("\n```\n\n")

    print(
        f"Successfully generated {out_file.name} containing {len(valid_files)} files."
    )


if __name__ == "__main__":
    main()

# import time
# from pathlib import Path
# from typing import List


# def main():
#     # 1. Define base directory and target paths clearly
#     base_dir = Path(__file__).resolve().parent
#     print(f"Generating master context from: {base_dir}")

#     target_paths: List[Path] = [
#         base_dir / "config",
#         # # base_dir / "hvar",
#         base_dir / "market_data",
#         # # base_dir / "pfe",
#         base_dir / "portfolio",
#         base_dir / "simlab",
#         base_dir / "utils",
#         base_dir / "valuation",
#         base_dir / "constants.py",
#         base_dir / "exceptions.py",
#         base_dir / "rq_logging.py",
#     ]

#     target_paths: List[Path] = [
#         # base_dir / "market_data" / "risk_factors",
#         # base_dir / "simlab" / "apply",
#         # base_dir / "simlab" / "scenarios",
#         # base_dir / "simlab" / "shocks",
#         base_dir / "simlab",
#         # base_dir / "market_data",
#     ]

#     out_file = base_dir.parent.parent / "master_context.txt"

#     # Added .pytest_cache and tests (unless you want me to review your tests too!)
#     exclude_dirs = {
#         "__pycache__",
#         ".git",
#         ".idea",
#         ".venv",
#         "venv",
#         ".pytest_cache",
#         "tests",
#         "_create_master_context.py",
#         ".ruff_cache",
#         "ruff_cache",
#     }

#     # 2. Two-pass approach: First, gather all valid files
#     valid_files: List[Path] = []

#     for path in target_paths:
#         if not path.exists():
#             print(f"Warning: Path not found: {path}")
#             continue

#         # Handle both individual files and directories
#         paths_to_check = path.rglob("*.py") if path.is_dir() else [path]

#         for subpath in paths_to_check:
#             if not any(part in exclude_dirs for part in subpath.parts):
#                 valid_files.append(subpath)

#     # Sort files to ensure the file order remains consistent across runs
#     valid_files.sort()

#     # Delete existing output file if it exists to ensure a clean slate
#     if out_file.exists():
#         out_file.unlink()
#         time.sleep(1)  # Small delay to ensure file system updates

#     # 3. Write the optimized context file
#     with out_file.open("w", encoding="utf-8") as out:

#         # --- HEADER & TABLE OF CONTENTS ---
#         out.write("# MASTER CONTEXT FILE FOR RISKQUANT LIBRARY\n\n")
#         out.write("### DIRECTORY STRUCTURE & FILES INCLUDED ###\n")
#         for f in valid_files:
#             # Use relative_to for cleaner paths in the TOC
#             out.write(f"- {f.relative_to(base_dir)}\n")
#         out.write("\n" + "=" * 60 + "\n\n")

#         # --- CODE INJECTION WITH MARKDOWN ---
#         for subpath in valid_files:
#             rel_path = subpath.relative_to(base_dir)

#             # Use clear headers and markdown code blocks for LLM parsing
#             out.write(f"### FILE: {rel_path} ###\n")
#             out.write("```python\n")
#             out.write(subpath.read_text(encoding="utf-8"))
#             out.write("\n```\n\n")

#     print(
#         f"Successfully generated {out_file.name} containing {len(valid_files)} files."
#     )


# if __name__ == "__main__":
#     main()
