# src/riskquant/_create_master_context.py

import time
from pathlib import Path
from typing import List


def main():
    # 1. Define base directory and target paths clearly
    base_dir = Path(__file__).resolve().parent
    print(f"Generating master context from: {base_dir}")

    target_paths: List[Path] = [
        base_dir / "config",
        # base_dir / "hvar",
        base_dir / "market_data",
        base_dir / "portfolio",
        # base_dir / "simlab",
        base_dir / "utils",
        # base_dir / "valuation",
        base_dir / "constants.py",
        base_dir / "exceptions.py",
        base_dir / "rq_logging.py",
    ]

    out_file = base_dir / "master_context.txt"

    # Added .pytest_cache and tests (unless you want me to review your tests too!)
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

    # 2. Two-pass approach: First, gather all valid files
    valid_files: List[Path] = []

    for path in target_paths:
        if not path.exists():
            print(f"Warning: Path not found: {path}")
            continue

        # Handle both individual files and directories
        paths_to_check = path.rglob("*.py") if path.is_dir() else [path]

        for subpath in paths_to_check:
            if not any(part in exclude_dirs for part in subpath.parts):
                valid_files.append(subpath)

    # Sort files to ensure the file order remains consistent across runs
    valid_files.sort()

    # Delete existing output file if it exists to ensure a clean slate
    if out_file.exists():
        out_file.unlink()
        time.sleep(1)  # Small delay to ensure file system updates

    # 3. Write the optimized context file
    with out_file.open("w", encoding="utf-8") as out:

        # --- HEADER & TABLE OF CONTENTS ---
        out.write("# MASTER CONTEXT FILE FOR RISKQUANT LIBRARY\n\n")
        out.write("### DIRECTORY STRUCTURE & FILES INCLUDED ###\n")
        for f in valid_files:
            # Use relative_to for cleaner paths in the TOC
            out.write(f"- {f.relative_to(base_dir)}\n")
        out.write("\n" + "=" * 60 + "\n\n")

        # --- CODE INJECTION WITH MARKDOWN ---
        for subpath in valid_files:
            rel_path = subpath.relative_to(base_dir)

            # Use clear headers and markdown code blocks for LLM parsing
            out.write(f"### FILE: {rel_path} ###\n")
            out.write("```python\n")
            out.write(subpath.read_text(encoding="utf-8"))
            out.write("\n```\n\n")

    print(
        f"Successfully generated {out_file.name} containing {len(valid_files)} files."
    )


if __name__ == "__main__":
    main()
