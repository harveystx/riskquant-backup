# src/riskquant/_create_master_context.py
import argparse
import time
from pathlib import Path


def main(profile: str | None = None):
    # 1. Handle Inputs: Check if called directly vs from CLI
    if profile is None:
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
                "simlab_extended",
                "config",
            ],
            help="Select the domain profile to generate context for.",
        )
        args = parser.parse_args()
        profile = args.profile

    # Handle interactive environments where __file__ might not be defined
    try:
        base_dir = Path(__file__).resolve().parent
        base_dir_parent = base_dir.parent.parent
    except NameError:
        # Fallback for interactive windows/REPLs
        # Assumes you are running this from the project root
        base_dir = Path().resolve() / "src" / "riskquant"
        base_dir_parent = base_dir.parent.parent

    print(f"Generating context for profile '{profile}' from: {base_dir}")

    # 2. Define Domain Profiles
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
            base_dir / "market_data" / "state.py",
            base_dir / "market_data" / "risk_factors",
            base_dir / "config",
            base_dir / "constants.py",
        ],
        "simlab_reduced": [
            base_dir / "simlab",
            base_dir / "constants.py",
        ],
        "simlab_extended": [
            base_dir / "simlab",
            base_dir / "portfolio",
            base_dir / "market_data",
            base_dir / "valuation",
            base_dir / "config",
            base_dir / "constants.py",
            base_dir / "utils",
            base_dir / "pnl" / "results.py",
        ],
        "valuation": [
            base_dir / "valuation",
            base_dir / "constants.py",
            base_dir_parent / "scripts" / "demos" / "valuation",
        ],
        "pricers": [
            base_dir / "valuation",
            base_dir / "constants.py",
            base_dir_parent / "scripts" / "demos" / "valuation",
        ],
        "pricer_kernels": [
            base_dir / "valuation" / "kernels",
            base_dir_parent / "scripts" / "demos" / "valuation",
        ],
        "config": [
            base_dir / "config",
            base_dir / "utils" / "tenors.py",
        ],
        "config_static": [
            base_dir / "config" / "enums",
            base_dir / "config" / "static_config",
            base_dir / "market_data" / "curves",
            base_dir / "utils" / "tenors.py",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
        ],
        "config_proxy": [
            base_dir / "config" / "enums",
            base_dir / "config" / "proxy_config",
            base_dir / "market_data" / "curves",
            base_dir / "utils" / "tenors.py",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
        ],
        "ce_jtd": [
            base_dir / "portfolio",
            base_dir / "simlab" / "scenarios" / "base.py",
            base_dir / "simlab" / "scenarios" / "parametric_copula_t.py",
            base_dir / "simlab" / "scenarios" / "parametric_copula_gaussian.py",
            base_dir / "valuation",
            base_dir / "constants.py",
        ],
        "ingest": [
            base_dir / "config",
            # base_dir / "credit",
            base_dir / "ingest_fixtures",
            base_dir / "market_data",
            base_dir / "portfolio",
            base_dir / "proxy_resolver",
            base_dir / "simlab",
            base_dir / "utils",
            base_dir / "valuation",
            base_dir / "_njit.py",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
            base_dir_parent / "scripts" / "demos" / "config",
            base_dir_parent / "scripts" / "demos" / "credit" / "rc",
            base_dir_parent / "scripts" / "demos" / "valuation",
            base_dir_parent / "scripts" / "demos" / "simlab",
        ],
        "ingest_fixtures": [
            base_dir / "ingest_fixtures",
        ],
        "ingest_fixtures_market_data": [
            base_dir / "ingest_fixtures" / "config",
            base_dir / "ingest_fixtures" / "market_data",
            base_dir / "market_data",
            base_dir_parent
            / "scripts"
            / "demos"
            / "market_data"
            / "risk_factors"
            / "demo_snapshot.py",
        ],
        "ingest_fixtures_risk_static_config": [
            base_dir / "ingest_fixtures" / "config" / "risk_static_config",
        ],
        "risk_quant": [
            # base_dir / "portfolio" / "ingest" / "schema.py",
            # base_dir / "portfolio" / "ingest" / "canonicalize.py",
            # base_dir / "portfolio" / "ingest" / "materialize.py",
            # base_dir / "portfolio" / "ingest" / "validate.py",
            # base_dir / "portfolio" / "ingest" / "build.py",
            base_dir / "config",
            base_dir / "ingest_fixtures",
            base_dir / "market_data",
            base_dir / "portfolio",
            # base_dir / "portfolio" / "ingest",
            # base_dir / "portfolio" / "blocks.py",
            # base_dir / "portfolio" / "compilers.py",
            # base_dir / "portfolio" / "instruments.py",
            base_dir / "valuation" / "pricers",
            base_dir / "valuation" / "engine.py",
            base_dir / "simlab",
            base_dir / "utils",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
        ],
        "risk_quant_adhoc": [
            base_dir / "ingest_fixtures",
            base_dir / "portfolio",
            base_dir / "valuation" / "pricers",
            base_dir / "valuation" / "engine.py",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
            base_dir / "utils",
        ],
        # "pricers": [
        #     base_dir / "valuation" / "pricers",
        #     base_dir / "valuation" / "engine.py",
        # ],
        "market_data": [
            base_dir / "market_data",
            base_dir / "ingest_fixtures",
            base_dir / "config",
            base_dir / "utils",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
        ],
        "market_data_reduced": [
            base_dir / "market_data",
            # base_dir / "ingest_fixtures",
            # base_dir / "config",
            # base_dir / "utils",
            # base_dir / "constants.py",
            # base_dir / "exceptions.py",
        ],
        "credit": [
            base_dir / "credit",
            base_dir / "config",
            base_dir / "ingest_fixtures",
            base_dir / "market_data",
            base_dir / "portfolio",
            base_dir / "valuation" / "pricers",
            base_dir / "valuation" / "engine.py",
            base_dir / "simlab",
            base_dir / "utils",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
            # base_dir_parent / "tests" / "credit" / "exposure",
            base_dir_parent / "scripts" / "demos" / "credit",
            # base_dir_parent / "scripts" / "reporting" / "credit_risk" / "stress_tests",
        ],
        "credit_reduced": [
            base_dir / "credit",
            # base_dir / "config",
            # base_dir / "ingest_fixtures",
            base_dir / "market_data",
            base_dir / "portfolio",
            base_dir / "valuation",
            base_dir / "simlab",
            base_dir / "utils",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
            # base_dir_parent / "tests" / "credit" / "exposure",
            # base_dir_parent / "scripts" / "reporting" / "credit_risk" / "stress_tests",
        ],
        "portfolio": [
            base_dir / "portfolio",
            base_dir / "utils",
            base_dir / "constants.py",
            base_dir / "exceptions.py",
        ],
    }

    # print("base_dir_parent`:", base_dir_parent)

    if profile not in PROFILES:
        print(
            f"Error: Profile '{profile}' is not valid. Choose from {list(PROFILES.keys())}."
        )
        return

    target_paths: list[Path] = PROFILES[profile]
    out_file = base_dir.parent.parent / "_contexts" / f"context_{profile}.txt"

    exclude_dirs = {
        "__pycache__",
        ".git",
        ".idea",
        ".venv",
        ".env",
        "venv",
        ".pytest_cache",
        "tests",
        "_create_master_context.py",
        ".ruff_cache",
        "ruff_cache",
    }

    # 3. Gather all valid files
    valid_files: list[Path] = []

    for path in target_paths:
        if not path.exists():
            print(f"Warning: Path not found: {path}")
            continue

        paths_to_check = path.rglob("*.py") if path.is_dir() else [path]

        for subpath in paths_to_check:
            if (
                not any(part in exclude_dirs for part in subpath.parts)
                and subpath not in valid_files
                and not subpath.name.startswith(".")  # Exclude hidden files
            ):
                valid_files.append(subpath)

    valid_files.sort()

    if out_file.exists():
        out_file.unlink()
        time.sleep(0.5)

    # 4. Write the optimized context file using XML tagging
    with out_file.open("w", encoding="utf-8") as out:
        # Preamble to prime the AI
        out.write("<purpose>\n")
        out.write(
            f"This document contains the source code for the RiskQuant '{profile.upper()}' profile.\n"
        )
        out.write(
            "Use this structural map and source code as the ground-truth context for answering subsequent queries.\n"
            "This is god-tier quantitative code, not typical Python scripts. Pay close attention to the structure and patterns used.\n"
            "RiskQuant prioritizes Data-Oriented Design (DOD), Structures-of-Arrays (SoA), bare-metal memory sympathy, and absolute maximum throughput. The core philosophy of this codebase is: Human-readable at the edge, C-contiguous at the core. Strings must die early. Bare-metal performance is the north star.\n"
        )
        out.write("</purpose>\n\n")

        # Give the AI a mental map of the directory before dumping the code
        out.write("<directory_structure>\n")
        project_folder = base_dir_parent.parent
        for f in valid_files:
            # print(f)
            rel = f.relative_to(project_folder)
            out.write(f"{rel}\n")
        out.write("</directory_structure>\n\n")

        # Isolate the code blocks cleanly
        out.write("<files>\n")
        for subpath in valid_files:
            rel_path = subpath.relative_to(project_folder)
            # .strip() removes excessive trailing newlines to save tokens
            content = subpath.read_text(encoding="utf-8").strip()

            out.write(f'<file path="{rel_path}">\n')
            out.write(f"```python\n{content}\n```\n")
            out.write("</file>\n\n")
        out.write("</files>\n")

    print(
        f"Successfully generated {out_file.name} containing {len(valid_files)} files."
    )


if __name__ == "__main__":
    main(profile="all")
    main(profile="config")
    main(profile="config_static")
    main(profile="config_proxy")
    main(profile="valuation")
    main(profile="simlab")
    main(profile="simlab_reduced")
    main(profile="simlab_extended")
    main(profile="ingest")
    main(profile="ce_jtd")
    main(profile="risk_quant")
    main(profile="risk_quant_adhoc")
    main(profile="pricers")
    main(profile="market_data")
    main(profile="market_data_reduced")
    main(profile="credit")
    main(profile="credit_reduced")
    main(profile="ingest_fixtures")
    main(profile="ingest_fixtures_risk_static_config")
    main(profile="ingest_fixtures_market_data")
    main(profile="portfolio")
    main(profile="pricer_kernels")
