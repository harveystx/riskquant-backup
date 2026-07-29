# src/riskquant/_create_master_context.py
import argparse
import ast
import time
from dataclasses import dataclass
from pathlib import Path

from riskquant.utils.printer import print_calling_file, print_success_message


@dataclass(slots=True)
class FileSummary:
    path: Path
    module_docstring: str | None
    imports: list[str]
    classes: list[str]
    functions: list[str]


def analyse_file(path: Path) -> FileSummary:
    """
    Lightweight AST analysis for LLM navigation.

    Extracts:

    - module docstring
    - public imports
    - public classes
    - public functions

    The goal is architectural navigation rather than full semantic analysis.
    """

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return FileSummary(
            path=path,
            module_docstring=None,
            imports=[],
            classes=[],
            functions=[],
        )

    imports: set[str] = set()
    classes: list[str] = []
    functions: list[str] = []

    for node in tree.body:
        # ==========================================================
        # Imports
        # ==========================================================

        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue

        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            continue

        # ==========================================================
        # Public classes
        # ==========================================================

        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            continue

        # ==========================================================
        # Public functions
        # ==========================================================

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            continue

    return FileSummary(
        path=path,
        module_docstring=ast.get_docstring(tree),
        imports=sorted(imports),
        classes=sorted(classes),
        functions=sorted(functions),
    )


def build_directory_tree(
    files: list[Path],
    root: Path,
) -> str:
    """
    Build a unicode directory tree from a list of files.
    """

    tree: dict[str, dict] = {}

    for file in files:
        rel = file.relative_to(root)

        node = tree

        for part in rel.parts:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def walk(
        node: dict[str, dict],
        prefix: str = "",
    ) -> None:

        items = sorted(node.items())

        for i, (name, child) in enumerate(items):
            is_last = i == len(items) - 1

            connector = "└── " if is_last else "├── "

            lines.append(prefix + connector + name)

            extension = "    " if is_last else "│   "

            walk(
                child,
                prefix + extension,
            )

    walk(tree)

    return "\n".join(lines)


def main(profile: str | None = None):
    print_calling_file()

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
        rq_project_directory = Path(__file__).resolve().parent
    except NameError:
        # Fallback for interactive windows/REPLs
        # Assumes you are running this from the project root
        rq_project_directory = Path().resolve()

    rq_src_path = rq_project_directory / "src" / "riskquant"
    rq_lab_src_path = rq_project_directory / "src" / "riskquant_lab"

    print("\n" + "-" * 100)
    print("PROFILE:", profile)
    print(f"Generating context for profile '{profile}' from: {rq_project_directory}")

    # 2. Define Domain Profiles
    PROFILES = {
        "all": [
            rq_src_path / "config",
            rq_src_path / "market_data",
            rq_src_path / "portfolio",
            rq_src_path / "simlab",
            rq_src_path / "valuation",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
        ],
        "simlab": [
            rq_src_path / "simlab",
            rq_src_path / "market_data" / "state.py",
            rq_src_path / "market_data" / "risk_factors",
            rq_src_path / "config",
            rq_src_path / "constants.py",
        ],
        "simlab_reduced": [
            rq_src_path / "simlab",
            rq_src_path / "constants.py",
        ],
        "simlab_extended": [
            rq_src_path / "simlab",
            rq_src_path / "portfolio",
            rq_src_path / "market_data",
            rq_src_path / "valuation",
            rq_src_path / "config",
            rq_src_path / "constants.py",
            rq_src_path / "utils",
            rq_src_path / "pnl" / "results.py",
        ],
        "valuation": [
            rq_src_path / "market_data" / "state.py",
            rq_src_path / "utils",
            rq_src_path / "valuation",
            rq_src_path / "_arrays.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "jit.py",
        ],
        "pricers": [
            rq_src_path / "valuation",
            rq_src_path / "constants.py",
            rq_project_directory / "scripts" / "demos" / "valuation",
        ],
        "pricer_kernels": [
            rq_src_path / "valuation" / "kernels",
            rq_project_directory / "scripts" / "demos" / "valuation",
        ],
        "config": [
            rq_src_path / "config",
            rq_src_path / "utils" / "tenors.py",
        ],
        "config_static": [
            rq_src_path / "config" / "enums",
            rq_src_path / "config" / "static_config",
            rq_src_path / "market_data" / "curves",
            rq_src_path / "utils" / "tenors.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
        ],
        "config_proxy": [
            rq_src_path / "config" / "enums",
            rq_src_path / "config" / "proxy_config",
            rq_src_path / "market_data" / "curves",
            rq_src_path / "utils" / "tenors.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
        ],
        "ce_jtd": [
            rq_src_path / "portfolio",
            rq_src_path / "simlab" / "scenarios" / "base.py",
            rq_src_path / "valuation",
            rq_src_path / "constants.py",
        ],
        "ingest": [
            rq_src_path / "config",
            # rq_path / "credit",
            rq_src_path / "ingest_fixtures",
            rq_src_path / "market_data",
            rq_src_path / "pnl",
            rq_src_path / "portfolio",
            rq_src_path / "proxy_resolver",
            rq_src_path / "scenario_metrics",
            rq_src_path / "simlab",
            rq_src_path / "utils",
            rq_src_path / "valuation",
            rq_src_path / "_arrays.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "jit.py",
            # rq_path / "synthetic_level_generators.py",
            rq_src_path / "universe.py",
            # SCRIPTS
            # base_dir_parent / "scripts" / "demos" / "config",
            # base_dir_parent / "scripts" / "demos" / "credit",
            # base_dir_parent / "scripts" / "demos" / "e2e",
            # base_dir_parent / "scripts" / "demos" / "market_data",
            # base_dir_parent / "scripts" / "demos" / "misc",
            # base_dir_parent / "scripts" / "demos" / "simlab",
            # base_dir_parent / "scripts" / "demos" / "valuation",
        ],
        "ingest_fixtures": [
            rq_src_path / "ingest_fixtures",
        ],
        "ingest_fixtures_market_data": [
            rq_src_path / "ingest_fixtures" / "config",
            rq_src_path / "ingest_fixtures" / "market_data",
            rq_src_path / "market_data",
            rq_project_directory
            / "scripts"
            / "demos"
            / "market_data"
            / "risk_factors"
            / "demo_snapshot.py",
        ],
        "ingest_fixtures_risk_static_config": [
            rq_src_path / "ingest_fixtures" / "config" / "risk_static_config",
        ],
        "risk_quant": [
            # rq_path / "portfolio" / "ingest" / "schema.py",
            # rq_path / "portfolio" / "ingest" / "canonicalize.py",
            # rq_path / "portfolio" / "ingest" / "materialize.py",
            # rq_path / "portfolio" / "ingest" / "validate.py",
            # rq_path / "portfolio" / "ingest" / "build.py",
            rq_src_path / "config",
            rq_src_path / "ingest_fixtures",
            rq_src_path / "market_data",
            rq_src_path / "portfolio",
            # rq_path / "portfolio" / "ingest",
            # rq_path / "portfolio" / "blocks.py",
            # rq_path / "portfolio" / "compilers.py",
            # rq_path / "portfolio" / "instruments.py",
            rq_src_path / "valuation" / "pricers",
            rq_src_path / "valuation" / "engine.py",
            rq_src_path / "simlab",
            rq_src_path / "utils",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
        ],
        "risk_quant_adhoc": [
            rq_src_path / "ingest_fixtures",
            rq_src_path / "portfolio",
            rq_src_path / "valuation" / "pricers",
            rq_src_path / "valuation" / "engine.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "utils",
        ],
        # "pricers": [
        #     rq_path / "valuation" / "pricers",
        #     rq_path / "valuation" / "engine.py",
        # ],
        "market_data_only": [
            # rq_path / "config",
            # rq_path / "ingest_fixtures",
            rq_src_path / "market_data",
            # rq_path / "utils",
            # rq_path / "valuation",
            # rq_path / "_arrays.py",
            # rq_path / "constants.py",
            # rq_path / "exceptions.py",
            # rq_path / "jit.py",
            # rq_path / "universe.py",
            # # SCRIPTS
            # base_dir_parent / "scripts" / "demos" / "market_data",
            # # TESTS
            # base_dir_parent / "tests" / "market_data",
        ],
        "portfolio_only": [
            # rq_path / "config",
            rq_src_path / "portfolio",
            # rq_path / "utils",
            # rq_path / "valuation",
            # rq_path / "_arrays.py",
            # rq_path / "jit.py",
            # rq_path / "constants.py",
            # rq_path / "exceptions.py",
            # rq_path / "universe.py",
        ],
        "solutions_credit_var": [
            rq_src_path / "solutions" / "credit_var",
            rq_project_directory / "scripts" / "solutions" / "credit_var",
            rq_lab_src_path,
        ],
        "solutions_credit_exposure": [
            rq_src_path / "solutions" / "credit_exposure",
            rq_project_directory / "scripts" / "solutions" / "credit_exposure",
            rq_lab_src_path,
        ],
        "solutions_sensitivity_pnl": [
            rq_src_path / "solutions" / "sensitivity_pnl",
            rq_project_directory / "scripts" / "solutions" / "sensitivity_pnl",
            rq_lab_src_path,
        ],
        "market_data_reduced": [
            rq_src_path / "market_data",
            # rq_path / "ingest_fixtures",
            # rq_path / "config",
            rq_src_path / "valuation" / "kernels" / "_math.py",
            rq_src_path / "utils",
            rq_src_path / "_arrays.py",
            rq_src_path / "jit.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "universe.py",
        ],
        "market_data_calibrate": [
            rq_src_path / "market_data" / "calibrate",
            rq_src_path / "ingest_fixtures",
            rq_src_path / "jit.py",
            rq_src_path / "constants.py",
            # SCRIPTS
            rq_project_directory / "scripts" / "demos" / "market_data" / "calibrate",
            # TESTS
            rq_project_directory / "tests" / "market_data" / "calibrate",
        ],
        "credit": [
            # SOURCE CODE
            rq_src_path / "credit",
            # rq_path / "config",
            rq_src_path / "ingest_fixtures" / "credit",
            rq_src_path / "ingest_fixtures" / "as_of.py",
            rq_src_path / "ingest_fixtures" / "constants.py",
            rq_src_path / "ingest_fixtures" / "rng.py",
            rq_src_path / "ingest_fixtures" / "schemas.py",
            rq_src_path / "ingest_fixtures" / "utils.py",
            # rq_path / "market_data",
            # rq_path / "portfolio",
            # rq_path / "valuation" / "pricers",
            # rq_path / "valuation" / "engine.py",
            # rq_path / "simlab",
            rq_src_path / "solutions" / "credit_var",
            rq_src_path / "utils",
            rq_src_path / "scenario_metrics",
            rq_src_path / "_arrays.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "jit.py",
            rq_src_path / "universe.py",
            # SCRIPTS
            rq_project_directory / "scripts" / "demos" / "credit" / "exposure",
            rq_project_directory / "scripts" / "demos" / "credit" / "jtd",
            # # base_dir_parent / "scripts" / "reporting" / "credit_risk" / "stress_tests",
        ],
        "credit_jtd": [
            rq_src_path / "credit" / "factors",
            rq_src_path / "credit" / "jtd",
            rq_src_path / "utils",
            rq_src_path / "_arrays.py",
            rq_src_path / "jit.py",
        ],
        "credit_jtd_kernels": [
            rq_src_path / "credit" / "jtd" / "blocks",
            rq_src_path / "credit" / "jtd" / "kernels",
            rq_src_path / "credit" / "jtd" / "evaluators",
        ],
        "credit_exposure": [
            rq_src_path / "credit" / "exposure",
            rq_src_path / "constants.py",
            rq_src_path / "jit.py",
            # SCRIPTS
            rq_project_directory / "scripts" / "demos" / "credit" / "exposure",
        ],
        "basel": [
            rq_src_path / "basel",
            rq_src_path / "utils",
            rq_src_path / "_arrays.py",
            rq_src_path / "jit.py",
            # SCRIPTS
            rq_project_directory / "scripts" / "demos" / "basel",
        ],
        "credit_reduced": [
            # SOURCE CODE
            rq_src_path / "credit",
            # rq_path / "config",
            # rq_path / "ingest_fixtures" / "credit",
            # rq_path / "ingest_fixtures" / "as_of.py",
            # rq_path / "ingest_fixtures" / "constants.py",
            # rq_path / "ingest_fixtures" / "rng.py",
            # rq_path / "ingest_fixtures" / "schemas.py",
            # rq_path / "ingest_fixtures" / "utils.py",
            # rq_path / "market_data",
            # rq_path / "portfolio",
            # rq_path / "valuation" / "pricers",
            # rq_path / "valuation" / "engine.py",
            # rq_path / "simlab",
            rq_src_path / "solutions" / "credit_var",
            # rq_path / "utils",
            # rq_path / "scenario_metrics",
            # rq_path / "_arrays.py",
            # rq_path / "constants.py",
            # rq_path / "exceptions.py",
            # rq_path / "jit.py",
            # rq_path / "universe.py",
            # SCRIPTS
            # base_dir_parent / "scripts" / "demos" / "credit" / "exposure",
            # base_dir_parent / "scripts" / "demos" / "credit" / "jtd",
            # # base_dir_parent / "scripts" / "reporting" / "credit_risk" / "stress_tests",
        ],
        "credit_jtd": [
            # SOURCE CODE
            rq_src_path / "credit" / "factors",
            rq_src_path / "credit" / "jtd",
            rq_src_path / "credit" / "masters",
            rq_src_path / "ingest_fixtures" / "credit" / "credit_factors",
            rq_src_path / "ingest_fixtures" / "credit" / "jtd",
            rq_src_path / "ingest_fixtures" / "as_of.py",
            rq_src_path / "ingest_fixtures" / "constants.py",
            rq_src_path / "ingest_fixtures" / "rng.py",
            rq_src_path / "ingest_fixtures" / "schemas.py",
            rq_src_path / "ingest_fixtures" / "utils.py",
            rq_src_path / "utils",
            rq_src_path / "_arrays.py",
            rq_src_path / "jit.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "universe.py",
            # SCRIPTS
            rq_project_directory / "scripts" / "demos" / "credit" / "exposure",
            rq_project_directory / "scripts" / "demos" / "credit" / "jtd",
        ],
        "portfolio": [
            rq_src_path / "config",
            rq_src_path / "portfolio",
            rq_src_path / "utils",
            rq_src_path / "valuation",
            rq_src_path / "_arrays.py",
            rq_src_path / "jit.py",
            rq_src_path / "constants.py",
            rq_src_path / "exceptions.py",
            rq_src_path / "universe.py",
        ],
        "scenario_metrics": [
            rq_src_path / "scenario_metrics",
            rq_src_path / "jit.py",
        ],
        "sensitivity_pnl": [
            # CODE
            rq_src_path / "sensitivity_pnl",
            rq_src_path / "_arrays.py",
            rq_src_path / "jit.py",
            rq_src_path / "constants.py",
            # SCRIPTS
            rq_project_directory / "scripts" / "demos" / "sensitivity_pnl",
            # TESTS
            rq_project_directory / "tests" / "sensitivity_pnl",
        ],
    }

    # print("base_dir_parent`:", base_dir_parent)

    if profile not in PROFILES:
        print(
            f"Error: Profile '{profile}' is not valid. Choose from {list(PROFILES.keys())}."
        )
        return

    target_paths: list[Path] = PROFILES[profile]
    out_file = rq_project_directory / "_contexts" / f"context_{profile}.txt"

    exclude_dirs = {
        "__pycache__",
        ".git",
        ".idea",
        ".venv",
        ".env",
        "venv",
        ".pytest_cache",
        # "tests",
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

        # paths_to_check = path.rglob("*.py") if path.is_dir() else [path]

        paths_to_check = (
            [p for p in path.rglob("*") if p.suffix in {".py", ".md"}]
            if path.is_dir()
            else [path]
        )

        for subpath in paths_to_check:
            if (
                not any(part in exclude_dirs for part in subpath.parts)
                and subpath not in valid_files
                and not subpath.name.startswith(".")  # Exclude hidden files
                and not subpath.parent.name.startswith("_deprecated")
            ):
                valid_files.append(subpath)

    valid_files.sort()

    file_summaries = [analyse_file(path) for path in valid_files]

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
        out.write("Treat this source code as the authoritative implementation.\n")
        out.write("</purpose>\n\n")

        out.write("<llm_guidance>\n")
        out.write(
            "RiskQuant Design Principles:\n"
            "- Data-Oriented Design (DOD)\n"
            "- Structure-of-Arrays (SoA)\n"
            "- Human-readable at the edge\n"
            "- C-contiguous at the core\n"
            "- Strings must die early\n"
            "- Memory locality matters\n"
            "- CPU efficiency matters\n"
            "- Memory efficiency matters\n"
            "- Correctness over cleverness\n"
            "- Kernels do not contain policy - they are purely computational\n"
            "- Compilers are responsible for policy and orchestration; they must fail fast on invalid inputs\n"
            "- Avoid unnecessary branching in hot paths\n"
            "- Skip '__init__.py' files (they are always empty)\n\n"
            "- Polars over Pandas for DataFrames\n"
            "When proposing code:\n"
            "- Preserve existing SoA layouts\n"
            "- Avoid object-heavy designs in hot paths\n"
            "- Avoid pandas in performance-critical code\n"
            "- Minimize allocations\n"
            "- Prefer vectorized NumPy operations\n"
            "- Preserve existing architectural patterns\n"
        )
        out.write("</llm_guidance>\n\n")

        project_folder = rq_project_directory.parent
        # project_folder = base_dir
        out.write("<directory_tree>\n")
        out.write(
            build_directory_tree(
                valid_files,
                rq_project_directory.parent,
            )
        )
        out.write("\n</directory_tree>\n\n")

        out.write("<symbol_index>\n")

        for summary in file_summaries:
            rel = summary.path.relative_to(project_folder)

            for cls in summary.classes:
                out.write(f'<class name="{cls}" file="{rel}" />\n')

            for fn in summary.functions:
                out.write(f'<function name="{fn}" file="{rel}" />\n')

        out.write("</symbol_index>\n\n")

        # Isolate the code blocks cleanly
        out.write("<files>\n")
        for summary in file_summaries:
            subpath = summary.path

            rel_path = subpath.relative_to(project_folder)

            out.write("<file_summary>\n")
            out.write(f"path: {rel_path}\n")

            if summary.module_docstring:
                out.write(f"module_docstring: {summary.module_docstring[:120]}...\n")

            if summary.classes:
                out.write(f"classes: {', '.join(summary.classes)}\n")

            if summary.functions:
                out.write(f"functions: {', '.join(summary.functions)}\n")

            if summary.imports:
                out.write("imports:\n")

                for imp in summary.imports:
                    if imp.startswith("riskquant"):
                        out.write(f"- {imp}\n")

            out.write("</file_summary>\n\n")

            content = subpath.read_text(encoding="utf-8").strip()

            if subpath.name == "__init__.py" and not content:
                continue  # skip empty __init__.py files

            out.write(f'<file path="{rel_path}">\n')
            out.write(f"```python\n{content}\n```\n")
            out.write("</file>\n\n")
        out.write("</files>\n")

    print(
        f"SUCCESS! Successfully generated {out_file.name} containing {len(valid_files)} files."
    )
    print("-" * 100 + "\n")

    print_success_message()


if __name__ == "__main__":
    # main(profile="all")
    # main(profile="config")
    # main(profile="config_static")
    # main(profile="config_proxy")
    # main(profile="valuation")
    # main(profile="simlab")
    # main(profile="simlab_reduced")
    # main(profile="simlab_extended")
    main(profile="ingest")
    main(profile="market_data_only")
    main(profile="portfolio_only")
    main(profile="credit")
    main(profile="credit_jtd")
    main(profile="credit_jtd_kernels")
    main(profile="credit_exposure")
    main(profile="basel")
    main(profile="sensitivity_pnl")
    # SOLUTIONS
    main(profile="solutions_credit_var")
    main(profile="solutions_credit_exposure")
    main(profile="solutions_sensitivity_pnl")
    # main(profile="ce_jtd")
    # main(profile="risk_quant")
    # main(profile="risk_quant_adhoc")
    # main(profile="pricers")
    # main(profile="market_data")
    # main(profile="market_data_reduced")
    # main(profile="market_data_calibrate")
    # main(profile="credit_jtd")
    # main(profile="credit_reduced")
    # main(profile="ingest_fixtures")
    # main(profile="ingest_fixtures_risk_static_config")
    # main(profile="ingest_fixtures_market_data")
    # main(profile="pricer_kernels")
    # main(profile="scenario_metrics")
