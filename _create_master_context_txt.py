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
                "riskquant_all",
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
        "riskquant_all": [
            # CODE
            rq_src_path,
            rq_lab_src_path,
            # SCRIPTS
            rq_project_directory / "scripts",
            # TESTS
            rq_project_directory / "tests",
            # OTHER
            rq_project_directory / "pyproject.toml",
        ],
        #######
        # NEW #
        #######
        "riskquant_config": [
            "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/config",
            "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/config",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/_arrays.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/asset_classes.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/container_topology.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/discount_basis.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/quote_types.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/roll_methods.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/x_axis_kinds.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/y_axis_kinds.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/namespace.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/proxy_config/compile.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/proxy_config/loader.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/proxy_config/proxy_topology.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/proxy_config/schema_objects.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/proxy_config/validate.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/compile.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/loader.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/schema_objects.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/static_config.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/validate.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/constants.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/exceptions.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/curves/extrap_methods.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/curves/interp_methods.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/risk_factors/shift_types.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/utils/printer.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/utils/tenors.py",
            "C:/Users/mharvey/Documents/code/riskquant/tests/config",
        ],
        "riskquant_dependence": [
            "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/dependence",
            "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/dependence",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/constants.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence/_kernels/_kendall_tau_b.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence/_kernels/_symmetry.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence/_matrix.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence/contracts.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence/estimators/pearson.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence/taxonomy.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/jit.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/observations/_kernels.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/observations/_validate.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/observations/arrays.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/observations/compile.py",
            "C:/Users/mharvey/Documents/code/riskquant/tests/dependence",
        ],
        "riskquant_dependence_stress": [
            # "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/dependence_stress",
            # "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/dependence_stress",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/dependence_stress",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/solutions/dependence_stress",
            # "C:/Users/mharvey/Documents/code/riskquant/tests/dependence_stress",
        ],
        "riskquant_sensitivities": [
            # "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/sensitivities",
            # "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/sensitivities",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/constants.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/jit.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/state.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/portfolio/compiled.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/sensitivities/result.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/sensitivities/sensitizer_kernels/spot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/sensitivities/sensitizer_registry.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/sensitivities/sensitizers/spot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/solutions/sensitivity_pnl/api.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/universe.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/_base_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/block_type.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/spot_block.py",
            # "C:/Users/mharvey/Documents/code/riskquant/tests/sensitivities",
        ],
        "riskquant_config_static_config": [
            # "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/config/static_config",
            # "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/config/static_config",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/_arrays.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/asset_classes.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/container_topology.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/discount_basis.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/quote_types.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/roll_methods.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/x_axis_kinds.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/enums/y_axis_kinds.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/namespace.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/compile.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/loader.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/schema_objects.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/static_config.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/validate.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/constants.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/exceptions.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/curves/extrap_methods.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/curves/interp_methods.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/risk_factors/shift_types.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/utils/printer.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/utils/tenors.py",
            # "C:/Users/mharvey/Documents/code/riskquant/tests/config/static_config",
        ],
        "riskquant_valuation": [
            "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/valuation",
            "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/valuation",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/_arrays.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/constants.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/jit.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/state.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/surfaces/y_axes_types.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/portfolio/compiled.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/_base_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/block_type.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/discounted_cashflow_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/spot_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/vanilla_european_option_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/vanilla_forward_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/fixed_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/floating_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/principal_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/spot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/vanilla_european_option.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/vanilla_forward.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_registry.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/fixed_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/floating_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/principal_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/spot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/vanilla_european_option.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/vanilla_forward.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricing_models/black76/kernels.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricing_models/black76/model.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricing_models/math.py",
            "C:/Users/mharvey/Documents/code/riskquant/tests/valuation",
        ],
        "riskquant_valuation_light": [
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/_base_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/block_type.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/discounted_cashflow_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/spot_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/vanilla_european_option_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/blocks/vanilla_forward_block.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/fixed_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/floating_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/principal_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/spot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/vanilla_european_option.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_kernels/vanilla_forward.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricer_registry.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/fixed_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/floating_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/principal_cashflow.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/spot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/vanilla_european_option.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricers/vanilla_forward.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricing_models/black76/kernels.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricing_models/black76/model.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/pricing_models/math.py",
        ],
        "riskquant_simlab": [
            "C:/Users/mharvey/Documents/code/riskquant/scripts/demos/simlab",
            "C:/Users/mharvey/Documents/code/riskquant/scripts/solutions/simlab",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/_arrays.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/proxy_config/proxy_topology.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/config/static_config/static_config.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/constants.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/jit.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/assemble.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/risk_factors/matrix.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/risk_factors/shift_types.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/market_data/risk_factors/snapshot.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/pnl/results.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/portfolio/compiled.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/proxy_resolver/proxy.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/builder/shock_universe.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/scenarios/base.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/scenarios/core_math/ar1_filter.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/scenarios/core_math/ewma_volatility.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/scenarios/core_math/sbb_block_generator.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/shock_plan/plan.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/shocks/apply.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/simlab/shocks/shock_kernel.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/universe.py",
            "C:/Users/mharvey/Documents/code/riskquant/src/riskquant/valuation/engine.py",
            "C:/Users/mharvey/Documents/code/riskquant/tests/simlab",
        ],
        #######
        # OLD #
        #######
        "github": [
            rq_project_directory / ".github",
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
        ".deprecated",
        "_deprecated",
    }

    # 3. Gather all valid files
    valid_files: list[Path] = []

    for path in target_paths:
        path = Path(path)
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
                and not subpath.parent.name.startswith(".deprecated")
                and "_deprecated" not in subpath.parts
                and ".deprecated" not in subpath.parts
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
            "- Intentionally contiguous at the core\n"
            "- Strings must die early\n"
            "- Memory locality matters\n"
            "- CPU efficiency matters\n"
            "- Memory efficiency matters\n"
            "- Correctness over cleverness\n"
            "- Fail fast on invalid inputs at boundaries/ingest\n"
            "- Kernels do not contain policy - they are purely computational\n"
            "- Compilers are responsible for policy and orchestration; they must fail fast on invalid inputs\n"
            "- Avoid unnecessary branching in hot paths\n"
            "- Skip '__init__.py' files (they are always empty)\n"
            "- Polars over Pandas for DataFrames\n"
            "- `riskquant.jit` is the single source of truth for Numba policy and types\n\n"
            "Numba rules - NON-NEGOTIABLE:\n"
            "- ALWAYS use canonical decorators from `riskquant.jit`: `njit`, `parallel_njit`, `stable_math_parallel_njit`, `inline_njit`\n"
            "- NEVER use raw `@nb.njit` or define local Numba compiler configurations\n"
            "- ALWAYS use canonical `_NB_*` scalar types in signatures (`_NB_VOID`, `_NB_FLOAT64`, `_NB_FLOAT32`, `_NB_INT64`, `_NB_INT32`, `_NB_INT8`, `_NB_BOOL`)\n"
            "- NEVER use raw `nb.void`, `nb.float64`, `nb.int32`, etc. when a canonical alias exists\n"
            "- ALWAYS use canonical array aliases (`_F64_*`, `_F32_*`, `_I64_*`, `_I32_*`, `_I8_*`, `_BOOL_*`) with the correct `_C`, `_F`, and `_RO` contract\n"
            "- NEVER replace canonical aliases with ad-hoc `nb_types.Array` or generic strided signatures\n"
            "- Read-only inputs MUST use `_RO`; preserve C/F contiguity\n"
            "- If a required Numba type or configuration is missing, extend `riskquant.jit`; do NOT create a local workaround\n"
            "- Raw `numba` usage is allowed only where `riskquant.jit` has no equivalent abstraction (e.g. `nb.prange`)\n"
            "- Existing code is not precedent; do not propagate violations of these rules\n\n"
            "When proposing code:\n"
            "- Preserve existing SoA layouts\n"
            "- Avoid object-heavy designs in hot paths\n"
            "- Avoid polars in performance-critical code\n"
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
    main(profile="riskquant_all")
    main(profile="riskquant_config")
    main(profile="riskquant_config_static_config")
    main(profile="riskquant_sensitivities")
    main(profile="riskquant_simlab")
    main(profile="riskquant_valuation")
    main(profile="riskquant_valuation_light")
    main(profile="riskquant_dependence")
    main(profile="riskquant_dependence_stress")
    # main(profile="scenario_metrics")
    # main(profile="sensitivity_pnl")
    # main(profile="solutions_sensitivity_pnl") 
    # main(profile="github")
