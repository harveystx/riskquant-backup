# riskquant/_get_imported_packages.py
"""
Scan a directory or Python file and report all imported modules.

Usage:
    python _get_imported_packages.py <relative/path/inside/project>
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path

# --- Configuration -----------------------------------------------------------

EXCLUDE = {
    "pandas",
    "numpy",
    "numba",
    "requests",
    "matplotlib",
    "scipy",
    "sklearn",
    "polars",
    "json",
    "os",
    "sys",
    "seaborn",
}

EXCLUDE_STDLIB = True

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    "node_modules",
    "build",
    "dist",
}

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

SEARCH_ROOTS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
]


# --- Core --------------------------------------------------------------------


def top_level(name: str) -> str:
    return name.split(".", 1)[0]


def is_excluded(name: str) -> bool:
    tl = top_level(name)
    if tl in EXCLUDE:
        return True
    if EXCLUDE_STDLIB and tl in sys.stdlib_module_names:
        return True
    return False


def imports_in_file(path: Path) -> set[str]:
    """Return the full dotted module names imported by a file."""
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8", errors="replace"), filename=str(path)
        )
    except SyntaxError as e:
        print(f"[warn] skipping {path}: {e}", file=sys.stderr)
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                found.add(node.module)
    return found


def scan_path(target: Path) -> dict[str, set[Path]]:
    """
    Inspect a single .py file or walk a directory tree,
    returning {full_module_name: {set of files that import it}}.
    """
    results: dict[str, set[Path]] = defaultdict(set)

    if target.is_file():
        if target.suffix == ".py":
            for mod in imports_in_file(target):
                if not is_excluded(mod):
                    results[mod].add(target)
        else:
            print(f"[warn] target is not a python file: {target}", file=sys.stderr)
        return results

    for py_file in target.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        for mod in imports_in_file(py_file):
            if is_excluded(mod):
                continue
            results[mod].add(py_file)
    return results


def resolve_module(module: str) -> Path | None:
    """Resolve a dotted module name to a .py file or package dir."""
    parts = module.split(".")
    for base in SEARCH_ROOTS:
        f = base.joinpath(*parts).with_suffix(".py")
        if f.is_file():
            return f
        pkg = base.joinpath(*parts)
        if (pkg / "__init__.py").is_file():
            return pkg
    return None


def fmt_path(p: Path) -> str:
    """Return the complete, absolute filesystem path with normalized forward slashes."""
    return p.resolve().as_posix()


# --- CLI ---------------------------------------------------------------------


def main(
    target_path: str | Path | None = None,
    *extra_paths: str | Path,
    context_name: str | None = None,
    indent: int = 12,
):
    resolved: set[str] = set()
    unresolved: list[str] = []

    # 1. Resolve and scan file/directory target
    if target_path is not None:
        p = Path(target_path).expanduser()
        target = p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if not target.exists():
            print(f"error: {target} does not exist", file=sys.stderr)
            sys.exit(1)
        results = scan_path(target)
    elif not extra_paths:
        results = scan_path(PROJECT_ROOT)
    else:
        results = {}

    for mod in sorted(results):
        p = resolve_module(mod)
        if p is None:
            unresolved.append(mod)
        else:
            resolved.add(fmt_path(p))

    # 2. Append any explicitly passed extra paths
    for extra in extra_paths:
        p = Path(extra).expanduser()
        ep = p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if not ep.exists():
            print(f"[warn] extra path not found: {extra}", file=sys.stderr)
        resolved.add(fmt_path(ep))

    if unresolved:
        for u in unresolved:
            print(f"[warn] unresolved: {u}", file=sys.stderr)

    key = context_name or (Path(target_path).stem if target_path else "root")
    spaces = " " * indent

    print(f'"{key}": [')
    for file_path in sorted(resolved):
        print(f'{spaces}"{file_path}",')
    print("],")


if __name__ == "__main__":
    for sub_dir in [
        "basel",
        "config",
        "config/static_config",
        "credit",
        "dependence",
        "dependence_stress",
        "ingest_fixtures",
        "market_data",
        "market_state",
        "observations",
        "reporting",
        # .....
        "scenario_metrics",
        "sensitivities",
        "sensitivity_pnl",
        "simlab",
        "solutions",
        "utils",
        "valuation",
    ]:
        print(f"Processing context: {sub_dir}")
        main(
            f"src/riskquant/{sub_dir}",
            f"scripts/demos/{sub_dir}",
            f"scripts/solutions/{sub_dir}",
            f"tests/{sub_dir}",
            context_name=f"riskquant_{sub_dir}".replace("/", "_"),
        )
