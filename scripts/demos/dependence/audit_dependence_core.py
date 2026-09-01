# scripts/demos/dependence/audit_dependence_core.py

"""
Mechanical audit of the dependence engine's computational core.

Scans executable source only — docstrings, comments and string literals are
stripped with ``tokenize`` first, so prose that *names* a banned construct while
explaining why it was removed does not trip the gate.

``riskquant.observations.preprocess`` is excluded by name. It is the edge-tier
policy module and it is supposed to contain imputation; the point of the
refactor was to make that the only place where imputation exists.

Exit status is non-zero on any violation, so the script can run in CI.

Run:
    python scripts/demos/dependence/audit_dependence_core.py
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from collections.abc import Mapping
from pathlib import Path

#: Packages that form the computational core.
CORE_PACKAGES: tuple[str, ...] = (
    "src/riskquant/dependence",
    "src/riskquant/observations",
    "src/riskquant/market_state",
)

#: Edge-tier modules deliberately exempt from the core rules.
EXEMPT: frozenset[str] = frozenset({"preprocess.py"})

#: Pattern -> human-readable rule. Applied to code with strings removed.
FORBIDDEN: Mapping[str, str] = {
    r"\bimport pandas\b|\bfrom pandas\b": "no Pandas in the computational core",
    r"\bmatplotlib\b|\bseaborn\b|\bPdfPages\b": "no plotting imports in the core",
    r"(?<![\w.])print\s*\(": "no print statements",
    r"\bopen\s*\(|\.savefig|\.to_csv|\.write_text": "no file output",
    r"\bfillna\b|\bKNNImputer\b|\bnan_to_num\b": "no hidden imputation",
    r"np\.nan(mean|std|median|max|min|quantile|sum)": "no NaN-skipping reductions",
    r"\bTSNE\b": "no dimensionality reduction in the core",
    r"type:\s*ignore": "no type: ignore",
    r"\bglobal\b\s+\w": "no global mutable state",
    r"datetime\.now|time\.time\(\)": "no dependence on the current time",
}


def _executable_source(path: Path) -> str:
    """Source text with comments and string literals blanked out."""
    pieces: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    for token_type, token_text, _, _, _ in tokens:
        if token_type in (tokenize.COMMENT, tokenize.STRING):
            pieces.append(" ")
        else:
            pieces.append(token_text)
            pieces.append(" ")
    return "".join(pieces)


def core_modules() -> list[Path]:
    """Every core module, excluding the edge-tier exemptions."""
    modules: list[Path] = []
    for package in CORE_PACKAGES:
        modules.extend(
            path
            for path in sorted(Path(package).rglob("*.py"))
            if path.name not in EXEMPT
        )
    return modules


def run() -> int:
    """Return the violation count and print a per-rule report."""
    modules = core_modules()
    sources = {path: _executable_source(path) for path in modules}

    violations = 0
    for pattern, rule in FORBIDDEN.items():
        offenders = [
            str(path) for path, source in sources.items() if re.search(pattern, source)
        ]
        violations += len(offenders)
        print(f"[{'PASS' if not offenders else 'FAIL'}] {rule}")
        for offender in offenders:
            print(f"          {offender}")

    path_headers = [
        path
        for path in modules
        if path.stat().st_size
        and not path.read_text().splitlines()[0].startswith("# src/")
    ]
    violations += len(path_headers)
    print(
        f"[{'PASS' if not path_headers else 'FAIL'}] first line is the repo-relative path"
    )

    initialisers = [
        path for package in CORE_PACKAGES for path in Path(package).rglob("__init__.py")
    ]
    non_empty = [path for path in initialisers if path.stat().st_size]
    violations += len(non_empty)
    print(
        f"[{'PASS' if not non_empty else 'FAIL'}] all {len(initialisers)} "
        "__init__.py files are empty"
    )

    print(f"\ncore modules scanned : {len(modules)}")
    print(f"violations           : {violations}")
    return violations


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
