# scripts/tools/demo_import_smoke.py
"""
Demo import smoke (WS_CREDIT_POLARS C.19, O-8).

Execute only the top-level import statements of each demo file and report
each import that fails. The demo body does not run: the tool reads each file
with ``ast`` and executes each ``import`` and ``from ... import`` node of the
module body alone. Imports inside functions, ``if`` blocks and ``try`` blocks
do not run.

Usage
-----
    python scripts/tools/demo_import_smoke.py [path ...]

Without a path, the tool checks every ``scripts/solutions/credit_var/
demo_e2e_*.py`` file under the repository root. Exit status: 0 when every
import resolves; 1 when an import fails; 2 when a path is not a ``.py`` file
or when no file is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR = _REPO_ROOT / "scripts" / "solutions" / "credit_var"
_DEFAULT_GLOB = "demo_e2e_*.py"


def _import_failures(path: Path) -> list[str]:
    """
    Return one line per failed top-level import of ``path``.

    The demo directory is first on ``sys.path`` while its imports run, as it
    is when Python runs the demo as a script, so sibling helper modules
    resolve. All imports of one file share one namespace.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    namespace: dict[str, object] = {"__name__": "__demo_import_smoke__"}
    failures: list[str] = []
    sys.path.insert(0, str(path.parent))
    try:
        for node in tree.body:
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
            try:
                exec(code, namespace)
            except Exception as exc:  # every failure is a finding; report all
                failures.append(
                    f"{path}: line {node.lineno}: {ast.unparse(node)} -> "
                    f"{type(exc).__name__}: {exc}"
                )
    finally:
        sys.path.remove(str(path.parent))
    return failures


def main(argv: list[str]) -> int:
    """Check the demo files of ``argv`` (or the default set) and print a report."""
    if argv:
        paths = [Path(arg) for arg in argv]
        bad_paths = [p for p in paths if not (p.is_file() and p.suffix == ".py")]
        if bad_paths:
            print("not a .py file: " + ", ".join(str(p) for p in bad_paths))
            return 2
    else:
        paths = sorted(_DEFAULT_DIR.glob(_DEFAULT_GLOB))
        if not paths:
            print(f"no file matches {_DEFAULT_DIR / _DEFAULT_GLOB}")
            return 2

    failures = [line for path in paths for line in _import_failures(path)]
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} failed import(s) in {len(paths)} file(s)")
        return 1
    print(f"all demo imports resolve ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
