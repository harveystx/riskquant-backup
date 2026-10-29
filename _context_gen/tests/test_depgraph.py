# _context_gen/tests/test_depgraph.py
from __future__ import annotations

import ast

from _context_gen.ast_model import ModuleInfo, build_module_info
from _context_gen.depgraph import build_graph

INTERNAL = frozenset({"riskquant"})


def _mod(dotted: str, repo_relative: str, src: str) -> ModuleInfo:
    tree = ast.parse(src)
    return build_module_info(
        repo_relative=repo_relative,
        module_dotted=dotted,
        text=src,
        tree=tree,
        internal_prefixes=INTERNAL,
    )


def test_from_package_import_submodule_prefers_submodule_target() -> None:
    # Regression test for the real bug found against the RiskQuant repo:
    # `from riskquant.observations import _validate` must resolve to the
    # real submodule `riskquant.observations._validate`, not the bare
    # (possibly-empty) `riskquant.observations` package.
    compile_mod = _mod(
        "riskquant.observations.compile",
        "src/riskquant/observations/compile.py",
        "from riskquant.observations import _validate\n",
    )
    validate_mod = _mod(
        "riskquant.observations._validate",
        "src/riskquant/observations/_validate.py",
        "x = 1\n",
    )
    graph = build_graph([compile_mod, validate_mod], texts={}, root_package="riskquant")
    assert graph.edges["riskquant.observations.compile"] == frozenset({
        "riskquant.observations._validate"
    })


def test_from_package_import_attribute_falls_back_to_package() -> None:
    # `SomeName` is not a real submodule file -- must fall back to the
    # package itself, not silently invent a nonexistent target.
    mod = _mod(
        "riskquant.a.b",
        "src/riskquant/a/b.py",
        "from riskquant.config import SomeClass\n",
    )
    config_pkg = _mod("riskquant.config", "src/riskquant/config/__init__.py", "")
    graph = build_graph([mod, config_pkg], texts={}, root_package="riskquant")
    assert graph.edges["riskquant.a.b"] == frozenset({"riskquant.config"})


def test_plain_import_of_package_resolved_as_is() -> None:
    mod = _mod("riskquant.a", "src/riskquant/a.py", "import riskquant.jit\n")
    jit_mod = _mod("riskquant.jit", "src/riskquant/jit.py", "")
    graph = build_graph([mod, jit_mod], texts={}, root_package="riskquant")
    assert graph.edges["riskquant.a"] == frozenset({"riskquant.jit"})


def test_importer_domains_counts_distinct_domains_not_raw_importers() -> None:
    # Direct regression test for the report-1/report-2 finding: raw
    # importer count is the wrong core-membership signal. Many importers
    # inside ONE domain must count as breadth 1, not breadth N.
    mods = [
        _mod(
            f"riskquant.config.mod{i}",
            f"src/riskquant/config/mod{i}.py",
            "from riskquant.jit import njit\n",
        )
        for i in range(5)
    ]
    jit_mod = _mod("riskquant.jit", "src/riskquant/jit.py", "")
    graph = build_graph([*mods, jit_mod], texts={}, root_package="riskquant")
    assert graph.importer_domains["riskquant.jit"] == {"config"}  # breadth 1, not 5
