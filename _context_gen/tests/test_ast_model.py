# _context_gen/tests/test_ast_model.py
from __future__ import annotations

import ast

from _context_gen.ast_model import (
    build_module_info,
    dotted_name_for,
    extract_symbols,
    is_disabled,
    resolve_imports,
)

INTERNAL = frozenset({"riskquant"})


def test_dotted_name_simple_module() -> None:
    assert dotted_name_for("src/riskquant/universe.py", "src/riskquant/", "riskquant") == "riskquant.universe"


def test_dotted_name_nested_module() -> None:
    got = dotted_name_for("src/riskquant/valuation/blocks/spot_block.py", "src/riskquant/", "riskquant")
    assert got == "riskquant.valuation.blocks.spot_block"


def test_dotted_name_init_py() -> None:
    got = dotted_name_for("src/riskquant/valuation/__init__.py", "src/riskquant/", "riskquant")
    assert got == "riskquant.valuation"


def test_dotted_name_outside_root_is_none() -> None:
    assert dotted_name_for("tests/valuation/test_x.py", "src/riskquant/", "riskquant") is None


def test_top_level_function_and_class() -> None:
    src = '''
"""Module doc."""

class SpotBlock:
    """A block."""

    def evaluate(self, x):
        return x


def price_spot(x, y):
    return x + y
'''
    tree = ast.parse(src)
    syms = extract_symbols(tree)
    kinds = {(s.kind, s.qualname) for s in syms}
    assert ("class", "SpotBlock") in kinds
    assert ("method", "SpotBlock.evaluate") in kinds
    assert ("function", "price_spot") in kinds


def test_nested_class_and_its_methods_are_captured() -> None:
    # This is the direct regression test for the report-1 finding: a
    # top-level-only AST walk misses methods and nested classes entirely.
    src = '''
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
'''
    tree = ast.parse(src)
    syms = extract_symbols(tree)
    quals = {s.qualname for s in syms}
    assert "Outer" in quals
    assert "Outer.Inner" in quals
    assert "Outer.Inner.inner_method" in quals
    assert "Outer.outer_method" in quals
    kinds = {s.qualname: s.kind for s in syms}
    assert kinds["Outer.Inner"] == "nested_class"
    assert kinds["Outer.Inner.inner_method"] == "method"


def test_symbols_are_source_order_not_alphabetical() -> None:
    src = '''
class Zebra:
    pass

class Apple:
    pass
'''
    tree = ast.parse(src)
    syms = extract_symbols(tree)
    assert [s.name for s in syms] == ["Zebra", "Apple"]


def test_meaningful_decorator_kept() -> None:
    src = '''
from riskquant.jit import njit

@njit
def kernel(x):
    return x
'''
    tree = ast.parse(src)
    syms = extract_symbols(tree)
    fn = next(s for s in syms if s.name == "kernel")
    assert any("njit" in d for d in fn.decorators)


def test_irrelevant_decorator_dropped() -> None:
    src = '''
def deco(f):
    return f

@deco
def kernel(x):
    return x
'''
    tree = ast.parse(src)
    syms = extract_symbols(tree)
    fn = next(s for s in syms if s.name == "kernel")
    assert fn.decorators == ()


def test_base_class_rendered() -> None:
    src = '''
class SpotBlock(Block):
    pass
'''
    tree = ast.parse(src)
    syms = extract_symbols(tree)
    cls = next(s for s in syms if s.name == "SpotBlock")
    assert cls.bases == ("Block",)


def test_long_signature_omitted_short_signature_kept() -> None:
    short_src = "def f(x: int) -> int:\n    return x\n"
    long_src = (
        "def f(aaaaaaaaaa: int, bbbbbbbbbb: int, cccccccccc: int, dddddddddd: int,"
        " eeeeeeeeee: int, ffffffffff: int) -> int:\n    return 0\n"
    )
    short_syms = extract_symbols(ast.parse(short_src))
    long_syms = extract_symbols(ast.parse(long_src))
    assert short_syms[0].signature is not None and "->" in short_syms[0].signature
    assert long_syms[0].signature is None


def test_relative_import_single_level_resolved() -> None:
    # This is the direct regression test for the confirmed report-1 bug:
    # relative imports were entirely invisible because ast.ImportFrom.module
    # has no package prefix on its own.
    src = "from .permutations import _resolve_data_order\n"
    tree = ast.parse(src)
    edges = resolve_imports(tree, "riskquant.dependence_stress.benchmarks", INTERNAL)
    targets = {e.target for e in edges}
    assert "riskquant.dependence_stress.permutations" in targets


def test_relative_import_bare_dot_import() -> None:
    src = "from . import validation as val\n"
    tree = ast.parse(src)
    edges = resolve_imports(tree, "riskquant.dependence_stress.engine", INTERNAL)
    targets = {e.target for e in edges}
    assert "riskquant.dependence_stress.validation" in targets


def test_relative_import_two_levels_up() -> None:
    # `..common` resolves at module granularity: `shared_util` is a name
    # pulled from the `common` module, not necessarily a submodule of it.
    src = "from ..common import shared_util\n"
    tree = ast.parse(src)
    edges = resolve_imports(tree, "riskquant.a.b.c", INTERNAL)
    targets = {e.target for e in edges}
    assert "riskquant.a.common" in targets


def test_absolute_internal_import_resolved() -> None:
    src = "from riskquant.jit import njit\n"
    tree = ast.parse(src)
    edges = resolve_imports(tree, "riskquant.valuation.engine", INTERNAL)
    targets = {(e.target, e.is_internal) for e in edges}
    assert ("riskquant.jit", True) in targets


def test_external_import_kept_but_marked_external() -> None:
    src = "import numpy as np\nimport numba\n"
    tree = ast.parse(src)
    edges = resolve_imports(tree, "riskquant.valuation.engine", INTERNAL)
    ext = {e.target: e.is_internal for e in edges}
    assert ext["numpy"] is False
    assert ext["numba"] is False


def test_docstring_never_truncated() -> None:
    # Direct regression test for the confirmed report-1 bug: the previous
    # generator sliced docstrings to 120 chars and always appended "...".
    long_doc = "A" * 500
    src = f'"""{long_doc}"""\n'
    tree = ast.parse(src)
    info = build_module_info(
        repo_relative="src/riskquant/x.py",
        module_dotted="riskquant.x",
        text=src,
        tree=tree,
        internal_prefixes=INTERNAL,
    )
    assert info.docstring == long_doc
    assert len(info.docstring) == 500  # exact, untouched, no [:120] slice anywhere


def test_disabled_file_flagged_not_dropped() -> None:
    src = "\n".join(f"# disabled line {i}" for i in range(10)) + "\n"
    tree = ast.parse(src)
    assert is_disabled(src, tree) is True
    info = build_module_info(
        repo_relative="src/riskquant_lab/plotting/colors.py",
        module_dotted="riskquant_lab.plotting.colors",
        text=src,
        tree=tree,
        internal_prefixes=INTERNAL,
    )
    assert info.is_disabled is True
    assert info.symbols == ()  # correctly empty, but the module info itself still exists


def test_trivial_empty_file_not_flagged_disabled() -> None:
    src = "\n"
    tree = ast.parse(src)
    assert is_disabled(src, tree) is False
