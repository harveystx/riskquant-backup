# _context_gen/ast_model.py
"""
Mechanical AST extraction.

Everything here is derived by syntax alone. No inferred semantics, no
"what this function does" descriptions, no call-graph inference — that is
explicitly out of scope (see the architecture contract, §17/§20/§33 of the
governing audits). If a fact is not directly readable from the AST, it does
not appear in a ModuleInfo.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

MAX_INLINE_SIGNATURE_CHARS = 100

# Decorators kept in the navigation layer because they carry contract
# meaning in this codebase (Numba compilation policy, frozen/slots/kw_only
# dataclass contracts, dispatch/visibility contracts). This is a reviewable
# allowlist, not a semantic guess — extend it explicitly, never implicitly.
MEANINGFUL_DECORATOR_NAMES: frozenset[str] = frozenset(
    {
        "njit",
        "parallel_njit",
        "stable_math_parallel_njit",
        "inline_njit",
        "dataclass",
        "unique",
        "abstractmethod",
        "staticmethod",
        "classmethod",
        "property",
    }
)

# A module is flagged DISABLED/COMMENT-ONLY (never silently dropped, §26) when
# its AST body is empty AND it has a non-trivial amount of raw text — i.e. it
# looks like source that was deliberately commented out wholesale, not a
# genuinely trivial file.
DISABLED_MIN_LINES = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class ImportEdge:
    target: str
    is_internal: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RawImport:
    """
    A syntactic import statement, resolved only as far as pure syntax can
    take it: relative dots are turned into an absolute `base`, but whether
    each imported `name` is itself a submodule (`from pkg import
    submodule`) or an attribute defined inside `base` (`from pkg import
    SomeClass`) cannot be decided without knowing which files actually
    exist — that decision belongs to depgraph.py, which has the full
    module registry. This module never guesses it.
    """

    base: str  # fully resolved dotted base, e.g. "riskquant.observations"
    names: tuple[str, ...]  # imported names under `base`; () for a plain `import base`


@dataclass(frozen=True, slots=True, kw_only=True)
class SymbolInfo:
    kind: str  # "class" | "function" | "method" | "nested_class"
    name: str
    qualname: str
    line: int
    end_line: int | None
    bases: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    signature: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModuleInfo:
    dotted: str
    repo_relative: str
    docstring: str | None
    symbols: tuple[SymbolInfo, ...] = field(default_factory=tuple)
    imports: tuple[ImportEdge, ...] = field(default_factory=tuple)
    raw_imports: tuple[RawImport, ...] = field(default_factory=tuple)
    is_disabled: bool = False


def dotted_name_for(repo_relative: str, source_root_prefix: str, package_name: str) -> str | None:
    """
    Convert a repo-relative POSIX path under `source_root_prefix`
    (e.g. "src/riskquant/") into a dotted module name rooted at
    `package_name` (e.g. "riskquant").

    Returns None if `repo_relative` is not under `source_root_prefix`.
    """
    if not repo_relative.startswith(source_root_prefix):
        return None
    rest = repo_relative[len(source_root_prefix) :]
    if not rest.endswith(".py"):
        return None
    rest = rest[: -len(".py")]
    if rest.endswith("/__init__"):
        rest = rest[: -len("/__init__")]
    if rest == "__init__" or rest == "":
        return package_name
    return package_name + "." + rest.replace("/", ".")


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _rendered_decorators(nodes: list[ast.expr]) -> tuple[str, ...]:
    kept: list[str] = []
    for n in nodes:
        if _decorator_name(n) in MEANINGFUL_DECORATOR_NAMES:
            try:
                kept.append("@" + ast.unparse(n))
            except Exception:
                kept.append("@" + _decorator_name(n))
    return tuple(kept)


def _rendered_bases(bases: list[ast.expr]) -> tuple[str, ...]:
    out: list[str] = []
    for b in bases:
        try:
            out.append(ast.unparse(b))
        except Exception:
            continue
    return tuple(out)


def _rendered_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    try:
        args = ast.unparse(node.args)
    except Exception:
        return None
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    rendered = f"{prefix}{node.name}({args})"
    if node.returns is not None:
        try:
            rendered += f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    if len(rendered) > MAX_INLINE_SIGNATURE_CHARS:
        return None
    return rendered


def _walk_class(node: ast.ClassDef, qual_prefix: str, out: list[SymbolInfo]) -> None:
    qualname = f"{qual_prefix}{node.name}"
    out.append(
        SymbolInfo(
            kind="class" if not qual_prefix else "nested_class",
            name=node.name,
            qualname=qualname,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", None),
            bases=_rendered_bases(node.bases),
            decorators=_rendered_decorators(node.decorator_list),
        )
    )
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                SymbolInfo(
                    kind="method",
                    name=child.name,
                    qualname=f"{qualname}.{child.name}",
                    line=child.lineno,
                    end_line=getattr(child, "end_lineno", None),
                    decorators=_rendered_decorators(child.decorator_list),
                    signature=_rendered_signature(child),
                )
            )
        elif isinstance(child, ast.ClassDef):
            _walk_class(child, qual_prefix=f"{qualname}.", out=out)


def extract_symbols(tree: ast.Module) -> tuple[SymbolInfo, ...]:
    """
    Depth-unlimited, mechanical: top-level classes/functions, their methods,
    and nested classes. This is the direct fix for the report-1 finding that
    a top-level-only walk misses 24.8% of all callables in this codebase.
    """
    out: list[SymbolInfo] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _walk_class(node, qual_prefix="", out=out)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                SymbolInfo(
                    kind="function",
                    name=node.name,
                    qualname=node.name,
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", None),
                    decorators=_rendered_decorators(node.decorator_list),
                    signature=_rendered_signature(node),
                )
            )
    out.sort(key=lambda s: s.line)  # source-definition order, not alphabetical (§9 of the contract)
    return tuple(out)


def resolve_imports(tree: ast.Module, module_dotted: str, internal_prefixes: frozenset[str]) -> tuple[ImportEdge, ...]:
    """
    Resolve every import statement to an absolute dotted target.

    Relative imports (`from .foo import x`, `from . import x`, `from ..a
    import y`) are resolved using `ast.ImportFrom.level` against the
    *containing package* of `module_dotted` — this is the direct fix for
    the report-1 finding that relative imports were invisible because
    `ast.ImportFrom.module` has no package prefix on its own.

    `internal_prefixes` (e.g. {"riskquant", "riskquant_lab"}) decides
    `is_internal`; anything else is marked external and kept (never
    silently dropped) but never treated as a production-dependency-closure
    candidate.
    """
    edges: list[ImportEdge] = []
    pkg_parts = module_dotted.split(".")
    seen: set[str] = set()

    def add(target: str) -> None:
        if target in seen:
            return
        seen.add(target)
        is_internal = any(target == p or target.startswith(p + ".") for p in internal_prefixes)
        edges.append(ImportEdge(target=target, is_internal=is_internal))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = pkg_parts[:-1]  # containing package of this module
                climb = node.level - 1
                if climb > 0:
                    base = base[:-climb] if climb <= len(base) else []
                if node.module:
                    # `from ..common import shared_util` -> the module being
                    # imported FROM is riskquant.a.common; `shared_util` may
                    # be a class/function/constant inside it, not necessarily
                    # its own module, so the edge stops at module granularity.
                    full = ".".join([*base, *node.module.split(".")])
                    if full:
                        add(full)
                else:
                    # `from . import validation as val` -- with no further
                    # module component, each imported name conventionally
                    # names a submodule of `base` (the common Python-package
                    # idiom this codebase uses, e.g. `from . import
                    # validation`), so each alias is its own edge.
                    for alias in node.names:
                        full = ".".join([*base, alias.name])
                        if full:
                            add(full)
            elif node.module:
                add(node.module)

    edges.sort(key=lambda e: e.target)
    return tuple(edges)


def extract_raw_imports(tree: ast.Module, module_dotted: str) -> tuple[RawImport, ...]:
    """
    Syntactic extraction only: relative dots resolved to an absolute base,
    imported names kept alongside it, no decision made about whether a
    name is a submodule or an attribute. depgraph.py performs that
    resolution once it has the full module registry to check against.
    """
    raw: list[RawImport] = []
    pkg_parts = module_dotted.split(".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw.append(RawImport(base=alias.name, names=()))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base_parts = pkg_parts[:-1]
                climb = node.level - 1
                if climb > 0:
                    base_parts = base_parts[:-climb] if climb <= len(base_parts) else []
                base = ".".join([*base_parts, *node.module.split(".")]) if node.module else ".".join(base_parts)
            else:
                base = node.module or ""
            if not base:
                continue
            names = tuple(alias.name for alias in node.names if alias.name != "*")
            raw.append(RawImport(base=base, names=names))

    return tuple(raw)


def is_disabled(text: str, tree: ast.Module | None) -> bool:
    """
    Mechanical DISABLED/COMMENT-ONLY flag: the AST body is empty (no
    executable statements, no docstring-only module counts as empty too
    unless it also has a docstring) and the raw text is non-trivial. Never
    used to exclude a file (§26) — only to annotate it.
    """
    if tree is None:
        return False
    if tree.body:
        return False
    non_blank = [ln for ln in text.splitlines() if ln.strip()]
    return len(non_blank) >= DISABLED_MIN_LINES


def build_module_info(
    *,
    repo_relative: str,
    module_dotted: str,
    text: str,
    tree: ast.Module,
    internal_prefixes: frozenset[str],
) -> ModuleInfo:
    return ModuleInfo(
        dotted=module_dotted,
        repo_relative=repo_relative,
        docstring=ast.get_docstring(tree),
        symbols=extract_symbols(tree),
        imports=resolve_imports(tree, module_dotted, internal_prefixes),
        raw_imports=extract_raw_imports(tree, module_dotted),
        is_disabled=is_disabled(text, tree),
    )
