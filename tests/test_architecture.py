# tests/test_architecture.py

"""
Mechanical architecture gate.

Enforces the package boundary, the public surface, the allocation contracts and
the phase scope. These are the rules that documentation cannot enforce and that
a reviewer cannot reliably check by eye.
"""

from __future__ import annotations

import ast
import importlib
import io
import tokenize
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "riskquant"


def _modules(package: str) -> Iterator[Path]:
    yield from sorted((SOURCE_ROOT / package).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    """Absolute module names imported by a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


def _executable_source(path: Path) -> str:
    """Source with comments and string literals blanked, so prose cannot trip a gate."""
    pieces: list[str] = []
    for token_type, token_text, _, _, _ in tokenize.generate_tokens(
        io.StringIO(path.read_text(encoding="utf-8")).readline
    ):
        pieces.append(
            " " if token_type in (tokenize.COMMENT, tokenize.STRING) else token_text
        )
        pieces.append(" ")
    return "".join(pieces)


# -------------------------------------------------------------- import graph


def test_observations_imports_neither_dependence_nor_dependence_stress() -> None:
    for path in _modules("observations"):
        for name in _imported_modules(path):
            assert not name.startswith("riskquant.dependence"), (
                f"{path.name} imports {name}; observations must sit upstream of both engines."
            )


def test_dependence_does_not_import_dependence_stress() -> None:
    for path in _modules("dependence"):
        for name in _imported_modules(path):
            assert not name.startswith("riskquant.dependence_stress"), (
                f"{path.name} imports {name}; the two engines must not know about each other."
            )


def test_dependence_imports_observations() -> None:
    """The seam exists and is used, so the graph is not vacuously satisfied."""
    imported: set[str] = set()
    for path in _modules("dependence"):
        imported |= _imported_modules(path)
    assert any(name.startswith("riskquant.observations") for name in imported)


def test_market_state_does_not_import_dependence() -> None:
    for path in _modules("market_state"):
        for name in _imported_modules(path):
            assert not name.startswith("riskquant.dependence")


def test_every_init_file_is_empty() -> None:
    initialisers = list(SOURCE_ROOT.rglob("__init__.py"))
    assert initialisers
    for path in initialisers:
        assert path.stat().st_size == 0, f"{path} must be empty"


def test_every_module_declares_its_repository_relative_path() -> None:
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        expected = f"# src/riskquant/{path.relative_to(SOURCE_ROOT).as_posix()}"
        assert first_line == expected, f"{path}: got {first_line!r}"


# ---------------------------------------------------------- factor regression
# C.21 and C.22 of WS_FACTOR_REGRESSION. The core is private; only its facade
# and tests may import it, and it may import only the allow-list below.

REPOSITORY_ROOT = SOURCE_ROOT.parents[1]
FACTOR_REGRESSION_CORE = SOURCE_ROOT / "factor_regression"
FACTOR_REGRESSION_FACADE = SOURCE_ROOT / "solutions" / "factor_regression"
_CORE_ALLOWED_IMPORT_PREFIXES = (
    "numpy",
    "scipy.stats",
    "riskquant.exceptions",
    "riskquant.factor_regression",
)
_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "ReprEnum"})


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import_statements(path: Path, root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """
    Every import statement of a file as ``(module, imported names)``.

    ``TYPE_CHECKING`` blocks are included (``ast.walk`` visits every node). A
    relative import is resolved against the module name of ``path`` below
    ``root``; one that climbs above ``root`` fails. ``import a.b`` gives
    ``("a.b", ())``; ``from a import b, c`` gives ``("a", ("b", "c"))``.
    """
    module = _module_name(path, root)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    statements: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Import):
            statements.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                anchor = package.split(".") if package else []
                climb = node.level - 1
                assert climb <= len(anchor), f"{path}: relative import above root"
                kept = anchor[: len(anchor) - climb]
                base = ".".join([*kept, *([node.module] if node.module else [])])
            assert base, f"{path}: unresolved relative import"
            names = tuple(alias.name for alias in node.names if alias.name != "*")
            statements.append((base, names))
    return statements


def _import_targets(path: Path, root: Path) -> set[str]:
    """Absolute names a file imports; ``from a import b`` yields ``a`` and ``a.b``."""
    targets: set[str] = set()
    for base, names in _import_statements(path, root):
        targets.add(base)
        targets.update(f"{base}.{name}" for name in names)
    return targets


def _core_may_import(name: str) -> bool:
    import sys

    return name.partition(".")[0] in sys.stdlib_module_names or any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in _CORE_ALLOWED_IMPORT_PREFIXES
    )


def _is_core_target(name: str) -> bool:
    return name == "riskquant.factor_regression" or name.startswith(
        "riskquant.factor_regression."
    )


def test_factor_regression_import_direction() -> None:
    src_root = SOURCE_ROOT.parent

    core_files = sorted(FACTOR_REGRESSION_CORE.rglob("*.py"))
    assert core_files, "the guard scanned no core module"
    for path in core_files:
        for base, names in _import_statements(path, src_root):
            # ``from scipy import stats`` passes through the allowed child
            # ``scipy.stats``; ``import scipy`` and ``from scipy import linalg``
            # fail, because neither the base nor every child is allowed.
            allowed = _core_may_import(base) or (
                bool(names) and all(_core_may_import(f"{base}.{n}") for n in names)
            )
            assert allowed, f"{path}: the factor regression core imports {base}"

    # Both packages: every relative import resolves inside src (asserted in
    # _import_statements) and the core never imports the facade.
    for path in sorted(FACTOR_REGRESSION_FACADE.rglob("*.py")):
        _import_targets(path, src_root)
    for path in core_files:
        for name in _import_targets(path, src_root):
            assert not name.startswith("riskquant.solutions"), (
                f"{path}: the core imports the facade tier ({name})"
            )

    outside = [
        (path, src_root)
        for path in sorted(src_root.rglob("*.py"))
        if not path.is_relative_to(FACTOR_REGRESSION_CORE)
        and not path.is_relative_to(FACTOR_REGRESSION_FACADE)
    ] + [
        (path, REPOSITORY_ROOT)
        for path in sorted((REPOSITORY_ROOT / "scripts").rglob("*.py"))
    ]
    assert outside, "the guard scanned no module outside the two packages"
    for path, root in outside:
        for name in _import_targets(path, root):
            assert not _is_core_target(name), (
                f"{path}: imports {name}; only the facade may import the core"
            )


def _annotation_mentions_str(node: ast.expr | None) -> bool:
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == "str":
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "str":
            return True
        # A string annotation is parsed and scanned like code.
        if (
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and _annotation_mentions_str(ast.parse(sub.value, mode="eval").body)
        ):
            return True
    return False


def test_factor_regression_core_holds_no_str_type() -> None:
    core_files = sorted(FACTOR_REGRESSION_CORE.rglob("*.py"))
    assert core_files, "the guard scanned no core module"
    n_annotations = 0
    for path in core_files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            annotations: list[ast.expr | None] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                every_arg = [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *([arguments.vararg] if arguments.vararg else []),
                    *([arguments.kwarg] if arguments.kwarg else []),
                ]
                annotations.extend(arg.annotation for arg in every_arg)
                annotations.append(node.returns)
            elif isinstance(node, ast.AnnAssign):
                annotations.append(node.annotation)
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = (
                        base.id
                        if isinstance(base, ast.Name)
                        else base.attr
                        if isinstance(base, ast.Attribute)
                        else ""
                    )
                    assert base_name not in _ENUM_BASES, (
                        f"{path}: class {node.name} derives from {base_name}"
                    )
            for annotation in annotations:
                if annotation is None:
                    continue
                n_annotations += 1
                assert not _annotation_mentions_str(annotation), (
                    f"{path}:{annotation.lineno}: str in a core annotation"
                )
    assert n_annotations > 0, "the guard scanned no annotation"


# -------------------------------------------------------------- public surface


PUBLIC_OBSERVATIONS = [
    ("riskquant.observations.orientation", "ObservationOrientation"),
    ("riskquant.observations.compile", "CompiledSample"),
    ("riskquant.observations.compile", "RankRepresentation"),
    ("riskquant.observations.compile", "compile_sample"),
    ("riskquant.observations.compile", "compile_ranks"),
]

PUBLIC_DEPENDENCE = [
    ("riskquant.dependence.taxonomy", "DependenceEstimator"),
    ("riskquant.dependence.taxonomy", "PsdStatus"),
    ("riskquant.dependence.taxonomy", "Invariance"),
    ("riskquant.dependence.taxonomy", "ESTIMATOR_CONTRACT"),
    ("riskquant.dependence.taxonomy", "DependenceContract"),
    ("riskquant.dependence.contracts", "DependenceProvenance"),
    ("riskquant.dependence.contracts", "DependenceResult"),
    ("riskquant.dependence.contracts", "TailCountResult"),
    ("riskquant.dependence.estimators.pearson", "pearson"),
    ("riskquant.dependence.estimators.spearman", "spearman"),
    ("riskquant.dependence.estimators.kendall", "kendall_tau_b"),
    ("riskquant.dependence.estimators.tail_counts", "tail_counts"),
    ("riskquant.dependence.diagnostics.spectrum", "spectrum"),
    ("riskquant.dependence.diagnostics.agreement", "agreement"),
    ("riskquant.dependence.represent.dissimilarity", "angular_dissimilarity"),
    ("riskquant.dependence.represent.dissimilarity", "euclidean_dissimilarity"),
    ("riskquant.dependence.represent.dissimilarity", "absolute_dissimilarity"),
    ("riskquant.dependence.represent.dissimilarity", "one_minus_dependence"),
    ("riskquant.dependence.represent.dissimilarity", "linfoot_dissimilarity"),
]


PUBLIC_SCENARIO_METRICS = [
    ("riskquant.scenario_metrics.api.var_es", "compute_var_es"),
    ("riskquant.scenario_metrics._result", "VarEsResult"),
]


@pytest.mark.parametrize(
    ("module", "name"),
    PUBLIC_OBSERVATIONS + PUBLIC_DEPENDENCE + PUBLIC_SCENARIO_METRICS,
)
def test_the_declared_public_object_exists(
    module: str,
    name: str,
) -> None:
    assert hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "name",
    [
        "riskquant.observations.compile.NormalScores",
        "riskquant.observations.compile.compile_normal_scores",
    ],
)
def test_phase_two_objects_are_not_present(name: str) -> None:
    module_name, attribute = name.rsplit(".", 1)
    assert not hasattr(importlib.import_module(module_name), attribute)


# ------------------------------------------------------------------ core rules


FORBIDDEN = {
    r"import pandas": "no Pandas in the computational core",
    r"matplotlib|seaborn|PdfPages": "no plotting in the core",
    r"print\s*\(": "no print statements",
    r"type:\s*ignore": "no type: ignore",
    r"np\.nan(mean|std|median|max|min|quantile|sum)": "no NaN-skipping reductions",
}


def test_core_packages_obey_the_mechanical_rules() -> None:
    import re

    exempt = {"preprocess.py"}
    for package in (
        "observations",
        "dependence",
        "market_state",
        "scenario_metrics",
        "factor_regression",
    ):
        for path in _modules(package):
            if path.name in exempt:
                continue
            source = _executable_source(path)
            for pattern, rule in FORBIDDEN.items():
                assert not re.search(pattern, source), f"{path}: {rule}"


def test_no_tail_coefficient_is_computed_outside_the_authorised_module() -> None:
    """
    Phase 1 remains counts only; the coefficient layer is confined to Phase 1b-A.

    ``lambda_hat``, ``chi_hat`` and ``chi_bar_hat`` are now authorised, but only
    inside ``dependence.coefficients``. The Phase 1 counting path, the
    estimators, the diagnostics and the whole of ``observations`` must remain
    free of them, so the counts/coefficients split survives as a mechanical
    guarantee rather than a convention.
    """
    import re

    forbidden = re.compile(r"lambda_hat|chi_hat|chi_bar|tail_dependence_coefficient")

    # Two exemptions, and only two. ``dependence.coefficients`` is where the
    # coefficients are computed. ``contracts.py`` may *name* them, because it
    # declares the result container, but it must not compute them -- which the
    # companion test below enforces.
    checked = 0
    for package in ("observations", "dependence"):
        for path in _modules(package):
            if "coefficients" in path.parts or path.name == "contracts.py":
                continue
            checked += 1
            assert not forbidden.search(_executable_source(path)), (
                f"{path} computes a tail coefficient; that belongs to "
                "dependence.coefficients."
            )

    assert checked > 0, "the guard scanned nothing"


def test_the_contracts_module_declares_coefficients_but_computes_nothing() -> None:
    """
    ``contracts.py`` is exempt from the name guard, so it needs its own.

    A container module may carry the field names; it may not carry the
    arithmetic. Any logarithm or division there would mean a coefficient escaped
    the authorised module.
    """
    import re

    source = _executable_source(SOURCE_ROOT / "dependence" / "contracts.py")
    assert not re.search(r"\blog\b|np\.log|math\.log", source)
    assert "/" not in source


def test_the_phase_one_counting_path_names_no_coefficient() -> None:
    """The specific module the split exists to protect."""
    import re

    source = _executable_source(
        SOURCE_ROOT / "dependence" / "estimators" / "tail_counts.py"
    )
    assert not re.search(r"lambda_hat|chi_hat|chi_bar", source)


def test_phase_one_b_b_remains_unimplemented() -> None:
    """Bootstrap and the acceptance test are deferred, and must stay absent."""
    import re

    forbidden = re.compile(
        r"\bbootstrap\b|confidence_interval|asymptotic_independence"
        r"|gaussian_copula_test|minimum_credible_count"
    )
    for package in ("observations", "dependence"):
        for path in _modules(package):
            assert not forbidden.search(_executable_source(path)), (
                f"{path} references a Phase 1b-B capability."
            )


def test_no_asymptotic_coefficient_is_named_in_the_public_surface() -> None:
    """A finite-threshold estimate must never be named as the asymptotic limit."""
    from riskquant.dependence.contracts import TailCountResult

    fields = set(TailCountResult.__dataclass_fields__)
    assert fields == {
        "thresholds",
        "tail",
        "joint_count",
        "marginal_count",
        "provenance",
    }


# ----------------------------------------------------------------- allocation


def test_a_pearson_only_run_allocates_nothing_from_the_rank_pipeline() -> None:
    """
    Measured, not inspected.

    Compiling the sample and running Pearson costs the canonical copy plus one
    standardised buffer. The rank pipeline would add the ranks, the int64
    argsort result and its int32 cast: another 2.5 matrix-sized allocations.
    """
    import tracemalloc

    from riskquant.dependence.estimators.pearson import pearson
    from riskquant.observations.compile import compile_sample
    from riskquant.observations.orientation import ObservationOrientation

    rng = np.random.default_rng(0)
    n_variables, n_observations = 40, 3000
    matrix = np.ascontiguousarray(rng.standard_normal((n_variables, n_observations)))
    rows = ObservationOrientation.VARIABLES_ON_ROWS

    pearson(compile_sample(matrix, orientation=rows))  # warm the JIT

    tracemalloc.start()
    pearson(compile_sample(matrix, orientation=rows))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    matrix_bytes = n_variables * n_observations * 8
    assert peak < 3 * matrix_bytes


def test_compiling_ranks_is_what_costs_the_rank_pipeline() -> None:
    """Companion to the previous test, so its bound is not vacuous."""
    import tracemalloc

    from riskquant.observations.compile import compile_ranks, compile_sample
    from riskquant.observations.orientation import ObservationOrientation

    rng = np.random.default_rng(0)
    n_variables, n_observations = 40, 3000
    matrix = np.ascontiguousarray(rng.standard_normal((n_variables, n_observations)))
    sample = compile_sample(
        matrix, orientation=ObservationOrientation.VARIABLES_ON_ROWS
    )
    compile_ranks(sample)  # warm

    tracemalloc.start()
    compile_ranks(sample)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    matrix_bytes = n_variables * n_observations * 8
    assert peak > matrix_bytes
