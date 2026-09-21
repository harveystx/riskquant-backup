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
