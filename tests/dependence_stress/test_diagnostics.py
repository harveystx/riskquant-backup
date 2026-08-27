# tests/dependence_stress/test_diagnostics.py

"""Measurement-only module. It must never transform or import the engine."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from riskquant.dependence_stress import diagnostics
from riskquant.dependence_stress.diagnostics import (
    empirical_lower_tail_cooccurrence,
    lower_tail_dependence,
    pearson_correlation,
    spearman_correlation,
)
from riskquant.dependence_stress.types import FloatArray


def test_pearson_matches_numpy(base: FloatArray) -> None:
    result = pearson_correlation(base)

    assert np.allclose(result, np.corrcoef(base))
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert result.shape == (base.shape[0], base.shape[0])


def test_spearman_matches_scipy_with_ties() -> None:
    data = np.array(
        [
            [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
            [4.0, 5.0, 5.0, 6.0, 2.0, 2.0],
            [9.0, 1.0, 3.0, 3.0, 7.0, 5.0],
        ]
    )

    reference = spearmanr(data, axis=1).statistic

    assert np.allclose(spearman_correlation(data), reference)


def test_spearman_is_invariant_under_monotone_transform(
    base: FloatArray,
) -> None:
    transformed = np.exp(base / 10.0)

    assert np.allclose(
        spearman_correlation(base),
        spearman_correlation(transformed),
    )


def test_pearson_rejects_zero_variance_row() -> None:
    data = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="zero-variance observation rows"):
        pearson_correlation(data)


def test_spearman_rejects_fully_tied_row() -> None:
    """Every observation tied gives a constant rank row."""
    data = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="zero-variance rank rows"):
        spearman_correlation(data)


def test_spearman_accepts_partially_tied_row() -> None:
    data = np.array([[1.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

    assert np.all(np.isfinite(spearman_correlation(data)))


def test_error_message_identifies_offending_rows() -> None:
    data = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [7.0, 7.0, 7.0]])

    with pytest.raises(ValueError, match=r"rows=\[0, 2\]"):
        pearson_correlation(data)


# ======================================================================
# Lower-tail co-occurrence
# ======================================================================


def test_cooccurrence_diagonal_is_exactly_one(base: FloatArray) -> None:
    result = empirical_lower_tail_cooccurrence(base, 0.05)

    assert np.array_equal(np.diag(result), np.ones(base.shape[0]))


def test_cooccurrence_denominator_is_n_tail() -> None:
    """S=10, q=0.05 gives n_tail=1 by the ceiling, not 0.5."""
    data = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ]
    )

    result = empirical_lower_tail_cooccurrence(data, 0.05)

    assert np.allclose(result, 1.0)


def test_cooccurrence_is_zero_for_disjoint_tails() -> None:
    """Row 1 is reversed, so the two lower tails never share a scenario."""
    row = np.arange(100, dtype=np.float64)
    data = np.vstack([row, row[::-1]])

    result = empirical_lower_tail_cooccurrence(data, 0.05)

    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0


def test_cooccurrence_counts_shared_scenarios() -> None:
    """Hand-checked: n_tail=2, one shared tail scenario, so 0.5."""
    data = np.array(
        [
            [0.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [0.0, 9.0, 1.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]
    )

    result = empirical_lower_tail_cooccurrence(data, 0.2)

    assert result[0, 1] == pytest.approx(0.5)


def test_cooccurrence_is_symmetric(base: FloatArray) -> None:
    result = empirical_lower_tail_cooccurrence(base, 0.1)

    assert np.array_equal(result, result.T)


def test_cooccurrence_bounded_in_unit_interval(base: FloatArray) -> None:
    result = empirical_lower_tail_cooccurrence(base, 0.1)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1, np.nan])
def test_cooccurrence_rejects_invalid_threshold(
    base: FloatArray,
    bad_q: float,
) -> None:
    with pytest.raises(ValueError, match="probability"):
        empirical_lower_tail_cooccurrence(base, bad_q)


def test_deprecated_alias_is_the_same_function() -> None:
    assert lower_tail_dependence is empirical_lower_tail_cooccurrence


# ======================================================================
# Architectural constraints
# ======================================================================


def test_diagnostics_does_not_import_engine() -> None:
    """
    diagnostics must never depend on the orchestration layer.

    The check parses the import statements rather than grepping the
    source, so the word "engine" appearing in a docstring cannot cause a
    false failure and an aliased import cannot cause a false pass.
    """
    import ast

    with open(diagnostics.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert not any("engine" in module for module in imported)
    assert {"ranks", "types", "validation"} <= imported


def test_diagnostics_do_not_mutate_input(base: FloatArray) -> None:
    original = base.copy()

    pearson_correlation(base)
    spearman_correlation(base)
    empirical_lower_tail_cooccurrence(base, 0.05)

    assert np.array_equal(base, original)
