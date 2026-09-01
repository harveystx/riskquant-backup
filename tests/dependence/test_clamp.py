# tests/dependence/test_clamp.py

"""
The three bound regimes.

A finite declared range does **not** license clamping. The third test is the
essential one: an excursion beyond the numerical tolerance is an implementation
defect and must raise, not be silently converted into a plausible correlation.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.constants import NUMERICAL_EXCURSION_TOLERANCE
from riskquant.dependence._matrix import allocate_dependence_matrix, finalise_symmetric
from riskquant.dependence.taxonomy import (
    ESTIMATOR_CONTRACT,
    DependenceContract,
    DependenceEstimator,
    PsdStatus,
)

PEARSON_CONTRACT = ESTIMATOR_CONTRACT[DependenceEstimator.PEARSON]

UNBOUNDED_CONTRACT = DependenceContract(
    estimator=DependenceEstimator.PEARSON,
    value_range=(-float("inf"), float("inf")),
    diagonal_value=1.0,
    psd_status=PsdStatus.NOT_A_CORRELATION_MATRIX,
)


def _matrix_with(value: float) -> np.ndarray:
    matrix = allocate_dependence_matrix(2)
    matrix[0, 1] = value
    return matrix


def test_regime_one_value_inside_the_range_is_untouched() -> None:
    result = finalise_symmetric(_matrix_with(0.5), contract=PEARSON_CONTRACT)
    assert result[0, 1] == 0.5


def test_regime_two_excursion_inside_the_tolerance_is_clamped() -> None:
    """1.0000000000000002 is provable rounding error: Cauchy-Schwarz bounds the truth."""
    excursion = 1.0 + 2.0 * np.finfo(np.float64).eps
    assert excursion > 1.0

    result = finalise_symmetric(_matrix_with(excursion), contract=PEARSON_CONTRACT)
    assert result[0, 1] == 1.0


def test_regime_two_applies_at_the_lower_bound_too() -> None:
    excursion = -1.0 - 2.0 * np.finfo(np.float64).eps
    result = finalise_symmetric(_matrix_with(excursion), contract=PEARSON_CONTRACT)
    assert result[0, 1] == -1.0


def test_regime_two_boundary_exactly_at_the_tolerance_is_clamped() -> None:
    result = finalise_symmetric(
        _matrix_with(1.0 + NUMERICAL_EXCURSION_TOLERANCE), contract=PEARSON_CONTRACT
    )
    assert result[0, 1] == 1.0


@pytest.mark.parametrize("bad", [1.4, -1.4, 2.0, 1.0 + 1e-6, -1.0 - 1e-6])
def test_regime_three_excursion_beyond_the_tolerance_raises(bad: float) -> None:
    """
    The essential test.

    Under the rejected rule "finite range implies clamp", an estimator returning
    1.4 from an indexing bug would be clamped to 1.0 and would ship as a
    plausible correlation.
    """
    with pytest.raises(ValueError, match="implementation defect, not rounding error"):
        finalise_symmetric(_matrix_with(bad), contract=PEARSON_CONTRACT)


def test_the_raised_message_locates_the_offending_pair() -> None:
    matrix = allocate_dependence_matrix(4)
    matrix[np.triu_indices(4, k=1)] = 0.1
    matrix[1, 3] = 5.0

    with pytest.raises(ValueError) as excinfo:
        finalise_symmetric(matrix, contract=PEARSON_CONTRACT)

    message = str(excinfo.value)
    assert "(1, 3)" in message
    assert "PEARSON" in message


def test_an_infinite_bound_disables_the_corresponding_clamp() -> None:
    """
    An unbounded estimator has no bound to round to, and a negative value from a
    bias-corrected statistic is a real result rather than an excursion.
    """
    result = finalise_symmetric(_matrix_with(-42.0), contract=UNBOUNDED_CONTRACT)
    assert result[0, 1] == -42.0


def test_finalisation_sets_the_diagonal_and_mirrors_and_freezes() -> None:
    result = finalise_symmetric(_matrix_with(0.25), contract=PEARSON_CONTRACT)

    assert np.array_equal(result, result.T)
    assert np.array_equal(np.diag(result), np.ones(2))
    assert not result.flags.writeable


def test_a_real_estimate_needs_at_most_a_rounding_sized_correction() -> None:
    """The clamp must never be doing statistical work on real data."""
    from riskquant.dependence.estimators.pearson import pearson
    from tests.dependence.conftest import compile_both, continuous_sample

    matrix = continuous_sample(seed=13, n_variables=6, n_observations=300)
    sample, _ = compile_both(matrix)

    assert pearson(sample).matrix == pytest.approx(np.corrcoef(matrix), abs=1e-12)
