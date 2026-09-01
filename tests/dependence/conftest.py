# tests/dependence/conftest.py

"""Shared fixtures and helpers for the dependence test suite."""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.observations.compile import (
    CompiledSample,
    RankRepresentation,
    compile_ranks,
    compile_sample,
)
from riskquant.observations.orientation import ObservationOrientation

VARIABLES_ROWS = ObservationOrientation.VARIABLES_ON_ROWS


def compile_both(matrix: np.ndarray) -> tuple[CompiledSample, RankRepresentation]:
    """Compile a raw matrix into the sample and its rank representation."""
    sample = compile_sample(matrix, orientation=VARIABLES_ROWS)
    return sample, compile_ranks(sample)


def continuous_sample(seed: int, n_variables: int, n_observations: int) -> np.ndarray:
    """Correlated Gaussian panel, so the estimators do real work."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_observations)
    matrix = np.empty((n_variables, n_observations), dtype=np.float64)
    for variable in range(n_variables):
        loading = 0.3 + 0.6 * variable / max(1, n_variables - 1)
        matrix[variable] = loading * factor + np.sqrt(
            1.0 - loading**2
        ) * rng.standard_normal(n_observations)
    return matrix


def zero_inflated_sample(
    seed: int, n_variables: int, n_observations: int, zero_fraction: float = 0.4
) -> np.ndarray:
    """Tie-heavy panel: the regime in which rank conventions actually matter."""
    rng = np.random.default_rng(seed)
    base = continuous_sample(seed, n_variables, n_observations)
    return np.ascontiguousarray(
        np.where(rng.random((n_variables, n_observations)) < zero_fraction, 0.0, base)
    )


@pytest.fixture
def gaussian_pair() -> np.ndarray:
    """Small hand-inspectable correlated panel."""
    return continuous_sample(seed=7, n_variables=5, n_observations=300)
