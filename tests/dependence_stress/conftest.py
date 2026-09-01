# tests/dependence_stress/conftest.py

"""Shared fixtures for the dependence-stress suite."""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.types import FloatArray

K = 6
S = 400
SEED = 20260827


@pytest.fixture(scope="session")
def base() -> FloatArray:
    """Non-degenerate K x S matrix with heterogeneous marginal shapes."""
    rng = np.random.default_rng(SEED)

    rows = [
        rng.standard_normal(S),
        rng.standard_t(4.0, S),
        rng.lognormal(0.0, 0.7, S),
        rng.uniform(-3.0, 5.0, S),
        rng.standard_normal(S) * 12.0 - 4.0,
        -rng.exponential(2.0, S),
    ]

    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


@pytest.fixture(scope="session")
def equicorrelation() -> FloatArray:
    """Strictly positive-definite K x K equicorrelation matrix."""
    matrix = np.full((K, K), 0.55, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


@pytest.fixture
def tied() -> FloatArray:
    """Explicit tie fixture with hand-computed average ranks."""
    return np.array(
        [
            [1.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )

