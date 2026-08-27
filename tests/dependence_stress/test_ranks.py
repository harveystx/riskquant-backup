# tests/dependence_stress/test_ranks.py

"""Rank semantics. ranks.py is the single authority; ties are the point."""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.ranks import average_ranks, empirical_uniforms
from riskquant.dependence_stress.types import FloatArray

EXPECTED_RANKS = np.array(
    [
        [1.5, 1.5, 3.0, 4.0],
        [1.0, 2.5, 2.5, 4.0],
    ]
)


def test_average_ranks_tie_regression(tied: FloatArray) -> None:
    """Permanent regression against hand-computed mid-ranks."""
    assert np.array_equal(average_ranks(tied), EXPECTED_RANKS)


def test_average_ranks_differ_from_ordinal_ranks(tied: FloatArray) -> None:
    """argsort(argsort(...)) is wrong under ties and must stay rejected."""
    ordinal = np.argsort(np.argsort(tied, axis=1), axis=1) + 1.0

    assert not np.array_equal(average_ranks(tied), ordinal)


def test_average_ranks_span_one_to_s(base: FloatArray) -> None:
    ranks = average_ranks(base)
    s = base.shape[1]

    assert np.allclose(ranks.min(axis=1), 1.0)
    assert np.allclose(ranks.max(axis=1), float(s))
    assert np.allclose(ranks.sum(axis=1), s * (s + 1) / 2.0)


def test_average_ranks_dtype_and_layout(base: FloatArray) -> None:
    ranks = average_ranks(base)

    assert ranks.dtype == np.float64
    assert ranks.flags.c_contiguous
    assert ranks.shape == base.shape


def test_empirical_uniforms_mid_rank_identity(tied: FloatArray) -> None:
    uniforms = empirical_uniforms(tied)

    assert np.allclose(uniforms, (EXPECTED_RANKS - 0.5) / tied.shape[1])


def test_empirical_uniforms_strictly_inside_unit_interval(
    base: FloatArray,
) -> None:
    uniforms = empirical_uniforms(base)

    assert np.all(uniforms > 0.0)
    assert np.all(uniforms < 1.0)


def test_empirical_uniforms_extremes(base: FloatArray) -> None:
    """The lowest uniform is 0.5/S and the highest is (S-0.5)/S."""
    s = base.shape[1]
    uniforms = empirical_uniforms(base)

    assert np.allclose(uniforms.min(axis=1), 0.5 / s)
    assert np.allclose(uniforms.max(axis=1), (s - 0.5) / s)


def test_empirical_uniforms_does_not_mutate_input(base: FloatArray) -> None:
    """The in-place (r - 0.5)/S arithmetic must act on the rank buffer."""
    original = base.copy()

    empirical_uniforms(base)

    assert np.array_equal(base, original)


def test_ranks_reject_invalid_input() -> None:
    with pytest.raises(ValueError):
        average_ranks(np.array([[1.0, np.nan], [1.0, 2.0]]))

    with pytest.raises(ValueError):
        empirical_uniforms(np.arange(4, dtype=np.float64))
