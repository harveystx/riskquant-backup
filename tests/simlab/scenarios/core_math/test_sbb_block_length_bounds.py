# tests/simlab/scenarios/core_math/test_sbb_block_length_bounds.py
"""
G3 regression: SBB geometric block lengths must never be narrowed to a wrapped
int32 value.

``rng.geometric`` returns int64 draws >= 1. For a large (schema-valid)
``expected_block_size`` the draws exceed the int32 range; a bare int32 cast
wrapped them to small or negative block lengths, which broke contiguous blocks
or made the stitch kernel write out of bounds (process abort). Every draw >= h
is equivalent to h in the stitch kernel, so the expected paths below are
computed by hand from the documented protocol: a block that covers the whole
horizon yields ``start, start+1, ...`` with circular wrap into [1, T-1].
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)

_T = 14
_INT32_MAX = 2**31 - 1


class _FixedDrawRng:
    """Deterministic stand-in for ``np.random.Generator``: fixed int64 draws."""

    def __init__(self, geometric: list[int], starts: list[int]) -> None:
        self._geometric = np.asarray(geometric, dtype=np.int64)
        self._starts = starts

    def geometric(self, p: float, size: int) -> npt.NDArray[np.int64]:
        assert size == self._geometric.shape[0]
        return self._geometric.copy()

    def integers(
        self, *, low: int, high: int, size: int, dtype: type
    ) -> npt.NDArray[np.generic]:
        assert size == len(self._starts)
        return np.asarray(self._starts, dtype=dtype)


@pytest.mark.parametrize(
    "draw",
    [2**32 + 2, 2**31 + 5, np.iinfo(np.int64).max],
    ids=["wraps_to_2", "wraps_negative", "int64_max"],
)
def test_block_lengths_above_int32_cover_the_horizon(draw: int) -> None:
    """A draw above the int32 range is one block over the whole horizon."""
    rng = _FixedDrawRng(geometric=[draw] * 6, starts=[3, 9, 5, 11, 2, 7])
    out = generate_sbb_path_indices(
        t_max=_T,
        h=3,
        total_scenarios=2,
        expected_block_size=2.0,
        rng=rng,  # type: ignore[arg-type]
    )
    # Scenario 0 uses draw 0 (start 3); scenario 1 uses draw 1 (start 9).
    np.testing.assert_array_equal(out, np.array([[3, 4, 5], [9, 10, 11]]))


@pytest.mark.parametrize("draw", [0, -5], ids=["zero", "negative"])
def test_block_lengths_below_one_are_refused(draw: int) -> None:
    """A geometric draw is >= 1; anything else fails loud before stitching."""
    rng = _FixedDrawRng(geometric=[draw] * 6, starts=[3, 9, 5, 11, 2, 7])
    with pytest.raises(RuntimeError, match="block length < 1"):
        generate_sbb_path_indices(
            t_max=_T,
            h=3,
            total_scenarios=2,
            expected_block_size=2.0,
            rng=rng,  # type: ignore[arg-type]
        )


def test_horizon_above_int32_is_refused_before_any_draw() -> None:
    with pytest.raises(ValueError, match="int32 path contract"):
        generate_sbb_path_indices(
            t_max=_T,
            h=_INT32_MAX + 1,
            total_scenarios=1,
            expected_block_size=3.0,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize(
    "expected_block_size",
    [1e9, float(np.finfo(np.float64).max)],
    ids=["1e9", "max_finite"],
)
def test_huge_expected_block_size_gives_one_contiguous_block(
    expected_block_size: float,
) -> None:
    """Every path is one block: consecutive indices with circular wrap."""
    h = 5
    out = generate_sbb_path_indices(
        t_max=_T,
        h=h,
        total_scenarios=64,
        expected_block_size=expected_block_size,
        rng=np.random.default_rng(11),
    )
    assert out.shape == (64, h)
    assert int(out.min()) >= 1 and int(out.max()) <= _T - 1
    nxt = 1 + (out[:, :-1] % (_T - 1))
    np.testing.assert_array_equal(out[:, 1:], nxt)


def test_generator_with_huge_block_size_streams_every_scenario() -> None:
    levels = 100.0 * np.exp(np.cumsum(np.linspace(-0.02, 0.03, _T)))
    matrix = RiskFactorMatrix(
        dates=np.arange(_T, dtype=np.int32),
        factor_ids=np.array([7], dtype=np.int32),
        values=np.asfortranarray(levels[:, None]),
    )
    gen = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=37,
        expected_block_size=1e9,
        horizon=4,
        seed=5,
    )
    batches = [
        np.array(b.shocks, copy=True)
        for b in gen.iter_batches(
            driver_rf_ids=np.array([7], dtype=np.int32),
            chart_ids=np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8),
            batch_size=8,
        )
    ]
    shocks = np.vstack(batches)
    assert shocks.shape == (37, 1)
    assert bool(np.isfinite(shocks).all())
