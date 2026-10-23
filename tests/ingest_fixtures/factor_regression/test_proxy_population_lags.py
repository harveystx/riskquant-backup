# tests/ingest_fixtures/factor_regression/test_proxy_population_lags.py
"""
Tests of ``population_lagged_loadings`` in the proxy calibration fixture.

The closed form counts shared sessions, as ``population_loadings`` does. The
tests pin it in three ways: against ``population_loadings`` at lag 0, against
a simulation of the documented process for lags 0 and 1, and against the rule
that a lag of 2 or more shares no session and gets a loading of exactly 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.ingest_fixtures.factor_regression.proxy_calibration import (
    BETA_EU,
    BETA_US,
    population_lagged_loadings,
    population_loadings,
)

# Owner sandbox simulation of the documented process (2,000,000 days, seed 3,
# daily rows, EU at lag 0, US at lags 0 and 1). The sampling error is about
# 0.002 per loading, so the tolerance below is 0.005.
_SIMULATED_LAGS_0_1 = (1.043, 0.111, 0.073)


def test_lag_zero_equals_the_step_closed_form_at_one_day() -> None:
    got = population_lagged_loadings((0,))
    want = population_loadings(np.array([1], dtype=np.int32))[0]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_lags_zero_and_one_match_the_simulation() -> None:
    np.testing.assert_allclose(
        population_lagged_loadings((0, 1)), _SIMULATED_LAGS_0_1, rtol=0.0, atol=0.005
    )


@pytest.mark.parametrize("extra", [(2,), (2, 3), (5,)])
def test_a_lag_of_two_or_more_has_zero_loading_and_changes_nothing(
    extra: tuple[int, ...],
) -> None:
    base = population_lagged_loadings((0, 1))
    got = population_lagged_loadings((0, 1, *extra))
    np.testing.assert_allclose(got[:3], base, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(got[3:], 0.0, rtol=0.0, atol=1e-15)


def test_the_order_of_the_lags_orders_the_loadings() -> None:
    forward = population_lagged_loadings((0, 1))
    reverse = population_lagged_loadings((1, 0))
    np.testing.assert_allclose(reverse, forward[[0, 2, 1]], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("lags", [(0,), (0, 1), (0, 1, 2, 3)])
def test_the_eu_index_absorbs_part_of_the_us_effect(lags: tuple[int, ...]) -> None:
    # The EU index covers the target window and is correlated with the US
    # index, so the lags never recover BETA_US: the Dimson sum stays below it.
    got = population_lagged_loadings(lags)
    assert got[0] > BETA_EU
    assert 0.0 < float(got[1:].sum()) < BETA_US


@pytest.mark.parametrize(
    ("lags", "message"),
    [((), "at least one"), ((0, -1), ">= 0"), ((0, 1, 1), "distinct")],
)
def test_invalid_lags_raise_value_error(lags: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        population_lagged_loadings(lags)
