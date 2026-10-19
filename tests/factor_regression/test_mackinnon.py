# tests/factor_regression/test_mackinnon.py
"""
Slice S2 of WS_FACTOR_REGRESSION_COINTEGRATION: MacKinnon (2010) N = 1 critical
values (A.2, C.7).

The expected constants are typed again here from Tables 2 and 3 of MacKinnon
(2010), QED WP 1227, independently of the module (S2.T1). The Monte Carlo
oracle (S2.T2) builds the Dickey-Fuller statistic in closed form with numpy
only: no statsmodels, no ``fit_core``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import mackinnon

# ------------------------------------------------------------- typed tables
# Rows: level 0.01, 0.05, 0.10. Columns: b_inf, b1, b2, b3. Empty cell 0.0.

_TAU_C_LITERALS = (
    (-3.43035, -6.5393, -16.786, -79.433),
    (-2.86154, -2.8903, -4.234, -40.040),
    (-2.56677, -1.5384, -2.809, 0.0),
)
_TAU_CT_LITERALS = (
    (-3.95877, -9.0531, -28.428, -134.155),
    (-3.41049, -4.3904, -9.036, -45.374),
    (-3.12705, -2.5856, -3.925, -22.380),
)

# The same cells as decimal strings for the exact evaluation of S2.T5.
_TAU_C_TEXT = (
    ("-3.43035", "-6.5393", "-16.786", "-79.433"),
    ("-2.86154", "-2.8903", "-4.234", "-40.040"),
    ("-2.56677", "-1.5384", "-2.809", "0"),
)
_TAU_CT_TEXT = (
    ("-3.95877", "-9.0531", "-28.428", "-134.155"),
    ("-3.41049", "-4.3904", "-9.036", "-45.374"),
    ("-3.12705", "-2.5856", "-3.925", "-22.380"),
)

_TABLES = (mackinnon.TABLE_TAU_C, mackinnon.TABLE_TAU_CT)
_LITERALS = {
    mackinnon.TABLE_TAU_C: _TAU_C_LITERALS,
    mackinnon.TABLE_TAU_CT: _TAU_CT_LITERALS,
}
_TEXT = {mackinnon.TABLE_TAU_C: _TAU_C_TEXT, mackinnon.TABLE_TAU_CT: _TAU_CT_TEXT}

# ------------------------------------------------------------ S2.T1 constants


def test_t1_the_two_typed_test_tables_agree() -> None:
    # A typing error in one test table must not pass silently.
    for table in _TABLES:
        for literal_row, text_row in zip(_LITERALS[table], _TEXT[table], strict=True):
            assert tuple(float(t) for t in text_row) == literal_row


def test_t1_the_table_selectors_are_fixed() -> None:
    assert mackinnon.TABLE_TAU_C == 0
    assert mackinnon.TABLE_TAU_CT == 1
    assert type(mackinnon.TABLE_TAU_C) is int
    assert type(mackinnon.TABLE_TAU_CT) is int


def test_t1_every_constant_equals_the_typed_table() -> None:
    stored = mackinnon._COEFFICIENTS
    assert type(stored) is np.ndarray
    assert stored.dtype == np.float64
    assert stored.shape == (2, 3, 4)
    assert stored.flags.c_contiguous
    assert not stored.flags.writeable
    expected = np.array(
        [_LITERALS[mackinnon.TABLE_TAU_C], _LITERALS[mackinnon.TABLE_TAU_CT]],
        dtype=np.float64,
    )
    # Exact equality, cell by cell, so a failure names the cell.
    for index in np.ndindex(expected.shape):
        assert stored[index] == expected[index], index


def test_t1_the_store_holds_23_nonzero_constants_and_one_empty_cell() -> None:
    stored = mackinnon._COEFFICIENTS
    assert int(np.count_nonzero(stored)) == 23
    assert stored[mackinnon.TABLE_TAU_C, 2, 3] == 0.0


# ------------------------------------------------------- S2.T2 Monte Carlo

_MC_SEED = 20261019
_MC_REPLICATIONS = 200_000
_MC_CHUNK = 20_000
_MC_N = 100
_MC_LEVELS = (0.01, 0.05, 0.10)
_MC_TOLERANCE = (0.05, 0.03, 0.03)


def _projection_basis(n_rows: int, *, trend: bool) -> NDArray[np.float64]:
    # Orthonormal basis of the deterministic columns: ones, then the trend.
    columns = [np.ones(n_rows, dtype=np.float64)]
    if trend:
        columns.append(np.arange(n_rows, dtype=np.float64))
    basis, _ = np.linalg.qr(np.column_stack(columns))
    return np.ascontiguousarray(basis)


def _df_tau(
    dz: NDArray[np.float64], z_lag: NDArray[np.float64], *, trend: bool
) -> NDArray[np.float64]:
    """
    Lag-0 Dickey-Fuller tau for each row of ``dz`` on ``z_lag``.

    Frisch-Waugh-Lovell: partial the deterministic columns out of the response
    and the regressor, then ``gamma = x'y / x'x`` and
    ``se = sqrt(rss / (n - k) / x'x)`` with ``k`` the full column count
    (DF convention, MacKinnon 2010 section 3).
    """
    n_rows = dz.shape[1]
    basis = _projection_basis(n_rows, trend=trend)
    y = dz - (dz @ basis) @ basis.T
    x = z_lag - (z_lag @ basis) @ basis.T
    sxx = np.einsum("ij,ij->i", x, x)
    gamma = np.einsum("ij,ij->i", x, y) / sxx
    resid = y - gamma[:, None] * x
    rss = np.einsum("ij,ij->i", resid, resid)
    k = basis.shape[1] + 1
    se = np.sqrt(rss / (n_rows - k) / sxx)
    out: NDArray[np.float64] = gamma / se
    return out


@pytest.fixture(scope="module")
def monte_carlo_tau() -> dict[int, NDArray[np.float64]]:
    rng = np.random.default_rng(_MC_SEED)
    taus: dict[int, list[NDArray[np.float64]]] = {t: [] for t in _TABLES}
    for _ in range(_MC_REPLICATIONS // _MC_CHUNK):
        steps = rng.standard_normal((_MC_CHUNK, _MC_N))
        z = np.zeros((_MC_CHUNK, _MC_N + 1), dtype=np.float64)
        np.cumsum(steps, axis=1, out=z[:, 1:])
        dz = np.ascontiguousarray(z[:, 1:] - z[:, :-1])
        z_lag = np.ascontiguousarray(z[:, :-1])
        taus[mackinnon.TABLE_TAU_C].append(_df_tau(dz, z_lag, trend=False))
        taus[mackinnon.TABLE_TAU_CT].append(_df_tau(dz, z_lag, trend=True))
    return {t: np.concatenate(parts) for t, parts in taus.items()}


def test_t2_the_oracle_uses_the_declared_sample(
    monte_carlo_tau: dict[int, NDArray[np.float64]],
) -> None:
    for table in _TABLES:
        tau = monte_carlo_tau[table]
        assert tau.shape == (_MC_REPLICATIONS,)
        with np.errstate(all="ignore"):
            assert bool(np.all(np.isfinite(tau)))


@pytest.mark.parametrize("table", _TABLES)
def test_t2_monte_carlo_quantiles_match_c_100(
    table: int, monte_carlo_tau: dict[int, NDArray[np.float64]]
) -> None:
    simulated = np.quantile(monte_carlo_tau[table], _MC_LEVELS)
    tabulated = mackinnon.critical_values(table, _MC_N)
    deviation = np.abs(simulated - tabulated)
    assert bool(np.all(deviation <= np.asarray(_MC_TOLERANCE))), (
        simulated,
        tabulated,
        deviation,
    )


# ------------------------------------------------------- S2.T3 asymptote


@pytest.mark.parametrize("table", _TABLES)
def test_t3_a_very_large_n_gives_b_inf(table: int) -> None:
    values = mackinnon.critical_values(table, 10**9)
    for level in range(3):
        b_inf = _LITERALS[table][level][0]
        assert abs(float(values[level]) - b_inf) <= 1e-8


# ------------------------------------------------------- S2.T4 level order


def _grid() -> list[int]:
    dense = list(range(20, 201))
    log_grid = sorted({round(v) for v in np.geomspace(20.0, 1e6, 400).tolist()})
    assert len(log_grid) >= 200
    assert log_grid[-1] == 1_000_000
    return dense + log_grid


@pytest.mark.parametrize("table", _TABLES)
def test_t4_levels_are_ordered_and_negative(table: int) -> None:
    for n in _grid():
        c = mackinnon.critical_values(table, n)
        assert c[0] < c[1] < c[2] < 0.0, (n, c)


# --------------------------------------------- S2.T5 exact rational oracle


def _exact(text_row: tuple[str, str, str, str], n: int) -> Fraction:
    b_inf, b1, b2, b3 = (Fraction(t) for t in text_row)
    return b_inf + b1 / n + b2 / n**2 + b3 / n**3


@pytest.mark.parametrize("table", _TABLES)
@pytest.mark.parametrize("n", [20, 25, 100, 1000])
def test_t5_c_matches_an_exact_rational_evaluation(table: int, n: int) -> None:
    values = mackinnon.critical_values(table, n)
    for level in range(3):
        exact = _exact(_TEXT[table][level], n)
        relative = abs(Fraction(float(values[level])) - exact) / abs(exact)
        assert relative <= Fraction(1, 10**15), (table, n, level, float(relative))


def test_t5_detects_a_swap_of_b2_and_b3() -> None:
    # Companion check: the oracle separates the swapped polynomial, so S2.T5
    # is not vacuous for the defect that S2.T2 and S2.T3 cannot see.
    text = _TAU_C_TEXT[0]
    swapped = (text[0], text[1], text[3], text[2])
    gap = abs(_exact(swapped, 100) - _exact(text, 100)) / abs(_exact(text, 100))
    assert gap > Fraction(1, 10**4)


# --------------------------------------------------------------- checks


@pytest.mark.parametrize(
    "bad", [True, False, np.bool_(True), 0.0, 1.0, "0", None, Fraction(0)]
)
def test_a_table_that_is_not_an_integer_raises_type_error(bad: Any) -> None:
    with pytest.raises(TypeError):
        mackinnon.critical_values(bad, 100)


@pytest.mark.parametrize(
    "bad", [True, np.bool_(True), 100.0, np.float64(100.0), "100", None]
)
def test_an_n_that_is_not_an_integer_raises_type_error(bad: Any) -> None:
    with pytest.raises(TypeError):
        mackinnon.critical_values(mackinnon.TABLE_TAU_C, bad)


@pytest.mark.parametrize("bad", [-1, 2, 3, np.int8(2)])
def test_an_unknown_table_raises_value_error(bad: int) -> None:
    with pytest.raises(ValueError, match="table"):
        mackinnon.critical_values(bad, 100)


@pytest.mark.parametrize("bad", [0, -1, -(10**20), np.int64(0), np.int32(-5)])
def test_an_n_below_one_raises_value_error(bad: int) -> None:
    with pytest.raises(ValueError, match="n must be at least 1"):
        mackinnon.critical_values(mackinnon.TABLE_TAU_C, bad)


@pytest.mark.parametrize(
    ("table", "n"),
    [
        (np.int8(0), np.int64(100)),
        (np.int32(1), np.uint16(100)),
        (np.int64(0), np.int32(100)),
    ],
)
def test_numpy_integers_give_the_python_int_result(table: Any, n: Any) -> None:
    expected = mackinnon.critical_values(int(table), int(n))
    assert np.array_equal(mackinnon.critical_values(table, n), expected)


def test_n_equal_to_one_is_accepted() -> None:
    values = mackinnon.critical_values(mackinnon.TABLE_TAU_C, 1)
    with np.errstate(all="ignore"):
        assert bool(np.all(np.isfinite(values)))


# --------------------------------------------------------------- result


def test_the_result_is_a_new_read_only_float64_array() -> None:
    values = mackinnon.critical_values(mackinnon.TABLE_TAU_CT, 250)
    assert type(values) is np.ndarray
    assert values.dtype == np.float64
    assert values.shape == (3,)
    assert values.flags.c_contiguous
    assert not values.flags.writeable
    assert not np.shares_memory(values, mackinnon._COEFFICIENTS)
    with pytest.raises(ValueError, match="read-only"):
        values[0] = 0.0


def test_two_calls_are_equal_and_leave_the_constants_unchanged() -> None:
    before = mackinnon._COEFFICIENTS.copy()
    first = mackinnon.critical_values(mackinnon.TABLE_TAU_C, 100)
    second = mackinnon.critical_values(mackinnon.TABLE_TAU_C, 100)
    assert first is not second
    assert not np.shares_memory(first, second)
    assert np.array_equal(first, second)
    assert np.array_equal(mackinnon._COEFFICIENTS, before)
    assert not mackinnon._COEFFICIENTS.flags.writeable
