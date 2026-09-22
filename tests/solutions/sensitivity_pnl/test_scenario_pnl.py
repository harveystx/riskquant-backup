# tests/solutions/sensitivity_pnl/test_scenario_pnl.py
"""
Scenario P&L from `SensitivityPnlCalculator` on derived Greeks.

Subject: `riskquant.solutions.sensitivity_pnl` -- `build_sensitivity_book_spec`
-> `SensitivityPnlCalculator.compile` -> `FactorAxis.levels` -> `run`, plus the
calculator's Taylor conventions on hand-built books.

Inputs come from the production upstream stages, assembled once in
`tests/_support`: the spot sensitizer's Greeks (`sensitivities`) and the
proxy-routed shocked levels (`resolved`). Those stages are asserted in their
own suites (`tests/sensitivities`, `tests/simlab/shocks`); here they are data.

The oracle is derived from the ECONOMIC identities of the instruments, never
from the production sensitivity coefficients and never by re-running the
production algorithm.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
)
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError


@pytest.fixture(scope="module")
def pnl(sensitivities, resolved):
    calculator = SensitivityPnlCalculator.compile(
        build_sensitivity_book_spec(
        sensitivities=sensitivities,
        required_factor_ids=sensitivities.active_factor_rf_ids,
    )
    )
    axis = FactorAxis.from_keys(
        pl.Series("factor_id", resolved["required"].tolist(), dtype=pl.Int32)
    )
    return calculator.run(
        axis.levels(
            base_levels=resolved["base_levels"], shocked_levels=resolved["shocked"]
        )
    )


# =============================================================================
# Sensitivity P&L vs independent economic oracle
# =============================================================================


def test_pnl_matches_independent_economic_oracle(
    rsc, resolved, pnl, position_uids
) -> None:
    """
    Oracle from the instrument identities, not from the sensitivity book:

        AAPL  V = Q*S        -> dV = Q*dS
        WTI   V = Q*S        -> dV = Q*dS      (S proxy-resolved)
        Brent V = Q*B*F      -> dV = Q*(B0*dF + F0*dB + dB*dF)
    """
    aapl_uid, brent_uid, wti_uid = position_uids
    row = resolved["row_of"]
    base, shocked = resolved["base_levels"], resolved["shocked"]
    i_fx = row[int(rsc.rel_name_to_rf_id["USDEUR"])]
    i_aapl = row[int(rsc.rel_name_to_rf_id["AAPL_EQ"])]
    i_brent = row[int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]
    i_wti = row[int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]

    d_fx = shocked[i_fx] - base[i_fx]
    d_brent = shocked[i_brent] - base[i_brent]
    f0, b0 = float(base[i_fx]), float(base[i_brent])

    expected = {
        aapl_uid: 100.0 * (shocked[i_aapl] - base[i_aapl]),
        wti_uid: 75.0 * (shocked[i_wti] - base[i_wti]),
        brent_uid: -50.0 * (b0 * d_fx + f0 * d_brent + d_brent * d_fx),
    }

    for uid, expected_row in expected.items():
        produced = np.asarray(pnl.row_for(uid), dtype=np.float64)
        np.testing.assert_allclose(produced, expected_row, atol=5e-2)


def test_wti_pnl_uses_proxy_shock_not_a_raw_zero(pnl, position_uids) -> None:
    """
    WTI has no market data of its own. If the raw (absent) WTI shock were
    used, its P&L would be identically zero in every scenario.
    """
    _, _, wti_uid = position_uids
    assert float(np.abs(np.asarray(pnl.row_for(wti_uid))).sum()) > 1.0


def test_pnl_shape_and_row_identity(pnl, position_uids) -> None:
    aapl_uid, brent_uid, wti_uid = position_uids
    assert pnl.pnl.shape == (3, 3)
    assert pnl.position_uids.tolist() == [aapl_uid, brent_uid, wti_uid]


# =============================================================================
# Calculator conventions
# =============================================================================


def _one_position_calculator(cross_value: float) -> SensitivityPnlCalculator:
    axis = FactorAxis.from_keys(pl.Series("factor_id", [10, 20], dtype=pl.Int32))
    sens = pl.DataFrame({
        "position_id": pl.Series([1, 1], dtype=pl.Int64),
        "factor_id": pl.Series([10, 20], dtype=pl.Int32),
        "first_order": pl.Series([0.0, 0.0], dtype=pl.Float64),
        "second_order": pl.Series([0.0, 0.0], dtype=pl.Float64),
    })
    cross = pl.DataFrame({
        "position_id": pl.Series([1], dtype=pl.Int64),
        "factor_id_i": pl.Series([10], dtype=pl.Int32),
        "factor_id_j": pl.Series([20], dtype=pl.Int32),
        "cross_second_order": pl.Series([cross_value], dtype=pl.Float64),
    })
    meta = pl.DataFrame({
        "factor_id": pl.Series([10, 20], dtype=pl.Int32),
        "displacement_convention_id": pl.Series(
            [int(DisplacementConvention.ABSOLUTE)] * 2, dtype=pl.Int8
        ),
        "quote_base": pl.Series([1.0, 1.0], dtype=pl.Float64),
    })
    return SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=[1],
            cross_sensitivities=cross,
        )
    )


def test_cross_term_has_no_half_multiplier() -> None:
    """PnL cross term is `c * x_i * x_j`, NOT `0.5 * c * x_i * x_j`."""
    calc = _one_position_calculator(cross_value=1.0)
    axis = FactorAxis.from_keys(pl.Series("factor_id", [10, 20], dtype=pl.Int32))
    shocks = axis.shocks(np.array([[2.0], [3.0]], dtype=np.float32))
    produced = float(calc.run(shocks).pnl[0, 0])
    assert produced == pytest.approx(6.0, abs=1e-5)  # c*x*y
    assert produced != pytest.approx(3.0, abs=1e-5)  # not 0.5*c*x*y


def test_diagonal_second_order_does_carry_a_half() -> None:
    """The DIAGONAL second-order term is `0.5 * d2 * x^2`."""
    axis = FactorAxis.from_keys(pl.Series("factor_id", [10], dtype=pl.Int32))
    sens = pl.DataFrame({
        "position_id": pl.Series([1], dtype=pl.Int64),
        "factor_id": pl.Series([10], dtype=pl.Int32),
        "first_order": pl.Series([0.0], dtype=pl.Float64),
        "second_order": pl.Series([1.0], dtype=pl.Float64),
    })
    meta = pl.DataFrame({
        "factor_id": pl.Series([10], dtype=pl.Int32),
        "displacement_convention_id": pl.Series(
            [int(DisplacementConvention.ABSOLUTE)], dtype=pl.Int8
        ),
        "quote_base": pl.Series([1.0], dtype=pl.Float64),
    })
    calc = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=[1],
            cross_sensitivities=None,
        )
    )
    produced = float(
        calc.run(axis.shocks(np.array([[2.0]], dtype=np.float32))).pnl[0, 0]
    )
    assert produced == pytest.approx(2.0, abs=1e-5)  # 0.5 * 1 * 4


def test_factor_axis_ordering_is_respected_not_assumed() -> None:
    """
    A non-trivially ordered axis must still attach each shock to its own
    factor. Row order is free; the axis is the contract.
    """
    # Descending keys -> SLOT encoding (keys are not sorted ascending).
    axis = FactorAxis.from_keys(pl.Series("factor_id", [20, 10], dtype=pl.Int32))
    sens = pl.DataFrame({
        "position_id": pl.Series([1, 1], dtype=pl.Int64),
        "factor_id": pl.Series([10, 20], dtype=pl.Int32),
        "first_order": pl.Series([1.0, 1000.0], dtype=pl.Float64),
        "second_order": pl.Series([0.0, 0.0], dtype=pl.Float64),
    })
    meta = pl.DataFrame({
        "factor_id": pl.Series([10, 20], dtype=pl.Int32),
        "displacement_convention_id": pl.Series(
            [int(DisplacementConvention.ABSOLUTE)] * 2, dtype=pl.Int8
        ),
        "quote_base": pl.Series([1.0, 1.0], dtype=pl.Float64),
    })
    calc = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=[1],
            cross_sensitivities=None,
        )
    )
    # Row 0 is factor 20 (d1=1000), row 1 is factor 10 (d1=1).
    produced = float(
        calc.run(axis.shocks(np.array([[1.0], [7.0]], dtype=np.float32))).pnl[0, 0]
    )
    assert produced == pytest.approx(1000.0 * 1.0 + 1.0 * 7.0, abs=1e-4)


def test_missing_factor_row_fails_fast() -> None:
    """
    A sensitivity referencing a factor absent from the market axis must
    RAISE, never contribute a silent zero in every scenario.
    """
    axis = FactorAxis.from_keys(pl.Series("factor_id", [10], dtype=pl.Int32))
    sens = pl.DataFrame({
        "position_id": pl.Series([1, 1], dtype=pl.Int64),
        "factor_id": pl.Series([10, 999], dtype=pl.Int32),
        "first_order": pl.Series([1.0, 1.0], dtype=pl.Float64),
        "second_order": pl.Series([0.0, 0.0], dtype=pl.Float64),
    })
    meta = pl.DataFrame({
        "factor_id": pl.Series([10, 999], dtype=pl.Int32),
        "displacement_convention_id": pl.Series(
            [int(DisplacementConvention.ABSOLUTE)] * 2, dtype=pl.Int8
        ),
        "quote_base": pl.Series([1.0, 1.0], dtype=pl.Float64),
    })
    with pytest.raises((SensitivityPnlInputError, KeyError, ValueError)):
        SensitivityPnlCalculator.compile(
            SensitivityBookSpec(
                sensitivities=sens,
                factor_meta=meta,
                factor_axis=axis,
                position_uids=[1],
                cross_sensitivities=None,
            )
        )
