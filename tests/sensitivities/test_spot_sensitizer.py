# tests/sensitivities/test_spot_sensitizer.py
"""
Production tests for the SpotBlock sensitivity layer.

Exercises `sensitivities.engine.run_sensitivities` ->
`sensitivities.sensitizers.spot` -> `sensitizer_kernels.spot`, plus the
`sensitizer_registry` dispatch contract. The solution-tier consumer of this
output (`book_adapter` -> calculator) is tested in
`tests/solutions/sensitivity_pnl/test_spot_book_adapter.py`.

Fixtures are built directly from the production compilers (no demo import),
so these are tests of the production layer itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivities.sensitizer_registry import (
    BLOCK_SENSITIZERS,
    UNREGISTERED_BLOCK_TYPE_NAMES,
    resolve_sensitizer,
)
from riskquant.valuation.blocks.block_type import SPOT_BLOCK_TYPE_ID, BlockType


def _rows(sens: BlockSensitivities, uid: int) -> dict[int, float]:
    """first_order by rf_id for one position."""
    mask = sens.position_uids == uid
    return dict(
        zip(
            sens.factor_rf_ids[mask].tolist(),
            sens.first_order[mask].tolist(),
            strict=True,
        )
    )


# =============================================================================
# Spot WITHOUT an FX leg
# =============================================================================


def test_spot_without_fx_first_order_is_quantity(
    rsc, sensitivities, position_uids
) -> None:
    """qty=+100, spot=200, no FX leg -> dPV/dSpot = qty = +100."""
    aapl_uid, _, _ = position_uids
    aapl_rf = int(rsc.rel_name_to_rf_id["AAPL_EQ"])
    rows = _rows(sensitivities, aapl_uid)
    assert rows[aapl_rf] == pytest.approx(100.0, abs=1e-9)


def test_spot_without_fx_has_no_fx_row_and_no_cross_term(
    rsc, sensitivities, position_uids
) -> None:
    """A non-quanto trade must produce exactly one row and zero cross terms."""
    aapl_uid, _, wti_uid = position_uids
    usdeur_rf = int(rsc.rel_name_to_rf_id["USDEUR"])

    for uid in (aapl_uid, wti_uid):
        rows = _rows(sensitivities, uid)
        assert len(rows) == 1, f"uid={uid} should have exactly one sensitivity row"
        assert usdeur_rf not in rows, f"uid={uid} must not carry FX sensitivity"
        assert not (sensitivities.cross_position_uids == uid).any()


def test_spot_second_order_own_factor_is_zero(sensitivities) -> None:
    """The spot payoff is affine/bilinear: all own-factor gammas are exactly 0."""
    assert np.all(sensitivities.second_order == 0.0)


# =============================================================================
# Spot WITH a direct FX leg (quanto / cross-currency)
# =============================================================================


def test_spot_with_fx_produces_commodity_fx_and_cross(
    rsc, sensitivities, position_uids
) -> None:
    """
    qty=-50, spot=82.30, fx=1.08:
        dPV/dSpot      = qty * fx   = -54.00
        dPV/dFX        = qty * spot = -4115.00
        d2PV/dSpot dFX = qty        = -50.00
    """
    _, brent_uid, _ = position_uids
    brent_rf = int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])
    usdeur_rf = int(rsc.rel_name_to_rf_id["USDEUR"])

    rows = _rows(sensitivities, brent_uid)
    assert rows[brent_rf] == pytest.approx(-54.00, abs=1e-4)
    assert rows[usdeur_rf] == pytest.approx(-4115.00, abs=1e-3)

    mask = sensitivities.cross_position_uids == brent_uid
    assert mask.sum() == 1
    assert int(sensitivities.cross_factor_rf_ids_i[mask][0]) == brent_rf
    assert int(sensitivities.cross_factor_rf_ids_j[mask][0]) == usdeur_rf
    assert sensitivities.cross_second_order[mask][0] == pytest.approx(-50.0, abs=1e-9)


def test_wti_sensitivity_is_quantity_and_has_no_fx(
    rsc, sensitivities, position_uids
) -> None:
    """WTI is USD-priced/USD-settled: dPV/dWTI = +75, no FX exposure."""
    _, _, wti_uid = position_uids
    wti_rf = int(rsc.rel_name_to_rf_id["WTI_CRUDE"])
    rows = _rows(sensitivities, wti_uid)
    assert rows[wti_rf] == pytest.approx(75.0, abs=1e-9)
    assert len(rows) == 1


# =============================================================================
# Axis / identity conventions
# =============================================================================


def test_active_factor_axis_is_exactly_the_expected_rf_ids(rsc, sensitivities) -> None:
    expected = {
        int(rsc.rel_name_to_rf_id[n])
        for n in ("USDEUR", "AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE")
    }
    assert set(sensitivities.active_factor_rf_ids.tolist()) == expected


def test_result_dtypes_and_contiguity(sensitivities) -> None:
    """Canonical dtype contract: int64 uids, int32 rf_ids, float64 values."""
    assert sensitivities.position_uids.dtype == np.int64
    assert sensitivities.factor_rf_ids.dtype == np.int32
    assert sensitivities.first_order.dtype == np.float64
    assert sensitivities.second_order.dtype == np.float64
    assert sensitivities.cross_position_uids.dtype == np.int64
    assert sensitivities.cross_factor_rf_ids_i.dtype == np.int32
    assert sensitivities.cross_second_order.dtype == np.float64

    assert sensitivities.first_order.flags["C_CONTIGUOUS"]
    assert sensitivities.factor_rf_ids.flags["C_CONTIGUOUS"]

    # 3 price rows + 1 FX row; exactly 1 cross term.
    assert sensitivities.num_entries == 4
    assert sensitivities.num_cross_entries == 1


def test_factor_ids_are_global_rf_ids_not_block_local_uix(
    rsc, portfolio_mapper_market, sensitivities
) -> None:
    """
    The sensitizer must emit GLOBAL rf_ids. Confirm against the mapper's own
    universe rather than against a name.
    """
    _portfolio, mapper, _market = portfolio_mapper_market
    assert set(sensitivities.active_factor_rf_ids.tolist()).issubset(
        set(mapper.market_id_universe.tolist())
    )


# =============================================================================
# Registry dispatch contract
# =============================================================================


def test_spot_block_resolves_to_a_sensitizer(portfolio_mapper_market) -> None:
    portfolio, _mapper, _market = portfolio_mapper_market
    assert resolve_sensitizer(portfolio.spot) is BLOCK_SENSITIZERS[SPOT_BLOCK_TYPE_ID]


def test_unauthored_block_types_fail_fast_and_are_named(
    portfolio_mapper_market,
) -> None:
    """
    Coverage is incremental: SPOT and VANILLA_FORWARD (D103, WS-C) are
    authored. Unauthored block types must be named explicitly at import and
    must RAISE on resolution -- never silently yield zero risk.
    """
    from riskquant.sensitivities.sensitizers.vanilla_forward import (
        sensitize_vanilla_forward_block,
    )

    assert "SPOT" not in UNREGISTERED_BLOCK_TYPE_NAMES
    assert "VANILLA_FORWARD" not in UNREGISTERED_BLOCK_TYPE_NAMES
    assert "VANILLA_EUROPEAN_OPTION" in UNREGISTERED_BLOCK_TYPE_NAMES

    class _FakeForwardBlock:
        BLOCK_TYPE_ID = int(BlockType.VANILLA_FORWARD)

    class _FakeOptionBlock:
        BLOCK_TYPE_ID = int(BlockType.VANILLA_EUROPEAN_OPTION)

    assert resolve_sensitizer(_FakeForwardBlock()) is sensitize_vanilla_forward_block  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="No sensitizer registered"):
        resolve_sensitizer(_FakeOptionBlock())  # type: ignore[arg-type]


# =============================================================================
# Numerical-core purity
# =============================================================================


def test_numerical_core_is_free_of_polars(sensitivities) -> None:
    """
    Polars must not reach the numerical sensitivity core. The result object
    exposes only NumPy arrays, and neither the engine, the sensitizer, nor
    the kernel imports polars.
    """
    import inspect

    import riskquant.sensitivities.engine as engine_mod
    import riskquant.sensitivities.result as result_mod
    import riskquant.sensitivities.sensitizer_kernels.spot as kernel_mod
    import riskquant.sensitivities.sensitizers.spot as sensitizer_mod

    for mod in (engine_mod, result_mod, kernel_mod, sensitizer_mod):
        assert "polars" not in inspect.getsource(mod), (
            f"{mod.__name__} must not import polars"
        )

    assert isinstance(sensitivities.first_order, np.ndarray)
