from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from riskquant.market_data.curves.extrap_methods import EXTRAP_FLAT
from riskquant.market_data.curves.interp_methods import INTERP_LINEAR
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.market_data.surfaces.y_axes_types import SURFACE_Y_AXIS_TYPE_STRIKE
from riskquant.portfolio.engine import UixMapper
from riskquant.simlab.runner import (
    UniverseMapping,
    build_market_data_matrix_from_snapshot,
)


def _make_universe_map() -> UniverseMapping:
    required_factor_ids = np.array(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
        dtype=np.int32,
    )
    return UniverseMapping(
        required_factor_ids=required_factor_ids,
        required_shift_type_ids=np.zeros(required_factor_ids.shape[0], dtype=np.int8),
        idx_fx_spots=slice(0, 1),
        idx_prices=slice(1, 3),
        idx_fwd_nodes=slice(3, 4),
        idx_vol_nodes=slice(4, 5),
        idx_discount_curves=slice(5, 7),
        shape_discount_curves=(1, 2),
        idx_price_curves=slice(7, 9),
        shape_price_curves=(1, 2),
        idx_vol_surfaces=slice(9, 13),
        shape_vol_surfaces=(1, 2, 2),
        curve_interp_methods=np.array([INTERP_LINEAR], dtype=np.int8),
        curve_extrap_methods=np.array([EXTRAP_FLAT], dtype=np.int8),
        raw_discount_tenors=np.array([[100.0, 200.0]], dtype=np.float64),
        raw_price_curve_tenors=np.array([[100.0, 200.0]], dtype=np.float64),
        surface_interp_x_methods=np.array([INTERP_LINEAR], dtype=np.int8),
        surface_extrap_x_methods=np.array([EXTRAP_FLAT], dtype=np.int8),
        surface_y_axis_type_ids=np.array([SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8),
        raw_vol_surface_tenors=np.array([[100.0, 200.0]], dtype=np.float64),
        raw_vol_surface_strikes=np.array([[90.0, 110.0]], dtype=np.float64),
    )


def _make_uix_mapper() -> UixMapper:
    return UixMapper(
        market_id_universe=np.array([1], dtype=np.int32),
        discount_curve_id_universe=np.array([1], dtype=np.int32),
        date_ord32_universe=np.array([100, 200], dtype=np.int32),
        fx_effective_pairs=np.array([[-1, 0]], dtype=np.int32),
    )


def _make_snapshot(
    *, reverse: bool = False, drop_last: bool = False
) -> RiskFactorLevelSnapshot:
    factor_ids = np.array(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
        dtype=np.int32,
    )
    factor_levels = np.array(
        [
            1.25,
            100.0,
            200.0,
            101.0,
            0.2,
            np.log(0.99),
            np.log(0.97),
            50.0,
            55.0,
            0.20,
            0.25,
            0.22,
            0.27,
        ],
        dtype=np.float64,
    )
    if drop_last:
        factor_ids = factor_ids[:-1]
        factor_levels = factor_levels[:-1]
    if reverse:
        factor_ids = factor_ids[::-1]
        factor_levels = factor_levels[::-1]
    return RiskFactorLevelSnapshot(
        as_of_date=date(2026, 1, 1),
        factor_ids=np.ascontiguousarray(factor_ids),
        factor_levels=np.ascontiguousarray(factor_levels),
    )


def test_build_market_data_matrix_from_snapshot_success() -> None:
    md = build_market_data_matrix_from_snapshot(
        as_of_date=date(2026, 1, 1),
        uix_mapper=_make_uix_mapper(),
        universe_map=_make_universe_map(),
        snapshot=_make_snapshot(),
        num_scenarios=1,
    )

    assert md.num_scenarios == 1
    assert md.fx_spots_effective.shape == (1, 1)
    assert md.prices.shape == (2, 1)
    assert md.forward_nodes.shape == (1, 1)
    assert md.vol_nodes.shape == (1, 1)
    assert md.discount_factor_grid.shape == (1, 2, 1)
    assert md.price_curve_grid.shape == (1, 2, 1)
    assert md.vol_surface_strikes.shape == (1, 2)
    assert md.vol_surfaces.shape == (1, 2, 1, 2)
    assert md.tte_grid.shape == (2,)

    np.testing.assert_allclose(md.fx_spots_effective[:, 0], np.array([1.0]))
    np.testing.assert_allclose(md.prices[:, 0], np.array([100.0, 200.0]))
    np.testing.assert_allclose(md.forward_nodes[:, 0], np.array([101.0]))
    np.testing.assert_allclose(md.vol_nodes[:, 0], np.array([0.2]))
    np.testing.assert_allclose(
        md.discount_factor_grid[0, :, 0], np.array([0.99, 0.97]), atol=1e-12
    )
    np.testing.assert_allclose(md.price_curve_grid[0, :, 0], np.array([50.0, 55.0]))
    np.testing.assert_allclose(md.vol_surfaces[0, 0, 0, :], np.array([0.20, 0.25]))
    np.testing.assert_allclose(md.vol_surfaces[0, 1, 0, :], np.array([0.22, 0.27]))


def test_build_market_data_matrix_from_snapshot_unsorted_snapshot() -> None:
    md = build_market_data_matrix_from_snapshot(
        as_of_date=date(2026, 1, 1),
        uix_mapper=_make_uix_mapper(),
        universe_map=_make_universe_map(),
        snapshot=_make_snapshot(reverse=True),
        num_scenarios=1,
    )

    np.testing.assert_allclose(md.prices[:, 0], np.array([100.0, 200.0]))


def test_build_market_data_matrix_from_snapshot_missing_factor() -> None:
    with pytest.raises(ValueError, match="missing from the snapshot"):
        build_market_data_matrix_from_snapshot(
            as_of_date=date(2026, 1, 1),
            uix_mapper=_make_uix_mapper(),
            universe_map=_make_universe_map(),
            snapshot=_make_snapshot(drop_last=True),
            num_scenarios=1,
        )
