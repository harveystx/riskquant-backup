# tests/credit/exposure/test_compile_exposure.py
"""
Test suite for `compile_exposure` and its integration with `run_exposure`.

Layout tests exercise the uid-join and CSR banding directly (no kernel).
Integration tests synthesize an MtM matrix in `CompiledPortfolio.trade_uids`
row order and drive the full RC path: compile_exposure -> run_exposure.

The minimal fixture deliberately scrambles `portfolio.trade_uids` relative to
`batch.uids` so the uid-join is proven order-independent — a positional carry
would silently pass the sorted case and fail here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from riskquant.constants import NO_NETTING_SET_ID
from riskquant.credit.exposure.compile import (
    _join_netting_set_ids,
    _resolve_collateral,
    compile_exposure_rc,
)
from riskquant.credit.exposure.compiled import CompiledExposure
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.exposure.models import REPLACEMENT_COST_ID
from riskquant.credit.exposure.results import NettingSetExposurePaths
from riskquant.credit.exposure.views.replacement_cost import RcNettingSetView
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.valuation.blocks.spot_block import SpotBlock

_SENTINEL = int(NO_NETTING_SET_ID)

# Batch uid axis is sorted; portfolio uid axis is scrambled. Any uid-join bug
# that reads by position rather than by key fails the fixture-backed tests.
_BATCH_UIDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
_PORTFOLIO_UIDS = [1006, 1008, 1001, 1004, 1003, 1002, 1007, 1005]

# Ground-truth netting-set linkage. 1007/1008 carry no linkage.
_NS_OF_UID: dict[int, int | None] = {
    1001: 101,
    1002: 101,
    1003: 101,
    1004: 102,
    1005: 102,
    1006: 103,
    1007: None,
    1008: None,
}

# Per-uid MtM used by the integration tests.
_MTM_OF_UID: dict[int, float] = {
    1001: 100.0,
    1002: 50.0,
    1003: 30.0,
    1004: 20.0,
    1005: -90.0,
    1006: 500.0,
    1007: 7.0,
    1008: 8.0,
}


# ==============================================================================
# BUILDERS
# ==============================================================================


def _make_batch(*, uids, netting_set_id) -> IngestBatch:
    """Minimal valid `IngestBatch` carrying only uids + netting-set linkage."""
    uids = np.ascontiguousarray(uids, dtype=np.int64)
    netting_set_id = np.ascontiguousarray(netting_set_id, dtype=np.int64)
    n = uids.shape[0]

    i32 = lambda: np.zeros(n, dtype=np.int32)  # noqa: E731
    f64 = lambda: np.zeros(n, dtype=np.float64)  # noqa: E731

    return IngestBatch(
        size=n,
        instrument_ids=i32(),
        uids=uids,
        qty=f64(),
        strike_price=f64(),
        cp_mult=np.zeros(n, dtype=np.int8),
        settle_ord=i32(),
        delivery_ord=i32(),
        expiry_ord=i32(),
        unique_dates_ord=np.empty(0, dtype=np.int32),
        settle_date_uix=i32(),
        delivery_date_uix=i32(),
        expiry_date_uix=i32(),
        fx_under_to_settle_ids=i32(),
        fx_under_to_settle_invert=i32(),
        fx_strike_to_settle_ids=i32(),
        fx_strike_to_settle_invert=i32(),
        price_rf_ids=i32(),
        price_curve_ids=i32(),
        vol_rf_ids=i32(),
        vol_surface_rf_ids=i32(),
        under_discount_curve_ids=i32(),
        strike_discount_curve_ids=i32(),
        vol_surface_y_axis_type_ids=np.zeros(n, dtype=np.int8),
        ead=f64(),
        pd=f64(),
        lgd=f64(),
        beta=f64(),
        systemic_uix=i32(),
        idiosyncratic_uix=i32(),
        chi_squared_uix=i32(),
        netting_set_id=netting_set_id,
    )


def _make_portfolio(*, trade_uids) -> CompiledPortfolio:
    """Minimal `CompiledPortfolio`: one `SpotBlock` in the given uid order."""
    trade_uids = np.ascontiguousarray(trade_uids, dtype=np.int64)
    n = trade_uids.shape[0]
    block = SpotBlock(
        trade_uids=trade_uids,
        qty=np.zeros(n, dtype=np.float64),
        spot_price_uix=np.zeros(n, dtype=np.int32),
        fx_uix=np.zeros(n, dtype=np.int32),
        fx_invert=np.zeros(n, dtype=np.bool_),
    )
    return CompiledPortfolio(spot=block)


def _ns_array(uids) -> np.ndarray:
    """Ground-truth netting-set id array aligned to `uids` (None -> sentinel)."""
    return np.array(
        [
            _NS_OF_UID[int(u)] if _NS_OF_UID[int(u)] is not None else _SENTINEL
            for u in uids
        ],
        dtype=np.int64,
    )


def _make_mtm(portfolio, *, n_scenarios, scale_by_scenario=False) -> np.ndarray:
    """(M, S) MtM aligned to `portfolio.trade_uids`; row value = `_MTM_OF_UID`."""
    base = np.array(
        [_MTM_OF_UID[int(u)] for u in portfolio.trade_uids], dtype=np.float64
    )
    mtm = np.empty((base.shape[0], n_scenarios), dtype=np.float64)
    for s in range(n_scenarios):
        mtm[:, s] = base * (s + 1) if scale_by_scenario else base
    return np.ascontiguousarray(mtm)


def _band_netting_sets(view, portfolio) -> list[set[int | None]]:
    """Per band, the ground-truth netting-set ids of its member trades."""
    banded = portfolio.trade_uids[view.mtm_row_perm]
    off = view.netting_set_offsets
    return [
        {_NS_OF_UID[int(u)] for u in banded[off[b] : off[b + 1]]}
        for b in range(view.num_netting_sets)
    ]


@pytest.fixture
def minimal_case() -> SimpleNamespace:
    """Scrambled 8-trade portfolio: 3 netting sets + 2 unlinked trades."""
    return SimpleNamespace(
        portfolio=_make_portfolio(trade_uids=_PORTFOLIO_UIDS),
        batch=_make_batch(uids=_BATCH_UIDS, netting_set_id=_ns_array(_BATCH_UIDS)),
    )


# ==============================================================================
# UID JOIN
# ==============================================================================


class TestJoinNettingSetIds:
    def test_join_follows_uid_not_position(self, minimal_case):
        ns = _join_netting_set_ids(minimal_case.portfolio, minimal_case.batch)
        expected = _ns_array(_PORTFOLIO_UIDS)  # aligned to portfolio order
        np.testing.assert_array_equal(ns, expected)

    def test_raises_when_portfolio_uid_absent_from_batch(self):
        batch = _make_batch(uids=[1, 2, 3], netting_set_id=[10, 10, 20])
        portfolio = _make_portfolio(trade_uids=[1, 2, 9999])
        with pytest.raises(ValueError, match="absent from the ingest batch"):
            compile_exposure_rc(portfolio=portfolio, batch=batch)


# ==============================================================================
# COLLATERAL RESOLUTION
# ==============================================================================


class TestResolveCollateral:
    def test_none_yields_zeros(self):
        out = _resolve_collateral(np.array([101, 102, 103], dtype=np.int64), None)
        np.testing.assert_array_equal(out, np.zeros(3))
        assert out.dtype == np.float64

    def test_empty_mapping_yields_zeros(self):
        out = _resolve_collateral(np.array([101, 102], dtype=np.int64), {})
        np.testing.assert_array_equal(out, np.zeros(2))

    def test_partial_mapping_zero_fills_missing(self):
        out = _resolve_collateral(np.array([101, 102, 103], dtype=np.int64), {102: 5.0})
        np.testing.assert_array_equal(out, [0.0, 5.0, 0.0])

    def test_band_order_independent_of_dict_order(self):
        out = _resolve_collateral(
            np.array([101, 102, 103], dtype=np.int64), {103: 9.0, 101: 1.0}
        )
        np.testing.assert_array_equal(out, [1.0, 0.0, 9.0])

    def test_unknown_keys_ignored(self):
        out = _resolve_collateral(np.array([101], dtype=np.int64), {101: 1.0, 999: 9.0})
        np.testing.assert_array_equal(out, [1.0])

    def test_negative_collateral_raises(self):
        with pytest.raises(ValueError, match="negative collateral"):
            _resolve_collateral(np.array([101], dtype=np.int64), {101: -1.0})


# ==============================================================================
# COMPILE: CSR BAND LAYOUT
# ==============================================================================


class TestCompileExposureLayout:
    def test_returns_single_rc_view(self, minimal_case):
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        )
        assert isinstance(exposure, CompiledExposure)
        assert exposure.total_views == 1
        view = exposure.views[0]
        assert isinstance(view, RcNettingSetView)
        assert view.EXPOSURE_MODEL_ID == REPLACEMENT_COST_ID

    def test_netting_set_uids_and_offsets(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        np.testing.assert_array_equal(view.netting_set_uids, [101, 102, 103])
        np.testing.assert_array_equal(view.netting_set_offsets, [0, 3, 5, 6])
        assert view.num_netting_sets == 3
        assert view.num_trades == 6

    def test_mtm_row_perm_exact_permutation(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        # Deterministic given the fixture: stable group-by on the joined ids.
        np.testing.assert_array_equal(view.mtm_row_perm, [2, 4, 5, 3, 7, 0])

    def test_perm_selects_only_linked_trades(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        banded = {int(u) for u in minimal_case.portfolio.trade_uids[view.mtm_row_perm]}
        assert banded == {1001, 1002, 1003, 1004, 1005, 1006}
        assert 1007 not in banded and 1008 not in banded

    def test_bands_group_correct_netting_sets(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        assert _band_netting_sets(view, minimal_case.portfolio) == [
            {101},
            {102},
            {103},
        ]

    def test_within_band_order_is_stable(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        off = view.netting_set_offsets
        for b in range(view.num_netting_sets):
            band = view.mtm_row_perm[off[b] : off[b + 1]]
            assert np.all(np.diff(band) > 0), "band rows must stay in portfolio order"

    def test_view_array_dtypes_and_contiguity(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        assert view.mtm_row_perm.dtype == np.int32
        assert view.netting_set_offsets.dtype == np.int32
        assert view.netting_set_uids.dtype == np.int64
        assert view.collateral.dtype == np.float64
        for arr in (
            view.mtm_row_perm,
            view.netting_set_offsets,
            view.netting_set_uids,
            view.collateral,
        ):
            assert arr.flags.c_contiguous

    def test_collateral_resolved_into_band_order(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio,
            batch=minimal_case.batch,
            collateral={103: 1_000_000.0, 101: 250.0},
        ).views[0]
        np.testing.assert_array_equal(view.collateral, [250.0, 0.0, 1_000_000.0])

    def test_collateral_none_yields_zero_band(self, minimal_case):
        view = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        ).views[0]
        np.testing.assert_array_equal(view.collateral, np.zeros(3))

    def test_compiled_exposure_aggregate(self, minimal_case):
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        )
        np.testing.assert_array_equal(exposure.netting_set_uids, [101, 102, 103])
        np.testing.assert_array_equal(exposure.view_offsets, [0])
        assert exposure.total_netting_sets == 3


# ==============================================================================
# COMPILE: ERROR PATHS
# ==============================================================================


class TestCompileExposureErrors:
    def test_raises_when_no_trade_carries_a_netting_set(self):
        batch = _make_batch(uids=[1, 2], netting_set_id=[_SENTINEL, _SENTINEL])
        portfolio = _make_portfolio(trade_uids=[1, 2])
        with pytest.raises(ValueError, match="no priced trade carries"):
            compile_exposure_rc(portfolio=portfolio, batch=batch)

    def test_negative_collateral_rejected(self, minimal_case):
        with pytest.raises(ValueError, match="negative collateral"):
            compile_exposure_rc(
                portfolio=minimal_case.portfolio,
                batch=minimal_case.batch,
                collateral={101: -1.0},
            )


# ==============================================================================
# INTEGRATION: compile_exposure -> run_exposure
# ==============================================================================


class TestRunExposureIntegration:
    def test_result_shape_and_join_key(self, minimal_case):
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        )
        mtm = _make_mtm(minimal_case.portfolio, n_scenarios=4)
        result = run_exposure(exposure=exposure, mtm=mtm)

        assert isinstance(result, NettingSetExposurePaths)
        np.testing.assert_array_equal(result.uid, [101, 102, 103])
        assert result.exposure.shape == (3, 4)
        assert result.exposure.dtype == np.float64
        assert result.exposure.flags.c_contiguous

    def test_rc_floor_collateral_and_negative_netting(self, minimal_case):
        # V: 101 -> 180, 102 -> -70, 103 -> 500.  C: 101=80, 103 over-collateralized.
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio,
            batch=minimal_case.batch,
            collateral={101: 80.0, 102: 0.0, 103: 1_000_000.0},
        )
        mtm = _make_mtm(minimal_case.portfolio, n_scenarios=2)
        ead = run_exposure(exposure=exposure, mtm=mtm).exposure

        np.testing.assert_allclose(ead[0], [100.0, 100.0])  # 101: max(180-80,0)
        np.testing.assert_allclose(ead[1], [0.0, 0.0])  # 102: net short -> 0
        np.testing.assert_allclose(ead[2], [0.0, 0.0])  # 103: over-collateralized

    def test_rc_without_collateral_is_max_v_zero(self, minimal_case):
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio, batch=minimal_case.batch
        )
        mtm = _make_mtm(minimal_case.portfolio, n_scenarios=1)
        ead = run_exposure(exposure=exposure, mtm=mtm).exposure

        np.testing.assert_allclose(ead[:, 0], [180.0, 0.0, 500.0])

    def test_rc_floor_crosses_within_scenarios(self, minimal_case):
        # 101 V_s = 180*(s+1) = 180, 360, 540;  C = 400 -> RC = 0, 0, 140.
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio,
            batch=minimal_case.batch,
            collateral={101: 400.0},
        )
        mtm = _make_mtm(minimal_case.portfolio, n_scenarios=3, scale_by_scenario=True)
        ead = run_exposure(exposure=exposure, mtm=mtm).exposure

        np.testing.assert_allclose(ead[0], [0.0, 0.0, 140.0])

    def test_rc_floor_is_strict_at_v_equals_c(self, minimal_case):
        # 103 single-trade set, V = 500. C = 500 -> RC = 0; C = 499 -> RC = 1.
        mtm = _make_mtm(minimal_case.portfolio, n_scenarios=1)

        ead_eq = run_exposure(
            exposure=compile_exposure_rc(
                portfolio=minimal_case.portfolio,
                batch=minimal_case.batch,
                collateral={103: 500.0},
            ),
            mtm=mtm,
        ).exposure
        assert ead_eq[2, 0] == 0.0

        ead_below = run_exposure(
            exposure=compile_exposure_rc(
                portfolio=minimal_case.portfolio,
                batch=minimal_case.batch,
                collateral={103: 499.0},
            ),
            mtm=mtm,
        ).exposure
        np.testing.assert_allclose(ead_below[2, 0], 1.0)

    def test_ead_per_scenario_independent(self, minimal_case):
        # Scenario-scaled V proves the kernel sums per-scenario, not per-set-total.
        exposure = compile_exposure_rc(
            portfolio=minimal_case.portfolio,
            batch=minimal_case.batch,
            collateral={101: 80.0},
        )
        mtm = _make_mtm(minimal_case.portfolio, n_scenarios=3, scale_by_scenario=True)
        ead = run_exposure(exposure=exposure, mtm=mtm).exposure

        np.testing.assert_allclose(ead[0], [100.0, 280.0, 460.0])  # 180*(s+1)-80


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
