# tests/sensitivities/test_scenario_pnl.py
"""
Production tests for the deterministic scenario -> proxy -> sensitivity-P&L path.

Exercises the existing production components composed by the E2E demo:

    build_universe_mapping
    extract_levels_from_snapshot
    resolve_required_factor_ids_to_market_data_factor_ids
    build_shock_universe
    apply_shocks_to_base_levels
    FactorAxis.levels -> SensitivityPnlCalculator.run

No scenario adapter or scenario engine exists or is needed: every step above
is already a public production entry point.

The oracle used throughout is derived from the ECONOMIC identities of the
instruments, never from the production sensitivity coefficients and never by
re-running the production algorithm.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.proxy_resolver.proxy import (
    RES_DIRECT,
    RES_ID_PROXY,
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.sensitivities.book_adapter import build_sensitivity_book_spec
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.shocks.apply import (
    apply_shocks_to_base_levels,
    validate_apply_shocks_buffers,
)
from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
)
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.utils.dates import date_to_ord32

# Source factors that carry market data. WTI is deliberately absent so the
# proxy path is forced rather than incidental.
_SOURCE_NAMES = ("USDEUR", "AAPL_EQ", "BRENT_CRUDE")

# (S, K_source) log returns.
_SHOCKS = (
    (0.000, 0.010, -0.020),
    (-0.010, 0.000, 0.015),
    (0.020, -0.015, -0.010),
)

_PROXY_DF = pl.DataFrame({
    "proxying_level": ["RISK_FACTOR"],
    "required_name": ["WTI_CRUDE"],
    "proxy_name": ["BRENT_CRUDE"],
    "vol_scaling_factor": [0.95],
})


# =============================================================================
# Fixtures: the production pipeline, assembled once
# =============================================================================


@pytest.fixture(scope="module")
def proxy_topology(rsc):
    return compile_proxy_topology_from_df(df=_PROXY_DF, static_config=rsc)


@pytest.fixture(scope="module")
def snapshot(rsc, as_of_date, golden_levels):
    return build_risk_factor_level_snapshot_from_dict(
        as_of_date=as_of_date,
        levels_dict=golden_levels,
        static_config=rsc,
    )


@pytest.fixture(scope="module")
def portfolio_and_mapper(rsc, raw_positions, as_of_date):
    batch = compile_raw_df_into_ingest_batch(raw_df=raw_positions, static_config=rsc)
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(as_of_date)))
    return compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    ), mapper


@pytest.fixture(scope="module")
def raw_shocks() -> np.ndarray:
    return np.asfortranarray(np.array(_SHOCKS, dtype=np.float32))


@pytest.fixture(scope="module")
def resolved(rsc, proxy_topology, snapshot, portfolio_and_mapper, raw_shocks):
    """Run the full production scenario path once."""
    _portfolio, mapper = portfolio_and_mapper
    universe_map = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    required = universe_map.required_factor_ids
    base_levels = extract_levels_from_snapshot(required_ids=required, snapshot=snapshot)

    available = np.ascontiguousarray(
        [rsc.rel_name_to_rf_id[n] for n in _SOURCE_NAMES], dtype=np.int32
    )
    proxy_resolution = resolve_required_factor_ids_to_market_data_factor_ids(
        static_config=rsc,
        proxy_topology=proxy_topology,
        required_factor_ids=required,
        market_data_factor_ids=available,
    )
    shock_universe = build_shock_universe(
        required_shift_type_id=universe_map.required_shift_type_ids,
        source_factor_ids=proxy_resolution.source_factor_ids,
        available_factor_ids=available,
    )
    batch = ScenarioSetBatch(
        shocks=raw_shocks,
        required_to_shock_uix=shock_universe.required_to_shock_uix,
        shock_scale=proxy_resolution.shock_scale,
        required_shift_type_id=universe_map.required_shift_type_ids,
        scenario_offset=0,
    )
    shocked = np.empty((required.shape[0], len(_SHOCKS)), dtype=np.float64, order="C")
    validate_apply_shocks_buffers(
        base_levels=base_levels, shocks=raw_shocks, out=shocked
    )
    apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=shocked)
    return {
        "required": required,
        "base_levels": base_levels,
        "shocked": shocked,
        "proxy_resolution": proxy_resolution,
        "shock_universe": shock_universe,
        "row_of": {int(rf): i for i, rf in enumerate(required)},
    }


@pytest.fixture(scope="module")
def pnl(rsc, portfolio_and_mapper, snapshot, resolved, as_of_date):
    portfolio, mapper = portfolio_and_mapper
    market_data = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=as_of_date,
        uix_mapper=mapper,
        universe_map=build_universe_mapping(static_config=rsc, uix_mapper=mapper),
        snapshot=snapshot,
        num_scenarios=1,
    )
    sens = run_sensitivities(
        portfolio=portfolio, uix_mapper=mapper, market_data=market_data
    )
    calculator = SensitivityPnlCalculator.compile(
        build_sensitivity_book_spec(sensitivities=sens)
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
# Scenario construction conventions
# =============================================================================


def test_scenario_matrix_dtype_and_contiguity(raw_shocks) -> None:
    """`ScenarioSetBatch.shocks` is (S, K) float32 and MUST be F-contiguous."""
    assert raw_shocks.dtype == np.float32
    assert raw_shocks.shape == (3, 3)
    assert raw_shocks.flags["F_CONTIGUOUS"]


def test_shocked_levels_layout(resolved) -> None:
    """The levels buffer is (N_req, S) float64 C-contiguous."""
    shocked = resolved["shocked"]
    assert shocked.dtype == np.float64
    assert shocked.shape == (4, 3)
    assert shocked.flags["C_CONTIGUOUS"]


def test_required_factor_axis_ordering(rsc, resolved) -> None:
    """Factor rows are the required rf_ids, ascending, and include WTI."""
    required = resolved["required"].tolist()
    assert required == sorted(required)
    assert set(required) == {
        int(rsc.rel_name_to_rf_id[n])
        for n in ("USDEUR", "AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE")
    }


# =============================================================================
# Proxy application
# =============================================================================


def test_wti_resolves_to_brent_with_095_scale(rsc, resolved) -> None:
    res = resolved["proxy_resolution"]
    i = resolved["row_of"][int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]
    assert int(res.source_factor_ids[i]) == int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])
    assert float(res.shock_scale[i]) == pytest.approx(0.95, abs=1e-7)
    assert int(res.resolution_code[i]) == int(RES_ID_PROXY)


def test_unproxied_factors_resolve_directly(rsc, resolved) -> None:
    res = resolved["proxy_resolution"]
    for name in ("USDEUR", "AAPL_EQ", "BRENT_CRUDE"):
        i = resolved["row_of"][int(rsc.rel_name_to_rf_id[name])]
        assert int(res.source_factor_ids[i]) == int(rsc.rel_name_to_rf_id[name])
        assert float(res.shock_scale[i]) == pytest.approx(1.0, abs=1e-7)
        assert int(res.resolution_code[i]) == int(RES_DIRECT)


def test_wti_shock_column_is_brents(rsc, resolved) -> None:
    """WTI must be routed to BRENT's shock column, not its own."""
    su = resolved["shock_universe"]
    wti_i = resolved["row_of"][int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]
    brent_i = resolved["row_of"][int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]
    assert su.required_to_shock_uix[wti_i] == su.required_to_shock_uix[brent_i]


def test_base_levels_are_not_altered_by_proxying(rsc, resolved) -> None:
    """Proxying changes shock provenance ONLY. Base levels are untouched."""
    base = resolved["base_levels"]
    row = resolved["row_of"]
    assert float(base[row[int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]]) == pytest.approx(
        70.00, abs=1e-4
    )
    assert float(base[row[int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]]) == pytest.approx(
        82.30, abs=1e-4
    )
    assert float(base[row[int(rsc.rel_name_to_rf_id["USDEUR"])]]) == pytest.approx(
        1.0800, abs=1e-6
    )


# =============================================================================
# LOG_RETURN semantics
# =============================================================================


def test_log_return_is_exponential_not_additive(rsc, resolved, raw_shocks) -> None:
    """
    level = base * exp(shock), NOT base * (1 + shock).

    At a 2% return the two differ by ~0.02% of the level -- far above the
    tolerance here, so an additive slip cannot pass.
    """
    row = resolved["row_of"]
    brent_i = row[int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]
    base = float(resolved["base_levels"][brent_i])

    for s in range(len(_SHOCKS)):
        shock = float(raw_shocks[s, _SOURCE_NAMES.index("BRENT_CRUDE")])
        produced = float(resolved["shocked"][brent_i, s])
        assert produced == pytest.approx(base * math.exp(shock), abs=1e-5)
        if abs(shock) > 1e-9:
            assert abs(produced - base * (1.0 + shock)) > 1e-4


def test_proxy_scale_applies_inside_the_exponent(rsc, resolved, raw_shocks) -> None:
    """WTI level = 70 * exp(0.95 * brent_shock), not 0.95 * (a Brent level)."""
    row = resolved["row_of"]
    wti_i = row[int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]
    for s in range(len(_SHOCKS)):
        brent_shock = float(raw_shocks[s, _SOURCE_NAMES.index("BRENT_CRUDE")])
        assert float(resolved["shocked"][wti_i, s]) == pytest.approx(
            70.00 * math.exp(0.95 * brent_shock), abs=1e-5
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


# =============================================================================
# dtype seam: ProxyResolution.shock_scale (float32) -> kernel (float64)
# =============================================================================


def test_float32_proxy_shock_scale_is_accepted_by_shock_application() -> None:
    """
    `ProxyResolution.shock_scale` is float32 by contract (proxy_audit enforces
    it), while `_kernel_apply_shocks` is dtype-pinned to a float64 scale. The
    promotion happens at the execution boundary in `apply_shocks_to_base_levels`.

    Before that promotion existed the production path raised TypeError, so this
    regression test stays even if other tests happen to cover the path.
    """
    base = np.array([70.0, 82.30], dtype=np.float64)
    shocks = np.asfortranarray(
        np.array([[0.02, 0.01], [0.03, -0.01], [0.00, 0.02]], dtype=np.float32)
    )
    for dtype in (np.float32, np.float64):
        batch = ScenarioSetBatch(
            shocks=shocks,
            required_to_shock_uix=np.array([0, 1], dtype=np.int32),
            shock_scale=np.array([0.95, 1.0], dtype=dtype),
            required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * 2, dtype=np.int8),
            scenario_offset=0,
        )
        out = np.empty((2, 3), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(base_levels=base, batch=batch, out=out)

        # Independent arithmetic check, not a comparison against the kernel.
        # Tolerance is set by the float32 shock operand (~1e-7 relative), not
        # loosened to hide an error: an additive or unscaled slip would be
        # ~1e-2 on these levels, four orders of magnitude larger.
        for s in range(3):
            assert out[0, s] == pytest.approx(
                70.0 * math.exp(0.95 * float(shocks[s, 0])), rel=1e-6
            )
            assert out[1, s] == pytest.approx(
                82.30 * math.exp(1.0 * float(shocks[s, 1])), rel=1e-6
            )


def test_proxy_resolution_shock_scale_remains_float32(resolved) -> None:
    """The canonical ProxyResolution dtype must NOT have been widened."""
    assert resolved["proxy_resolution"].shock_scale.dtype == np.float32
