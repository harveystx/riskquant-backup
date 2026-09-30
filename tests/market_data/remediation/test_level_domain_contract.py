# tests/market_data/remediation/test_level_domain_contract.py
"""
AUD-05: every market-data producer route enforces the level-domain contract.

    all levels                 finite
    FX spot / VOL levels       > 0      (rf_id -> quote type, never the value)
    PRICE/RATE/SPREAD/INDEX    any finite value

Routes exercised: flat levels, compiled snapshot, direct snapshot constructor,
`RiskFactorMatrix.get_snapshot`, and the builder itself. Invalid levels must
die before any dense array is built; valid levels must still reconcile to the
independent oracle, with D059 mutations proving consumed rows stay connected.
"""

from __future__ import annotations

import dataclasses

import hostile_fixture as hf
import md_oracle as oracle
import numpy as np
import pandas as pd
import polars as pl
import pytest

import riskquant.market_data.builder as builder_mod
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.risk_factors.builder_matrix import (
    build_risk_factor_matrix_from_df,
)
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
    strictly_positive_level_mask,
)
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.portfolio.engine import build_uix_mapper
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.utils.dates import date_to_ord32

AS_OF_ORD = int(date_to_ord32(hf.AS_OF))
RTOL = 1e-13
DOMAIN_ERROR = r"\(AUD-05, D070\)"


@pytest.fixture(scope="module")
def rsc():
    dfs = hf.static_config_dfs()
    extra = pl.DataFrame(
        [
            dict(
                node_id=19,
                node_name="IDX_USD",
                quote_ccy="USD",
                asset_class="EQUITY",
                asset_subclass="EQ_INDEX",
                quote_type="INDEX",
                unit_scale="DECIMAL",
                shift_type="LOG_RETURN",
                fx_base_ccy=None,
            ),
            dict(
                node_id=23,
                node_name="SPRD_EUR",
                quote_ccy="EUR",
                asset_class="CREDIT",
                asset_subclass="CR_SINGLE_NAME",
                quote_type="SPREAD",
                unit_scale="DECIMAL",
                shift_type="DIFFERENCE",
                fx_base_ccy=None,
            ),
        ],
        schema=dfs["node_meta"].schema,
    )
    dfs["node_meta"] = dfs["node_meta"].vstack(extra)
    return compile_risk_static_config_from_dict_of_dfs(dfs)


BASE = {
    "EQ_GBP": 148.9,
    "PWR_SPREAD": -7.25,
    "EURGBP": 0.8561,
    "VOL_EQ_USD": 0.183,
    "IDX_USD": -12.0,
    "SPRD_EUR": -0.004,
}


@pytest.fixture(scope="module")
def universe(rsc):
    """Spot book with direct PRICE, INDEX, legal negative PRICE; VOL and SPREAD rfs carried via bypass."""
    book = pd.DataFrame([
        dict(
            uid=1,
            instrument="SPOT",
            quantity=10.0,
            underlying="EQ_GBP",
            settlement_ccy="EUR",
        ),
        dict(
            uid=2,
            instrument="SPOT",
            quantity=5.0,
            underlying="PWR_SPREAD",
            settlement_ccy="EUR",
        ),
        dict(
            uid=3,
            instrument="SPOT",
            quantity=2.0,
            underlying="IDX_USD",
            settlement_ccy="USD",
        ),
    ])
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(book), static_config=rsc
    )
    vol = batch.volatility_ids.copy()
    vol[0] = rsc.rel_name_to_rf_id["VOL_EQ_USD"]
    vol[1] = rsc.rel_name_to_rf_id["SPRD_EUR"]
    batch = dataclasses.replace(batch, volatility_ids=np.ascontiguousarray(vol))
    mapper = build_uix_mapper([batch], as_of_ord32=AS_OF_ORD)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    order = [names[int(i)] for i in um.required_factor_ids]
    return mapper, um, order


def _flat(order, overrides, s=1):
    lv = dict(BASE)
    lv.update(overrides)
    return np.ascontiguousarray(
        np.tile(np.array([[lv[n]] for n in order], dtype=np.float64), (1, s))
    )


def test_domain_flags_come_from_quote_type_not_value(rsc, universe):
    _, um, _ = universe
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    scalar_ids = um.required_factor_ids[um.idx_scalar_nodes]
    flags = dict(
        zip(
            [names[int(i)] for i in scalar_ids],
            um.scalar_node_strictly_positive.tolist(),
            strict=True,
        )
    )
    assert flags == {
        "VOL_EQ_USD": True,
        "EQ_GBP": False,
        "SPRD_EUR": False,
        "PWR_SPREAD": False,
        "IDX_USD": False,
    }
    ids = np.array(
        [
            rsc.rel_name_to_rf_id[n]
            for n in ("EURGBP", "VOL_EQ_USD", "EQ_GBP", "USD_OIS[3M]")
        ],
        np.int32,
    )
    assert strictly_positive_level_mask(rf_ids=ids, static_config=rsc).tolist() == [
        True,
        True,
        False,
        False,
    ]


LEGAL = [
    ("PRICE>0", {}),
    ("PRICE=0", {"EQ_GBP": 0.0}),
    ("PRICE<0", {"PWR_SPREAD": -99.0}),
    ("SPREAD<0", {"SPRD_EUR": -0.02}),
    ("INDEX<0", {"IDX_USD": -3.0}),
    ("FX>0", {"EURGBP": 0.5}),
    ("VOL>0", {"VOL_EQ_USD": 0.01}),
]
ILLEGAL = [
    ("FX=0", {"EURGBP": 0.0}),
    ("FX<0", {"EURGBP": -0.91}),
    ("VOL=0", {"VOL_EQ_USD": 0.0}),
    ("VOL<0", {"VOL_EQ_USD": -0.2}),
    ("PRICE NaN", {"EQ_GBP": np.nan}),
    ("PRICE +inf", {"EQ_GBP": np.inf}),
    ("PRICE -inf", {"EQ_GBP": -np.inf}),
    ("FX NaN", {"EURGBP": np.nan}),
    ("FX +inf", {"EURGBP": np.inf}),
    ("FX -inf", {"EURGBP": -np.inf}),
    ("VOL NaN", {"VOL_EQ_USD": np.nan}),
    ("VOL +inf", {"VOL_EQ_USD": np.inf}),
    ("VOL -inf", {"VOL_EQ_USD": -np.inf}),
]


@pytest.mark.parametrize(("label", "overrides"), LEGAL)
def test_legal_levels_are_accepted_on_the_flat_route(universe, label, overrides):
    mapper, um, order = universe
    mdm = build_market_data_matrix_from_flat_levels(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=um,
        flat_levels=_flat(order, overrides),
        num_scenarios=1,
    )
    assert np.isfinite(mdm.prices).all() and np.isfinite(mdm.fx_spots_effective).all()


@pytest.fixture
def spy(monkeypatch):
    calls = []
    for name in (
        "_build_scalar_node_levels",
        "_build_fx_spots_effective",
        "MarketDataMatrix",
    ):
        real = getattr(builder_mod, name)

        def wrapped(*a, _real=real, _name=name, **k):
            calls.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(builder_mod, name, wrapped)
    return calls


@pytest.mark.parametrize("s", [1, 4096])
@pytest.mark.parametrize(("label", "overrides"), ILLEGAL)
def test_illegal_levels_die_before_any_state_is_built_on_the_flat_route(
    universe, spy, label, overrides, s
):
    mapper, um, order = universe
    flat = _flat(order, overrides, s)
    junk = [np.full(flat.shape, 7.77e11) for _ in range(16)]
    del junk
    with pytest.raises(ValueError, match=DOMAIN_ERROR):
        build_market_data_matrix_from_flat_levels(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            flat_levels=flat,
            num_scenarios=s,
        )
    assert spy == []  # no dense array, no effective FX, no matrix


def test_one_bad_scenario_among_many_is_rejected(universe, spy):
    mapper, um, order = universe
    flat = _flat(order, {}, 4096)
    flat[order.index("EURGBP"), 4095] = -1e-300
    with pytest.raises(ValueError, match="FX spot levels must be strictly positive"):
        build_market_data_matrix_from_flat_levels(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            flat_levels=flat,
            num_scenarios=4096,
        )
    assert spy == []


@pytest.mark.parametrize(
    ("label", "overrides"),
    [x for x in ILLEGAL if "NaN" not in x[0] and "inf" not in x[0]],
)
def test_direct_snapshot_and_history_snapshot_routes_cannot_bypass_the_domain(
    rsc, universe, spy, label, overrides
):
    mapper, um, order = universe
    lv = dict(BASE)
    lv.update(overrides)
    snap = RiskFactorLevelSnapshot(
        as_of_date=hf.AS_OF,
        factor_ids=um.required_factor_ids.copy(),
        factor_levels=np.array([lv[n] for n in order], dtype=np.float64),
    )
    with pytest.raises(ValueError, match=DOMAIN_ERROR):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snap,
        )
    # History frames carry DECLARED source units (VOL_EQ_USD is PERCENT).
    source = {n: lv[n] * (100.0 if n == "VOL_EQ_USD" else 1.0) for n in order}
    history = build_risk_factor_matrix_from_df(
        df=pd.DataFrame(
            {n: [source[n]] for n in order}, index=pd.to_datetime([hf.AS_OF])
        ),
        static_config=rsc,
    )
    with pytest.raises(ValueError, match=DOMAIN_ERROR):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=history.get_snapshot(hf.AS_OF),
        )
    assert spy == []


def test_builder_entry_point_enforces_the_contract_itself(universe, spy):
    mapper, um, order = universe
    flat = _flat(order, {"VOL_EQ_USD": -0.1}, 2)
    with pytest.raises(ValueError, match="VOL levels must be strictly positive"):
        builder_mod.build_market_data_matrix(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            num_scenarios=2,
            raw_fx_spots=flat[um.idx_fx_spots],
            fx_spot_uix_in_universe=um.fx_spot_uix_in_universe,
            raw_scalar_nodes=flat[um.idx_scalar_nodes],
            scalar_node_uix_in_universe=um.scalar_node_uix_in_universe,
            scalar_node_strictly_positive=um.scalar_node_strictly_positive,
            raw_discount_tenors=um.raw_discount_tenors,
            raw_discount_levels=np.empty((0, 0, 2)),
            raw_price_curve_tenors=um.raw_price_curve_tenors,
            raw_price_curve_values=np.empty((0, 0, 2)),
            curve_interp_methods=um.curve_interp_methods,
            curve_extrap_methods=um.curve_extrap_methods,
            price_curve_uix_in_universe=um.price_curve_uix_in_universe,
            raw_vol_surface_tenors=um.raw_vol_surface_tenors,
            raw_vol_surface_strikes=um.raw_vol_surface_strikes,
            raw_vol_surface_values=np.empty((0, 0, 0, 2)),
            surface_interp_x_methods=um.surface_interp_x_methods,
            surface_extrap_x_methods=um.surface_extrap_x_methods,
            surface_y_axis_type_ids=um.surface_y_axis_type_ids,
        )
    assert spy == []


def test_negative_rate_is_legal_and_curve_levels_must_be_finite(rsc, universe, spy):
    # RATE: legal when negative at the canonical compile boundary (flag is False)
    snap = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=hf.AS_OF, levels_dict={"USD_OIS[3M]": -0.5}, static_config=rsc
    )
    assert snap.factor_levels.tolist() == [-0.005]
    # curve / surface raw levels reaching the builder must be finite, checked before D061
    mapper, um, order = universe
    flat = _flat(order, {}, 1)
    kwargs = dict(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        num_scenarios=1,
        raw_fx_spots=flat[um.idx_fx_spots],
        fx_spot_uix_in_universe=um.fx_spot_uix_in_universe,
        raw_scalar_nodes=flat[um.idx_scalar_nodes],
        scalar_node_uix_in_universe=um.scalar_node_uix_in_universe,
        scalar_node_strictly_positive=um.scalar_node_strictly_positive,
        raw_discount_tenors=np.zeros((1, 1)),
        raw_discount_levels=np.full((1, 1, 1), np.nan),
        raw_price_curve_tenors=um.raw_price_curve_tenors,
        raw_price_curve_values=np.empty((0, 0, 1)),
        curve_interp_methods=um.curve_interp_methods,
        curve_extrap_methods=um.curve_extrap_methods,
        price_curve_uix_in_universe=um.price_curve_uix_in_universe,
        raw_vol_surface_tenors=um.raw_vol_surface_tenors,
        raw_vol_surface_strikes=um.raw_vol_surface_strikes,
        raw_vol_surface_values=np.empty((0, 0, 0, 1)),
        surface_interp_x_methods=um.surface_interp_x_methods,
        surface_extrap_x_methods=um.surface_extrap_x_methods,
        surface_y_axis_type_ids=um.surface_y_axis_type_ids,
    )
    with pytest.raises(ValueError, match="discount curve levels contain non-finite"):
        builder_mod.build_market_data_matrix(**kwargs)
    # A finite NEGATIVE rate is legal (D078/D080): under CURVE the D080
    # constructor builds DF = exp(-zT) > 1 for it, with no clipping. The pillar
    # coordinate here is a positive canonical T because the layout plan would
    # have produced one; the domain gate is what this test certifies.
    kwargs["raw_discount_levels"] = np.full((1, 1, 1), -0.02)  # finite: domain passes
    kwargs["raw_discount_tenors"] = np.full((1, 1), 0.5)
    kwargs["discounting_mode"] = DiscountingMode.CURVE
    matrix = builder_mod.build_market_data_matrix(**kwargs)
    assert np.all(matrix.discount_factor_grid > 0.0)
    assert np.all(matrix.discount_factor_grid[:, 1:, :] >= 1.0)


# ------------------------------------------------------------ route consistency
def test_valid_levels_give_identical_state_on_every_route(rsc, universe):
    mapper, um, order = universe
    flat = _flat(order, {})
    via_flat = build_market_data_matrix_from_flat_levels(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=um,
        flat_levels=flat,
        num_scenarios=1,
    )
    compiled = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=hf.AS_OF,
        levels_dict={n: BASE[n] * (100.0 if n == "VOL_EQ_USD" else 1.0) for n in order},
        static_config=rsc,
    )  # VOL_EQ_USD is quoted PERCENT; canonical decimal is BASE
    via_compiled = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=compiled,
    )
    direct = RiskFactorLevelSnapshot(
        as_of_date=hf.AS_OF,
        factor_ids=um.required_factor_ids.copy(),
        factor_levels=flat[:, 0].copy(),
    )
    via_direct = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=direct,
    )
    for other in (via_compiled, via_direct):
        np.testing.assert_allclose(other.prices, via_flat.prices, rtol=1e-15, atol=0)
        np.testing.assert_array_equal(
            other.fx_spots_effective, via_flat.fx_spots_effective
        )


def test_invalid_levels_fail_on_the_compiled_route_too(rsc, universe):
    _, _, order = universe
    with pytest.raises(ValueError, match="strictly positive"):
        compile_risk_factor_level_snapshot_from_levels_dict(
            as_of_date=hf.AS_OF,
            levels_dict={
                n: (
                    -0.91
                    if n == "EURGBP"
                    else BASE[n] * (100.0 if n == "VOL_EQ_USD" else 1.0)
                )
                for n in order
            },
            static_config=rsc,
        )


# ------------------------------------------- valid economics + D059 mutations
def _pv_and_oracle(levels, s):
    run = hf.run_spot_book(
        static_config=_R, book=hf.SPOT_BOOK, levels=levels, num_scenarios=s
    )
    mkt = oracle.EconomicMarket(
        levels=levels,
        factor_ccy=hf.FACTOR_CCY,
        fx_quotes={p: levels[n] for p, n in hf.FX_QUOTES.items()},
    )
    pv = {int(u): run[4][i] for i, u in enumerate(run[0].trade_uids.tolist())}
    exp = {
        r.uid: oracle.spot_pv(
            mkt, qty=r.quantity, underlying=r.underlying, settle=r.settlement_ccy, n=s
        )
        for r in hf.SPOT_BOOK.itertuples(index=False)
    }
    return pv, exp


_R = compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())


def _mutated(name, fn):
    lv = {k: v.copy() for k, v in hf.LEVELS.items()}
    lv[name] = fn(lv[name])
    return lv


@pytest.mark.parametrize(
    ("label", "levels", "moved"),
    [
        ("price EQ_EUR", _mutated("EQ_EUR", lambda x: x * 1.02), [105]),
        ("direct FX USDEUR", _mutated("USDEUR", lambda x: x * 0.98), [101, 104]),
        ("inverse FX EURGBP", _mutated("EURGBP", lambda x: x * 1.03), [102, 105]),
        (
            "scenario level EQ_CHF[3]",
            _mutated("EQ_CHF", lambda x: np.where(np.arange(x.size) == 3, x * 1.5, x)),
            [103],
        ),
    ],
)
def test_validation_keeps_consumed_rows_connected(label, levels, moved):
    base_pv, base_exp = _pv_and_oracle(hf.LEVELS, hf.S)
    pv, exp = _pv_and_oracle(levels, hf.S)
    assert sorted(u for u in pv if not np.array_equal(pv[u], base_pv[u])) == sorted(
        moved
    )
    assert sorted(u for u in exp if not np.array_equal(exp[u], base_exp[u])) == sorted(
        moved
    )
    for u in pv:
        assert float(np.max(np.abs(pv[u] - exp[u]) / np.abs(exp[u]))) <= RTOL


@pytest.mark.parametrize("s", [1, 4096])
def test_valid_real_producer_state_reconciles_and_is_deterministic(s):
    levels = {k: np.resize(v, s) for k, v in hf.LEVELS.items()}
    junk = [np.full((8, s), -4.242e9) for _ in range(32)]
    del junk
    pv1, exp = _pv_and_oracle(levels, s)
    pv2, _ = _pv_and_oracle(levels, s)
    for u in exp:
        assert float(np.max(np.abs(pv1[u] - exp[u]) / np.abs(exp[u]))) <= RTOL
        np.testing.assert_array_equal(pv1[u], pv2[u])


def test_full_pipeline_never_values_an_invalid_state(universe, spy):
    """No MarketDataMatrix exists, so there is nothing to fold or value."""
    mapper, um, order = universe
    result = None
    with pytest.raises(ValueError, match=DOMAIN_ERROR):
        result = build_market_data_matrix_from_flat_levels(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            flat_levels=_flat(order, {"EURGBP": -0.9}),
            num_scenarios=1,
        )
    assert result is None
    assert spy == []
