# tests/market_data/remediation/test_contract_failure_paths.py
"""
Fail-closed evidence for every remediated market-data contract (D060-D072).

Each test drives a PRODUCTION entry point with a hostile but syntactically
valid input and asserts that the run stops with the contract's own error
before any gather, fold or kernel can consume the input.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import hostile_fixture as hf
import numpy as np
import pandas as pd
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.constants import INVALID_UIX, NO_FX_RF_REQUIRED
from riskquant.exceptions import PrettyKeyError
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_df,
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.universe import _mixed_radix_key, lookup_fx_effective_uix
from riskquant.utils.dates import date_to_ord32

AS_OF_ORD = int(date_to_ord32(hf.AS_OF))


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())


def _spot(uid, und, settle, qty=10.0):
    return dict(
        uid=uid, instrument="SPOT", quantity=qty, underlying=und, settlement_ccy=settle
    )


def _fwd(uid, **kw):
    row = dict(
        uid=uid,
        instrument="VANILLA_FORWARD_NODE",
        quantity=100.0,
        underlying="BRENT_FWD[M+0]",
        settlement_ccy="USD",
        strike_price=80.0,
        strike_ccy="USD",
        delivery_date=date(2027, 2, 15),
        settlement_date=date(2027, 2, 17),
        under_discount_curve="USD_OIS",
        strike_discount_curve="USD_OIS",
    )
    row.update(kw)
    return row


def _opt(uid, **kw):
    row = _fwd(
        uid,
        instrument="VANILLA_EUROPEAN_OPTION_NODE",
        expiry_date=date(2027, 1, 29),
        is_call=1,
        volatility_container_name="VOL_EQ_USD",
    )
    row.update(kw)
    return row


def _ingest(rsc, rows, as_of=hf.AS_OF):
    return compile_raw_df_into_ingest_batch(
        raw_df=pd.DataFrame(rows), static_config=rsc, copy=True, as_of_date=as_of
    )


# ---------------------------------------------------------------- D060 FX sentinel
def test_static_config_fx_matrix_fill_is_the_canonical_sentinel(rsc):
    cells = rsc.fx_spot_matrix
    configured = {(b, q) for (b, q) in rsc.fx_spot_lookup}
    empty = np.ones(cells.shape, dtype=bool)
    for b, q in configured:
        empty[b, q] = False
    assert np.all(cells[empty] == NO_FX_RF_REQUIRED)


def test_missing_fx_pair_fails_at_ingest(rsc):
    with pytest.raises(PrettyKeyError, match="No FX spot rf configured"):
        _ingest(rsc, [_spot(1, "EQ_JPY", "EUR")])


def test_missing_fx_pair_in_mixed_book_fails_at_ingest(rsc):
    with pytest.raises(PrettyKeyError, match="No FX spot rf configured"):
        _ingest(rsc, [_spot(1, "EQ_USD", "EUR"), _spot(2, "EQ_JPY", "EUR")])


def test_inverse_only_pair_resolves_to_inverted_leg(rsc):
    batch = _ingest(rsc, [_spot(1, "EQ_GBP", "EUR")])
    assert int(batch.fx_u2s_ids[0]) == rsc.rel_name_to_rf_id["EURGBP"]
    assert int(batch.fx_u2s_inv[0]) == 1


def test_negative_non_sentinel_fx_id_cannot_become_identity():
    with pytest.raises(ValueError, match="not NO_FX_RF_REQUIRED"):
        lookup_fx_effective_uix(
            market_id_universe=np.array([7, 31], np.int32),
            fx_effective_pairs=np.array([[1, 0]], np.int32),
            raw_fx_ids=np.array([-999_999], np.int32),
            raw_fx_inv=np.array([0], np.int8),
        )


def test_reciprocal_fx_orientations_are_rejected_in_static_config():
    import polars as pl

    dfs = hf.static_config_dfs()
    schema = dfs["node_meta"].schema
    row = dict(
        node_id=6001,
        node_name="GBPEUR",
        quote_ccy="EUR",
        asset_class="FX",
        asset_subclass="FX_G10",
        quote_type="FX_RATE",
        unit_scale="DECIMAL",
        shift_type="LOG_RETURN",
        fx_base_ccy="GBP",
    )
    dfs["node_meta"] = dfs["node_meta"].vstack(pl.DataFrame([row], schema=schema))
    with pytest.raises(Exception, match="reciprocal"):
        compile_risk_static_config_from_dict_of_dfs(dfs)


# ---------------------------------------------------------------- D062 key packing
@pytest.mark.parametrize("bad", [INVALID_UIX, -1, 7])
def test_packed_coordinate_outside_domain_is_rejected(bad):
    cols = (np.array([0, 1], np.int64), np.array([bad, 0], np.int64))
    with pytest.raises(ValueError, match="outside"):
        _mixed_radix_key(cols, (2, 5), ctx="probe")


@pytest.mark.parametrize(
    "missing",
    [
        "delivery_date",
        "settlement_date",
        "under_discount_curve",
        "strike_discount_curve",
    ],
)
def test_forward_missing_coordinate_fails_at_ingest(rsc, missing):
    with pytest.raises(ValueError, match=rf"missing '{missing}'"):
        _ingest(rsc, [_fwd(1, **{missing: None}), _fwd(2)])


def test_md03_alias_fixture_cannot_reach_the_mapper_even_bypassing_ingest(rsc):
    good = _ingest(rsc, [_fwd(1), _fwd(2, settlement_date=date(2027, 2, 18))])
    settle = good.settle_ord.copy()
    settle[1] = -99_999
    bypass = dataclasses.replace(good, settle_ord=np.ascontiguousarray(settle))
    mapper = build_uix_mapper([bypass])
    with pytest.raises(ValueError, match="outside"):
        compile_portfolio_from_ingest_batch(ingest_batch=bypass, uix_mapper=mapper)


# ---------------------------------------------------------------- D063 source kind
@pytest.mark.parametrize("und", ["USD_OIS", "BRENT_FWD", "VOL_EQ_USD"])
def test_spot_source_kind_mismatch_is_rejected(rsc, und):
    with pytest.raises(PrettyKeyError, match=r"source kind|NEITHER"):
        _ingest(rsc, [_spot(1, und, "USD")])


def test_forward_on_plain_price_node_is_rejected(rsc):
    with pytest.raises(PrettyKeyError, match="source kind"):
        _ingest(rsc, [_fwd(1, underlying="EQ_USD")])


def test_option_volatility_must_be_a_vol_rf(rsc):
    with pytest.raises(PrettyKeyError, match="source kind"):
        _ingest(rsc, [_opt(1, volatility_container_name="EQ_USD")])


# ---------------------------------------------------------------- D065 contract identity
def test_forward_delivery_outside_resolved_contract_is_rejected(rsc):
    with pytest.raises(PrettyKeyError, match=r"202702 contract"):
        _ingest(rsc, [_fwd(1, delivery_date=date(2027, 3, 15))])


def test_forward_delivery_inside_resolved_contract_is_accepted(rsc):
    batch = _ingest(rsc, [_fwd(1, delivery_date=date(2027, 2, 27))])
    assert batch.as_of_ord32 == AS_OF_ORD


def test_forward_rows_require_an_as_of(rsc):
    with pytest.raises(ValueError, match="as-of"):
        _ingest(rsc, [_fwd(1)], as_of=None)


# ---------------------------------------------------------------- D066 discount linkage
def test_curve_currency_must_equal_leg_currency(rsc):
    with pytest.raises(PrettyKeyError, match="curve currency must equal leg"):
        _ingest(rsc, [_fwd(1, strike_ccy="EUR", strike_discount_curve="USD_OIS")])


def test_non_discount_curve_is_rejected(rsc):
    with pytest.raises(PrettyKeyError, match="not a discount curve"):
        _ingest(rsc, [_fwd(1, under_discount_curve="BRENT_FWD")])


# ---------------------------------------------------------------- D061 discount producer
def test_discount_dependent_book_fails_closed_before_any_kernel(rsc):
    # D061 is keyed on the requested discounting mode: CURVE (market-derived
    # discounting) is unimplemented and fails before any curve machinery runs.
    batch = _ingest(rsc, [_fwd(1), _opt(2)])
    mapper = build_uix_mapper([batch])
    compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=mapper)
    um = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    levels = {"BRENT_FWD[M+0]": 81.0, "VOL_EQ_USD": 18.3, "USD_OIS[3M]": 4.25}
    snap = build_risk_factor_level_snapshot_from_dict(
        as_of_date=hf.AS_OF, levels_dict=levels, static_config=rsc
    )
    with pytest.raises(NotImplementedError, match="D061"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.CURVE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snap,
        )


# ---------------------------------------------------------------- D067 as-of identity
def _spot_run(rsc, mapper_as_of=AS_OF_ORD):
    batch = _ingest(rsc, [_spot(1, "EQ_USD", "EUR")], as_of=None)
    mapper = build_uix_mapper([batch], as_of_ord32=mapper_as_of)
    um = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    return batch, mapper, um


def test_snapshot_as_of_mismatch_is_rejected(rsc):
    _, mapper, um = _spot_run(rsc, AS_OF_ORD)
    snap = build_risk_factor_level_snapshot_from_dict(
        as_of_date=hf.AS_OF - timedelta(days=1),
        levels_dict={"EQ_USD": 1.0, "USDEUR": 0.9},
        static_config=rsc,
    )
    with pytest.raises(ValueError, match="D067"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snap,
        )


@pytest.mark.parametrize("mapper_as_of", [None, AS_OF_ORD + 1])
def test_mapper_as_of_must_equal_matrix_as_of(rsc, mapper_as_of):
    _, mapper, um = _spot_run(rsc, mapper_as_of)
    with pytest.raises(ValueError, match="D067"):
        build_market_data_matrix_from_flat_levels(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=hf.AS_OF,
            uix_mapper=mapper,
            universe_map=um,
            flat_levels=np.ones((um.required_factor_ids.shape[0], 1)),
            num_scenarios=1,
        )


def test_ingest_as_of_and_mapper_as_of_cannot_disagree(rsc):
    batch = _ingest(rsc, [_fwd(1)])
    with pytest.raises(ValueError, match="D067"):
        build_uix_mapper([batch], as_of_ord32=AS_OF_ORD + 3)
    assert build_uix_mapper([batch]).as_of_ord32 == AS_OF_ORD


# ---------------------------------------------------------------- D064 mapper identity
def test_matrix_built_for_another_mapper_is_rejected_by_the_fold(rsc):
    run_a = hf.run_spot_book(
        static_config=rsc,
        book=pd.DataFrame([_spot(1, "EQ_USD", "EUR")]),
        levels=hf.LEVELS,
        num_scenarios=hf.S,
    )
    run_b = hf.run_spot_book(
        static_config=rsc,
        book=pd.DataFrame([_spot(1, "EQ_GBP", "EUR")]),
        levels=hf.LEVELS,
        num_scenarios=hf.S,
    )
    assert run_a[2].prices.shape == run_b[2].prices.shape
    with pytest.raises(ValueError, match="different UixMapper"):
        assemble_valuation_state(matrix=run_a[2], uix_mapper=run_b[1])


# ---------------------------------------------------------------- D068 expired options
def test_expired_option_is_rejected(rsc):
    batch = _ingest(rsc, [_opt(1, expiry_date=date(2027, 1, 3))])
    mapper = build_uix_mapper([batch])
    with pytest.raises(ValueError, match="D068"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=mapper)


def test_option_expiring_on_the_as_of_compiles_at_tte_floor(rsc):
    batch = _ingest(rsc, [_opt(1, expiry_date=hf.AS_OF)])
    mapper = build_uix_mapper([batch])
    port = compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=mapper)
    assert port.vanilla_european_option.sqrt_tte[0] == pytest.approx(np.sqrt(1e-9))


# ---------------------------------------------------------------- D069/D070 snapshot
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_snapshot_rejects_non_finite_levels(bad):
    with pytest.raises(ValueError, match="non-finite"):
        RiskFactorLevelSnapshot(
            as_of_date=hf.AS_OF,
            factor_ids=np.array([7, 31], np.int32),
            factor_levels=np.array([1.0, bad]),
        )


@pytest.mark.parametrize(
    "name,level", [("USDEUR", 0.0), ("USDEUR", -0.9), ("VOL_EQ_USD", -18.3)]
)
def test_strictly_positive_quote_types_reject_non_positive_levels(rsc, name, level):
    with pytest.raises(ValueError, match="strictly positive"):
        compile_risk_factor_level_snapshot_from_levels_dict(
            as_of_date=hf.AS_OF, levels_dict={name: level}, static_config=rsc
        )


def test_negative_price_level_is_accepted(rsc):
    snap = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=hf.AS_OF, levels_dict={"PWR_SPREAD": -7.25}, static_config=rsc
    )
    assert snap.factor_levels.tolist() == [-7.25]


def test_every_snapshot_route_yields_the_same_canonical_units_and_dtype(rsc):
    lv = {"VOL_EQ_USD": 18.3, "EQ_USD": 212.37}
    a = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=hf.AS_OF, levels_dict=lv, static_config=rsc
    )
    b = build_risk_factor_level_snapshot_from_dict(
        as_of_date=hf.AS_OF, levels_dict=lv, static_config=rsc
    )
    c = build_risk_factor_level_snapshot_from_df(
        as_of_date=hf.AS_OF,
        df=pd.DataFrame({"risk_factor_name": list(lv), "level": list(lv.values())}),
        static_config=rsc,
    )
    for snap in (a, b, c):
        assert snap.factor_levels.dtype == np.float64
        np.testing.assert_array_equal(snap.factor_ids, a.factor_ids)
        np.testing.assert_array_equal(snap.factor_levels, a.factor_levels)
    assert dict(zip(a.factor_ids.tolist(), a.factor_levels.tolist(), strict=True))[
        17
    ] == pytest.approx(0.183, abs=0.0, rel=1e-15)


def test_df_route_rejects_missing_level_instead_of_dropping(rsc):
    df = pd.DataFrame({
        "risk_factor_name": ["EQ_USD", "USDEUR"],
        "level": [212.37, np.nan],
    })
    with pytest.raises(ValueError, match="Missing"):
        build_risk_factor_level_snapshot_from_df(
            as_of_date=hf.AS_OF, df=df, static_config=rsc
        )
