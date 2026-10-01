# tests/market_data/remediation/test_node_option_delivery_contract.py
"""
NODE option / NODE forward date contract (D062 / D065, option half).

Locked contract:

* a NODE option's value is a function of expiry (forward measure, vol, TTE) and
  settlement (payoff discount). `delivery_date` is NOT part of the option
  contract and MUST NOT be required;
* a RELATIVE (OFFSET) NODE forward's `delivery_date` IS required: it is the
  asset leg's discount date and must fall in the resolved contract period
  (D065). An EXACT (LABEL) NODE forward may omit it; its asset leg is then
  discounted at settlement (D101);
* a relative name plus the RUN as-of resolves to exactly one rf id. An option's
  expiry MUST NOT re-resolve it (D067 / D064 unchanged).

Underlyings here are exact market points: `BRENT_CRUDE_FWD[M+0]` is a curve
POINT, read from one universe row exactly like a node, never interpolated.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.utils.dates import date_to_ord32

A = CANONICAL_AS_OF_DATE  # 2026-01-15; BRENT_CRUDE_FWD[M+0] is the 202602 contract
NEXT_MONTH = date(2026, 2, 1)  # CALENDAR_PERIOD roll boundary
PILLAR = "BRENT_CRUDE_FWD[M+0]"


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )


def _option(**overrides) -> pd.DataFrame:
    row = dict(
        uid=1,
        instrument="VANILLA_EUROPEAN_OPTION_NODE",
        quantity=1_000.0,
        underlying=PILLAR,
        settlement_ccy="USD",
        strike_price=70.0,
        strike_ccy="USD",
        is_call=1.0,
        expiry_date=date(2026, 2, 18),
        settlement_date=date(2026, 2, 20),
        volatility_container_name="BRENT_CRUDE_ATM_VOL",
        under_discount_curve="USD_SOFR_OIS",
        strike_discount_curve="USD_SOFR_OIS",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def _forward(**overrides) -> pd.DataFrame:
    row = dict(
        uid=2,
        instrument="VANILLA_FORWARD_NODE",
        quantity=1_000.0,
        underlying=PILLAR,
        settlement_ccy="USD",
        strike_price=70.0,
        strike_ccy="USD",
        delivery_date=date(2026, 2, 15),
        settlement_date=date(2026, 2, 17),
        under_discount_curve="USD_SOFR_OIS",
        strike_discount_curve="USD_SOFR_OIS",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def _ingest_and_compile(rsc, frame, as_of=A):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(frame), static_config=rsc, as_of_date=as_of
    )
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    return batch, mapper, portfolio


# --------------------------------------------------------------- Test 1
def test_node_option_without_delivery_ingests_and_compiles(rsc):
    frame = _option()
    assert "delivery_date" not in frame.columns
    batch, mapper, portfolio = _ingest_and_compile(rsc, frame)

    assert int(batch.underlying_ids[0]) == int(rsc.rel_name_to_rf_id[PILLAR])
    assert int(batch.delivery_ord[0]) < 0  # no delivery carried
    assert int(batch.expiry_ord[0]) == int(date_to_ord32(date(2026, 2, 18)))
    assert int(batch.settle_ord[0]) == int(date_to_ord32(date(2026, 2, 20)))

    option = portfolio.vanilla_european_option
    assert option is not None
    assert option.trade_uids.tolist() == [1]
    assert portfolio.uix_fingerprint == mapper.fingerprint
    assert float(option.sqrt_tte[0]) > 0.0

    # The producer builds under the supported NONE discounting mode: the
    # option's discount rows are DF = 1 (D061), and no curve level is read.
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A,
        levels_dict={names[int(i)]: 80.0 for i in um.required_factor_ids},
        static_config=rsc,
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    assert matrix.discount_factor_grid.shape[0] == 1  # the USD_SOFR_OIS curve
    assert np.all(matrix.discount_factor_grid == 1.0)


def test_option_economics_do_not_change_when_delivery_metadata_is_supplied(rsc):
    """Delivery is optional metadata for an option, never an economic input."""
    _, _, without = _ingest_and_compile(rsc, _option())
    _, _, with_delivery = _ingest_and_compile(
        rsc, _option(delivery_date=date(2026, 2, 15))
    )
    a, b = without.vanilla_european_option, with_delivery.vanilla_european_option
    for field in ("qty", "strike", "log_strike", "cp_mult", "sqrt_tte", "half_tte"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


# --------------------------------------------------------------- Test 3
def test_relative_node_forward_still_requires_delivery_and_its_contract_period(rsc):
    """
    BRENT_CRUDE_FWD[M+0] is a CALENDAR_PERIOD (relative) NODE.
    Delivery remains required for relative NODEs (D065, D062 amended).
    The error now comes from the identity gate (identity.py), not the family
    validator (validate.py), because delivery_date is no longer in the family
    mandatory-column list.
    """
    # Column absent: _get_date_col returns all-null → NO_DATE_ORD32 →
    # identity gate fires for CALENDAR_PERIOD underlying.
    with pytest.raises(
        Exception, match=r"forward contract identity requires delivery_date"
    ):
        compile_raw_df_into_ingest_batch(
            raw_df=pl.from_pandas(
                _forward().drop(columns=["delivery_date"])
            ),  # column absent
            static_config=rsc,
            as_of_date=A,
        )
    # Null delivery on a present column takes the same path.
    with pytest.raises(
        Exception, match=r"forward contract identity requires delivery_date"
    ):
        compile_raw_df_into_ingest_batch(
            raw_df=pl.from_pandas(_forward(delivery_date=None)),
            static_config=rsc,
            as_of_date=A,
        )
    # The D065 period check is untouched for relative NODEs.
    with pytest.raises(Exception, match="is the 202602 contract"):
        compile_raw_df_into_ingest_batch(
            raw_df=pl.from_pandas(
                _forward(
                    delivery_date=date(2026, 3, 15), settlement_date=date(2026, 3, 17)
                )
            ),
            static_config=rsc,
            as_of_date=A,
        )
    # A well-formed relative-NODE forward still ingests and compiles.
    _, _, portfolio = _ingest_and_compile(rsc, _forward())
    assert portfolio.vanilla_forward is not None


def test_forward_and_option_still_require_the_run_as_of(rsc):
    for frame in (_forward(), _option()):
        with pytest.raises(ValueError, match="require the run as-of date"):
            compile_raw_df_into_ingest_batch(
                raw_df=pl.from_pandas(frame), static_config=rsc, as_of_date=None
            )


# --------------------------------------------------------------- Test 4
def test_relative_identity_is_fixed_by_the_run_as_of_not_by_expiry(rsc):
    pillar = int(rsc.rel_name_to_rf_id[PILLAR])

    near, mapper_near, _ = _ingest_and_compile(rsc, _option())
    far, mapper_far, _ = _ingest_and_compile(
        rsc,
        _option(expiry_date=date(2026, 11, 18), settlement_date=date(2026, 11, 20)),
    )
    # expiry does NOT re-resolve the relative name
    assert int(near.underlying_ids[0]) == int(far.underlying_ids[0]) == pillar
    assert mapper_near.as_of_ord32 == mapper_far.as_of_ord32 == int(date_to_ord32(A))

    # a different run as-of keeps the pillar but is a different run identity
    _, mapper_rolled, _ = _ingest_and_compile(rsc, _option(), as_of=NEXT_MONTH)
    assert mapper_rolled.as_of_ord32 == int(date_to_ord32(NEXT_MONTH))
    assert mapper_rolled.fingerprint != mapper_near.fingerprint
