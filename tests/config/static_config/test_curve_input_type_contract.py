# tests/config/static_config/test_curve_input_type_contract.py
"""
`CurveInputType` metadata contract (D079 clause 3).

What this pins:

* the enum follows the repository's `(str, Enum)` config convention and has a
  `VALID_*` frozenset, like `QuoteType` / `UnitScale`;
* the value is declared ONCE, on the curve, and reaches `RiskStaticConfig`;
* curve points inherit it through `curve_id` — `CurvePointMeta` carries no
  competing field, so there is one source of truth;
* it is orthogonal to `QuoteType`: the same `RATE`/`PERCENT` curve can be
  `PAR_RATE` or `ZERO_RATE`, and neither is inferred from the other;
* an unknown value fails closed at the configuration boundary;
* the canonical snapshot stays a flat `relative name -> level` mapping with no
  curve semantics in the payload;
* valuation never sees the enum;
* D061 stays fail-closed: declaring a curve input type makes no discounting
  path reachable.
"""

from __future__ import annotations

import dataclasses
import pathlib

import polars as pl
import pytest

from riskquant.config.enums.curve_input_types import (
    VALID_CURVE_INPUT_TYPES,
    CurveInputType,
)
from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.enums.quote_types import QuoteType
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.schema_objects import CurveMeta, CurvePointMeta
from riskquant.exceptions import ConfigurationError
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)

_DISCOUNT_CURVES = (
    "USD_SOFR_OIS",
    "USD_LIBOR_3M",
    "USD_CSA_EUR",
    "EUR_ESTR_OIS",
    "EUR_EURIBOR_6M",
    "GBP_SONIA_OIS",
    "CHF_SARON_OIS",
    "AUD_AONIA_OIS",
)


def _dfs(curve_input_types: dict[str, str | None] | None = None):
    """Canonical config, optionally with a `curve_input_type` column added."""
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    if curve_input_types is not None:
        cm = dfs["curve_meta"]
        dfs["curve_meta"] = cm.with_columns(
            pl.Series(
                "curve_input_type",
                [curve_input_types.get(n) for n in cm["curve_name"].to_list()],
                dtype=pl.String,
            )
        )
    return dfs


# ------------------------------------------------------------------ enum contract
def test_enum_follows_repository_conventions():
    assert issubclass(CurveInputType, str)
    members = [e.value for e in CurveInputType]
    assert members == [
        "PAR_RATE",
        "ZERO_RATE",
        "DISCOUNT_FACTOR",
        "LOG_DISCOUNT_FACTOR",
    ]
    assert len(set(members)) == len(members)
    assert all(e.name == e.value for e in CurveInputType)
    assert frozenset(members) == VALID_CURVE_INPUT_TYPES
    path = pathlib.Path("src/riskquant/config/enums/curve_input_types.py")
    assert (
        str(CurveInputType.PAR_RATE) == "PAR_RATE"
    )  # CurveInputType is a String-based Enum
    assert path.name == "curve_input_types.py"


def test_enum_is_orthogonal_to_quote_type():
    """No member of one taxonomy is a member of the other."""
    assert VALID_CURVE_INPUT_TYPES.isdisjoint({e.value for e in QuoteType})


# ------------------------------------------------------------ config propagation
def test_curve_input_type_reaches_the_compiled_static_config():
    declared = {"USD_SOFR_OIS": "PAR_RATE", "EUR_ESTR_OIS": "ZERO_RATE"}
    rsc = compile_risk_static_config_from_dict_of_dfs(_dfs(declared))
    by_name = {c.curve_name: c for c in rsc.curve_meta_dict.values()}
    assert by_name["USD_SOFR_OIS"].curve_input_type == "PAR_RATE"
    assert by_name["EUR_ESTR_OIS"].curve_input_type == "ZERO_RATE"
    # same generic market-data metadata on both: the distinction is carried
    # only by curve_input_type
    for name in ("USD_SOFR_OIS", "EUR_ESTR_OIS"):
        assert by_name[name].quote_type == QuoteType.RATE.value
        assert by_name[name].unit_scale == "PERCENT"
    # undeclared curves stay None: never inferred from quote_type
    assert by_name["GBP_SONIA_OIS"].curve_input_type is None


@pytest.mark.parametrize(
    "value", ["PAR_RATE", "ZERO_RATE", "DISCOUNT_FACTOR", "LOG_DISCOUNT_FACTOR"]
)
def test_every_member_survives_compilation(value):
    rsc = compile_risk_static_config_from_dict_of_dfs(_dfs({"USD_SOFR_OIS": value}))
    by_name = {c.curve_name: c for c in rsc.curve_meta_dict.values()}
    assert by_name["USD_SOFR_OIS"].curve_input_type == value


def test_absent_column_compiles_and_leaves_every_curve_undeclared():
    rsc = compile_risk_static_config_from_dict_of_dfs(_dfs())
    assert (
        "curve_input_type"
        not in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS["curve_meta"].columns
    )
    assert all(c.curve_input_type is None for c in rsc.curve_meta_dict.values())


def test_unknown_value_fails_closed_at_the_configuration_boundary():
    with pytest.raises(ConfigurationError, match="CurveInputType"):
        compile_risk_static_config_from_dict_of_dfs(
            _dfs({"USD_SOFR_OIS": "PAR"})  # not a member
        )


# ------------------------------------------------------------- point inheritance
def test_curve_points_inherit_and_cannot_disagree():
    """`CurvePointMeta` has no curve_input_type field: one source of truth."""
    point_fields = {f.name for f in dataclasses.fields(CurvePointMeta)}
    assert "curve_input_type" not in point_fields
    assert "curve_input_type" in {f.name for f in dataclasses.fields(CurveMeta)}

    rsc = compile_risk_static_config_from_dict_of_dfs(
        _dfs({"USD_SOFR_OIS": "PAR_RATE"})
    )
    curve_id = rsc.container_name_to_id["USD_SOFR_OIS"]
    points = [p for p in rsc.curve_point_meta_dict.values() if p.curve_id == curve_id]
    assert len(points) == 9  # ON .. 10Y
    for point in points:
        assert rsc.curve_meta_dict[point.curve_id].curve_input_type == "PAR_RATE"


# ---------------------------------------------------------------- snapshot impact
def test_canonical_snapshot_stays_a_flat_level_mapping():
    snapshot = CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT
    assert all(isinstance(k, str) for k in snapshot)
    assert all(isinstance(v, float | int) for v in snapshot.values())
    assert "USD_SOFR_OIS[1Y]" in snapshot and snapshot["USD_SOFR_OIS[1Y]"] == 4.80
    # no curve semantics encoded in relative names, and no enum in the payload
    assert not [k for k in snapshot for m in VALID_CURVE_INPUT_TYPES if m in k]
    assert not [v for v in snapshot.values() if isinstance(v, CurveInputType)]


# --------------------------------------------------------------- isolation / D061
def test_no_valuation_or_kernel_module_references_curve_input_type():
    roots = [
        pathlib.Path("src/riskquant/valuation"),
        pathlib.Path("src/riskquant/sensitivities"),
    ]
    offenders = [
        str(p)
        for root in roots
        for p in root.rglob("*.py")
        if "CurveInputType" in p.read_text() or "curve_input_type" in p.read_text()
    ]
    assert offenders == []


def test_declaring_a_curve_input_type_does_not_make_curve_discounting_reachable():
    """D061 stays fail-closed: the metadata names a contract, it does not enable one."""
    import datetime as dt

    import pandas as pd

    from riskquant.market_data.assemble import (
        build_market_data_matrix_from_snapshot,
        build_universe_mapping,
    )
    from riskquant.market_data.risk_factors.compile_snapshot import (
        compile_risk_factor_level_snapshot_from_levels_dict,
    )
    from riskquant.portfolio.engine import build_uix_mapper
    from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

    as_of = dt.date(2026, 1, 15)
    rsc = compile_risk_static_config_from_dict_of_dfs(
        _dfs({name: "ZERO_RATE" for name in _DISCOUNT_CURVES})
    )
    frame = pd.DataFrame([
        dict(
            uid=1,
            instrument="VANILLA_FORWARD_NODE",
            quantity=1.0,
            underlying="BRENT_CRUDE_FWD[M+0]",
            settlement_ccy="USD",
            strike_price=70.0,
            strike_ccy="USD",
            delivery_date=dt.date(2026, 2, 15),
            settlement_date=dt.date(2026, 2, 17),
            under_discount_curve="USD_SOFR_OIS",
            strike_discount_curve="USD_SOFR_OIS",
        )
    ])
    batch = compile_raw_df_into_ingest_batch(
        raw_df=frame, static_config=rsc, copy=True, as_of_date=as_of
    )
    mapper = build_uix_mapper([batch])
    um = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=as_of,
        levels_dict={names[int(i)]: 80.0 for i in um.required_factor_ids},
        static_config=rsc,
    )
    with pytest.raises(NotImplementedError, match="D061"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.CURVE,
            as_of_date=as_of,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snapshot,
        )


def test_commodity_curves_are_not_forced_to_declare_a_curve_input_type():
    """The metadata is optional; a PRICE forward curve may leave it undeclared."""
    rsc = compile_risk_static_config_from_dict_of_dfs(
        _dfs({"USD_SOFR_OIS": "PAR_RATE"})
    )
    by_name = {c.curve_name: c for c in rsc.curve_meta_dict.values()}
    assert by_name["BRENT_CRUDE_FWD"].quote_type == QuoteType.PRICE.value
    assert by_name["BRENT_CRUDE_FWD"].curve_input_type is None
