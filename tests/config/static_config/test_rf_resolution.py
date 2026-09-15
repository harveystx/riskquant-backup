# tests/config/static_config/test_rf_resolution.py
"""
Regression tests for `riskquant.config.static_config.rf_resolution`.

Coverage:
  - Each of the 6 (XAxisKind, RollMethod) resolver functions individually,
    with oracle values computed independently from the module's documented
    formulas (not mirrored from the implementation).
  - The half-year boundary matrix around every FH/BH transition, including
    the exact boundary date and a year rollover.
  - The shared `parse_fh_bh_mmdd` parser (format + month/day validity).
  - `RiskFactorAbsToRelNameIdMap` bidirectional-index construction and its
    internal consistency checks.
  - `build_risk_factor_name_mapping_for_as_of_date` end-to-end against the
    real canonical static-config fixture.
  - Every documented resolver error path.
  - Drift guard: the dispatch registry keyset equals
    ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS and the parameter contract covers
    every RollMethod.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from riskquant.config.enums.roll_methods import (
    ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS,
    ROLL_METHOD_REQUIRED_PARAMS,
    RollMethod,
    parse_fh_bh_mmdd,
)
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    _OFFSET_RESOLVERS,
    RiskFactorAbsToRelNameIdMap,
    _half_year_boundary_by_date,
    _half_year_label_suffix,
    _resolve_curve_or_surface_suffix,
    _resolve_offset_hy_half_year_by_date,
    _resolve_offset_hy_half_year_by_label,
    _resolve_offset_m_calendar_period,
    _resolve_offset_m_half_year_by_date,
    _resolve_offset_y_calendar_period,
    _resolve_offset_y_roll_on_mmdd,
    _ResolverContext,
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)


def _ctx(
    *,
    token_offset: int,
    as_of: date,
    anchor_offset: int | None = None,
    roll_month: int | None = None,
    roll_day: int | None = None,
    fh_bh_mmdd: str | None = None,
) -> _ResolverContext:
    """Build a resolver context carrying roll params directly (no rule object)."""
    return _ResolverContext(
        container_name="TEST",
        relative_name="TEST[X]",
        token_offset=token_offset,
        as_of_date=as_of,
        anchor_offset=anchor_offset,
        roll_month=roll_month,
        roll_day=roll_day,
        fh_bh_mmdd=fh_bh_mmdd,
    )


# ---------------------------------------------------------------------------
# Individual resolver functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor_offset,token_offset,as_of,expected",
    [
        (1, 0, date(2026, 5, 23), "202606"),
        (1, 2, date(2026, 5, 23), "202608"),
        (0, 0, date(2026, 5, 23), "202605"),
        (1, -1, date(2026, 5, 23), "202605"),
        (1, 0, date(2026, 12, 15), "202701"),  # year rollover
    ],
)
def test_resolve_offset_m_calendar_period(anchor_offset, token_offset, as_of, expected):
    ctx = _ctx(anchor_offset=anchor_offset, token_offset=token_offset, as_of=as_of)
    assert _resolve_offset_m_calendar_period(ctx) == expected


@pytest.mark.parametrize(
    "anchor_offset,token_offset,as_of,expected",
    [
        (1, 0, date(2026, 5, 23), "2027"),
        (0, 0, date(2026, 5, 23), "2026"),
        (0, 1, date(2026, 5, 23), "2027"),
        (0, -1, date(2026, 5, 23), "2025"),
    ],
)
def test_resolve_offset_y_calendar_period(anchor_offset, token_offset, as_of, expected):
    ctx = _ctx(anchor_offset=anchor_offset, token_offset=token_offset, as_of=as_of)
    assert _resolve_offset_y_calendar_period(ctx) == expected


@pytest.mark.parametrize(
    "as_of,expected",
    [
        (date(2026, 5, 23), "2026"),  # after Apr-1 roll: current year
        (date(2026, 4, 1), "2026"),  # on the roll date itself: current year
        (date(2026, 3, 31), "2025"),  # day before roll: previous year
        (date(2026, 1, 1), "2025"),  # start of year, before roll
    ],
)
def test_resolve_offset_y_roll_on_mmdd(as_of, expected):
    ctx = _ctx(roll_month=4, roll_day=1, token_offset=0, as_of=as_of)
    assert _resolve_offset_y_roll_on_mmdd(ctx) == expected


def test_resolve_offset_hy_half_year_by_label():
    ctx = _ctx(fh_bh_mmdd="0401|1001", token_offset=0, as_of=date(2026, 5, 23))
    assert _resolve_offset_hy_half_year_by_label(ctx) == "2026_BH"


def test_resolve_offset_hy_half_year_by_date():
    ctx = _ctx(fh_bh_mmdd="0401|1001", token_offset=0, as_of=date(2026, 5, 23))
    assert _resolve_offset_hy_half_year_by_date(ctx) == "20261001"


def test_resolve_offset_m_half_year_by_date():
    ctx = _ctx(fh_bh_mmdd="0401|1001", token_offset=0, as_of=date(2026, 5, 23))
    assert _resolve_offset_m_half_year_by_date(ctx) == "20261001"


def test_resolve_calendar_period_missing_anchor_offset_raises():
    ctx = _ctx(token_offset=0, as_of=date(2026, 5, 23))  # anchor_offset None
    with pytest.raises(ValueError, match="anchor_offset"):
        _resolve_offset_m_calendar_period(ctx)


@pytest.mark.parametrize(
    "anchor_offset,token_offset,expected",
    [
        (0, 0, "202605"),  # neither shift
        (1, 0, "202606"),  # anchor_offset shifts the spot anchor +1 month
        (0, 1, "202606"),  # token offset [M+1] shifts +1 month
        (1, 1, "202607"),  # both compose additively
        (3, 0, "202608"),  # anchor-only, +3
        (0, 3, "202608"),  # token-only, +3
    ],
)
def test_anchor_offset_distinct_from_token_offset(
    anchor_offset, token_offset, expected
):
    """
    anchor_offset (per-curve constant) and the [M+N] token offset (per-point)
    are DISTINCT inputs. For CALENDAR_PERIOD:
        resolved = as_of_period + anchor_offset + token_offset.
    Both contribute; neither is the other. as_of fixed at 2026-05.
    """
    ctx = _ctx(
        anchor_offset=anchor_offset,
        token_offset=token_offset,
        as_of=date(2026, 5, 23),
    )
    assert _resolve_offset_m_calendar_period(ctx) == expected


def test_resolve_roll_on_mmdd_missing_params_raises():
    ctx = _ctx(token_offset=0, as_of=date(2026, 5, 23))  # roll_month/day None
    with pytest.raises(ValueError, match="roll_month and roll_day"):
        _resolve_offset_y_roll_on_mmdd(ctx)


# ---------------------------------------------------------------------------
# Half-year boundary matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of,expected_label",
    [
        (date(2026, 3, 31), "2026_FH"),
        (date(2026, 4, 1), "2026_FH"),
        (date(2026, 4, 2), "2026_BH"),
        (date(2026, 9, 30), "2026_BH"),
        (date(2026, 10, 1), "2026_BH"),
        (date(2026, 10, 2), "2026_BH"),
        (date(2026, 12, 31), "2026_BH"),
        (date(2027, 1, 1), "2027_FH"),
    ],
)
def test_half_year_label_suffix_boundary_matrix(as_of, expected_label):
    assert (
        _half_year_label_suffix(
            as_of_date=as_of, fh_bh_mmdd="0401|1001", token_offset=0
        )
        == expected_label
    )


@pytest.mark.parametrize(
    "as_of,expected_date",
    [
        (date(2026, 3, 31), date(2026, 4, 1)),
        (date(2026, 4, 1), date(2026, 4, 1)),
        (date(2026, 4, 2), date(2026, 10, 1)),
        (date(2026, 9, 30), date(2026, 10, 1)),
        (date(2026, 10, 1), date(2026, 10, 1)),
        (date(2026, 10, 2), date(2027, 4, 1)),
        (date(2026, 12, 31), date(2027, 4, 1)),
        (date(2027, 1, 1), date(2027, 4, 1)),
    ],
)
def test_half_year_boundary_by_date_matrix(as_of, expected_date):
    result = _half_year_boundary_by_date(
        as_of_date=as_of,
        fh_bh_mmdd="0401|1001",
        token_offset=0,
        relative_name="X",
        container_name="Y",
    )
    assert result == expected_date


@pytest.mark.parametrize(
    "token_offset,expected_label,expected_date",
    [
        (-1, "2026_FH", date(2026, 4, 1)),
        (0, "2026_BH", date(2026, 10, 1)),
        (1, "2027_FH", date(2027, 4, 1)),
        (2, "2027_BH", date(2027, 10, 1)),
    ],
)
def test_half_year_offsets_advance_monotonically(
    token_offset, expected_label, expected_date
):
    as_of = date(2026, 5, 23)
    assert (
        _half_year_label_suffix(
            as_of_date=as_of, fh_bh_mmdd="0401|1001", token_offset=token_offset
        )
        == expected_label
    )
    assert (
        _half_year_boundary_by_date(
            as_of_date=as_of,
            fh_bh_mmdd="0401|1001",
            token_offset=token_offset,
            relative_name="X",
            container_name="Y",
        )
        == expected_date
    )


# ---------------------------------------------------------------------------
# parse_fh_bh_mmdd (shared parser)
# ---------------------------------------------------------------------------


def test_parse_fh_bh_mmdd_valid():
    assert parse_fh_bh_mmdd("0401|1001") == (4, 1, 10, 1)
    assert parse_fh_bh_mmdd("1231|0630") == (12, 31, 6, 30)


@pytest.mark.parametrize(
    "bad",
    [
        "0401",
        "0401|1001|extra",
        "040|1001",
        "0401|100",
        "",
        "abcd|efgh",
        "1301|1001",  # month 13
        "0001|1001",  # month 0
        "0230|1001",  # Feb 30 invalid
        "0401|0000",  # day 0
    ],
)
def test_parse_fh_bh_mmdd_invalid_raises(bad):
    with pytest.raises(ValueError):
        parse_fh_bh_mmdd(bad)


# ---------------------------------------------------------------------------
# Drift guard: one taxonomy shared by resolver and validator
# ---------------------------------------------------------------------------


def test_resolver_registry_matches_allowed_pairs():
    assert set(_OFFSET_RESOLVERS) == ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS


def test_param_contract_covers_every_roll_method():
    assert set(ROLL_METHOD_REQUIRED_PARAMS) == set(RollMethod)


# ---------------------------------------------------------------------------
# Error paths in _resolve_curve_or_surface_suffix
# ---------------------------------------------------------------------------


def test_label_container_with_offset_token_raises():
    with pytest.raises(ValueError, match="label-only tokens"):
        _resolve_curve_or_surface_suffix(
            container_name="FOO",
            container_kind="curve",
            x_axis_kind_name="LABEL",
            roll_method_name=None,
            anchor_offset=None,
            roll_month=None,
            roll_day=None,
            fh_bh_mmdd=None,
            tenor_token="[M+0]",
            relative_name="FOO[M+0]",
            as_of_date=date(2026, 5, 23),
        )


def test_offset_container_missing_roll_method_raises():
    with pytest.raises(ValueError, match="no roll_method"):
        _resolve_curve_or_surface_suffix(
            container_name="FOO",
            container_kind="curve",
            x_axis_kind_name="OFFSET_M",
            roll_method_name=None,
            anchor_offset=None,
            roll_month=None,
            roll_day=None,
            fh_bh_mmdd=None,
            tenor_token="[M+0]",
            relative_name="FOO[M+0]",
            as_of_date=date(2026, 5, 23),
        )


def test_offset_container_requires_signed_offset_token_raises():
    with pytest.raises(ValueError, match="requires a signed integer offset"):
        _resolve_curve_or_surface_suffix(
            container_name="FOO",
            container_kind="curve",
            x_axis_kind_name="OFFSET_M",
            roll_method_name="CALENDAR_PERIOD",
            anchor_offset=1,
            roll_month=None,
            roll_day=None,
            fh_bh_mmdd=None,
            tenor_token="[ON]",  # label-only token, no offset
            relative_name="FOO[ON]",
            as_of_date=date(2026, 5, 23),
        )


def test_unsupported_axis_kind_roll_method_pair_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_curve_or_surface_suffix(
            container_name="FOO",
            container_kind="curve",
            x_axis_kind_name="OFFSET_M",  # OFFSET_M + ROLL_ON_MMDD not registered
            roll_method_name="ROLL_ON_MMDD",
            anchor_offset=None,
            roll_month=4,
            roll_day=1,
            fh_bh_mmdd=None,
            tenor_token="[M+0]",
            relative_name="FOO[M+0]",
            as_of_date=date(2026, 5, 23),
        )


def test_label_container_returns_bare_label():
    result = _resolve_curve_or_surface_suffix(
        container_name="FOO",
        container_kind="curve",
        x_axis_kind_name="LABEL",
        roll_method_name=None,
        anchor_offset=None,
        roll_month=None,
        roll_day=None,
        fh_bh_mmdd=None,
        tenor_token="[3M]",
        relative_name="FOO[3M]",
        as_of_date=date(2026, 5, 23),
    )
    assert result == "3M"


# ---------------------------------------------------------------------------
# RiskFactorAbsToRelNameIdMap construction and lookups
# ---------------------------------------------------------------------------


def test_abs_to_rel_map_basic_lookups():
    m = RiskFactorAbsToRelNameIdMap(
        as_of_date=date(2026, 5, 23),
        abs_to_rf_id={"A_202606": 1, "B": 2},
        abs_to_rel_name={"A_202606": "A[M+0]", "B": "B"},
    )
    assert m.get_rf_id("A_202606") == 1
    assert m.get_rf_id("A[M+0]") == 1
    assert m.get_rf_id_for_abs_name("A_202606") == 1
    assert m.get_rf_id_for_rel_name("A[M+0]") == 1
    assert m.get_abs_name_for_rel_name("A[M+0]") == "A_202606"
    assert m.get_rel_name_for_abs_name("A_202606") == "A[M+0]"
    assert m.num_risk_factors == 2


def test_abs_to_rel_map_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="disagree"):
        RiskFactorAbsToRelNameIdMap(
            as_of_date=date(2026, 5, 23),
            abs_to_rf_id={"A": 1},
            abs_to_rel_name={"B": "B"},
        )


def test_abs_to_rel_map_rejects_duplicate_relative_name():
    with pytest.raises(ValueError, match="Duplicate relative name"):
        RiskFactorAbsToRelNameIdMap(
            as_of_date=date(2026, 5, 23),
            abs_to_rf_id={"A_202606": 1, "A_202607": 2},
            abs_to_rel_name={"A_202606": "A[M+0]", "A_202607": "A[M+0]"},
        )


def test_abs_to_rel_map_list_lookups():
    m = RiskFactorAbsToRelNameIdMap(
        as_of_date=date(2026, 5, 23),
        abs_to_rf_id={"A": 1, "B": 2},
        abs_to_rel_name={"A": "A", "B": "B"},
    )
    assert m.get_rf_id_for_rel_name(["A", "B"]) == [1, 2]
    assert m.get_abs_name_for_rel_name(["A", "B"]) == ["A", "B"]


# ---------------------------------------------------------------------------
# End-to-end against the real canonical fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_static_config():
    dict_of_pl = {
        name: pl.DataFrame(table_data)
        for name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }
    return compile_risk_static_config_from_dict_of_dfs(dict_of_pl)


def test_build_mapping_end_to_end_resolves_every_risk_factor(real_static_config):
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date=date(2026, 5, 23)
    )
    expected_count = (
        len(real_static_config.node_meta_dict)
        + len(real_static_config.curve_point_meta_dict)
        + len(real_static_config.surface_point_meta_dict)
    )
    assert mapping.num_risk_factors == expected_count


def test_build_mapping_pins_canonical_resolutions(real_static_config):
    """Bit-for-bit oracle: the documented resolutions must not drift."""
    m = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date=date(2026, 5, 23)
    )
    assert (
        m.get_abs_name_for_rel_name("BRENT_CRUDE_FWD[M+0]") == "BRENT_CRUDE_FWD_202606"
    )
    assert (
        m.get_abs_name_for_rel_name("BRENT_CRUDE_FWD[M+3]") == "BRENT_CRUDE_FWD_202609"
    )
    assert (
        m.get_abs_name_for_rel_name("CO2_EUA_ANNUAL_FWD[Y+0]")
        == "CO2_EUA_ANNUAL_FWD_2026"
    )
    assert (
        m.get_abs_name_for_rel_name("CO2_EUA_ANNUAL_FWD[Y+1]")
        == "CO2_EUA_ANNUAL_FWD_2027"
    )
    assert (
        m.get_abs_name_for_rel_name("POWER_CH_BASELOAD_HY[HY+0]")
        == "POWER_CH_BASELOAD_HY_2026_BH"
    )
    assert (
        m.get_abs_name_for_rel_name("POWER_CH_BASELOAD_HY[HY+1]")
        == "POWER_CH_BASELOAD_HY_2027_FH"
    )


def test_build_mapping_accepts_iso_string_as_of_date(real_static_config):
    from_str = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date="2026-05-23"
    )
    from_date = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date=date(2026, 5, 23)
    )
    assert from_str.abs_to_rf_id == from_date.abs_to_rf_id


def test_build_mapping_none_as_of_date_raises(real_static_config):
    with pytest.raises(ValueError, match="required"):
        build_risk_factor_name_mapping_for_as_of_date(
            risk_static_config=real_static_config, as_of_date=None
        )


def test_build_mapping_node_abs_name_equals_rel_name(real_static_config):
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date=date(2026, 5, 23)
    )
    for node in real_static_config.node_meta_dict.values():
        assert mapping.get_rel_name_for_abs_name(node.node_name) == node.node_name
        assert mapping.get_rf_id(node.node_name) == int(node.node_id)


def test_build_mapping_different_as_of_dates_can_produce_different_names(
    real_static_config,
):
    m1 = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date=date(2026, 5, 23)
    )
    m2 = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=real_static_config, as_of_date=date(2027, 5, 23)
    )
    assert m1.abs_to_rf_id.keys() != m2.abs_to_rf_id.keys()
    for node in real_static_config.node_meta_dict.values():
        assert node.node_name in m1.abs_to_rf_id
        assert node.node_name in m2.abs_to_rf_id
        assert m1.get_rf_id(node.node_name) == m2.get_rf_id(node.node_name)
