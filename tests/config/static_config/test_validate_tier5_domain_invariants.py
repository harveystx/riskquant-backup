# tests/config/static_config/test_validate_tier5_domain_invariants.py
"""
Regression tests for `static_config/validate.py`'s Tier 3/4/5 checks
(RQ-AUD-022 remediation).

Before this file, `validate_static_config_dfs`'s ~15 distinct validation
functions (enum membership, FK referential integrity, FX node naming
convention, (XAxisKind, RollMethod) allowed-pair table, discount-curve
default/basis invariants, global name/rf_id uniqueness) were exercised only
by ONE happy-path fixture (`test_full_pipeline_equivalence.py`'s baseline
compile). None of their failure paths -- the actual safety-critical part of
a validator -- had any test coverage.

Method: start from the real, known-valid canonical fixture and apply a
single surgical mutation per test, so each test exercises exactly the
targeted check without accidentally tripping an unrelated one first.
"""

from __future__ import annotations

import polars as pl
import pytest

from riskquant.config.static_config.validate import validate_static_config_dfs
from riskquant.exceptions import ConfigurationError
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)


@pytest.fixture()
def baseline() -> dict[str, pl.DataFrame]:
    return {
        name: pl.DataFrame(table_data)
        for name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }


def test_baseline_is_valid(baseline):
    validate_static_config_dfs(baseline)  # must not raise


# ---------------------------------------------------------------------------
# Tier 3 — enum membership
# ---------------------------------------------------------------------------


def test_invalid_asset_class_enum_raises(baseline):
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.when(pl.col("node_name") == "USDEUR")
        .then(pl.lit("NOT_A_REAL_ASSET_CLASS"))
        .otherwise(pl.col("asset_class"))
        .alias("asset_class")
    )
    with pytest.raises(ConfigurationError, match="Enum violation"):
        validate_static_config_dfs(baseline)


def test_null_in_non_nullable_enum_column_raises(baseline):
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.when(pl.col("node_name") == "USDEUR")
        .then(None)
        .otherwise(pl.col("quote_type"))
        .alias("quote_type")
    )
    with pytest.raises(ConfigurationError, match="non-nullable|Blank string"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 4 — foreign keys
# ---------------------------------------------------------------------------


def test_unknown_quote_ccy_fk_violation_raises(baseline):
    baseline["curve_meta"] = baseline["curve_meta"].with_columns(
        pl.when(pl.col("curve_name") == "USD_SOFR_OIS")
        .then(pl.lit("ZZZ"))  # not a real currency in the ccy table
        .otherwise(pl.col("quote_ccy"))
        .alias("quote_ccy")
    )
    with pytest.raises(ConfigurationError, match="FK violation"):
        validate_static_config_dfs(baseline)


def test_unknown_curve_id_in_curve_point_raises(baseline):
    baseline["curve_point_meta"] = baseline["curve_point_meta"].with_columns(
        pl.when(pl.col("curve_id") == baseline["curve_point_meta"]["curve_id"][0])
        .then(999_999)
        .otherwise(pl.col("curve_id"))
        .alias("curve_id")
    )
    with pytest.raises(ConfigurationError, match="FK violation"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 5 — FX node invariants
# ---------------------------------------------------------------------------


def test_fx_rate_node_missing_base_ccy_raises(baseline):
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.when(pl.col("node_name") == "USDEUR")
        .then(None)
        .otherwise(pl.col("fx_base_ccy"))
        .alias("fx_base_ccy")
    )
    with pytest.raises(ConfigurationError, match="must specify fx_base_ccy"):
        validate_static_config_dfs(baseline)


def test_non_fx_rate_node_with_base_ccy_raises(baseline):
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.when(pl.col("node_name") == "AAPL_EQ")
        .then(pl.lit("USD"))
        .otherwise(pl.col("fx_base_ccy"))
        .alias("fx_base_ccy")
    )
    with pytest.raises(ConfigurationError, match="only FX_RATE nodes"):
        validate_static_config_dfs(baseline)


def test_fx_rate_node_name_mismatch_raises(baseline):
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.when(pl.col("node_name") == "USDEUR")
        .then(pl.lit("WRONGNAME"))
        .otherwise(pl.col("node_name"))
        .alias("node_name")
    )
    with pytest.raises(ConfigurationError, match="node_name must equal"):
        validate_static_config_dfs(baseline)


def test_duplicate_fx_pair_raises(baseline):
    # Make EURCHF's (base, quote) collide with an existing FX_RATE pair by
    # renaming it AND its base/quote to duplicate USDEUR's (USD, EUR) pair,
    # while keeping node_name consistent with fx_base_ccy+quote_ccy so the
    # name-mismatch check (tested above) doesn't fire first.
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        [
            pl.when(pl.col("node_name") == "EURCHF")
            .then(pl.lit("USD"))
            .otherwise(pl.col("fx_base_ccy"))
            .alias("fx_base_ccy"),
            pl.when(pl.col("node_name") == "EURCHF")
            .then(pl.lit("EUR"))
            .otherwise(pl.col("quote_ccy"))
            .alias("quote_ccy"),
            pl.when(pl.col("node_name") == "EURCHF")
            .then(pl.lit("USDEUR"))
            .otherwise(pl.col("node_name"))
            .alias("node_name"),
        ]
    )
    with pytest.raises(ConfigurationError, match="duplicate|Uniqueness violation"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 5 — (XAxisKind, RollMethod) axis/roll pairs
# ---------------------------------------------------------------------------


def _set_curve_col(baseline, curve_name, col, value):
    """Surgically set one column for one curve, leaving all other rows intact."""
    baseline["curve_meta"] = baseline["curve_meta"].with_columns(
        pl.when(pl.col("curve_name") == curve_name)
        .then(pl.lit(value))
        .otherwise(pl.col(col))
        .alias(col)
    )
    return baseline


def test_offset_curve_missing_roll_method_raises(baseline):
    _set_curve_col(baseline, "BRENT_CRUDE_FWD", "roll_method", None)
    with pytest.raises(ConfigurationError, match="missing roll_method"):
        validate_static_config_dfs(baseline)


def test_label_curve_with_populated_roll_method_raises(baseline):
    # USD_SOFR_OIS is a LABEL curve; giving it a roll_method must be rejected.
    _set_curve_col(baseline, "USD_SOFR_OIS", "roll_method", "CALENDAR_PERIOD")
    with pytest.raises(ConfigurationError, match="LABEL containers with populated"):
        validate_static_config_dfs(baseline)


def test_disallowed_axis_roll_pair_raises(baseline):
    # BRENT_CRUDE_FWD is OFFSET_M; (OFFSET_M, ROLL_ON_MMDD) is not an allowed pair.
    _set_curve_col(baseline, "BRENT_CRUDE_FWD", "roll_method", "ROLL_ON_MMDD")
    with pytest.raises(ConfigurationError, match="disallowed"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 5 — method-specific roll-parameter presence / absence / ranges
# ---------------------------------------------------------------------------


def test_calendar_period_missing_anchor_offset_raises(baseline):
    # BRENT_CRUDE_FWD is CALENDAR_PERIOD; anchor_offset is required.
    _set_curve_col(baseline, "BRENT_CRUDE_FWD", "anchor_offset", None)
    with pytest.raises(ConfigurationError, match="requires"):
        validate_static_config_dfs(baseline)


def test_offset_curve_extraneous_param_raises(baseline):
    # BRENT_CRUDE_FWD is CALENDAR_PERIOD; roll_month is a forbidden param.
    _set_curve_col(baseline, "BRENT_CRUDE_FWD", "roll_month", 5)
    with pytest.raises(ConfigurationError, match="forbids"):
        validate_static_config_dfs(baseline)


def test_roll_on_mmdd_month_out_of_range_raises(baseline):
    # CO2_EUA_ANNUAL_FWD is ROLL_ON_MMDD (roll_month=4).
    _set_curve_col(baseline, "CO2_EUA_ANNUAL_FWD", "roll_month", 13)
    with pytest.raises(ConfigurationError, match="roll_month 13 out of range"):
        validate_static_config_dfs(baseline)


def test_roll_on_mmdd_day_out_of_range_raises(baseline):
    _set_curve_col(baseline, "CO2_EUA_ANNUAL_FWD", "roll_day", 0)
    with pytest.raises(ConfigurationError, match="roll_day 0 out of range"):
        validate_static_config_dfs(baseline)


def test_half_year_malformed_fh_bh_mmdd_raises(baseline):
    # POWER_CH_BASELOAD_HY is HALF_YEAR_BY_LABEL; back-half month 13 is invalid.
    _set_curve_col(baseline, "POWER_CH_BASELOAD_HY", "fh_bh_mmdd", "0401|1301")
    with pytest.raises(ConfigurationError, match="invalid fh_bh_mmdd"):
        validate_static_config_dfs(baseline)


def test_invalid_roll_method_enum_raises(baseline):
    _set_curve_col(baseline, "BRENT_CRUDE_FWD", "roll_method", "NOT_A_METHOD")
    with pytest.raises(ConfigurationError, match="Enum violation"):
        validate_static_config_dfs(baseline)


def test_half_year_extraneous_roll_month_raises(baseline):
    # POWER_CH_BASELOAD_HY is HALF_YEAR_BY_LABEL; roll_month is forbidden there.
    _set_curve_col(baseline, "POWER_CH_BASELOAD_HY", "roll_month", 4)
    with pytest.raises(ConfigurationError, match="forbids"):
        validate_static_config_dfs(baseline)


def test_non_integral_anchor_offset_raises(baseline):
    # A Float64 anchor_offset of 1.5 must be rejected, never truncated to 1.
    _set_curve_col(baseline, "BRENT_CRUDE_FWD", "anchor_offset", 1.5)
    with pytest.raises(ConfigurationError, match="non-integral"):
        validate_static_config_dfs(baseline)


def test_label_curve_with_populated_roll_param_raises(baseline):
    # USD_SOFR_OIS is a LABEL curve (roll_method NULL); an anchor_offset must
    # still be rejected -- LABEL axes carry no roll metadata.
    _set_curve_col(baseline, "USD_SOFR_OIS", "anchor_offset", 5)
    with pytest.raises(ConfigurationError, match="LABEL axis"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 5 — discount-curve invariants
# ---------------------------------------------------------------------------


def test_default_discount_flag_on_non_discount_curve_raises(baseline):
    baseline["curve_meta"] = baseline["curve_meta"].with_columns(
        pl.when(pl.col("curve_name") == "BRENT_CRUDE_FWD")  # NOT_A_DISCOUNT_CURVE
        .then(True)
        .otherwise(pl.col("is_default_discount_for_ccy"))
        .alias("is_default_discount_for_ccy")
    )
    with pytest.raises(ConfigurationError, match="non-discount curve"):
        validate_static_config_dfs(baseline)


def test_zero_defaults_for_a_currency_raises(baseline):
    baseline["curve_meta"] = baseline["curve_meta"].with_columns(
        pl.when(pl.col("curve_name") == "USD_SOFR_OIS")
        .then(False)
        .otherwise(pl.col("is_default_discount_for_ccy"))
        .alias("is_default_discount_for_ccy")
    )
    with pytest.raises(ConfigurationError, match="exactly one"):
        validate_static_config_dfs(baseline)


def test_two_defaults_for_a_currency_raises(baseline):
    # Must also give the second curve the firm-standard OIS basis, or the
    # earlier "default basis must equal firm standard" check fires first
    # (USD_LIBOR_3M's real basis is IBOR_3M) instead of the "exactly one
    # default" check this test targets.
    baseline["curve_meta"] = baseline["curve_meta"].with_columns(
        [
            pl.when(pl.col("curve_name").is_in(["USD_SOFR_OIS", "USD_LIBOR_3M"]))
            .then(True)
            .otherwise(pl.col("is_default_discount_for_ccy"))
            .alias("is_default_discount_for_ccy"),
            pl.when(pl.col("curve_name") == "USD_LIBOR_3M")
            .then(pl.lit("OIS"))
            .otherwise(pl.col("discount_basis"))
            .alias("discount_basis"),
        ]
    )
    with pytest.raises(ConfigurationError, match="exactly one"):
        validate_static_config_dfs(baseline)


def test_discount_curve_wrong_asset_class_raises(baseline):
    # Must also change asset_subclass to a COMMODITY-consistent value, or the
    # earlier asset_class/asset_subclass consistency check fires first
    # (asset_subclass stays IR_OIS, which no longer matches asset_class).
    baseline["curve_meta"] = baseline["curve_meta"].with_columns(
        [
            pl.when(pl.col("curve_name") == "USD_SOFR_OIS")
            .then(pl.lit("COMMODITY"))
            .otherwise(pl.col("asset_class"))
            .alias("asset_class"),
            pl.when(pl.col("curve_name") == "USD_SOFR_OIS")
            .then(pl.lit("CM_OIL"))
            .otherwise(pl.col("asset_subclass"))
            .alias("asset_subclass"),
        ]
    )
    with pytest.raises(ConfigurationError, match="asset_class=='IR'"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 5 — asset_class / asset_subclass consistency
# ---------------------------------------------------------------------------


def test_asset_class_subclass_mismatch_raises(baseline):
    # AAPL_EQ is EQUITY/EQ_SINGLE_NAME; force a COMMODITY subclass onto it
    # while asset_class stays EQUITY.
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.when(pl.col("node_name") == "AAPL_EQ")
        .then(pl.lit("CM_AGRI"))
        .otherwise(pl.col("asset_subclass"))
        .alias("asset_subclass")
    )
    with pytest.raises(ConfigurationError, match="inconsistent"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 2 — uniqueness
# ---------------------------------------------------------------------------


def test_duplicate_node_id_raises(baseline):
    ids = baseline["node_meta"]["node_id"].to_list()
    ids[1] = ids[0]  # force a collision
    baseline["node_meta"] = baseline["node_meta"].with_columns(
        pl.Series("node_id", ids)
    )
    with pytest.raises(ConfigurationError, match="Uniqueness violation"):
        validate_static_config_dfs(baseline)


def test_duplicate_curve_point_composite_key_raises(baseline):
    tokens = baseline["curve_point_meta"]["tenor_token"].to_list()
    curve_ids = baseline["curve_point_meta"]["curve_id"].to_list()
    # Force row 1's (curve_id, tenor_token) to equal row 0's.
    curve_ids[1] = curve_ids[0]
    tokens[1] = tokens[0]
    baseline["curve_point_meta"] = baseline["curve_point_meta"].with_columns(
        [pl.Series("curve_id", curve_ids), pl.Series("tenor_token", tokens)]
    )
    with pytest.raises(ConfigurationError, match="Composite uniqueness violation"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 5 — global rf_id namespace / uniqueness
# ---------------------------------------------------------------------------


def test_rf_id_collision_across_namespaces_raises(baseline):
    # Force a curve_point rf_id to collide with an existing node_id.
    existing_node_id = baseline["node_meta"]["node_id"][0]
    rf_ids = baseline["curve_point_meta"]["rf_id"].to_list()
    rf_ids[0] = int(existing_node_id)
    baseline["curve_point_meta"] = baseline["curve_point_meta"].with_columns(
        pl.Series("rf_id", rf_ids)
    )
    with pytest.raises(ConfigurationError, match="rf_id collision"):
        validate_static_config_dfs(baseline)


# ---------------------------------------------------------------------------
# Tier 1 — required sheets / columns
# ---------------------------------------------------------------------------


def test_missing_required_sheet_raises(baseline):
    del baseline["ccy"]
    with pytest.raises(ConfigurationError, match="Missing required sheet"):
        validate_static_config_dfs(baseline)


def test_missing_required_column_raises(baseline):
    baseline["node_meta"] = baseline["node_meta"].drop("asset_class")
    with pytest.raises(ConfigurationError, match="missing required columns"):
        validate_static_config_dfs(baseline)
