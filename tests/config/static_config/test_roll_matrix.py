# tests/config/static_config/test_roll_matrix.py
"""
Complete (XAxisKind, RollMethod) resolution matrix.

Proves that EVERY pair in ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS
validates, compiles, and resolves end-to-end against the real canonical
fixture, and that the matrix covers exactly the authoritative pair set (no
pair that validation accepts but the resolver cannot handle, and none the
resolver handles but validation rejects). Also proves representative invalid
pairs fail fast.

The three OFFSET curves in the canonical fixture each carry one pair by
default; the other three pairs are exercised by surgically re-declaring one
curve's roll convention. The canonical fixture itself is never grown.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from riskquant.config.enums.roll_methods import (
    ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS,
    RollMethod,
)
from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.config.static_config.validate import validate_static_config_dfs
from riskquant.exceptions import ConfigurationError
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS as CANONICAL_DFS,
)

_AS_OF = date(2026, 5, 23)


def _baseline() -> dict[str, pl.DataFrame]:
    return {name: df.clone() for name, df in CANONICAL_DFS.items()}


def _apply_roll(
    dfs: dict[str, pl.DataFrame],
    curve_name: str,
    *,
    roll_method: str,
    anchor_offset: int | None = None,
    roll_month: int | None = None,
    roll_day: int | None = None,
    fh_bh_mmdd: str | None = None,
) -> dict[str, pl.DataFrame]:
    """Re-declare one curve's complete roll convention, nulling the rest."""
    cm = dfs["curve_meta"]

    def setcol(col: str, val: object) -> pl.Expr:
        return (
            pl.when(pl.col("curve_name") == curve_name)
            .then(pl.lit(val))
            .otherwise(pl.col(col))
            .alias(col)
        )

    dfs["curve_meta"] = cm.with_columns(
        [
            setcol("roll_method", roll_method),
            setcol("anchor_offset", anchor_offset),
            setcol("roll_month", roll_month),
            setcol("roll_day", roll_day),
            setcol("fh_bh_mmdd", fh_bh_mmdd),
        ]
    )
    return dfs


# (pair, curve, roll-config, probe relative name, expected absolute name).
# Covers all six ALLOWED pairs: three canonical defaults + three re-declared.
_MATRIX: list[tuple[XAxisKind, RollMethod, str, dict, str, str]] = [
    (
        XAxisKind.OFFSET_M,
        RollMethod.CALENDAR_PERIOD,
        "BRENT_CRUDE_FWD",
        {"roll_method": "CALENDAR_PERIOD", "anchor_offset": 1},
        "BRENT_CRUDE_FWD[M+0]",
        "BRENT_CRUDE_FWD_202606",
    ),
    (
        XAxisKind.OFFSET_M,
        RollMethod.HALF_YEAR_BY_DATE,
        "BRENT_CRUDE_FWD",
        {"roll_method": "HALF_YEAR_BY_DATE", "fh_bh_mmdd": "0401|1001"},
        "BRENT_CRUDE_FWD[M+0]",
        "BRENT_CRUDE_FWD_20261001",
    ),
    (
        XAxisKind.OFFSET_Y,
        RollMethod.CALENDAR_PERIOD,
        "CO2_EUA_ANNUAL_FWD",
        {"roll_method": "CALENDAR_PERIOD", "anchor_offset": 0},
        "CO2_EUA_ANNUAL_FWD[Y+0]",
        "CO2_EUA_ANNUAL_FWD_2026",
    ),
    (
        XAxisKind.OFFSET_Y,
        RollMethod.ROLL_ON_MMDD,
        "CO2_EUA_ANNUAL_FWD",
        {"roll_method": "ROLL_ON_MMDD", "roll_month": 4, "roll_day": 1},
        "CO2_EUA_ANNUAL_FWD[Y+0]",
        "CO2_EUA_ANNUAL_FWD_2026",
    ),
    (
        XAxisKind.OFFSET_HY,
        RollMethod.HALF_YEAR_BY_DATE,
        "POWER_CH_BASELOAD_HY",
        {"roll_method": "HALF_YEAR_BY_DATE", "fh_bh_mmdd": "0401|1001"},
        "POWER_CH_BASELOAD_HY[HY+0]",
        "POWER_CH_BASELOAD_HY_20261001",
    ),
    (
        XAxisKind.OFFSET_HY,
        RollMethod.HALF_YEAR_BY_LABEL,
        "POWER_CH_BASELOAD_HY",
        {"roll_method": "HALF_YEAR_BY_LABEL", "fh_bh_mmdd": "0401|1001"},
        "POWER_CH_BASELOAD_HY[HY+0]",
        "POWER_CH_BASELOAD_HY_2026_BH",
    ),
]


def test_matrix_covers_every_allowed_pair_exactly():
    covered = {(xk, rm) for xk, rm, *_ in _MATRIX}
    assert covered == ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS


@pytest.mark.parametrize(
    "x_kind,method,curve,cfg,rel_name,expected",
    _MATRIX,
    ids=[f"{xk.name}+{rm.name}" for xk, rm, *_ in _MATRIX],
)
def test_allowed_pair_validates_compiles_resolves(
    x_kind, method, curve, cfg, rel_name, expected
):
    dfs = _apply_roll(_baseline(), curve, **cfg)
    validate_static_config_dfs(dfs)  # must not raise
    rsc = compile_risk_static_config_from_dict_of_dfs(dfs)  # must not raise
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=rsc, as_of_date=_AS_OF
    )
    assert mapping.get_abs_name_for_rel_name(rel_name) == expected


@pytest.mark.parametrize(
    "curve,cfg,match",
    [
        # LABEL curve given a roll method -> extraneous.
        (
            "USD_SOFR_OIS",
            {"roll_method": "CALENDAR_PERIOD", "anchor_offset": 1},
            "LABEL containers with populated",
        ),
        # OFFSET_M + ROLL_ON_MMDD is not an allowed pair.
        (
            "BRENT_CRUDE_FWD",
            {"roll_method": "ROLL_ON_MMDD", "roll_month": 4, "roll_day": 1},
            "disallowed",
        ),
    ],
)
def test_invalid_pair_fails_fast(curve, cfg, match):
    dfs = _apply_roll(_baseline(), curve, **cfg)
    with pytest.raises(ConfigurationError, match=match):
        validate_static_config_dfs(dfs)
