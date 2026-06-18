# src/riskquant/config/static_config/compile.py
"""
Single entry-point for building a RiskStaticConfig from source data.

Pipeline:
    dict_of_dfs  ->  validate  ->  compile (typed scalars)  ->  RiskStaticConfig

The 7 (+ ccy + spot_roll_rule + holiday_calendars = 10) required sheets are:
    ccy, holiday_calendars, spot_roll_rule,
    node_meta, curve_meta, curve_point_meta,
    surface_meta, surface_point_meta.

All system-defined taxonomy (asset class, unit, shift, interp/extrap,
x/y axis kind, roll method, discount basis) lives in
riskquant.config.enums / riskquant.market_data and is resolved by the
RiskStaticConfig constructor.

Nothing in this module is imported by the valuation or simlab packages.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from riskquant.config.static_config.loader import (
    load_risk_static_config_xlsx_into_dict,
)
from riskquant.config.static_config.schema_objects import (
    Ccy,
    CurveMeta,
    CurvePointMeta,
    HolidayCalendar,
    NodeMeta,
    SpotRollRule,
    SurfaceMeta,
    SurfacePointMeta,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.config.static_config.validate import validate_static_config_dfs
from riskquant.constants import DATE_EPOCH

# ---------------------------------------------------------------------------
# Cell extraction primitives
# ---------------------------------------------------------------------------
# The only place raw DataFrame cell values become typed Python scalars.
# Deliberately thin — no business logic, just safe type coercion.


def _iter(df: pd.DataFrame) -> list[dict]:
    """
    Materialize a DataFrame into plain dicts with None for missing values.

    Per-column conversion avoids .where(..., None), which would force
    every column to object dtype.
    """
    if df is None or df.empty:
        return []

    cols: dict[str, list] = {}
    for col in df.columns:
        lst = df[col].tolist()
        cols[col] = [
            None if (v is pd.NA or (isinstance(v, float) and v != v)) else v
            for v in lst
        ]
    keys = list(cols.keys())
    return [
        dict(zip(keys, row, strict=True)) for row in zip(*cols.values(), strict=True)
    ]


def _str(v: object, *, upper: bool = False) -> str:
    if v is None:
        raise ValueError("Expected a non-null string cell, got None.")
    s = str(v).strip()
    return s.upper() if upper else s


def _opt_str(v: object, *, upper: bool = False) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    return s.upper() if upper else s


def _int(v: object) -> int:
    if v is None:
        raise ValueError("Expected a non-null int cell, got None.")
    return int(v)  # type: ignore[arg-type]


def _opt_int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return int(v)  # type: ignore[arg-type]


def _bool(v: object) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    s = str(v).strip().upper()
    if s in ("TRUE", "1", "Y", "YES"):
        return True
    if s in ("FALSE", "0", "N", "NO", ""):
        return False
    raise ValueError(f"Cannot interpret {v!r} as boolean.")


# ---------------------------------------------------------------------------
# Per-table compilers
# ---------------------------------------------------------------------------


def _compile_ccy(df: pd.DataFrame) -> dict[str, Ccy]:
    out: dict[str, Ccy] = {}
    for r in _iter(df):
        name = _str(r["ccy_name"], upper=True)
        out[name] = Ccy(ccy_name=name, ccy_id=_int(r["ccy_id"]))
    return out


def _compile_holiday_calendars(df: pd.DataFrame) -> dict[str, HolidayCalendar]:
    out: dict[str, HolidayCalendar] = {}
    if df is None or df.empty:
        return out

    epoch_ts = pd.Timestamp(DATE_EPOCH)
    dt_series = pd.to_datetime(df["holiday_date"], format="%Y-%m-%d", errors="raise")
    work = df.copy()
    work["_ord"] = (dt_series - epoch_ts).dt.days.astype(np.int32)

    for cal_name, group in work.groupby("holiday_calendar_name"):
        out[str(cal_name)] = HolidayCalendar(
            holiday_calendar_name=str(cal_name),
            holiday_dates_ord=np.sort(group["_ord"].unique().astype(np.int32)),
        )
    return out


def _compile_spot_roll_rule(df: pd.DataFrame) -> dict[str, SpotRollRule]:
    out: dict[str, SpotRollRule] = {}
    for r in _iter(df):
        name = _str(r["rule_name"], upper=True)
        out[name] = SpotRollRule(
            rule_name=name,
            roll_method=_str(r["roll_method"], upper=True),
            offset_in_tenor_units=_int(r["offset_in_tenor_units"]),
            roll_month=_opt_int(r.get("roll_month")),
            roll_day=_opt_int(r.get("roll_day")),
            fh_bh_mmdd=_opt_str(r.get("fh_bh_mmdd")),
            post_roll_shift=_int(r["post_roll_shift"]),
        )
    return out


def _compile_node_meta(df: pd.DataFrame) -> dict[int, NodeMeta]:
    out: dict[int, NodeMeta] = {}
    for r in _iter(df):
        node_id = _int(r["node_id"])
        out[node_id] = NodeMeta(
            node_id=node_id,
            node_name=_str(r["node_name"], upper=True),
            quote_ccy=_str(r["quote_ccy"], upper=True),
            asset_class=_str(r["asset_class"], upper=True),
            asset_subclass=_str(r["asset_subclass"], upper=True),
            quote_type=_str(r["quote_type"], upper=True),
            unit_scale=_str(r["unit_scale"], upper=True),
            shift_type=_str(r["shift_type"], upper=True),
            fx_base_ccy=_opt_str(r.get("fx_base_ccy"), upper=True),
        )
    return out


def _compile_curve_meta(df: pd.DataFrame) -> dict[int, CurveMeta]:
    out: dict[int, CurveMeta] = {}
    for r in _iter(df):
        curve_id = _int(r["curve_id"])
        out[curve_id] = CurveMeta(
            curve_id=curve_id,
            curve_name=_str(r["curve_name"], upper=True),
            quote_ccy=_str(r["quote_ccy"], upper=True),
            asset_class=_str(r["asset_class"], upper=True),
            asset_subclass=_str(r["asset_subclass"], upper=True),
            quote_type=_str(r["quote_type"], upper=True),
            unit_scale=_str(r["unit_scale"], upper=True),
            shift_type=_str(r["shift_type"], upper=True),
            x_axis_kind=_str(r["x_axis_kind"], upper=True),
            x_interp_method=_str(r["x_interp_method"], upper=True),
            x_extrap_method=_str(r["x_extrap_method"], upper=True),
            spot_roll_rule_name=_opt_str(r.get("spot_roll_rule_name"), upper=True),
            discount_basis=_str(r["discount_basis"], upper=True),
            is_default_discount_for_ccy=_bool(r.get("is_default_discount_for_ccy")),
        )
    return out


def _compile_curve_point_meta(
    df: pd.DataFrame,
    curve_meta: dict[int, CurveMeta],
) -> dict[int, CurvePointMeta]:
    out: dict[int, CurvePointMeta] = {}
    for r in _iter(df):
        curve_id = _int(r["curve_id"])
        rf_id = _int(r["rf_id"])
        out[rf_id] = CurvePointMeta(
            rf_id=rf_id,
            curve_id=curve_id,
            curve_name=curve_meta[curve_id].curve_name,
            tenor_token=_str(r["tenor_token"], upper=True),
        )
    return out


def _compile_surface_meta(df: pd.DataFrame) -> dict[int, SurfaceMeta]:
    out: dict[int, SurfaceMeta] = {}
    for r in _iter(df):
        surface_id = _int(r["surface_id"])
        out[surface_id] = SurfaceMeta(
            surface_id=surface_id,
            surface_name=_str(r["surface_name"], upper=True),
            quote_ccy=_str(r["quote_ccy"], upper=True),
            asset_class=_str(r["asset_class"], upper=True),
            asset_subclass=_str(r["asset_subclass"], upper=True),
            quote_type=_str(r["quote_type"], upper=True),
            unit_scale=_str(r["unit_scale"], upper=True),
            shift_type=_str(r["shift_type"], upper=True),
            x_axis_kind=_str(r["x_axis_kind"], upper=True),
            x_interp_method=_str(r["x_interp_method"], upper=True),
            x_extrap_method=_str(r["x_extrap_method"], upper=True),
            spot_roll_rule_name=_opt_str(r.get("spot_roll_rule_name"), upper=True),
            y_axis_kind=_str(r["y_axis_kind"], upper=True),
            y_interp_method=_str(r["y_interp_method"], upper=True),
            y_extrap_method=_str(r["y_extrap_method"], upper=True),
        )
    return out


def _compile_surface_point_meta(
    df: pd.DataFrame,
    surface_meta: dict[int, SurfaceMeta],
) -> dict[int, SurfacePointMeta]:
    out: dict[int, SurfacePointMeta] = {}
    for r in _iter(df):
        surface_id = _int(r["surface_id"])
        rf_id = _int(r["rf_id"])
        out[rf_id] = SurfacePointMeta(
            rf_id=rf_id,
            surface_id=surface_id,
            surface_name=surface_meta[surface_id].surface_name,
            x_tenor_token=_str(r["x_tenor_token"], upper=True),
            y_label=_str(r["y_label"], upper=False),
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_risk_static_config_from_dict_of_dfs(
    dict_of_dfs: dict[str, pd.DataFrame],
) -> RiskStaticConfig:
    """
    Validate and compile a dict of DataFrames into a RiskStaticConfig.

    Parameters
    ----------
    dict_of_dfs
        Sheet-name -> DataFrame produced by the loader. All 8 required
        sheets must be present and well-formed; see REQUIRED_SHEETS in
        validate.py.

    Returns
    -------
    RiskStaticConfig
        Fully compiled, Pandas-free runtime config with every derived
        index already built.
    """
    validate_static_config_dfs(dict_of_dfs)

    ccy_dict = _compile_ccy(dict_of_dfs["ccy"])
    holiday_calendars_dict = _compile_holiday_calendars(
        dict_of_dfs["holiday_calendars"]
    )
    spot_roll_rule_dict = _compile_spot_roll_rule(dict_of_dfs["spot_roll_rule"])
    node_meta_dict = _compile_node_meta(dict_of_dfs["node_meta"])
    curve_meta_dict = _compile_curve_meta(dict_of_dfs["curve_meta"])
    curve_point_meta_dict = _compile_curve_point_meta(
        dict_of_dfs["curve_point_meta"], curve_meta_dict
    )
    surface_meta_dict = _compile_surface_meta(dict_of_dfs["surface_meta"])
    surface_point_meta_dict = _compile_surface_point_meta(
        dict_of_dfs["surface_point_meta"], surface_meta_dict
    )

    return RiskStaticConfig(
        ccy_dict=ccy_dict,
        holiday_calendars_dict=holiday_calendars_dict,
        spot_roll_rule_dict=spot_roll_rule_dict,
        node_meta_dict=node_meta_dict,
        curve_meta_dict=curve_meta_dict,
        curve_point_meta_dict=curve_point_meta_dict,
        surface_meta_dict=surface_meta_dict,
        surface_point_meta_dict=surface_point_meta_dict,
    )


def compile_risk_static_config_from_xlsx(path: str | Path) -> RiskStaticConfig:
    """
    Load, validate, and compile an .xlsx workbook into a RiskStaticConfig.

    Parameters
    ----------
    path
        Path to the Excel workbook with the 8 required sheets.

    Returns
    -------
    RiskStaticConfig
        Compiled runtime configuration.
    """
    dict_of_dfs = load_risk_static_config_xlsx_into_dict(path=path)
    return compile_risk_static_config_from_dict_of_dfs(dict_of_dfs)
