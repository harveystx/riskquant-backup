# src/riskquant/config/static_config/compile.py
"""
Single entry-point for building a RiskStaticConfig from source data.

Pipeline:
    dict_of_dfs  ->  validate  ->  compile (typed scalars)  ->  RiskStaticConfig

The 7 required sheets are:
    ccy, holiday_calendars,
    node_meta, curve_meta, curve_point_meta,
    surface_meta, surface_point_meta.

There is no spot_roll_rule sheet: each curve / surface owns its
tenor-resolution convention directly.

All system-defined taxonomy (asset class, unit, shift, interp/extrap,
x/y axis kind, roll method, discount basis) lives in
riskquant.config.enums / riskquant.market_data and is resolved by the
RiskStaticConfig constructor.

Nothing in this module is imported by the valuation or simlab packages.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from riskquant.config.static_config.loader import (
    load_risk_static_config_xlsx_into_dict,
)
from riskquant.config.static_config.schema_objects import (
    Ccy,
    CurveMeta,
    CurvePointMeta,
    HolidayCalendar,
    NodeMeta,
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


def _iter(df: pl.DataFrame) -> list[dict]:
    """
    Materialize a DataFrame into plain dicts with None for missing values.

    Per-column conversion avoids forcing every column to a lowest-common
    Python object type; each column keeps its native dtype until the final
    per-row dict assembly.
    """
    if df is None or df.is_empty():
        return []

    cols: dict[str, list] = {}
    for col in df.columns:
        series = df[col]
        lst = series.to_list()
        cols[col] = [
            None if (v is None or (isinstance(v, float) and v != v)) else v for v in lst
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


def _require_integral(v: float) -> int:
    """Return v as int iff it is integral; raise otherwise. No silent truncation."""
    if not v.is_integer():
        raise ValueError(
            f"Expected an integral numeric value, got {v!r}. "
            f"Non-integral values are rejected, not truncated."
        )
    return int(v)


def _int(v: object) -> int:
    if v is None:
        raise ValueError("Expected a non-null int cell, got None.")
    if isinstance(v, bool):
        raise ValueError(f"Expected an int cell, got bool {v!r}.")
    if isinstance(v, float):
        # An Excel numeric cell materialises as float; accept only when it is
        # exactly integral (1.0 -> 1) and reject 1.5 rather than truncate it.
        return _require_integral(v)
    # `v` is a raw materialised cell value (object) by construction -- the
    # boundary genuinely cannot narrow it further. `cast` documents that the
    # runtime behaviour is unchanged: `int(v)` still raises naturally on a
    # non-numeric value. Narrower than a blanket suppression, which
    # copilot-instructions.md §16 prohibits. (RQ-AUD-021.)
    return int(cast("int | str", v))


def _opt_int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"Expected an int cell, got bool {v!r}.")
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return _require_integral(v)
    # See _int above: cast documents the boundary, not a behaviour change.
    return int(cast("int | str", v))


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


def _compile_ccy(df: pl.DataFrame) -> dict[str, Ccy]:
    out: dict[str, Ccy] = {}
    for r in _iter(df):
        name = _str(r["ccy_name"], upper=True)
        out[name] = Ccy(ccy_name=name, ccy_id=_int(r["ccy_id"]))
    return out


def _compile_holiday_calendars(df: pl.DataFrame) -> dict[str, HolidayCalendar]:
    out: dict[str, HolidayCalendar] = {}
    if df is None or df.is_empty():
        return out

    epoch = DATE_EPOCH if isinstance(DATE_EPOCH, date) else date(1970, 1, 1)

    holiday_date = df["holiday_date"]
    if holiday_date.dtype in (pl.Date, pl.Datetime):
        date_col = holiday_date.cast(pl.Date)
    else:
        # Raw fixture / xlsx string cells: parse explicitly rather than
        # relying on an implicit engine-specific date inference.
        date_col = holiday_date.cast(pl.Utf8).str.strptime(
            pl.Date, format="%Y-%m-%d", strict=True
        )

    ord_col = (date_col - epoch).dt.total_days().cast(pl.Int32)
    work = df.with_columns(ord_col.alias("_ord"))

    for (cal_name,), group in work.group_by(["holiday_calendar_name"]):
        out[str(cal_name)] = HolidayCalendar(
            holiday_calendar_name=str(cal_name),
            holiday_dates_ord=np.sort(group["_ord"].unique().to_numpy()),
        )
    return out


def _compile_node_meta(df: pl.DataFrame) -> dict[int, NodeMeta]:
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


def _compile_curve_meta(df: pl.DataFrame) -> dict[int, CurveMeta]:
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
            roll_method=_opt_str(r.get("roll_method"), upper=True),
            anchor_offset=_opt_int(r.get("anchor_offset")),
            roll_month=_opt_int(r.get("roll_month")),
            roll_day=_opt_int(r.get("roll_day")),
            fh_bh_mmdd=_opt_str(r.get("fh_bh_mmdd")),
            # D079 clause 3: declared once on the curve; points inherit it.
            # Absent means "not yet declared" - never inferred from quote_type.
            curve_input_type=_opt_str(r.get("curve_input_type"), upper=True),
            discount_basis=_str(r["discount_basis"], upper=True),
            is_default_discount_for_ccy=_bool(r.get("is_default_discount_for_ccy")),
        )
    return out


def _compile_curve_point_meta(
    df: pl.DataFrame,
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


def _compile_surface_meta(df: pl.DataFrame) -> dict[int, SurfaceMeta]:
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
            roll_method=_opt_str(r.get("roll_method"), upper=True),
            anchor_offset=_opt_int(r.get("anchor_offset")),
            roll_month=_opt_int(r.get("roll_month")),
            roll_day=_opt_int(r.get("roll_day")),
            fh_bh_mmdd=_opt_str(r.get("fh_bh_mmdd")),
            y_axis_kind=_str(r["y_axis_kind"], upper=True),
            y_interp_method=_str(r["y_interp_method"], upper=True),
            y_extrap_method=_str(r["y_extrap_method"], upper=True),
        )
    return out


def _compile_surface_point_meta(
    df: pl.DataFrame,
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
    dict_of_dfs: dict[str, pl.DataFrame],
) -> RiskStaticConfig:
    """
    Validate and compile a dict of DataFrames into a RiskStaticConfig.

    Parameters
    ----------
    dict_of_dfs
        Sheet-name -> DataFrame produced by the loader. All 8 required
        sheets must be present and well-formed; see REQUIRED_SHEETS in
        validate.py (7 tables).

    Returns
    -------
    RiskStaticConfig
        Fully compiled, dataframe-free runtime config with every derived
        index already built.
    """
    validate_static_config_dfs(dict_of_dfs)

    ccy_dict = _compile_ccy(dict_of_dfs["ccy"])
    holiday_calendars_dict = _compile_holiday_calendars(
        dict_of_dfs["holiday_calendars"]
    )
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
        Path to the Excel workbook with the 7 required sheets.

    Returns
    -------
    RiskStaticConfig
        Compiled runtime configuration.
    """
    dict_of_dfs = load_risk_static_config_xlsx_into_dict(path=path)
    return compile_risk_static_config_from_dict_of_dfs(dict_of_dfs)
