# src/riskquant/config/builder.py
"""
Builder module for risk configuration.
Isolates Pandas and DataFrame memory from the runtime config objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.config.static_config.schema_objects import (
    AliasSchema,
    AssetClass,
    AssetSubclass,
    BusinessDayConvention,
    Ccy,
    CompoundingConvention,
    ContainerMeta,
    DayCountConvention,
    ExtrapolationMethod,
    HolidayCalendar,
    InterpolationMethod,
    RiskFactor,
    ShiftType,
    SpotRollRule,
    SurfaceYAxisType,
    TenorSchema,
    UnitScale,
    UnitType,
)
from riskquant.market_data.risk_factors.shift_types import (
    validate_shift_types_dict,
)


@dataclass(frozen=True, slots=True)
class RiskStaticConfigFrames:
    """
    Temporary container that holds pandas DataFrames loaded from Excel.
    Call .compile() to create a runtime RiskStaticConfig that holds no DataFrames.

    The tab names in the Excel workbook must match the field names in this dataclass, which are:
    - container_meta,
    - risk_factors,
    - tenor_schema,
    - alias_schema,
    - spot_roll_rules,
    - asset_classes,
    - asset_subclasses,
    - unit_types,
    - unit_scales,
    - surface_y_axis_schema,
    - ccy,
    - shift_types,
    - interpolation_methods,
    - extrapolation_methods,
    - holiday_calendars,
    - day_count_conventions,
    - business_day_conventions,
    - compounding_conventions.

    """

    # Core
    container_meta: pd.DataFrame
    risk_factors: pd.DataFrame

    # Schemas
    tenor_schema: pd.DataFrame
    alias_schema: pd.DataFrame
    spot_roll_rules: pd.DataFrame
    surface_y_axis_schema: pd.DataFrame

    # Lookups
    asset_classes: pd.DataFrame
    asset_subclasses: pd.DataFrame
    unit_types: pd.DataFrame
    unit_scales: pd.DataFrame
    ccy: pd.DataFrame
    shift_types: pd.DataFrame
    interpolation_methods: pd.DataFrame
    extrapolation_methods: pd.DataFrame

    # Calendars / conventions
    holiday_calendars: pd.DataFrame
    day_count_conventions: pd.DataFrame
    business_day_conventions: pd.DataFrame
    compounding_conventions: pd.DataFrame

    def compile(self) -> RiskStaticConfig:
        """
        Compile validated DataFrames into compact Python structures and return runtime config.
        """

        # --- Helper for fast, null-safe DataFrame iteration ---
        def _iter_df(df: pd.DataFrame) -> list[dict]:
            if df is None or df.empty:
                return []
            # Replaces pandas <NA> and np.nan with native Python None for clean extraction
            return df.where(pd.notnull(df), None).to_dict("records")

        # ======================= #
        # --- LOOKUPS & CONVS --- #
        # ======================= #
        asset_classes_dict = {
            str(r["asset_class"]): AssetClass(
                asset_class=str(r["asset_class"]),
                asset_class_id=int(r["asset_class_id"]),
            )
            for r in _iter_df(self.asset_classes)
        }

        asset_subclasses_dict = {
            str(r["asset_subclass"]): AssetSubclass(
                asset_subclass=str(r["asset_subclass"]),
                asset_subclass_id=int(r["asset_subclass_id"]),
                asset_class=str(r["asset_class"]),
            )
            for r in _iter_df(self.asset_subclasses)
        }

        unit_types_dict = {
            str(r["unit_type_name"]): UnitType(
                unit_type_name=str(r["unit_type_name"]),
                unit_type_name_id=int(r["unit_type_name_id"]),
            )
            for r in _iter_df(self.unit_types)
        }

        unit_scales_dict = {
            str(r["unit_scale_name"]): UnitScale(
                unit_scale_name=str(r["unit_scale_name"]),
                unit_scale_name_id=int(r["unit_scale_name_id"]),
                scale_to_base=float(r["scale_to_base"]),
            )
            for r in _iter_df(self.unit_scales)
        }

        day_count_conventions_dict = {
            str(r["day_count_convention"]): DayCountConvention(
                day_count_convention=str(r["day_count_convention"]),
                day_count_convention_id=int(r["day_count_convention_id"]),
            )
            for r in _iter_df(self.day_count_conventions)
        }

        business_day_conventions_dict = {
            str(r["business_day_convention"]): BusinessDayConvention(
                business_day_convention=str(r["business_day_convention"]),
                business_day_convention_id=int(r["business_day_convention_id"]),
            )
            for r in _iter_df(self.business_day_conventions)
        }

        compounding_conventions_dict = {
            str(r["compounding_convention"]): CompoundingConvention(
                compounding_convention=str(r["compounding_convention"]),
                compounding_convention_id=int(r["compounding_convention_id"]),
                compounding_type=str(r["compounding_type"]),
                frequency_per_year=(
                    float(r["frequency_per_year"])
                    if r.get("frequency_per_year") is not None
                    else None
                ),
            )
            for r in _iter_df(self.compounding_conventions)
        }

        interpolation_methods_dict = {
            str(r["interpolation_method"]): InterpolationMethod(
                interpolation_method=str(r["interpolation_method"]),
                interpolation_method_id=int(r["interpolation_method_id"]),
            )
            for r in _iter_df(self.interpolation_methods)
        }

        extrapolation_methods_dict = {
            str(r["extrapolation_method"]): ExtrapolationMethod(
                extrapolation_method=str(r["extrapolation_method"]),
                extrapolation_method_id=int(r["extrapolation_method_id"]),
            )
            for r in _iter_df(self.extrapolation_methods)
        }

        # Fixed: Now mapping using the actual schema objects
        ccy_dict = {
            str(r["ccy"]).upper(): Ccy(
                ccy=str(r["ccy"]).upper(), ccy_id=int(r["ccy_id"])
            )
            for r in _iter_df(self.ccy)
        }

        shift_types_dict = {
            str(r["shift_type"]).upper(): ShiftType(
                shift_type=str(r["shift_type"]).upper(),
                shift_type_id=int(r["shift_type_id"]),
            )
            for r in _iter_df(self.shift_types)
        }
        # Pass a mapped `{str: int}` dict safely into the validator to prevent it from breaking
        validate_shift_types_dict(
            {k: v.shift_type_id for k, v in shift_types_dict.items()}
        )

        # HOLIDAY CALENDARS

        # Get df.
        holiday_df = self.holiday_calendars

        # print(holiday_df.to_string())

        # Assuming df is your loaded holiday calendar DataFrame
        # 1. Ensure the dates are actual Python date objects (not pandas Timestamps or strings)
        holiday_df["holiday_date"] = pd.to_datetime(
            holiday_df["holiday_date"]
        ).dt.date

        # 2. Group by the calendar name and aggregate the dates into a set
        # We also rename the output column to 'holiday_dates' to match builder.py
        aggregated_calendars = (
            holiday_df.groupby("holiday_calendar_name")["holiday_date"]
            .apply(set)
            .reset_index(name="holiday_dates")
        )

        # 3. Now you can convert this to a list of dicts to feed your builder!
        calendar_records = aggregated_calendars.to_dict(orient="records")

        # Example of what calendar_records looks like now:
        # [
        #     {'holiday_calendar_name': 'NL', 'holiday_dates': {datetime.date(2026, 1, 1)}},
        #     {'holiday_calendar_name': 'US', 'holiday_dates': {datetime.date(2026, 1, 1)}}
        # ]

        # Instantiating the HolidayCalendar schema
        holiday_calendars_dict = {
            str(r["holiday_calendar_name"]): HolidayCalendar(
                holiday_calendar_name=str(r["holiday_calendar_name"]),
                holiday_dates=(
                    set(r["holiday_dates"])
                    if r.get("holiday_dates") is not None
                    else set()
                ),
            )
            for r in calendar_records
        }

        # ======================= #
        # --- CORE META & RF  --- #
        # ======================= #
        container_meta_dict: Dict[str, ContainerMeta] = {}
        for r in _iter_df(self.container_meta):
            if not bool(r.get("active", 1)):
                continue

            c_name = str(r["container_name"])
            base_ccy = r.get("base_ccy")
            quote_ccy = str(r["quote_ccy"])
            base_ccy = str(base_ccy) if base_ccy is not None else quote_ccy

            container_meta_dict[c_name] = ContainerMeta(
                container_name=c_name,
                container_name_id=int(r["container_name_id"]),
                base_ccy=base_ccy,
                quote_ccy=quote_ccy,
                unit_type_name=str(r["unit_type_name"]),
                unit_scale_name=str(r["unit_scale_name"]),
                container_dim=int(r["container_dim"]),
                asset_class=str(r["asset_class"]),
                asset_subclass=str(r["asset_subclass"]),
                tenor_type_name=r.get("tenor_type_name"),
                spot_roll_rule_name=r.get("spot_roll_rule_name"),
                alias_type_name=r.get("alias_type_name"),
                surface_y_axis_type_name=r.get("surface_y_axis_type_name"),
                shift_type_name=str(r["shift_type_name"]),
                interp_method_name_x=r.get("interp_method_name_x"),
                interp_method_name_y=r.get("interp_method_name_y"),
                extrap_method_name_x=r.get("extrap_method_name_x"),
                extrap_method_name_y=r.get("extrap_method_name_y"),
            )

        risk_factors_dict: Dict[str, RiskFactor] = {}
        for r in _iter_df(self.risk_factors):
            if not bool(r.get("active", 1)):
                continue

            rf_name = str(r["risk_factor_name"])
            risk_factors_dict[rf_name] = RiskFactor(
                risk_factor_name=rf_name,
                risk_factor_name_id=int(r["risk_factor_name_id"]),
                tenor_token=r.get("tenor_token"),
                container_name=str(r["container_name"]),
            )

        # ======================= #
        # --- SCHEMAS / RULES --- #
        # ======================= #

        # TENOR SCHEMA
        tenor_schema_dict = {
            str(r["tenor_type_name"]): TenorSchema(
                tenor_type_name=str(r["tenor_type_name"]),
                tenor_unit=str(r["tenor_unit"]),
                digits=int(r["digits"]),
                token_regex=(
                    re.compile(str(r["token_regex"]))
                    if pd.notna(r.get("token_regex"))
                    else None
                ),
                abs_name_regex=(
                    re.compile(str(r["abs_name_regex"]))
                    if pd.notna(r.get("abs_name_regex"))
                    else None
                ),
                rel_name_regex=(
                    re.compile(str(r["rel_name_regex"]))
                    if pd.notna(r.get("rel_name_regex"))
                    else None
                ),
                abs_name_example=str(r.get("abs_name_example", "")),
                rel_name_example=str(r.get("rel_name_example", "")),
            )
            for r in _iter_df(self.tenor_schema)
        }

        # ALIAS SCHEMA
        alias_schema_dict = {
            str(r["alias_type_name"]): AliasSchema(
                alias_type_name=str(r["alias_type_name"]),
                alias_kind=str(r["alias_kind"]),
                alias_unit=r.get("alias_unit"),
            )
            for r in _iter_df(self.alias_schema)
        }

        # SURFACE Y-AXIS SCHEMA
        surface_y_axis_schema_dict = {
            str(r["y_axis_type_name"]): SurfaceYAxisType(
                y_axis_type_name=str(r["y_axis_type_name"])
            )
            for r in _iter_df(self.surface_y_axis_schema)
        }

        # SPOT ROLL RULES SCHEMA
        spot_roll_rules_dict: Dict[Tuple[str, str], SpotRollRule] = {}
        for r in _iter_df(self.spot_roll_rules):
            key = (str(r["spot_roll_rule_name"]), str(r["tenor_type_name"]))
            spot_roll_rules_dict[key] = SpotRollRule(
                spot_roll_rule_name=key[0],
                tenor_type_name=key[1],
                spot_roll_method=str(r["spot_roll_method"]),
                offset_in_tenor_units=r.get("offset_in_tenor_units"),
                roll_month=r.get("roll_month"),
                roll_day=r.get("roll_day"),
                fh_bh_ddmm=r.get("fh_bh_ddmm"),
                post_roll_shift=r.get("post_roll_shift"),
            )

        # # ======================= #
        # # --- PROXIES         --- #
        # # ======================= #

        # # Container proxies
        # container_proxies_dict = {
        #     str(r["container_name"]): ContainerProxyMap(
        #         container_name=str(r["container_name"]),
        #         proxy_container_name=str(r["proxy_container_name"]),
        #         proxy_container_vol_scaling_factor=float(
        #             r.get("proxy_container_vol_scaling_factor", 1.0) or 1.0
        #         ),
        #     )
        #     for r in _iter_df(self.container_proxies)
        # }

        # # Risk factor proxies
        # risk_factor_proxies_dict = {
        #     str(r["risk_factor_name"]): RiskFactorProxyMap(
        #         risk_factor_name=str(r["risk_factor_name"]),
        #         proxy_risk_factor_name=str(r["proxy_risk_factor_name"]),
        #         proxy_risk_factor_vol_scaling_factor=float(
        #             r.get("proxy_risk_factor_vol_scaling_factor", 1.0) or 1.0
        #         ),
        #     )
        #     for r in _iter_df(self.risk_factor_proxies)
        # }

        # ======================= #
        # --- FINALIZE CONFIG --- #
        # ======================= #
        return RiskStaticConfig(
            container_meta_dict=container_meta_dict,
            risk_factors_dict=risk_factors_dict,
            tenor_schema_dict=tenor_schema_dict,
            alias_schema_dict=alias_schema_dict,
            spot_roll_rules_dict=spot_roll_rules_dict,
            surface_y_axis_schema_dict=surface_y_axis_schema_dict,
            asset_classes_dict=asset_classes_dict,
            asset_subclasses_dict=asset_subclasses_dict,
            unit_types_dict=unit_types_dict,
            unit_scales_dict=unit_scales_dict,
            day_count_conventions_dict=day_count_conventions_dict,
            business_day_conventions_dict=business_day_conventions_dict,
            compounding_conventions_dict=compounding_conventions_dict,
            interpolation_methods_dict=interpolation_methods_dict,
            extrapolation_methods_dict=extrapolation_methods_dict,
            ccy_dict=ccy_dict,
            shift_types_dict=shift_types_dict,
            holiday_calendars_dict=holiday_calendars_dict,
        )
