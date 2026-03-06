# src/riskquant/config/configs.py
"""
Runtime configuration containers.
Strictly native Python structures (no Pandas/DataFrames) for maximum efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from riskquant.config.schema_objects import (
    AliasSchema,
    AssetClass,
    AssetSubclass,
    BusinessDayConvention,
    Ccy,
    CompoundingConvention,
    ContainerMeta,
    ContainerProxyMap,
    DayCountConvention,
    ExtrapolationMethod,
    HolidayCalendar,
    InterpolationMethod,
    RiskFactor,
    RiskFactorProxyMap,
    ShiftType,
    SpotRollRule,
    SurfaceYAxisType,
    TenorSchema,
    UnitScale,
    UnitType,
)
from riskquant.utils.tenors import parse_bracket_token


@dataclass(frozen=True, slots=True)
class RiskStaticConfig:
    """
    Container for risk static configuration tables for runtime use.
    No pandas DataFrames here - just pure Python structures for maximum efficiency.
    """

    # Core
    container_meta_dict: Dict[str, ContainerMeta] = field(
        init=True,
        repr=False,
    )
    risk_factors_dict: Dict[str, RiskFactor] = field(
        init=True,
        repr=False,
    )

    # Schemas & Rules
    tenor_schema_dict: Dict[str, TenorSchema] = field(
        init=True,
        repr=False,
    )
    alias_schema_dict: Dict[str, AliasSchema] = field(
        init=True,
        repr=False,
    )
    spot_roll_rules_dict: Dict[Tuple[str, str], SpotRollRule] = field(
        init=True, repr=False
    )
    surface_y_axis_schema_dict: Dict[str, SurfaceYAxisType] = field(
        init=True, repr=False
    )

    # Lookups
    asset_classes_dict: Dict[str, AssetClass] = field(
        init=True,
        repr=False,
    )
    asset_subclasses_dict: Dict[str, AssetSubclass] = field(
        init=True, repr=False
    )
    unit_types_dict: Dict[str, UnitType] = field(
        init=True,
        repr=False,
    )
    unit_scales_dict: Dict[str, UnitScale] = field(
        init=True,
        repr=False,
    )
    day_count_conventions_dict: Dict[str, DayCountConvention] = field(
        init=True, repr=False
    )
    business_day_conventions_dict: Dict[str, BusinessDayConvention] = field(
        init=True, repr=False
    )
    compounding_conventions_dict: Dict[str, CompoundingConvention] = field(
        init=True, repr=False
    )
    interpolation_methods_dict: Dict[str, InterpolationMethod] = field(
        init=True, repr=False
    )
    extrapolation_methods_dict: Dict[str, ExtrapolationMethod] = field(
        init=True, repr=False
    )
    ccy_dict: Dict[str, Ccy] = field(
        init=True,
        repr=False,
    )
    shift_types_dict: Dict[str, ShiftType] = field(
        init=True,
        repr=False,
    )

    # Proxies & Calendars
    curve_proxies_dict: Dict[str, ContainerProxyMap] = field(
        init=True,
        repr=False,
    )
    risk_factor_proxies_dict: Dict[str, RiskFactorProxyMap] = field(
        init=True,
        repr=False,
    )
    holiday_calendars_dict: Dict[str, HolidayCalendar] = field(
        init=True,
        repr=False,
    )

    # --- INTERNAL CACHES --- #
    # Initialized as None; populated lazily on first access using object.__setattr__
    _container_name_to_container_id: Optional[Dict[str, int]] = field(
        default=None,
        repr=False,
        init=False,
    )
    _rf_name_to_rf_id: Optional[Dict[str, int]] = field(
        default=None,
        repr=False,
        init=False,
    )
    _rf_id_to_proxy_rf_id: Optional[Dict[int, Tuple[int, float]]] = field(
        default=None,
        repr=False,
        init=False,
    )
    _rf_id_to_risk_factor: Optional[Dict[int, RiskFactor]] = field(
        default=None,
        repr=False,
        init=False,
    )
    _rf_id_to_shift_type_id: Optional[Dict[int, int]] = field(
        default=None,
        repr=False,
        init=False,
    )
    _curve_tenor_index: Optional[
        Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]
    ] = field(
        default=None,
        repr=False,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.container_meta_dict:
            raise ValueError(
                "container_meta_dict is empty. Check your container_meta sheet and 'active' flags."
            )
        if not self.risk_factors_dict:
            raise ValueError(
                "risk_factors_dict is empty. Check your risk_factors sheet and 'active' flags."
            )

    def summary(self) -> str:
        lines = [
            "=== Risk Static Config Summary ===",
            f"Containers Mapped: {len(self.container_meta_dict)}",
            f"Risk Factors: {len(self.risk_factors_dict)}",
            f"Asset Classes: {len(self.asset_classes_dict)}",
            f"Currencies: {len(self.ccy_dict)}",
            f"Shift Types: {len(self.shift_types_dict)}",
            f"Container Proxies: {len(self.curve_proxies_dict)}",
            f"Risk Factor Proxies: {len(self.risk_factor_proxies_dict)}",
            "==================================",
        ]
        return "\n".join(lines)

    def get_risk_factor_id(self, risk_factor_name: str) -> int:
        return self.risk_factors_dict[risk_factor_name].risk_factor_name_id

    @property
    def container_name_to_container_id(self) -> Dict[str, int]:
        """A build-once cache: container_name -> container_id"""
        if self._container_name_to_container_id is None:
            object.__setattr__(
                self,
                "_container_name_to_container_id",
                {
                    k: v.container_name_id
                    for k, v in self.container_meta_dict.items()
                },
            )
        return self._container_name_to_container_id

    @property
    def rf_name_to_rf_id(self) -> Dict[str, int]:
        """A build-once cache: risk_factor_name -> risk_factor_name_id"""
        if self._rf_name_to_rf_id is None:
            object.__setattr__(
                self,
                "_rf_name_to_rf_id",
                {
                    k: v.risk_factor_name_id
                    for k, v in self.risk_factors_dict.items()
                },
            )
        return self._rf_name_to_rf_id

    @property
    def rf_id_to_proxy_rf_id(self) -> Dict[int, Tuple[int, float]]:
        """A build-once cache of RF ID-level proxy overrides: required_rf_id -> (proxy_rf_id, vol_scale)"""
        if self._rf_id_to_proxy_rf_id is None:
            rf_name_map = self.rf_name_to_rf_id
            out: Dict[int, Tuple[int, float]] = {}
            for p in self.risk_factor_proxies_dict.values():
                rf_id = int(rf_name_map[p.risk_factor_name])
                proxy_rf_id = int(rf_name_map[p.proxy_risk_factor_name])
                scale = float(p.proxy_risk_factor_vol_scaling_factor or 1.0)
                out[rf_id] = (proxy_rf_id, scale)
            object.__setattr__(self, "_rf_id_to_proxy_rf_id", out)
        return self._rf_id_to_proxy_rf_id

    @property
    def rf_id_to_risk_factor(self) -> Dict[int, RiskFactor]:
        """Cached mapping: risk_factor_name_id (int) -> RiskFactor"""
        if self._rf_id_to_risk_factor is None:
            out: Dict[int, RiskFactor] = {}
            for rf in self.risk_factors_dict.values():
                rf_id = int(rf.risk_factor_name_id)
                if rf_id in out:
                    raise ValueError(
                        f"Duplicate risk_factor_name_id detected: {rf_id}"
                    )
                out[rf_id] = rf
            object.__setattr__(self, "_rf_id_to_risk_factor", out)
        return self._rf_id_to_risk_factor

    @property
    def rf_id_to_shift_type_id(self) -> Dict[int, int]:
        """Cached mapping: risk_factor_name_id (int) -> shift_type_id (int)"""
        if self._rf_id_to_shift_type_id is None:
            out: Dict[int, int] = {}
            for rf in self.risk_factors_dict.values():
                rf_id = int(rf.risk_factor_name_id)
                cm = self.container_meta_dict.get(rf.container_name)
                if cm is None:
                    raise KeyError(
                        f"curve_name={rf.container_name!r} not found in container_meta_dict."
                    )

                st_obj = self.shift_types_dict.get(cm.shift_type_name)
                if st_obj is None:
                    raise KeyError(
                        f"shift_type_name={cm.shift_type_name!r} not found in shift_types_dict."
                    )

                out[rf_id] = int(st_obj.shift_type_id)
            object.__setattr__(self, "_rf_id_to_shift_type_id", out)
        return self._rf_id_to_shift_type_id

    @property
    def curve_tenor_index(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
        """Cached per-curve tenor lookup used by curve-level proxy resolution."""
        if self._curve_tenor_index is not None:
            return self._curve_tenor_index

        offsets: Dict[str, list[int]] = {}
        offset_ids: Dict[str, list[int]] = {}
        labels: Dict[str, Dict[str, int]] = {}

        for rf_id, rf in self.rf_id_to_risk_factor.items():
            rf_curve_name = rf.container_name
            rf_tenor_token = rf.tenor_token

            if rf_tenor_token is None:
                labels.setdefault(rf_curve_name, {})[""] = int(rf_id)
                continue

            rf_unit, rf_offset = parse_bracket_token(rf_tenor_token)

            if rf_offset is None:
                labels.setdefault(rf_curve_name, {})[rf_unit] = int(rf_id)
            else:
                offsets.setdefault(rf_curve_name, []).append(int(rf_offset))
                offset_ids.setdefault(rf_curve_name, []).append(int(rf_id))

        out: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = {}

        for rf_curve_name, rf_offset_list in offsets.items():
            rf_ids_list = offset_ids[rf_curve_name]
            rf_offsets_arr = np.asarray(rf_offset_list, dtype=np.int32)
            rf_ids_arr = np.asarray(rf_ids_list, dtype=np.int32)

            rf_offset_idx = np.argsort(rf_offsets_arr, kind="mergesort")

            out[rf_curve_name] = (
                rf_offsets_arr[rf_offset_idx],
                rf_ids_arr[rf_offset_idx],
                labels.get(rf_curve_name, {}),
            )

        for rf_curve_name, rf_curve_label_to_rf_id in labels.items():
            if rf_curve_name not in out:
                out[rf_curve_name] = (
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    rf_curve_label_to_rf_id,
                )

        object.__setattr__(self, "_curve_tenor_index", out)
        return out


if __name__ == "__main__":
    pass
