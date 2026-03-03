# src/riskquant/config/schema_objects.py
"""
Schema objects for config data.

Design intent:
- Immutable, slot-based value objects used by config loading/compiling/validation.
- Keep this module lightweight: avoid importing pandas/numpy here to prevent
  config schema from dragging heavy deps into runtime/model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Pattern, Tuple

from riskquant.utils.printer import print_success_message


# =============== #
# --- LOOKUPS --- #
# =============== #


@dataclass(frozen=True, slots=True)
class Ccy:
    ccy: str
    ccy_id: int


@dataclass(frozen=True, slots=True)
class AssetClass:
    asset_class: str
    asset_class_id: int


@dataclass(frozen=True, slots=True)
class AssetSubclass:
    asset_subclass: str
    asset_subclass_id: int
    asset_class: str  # FK -> AssetClass.asset_class (validated in compile step)


@dataclass(frozen=True, slots=True)
class UnitType:
    unit_type_name: str
    unit_type_name_id: int


@dataclass(frozen=True, slots=True)
class UnitScale:
    unit_scale_name: str
    unit_scale_name_id: int
    scale_to_base: float


@dataclass(frozen=True, slots=True)
class ShiftType:
    shift_type: str
    shift_type_id: int


@dataclass(frozen=True, slots=True)
class InterpolationMethod:
    interpolation_method: str
    interpolation_method_id: int


@dataclass(frozen=True, slots=True)
class ExtrapolationMethod:
    extrapolation_method: str
    extrapolation_method_id: int


@dataclass(frozen=True, slots=True)
class SurfaceYAxisType:
    y_axis_type_name: str


# =================== #
# --- CONVENTIONS --- #
# =================== #


@dataclass(frozen=True, slots=True)
class DayCountConvention:
    day_count_convention: str
    day_count_convention_id: int


@dataclass(frozen=True, slots=True)
class BusinessDayConvention:
    business_day_convention: str
    business_day_convention_id: int


@dataclass(frozen=True, slots=True)
class CompoundingConvention:
    compounding_convention: str
    compounding_convention_id: int
    compounding_type: str
    frequency_per_year: Optional[float]


# ===================== #
# --- CORE ENTITIES --- #
# ===================== #


@dataclass(frozen=True, slots=True)
class ContainerMeta:
    """
    Metadata for a container (curve, surface, FX spot, etc).

    Notes:
    - tenor_type_name / spot_roll_rule_name / alias_type_name are typically None for container_dim == 0 (e.g., FX).
    - surface_y_axis_type_name / interp_method_name_y / extrap_method_name_y are typically None for non-surfaces.
    """

    container_name: str
    container_name_id: int

    base_ccy: str
    quote_ccy: str

    unit_type_name: str
    unit_scale_name: str

    container_dim: int  # 0 or 1 for now (extend later)

    asset_class: str
    asset_subclass: str

    tenor_type_name: Optional[str]
    spot_roll_rule_name: Optional[str]
    alias_type_name: Optional[str]

    surface_y_axis_type_name: Optional[str]

    shift_type_name: str

    interp_method_name_x: Optional[str]
    interp_method_name_y: Optional[str]
    extrap_method_name_x: Optional[str]
    extrap_method_name_y: Optional[str]

    def __post_init__(self) -> None:
        if self.container_dim not in (0, 1):
            raise ValueError(
                f"container_dim must be 0 or 1 for now; got {self.container_dim!r} "
                f"for container_name={self.container_name!r}"
            )

        # Basic "dim implies curve metadata" checks.
        if self.container_dim == 0:
            # Scalar containers (FX spot etc) should not specify tenor/roll/alias.
            if self.tenor_type_name is not None:
                raise ValueError(
                    "tenor_type_name must be None when container_dim==0 "
                    f"(container_name={self.container_name!r}, tenor_type_name={self.tenor_type_name!r})"
                )
            if self.spot_roll_rule_name is not None:
                raise ValueError(
                    "spot_roll_rule_name must be None when container_dim==0 "
                    f"(container_name={self.container_name!r}, spot_roll_rule_name={self.spot_roll_rule_name!r})"
                )
            if self.alias_type_name is not None:
                raise ValueError(
                    "alias_type_name must be None when container_dim==0 "
                    f"(container_name={self.container_name!r}, alias_type_name={self.alias_type_name!r})"
                )

        # Strong defaults that catch config typos early.
        # (Hard errors only for the really obvious ones.)
        if self.asset_class == "FX":
            if self.container_dim != 0:
                raise ValueError(
                    "FX containers must have container_dim==0 "
                    f"(container_name={self.container_name!r}, container_dim={self.container_dim!r})"
                )
            if self.unit_type_name != "FX_RATE":
                raise ValueError(
                    "FX containers must have unit_type_name=='FX_RATE' "
                    f"(container_name={self.container_name!r}, unit_type_name={self.unit_type_name!r})"
                )

        if self.asset_class == "IR":
            # IR should never be FX_RATE; this catches copy/paste errors.
            if self.unit_type_name == "FX_RATE":
                raise ValueError(
                    "IR containers cannot have unit_type_name=='FX_RATE' "
                    f"(container_name={self.container_name!r})"
                )


@dataclass(frozen=True, slots=True)
class RiskFactor:
    risk_factor_name: str
    risk_factor_name_id: int
    tenor_token: Optional[str]
    container_name: str

    def __post_init__(self) -> None:
        # Remove tenor token as a suffix only (safer than replace)
        rf_name_prefix = (
            self.risk_factor_name.removesuffix(self.tenor_token)
            if self.tenor_token
            else self.risk_factor_name
        )

        if rf_name_prefix != self.container_name:
            raise ValueError(
                f"risk_factor_name ({self.risk_factor_name!r}) must match "
                f"container_name ({self.container_name!r}) "
                f"after removing tenor_token ({self.tenor_token!r})."
            )


# ===================== #
# --- SCHEMA TABLES --- #
# ===================== #


@dataclass(frozen=True, slots=True)
class TenorSchema:
    tenor_type_name: str
    tenor_units: str
    digits: int
    regex: Pattern[str]


@dataclass(frozen=True, slots=True)
class AliasSchema:
    alias_type_name: str
    alias_kind: str  # OFFSET or LABEL
    alias_unit: Optional[str]


@dataclass(frozen=True, slots=True)
class SpotRollRule:
    spot_roll_rule_name: str
    tenor_type_name: str
    spot_roll_method: str
    offset_in_tenor_units: Optional[int]
    roll_month: Optional[int]
    roll_day: Optional[int]
    fh_bh_ddmm: Optional[str]
    post_roll_shift: Optional[int]


# ===================== #
# --- PROXY MAPPING --- #
# ===================== #


@dataclass(frozen=True, slots=True)
class ContainerProxyMap:
    container_name: str
    proxy_container_name: str
    proxy_container_vol_scaling_factor: Optional[float]


@dataclass(frozen=True, slots=True)
class RiskFactorProxyMap:
    risk_factor_name: str
    proxy_risk_factor_name: str
    proxy_risk_factor_vol_scaling_factor: Optional[float]


# ======================== #
# --- HOLIDAY CALENDAR --- #
# ======================== #


@dataclass(frozen=True, slots=True)
class HolidayCalendar:
    holiday_calendar_name: str
    holiday_dates: Optional[Tuple[date, ...]]


if __name__ == "__main__":
    Ccy(ccy="USDEUR", ccy_id=1)

    # Print success message
    print_success_message()
