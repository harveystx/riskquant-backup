# src/riskquant/config/enums/unit_types.py
"""
Unit type and unit scale taxonomy.

System-defined constants — not user configuration.
Users reference these string values in node_meta / curve_meta /
surface_meta under unit_type and unit_scale.
"""

from __future__ import annotations

from enum import Enum


class UnitType(str, Enum):
    PRICE = "PRICE"
    FX_RATE = "FX_RATE"
    RATE = "RATE"
    VOL = "VOL"
    INDEX = "INDEX"
    SPREAD = "SPREAD"


class UnitScale(str, Enum):
    DECIMAL = "DECIMAL"
    PERCENT = "PERCENT"
    BP = "BP"


# Multiplier converting a value stored at this scale into its base (decimal) form.
# e.g. a rate stored as 5.0 PERCENT -> 0.05 base.
UNIT_SCALE_TO_BASE: dict[UnitScale, float] = {
    UnitScale.DECIMAL: 1.0,
    UnitScale.PERCENT: 0.01,
    UnitScale.BP: 0.0001,
}

VALID_UNIT_TYPES: frozenset[str] = frozenset(e.value for e in UnitType)
VALID_UNIT_SCALES: frozenset[str] = frozenset(e.value for e in UnitScale)
