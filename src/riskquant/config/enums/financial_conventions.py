# src/riskquant/config/enums/financial_conventions.py
"""
Standard financial convention enumerations.

These are system-defined constants — not user configuration.
Users reference these string values in instrument definitions and
curve configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Day count conventions
# ---------------------------------------------------------------------------


class DayCount(StrEnum):
    ACT_365F = "ACT_365F"
    ACT_360 = "ACT_360"
    DC_30_360 = "30_360"


# ---------------------------------------------------------------------------
# Business day conventions
# ---------------------------------------------------------------------------


class BusinessDayConvention(StrEnum):
    FOLLOWING = "FOLLOWING"
    MOD_FOLLOWING = "MOD_FOLLOWING"
    PRECEDING = "PRECEDING"
    MOD_PRECEDING = "MOD_PRECEDING"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Compounding conventions
# ---------------------------------------------------------------------------


class CompoundingType(StrEnum):
    SIMPLE = "SIMPLE"
    CONTINUOUS = "CONTINUOUS"
    PERIODIC = "PERIODIC"


@dataclass(frozen=True, slots=True)
class CompoundingConvention:
    name: str
    compounding_type: CompoundingType
    # None for SIMPLE and CONTINUOUS — frequency is not applicable.
    frequency_per_year: int | None


COMPOUNDING_CONVENTIONS: dict[str, CompoundingConvention] = {
    "SIMPLE": CompoundingConvention("SIMPLE", CompoundingType.SIMPLE, None),
    "CONTINUOUS": CompoundingConvention("CONTINUOUS", CompoundingType.CONTINUOUS, None),
    "ANNUAL": CompoundingConvention("ANNUAL", CompoundingType.PERIODIC, 1),
    "SEMI_ANNUAL": CompoundingConvention("SEMI_ANNUAL", CompoundingType.PERIODIC, 2),
    "QUARTERLY": CompoundingConvention("QUARTERLY", CompoundingType.PERIODIC, 4),
    "MONTHLY": CompoundingConvention("MONTHLY", CompoundingType.PERIODIC, 12),
    "BIWEEKLY": CompoundingConvention("BIWEEKLY", CompoundingType.PERIODIC, 26),
    "WEEKLY": CompoundingConvention("WEEKLY", CompoundingType.PERIODIC, 52),
}

# Flat sets for O(1) membership checks in validation.
VALID_DAY_COUNTS: frozenset[str] = frozenset(e.value for e in DayCount)
VALID_BUSINESS_DAY_CONVENTIONS: frozenset[str] = frozenset(
    e.value for e in BusinessDayConvention
)
VALID_COMPOUNDING_CONVENTIONS: frozenset[str] = frozenset(
    COMPOUNDING_CONVENTIONS.keys()
)


if __name__ == "__main__":
    print("=" * 70)
    print(" RISKQUANT FINANCIAL CONVENTIONS REVIEW ".center(70, "#"))
    print("=" * 70)

    # 1. Print Day Count Conventions
    print("\n--- Day Count Conventions ---")
    for dc in DayCount:
        print(f"  ├── {dc.value}")

    # 2. Print Business Day Conventions
    print("\n--- Business Day Conventions ---")
    for bdc in BusinessDayConvention:
        print(f"  ├── {bdc.value}")

    # 3. Print Compounding Conventions Map
    print("\n--- Compounding Conventions Summary ---")
    print(f"{'Key':<15} | {'Type':<12} | {'Frequency / Year':<16}")
    print("-" * 50)
    for key, conv in COMPOUNDING_CONVENTIONS.items():
        freq_str = (
            str(conv.frequency_per_year)
            if conv.frequency_per_year is not None
            else "N/A"
        )
        print(f"{key:<15} | {conv.compounding_type.value:<12} | {freq_str:<16}")

    # 4. Integrity Checks
    print("\n--- Static Set Validation Checks ---")
    print(f"✓ Day Count Lookups         : {len(VALID_DAY_COUNTS)} values loaded.")
    print(
        f"✓ Business Day Lookups      : {len(VALID_BUSINESS_DAY_CONVENTIONS)} values loaded."
    )
    print(
        f"✓ Compounding Type Lookups  : {len(VALID_COMPOUNDING_CONVENTIONS)} keys loaded."
    )

    # Assert that all keys in the lookups match expectation
    assert len(VALID_COMPOUNDING_CONVENTIONS) == len(COMPOUNDING_CONVENTIONS)
    print("✓ Validation Integrity      : Pass")
    print("=" * 70)
