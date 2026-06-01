# src/riskquant/portfolio/ingest/schema.py

"""
Structural ingest schema definitions.

This module defines the canonical structural ingest contract for RiskQuant
portfolio ingestion. It validates ONLY structural invariants (column presence,
uniqueness, allowed universe). Semantic invariants (value ranges, cross-column
logic, instrument correctness) belong in the validation dispatcher.
"""

from __future__ import annotations

from typing import Final

# ==============================================================================
# CORE REQUIRED SCHEMA CONTRACT (REQUIRED FOR ALL INGEST BATCHES)
# ==============================================================================

UNIVERSAL_REQUIRED_COLS: Final[tuple[str, ...]] = (
    "uid",
    "instrument",
)

# ==============================================================================
# DOMAIN COLUMN REGISTRIES (REFLECTING DIFFERENT RISK DOMAINS)
# ==============================================================================

# Market-risk-related columns. May not be required for all instruments.
MARKET_DOMAIN_COLS: Final[tuple[str, ...]] = (
    "quantity",
    "underlying",
    "settlement_ccy",
)

# Credit-exposure linkage columns. Optional for all instruments.
CREDIT_EXPOSURE_DOMAIN_COLS: Final[tuple[str, ...]] = ("netting_set_id",)

# # JTD loss model-related columns. May not be required for all instruments.
# JTD_LOSS_MODEL_DOMAIN_COLS: Final[tuple[str, ...]] = (
#     "ead",
#     "pd",
#     "lgd",
# )

# ==============================================================================
# INSTRUMENT EXTENSION COLS
# ==============================================================================

MARKET_DOMAIN_EXTENSION_COLS: Final[tuple[str, ...]] = (
    "strike_price",
    "strike_ccy",
    "is_call",
    "expiry_date",
    "delivery_date",
    "settlement_date",
    "under_discount_curve",
    "strike_discount_curve",
    # Container-name pointer for the option's implied-vol source. Optional
    # at the structural layer (SPOT / FORWARD rows leave it None); the
    # OPTION family validator enforces non-null.
    "volatility_container_name",
)

# # `beta` is the single-factor copula loading consumed by the JTD block compilers.
# JTD_LOSS_MODEL_DOMAIN_EXTENSION_COLS: Final[tuple[str, ...]] = (
#     "beta",
#     "systemic_factor",
#     "idiosyncratic_factor",
#     "chi_squared_factor",
# )

# ==============================================================================
# GLOBAL STRUCTURAL REGISTRY
# ==============================================================================

ALL_ALLOWED_COLS_SET: Final[frozenset[str]] = frozenset(
    UNIVERSAL_REQUIRED_COLS
    + MARKET_DOMAIN_COLS
    # + JTD_LOSS_MODEL_DOMAIN_COLS
    + CREDIT_EXPOSURE_DOMAIN_COLS
    + MARKET_DOMAIN_EXTENSION_COLS
    # + JTD_LOSS_MODEL_DOMAIN_EXTENSION_COLS
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    UNIVERSAL_REQUIRED_COLS
    + MARKET_DOMAIN_COLS
    # + JTD_LOSS_MODEL_DOMAIN_COLS
    + CREDIT_EXPOSURE_DOMAIN_COLS
    + MARKET_DOMAIN_EXTENSION_COLS
    # + JTD_LOSS_MODEL_DOMAIN_EXTENSION_COLS
)

# ==============================================================================
# TYPE REGISTRIES
# ==============================================================================

STRING_COLS: Final[tuple[str, ...]] = (
    # Market-specific string columns
    "instrument",
    "underlying",
    "strike_ccy",
    "settlement_ccy",
    "under_discount_curve",
    "strike_discount_curve",
    "volatility_container_name",
    # # JTD loss model-specific string columns
    # "systemic_factor",
    # "idiosyncratic_factor",
    # "chi_squared_factor",
)

FLOAT_COLS: Final[tuple[str, ...]] = (
    # Market-specific float columns
    "quantity",
    "strike_price",
    # # JTD loss model-specific float columns
    # "ead",
    # "pd",
    # "lgd",
    # "beta",
)

INT_COLS: Final[tuple[str, ...]] = ("uid",)

# Nullable integer columns: exact integers, NA permitted (NA -> typed sentinel
# downstream). Distinct from INT_COLS, which are strictly non-nullable.
NULLABLE_INT_COLS: Final[tuple[str, ...]] = ("netting_set_id",)

FLAG_COLS: Final[tuple[str, ...]] = ("is_call",)

DATE_COLS: Final[tuple[str, ...]] = (
    "delivery_date",
    "settlement_date",
    "expiry_date",
)

DATETIME_COLS: Final[tuple[str, ...]] = ()

ALL_TYPED_COLS: Final[tuple[str, ...]] = (
    STRING_COLS
    + FLOAT_COLS
    + INT_COLS
    + NULLABLE_INT_COLS
    + FLAG_COLS
    + DATE_COLS
    + DATETIME_COLS
)

if len(ALL_TYPED_COLS) != len(set(ALL_TYPED_COLS)):
    raise RuntimeError("Column appears in multiple dtype registries.")

ALL_TYPED_COLS_SET: Final[frozenset[str]] = frozenset(ALL_TYPED_COLS)

typed_missing = ALL_ALLOWED_COLS_SET - ALL_TYPED_COLS_SET
if typed_missing:
    raise RuntimeError(
        f"Allowed columns missing dtype registry:\n{sorted(typed_missing)}"
    )

illegal_typed = ALL_TYPED_COLS_SET - ALL_ALLOWED_COLS_SET
if illegal_typed:
    raise RuntimeError(
        f"Typed registry contains unknown columns:\n{sorted(illegal_typed)}"
    )

# ==============================================================================
# NON-NULLABLE REGISTRY
# ==============================================================================

NON_NULLABLE_COLS: Final[tuple[str, ...]] = UNIVERSAL_REQUIRED_COLS

NON_NULLABLE_COLS_SET: Final[frozenset[str]] = frozenset(NON_NULLABLE_COLS)

unknown_non_nullable = NON_NULLABLE_COLS_SET - ALL_ALLOWED_COLS_SET
if unknown_non_nullable:
    raise RuntimeError(
        f"NON_NULLABLE_COLS contains unknown cols:\n{sorted(unknown_non_nullable)}"
    )

if __name__ == "__main__":
    print(ALL_TYPED_COLS)
