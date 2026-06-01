# src/riskquant/config/enums/discount_basis.py
"""
Discount basis taxonomy for IR discount curves.

A single currency carries multiple discount curves: one per OIS / IBOR
tenor / CSA collateral choice. The basis enum is the per-(currency,
basis) discriminator the ingest layer uses to resolve a trade's
discount-curve linkage when it specifies a non-default basis.

Sentinel:
  NOT_A_DISCOUNT_CURVE - explicitly marks a curve as non-discount. All
  curve_meta rows carry a basis; non-discount curves carry this value.
  The validator rejects any attempt to use a non-discount curve as a
  discount-curve linkage.

Standard convention: the firm-wide default discount basis is OIS. The
static-config validator enforces exactly one is_default_discount=True
curve per currency that has any discount curve, and that the default's
basis is OIS.
"""

from __future__ import annotations

from enum import IntEnum


class DiscountBasis(IntEnum):
    NOT_A_DISCOUNT_CURVE = 0

    OIS = 1

    IBOR_1M = 11
    IBOR_3M = 13
    IBOR_6M = 16

    CSA_USD = 30
    CSA_EUR = 31
    CSA_GBP = 32
    CSA_CHF = 33
    CSA_JPY = 34

    INTERNAL_FVA = 90


# Firm-wide standard. The default discount curve per currency must use this basis.
FIRM_DEFAULT_DISCOUNT_BASIS: DiscountBasis = DiscountBasis.OIS


VALID_DISCOUNT_BASES: frozenset[str] = frozenset(b.name for b in DiscountBasis)

DISCOUNT_BASES_EXCLUDING_SENTINEL: frozenset[str] = frozenset(
    b.name for b in DiscountBasis if b is not DiscountBasis.NOT_A_DISCOUNT_CURVE
)
