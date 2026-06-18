# src/riskquant/credit/masters/enums/credit_factor_kind.py
"""
Credit factor taxonomy.

System-defined constants — not user configuration.

Users reference these string values in the credit factor universe
authoring table. The loader converts the strings into the corresponding
enum values during ingest.
"""

from __future__ import annotations

__all__ = [
    "CreditFactorKind",
]

from enum import IntEnum


class CreditFactorKind(IntEnum):
    SYSTEMIC = 0
    IDIOSYNCRATIC = 1
    CHI = 2
    UNIFORM = 3
    STRESS_INDICATOR = 4
    BETA_PD = 5


VALID_CREDIT_FACTOR_KIND_NAMES: frozenset[str] = frozenset(
    e.name for e in CreditFactorKind
)
VALID_CREDIT_FACTOR_KIND_IDS: frozenset[int] = frozenset(
    e.value for e in CreditFactorKind
)
