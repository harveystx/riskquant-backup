# src/riskquant/portfolio/fx.py
from __future__ import annotations

import re
from typing import Iterable

_ccy_re = re.compile(r"^[A-Z]{3}$")


def validate_ccy(ccy: str) -> str:
    if not isinstance(ccy, str):
        raise TypeError(f"Currency must be str; got {type(ccy).__name__}")
    if len(ccy) != 3:
        raise ValueError(
            f"Currency must be exactly 3 letters; got '{ccy}' ({len(ccy)})"
        )
    if _ccy_re.match(ccy) is None:
        raise ValueError(f"Currency must be 3 uppercase letters A–Z; got '{ccy}'")
    return ccy


def validate_ccys_unique(ccys: Iterable[str]) -> None:
    """Validate each distinct CCY once; fail fast on first invalid."""
    for c in set(ccys):
        validate_ccy(c)


def get_fx_pair_for_under_to_settle_conversion(
    *,
    underlying_ccy: str,
    settlement_ccy: str,
) -> str:
    # Validate scalars only (no allocations)
    u = validate_ccy(underlying_ccy)
    s = validate_ccy(settlement_ccy)
    if u == s:
        return ""
    return f"{u}{s}"
