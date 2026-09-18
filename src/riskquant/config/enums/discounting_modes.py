# src/riskquant/config/enums/discounting_modes.py
"""
Run-level discounting mode (D061).

`NONE` is an EXPLICIT economic instruction: the run prices with

    DF(as_of, target_date, scenario) = 1.0

for every valid target date. It is not "no curve", "no data", "undeclared
`CurveInputType`", an invalid UIX, or any other fallback; every one of those
remains a hard failure. Curve risk factors may still be configured, linked and
present in the snapshot under `NONE` - their levels are simply not read.

`CURVE` selects market-derived discounting, which is not implemented: the
market-data producer fails closed with `NotImplementedError(D061)` before any
curve construction runs.

The mode is a market-data producer input. It MUST NOT reach valuation blocks,
`ValuationState`, sensitivities or pricing kernels, which consume resolved
numerical discount factors only.
"""

from __future__ import annotations

from enum import StrEnum


class DiscountingMode(StrEnum):
    NONE = "NONE"
    CURVE = "CURVE"


VALID_DISCOUNTING_MODES: frozenset[str] = frozenset(m.value for m in DiscountingMode)
