# src/riskquant/config/enums/curve_input_types.py
"""
Curve input taxonomy — the ECONOMIC representation of a curve's own levels.

System-defined constants, not user configuration. Users reference these string
values in `curve_meta.curve_input_type`.

`CurveInputType` is orthogonal to `QuoteType` (D079 clause 3):

    quote_type = RATE, unit_scale = PERCENT, curve_input_type = PAR_RATE
    quote_type = RATE, unit_scale = PERCENT, curve_input_type = ZERO_RATE

are different economic contracts carrying identical generic market-data
metadata. `QuoteType` says "these levels are rates"; `CurveInputType` says what
the future curve-construction layer must DO with them. It MUST NOT be inferred
from `QuoteType`, from the curve name, from tenor labels, or from the numerical
magnitude of the levels.

The value is declared once, on the CURVE. Curve points inherit it through their
`curve_id`; there is no point-level override, so there is exactly one source of
truth.

Scope: the enum names the contract only. Curve construction itself remains
fail-closed under D061 (`discounting_mode = NONE`), and nothing in the
market-data or valuation path consumes these values today.
"""

from __future__ import annotations

from enum import StrEnum


class CurveInputType(StrEnum):
    #: Quoted market instrument rates (OIS / deposit / swap par rates). A
    #: construction model (bootstrap) is required before discount factors exist.
    PAR_RATE = "PAR_RATE"
    #: Zero (spot) rates already stripped of instrument structure. Only a
    #: declared compounding convention and a rate -> DF conversion are needed.
    ZERO_RATE = "ZERO_RATE"
    #: Levels that already ARE discount factors for the point's maturity.
    DISCOUNT_FACTOR = "DISCOUNT_FACTOR"
    #: Levels that already ARE natural logarithms of discount factors.
    LOG_DISCOUNT_FACTOR = "LOG_DISCOUNT_FACTOR"


VALID_CURVE_INPUT_TYPES: frozenset[str] = frozenset(e.value for e in CurveInputType)
