# src\riskquant\portfolio\ingest\schema.py


from typing import Final, Tuple


_REQUIRED_BASE_COLS: Final[Tuple[str, ...]] = (
    "uid",
    "instrument",
    "quantity",
    "underlying",
    "underlying_ccy",
    "settlement_ccy",
)

# Optional “extended schema” cols (may be missing for SPOT-only sheets)
_OPTIONAL_COLS: Final[Tuple[str, ...]] = (
    "strike_price",
    "strike_ccy",
    "is_call",
    "expiry_date",
    "settlement_date",
)
