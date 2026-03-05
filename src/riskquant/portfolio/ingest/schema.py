# src\riskquant\portfolio\ingest\schema.py


from typing import Final, Tuple

# Required “base schema” cols (must be present in all sheets, including SPOT-only sheets)
REQUIRED_BASE_COLS: Final[Tuple[str, ...]] = (
    "uid",
    "instrument",
    "quantity",
    "underlying",
    "underlying_ccy",
    "settlement_ccy",
)

# Optional “extended schema” cols (may be missing for SPOT-only sheets)
# Cols are required for any non-SPOT instruments (e.g. options, futures, swaps) and may be missing for SPOT-only sheets.
OPTIONAL_COLS: Final[Tuple[str, ...]] = (
    "strike_price",
    "strike_ccy",
    "is_call",
    "expiry_date",
    "delivery_date",
    "settlement_date",
)
