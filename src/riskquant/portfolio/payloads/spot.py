"""
Payload for spot positions.

This lightweight, immutable container holds the minimal numeric
information required to price and risk a spot position. It stores only
primitive numeric types (NumPy dtypes) so it can be stored in compact
arrays and passed efficiently to vectorized valuation kernels.

Fields
------
quantity
    Signed position size (float64). Positive values represent long
    exposure; negative values represent short exposure.
price_rf_id
    Integer identifier (np.int32) referring to the price/forward risk
    factor that provides market data for this spot position.
fx_rf_id
    Integer identifier (np.int32) for FX conversion underlying_ccy ->
    settlement_ccy. Uses the sentinel `NO_FX_RF_REQUIRED` when no FX
    conversion is needed.
"""

from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.constants import NO_FX_RF_REQUIRED


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotPayload:
    """Immutable payload for a single spot trade.

    The class is frozen to guarantee immutability and uses slots to
    reduce per-instance memory overhead. All fields are NumPy dtypes
    to remain compatible with array-based processing.
    """

    # ========== ECONOMIC TERMS ==========

    # Signed quantity / notional for the spot position. dtype: np.float64
    quantity: np.float64

    # ========== UNDERLYING LINKAGES ==========

    # Price risk factor id used to look up market data for this position. dtype: np.int32
    price_rf_id: np.int32

    # ========== FX LINKAGES ==========

    # FX risk factor id for underlying->settlement conversion, or sentinel when not required.
    fx_rf_id: np.int32 = NO_FX_RF_REQUIRED

    def __str__(self) -> str:
        # Single-line, human-readable representation for logging and debugging
        return (
            f"SpotPayload("
            f"quantity={self.quantity}, "
            f"price_rf_id={self.price_rf_id}, "
            f"fx_rf_id={self.fx_rf_id}"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    # Small usage example showing construction and printing of a SpotPayload.
    # Example values chosen for illustration only.
    example = SpotPayload(
        price_rf_id=np.int32(101),
        quantity=np.float64(1_000.0),
        fx_rf_id=NO_FX_RF_REQUIRED,
    )
    print(example)
