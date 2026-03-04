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

from riskquant.portfolio.constants import NO_FX_RF_REQUIRED, NO_ID


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

    # ========== VALIDATION IN __post_init__ ==========

    def __post_init__(self) -> None:
        # Force strict NumPy dtypes to prevent Python object boxing in kernels
        object.__setattr__(
            self,
            "quantity",
            np.float64(self.quantity),
        )
        object.__setattr__(
            self,
            "price_rf_id",
            np.int32(self.price_rf_id),
        )
        object.__setattr__(
            self,
            "fx_rf_id",
            np.int32(self.fx_rf_id),
        )

        # Validation to catch common errors early
        # price_rf_id must be a valid non-negative integer (no sentinel allowed)
        if self.price_rf_id == NO_ID:
            raise ValueError(
                "SpotPayload requires price_rf_id (cannot be sentinel)."
            )
        if self.price_rf_id < 0:
            raise ValueError("price_rf_id must be non-negative.")

        # fx_rf_id must be either a valid non-negative integer or the NO_FX_RF_REQUIRED sentinel
        if self.fx_rf_id == NO_ID:
            raise ValueError(
                "SpotPayload requires fx_rf_id (cannot be NO_ID). Use NO_FX_RF_REQUIRED if no FX conversion is needed."
            )

        # fx_rf_id must be non-negative or the NO_FX_RF_REQUIRED sentinel (negative values other than the sentinel are not allowed)
        if self.fx_rf_id < 0 and self.fx_rf_id != NO_FX_RF_REQUIRED:
            raise ValueError(
                "fx_rf_id must be a non-negative integer or NO_FX_RF_REQUIRED sentinel."
            )

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
