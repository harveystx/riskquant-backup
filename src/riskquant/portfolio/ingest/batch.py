# src/riskquant/portfolio/ingest/batch.py
"""
Struct-of-Arrays (SoA) container for a validated batch of trades.
"""

from dataclasses import dataclass
import numpy as np
from pandas import DataFrame, Series, NaT

from riskquant.constants import (
    NO_DISCOUNT_CURVE_ID,
    NO_FX_RF_REQUIRED,
    NO_DELIVERY_DATE_ORDINAL32,
    NO_SETTLEMENT_DATE_ORDINAL32,
    NO_EXPIRY_DATE_ORDINAL32,
    NOT_OPTION,
)
from riskquant.portfolio import fx
from riskquant.utils.dates import ordinal32_to_datetime64D


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestBatch:
    """
    Pristine, contiguous array container for trade ingestion.
    """

    # Number of trades in this batch
    size: int

    # ========== ROUTING / DISPATCH ==========
    instrument_ids: np.ndarray

    # ========== CORE IDENTITY & ECONOMICS ==========
    uids: np.ndarray
    qty: np.ndarray
    strike_price: np.ndarray
    is_call: np.ndarray

    # ========== DATES (ORDINAL32) ==========
    settle_ord: np.ndarray
    delivery_ord: np.ndarray
    expiry_ord: np.ndarray

    # ========== MARKET & FX LINKAGES (UNIVERSE INDICES) ==========
    fx_under_to_settle_ids: np.ndarray
    fx_under_to_settle_invert: np.ndarray
    fx_strike_to_settle_ids: np.ndarray
    fx_strike_to_settle_invert: np.ndarray
    price_rf_ids: np.ndarray
    price_curve_ids: np.ndarray
    vol_rf_ids: np.ndarray
    vol_surface_rf_ids: np.ndarray
    under_discount_curve_ids: np.ndarray
    strike_discount_curve_ids: np.ndarray

    def to_df(self) -> DataFrame:
        """
        Reconstructs a Pandas DataFrame from the SoA arrays for auditing and debugging.
        Translates ordinals back to datetimes and applies NaT/NaN for sentinels.
        """

        def decode_dates(ord_array: np.ndarray, sentinel: int) -> Series:
            mask = ord_array == sentinel
            dts = ordinal32_to_datetime64D(ord_array)
            s = Series(dts)
            s[mask] = NaT
            return s

        def mask_fx(fx_array: np.ndarray) -> Series:
            s = Series(fx_array, dtype=object)
            s[fx_array == NO_FX_RF_REQUIRED] = "NONE"
            return s

        def mask_fx_invert(fx_invert_array: np.ndarray) -> Series:
            s = Series(fx_invert_array, dtype=bool)
            return s

        def mask_curve(curve_array: np.ndarray) -> Series:
            s = Series(curve_array, dtype=object)
            s[curve_array == NO_DISCOUNT_CURVE_ID] = "NONE"
            return s

        return DataFrame(
            {
                # IDS
                "uid": self.uids,
                "instrument_id": self.instrument_ids,
                # ECONOMICS
                "qty": self.qty,
                "strike_price": Series(self.strike_price).replace(0.0, np.nan),
                "is_call": Series(self.is_call).replace(NOT_OPTION, np.nan),
                # DATES
                "expiry_date": decode_dates(
                    self.expiry_ord,
                    NO_EXPIRY_DATE_ORDINAL32,
                ),
                "delivery_date": decode_dates(
                    self.delivery_ord,
                    NO_DELIVERY_DATE_ORDINAL32,
                ),
                "settle_date": decode_dates(
                    self.settle_ord,
                    NO_SETTLEMENT_DATE_ORDINAL32,
                ),
                # LINKAGES
                "price_rf_id": self.price_rf_ids,
                "price_curve_id": self.price_curve_ids,
                "vol_rf_id": self.vol_rf_ids,
                "vol_surface_rf_id": self.vol_surface_rf_ids,
                "fx_u2s": mask_fx(self.fx_under_to_settle_ids),
                "fx_u2s_invert": mask_fx_invert(self.fx_under_to_settle_invert),
                "fx_k2s": mask_fx(self.fx_strike_to_settle_ids),
                "fx_k2s_invert": mask_fx_invert(
                    self.fx_strike_to_settle_invert
                ),
                "under_discount_curve_id": mask_curve(
                    self.under_discount_curve_ids
                ),
                "strike_discount_curve_id": mask_curve(
                    self.strike_discount_curve_ids
                ),
            }
        )

    @property
    def df(self) -> DataFrame:
        """
        Alias for .to_df() to allow easy access to the reconstructed DataFrame.
        """
        return self.to_df()
