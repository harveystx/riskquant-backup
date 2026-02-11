# riskquant/portfolio/payloads/vanilla_european_option.py
from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.constants import (
    FX_NONE,
    NO_EXPIRY_DATE,
)
from riskquant.portfolio.payloads.base import Payload


@dataclass(frozen=True, slots=True)
class VanillaEuropeanOptionPayload(Payload):
    """
    Payload container for a plain-vanilla (single-underlying) European option position.

    This payload is intentionally MODEL-NEUTRAL:
      - It does NOT encode pricing model (Black, local vol, etc.)
      - It does NOT encode exercise style (European/American/Bermudan)
      - Those are pricer concerns.

    What it DOES encode is the minimal economic definition + market linkages:
      - underlying forward/price risk factor id used by the pricer
      - strike (in trade currency units)
      - time-to-expiry anchor via an expiry risk factor id (or expiry key) if you choose that pattern
      - implied volatility linkage via a single vol risk factor id
      - optional FX links if underlying/trade currencies differ from settlement currency

    Attributes
    ----------
    underlying_price_rf_id : int32
        Risk factor id for the underlying price/forward used by the option pricer.
        (E.g. forward node F(t,T) if you model options on forwards.)

    quantity : float64
        Signed position size (long > 0, short < 0). Kept float64 for safety in big books.

    strike : float64
        Strike K in strike currency units (strike_ccy).

    is_call : int8
        1 for call, 0 for put. int8 to keep compact and branch-friendly.

    expiry_rf_id : int32
        Risk factor id that identifies the expiry anchor required by the pricer.
        This can be:
          - a time/expiry key factor,
          - or a forward node / tenor id that implies expiry,
          - depending on your market data design.
        Keep it as int32 for compact numeric dispatch.

    vol_rf_id : int32
        Risk factor id for the implied volatility input used by the pricer.

        Recommended convention:
          - risk_factor_group = "VOL" (single group for all vols)
          - different vol "types" (ATM / RR / BF / etc.) are represented as different curves
            that all share group "VOL", with their own curve_name and tenor grid.
          - shift type for group "VOL" is typically DIFFERENCE (absolute vol moves),
            but the resolver treats this as a pure source-selection problem; shift-type logic
            belongs downstream.

        This design integrates cleanly with:
          - proxy resolver (ID-level / curve-level / tenor fallback),
          - RiskFactorMatrix (historical vol series),
          - RiskFactorSnapshot (current vol level, potentially from a different provider).

    fx_under_to_settle_rf_id : int32
        FX spot risk factor id for underlying_ccy -> settlement_ccy conversion.
        FX_NONE means no conversion required.

    fx_strike_to_settle_rf_id : int32
        FX spot risk factor id for strike_ccy -> settlement_ccy conversion.
        FX_NONE means no conversion required.

    Notes
    -----
    - Discounting is NOT stored here. The pricer will map expiry/curve metadata to a discount factor source.
    - vol_rf_id is intentionally a single ID for now (one series per trade). If/when you add smile/surface
      representations, you can extend by adding additional vol-related rf_ids (e.g. rr25_rf_id, bf25_rf_id),
      still under risk_factor_group="VOL".
    """

    underlying_price_rf_id: np.int32
    quantity: np.float64
    strike: np.float64
    is_call: np.int8

    # --- VOL LINKAGE --- #
    vol_rf_id: np.int32

    # --- EXPIRY DATE --- #
    expiry_rf_id: np.int32 = NO_EXPIRY_DATE

    # --- FX LINKAGE --- #
    fx_under_to_settle_rf_id: np.int32 = FX_NONE
    fx_strike_to_settle_rf_id: np.int32 = FX_NONE

    def __str__(self) -> str:
        return (
            f"VanillaEuropeanOptionPayload("
            f"underlying_price_rf_id={self.underlying_price_rf_id}, "
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"is_call={self.is_call}, "
            f"vol_rf_id={self.vol_rf_id}, "
            f"expiry_rf_id={self.expiry_rf_id}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}"
            f")"
        )
