# riskquant/portfolio/payloads/forward.py

from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.constants import (
    NO_SETTLEMENT_DATE,
    FX_NONE,
)
from riskquant.portfolio.payloads.base import Payload


@dataclass(frozen=True, slots=True)
class ForwardPayload(Payload):
    """
    Immutable payload for a single forward contract position used in portfolio valuation
    and historical simulation risk analytics within the riskquant framework.

    This payload is intentionally MODEL-NEUTRAL and TIME-DECOUPLED:
      - price risk exposure is defined by a forward curve point (`price_rf_id`)
      - cash settlement timing is defined independently by `settlement_date_ordinal32`
      - pricing, discounting, and FX logic are applied externally by pricers

    Attributes
    ----------
    price_rf_id : np.int32
        Identifier of the underlying forward price risk factor F(t, T).

        This identifies the forward curve point or surface coordinate that
        defines the economic exposure of the contract and determines which
        historical series is shocked in simulation.

        IMPORTANT:
          - `price_rf_id` defines *exposure timing* via the curve tenor.
          - It does NOT imply a contractual delivery or settlement date.

    quantity : np.float64
        Signed contract quantity or notional.
        Positive values represent long exposure, negative values represent short exposure.

    strike : np.float64
        Contract strike price K, expressed in trade currency units.

    settlement_date_ordinal32 : np.int32
        Contractual payment (settlement) date, expressed as days since
        the riskquant DATE_EPOCH (1970-01-01).

        This field is used **exclusively for discounting** the forward payoff.

        Sentinel behavior:
          - If `settlement_date_ordinal32 == NO_SETTLEMENT_DATE (-1)`, no discounting is applied
            and the discount factor is assumed to be 1.0.

        Design principles:
          - The economic exposure is defined solely by `price_rf_id`.
          - The payment date does NOT affect scenario shocks or risk factor selection.
          - The payment date does NOT need to coincide with the tenor implied by
            `price_rf_id`.

        This design intentionally decouples:
          - exposure timing (forward curve point), and
          - cash settlement timing (payment date),

        enabling support for:
          - forwards on long-dated production with arbitrary settlement dates,
          - power and commodity contracts with deferred payment schedules,
          - instruments where discounting is disabled by construction.

        Units:
          - int32 days since 1970-01-01 (DATE_EPOCH)

    fx_under_to_settle_rf_id : np.int32
        Identifier of the FX spot risk factor mapping underlying_ccy -> settlement_ccy,
        or FX_NONE (-1) if no conversion is required.

    fx_strike_to_settle_rf_id : np.int32
        Identifier of the FX spot risk factor mapping strike_ccy -> settlement_ccy,
        or FX_NONE (-1) if no conversion is required.

    Design Notes
    ------------
    - This payload carries only minimal economic definition and risk-factor identifiers.
    - Pricing, discounting, FX conversion, and scenario application are performed
      externally by pricers and valuation engines.
    - `frozen=True` and `slots=True` enforce immutability and compact memory layout,
      minimizing allocation overhead in large portfolios and high-throughput
      simulation workloads.
    - Supports up to two FX linkages without runtime currency strings, preserving
      correctness for:
        * direct forwards (underlying_ccy == settlement_ccy == strike_ccy)
        * quanto forwards (underlying_ccy != settlement_ccy)
        * cross-currency strike forwards (strike_ccy != settlement_ccy)
        * combined quanto + cross-currency structures
    - Suitable for large books, vectorized valuation, and multiprocessing execution.

    Typical valuation identity (resolved by pricers, external to the payload)
    -------------------------------------------------------------------------
        V(t, s) = D(t, settlement_date, s) *
                 quantity *
                 (F_s * FX_s_under->settle - K * FX_s_strike->settle)

        where:
          - D(t, settlement_date, s) is the scenario discount factor in settlement currency
          - F_s is the shocked forward price under scenario s
          - FX_s_* are FX spot levels under scenario s
    """

    price_rf_id: np.int32
    quantity: np.float64
    strike: np.float64
    settlement_date_ordinal32: np.int32 = NO_SETTLEMENT_DATE
    fx_under_to_settle_rf_id: np.int32 = FX_NONE
    fx_strike_to_settle_rf_id: np.int32 = FX_NONE

    def __str__(self) -> str:
        return (
            f"ForwardPayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"settlement_date_ordinal32={self.settlement_date_ordinal32}, "
            f"price_rf_id={self.price_rf_id}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}"
            f")"
        )
