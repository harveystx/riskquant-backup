# src/riskquant/valuation/blocks/vanilla_forward_block.py
"""
Compiled blocks for vanilla forward trades.

Economic contract
-----------------
A vanilla forward is priced as TWO INDEPENDENT PRESENT-VALUE LEGS:

    PV(0) = Qty * [ PV_settle(receive one unit of the asset at T)
                    - PV_settle(pay Strike at T) ]

          = Qty * [ F_a(0,T) * DF_a(0,T) * FX_spot(a -> settle)
                    - K       * DF_k(0,T) * FX_spot(k -> settle) ]

where `a` is the asset (forward-price) currency and `k` the strike currency.

Settlement assumption
---------------------
The two-leg form is the value of a PHYSICALLY SETTLED forward: the holder
receives the asset and pays the strike, each leg discounted on its own curve
and converted at its own FX rate. It is therefore correct for a
cross-currency physical forward, and it degenerates exactly to the familiar

    PV = Qty * (F - K) * DF * FX

whenever both legs share one curve and one FX rate -- which is the case for a
single-currency forward and for a CASH-SETTLED forward of any currency, whose
single net cashflow Qty*(S_T - K) is discounted once.

The compiler MUST NOT pair two different discount curves on a cash-settled
forward. Doing so would silently produce the physical-settlement value, which
differs whenever DF_a != DF_k, and no layer below can detect it (D045/VAL-15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import (
    BlockType,
)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VanillaForwardBlock(Block):
    """Common SoA fields shared by all vanilla forward blocks."""

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_FORWARD)

    # --- TRADE SPECIFICS ---

    # Signed quantity. Long forward > 0, short < 0.
    qty: npt.NDArray[np.float64]

    # Contractual strike, in the STRIKE currency, payable at T. Undiscounted.
    strike: npt.NDArray[np.float64]

    # ---  LINKAGES (UIX) ---

    # Index into `ValuationState.effective_forward_prices`.
    #
    # WARNING: despite the field name, this row is NOT a forward price. It is
    # the fully-synthesised PRESENT VALUE of the underlying leg -- the forward
    # price, its own discount factor and its own FX rate already multiplied
    # together by the market-data compiler:
    #
    #   Economic quantity : PV at t=0 of receiving one unit of the asset at T
    #   Value             : FWD_PRICE_CCY(0,T) * DF_CCY(0,T)
    #                       * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #   Currency          : SETTLEMENT
    #   Date              : t = 0 (for delivery at T)
    #   Discounted        : YES (already applied -- do NOT discount again)
    #   FX-converted      : YES (spot FX at t = 0)
    #   Scenario-dependent: YES
    #
    # Populating this with a raw forward price would subtract a discounted PV
    # from an undiscounted price and produce a dimensionally invalid number
    # with no error (D045/VAL-06).
    effective_forward_price_uix: npt.NDArray[np.int32]

    # Index into `ValuationState.effective_discount_factors`, used to discount
    # and convert the STRIKE leg.
    #
    #   Economic quantity : combined discount-and-convert multiplier
    #   Value             : DF_CCY(0,T) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #   Currency          : dimensionless (STRIKE ccy -> SETTLEMENT ccy)
    #   Date              : 0 -> T, where T is the strike payment date
    #   Discounted        : it IS the discount factor
    #   FX-converted      : YES (spot FX at t = 0)
    #   Scenario-dependent: YES
    #
    # An amount multiplied by this factor is BOTH discounted to t = 0 AND
    # converted to settlement currency, in one multiply. The row key is
    # (curve_currency, date); CCY here is the STRIKE currency.
    effective_discount_factor_uix: npt.NDArray[np.int32]
