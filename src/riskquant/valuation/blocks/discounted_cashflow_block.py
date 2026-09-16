# src/riskquant/valuation/blocks/discounted_cashflow_block.py
"""
Compiled blocks for discounted cashflows.

Economic contract
-----------------
Every block here is a single deterministic or forward-implied cashflow at a
date T, discounted and converted to settlement currency in one multiply:

    PV(0) = CashFlow(T) * EffectiveDiscountFactor(0, T)

FX conversion is applied PER CASHFLOW, not after aggregation, because each
cashflow may sit on its own curve, currency and date.

Sign conventions
----------------
The three blocks deliberately differ, and the difference is contractual:

    FixedCashflowBlock     signed by `pay_receive_mult` (pay = -1, receive = +1)
    FloatingCashflowBlock  signed by `pay_receive_mult` (pay = -1, receive = +1)
    PrincipalCashflowBlock signed by `amount` itself -- there is NO multiplier

A principal exchange carries its own sign so that the two legs of a
cross-currency notional exchange can be expressed as one positive and one
negative amount. Applying a pay/receive multiplier as well would double-sign
it. The compiler owns this (D045/VAL-16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import (
    UNSET_BLOCK_TYPE_ID_SENTINEL,
    BlockType,
)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _DiscountedCashflowBlock(Block):
    """Common SoA fields shared by all discounted cashflow blocks."""

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = UNSET_BLOCK_TYPE_ID_SENTINEL

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.effective_discount_factors`.
    #
    #   Economic quantity : combined discount-and-convert multiplier
    #   Value             : DF_CCY(0, T) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #   Currency          : dimensionless (cashflow ccy -> SETTLEMENT ccy)
    #   Date              : 0 -> T, the cashflow's own payment date
    #   Discounted        : it IS the discount factor
    #   FX-converted      : YES (spot FX at t = 0)
    #   Scenario-dependent: YES
    #
    # Discounts the cashflow to t = 0 and converts to settlement currency
    # in one step. Row key is (curve_currency, payment_date).
    effective_discount_factor_uix: npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PrincipalCashflowBlock(_DiscountedCashflowBlock):
    """
    Compiled deterministic principal cashflows.

    CashFlow(T) = Amount

    PV(0) = CashFlow(T) * EffectiveDiscountFactor(0, T)

    Examples
    --------
    - Bond redemption
    - Deposit repayment
    - Cross-currency swap notional exchange
    - Bullet loan principal repayment
    - Final notional return on an FRN
    """

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.PRINCIPAL_CASHFLOW)

    # --- TRADE SPECIFICS ---

    # SIGNED cashflow amount in the cashflow currency, payable at T.
    # Receive > 0, pay < 0. There is no pay/receive multiplier on this block --
    # see the module docstring.
    amount: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FixedCashflowBlock(_DiscountedCashflowBlock):
    """
    Compiled fixed-rate cashflows.

    CashFlow(T) = Notional * AccrualFactor * FixedRate * PayReceiveMult

    PV(0) = CashFlow(T) * EffectiveDiscountFactor(0, T)
    """

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.FIXED_CASHFLOW)

    # --- TRADE SPECIFICS ---

    # Unsigned notional in the cashflow currency.
    notional: npt.NDArray[np.float64]

    # Year fraction of the accrual period, on the leg's own day-count basis.
    accrual_factor: npt.NDArray[np.float64]

    # Annualised fixed rate, decimal (0.0425, not 4.25).
    fixed_rate: npt.NDArray[np.float64]

    pay_receive_mult: npt.NDArray[np.int8]  # pay=-1, receive=+1


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FloatingCashflowBlock(_DiscountedCashflowBlock):
    """
    Compiled floating-rate cashflows.

    FloatingRate(0, s, T) = ForwardRate(0, s, T) + Spread

    CashFlow(T) = Notional * FloatingRate(0, s, T) * AccrualFactor * PayReceiveMult

    PV(0) = CashFlow(T) * EffectiveDiscountFactor(0, T)
    """

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.FLOATING_CASHFLOW)

    # --- TRADE SPECIFICS ---

    # Unsigned notional in the cashflow currency.
    notional: npt.NDArray[np.float64]

    # Year fraction of the accrual period, on the leg's own day-count basis.
    accrual_factor: npt.NDArray[np.float64]

    # Additive spread over the forward rate, annualised decimal. Same units and
    # basis as the forward rate it is added to.
    spread: npt.NDArray[np.float64]

    pay_receive_mult: npt.NDArray[np.int8]  # pay=-1, receive=+1

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.forward_rates` for the forward rate
    # underlying this cashflow.
    #
    #   Economic quantity : annualised SIMPLE forward rate for the accrual
    #                       period [s, T]
    #   Currency          : dimensionless
    #   Date              : fixing/reset s, accruing to payment T
    #   Discounted        : NO
    #   FX-converted      : NO
    #   Scenario-dependent: YES
    #
    # Simple, not compounded: the kernel forms notional * rate * accrual_factor.
    forward_rate_uix: npt.NDArray[np.int32]
