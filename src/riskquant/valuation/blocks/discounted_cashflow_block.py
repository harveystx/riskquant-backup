# src/riskquant/valuation/blocks/discounted_cashflow_block.py
"""
Compiled blocks for discounted cashflows.
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _DiscountedCashflowBlock(Block):
    """Common SoA fields shared by all discounted cashflow blocks."""

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = UNSET_BLOCK_TYPE_ID_SENTINEL

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.effective_discount_factors`.
    # Discounts cashflow to t = 0 and converts to settlement currency
    # at t = 0 in one step.
    effective_discount_factor_uix: npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True, kw_only=True)
class PrincipalCashflowBlock(_DiscountedCashflowBlock):
    """
    Compiled deterministic principal cashflows.

    CashFlow(T) = Amount

    PV(0) = CashFlow(T) * DiscountFactor(0, T) * FxSpot(0)

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
    amount: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedCashflowBlock(_DiscountedCashflowBlock):
    """
    Compiled fixed-rate cashflows.

    CashFlow(T) = Notional * AccrualFactor * FixedRate * PayReceiveMult

    PV(0) = CashFlow(T) * DiscountFactor(0, T) * FxSpot(0)

    PV(0) = CashFlow(T) * EffectiveDiscountFactor(0, T)

    """

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.FIXED_CASHFLOW)

    # --- TRADE SPECIFICS ---

    notional: npt.NDArray[np.float64]
    accrual_factor: npt.NDArray[np.float64]
    fixed_rate: npt.NDArray[np.float64]
    pay_receive_mult: npt.NDArray[np.int8]  # pay=-1, receive=+1


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatingCashflowBlock(_DiscountedCashflowBlock):
    """
    Compiled floating-rate cashflows.

    FloatingRate(0, s, T) = ForwardRate(0, s, T) + Spread

    CashFlow(T) = Notional * FloatingRate(0, s, T) * AccrualFactor * PayReceiveMult

    PV(0) = CashFlow(T) * DiscountFactor(0, T) * FxSpot(0)
    """

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.FLOATING_CASHFLOW)

    # --- TRADE SPECIFICS ---

    notional: npt.NDArray[np.float64]
    accrual_factor: npt.NDArray[np.float64]
    spread: npt.NDArray[np.float64]
    pay_receive_mult: npt.NDArray[np.int8]  # pay=-1, receive=+1

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.forward_rates` for the forward rate
    # underlying this cashflow.
    forward_rate_uix: npt.NDArray[np.int32]
