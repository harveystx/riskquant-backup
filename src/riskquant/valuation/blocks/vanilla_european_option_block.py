# src/riskquant/valuation/blocks/vanilla_european_option_block.py
"""
Compiled blocks for vanilla European option trades.

Economic contract
-----------------
Black-76 on the strike-currency forward, undiscounted future value at expiry,
then discounted and converted in one multiply:

    FV(T_exp) = cp * [ F_k * Phi(cp * d1) - K * Phi(cp * d2) ]     (strike ccy)

    d1 = (ln(F_k / K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    PV(0)     = Qty * FV(T_exp) * EffectiveDiscountFactor(0, T_settle)

`cp_mult` packs `is_call` into +1 / -1 for a branchless payoff: cp = +1 gives
F*Phi(d1) - K*Phi(d2); cp = -1 gives K*Phi(-d2) - F*Phi(-d1).

Architectural contract
----------------------
* Row `i` of every array corresponds to the trade with UID `trade_uids[i]`.
* Every `_uix` field is a zero-based, contiguous execution index produced by
  the portfolio engine's `UixMapper` -- NOT a global static-config ID.
* FX is NOT carried by this block. It is folded into
  `effective_discount_factors` by the market-data compiler, and into the
  strike-currency forward by the forward translation. No invert flags survive
  into the block layer.
* There is NO sentinel branch. Every `_uix` field must be a valid dense row
  index. The kernel runs under `boundscheck=False`: a negative value wraps to
  the last row of the target matrix and an out-of-range value reads arbitrary
  memory. Earlier revisions of this docstring claimed an `INVALID_UIX (< 0)`
  branch and `fx_*_to_settle_effective_uix` fields; neither ever existed in
  the kernel or in this class (D045/VAL-07).

Factor pairing invariant
------------------------
`forward_price_strike_ccy_uix` and `option_log_forward_price_strike_ccy_uix`
index TWO SEPARATE factor universes that must agree row by row:

    option_log_forward_prices_strike_ccy[j, s]
        == ln( forward_prices_strike_ccy[i, s] )

for the paired (i, j) of every row and every scenario `s`. The kernel uses the
log for d1/d2 and the level for the payoff coefficient, so a desynchronised
pair produces a finite, plausible, wrong price. Establishing this invariant is
the option compiler's responsibility; the kernel must not rediscover or
re-validate it per scenario (D045/VAL-11).

Time-to-expiry redundancy
-------------------------
`sqrt_tte` and `half_tte` encode ONE T twice, precomputed so the scenario loop
does no roots or halvings:

    sqrt_tte == sqrt(T)        half_tte == 0.5 * T
    => half_tte == 0.5 * sqrt_tte ** 2

The compiler must derive both from a single T. Sourcing them independently --
from an expiry date and a settlement date, say -- desynchronises the variance
drift term from the diffusion term silently (D045/VAL-16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import BlockType


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VanillaEuropeanOptionBlock(Block):
    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_EUROPEAN_OPTION)

    # --- TRADE SPECIFICS ---

    # Signed quantity. Long option > 0, short < 0.
    qty: NDArray[np.float64]

    # Strike in the STRIKE currency, at the expiry date. Undiscounted.
    strike: NDArray[np.float64]

    # ln(strike). Precomputed so the scenario loop does no logs.
    # Invariant: log_strike == log(strike).
    log_strike: NDArray[np.float64]

    # Call/put packed for a branchless payoff: call = +1, put = -1.
    cp_mult: NDArray[np.int8]

    # sqrt(T) where T is year-fraction to EXPIRY. See the module docstring.
    sqrt_tte: NDArray[np.float64]

    # 0.5 * T, same T as `sqrt_tte`. See the module docstring.
    half_tte: NDArray[np.float64]

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.forward_prices_strike_ccy`.
    #
    #   Economic quantity : forward price of the underlying, re-expressed in
    #                       strike-currency units
    #   Currency          : STRIKE
    #   Date              : T_expiry (a forward price, NOT a present value)
    #   Discounted        : NO  (re-accumulated to expiry on the strike curve)
    #   FX-converted      : YES (forward FX, via the covered-interest-parity
    #                       construction documented in market_data.state)
    #   Scenario-dependent: YES
    #
    # The volatility indexed by `vol_uix` must be the volatility OF THIS
    # QUANTITY. For a cross-currency option the strike-ccy forward is a product
    # of the asset forward and an FX forward, so its volatility is not the
    # asset forward's volatility.
    forward_price_strike_ccy_uix: NDArray[np.int32]

    # Index into `ValuationState.option_log_forward_prices_strike_ccy`.
    # Must satisfy the pairing invariant in the module docstring.
    option_log_forward_price_strike_ccy_uix: NDArray[np.int32]

    # Index into `ValuationState.vols`.
    #
    #   Economic quantity : annualised lognormal volatility of the
    #                       STRIKE-CURRENCY forward above
    #   Currency          : dimensionless
    #   Date              : T_expiry
    #   Scenario-dependent: YES
    vol_uix: NDArray[np.int32]

    # Index into `ValuationState.effective_discount_factors`.
    #
    #   Economic quantity : combined discount-and-convert multiplier
    #   Value             : DF_STRIKE_CCY(0, T_settle)
    #                       * FX_SPOT_EFFECTIVE(StrikeCCY -> SettleCCY)
    #   Currency          : dimensionless (STRIKE ccy -> SETTLEMENT ccy)
    #   Date              : 0 -> T_settle, the option's SETTLEMENT date, which
    #                       may differ from T_expiry for deferred settlement
    #   Scenario-dependent: YES
    #
    # Pairing the T_expiry forward measure with a T_settle discount factor is
    # exact under deterministic rates and is the standard market convention.
    effective_discount_factor_uix: NDArray[np.int32]
