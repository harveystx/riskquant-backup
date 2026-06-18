# src/riskquant/valuation/pricers/principal_cashflow.py
"""
Principal cashflow pricer.

PV = principal_amount * discount_factor * fx_effective

# TODO: Currently this block does not support per-trade FX. We may need to consider this
for cross-currency swaps with notional exchanges. If so, we can add an `fx_effective_uix`

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.pricer_kernels.principal_cashflow import (
    _kernel_price_principal_cashflow,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import ValuationState
    from riskquant.valuation.blocks.discounted_cashflow_block import (
        PrincipalCashflowBlock,
    )


def price_principal_cashflow_block(
    *,
    block: PrincipalCashflowBlock,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """Principal cashflow pricer entrypoint. All arrays must be C-contiguous and dtype-stable."""
    _kernel_price_principal_cashflow(
        out,
        valuation_state.discount_factors,
        valuation_state.fx_spots_effective,
        block.amount,
        block.discount_factor_uix,
        block.fx_spot_effective_uix,
    )
