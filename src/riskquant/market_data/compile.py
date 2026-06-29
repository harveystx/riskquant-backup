# src/riskquant/market_data/compile.py
"""
The market-data fold: `MarketDataMatrix` -> `ValuationState`.

This is the compile step that turns raw, universe-aligned market grids into the
pre-multiplied EFFECTIVE factors the pricing kernels gather. Every effective
factor is keyed by a dedup table on `UixMapper`; this module reads the table
and folds the matching raw rows into a dense `(K, S)` matrix.

Pricing kernels stay dumb: all FX translation, discounting, and source-kind
resolution happens HERE, once, at compile time -- never in the scenario loop.

Universe status
---------------
Wired:    effective_spot_prices  (spot_effective_pairs)
Pending:  effective_discount_factors, effective_forward_prices,
          forward_prices_strike_ccy, option_log_forward_prices_strike_ccy,
          vols, forward_rates -- land as their dedup universes and block
          compilers come online. Until then their fields stay empty, which is
          correct: no compiled block references them yet.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.state import MarketDataMatrix, ValuationState
from riskquant.universe import UixMapper


def _fold_effective_spot_prices(
    *,
    matrix: MarketDataMatrix,
    spot_effective_pairs: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """
    Fold `effective_spot_prices[k] = prices[price_uix] * fx_spots_effective[
    fx_eff_uix]`, with identity-FX rows (`fx_eff_uix == INVALID_UIX`)
    multiplying by 1.0.

    `spot_effective_pairs[:, 0]` indexes `matrix.prices` (market_id_universe);
    `[:, 1]` indexes `matrix.fx_spots_effective`. The FX multiply is a vectorized
    scatter over the non-identity rows; identity rows are left as the raw spot
    level.
    """
    k_rows = spot_effective_pairs.shape[0]
    s = matrix.num_scenarios

    out = np.empty((k_rows, s), dtype=np.float64)
    if k_rows == 0:
        return out

    price_uix = spot_effective_pairs[:, 0]
    fx_eff_uix = spot_effective_pairs[:, 1]

    # Gather raw spot levels (all rows).
    out[:] = matrix.prices[price_uix]

    # Scale the rows that carry a real FX leg. `out[rows, :] *= x` is
    # getitem -> in-place multiply -> setitem, so it scatters back into `out`
    # (a plain boolean-mask getitem would write to a throwaway copy).
    fx_rows = np.flatnonzero(fx_eff_uix >= 0)
    if fx_rows.size:
        out[fx_rows, :] *= matrix.fx_spots_effective[fx_eff_uix[fx_rows]]

    return out


def assemble_valuation_state(
    *,
    matrix: MarketDataMatrix,
    uix_mapper: UixMapper,
) -> ValuationState:
    """
    Fold a `MarketDataMatrix` into the kernel-ready `ValuationState`.

    `batch_size` is taken from `matrix.num_scenarios`; the matrix has already
    validated that every grid's scenario axis agrees with it.
    """
    effective_spot_prices = _fold_effective_spot_prices(
        matrix=matrix,
        spot_effective_pairs=uix_mapper.spot_effective_pairs,
    )

    return ValuationState(
        batch_size=matrix.num_scenarios,
        effective_spot_prices=effective_spot_prices,
    )
