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
Wired:    effective_spot_prices       (spot_effective_pairs)
          effective_discount_factors  (edf_effective_triples)
          effective_forward_prices    (efp_effective_quads, NODE-sourced)
Pending:  effective_forward_prices    (CURVE-sourced),
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


def _fold_effective_discount_factors(
    *,
    matrix: MarketDataMatrix,
    edf_effective_triples: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """
    Materialise `(K_edf, S)` = `DF_ccy(0, T) * FX_SPOT_EFFECTIVE(ccy -> settle)`.

    `edf_effective_triples[:, 0]` indexes `matrix.discount_factor_grid` axis 0
    (the DISCOUNT-CURVE universe, distinct from market_id_universe); `[:, 1]`
    indexes its date axis; `[:, 2]` indexes `matrix.fx_spots_effective`, or is
    `INVALID_UIX` for a leg already in settlement currency.

    Discounting and FX are pre-multiplied here precisely so the kernel performs
    one gather and one multiply. An amount multiplied by a row of the result is
    BOTH discounted to t = 0 AND converted to settlement currency.
    """
    k_rows = edf_effective_triples.shape[0]
    s = matrix.num_scenarios

    out = np.empty((k_rows, s), dtype=np.float64)
    if k_rows == 0:
        return out

    curve_uix = edf_effective_triples[:, 0]
    date_uix = edf_effective_triples[:, 1]
    fx_eff_uix = edf_effective_triples[:, 2]

    out[:] = matrix.discount_factor_grid[curve_uix, date_uix]

    fx_rows = np.flatnonzero(fx_eff_uix >= 0)
    if fx_rows.size:
        out[fx_rows, :] *= matrix.fx_spots_effective[fx_eff_uix[fx_rows]]

    return out


def _fold_effective_forward_prices(
    *,
    matrix: MarketDataMatrix,
    efp_effective_quads: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """
    Materialise `(K_efp, S)`
        = `F_ccy(0, T) * DF_ccy(0, T) * FX_SPOT_EFFECTIVE(ccy -> settle)`.

    This is the settlement-currency PRESENT VALUE of receiving one unit of the
    asset at T -- NOT a forward price. The forward kernel subtracts a discounted
    strike leg from it, so both operands must already be PVs in one currency.

    Columns are `(price_uix, curve_uix, date_uix, fx_eff_uix)`. `price_uix`
    indexes `matrix.forward_nodes` (NODE-sourced forwards only in this phase;
    curve-sourced forwards would read `matrix.price_curve_grid` and need a
    source-kind column).
    """
    k_rows = efp_effective_quads.shape[0]
    s = matrix.num_scenarios

    out = np.empty((k_rows, s), dtype=np.float64)
    if k_rows == 0:
        return out

    price_uix = efp_effective_quads[:, 0]
    curve_uix = efp_effective_quads[:, 1]
    date_uix = efp_effective_quads[:, 2]
    fx_eff_uix = efp_effective_quads[:, 3]

    out[:] = matrix.forward_nodes[price_uix]
    out *= matrix.discount_factor_grid[curve_uix, date_uix]

    fx_rows = np.flatnonzero(fx_eff_uix >= 0)
    if fx_rows.size:
        out[fx_rows, :] *= matrix.fx_spots_effective[fx_eff_uix[fx_rows]]

    return out


def _fold_vols(
    *,
    matrix: MarketDataMatrix,
    vol_effective_uix: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """
    Materialise `(K_vol, S)` by gathering `matrix.vol_nodes`.

    A `vols` row is a plain NODE volatility level; there is nothing to
    pre-multiply. It is folded here anyway so that every factor a kernel
    gathers lives in one container with one dense coordinate space.

    SEMANTIC CONTRACT: the row must be the volatility OF THE STRIKE-CURRENCY
    FORWARD the paired `forward_prices_strike_ccy` row carries. For an option
    whose asset and strike currencies differ, those are different stochastic
    quantities, and nothing in `IngestBatch` or `MarketDataMatrix` supplies the
    FX volatility or correlation needed to convert one into the other. The
    quoted surface must therefore already be in the option's own measure. This
    layer cannot verify that; it is a requirement on the feed (D045/X-07).
    """
    k_rows = vol_effective_uix.shape[0]
    s = matrix.num_scenarios

    out = np.empty((k_rows, s), dtype=np.float64)
    if k_rows == 0:
        return out
    out[:] = matrix.vol_nodes[vol_effective_uix]
    return out


def _fold_forward_prices_strike_ccy(
    *,
    effective_forward_prices: npt.NDArray[np.float64],
    effective_discount_factors: npt.NDArray[np.float64],
    fk_effective_pairs: npt.NDArray[np.int32],
    num_scenarios: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Materialise the strike-currency forward AND its log cache, row-aligned.

    Substituting the two effective definitions:

        effective_forward_prices[efp]   = F_ccy(T) DF_ccy(0,T) FX(ccy->settle)
        effective_discount_factors[edf] = DF_K(0,T) FX(K->settle)

        ratio = F_ccy(T) DF_ccy(0,T) FX(ccy->settle)
                / ( DF_K(0,T) FX(K->settle) )
              = F_K(T)

    which is exactly the construction `market_data.state` documents: discount
    the asset-currency forward to PV, translate through the settlement
    numeraire, re-accumulate on the strike curve. Both legs are taken at the
    option's EXPIRY.

    Returns `(forward_prices_strike_ccy, option_log_forward_prices_strike_ccy)`.
    The log is an elementwise transform of the SAME table, so

        option_log_forward_prices_strike_ccy[k] == log(forward_prices_strike_ccy[k])

    holds for every row by construction, and one `k` addresses both. That is the
    VAL-11 pairing invariant, established here rather than asserted downstream.
    """
    k_rows = fk_effective_pairs.shape[0]

    fwd = np.empty((k_rows, num_scenarios), dtype=np.float64)
    if k_rows == 0:
        return fwd, np.empty((0, num_scenarios), dtype=np.float64)

    fwd[:] = effective_forward_prices[fk_effective_pairs[:, 0]]
    fwd /= effective_discount_factors[fk_effective_pairs[:, 1]]

    return fwd, np.ascontiguousarray(np.log(fwd))


def assemble_valuation_state(
    *,
    matrix: MarketDataMatrix,
    uix_mapper: UixMapper,
) -> ValuationState:
    """
    Fold a `MarketDataMatrix` into the kernel-ready `ValuationState`.

    `batch_size` is taken from `matrix.num_scenarios`; the matrix has already
    validated that every grid's scenario axis agrees with it.

    D064: the matrix must have been built against THIS mapper. Every effective
    table read below indexes the matrix's axes, and two mappers with equal
    universe sizes produce in-range but economically foreign rows.
    """
    if matrix.uix_fingerprint != uix_mapper.fingerprint:
        raise ValueError(
            "MarketDataMatrix was built against a different UixMapper "
            f"(matrix {matrix.uix_fingerprint[:12]}..., mapper "
            f"{uix_mapper.fingerprint[:12]}...). Rebuild the matrix with this mapper."
        )
    effective_spot_prices = _fold_effective_spot_prices(
        matrix=matrix,
        spot_effective_pairs=uix_mapper.spot_effective_pairs,
    )
    effective_discount_factors = _fold_effective_discount_factors(
        matrix=matrix,
        edf_effective_triples=uix_mapper.edf_effective_triples,
    )
    effective_forward_prices = _fold_effective_forward_prices(
        matrix=matrix,
        efp_effective_quads=uix_mapper.efp_effective_quads,
    )
    forward_prices_strike_ccy, log_forward_prices_strike_ccy = (
        _fold_forward_prices_strike_ccy(
            effective_forward_prices=effective_forward_prices,
            effective_discount_factors=effective_discount_factors,
            fk_effective_pairs=uix_mapper.fk_effective_pairs,
            num_scenarios=matrix.num_scenarios,
        )
    )
    vols = _fold_vols(
        matrix=matrix,
        vol_effective_uix=uix_mapper.vol_effective_uix,
    )

    return ValuationState(
        batch_size=matrix.num_scenarios,
        uix_fingerprint=uix_mapper.fingerprint,
        effective_spot_prices=effective_spot_prices,
        effective_discount_factors=effective_discount_factors,
        effective_forward_prices=effective_forward_prices,
        vols=vols,
        forward_prices_strike_ccy=forward_prices_strike_ccy,
        option_log_forward_prices_strike_ccy=log_forward_prices_strike_ccy,
    )
