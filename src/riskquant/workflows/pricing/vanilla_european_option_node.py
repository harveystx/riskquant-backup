# src/riskquant/workflows/pricing/vanilla_european_option_node.py

"""Workflow for pricing vanilla European option trades, node variant."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionNodeBlock,
)
from riskquant.valuation.pricers._deprecated.vanilla_european_option_node_v1 import (
    price_vanilla_european_option_node_block,
)

_F64_1D = npt.NDArray[np.float64]
_I8_1D = npt.NDArray[np.int8]
_I32_1D = npt.NDArray[np.int32]


# ============================================================================
# INPUT HELPERS
# ============================================================================


def _as_float_array(
    x: float | list[float] | tuple[float, ...] | _F64_1D,
) -> _F64_1D:
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    return np.array(
        arr,
        dtype=np.float64,
        copy=True,
        order="C",
    )


def _as_int8_array(
    x: int | list[int] | tuple[int, ...] | _I8_1D,
) -> _I8_1D:
    arr = np.asarray(x, dtype=np.int8)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    return np.array(
        arr,
        dtype=np.int8,
        copy=True,
        order="C",
    )


def _broadcast_f64(
    arr: _F64_1D,
    n: int,
) -> _F64_1D:
    return np.array(
        np.broadcast_to(arr, (n,)),
        dtype=np.float64,
        copy=True,
        order="C",
    )


def _broadcast_i8(
    arr: _I8_1D,
    n: int,
) -> _I8_1D:
    return np.array(
        np.broadcast_to(arr, (n,)),
        dtype=np.int8,
        copy=True,
        order="C",
    )


def _validate_input_lengths(
    *,
    forward_price: _F64_1D,
    strike: _F64_1D,
    vol: _F64_1D,
    tte: _F64_1D,
    qty: _F64_1D,
    discount_factor: _F64_1D,
    cp_mult: _I8_1D,
) -> int:
    """
    Validate broadcastable workflow inputs.

    All arrays must either:
        - have length 1
        - have length n

    Returns
    -------
    int
        Portfolio size n.
    """

    lengths = {
        len(forward_price),
        len(strike),
        len(vol),
        len(tte),
        len(qty),
        len(discount_factor),
        len(cp_mult),
    }

    non_scalar_lengths = {length for length in lengths if length > 1}

    if len(non_scalar_lengths) > 1:
        raise ValueError(
            "Input arrays have inconsistent lengths: "
            f"forward_price={len(forward_price)}, "
            f"strike={len(strike)}, "
            f"vol={len(vol)}, "
            f"tte={len(tte)}, "
            f"qty={len(qty)}, "
            f"discount_factor={len(discount_factor)}, "
            f"cp_mult={len(cp_mult)}."
        )

    if non_scalar_lengths:
        return next(iter(non_scalar_lengths))

    return 1


# ============================================================================
# SYNTHETIC BLOCK BUILDER
# ============================================================================


def _build_block(
    *,
    qty: _F64_1D,
    strike: _F64_1D,
    cp_mult: _I8_1D,
) -> VanillaEuropeanOptionNodeBlock:

    n = len(qty)

    expiry_date_uix = np.arange(
        n,
        dtype=np.int32,
    )

    settlement_date_uix = np.arange(
        n,
        2 * n,
        dtype=np.int32,
    )

    invalid_uix = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    return VanillaEuropeanOptionNodeBlock(
        trade_uids=np.arange(
            n,
            dtype=np.int64,
        ),
        qty=qty,
        strike=strike,
        cp_mult=cp_mult,
        expiry_date_uix=expiry_date_uix,
        settlement_date_uix=settlement_date_uix,
        fx_under_to_settle_effective_uix=invalid_uix,
        fx_strike_to_settle_effective_uix=invalid_uix,
        under_discount_curve_uix=invalid_uix,
        strike_discount_curve_uix=np.zeros(
            n,
            dtype=np.int32,
        ),
        underlying_price_uix=np.arange(
            n,
            dtype=np.int32,
        ),
        vol_uix=np.arange(
            n,
            dtype=np.int32,
        ),
    )


# ============================================================================
# SYNTHETIC MARKET DATA BUILDER
# ============================================================================


def _build_market_data(
    *,
    forward_price: _F64_1D,
    vol: _F64_1D,
    tte: _F64_1D,
    discount_factor: _F64_1D,
) -> MarketDataMatrix:

    n = len(forward_price)

    forward_nodes = np.array(
        forward_price.reshape(n, 1),
        dtype=np.float64,
        copy=True,
        order="C",
    )

    vol_nodes = np.array(
        vol.reshape(n, 1),
        dtype=np.float64,
        copy=True,
        order="C",
    )

    #
    # Single synthetic curve.
    #
    # Curve 0.
    #
    # Date rows:
    #   0..n-1     expiry
    #   n..2n-1    settlement
    #

    discount_factor_grid = np.ones(
        (1, 2 * n, 1),
        dtype=np.float64,
    )

    discount_factor_grid[
        0,
        n : 2 * n,
        0,
    ] = discount_factor

    tte_grid = np.empty(
        2 * n,
        dtype=np.float64,
    )

    tte_grid[:n] = tte
    tte_grid[n:] = tte

    return MarketDataMatrix(
        num_scenarios=1,
        prices=np.empty(
            (0, 0),
            dtype=np.float64,
        ),
        forward_nodes=forward_nodes,
        vol_nodes=vol_nodes,
        fx_spots_effective=np.empty(
            (0, 1),
            dtype=np.float64,
        ),
        discount_factor_grid=discount_factor_grid,
        tte_grid=tte_grid,
    )


# ============================================================================
# PUBLIC WORKFLOW
# ============================================================================


def price_vanilla_european_option_node(
    *,
    forward_price: float | list[float] | _F64_1D,
    strike: float | list[float] | _F64_1D,
    vol: float | list[float] | _F64_1D,
    tte: float | list[float] | _F64_1D,
    cp_mult: int | list[int] | _I8_1D,
    qty: float | list[float] | _F64_1D = 1.0,
    discount_factor: float | list[float] | _F64_1D = 1.0,
) -> _F64_1D:
    """
    Price vanilla European options using the production
    node-option pricing stack.
    """

    forward_price = _as_float_array(forward_price)
    strike = _as_float_array(strike)
    vol = _as_float_array(vol)
    tte = _as_float_array(tte)
    qty = _as_float_array(qty)
    discount_factor = _as_float_array(discount_factor)
    cp_mult = _as_int8_array(cp_mult)

    n = _validate_input_lengths(
        forward_price=forward_price,
        strike=strike,
        vol=vol,
        tte=tte,
        qty=qty,
        discount_factor=discount_factor,
        cp_mult=cp_mult,
    )

    forward_price = _broadcast_f64(forward_price, n)
    strike = _broadcast_f64(strike, n)
    vol = _broadcast_f64(vol, n)
    tte = _broadcast_f64(tte, n)
    qty = _broadcast_f64(qty, n)
    discount_factor = _broadcast_f64(discount_factor, n)
    cp_mult = _broadcast_i8(cp_mult, n)

    block = _build_block(
        qty=qty,
        strike=strike,
        cp_mult=cp_mult,
    )

    market_data_mtx = _build_market_data(
        forward_price=forward_price,
        vol=vol,
        tte=tte,
        discount_factor=discount_factor,
    )

    out = np.empty(
        (n, 1),
        dtype=np.float64,
    )

    price_vanilla_european_option_node_block(
        block=block,
        market_data_mtx=market_data_mtx,
        out=out,
    )

    return out[:, 0]


if __name__ == "__main__":
    pv = price_vanilla_european_option_node(
        forward_price=[100.0, 100.0, 100.0, 100.0],
        strike=[100.0, 110.0, 100.0, 110.0],
        vol=[0.20, 0.20, 0.20, 0.20],
        tte=[1.0, 1.0, 1.0, 1.0],
        cp_mult=[1, -1, 1, -1],
        qty=[1.0, 1.0, 1.0, 1.0],
        discount_factor=[1.0, 1.0, 0.5, 0.5],
    )

    print(pv)
