# tests/valuation/test_engine.py
"""
Current `run_valuation` semantics.

Current signature:

    run_valuation(*, portfolio, valuation_state, out) -> None
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.state import ValuationState
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.valuation.blocks.spot_block import SpotBlock
from riskquant.valuation.blocks.vanilla_forward_block import VanillaForwardBlock
from riskquant.valuation.engine import run_valuation

S = 3
NO_DATES = np.array([], dtype=np.int32)
# Hand-built blocks and state share one hand-built universe; they declare it with
# the same identity token so `run_valuation`'s AUD-02 pairing check applies.
_UNIVERSE = "hand-built:tests/valuation/test_engine.py"


def _state(**kw) -> ValuationState:
    return ValuationState(batch_size=S, uix_fingerprint=_UNIVERSE, **kw)


def _spot_block(uids, qty, uix) -> SpotBlock:
    return SpotBlock(
        trade_uids=np.asarray(uids, dtype=np.int64),
        qty=np.asarray(qty, dtype=np.float64),
        effective_spot_price_uix=np.asarray(uix, dtype=np.int32),
    )


def _fwd_block(uids, qty, strike, fuix, duix) -> VanillaForwardBlock:
    return VanillaForwardBlock(
        trade_uids=np.asarray(uids, dtype=np.int64),
        qty=np.asarray(qty, dtype=np.float64),
        strike=np.asarray(strike, dtype=np.float64),
        effective_forward_price_uix=np.asarray(fuix, dtype=np.int32),
        effective_discount_factor_uix=np.asarray(duix, dtype=np.int32),
    )


EFF_SPOT = np.ascontiguousarray([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
EFF_FWD = np.ascontiguousarray([[70.0, 80.0, 60.0]])
EFF_DF = np.ascontiguousarray([[0.95, 0.94, 0.96]])


def test_single_spot_block():
    p = CompiledPortfolio(
        uix_fingerprint=_UNIVERSE,
        unique_dates_ord32=NO_DATES,
        spot=_spot_block([9007, 41], [2.0, -3.0], [1, 0]),
    )
    out = np.empty((p.total_trades, S))
    run_valuation(
        portfolio=p, valuation_state=_state(effective_spot_prices=EFF_SPOT), out=out
    )
    np.testing.assert_allclose(out[0], 2.0 * EFF_SPOT[1])
    np.testing.assert_allclose(out[1], -3.0 * EFF_SPOT[0])


def test_multiple_block_types_land_at_the_right_offsets():
    p = CompiledPortfolio(
        uix_fingerprint=_UNIVERSE,
        unique_dates_ord32=NO_DATES,
        spot=_spot_block([9007], [2.0], [1]),
        vanilla_forward=_fwd_block([41, 5], [10.0, -1.0], [65.0, 55.0], [0, 0], [0, 0]),
    )
    assert p.block_offsets.tolist() == [0, 1]
    assert p.total_trades == 3
    out = np.empty((p.total_trades, S))
    run_valuation(
        portfolio=p,
        valuation_state=_state(
            effective_spot_prices=EFF_SPOT,
            effective_forward_prices=EFF_FWD,
            effective_discount_factors=EFF_DF,
        ),
        out=out,
    )
    np.testing.assert_allclose(out[0], 2.0 * EFF_SPOT[1])
    np.testing.assert_allclose(out[1], 10.0 * (EFF_FWD[0] - 65.0 * EFF_DF[0]))
    np.testing.assert_allclose(out[2], -1.0 * (EFF_FWD[0] - 55.0 * EFF_DF[0]))


def test_trade_uids_are_concatenated_in_block_slot_order():
    p = CompiledPortfolio(
        uix_fingerprint=_UNIVERSE,
        unique_dates_ord32=NO_DATES,
        spot=_spot_block([9007], [1.0], [0]),
        vanilla_forward=_fwd_block([41, 5], [1.0, 1.0], [1.0, 1.0], [0, 0], [0, 0]),
    )
    assert p.trade_uids.tolist() == [9007, 41, 5]
    assert p.trade_uids.tolist() != sorted(p.trade_uids.tolist())


def test_empty_portfolio_prices_nothing():
    p = CompiledPortfolio(uix_fingerprint=_UNIVERSE, unique_dates_ord32=NO_DATES)
    assert p.total_trades == 0 and p.total_blocks == 0
    out = np.empty((0, S))
    run_valuation(portfolio=p, valuation_state=_state(), out=out)
    assert out.shape == (0, S)


# =========================================================================
# VAL-03 output contract (production fix applied in the current source)
# =========================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1, S),
        (3, S),
        (2, S - 1),
        (2, S + 1),
    ],
)
def test_mis_sized_output_is_rejected(shape):
    p = CompiledPortfolio(
        uix_fingerprint=_UNIVERSE,
        unique_dates_ord32=NO_DATES,
        spot=_spot_block([9007, 41], [1.0, 1.0], [0, 0]),
    )
    with pytest.raises(ValueError, match="out has shape"):
        run_valuation(
            portfolio=p,
            valuation_state=_state(effective_spot_prices=EFF_SPOT),
            out=np.empty(shape),
        )


def test_non_contiguous_output_is_rejected():
    p = CompiledPortfolio(
        uix_fingerprint=_UNIVERSE,
        unique_dates_ord32=NO_DATES,
        spot=_spot_block([9007, 41], [1.0, 1.0], [0, 0]),
    )
    wide = np.empty((2, S + 2))
    with pytest.raises(ValueError):
        run_valuation(
            portfolio=p,
            valuation_state=_state(effective_spot_prices=EFF_SPOT),
            out=wide[:, :S],
        )


def test_wrong_output_dtype_is_rejected():
    p = CompiledPortfolio(
        uix_fingerprint=_UNIVERSE,
        unique_dates_ord32=NO_DATES,
        spot=_spot_block([9007], [1.0], [0]),
    )
    with pytest.raises(ValueError, match="float64"):
        run_valuation(
            portfolio=p,
            valuation_state=_state(effective_spot_prices=EFF_SPOT),
            out=np.empty((1, S), dtype=np.float32),  # type: ignore
        )
