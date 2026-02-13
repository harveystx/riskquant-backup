# riskquant/portfolio/compile.py
from __future__ import annotations

from typing import Iterator, Dict

import numpy as np

from riskquant.config.configs import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.blocks.forward_block import ForwardBlock
from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.constants import NO_FX_RF_REQUIRED
from riskquant.portfolio.payloads.forward import ForwardPayload
from riskquant.portfolio.payloads.spot import SpotPayload
from riskquant.portfolio.trades import TradeRecord


def _grow_1d(a: np.ndarray, new_cap: int) -> np.ndarray:
    b = np.empty(new_cap, dtype=a.dtype)
    b[: a.shape[0]] = a
    return b


def compile_portfolio_streaming(
    *,
    trades: Iterator[TradeRecord],
    cfg: RiskStaticConfig,
    initial_capacity: int = 4096,
) -> CompiledPortfolio:
    """
    Streaming compiler focused on CPU/RAM efficiency.

    - Single pass over trades iterator
    - No materialization of TradeRecord list
    - Builds compact numeric arrays only
    - Builds factor universes in first-seen order
    - Today: supports SPOT only (raises on other instruments)
    """
    # ============================== #
    # --- RESOLVE INSTRUMENT IDS --- #
    # ============================== #

    # --- SPOT --- #

    # Get inst ID.
    spot_inst_id = cfg.get_instrument_id("SPOT")

    # Cannot be None.
    if spot_inst_id is None:
        raise PrettyKeyError("Missing instrument id for 'SPOT' in config")

    # Force int.
    spot_inst_id = int(spot_inst_id)

    # --- FORWARD --- #

    # Get inst ID.
    fwd_inst_id = cfg.get_instrument_id("FORWARD")

    # Cannot be None.
    if fwd_inst_id is None:
        raise PrettyKeyError("Missing instrument id for 'FORWARD' in config")

    # Force int.
    fwd_inst_id = int(fwd_inst_id)

    # ==================================== #
    # --- INITIAL ALLOCATION OF MEMORY --- #
    # ==================================== #

    # Set cap.
    cap0 = int(initial_capacity)
    if cap0 <= 0:
        cap0 = 1024

    # Per-instrument capacities (prevents skewed books from growing unused arrays)
    cap_spot = cap0
    cap_fwd = cap0

    # Dynamic buffers (spot)
    spot_trade_uids = np.empty(cap_spot, dtype=np.int64)
    spot_qty = np.empty(cap_spot, dtype=np.float64)
    spot_price_uix = np.empty(cap_spot, dtype=np.int32)
    spot_fx_uix = np.empty(cap_spot, dtype=np.int32)

    # Dynamic buffers (forward)
    fwd_trade_uids = np.empty(cap_fwd, dtype=np.int64)
    fwd_qty = np.empty(cap_fwd, dtype=np.float64)
    fwd_strike = np.empty(cap_fwd, dtype=np.float64)
    fwd_price_uix = np.empty(cap_fwd, dtype=np.int32)
    fwd_fx_under_to_settle_uix = np.empty(cap_fwd, dtype=np.int32)
    fwd_fx_trade_to_settle_uix = np.empty(cap_fwd, dtype=np.int32)

    # Factor universes built incrementally
    # rf_id -> universe index
    price_map: Dict[int, int] = {}
    fx_map: Dict[int, int] = {}
    price_ids: list[np.int32] = []
    fx_ids: list[np.int32] = []

    # ====================== #
    # --- LOCAL BINDINGS --- #
    # ====================== #

    # local bindings for speed in loop
    price_map_get = price_map.get
    fx_map_get = fx_map.get
    price_map_set = price_map.__setitem__
    fx_map_set = fx_map.__setitem__
    price_ids_append = price_ids.append
    fx_ids_append = fx_ids.append
    fx_none = int(NO_FX_RF_REQUIRED)

    # ============================================================ #
    # --- STREAMING-BASED COMPILATION OF TRADES INTO PORTFOLIO --- #
    # ============================================================ #

    # Init m (trade position id in iterator)
    m = 0

    # Init per-instrument counters
    m_spot = 0
    m_fwd = 0

    # Stream trades from iterator
    for t in trades:
        # Get inst id of mth trade.
        inst_id = int(t.instrument_id)

        # Check that the instrument ID is valid (i.e., is recognised and processing exists for it).
        if inst_id != spot_inst_id and inst_id != fwd_inst_id:
            raise NotImplementedError(
                f"Encountered instrument_id={inst_id} for trade_uid={t.trade_uid}!\n"
                "compile_portfolio_streaming currently supports the following instruments: \n"
                "- SPOT, \n"
                "- FORWARD. \n"
            )

        # --- SPOT --- #

        # If instrument = 'SPOT'
        if inst_id == spot_inst_id:
            # grow if needed (geometric growth; amortized O(1))
            if m_spot == cap_spot:
                # New cap.
                new_cap = cap_spot * 2

                # Grow arrays (spot)
                spot_trade_uids = _grow_1d(spot_trade_uids, new_cap)
                spot_qty = _grow_1d(spot_qty, new_cap)
                spot_price_uix = _grow_1d(spot_price_uix, new_cap)
                spot_fx_uix = _grow_1d(spot_fx_uix, new_cap)

                # Update cap of spot
                cap_spot = new_cap

            # Get payload.
            p = t.payload

            # Must be a SpotPayload
            if not isinstance(p, SpotPayload):
                raise TypeError(
                    f"SPOT trade has non-SpotPayload: trade_uid={t.trade_uid}, payload={type(p).__name__}"
                )

            # Unpack.
            spot_trade_uids[m_spot] = int(t.trade_uid)
            spot_qty[m_spot] = float(p.quantity)

            # PRICE RISK FACTOR ID -> PRICE UIX

            # Extract the price RF ID from the payload
            price_rf_id_ = int(p.price_rf_id)

            # Look up whether we've seen this risk factor ID before in price_map dict (price_rf_id -> price_uix).
            price_uix = price_map_get(price_rf_id_)

            # If the risk factor ID doesn't map to anything (we haven't seen it), assign it the next universe index:
            # (1) Count the number of prices seen
            # (2) Map the unseen ID to this count.
            # (3) Append the ID to the list storing seen price RF IDs.
            if price_uix is None:
                # (1) Count the number of prices seen (next free index).
                # If len(price_ids) = 2 then 2 is next free index.
                price_uix = len(price_ids)

                # (2) Map the unseen ID to this count.
                price_map_set(price_rf_id_, price_uix)

                # (3) Append the ID to the list storing seen price RF IDs.
                price_ids_append(np.int32(price_rf_id_))

            # Update the price_uix for the mth SPOT trade.
            spot_price_uix[m_spot] = np.int32(price_uix)

            # --> FX RISK FACTOR ID -> FX UIX

            # Extract FX RF ID from payload
            fx_rf_id_ = int(p.fx_rf_id)

            # If fx_rf_id_ = fx_none, no quanto.
            if fx_rf_id_ == fx_none:
                spot_fx_uix[m_spot] = np.int32(-1)
            else:
                # Look up whether we've seen this FX ID before in fx_map dict (fx_rf_id -> fx_uix).
                fx_uix = fx_map_get(fx_rf_id_)

                # If we have not seen it.
                if fx_uix is None:
                    # (1) Count the number of FX IDs seen (next free index).
                    fx_uix = len(fx_ids)

                    # (2) Map the unseen FX ID to this count (next free uix).
                    fx_map_set(fx_rf_id_, fx_uix)

                    # (3) Append the FX ID to the list storing seen FX RF IDs.
                    fx_ids_append(np.int32(fx_rf_id_))

                # Update the fx_uix for the mth SPOT trade.
                spot_fx_uix[m_spot] = np.int32(fx_uix)

            # INCREMENT COUNTERS

            # Increment the trade counter.
            m_spot += 1
            m += 1
            continue

        # --- FORWARD --- #

        # If instrument = 'FORWARD'
        if inst_id == fwd_inst_id:
            # grow if needed (geometric growth; amortized O(1))
            if m_fwd == cap_fwd:
                # New cap.
                new_cap = cap_fwd * 2

                # Grow arrays (forward)
                fwd_trade_uids = _grow_1d(fwd_trade_uids, new_cap)
                fwd_qty = _grow_1d(fwd_qty, new_cap)
                fwd_strike = _grow_1d(fwd_strike, new_cap)
                fwd_price_uix = _grow_1d(fwd_price_uix, new_cap)
                fwd_fx_under_to_settle_uix = _grow_1d(
                    fwd_fx_under_to_settle_uix, new_cap
                )
                fwd_fx_trade_to_settle_uix = _grow_1d(
                    fwd_fx_trade_to_settle_uix, new_cap
                )

                # Update cap for forwards
                cap_fwd = new_cap

            # Get payload.
            p = t.payload

            # Must be a ForwardPayload.
            if not isinstance(p, ForwardPayload):
                raise TypeError(
                    f"FORWARD trade has non-ForwardPayload: trade_uid={t.trade_uid}, payload={type(p).__name__}"
                )

            # Unpack
            fwd_trade_uids[m_fwd] = int(t.trade_uid)
            fwd_qty[m_fwd] = float(p.quantity)
            fwd_strike[m_fwd] = float(p.strike)

            prf = int(p.price_rf_id)
            pi = price_map_get(prf)
            if pi is None:
                pi = len(price_ids)
                price_map_set(prf, pi)
                price_ids_append(np.int32(prf))
            fwd_price_uix[m_fwd] = np.int32(pi)

            frf = int(p.fx_under_to_settle_rf_id)
            if frf == fx_none:
                fwd_fx_under_to_settle_uix[m_fwd] = np.int32(-1)
            else:
                fi = fx_map_get(frf)
                if fi is None:
                    fi = len(fx_ids)
                    fx_map_set(frf, fi)
                    fx_ids_append(np.int32(frf))
                fwd_fx_under_to_settle_uix[m_fwd] = np.int32(fi)

            trf = int(p.fx_strike_to_settle_rf_id)
            if trf == fx_none:
                fwd_fx_trade_to_settle_uix[m_fwd] = np.int32(-1)
            else:
                ti = fx_map_get(trf)
                if ti is None:
                    ti = len(fx_ids)
                    fx_map_set(trf, ti)
                    fx_ids_append(np.int32(trf))
                fwd_fx_trade_to_settle_uix[m_fwd] = np.int32(ti)

            # INCREMENT COUNTERS

            # Increment forward and trade counter.
            m_fwd += 1
            m += 1
            continue

    # Smoke test - no trades compiled error.
    if m == 0:
        raise ValueError("No trades compiled!")

    # Price factor IDs and FX factor IDs to an array.
    price_factor_ids = np.asarray(price_ids, dtype=np.int32)
    fx_factor_ids = np.asarray(fx_ids, dtype=np.int32)

    # --- SPOT BLOCK --- #

    # Init the spot block.
    spot_block = None
    if m_spot:  # m_spot = 0 evaluates to False
        spot_block = SpotBlock(
            trade_uids=spot_trade_uids[:m_spot],
            qty=spot_qty[:m_spot],
            price_uix=spot_price_uix[:m_spot],
            fx_uix=spot_fx_uix[:m_spot],
        )

    # --- FORWARD BLOCK --- #

    # Init the forward block.
    forward_block = None
    if m_fwd:  # m_fwd = 0 evaluates to False
        forward_block = ForwardBlock(
            trade_uids=fwd_trade_uids[:m_fwd],
            qty=fwd_qty[:m_fwd],
            strike=fwd_strike[:m_fwd],
            price_uix=fwd_price_uix[:m_fwd],
            fx_under_to_settle_uix=fwd_fx_under_to_settle_uix[:m_fwd],
            fx_trade_to_settle_uix=fwd_fx_trade_to_settle_uix[:m_fwd],
        )

    # --- COMPILED PORTFOLIO --- #

    # Return
    return CompiledPortfolio(
        M=m,
        price_factor_ids=price_factor_ids,
        fx_factor_ids=fx_factor_ids,
        spot=spot_block,
        forward=forward_block,
    )
