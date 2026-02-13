# riskquant/valuation/pnl/pnl_engine_streaming.py
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from riskquant.config.configs import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
    SHIFT_DIFFERENCE,
)
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.simlab.resolve.proxy import ProxyResolution
from riskquant.simlab.scenarios.base import ScenarioGenerator
from riskquant.valuation.pnl.pnl_writer import PnlBatchWriter


def _build_trade_uid_order(portfolio: CompiledPortfolio) -> np.ndarray:
    trade_uids = np.empty(int(portfolio.M), dtype=np.int64)
    r = 0
    if portfolio.spot is not None:
        m = int(portfolio.spot.trade_uids.shape[0])
        trade_uids[r : r + m] = portfolio.spot.trade_uids
        r += m
    if portfolio.forward is not None:
        m = int(portfolio.forward.trade_uids.shape[0])
        trade_uids[r : r + m] = portfolio.forward.trade_uids
        r += m
    return trade_uids


def _snapshot_id_to_level(
    snapshot: RiskFactorLevelSnapshot,
) -> dict[int, float]:
    ids = snapshot.factor_ids
    lv = snapshot.factor_levels
    return {int(ids[i]): float(lv[i]) for i in range(ids.shape[0])}


def _fetch_levels(
    *,
    factor_ids: np.ndarray,  # (N,) int32
    id_to_level: dict[int, float],
    name: str,
    dtype: np.dtype,
) -> np.ndarray:
    out = np.empty(int(factor_ids.shape[0]), dtype=dtype)
    for i in range(out.shape[0]):
        fid = int(factor_ids[i])
        try:
            out[i] = id_to_level[fid]
        except KeyError as e:
            raise PrettyKeyError(
                f"Missing required current level in snapshot for {name}:\n- factor_id={fid}\n"
            ) from e
    return out


def _levels_from_shocks_for_uix(
    *,
    base_levels: np.ndarray,  # (N,) float32
    shift_type_ids: np.ndarray,  # (N,) int8
    required_to_shock_uix: np.ndarray,  # (N,) int32
    shock_scale: np.ndarray,  # (N,) float32
    shocks: np.ndarray,  # (B,K) float32
    out: np.ndarray,  # (B,N) float32 (preallocated)
) -> None:
    """
    Materialize scenario levels (B,N) for one universe (prices or fx).
    This is a pragmatic first version; later you can replace with lazy gathers per block.
    """
    B = int(shocks.shape[0])
    N = int(base_levels.shape[0])

    for j in range(N):
        st = np.int8(shift_type_ids[j])
        k = int(required_to_shock_uix[j])
        sc = float(shock_scale[j])

        sh = shocks[:, k]  # (B,)

        if st == SHIFT_LOG_RETURN:
            out[:, j] = base_levels[j] * np.exp(sh * sc)
        elif st == SHIFT_SIMPLE_RETURN:
            out[:, j] = base_levels[j] * (1.0 + sh * sc)
        elif st == SHIFT_DIFFERENCE:
            out[:, j] = base_levels[j] + (sh * sc)
        else:
            raise ValueError(f"Unknown shift_type_id={int(st)} at j={j}")


def run_pnl_engine_streaming(
    *,
    cfg: RiskStaticConfig,
    portfolio: CompiledPortfolio,
    snapshot: RiskFactorLevelSnapshot,
    proxy: ProxyResolution,
    scenario_gen: ScenarioGenerator,
    scenario_batch_size: int,
    writer: PnlBatchWriter,
    run_id: Optional[str] = None,
    # dtypes
    level_dtype: np.dtype = np.float32,
    pnl_dtype: np.dtype = np.float32,
) -> str:
    """
    Streaming HVaR engine:
      - iterates scenario batches
      - produces trade-level PnL (M,B) for each batch
      - writes batches to DB via writer

    Contract:
      proxy.required_factor_ids must match required_ids ordering:
        required_ids = [price_factor_ids, fx_factor_ids]
    """
    if run_id is None:
        run_id = uuid.uuid4().hex

    price_ids = portfolio.price_factor_ids
    fx_ids = portfolio.fx_factor_ids
    required_ids = np.concatenate((price_ids, fx_ids)).astype(
        np.int32, copy=False
    )

    # Enforce ordering alignment with proxy for direct slicing
    if required_ids.shape != proxy.required_factor_ids.shape or np.any(
        required_ids != proxy.required_factor_ids
    ):
        raise ValueError(
            "ProxyResolution.required_factor_ids must match valuation required_ids ordering "
            "([price_factor_ids, fx_factor_ids])."
        )

    # Required shift types from cfg cache
    rf_shift = cfg.rf_id_to_shift_type_id
    required_shift_type_id = np.empty(required_ids.shape[0], dtype=np.int8)
    for i in range(required_ids.shape[0]):
        rid = np.int32(required_ids[i])
        try:
            required_shift_type_id[i] = np.int8(rf_shift[rid])
        except KeyError as e:
            raise PrettyKeyError(
                f"Missing shift type id for required_factor_id={rid}"
            ) from e

    # Snapshot levels
    id_to_level = _snapshot_id_to_level(snapshot)
    price0 = _fetch_levels(
        factor_ids=price_ids,
        id_to_level=id_to_level,
        name="price_factor_ids",
        dtype=level_dtype,
    )
    fx0 = _fetch_levels(
        factor_ids=fx_ids,
        id_to_level=id_to_level,
        name="fx_factor_ids",
        dtype=level_dtype,
    )

    N_price = int(price_ids.shape[0])
    N_fx = int(fx_ids.shape[0])

    # trade uid ordering matching pnl row order
    trade_uids = _build_trade_uid_order(portfolio)
    M = int(portfolio.M)

    # Iterate scenarios in batches
    for batch in scenario_gen.iter_batches(
        required_factor_ids=required_ids,
        required_shift_type_id=required_shift_type_id,
        source_factor_ids=proxy.source_factor_ids,
        shock_scale=proxy.shock_scale,
        batch_size=int(scenario_batch_size),
    ):
        B = int(batch.shocks.shape[0])

        # Preallocate scenario levels for universes (first version)
        price_s = np.empty((B, N_price), dtype=level_dtype, order="C")
        fx_s = np.empty((B, N_fx), dtype=level_dtype, order="C")

        # Apply shocks to snapshot levels
        _levels_from_shocks_for_uix(
            base_levels=price0,
            shift_type_ids=batch.required_shift_type_id[:N_price],
            required_to_shock_uix=batch.required_to_shock_uix[:N_price],
            shock_scale=batch.shock_scale[:N_price],
            shocks=batch.shocks,
            out=price_s,
        )
        _levels_from_shocks_for_uix(
            base_levels=fx0,
            shift_type_ids=batch.required_shift_type_id[N_price:],
            required_to_shock_uix=batch.required_to_shock_uix[N_price:],
            shock_scale=batch.shock_scale[N_price:],
            shocks=batch.shocks,
            out=fx_s,
        )

        # Allocate trade-level pnl for this batch
        pnl = np.empty((M, B), dtype=pnl_dtype, order="C")

        # Fill pnl block-by-block in deterministic row order
        row0 = 0

        # SPOT block
        if portfolio.spot is not None:
            b = portfolio.spot
            m = int(b.trade_uids.shape[0])
            # vectorized over scenarios, loop over trades (first version)
            for i in range(m):
                q = float(b.qty[i])
                pu = int(b.price_uix[i])
                fu = int(b.fx_uix[i])
                if fu < 0:
                    pv0 = q * float(price0[pu])
                    pvS = q * price_s[:, pu]
                else:
                    pv0 = q * float(price0[pu]) * float(fx0[fu])
                    pvS = q * price_s[:, pu] * fx_s[:, fu]
                pnl[row0 + i, :] = (pvS - pv0).astype(pnl_dtype, copy=False)
            row0 += m

        # FORWARD block
        if portfolio.forward is not None:
            b = portfolio.forward
            m = int(b.trade_uids.shape[0])
            for i in range(m):
                q = float(b.qty[i])
                K = float(b.strike[i])
                pu = int(b.price_uix[i])
                fu = int(b.fx_under_to_settle_uix[i])
                ft = int(b.fx_trade_to_settle_uix[i])

                F0 = float(price0[pu])
                under_fx0 = 1.0 if fu < 0 else float(fx0[fu])
                trade_fx0 = 1.0 if ft < 0 else float(fx0[ft])
                pv0 = q * (F0 * under_fx0 - K * trade_fx0)

                FS = price_s[:, pu]
                under_fxS = 1.0 if fu < 0 else fx_s[:, fu]
                trade_fxS = 1.0 if ft < 0 else fx_s[:, ft]
                pvS = q * (FS * under_fxS - K * trade_fxS)

                pnl[row0 + i, :] = (pvS - pv0).astype(pnl_dtype, copy=False)
            row0 += m

        # Write batch to persistence
        writer.write_batch(
            run_id=run_id,
            trade_uids=trade_uids,
            scenario_offset=int(batch.scenario_offset),
            pnl=pnl,
            scenario_count_total=batch.scenario_count_total,
        )

    return run_id
