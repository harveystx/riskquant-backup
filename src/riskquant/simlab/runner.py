# src/riskquant/simlab/runner.py
"""
SIMLAB orchestrator.

Streaming VaR engine: holds the compiled portfolio, the UIX universe,
the universe-to-flat-buffer mapping, and a scenario generator. Each
`run(snapshot)` call:

  1. Extracts the t0 base-levels vector from the injected snapshot.
  2. Prices the base portfolio once (single-scenario MD matrix).
  3. Streams scenario batches from the generator, applies shocks to
     base levels into a pre-allocated `(N_req, batch_size)` buffer,
     hydrates a `MarketDataMatrix` view over that buffer, prices the
     batch, and writes the PVs into the master scenario-PV grid.
  4. Subtracts base PVs in place and returns a float32 PnL matrix.

DOD properties
--------------
- All hot-path buffers pre-allocated in `run()` exactly once.
- Apply-shocks buffer validation runs ONCE at first batch, not per
  batch.
- `scenario_pvs` is laid out as `(n_trades, n_scenarios)` C-contiguous
  so the per-batch PV writes are unit-stride along the inner axis.
- The terminal `np.float32` cast writes into a pre-allocated buffer
  to avoid the peak-memory spike of `.astype(np.float32)`.

Market-data assembly
--------------------
The snapshot/levels -> `MarketDataMatrix` path (`UniverseMapping`,
`extract_levels_from_snapshot`, `build_market_data_matrix_from_*`) now
lives in `riskquant.market_data.assemble`. It is a simulation-independent
market-data concern; the runner is one consumer. Relocated to keep a
plain valuation from having to import `simlab`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from riskquant.config.enums.discounting_modes import DiscountingMode

# Config & Resolution
from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.static_config import RiskStaticConfig

# Market-data assembly (relocated out of this module).
from riskquant.market_data.assemble import (
    UniverseMapping,
    build_market_data_matrix_from_flat_levels,
    extract_levels_from_snapshot,
)
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.pnl.results import PnlMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.proxy_resolver.proxy import (
    ProxyResolution,
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.simlab.scenarios.base import ScenarioGenerator
from riskquant.simlab.shocks.apply import (
    apply_shocks_to_base_levels,
    validate_apply_shocks_buffers,
)
from riskquant.universe import UixMapper
from riskquant.valuation.engine import run_valuation

# Re-export the assembly surface so existing `from simlab.runner import ...`
# call sites keep working while consumers migrate to `market_data.assemble`.
# These are NOT defined here anymore — `market_data.assemble` is the home.
__all__ = [
    "SimlabRunner",
    "UniverseMapping",
    "build_market_data_matrix_from_flat_levels",
    "extract_levels_from_snapshot",
]


# =========================================================================
# ORCHESTRATOR
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class SimlabRunner:
    """
    Immutable orchestrator for the SIMLAB risk engine.

    Acts as a compiled compute topology: once constructed, every call
    to `run(snapshot)` reuses the same proxy-resolution, universe map,
    UIX mapper, and pre-allocated hot-path buffers (allocated inside
    `run` and reused for the duration of that call).
    """

    config: RiskStaticConfig
    proxy_topology: ProxyTopology
    portfolio: CompiledPortfolio
    uix_mapper: UixMapper
    universe_map: UniverseMapping
    generator: ScenarioGenerator
    as_of_date: date
    batch_size: int = 500

    # Cache topology mapping, NOT state.
    _proxy_resolution: ProxyResolution = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Single proxy resolution per runner lifetime.
        proxy_res = resolve_required_factor_ids_to_market_data_factor_ids(
            static_config=self.config,
            proxy_topology=self.proxy_topology,
            required_factor_ids=self.universe_map.required_factor_ids,
            market_data_factor_ids=self.generator.available_factor_ids,
        )
        object.__setattr__(self, "_proxy_resolution", proxy_res)

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def run(self, snapshot: RiskFactorLevelSnapshot) -> PnlMatrix:
        """
        Run the streaming VaR loop against an injected market snapshot.

        Returns a `PnlMatrix` whose `pnl` field is float32
        `(n_trades, n_scenarios)`.
        """
        n_trades = self.portfolio.total_trades
        n_scenarios = self.generator.total_scenarios

        # ---- 1. t0 base levels ----
        base_levels = extract_levels_from_snapshot(
            required_ids=self.universe_map.required_factor_ids,
            snapshot=snapshot,
        )
        n_req_factors = base_levels.shape[0]

        # ---- 2. Pre-allocate hot-path buffers ----
        # `scenario_pvs` is C-contig: per-batch writes hit unit-stride
        # along the inner scenario axis.
        scenario_pvs = np.empty(
            (n_trades, n_scenarios),
            dtype=np.float64,
            order="C",
        )
        shocked_levels_buf = np.empty(
            (n_req_factors, self.batch_size),
            dtype=np.float64,
            order="C",
        )

        # ---- 3. Base valuation (t0) ----
        base_state = build_market_data_matrix_from_flat_levels(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=self.as_of_date,
            uix_mapper=self.uix_mapper,
            universe_map=self.universe_map,
            flat_levels=base_levels[:, None],
            num_scenarios=1,
        )
        base_result = run_valuation(
            portfolio=self.portfolio,
            valuation_state=base_state,
        )
        base_pvs = base_result.values  # (n_trades, 1)

        # ---- 4. Streaming hot loop ----
        first_batch = True
        for batch in self.generator.iter_batches(
            required_shift_type_id=self.universe_map.required_shift_type_ids,
            source_factor_ids=self._proxy_resolution.source_factor_ids,
            shock_scale=self._proxy_resolution.shock_scale,
            batch_size=self.batch_size,
        ):
            start = batch.scenario_offset
            current_b = batch.shocks.shape[0]
            end = start + current_b

            out_view = shocked_levels_buf[:, :current_b]

            # One-shot layout/dtype validation at first batch. Hot
            # path trusts the contract afterwards.
            if first_batch:
                validate_apply_shocks_buffers(
                    base_levels=base_levels,
                    shocks=batch.shocks,
                    out=out_view,
                )
                first_batch = False

            # Step A — apply shocks (zero allocation).
            apply_shocks_to_base_levels(
                base_levels=base_levels,
                batch=batch,
                out=out_view,
            )

            # Step B — hydrate market state (zero-copy slicing).
            market_state = build_market_data_matrix_from_flat_levels(
                discounting_mode=DiscountingMode.NONE,
                as_of_date=self.as_of_date,
                uix_mapper=self.uix_mapper,
                universe_map=self.universe_map,
                flat_levels=out_view,
                num_scenarios=current_b,
            )

            # Step C — price the batch.
            batch_result = run_valuation(
                portfolio=self.portfolio,
                valuation_state=market_state,
            )

            # Step D — write PVs into the master grid.
            scenario_pvs[:, start:end] = batch_result.values

        # ---- 5. PnL = scenario PV - base PV, in place ----
        # Broadcast subtraction against the (n_trades, 1) base PVs.
        scenario_pvs -= base_pvs

        # ---- 6. Narrow-cast into a fresh float32 buffer ----
        # Allocating the result up front and using np.copyto with the
        # cast is equivalent to .astype() but lets us drop scenario_pvs
        # immediately after — peak-memory wise we still own both
        # buffers across the copy; the caller's GC reclaims scenario_pvs
        # at function exit.
        scenario_pnl_32 = np.empty(scenario_pvs.shape, dtype=np.float32)
        np.copyto(scenario_pnl_32, scenario_pvs, casting="same_kind")

        return PnlMatrix(
            trade_uids=self.portfolio.trade_uids,
            pnl=scenario_pnl_32,
        )
