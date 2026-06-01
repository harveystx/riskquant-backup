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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt

# Config & Resolution
from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.static_config import RiskStaticConfig

# State & Valuation
from riskquant.market_data.builder import build_market_data_matrix
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.market_data.state import MarketDataMatrix
from riskquant.pnl.results import PnlMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.engine import UixMapper
from riskquant.proxy_resolver.proxy import (
    ProxyResolution,
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.simlab.scenarios.base import ScenarioGenerator
from riskquant.simlab.shocks.apply import (
    apply_shocks_to_base_levels,
    validate_apply_shocks_buffers,
)
from riskquant.valuation.engine import run_valuation

# =========================================================================
# UNIVERSE MAPPING
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class UniverseMapping:
    """
    Slicing plan to extract the structured market-data arrays from one
    flat `(N_req, B)` levels matrix.

    The `idx_*` slices index the leading factor axis; reshape information
    for curves and surfaces lives in the `shape_*` triples.

    All `np.ndarray` fields hold per-config metadata pinned to the run's
    universe. Validation lives at build time; the runner trusts the
    contract.
    """

    required_factor_ids: npt.NDArray[np.int32]
    required_shift_type_ids: npt.NDArray[np.int8]

    # Scalars
    idx_fx_spots: slice
    idx_prices: slice
    idx_fwd_nodes: slice
    idx_vol_nodes: slice

    # Curves / surfaces
    idx_discount_curves: slice
    shape_discount_curves: tuple[int, int]  # (C_curves, M_tenors)

    idx_price_curves: slice
    shape_price_curves: tuple[int, int]

    idx_vol_surfaces: slice
    shape_vol_surfaces: tuple[int, int, int]  # (C_surfaces, M_tenors, K_strikes)

    # 1-D curve configs
    curve_interp_methods: npt.NDArray[np.int8]
    curve_extrap_methods: npt.NDArray[np.int8]
    raw_discount_tenors: npt.NDArray[np.float64]
    raw_price_curve_tenors: npt.NDArray[np.float64]

    # 2-D surface configs
    surface_interp_x_methods: npt.NDArray[np.int8]
    surface_extrap_x_methods: npt.NDArray[np.int8]
    surface_y_axis_type_ids: npt.NDArray[np.int8]
    raw_vol_surface_tenors: npt.NDArray[np.float64]
    raw_vol_surface_strikes: npt.NDArray[np.float64]


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
        base_levels = self._extract_base_levels(
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
        base_state = self._hydrate_state(
            flat_levels=base_levels[:, None],
            num_scenarios=1,
        )
        base_result = run_valuation(
            portfolio=self.portfolio,
            market_data_mtx=base_state,
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
            market_state = self._hydrate_state(
                flat_levels=out_view,
                num_scenarios=current_b,
            )

            # Step C — price the batch.
            batch_result = run_valuation(
                portfolio=self.portfolio,
                market_data_mtx=market_state,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_base_levels(
        self,
        *,
        required_ids: npt.NDArray[np.int32],
        snapshot: RiskFactorLevelSnapshot,
    ) -> npt.NDArray[np.float64]:
        """
        Vectorised O(N log K) gather of base levels by required-id.

        `snapshot.factor_ids` is sorted ascending (the Snapshot class
        invariant), enabling one-shot `searchsorted` membership check
        and gather.
        """
        idx = np.searchsorted(snapshot.factor_ids, required_ids)
        idx = np.clip(idx, 0, snapshot.factor_ids.shape[0] - 1)
        if np.any(snapshot.factor_ids[idx] != required_ids):
            missing = required_ids[snapshot.factor_ids[idx] != required_ids]
            raise ValueError(
                f"Compiled portfolio requires risk factors missing from the "
                f"snapshot: {np.unique(missing).tolist()}"
            )
        return np.ascontiguousarray(snapshot.factor_levels[idx]).astype(
            np.float64, copy=False
        )

    def _hydrate_state(
        self,
        *,
        flat_levels: npt.NDArray[np.float64],
        num_scenarios: int,
    ) -> MarketDataMatrix:
        """
        Slice the flat `(N_req, B)` levels matrix into the dimensional
        grids expected by the valuation engine, and call the builder.

        Slicing produces views, not copies. The builder owns its own
        per-build allocations (FX-effective gather, curve grids).
        """
        u = self.universe_map

        # ---- Scalars ----
        raw_fx_spots = flat_levels[u.idx_fx_spots, :]
        raw_prices = flat_levels[u.idx_prices, :]
        raw_forward_nodes = flat_levels[u.idx_fwd_nodes, :]
        raw_vol_nodes = flat_levels[u.idx_vol_nodes, :]

        # ---- Curves ----
        raw_discount_ln_dfs = flat_levels[u.idx_discount_curves, :].reshape(
            u.shape_discount_curves[0],
            u.shape_discount_curves[1],
            num_scenarios,
        )
        raw_price_curve_values = flat_levels[u.idx_price_curves, :].reshape(
            u.shape_price_curves[0],
            u.shape_price_curves[1],
            num_scenarios,
        )

        # ---- Surfaces ----
        raw_vol_surface_values = flat_levels[u.idx_vol_surfaces, :].reshape(
            u.shape_vol_surfaces[0],
            u.shape_vol_surfaces[1],
            u.shape_vol_surfaces[2],
            num_scenarios,
        )

        return build_market_data_matrix(
            as_of_date=self.as_of_date,
            uix_mapper=self.uix_mapper,
            num_scenarios=num_scenarios,
            raw_fx_spots=raw_fx_spots,
            raw_prices=raw_prices,
            raw_forward_nodes=raw_forward_nodes,
            raw_vol_nodes=raw_vol_nodes,
            raw_discount_tenors=u.raw_discount_tenors,
            raw_discount_ln_dfs=raw_discount_ln_dfs,
            raw_price_curve_tenors=u.raw_price_curve_tenors,
            raw_price_curve_values=raw_price_curve_values,
            curve_interp_methods=u.curve_interp_methods,
            curve_extrap_methods=u.curve_extrap_methods,
            raw_vol_surface_tenors=u.raw_vol_surface_tenors,
            raw_vol_surface_strikes=u.raw_vol_surface_strikes,
            raw_vol_surface_values=raw_vol_surface_values,
            surface_interp_x_methods=u.surface_interp_x_methods,
            surface_extrap_x_methods=u.surface_extrap_x_methods,
            surface_y_axis_type_ids=u.surface_y_axis_type_ids,
        )
