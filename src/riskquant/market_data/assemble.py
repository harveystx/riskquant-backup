# src/riskquant/market_data/assemble.py
"""
Market-data assembly: risk-factor levels -> `MarketDataMatrix`.

The producer-side tier directly above `market_data.builder`. Where
`builder.build_market_data_matrix` constructs a matrix from raw, typed,
universe-aligned arrays, this module assembles those arrays from a flat
`(N_req, S)` levels buffer (or a `RiskFactorLevelSnapshot`) using a
`UniverseMapping` slicing plan, then delegates down to the builder.

Layering
--------
This is the single, simulation-independent path from a levels snapshot to
a valuation-ready `MarketDataMatrix`. It lives in `market_data` precisely
so a plain valuation never has to import `simlab`: SIMLAB simulates the
levels; assembling a matrix from already-known levels is a market-data
concern. `simlab.runner.SimlabRunner` is one consumer; any base/t0
valuation is another.

`UniverseMapping` is the typed layout plan that extends the run's
`UixMapper` ID-universe into the dense, kernel-facing market-data axes.
`UixMapper` (built from the portfolio's ingest batches, now living in the
leaf module `riskquant.universe`) is the authority for the FX-effective
pair table, the date axis, and the market-/discount-id universes;
`UniverseMapping` adds the partition of those universes into the
`prices / forward_nodes / vol_nodes / fx_spots` scalar blocks plus the
curve/surface tenor grids, interp/extrap method codes, and surface y-axis
metadata sourced from `RiskStaticConfig`.

Scalar-node universe alignment (item C, option A1)
--------------------------------------------------
The compiled blocks index `prices` / `forward_nodes` / `vol_nodes` with a
`market_id_universe` UIX. So the dense scalar-node levels gathered here are
scattered by the builder into one `(len(market_id_universe), S)` matrix at
the universe positions carried by `scalar_node_uix_in_universe`; the three
pricer-facing fields alias that single matrix. The old behaviour (dense
`idx_*` slices handed straight to the matrix) was only correct when price
ids happened to occupy the front contiguous run of `market_id_universe`.

NOTE (carried forward): `UniverseMapping` currently straddles two
concerns. Every field except `required_factor_ids` / `required_shift_type_
ids` is pure market-data layout; those two are simlab shock-path inputs
(`extract_levels_from_snapshot` uses `required_factor_ids`; the shift-type
ids are consumed only by the runner's generator). A future split is
warranted — tracked in notes/REFACTORS_NEEDED.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_raise
from riskquant.config.namespace import GlobalIdNamespace
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.builder import build_market_data_matrix
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.market_data.state import MarketDataMatrix
from riskquant.universe import UixMapper

# =========================================================================
# UNIVERSE MAPPING (TYPED LAYOUT PLAN)
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

    # Scalar-node universe alignment (item C / A1)
    # --------------------------------------------
    # `idx_scalar_nodes` spans the scalar-node sub-block of `flat_levels`
    # (every NODE-topology rf in the run universe that is NOT an FX spot).
    # `scalar_node_uix_in_universe` is the `market_id_universe` row for each
    # of those dense rows, so the builder scatters them into the universe-
    # aligned `(len(market_id_universe), S)` scalar matrix. Under A1
    # `prices`/`forward_nodes`/`vol_nodes` alias that one matrix, so there is
    # no per-class (price / fwd / vol) storage partition: those distinctions
    # are pricer-side consumption, not a layout concern, and were removed
    # (S1). A node's economic kind is a config property; its level lives at
    # its universe row regardless of which block reads it.
    idx_scalar_nodes: slice
    scalar_node_uix_in_universe: npt.NDArray[np.int32]

    # Curves / surfaces
    idx_discount_curves: slice
    shape_discount_curves: tuple[int, int]  # (C_curves, M_tenors)

    idx_price_curves: slice
    shape_price_curves: tuple[int, int]

    # Price-curve universe alignment (item C / A1, curve layer)
    # ---------------------------------------------------------
    # Forward-curve underlyings resolve to a curve CONTAINER id that lands
    # in `market_id_universe` (`compile_vanilla_forward_curve` sets
    # `price_curve_uix = map_market_ids_to_uix(...)`), and the kernel reads
    # `price_curve_grid[price_curve_uix, delivery_uix, s]`. So the builder
    # must scatter the dense `(C, D, S)` interpolated grid into a
    # `(len(market_id_universe), D, S)` array at these universe rows — the
    # exact A1 treatment, one axis up. `price_curve_uix_in_universe[c]` is
    # the `market_id_universe` row of the c-th price-curve container (same
    # order as the `idx_price_curves` curve blocks). Empty when the run
    # references no forward curves.
    price_curve_uix_in_universe: npt.NDArray[np.int32]

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

    # Container ids for each row of the surface batch — length C_surfaces,
    # int32. Lets builder-side consumers map a `c` row index in the
    # surface arrays back to its `market_id_universe` slot via
    # `np.searchsorted(uix_mapper.market_id_universe, surface_ids)`.
    # Required for any future scatter-into-universe pass (e.g. surface-
    # ATM synthesis of `vol_nodes` slots, market-id-universe-aligned
    # surface metadata propagation). Defaulted to an empty array so
    # universes that declare zero surfaces remain valid.
    surface_ids: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
    )


# =========================================================================
# UNIVERSE MAPPING BUILDER (config SSOT -> run-scoped layout plan)
# =========================================================================


def _classify_run_universe(
    *,
    market_id_universe: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
) -> tuple[
    npt.NDArray[np.int32],  # scalar-RF ids   (node OR curve/surface POINT)
    npt.NDArray[np.int32],  # fx-spot rf ids  (subset of scalar)
    npt.NDArray[np.int32],  # price-curve CONTAINER ids
    npt.NDArray[np.int32],  # vol-surface CONTAINER ids
]:
    """
    Partition the run's `market_id_universe` by GlobalIdNamespace RANGE.

    Contract (pinned):
      - `< 100M`  (NODE range): a SCALAR risk factor an instrument prices off
        as a single value. This is EITHER a standalone node (node_meta) OR a
        curve/surface POINT (a pillar rf an instrument references directly).
        Both are read as one scalar level and both scatter into the A1
        `scalar_levels` matrix at their `market_id_universe` row. The point-
        vs-node topology distinction is irrelevant to the scalar read path,
        so we deliberately do NOT branch on `rf_id_to_parent_topology_type`
        here — range is the role boundary.
      - `[100M, 200M)` (CURVE range): a price-curve CONTAINER an instrument
        interpolates off. Routed to its own grid universe.
      - `[200M, 300M)` (SURFACE range): a vol-surface CONTAINER.

    FX spots are the scalar subset whose id is in `fx_spot_lookup`'s values.

    Membership comes from the validated run universe; the split is pure
    `GlobalIdNamespace` arithmetic — nothing inferred from the portfolio.
    """
    ids = market_id_universe
    is_node_range = (ids >= GlobalIdNamespace.NODES.min_id) & (
        ids <= GlobalIdNamespace.NODES.max_id
    )
    is_curve_range = (ids >= GlobalIdNamespace.CURVES.min_id) & (
        ids <= GlobalIdNamespace.CURVES.max_id
    )
    is_surface_range = (ids >= GlobalIdNamespace.SURFACES.min_id) & (
        ids <= GlobalIdNamespace.SURFACES.max_id
    )

    out_of_range = ~(is_node_range | is_curve_range | is_surface_range)
    if out_of_range.any():
        raise ValueError(
            "market_id_universe contains ids outside every GlobalIdNamespace "
            f"block: {ids[out_of_range][:10].tolist()}."
        )

    fx_spot_ids = frozenset(int(v) for v in static_config.fx_spot_lookup.values())
    is_fx = np.fromiter(
        (int(i) in fx_spot_ids for i in ids),
        dtype=np.bool_,
        count=ids.shape[0],
    )

    scalar_rfs = ids[is_node_range & ~is_fx]
    fx_spots = ids[is_node_range & is_fx]
    price_curve_containers = ids[is_curve_range]
    vol_surface_containers = ids[is_surface_range]
    return scalar_rfs, fx_spots, price_curve_containers, vol_surface_containers


def build_universe_mapping(
    *,
    static_config: RiskStaticConfig,
    uix_mapper: UixMapper,
    vol_surface_ids: Sequence[int] = (),
) -> UniverseMapping:
    """
    Build the run-scoped market-data layout plan from config SSOT.

    The run universe (`uix_mapper.market_id_universe` + the discount uix
    universe) decides WHICH factors are laid out; `RiskStaticConfig` decides
    WHAT each one is. Membership is the validated subset the portfolio
    references; classification is pure config lookup. Nothing is inferred
    from the portfolio.

    Price (forward) curves are NOT passed in — a forward-curve underlying
    resolves to a curve CONTAINER id that lands in `market_id_universe`
    (`compile_vanilla_forward_curve` -> `map_market_ids_to_uix`), so the
    CURVE-topology members of `market_id_universe` ARE the run's price
    curves. Discount curves come from `discount_curve_id_universe`.

    Layout / `required_factor_ids` order (the `flat_levels` row order the
    `idx_*` slices then re-slice):

        [ fx_spots | scalar_nodes | discount_curve_pts | price_curve_pts ]

    Curve points are emitted curve-by-curve in `curve_id_to_tenor_index`
    offset order so the `(C, M)` reshape in the builder is exact.

    Universe alignment (item C / A1): `scalar_node_uix_in_universe` and
    `price_curve_uix_in_universe` carry each scalar node's / price-curve
    container's `market_id_universe` row, so the builder scatters both the
    dense scalar block and the dense `(C, D, S)` curve grid into universe-
    aligned arrays.

    Raises
    ------
    NotImplementedError
        If the run references vol surfaces (any SURFACE-topology member of
        `market_id_universe`, or a non-empty `vol_surface_ids`). The surface
        `(C, M, K)` grid assembly from raw `surface_point_meta` has no
        reference producer to verify against and is intentionally NOT
        fabricated blind — see notes/REFACTORS_NEEDED.md (P2 follow-up).
        Scalar nodes, FX spots, discount curves, and price (forward) curves
        are fully supported.
    """
    mkt_uni = uix_mapper.market_id_universe
    disc_uni = uix_mapper.discount_curve_id_universe

    scalar_rfs, fx_spots, price_curve_ids, surface_ids = _classify_run_universe(
        market_id_universe=mkt_uni, static_config=static_config
    )
    if surface_ids.size or vol_surface_ids:
        raise NotImplementedError(
            "build_universe_mapping: vol-surface layout is not yet implemented "
            "(the (C, M, K) grid assembly from surface_point_meta has no "
            f"reference producer to verify against). Surface containers in run: "
            f"{surface_ids[:5].tolist()}. Scalar nodes, FX spots, discount "
            "curves, and price curves are supported. See REFACTORS_NEEDED.md."
        )

    interp = static_config.container_interp_x_id
    extrap = static_config.container_extrap_x_id
    tenor_index = static_config.curve_id_to_tenor_index

    # ---- Curve point layout: discount then price, curve-by-curve --------
    def _curve_block(
        container_ids: npt.NDArray[np.int32],
    ) -> tuple[list[int], list[float], list[int], list[int], int]:
        child_ids: list[int] = []
        tenors: list[float] = []
        interps: list[int] = []
        extraps: list[int] = []
        m_ref: int | None = None
        for cid in container_ids.tolist():
            ti = tenor_index[int(cid)]
            offs = ti.sorted_offsets
            ids = ti.rf_ids_by_offset
            if m_ref is None:
                m_ref = offs.shape[0]
            elif offs.shape[0] != m_ref:
                raise ValueError(
                    f"curve {cid} has {offs.shape[0]} tenors but the batch "
                    f"expects a rectangular {m_ref}; ragged curve grids are "
                    "not supported in one batch."
                )
            child_ids.extend(int(x) for x in ids.tolist())
            tenors.extend(float(x) for x in offs.tolist())
            cslot = static_config.container_slot(int(cid))
            interps.append(int(interp[cslot]))
            extraps.append(int(extrap[cslot]))
        return child_ids, tenors, interps, extraps, (m_ref or 0)

    disc_ids = np.asarray(disc_uni, dtype=np.int32)
    price_ids = np.asarray(price_curve_ids, dtype=np.int32)  # CURVE members of mkt_uni

    disc_children, disc_tenors, _disc_interp, _disc_extrap, disc_m = _curve_block(
        disc_ids
    )
    price_children, price_tenors, price_interp, price_extrap, price_m = _curve_block(
        price_ids
    )

    # ---- Assemble required_factor_ids + idx_* in one pass ---------------
    n_fx = fx_spots.shape[0]
    n_scalar = scalar_rfs.shape[0]
    n_disc = len(disc_children)
    n_price = len(price_children)

    required = np.concatenate(
        [
            fx_spots,
            scalar_rfs,
            np.asarray(disc_children, dtype=np.int32),
            np.asarray(price_children, dtype=np.int32),
        ]
    ).astype(np.int32, copy=False)

    off = 0
    idx_fx = slice(off, off + n_fx)
    off += n_fx
    idx_scalar = slice(off, off + n_scalar)
    off += n_scalar
    idx_disc = slice(off, off + n_disc)
    off += n_disc
    idx_price = slice(off, off + n_price)
    off += n_price

    # Scalar-RF + price-curve universe rows (A1 scatter targets).
    scalar_uix = np.searchsorted(mkt_uni, scalar_rfs).astype(np.int32, copy=False)
    price_curve_uix = np.searchsorted(mkt_uni, price_ids).astype(np.int32, copy=False)

    required_shift = static_config.rf_id_to_shift_type_id[required].astype(
        np.int8, copy=False
    )

    return UniverseMapping(
        required_factor_ids=required,
        required_shift_type_ids=required_shift,
        idx_fx_spots=idx_fx,
        idx_scalar_nodes=idx_scalar,
        scalar_node_uix_in_universe=scalar_uix,
        idx_discount_curves=idx_disc,
        shape_discount_curves=(disc_ids.shape[0], disc_m),
        idx_price_curves=idx_price,
        shape_price_curves=(price_ids.shape[0], price_m),
        price_curve_uix_in_universe=price_curve_uix,
        idx_vol_surfaces=slice(off, off),
        shape_vol_surfaces=(0, 0, 0),
        # Discount curves use a hardwired log-linear kernel and ignore method
        # codes; `curve_interp/extrap_methods` feed ONLY the price-curve grid
        # builder, so they carry price-curve entries exclusively (length
        # C_price). `disc_interp`/`disc_extrap` are computed but unused.
        curve_interp_methods=np.asarray(price_interp, dtype=np.int8),
        curve_extrap_methods=np.asarray(price_extrap, dtype=np.int8),
        raw_discount_tenors=np.asarray(disc_tenors, dtype=np.float64).reshape(
            disc_ids.shape[0], disc_m
        ),
        raw_price_curve_tenors=np.asarray(price_tenors, dtype=np.float64).reshape(
            price_ids.shape[0], price_m
        ),
        surface_interp_x_methods=np.empty(0, dtype=np.int8),
        surface_extrap_x_methods=np.empty(0, dtype=np.int8),
        surface_y_axis_type_ids=np.empty(0, dtype=np.int8),
        raw_vol_surface_tenors=np.empty(0, dtype=np.float64),
        raw_vol_surface_strikes=np.empty(0, dtype=np.float64),
        surface_ids=np.empty(0, dtype=np.int32),
    )


# =========================================================================
# LEVELS GATHER
# =========================================================================


def extract_levels_from_snapshot(
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
    idx = searchsorted_or_raise(
        universe=snapshot.factor_ids,
        raw=required_ids,
        ctx="risk factors required by the compiled portfolio but missing "
        "from the snapshot",
    )
    return np.ascontiguousarray(snapshot.factor_levels[idx]).astype(
        np.float64, copy=False
    )


# =========================================================================
# MATRIX ASSEMBLY
# =========================================================================


def build_market_data_matrix_from_flat_levels(
    *,
    as_of_date: date,
    uix_mapper: UixMapper,
    universe_map: UniverseMapping,
    flat_levels: npt.NDArray[np.float64],
    num_scenarios: int,
) -> MarketDataMatrix:
    """
    Build a `MarketDataMatrix` from a flat `(N_req, S)` levels matrix.

    Slicing produces views, not copies. The builder owns its own
    per-build allocations (FX-effective gather, curve grids).
    """
    u = universe_map

    raw_fx_spots = flat_levels[u.idx_fx_spots, :]
    raw_scalar_nodes = flat_levels[u.idx_scalar_nodes, :]

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

    raw_vol_surface_values = flat_levels[u.idx_vol_surfaces, :].reshape(
        u.shape_vol_surfaces[0],
        u.shape_vol_surfaces[1],
        u.shape_vol_surfaces[2],
        num_scenarios,
    )

    return build_market_data_matrix(
        as_of_date=as_of_date,
        uix_mapper=uix_mapper,
        num_scenarios=num_scenarios,
        raw_fx_spots=raw_fx_spots,
        raw_scalar_nodes=raw_scalar_nodes,
        scalar_node_uix_in_universe=u.scalar_node_uix_in_universe,
        raw_discount_tenors=u.raw_discount_tenors,
        raw_discount_ln_dfs=raw_discount_ln_dfs,
        raw_price_curve_tenors=u.raw_price_curve_tenors,
        raw_price_curve_values=raw_price_curve_values,
        curve_interp_methods=u.curve_interp_methods,
        curve_extrap_methods=u.curve_extrap_methods,
        price_curve_uix_in_universe=u.price_curve_uix_in_universe,
        raw_vol_surface_tenors=u.raw_vol_surface_tenors,
        raw_vol_surface_strikes=u.raw_vol_surface_strikes,
        raw_vol_surface_values=raw_vol_surface_values,
        surface_interp_x_methods=u.surface_interp_x_methods,
        surface_extrap_x_methods=u.surface_extrap_x_methods,
        surface_y_axis_type_ids=u.surface_y_axis_type_ids,
    )


def build_market_data_matrix_from_snapshot(
    *,
    as_of_date: date,
    uix_mapper: UixMapper,
    universe_map: UniverseMapping,
    snapshot: RiskFactorLevelSnapshot,
    num_scenarios: int = 1,
) -> MarketDataMatrix:
    """
    Build a `MarketDataMatrix` directly from a `RiskFactorLevelSnapshot`.
    """
    base_levels = extract_levels_from_snapshot(
        required_ids=universe_map.required_factor_ids,
        snapshot=snapshot,
    )
    if num_scenarios == 1:
        flat_levels = base_levels[:, None]
    else:
        flat_levels = np.broadcast_to(
            base_levels[:, None], (base_levels.shape[0], num_scenarios)
        )
    return build_market_data_matrix_from_flat_levels(
        as_of_date=as_of_date,
        uix_mapper=uix_mapper,
        universe_map=universe_map,
        flat_levels=flat_levels,
        num_scenarios=num_scenarios,
    )
