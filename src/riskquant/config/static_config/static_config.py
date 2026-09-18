# src/riskquant/config/static_config/static_config.py
"""
Runtime container for the compiled god-tier static config.

Strictly native Python structures plus typed NumPy arrays — no Pandas.
All derived lookups are computed eagerly in `__post_init__`. The object
is fully valid and fully indexed the moment construction completes.

Conceptual model
----------------
Three disjoint container topology types:

  Node     - scalar observable; IS its own risk factor.
             (FX rates, equity spot, single-name credit spread proxy, etc.)
  Curve    - 1D function; addressable risk-factor children.
             (Commodity forward curves, IR discount curves, etc.)
  Surface  - 2D function; addressable risk-factor children.
             (Vol surfaces in (expiry, strike) or equivalent.)

Name namespaces are kept globally unique: every node_name, curve_name,
surface_name, curve-point relative name, and surface-point relative name
is distinct from every other in the same RiskStaticConfig. The validator
enforces this at compile time so the ingest layer can classify a string
deterministically.

Two lookup tables matter at ingest:

  name_to_rf_id        : node_name | curve_point_rel_name | surface_point_rel_name
                         -> rf_id (int32). Resolves directly to a runtime
                         risk factor.
  container_name_to_id : curve_name | surface_name -> container_id (int32).
                         Resolves to a parent container that requires
                         interpolation at runtime.

A trade's `underlying` resolves into exactly one of these two tables.
Both-match and neither-match are validation errors.

FX resolution
-------------
fx_spot_lookup is the typed table the ingest FX resolver consumes:

  (under_ccy_id: int8, settle_ccy_id: int8) -> rf_id: int32

It is built from NodeMeta rows tagged asset_class="FX" using their
declared (fx_base_ccy, quote_ccy). No string heuristics, no "[ON]"
guessing.

Discount resolution
-------------------
Two layers:

  ccy_id_to_default_discount_curve_id  : per-ccy default (OIS).
  discount_curve_lookup                : (ccy_id, basis_id) -> curve_id
                                          for explicit (basis) overrides.

The compiler enforces every active currency has at least one discount
curve and exactly one default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_raise
from riskquant.config.enums.container_topology import ContainerTopologyType
from riskquant.config.enums.discount_basis import DiscountBasis
from riskquant.config.enums.quote_types import UNIT_SCALE_TO_BASE, UnitScale
from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.static_config.schema_objects import (
    Ccy,
    CurveMeta,
    CurvePointMeta,
    HolidayCalendar,
    NodeMeta,
    SurfaceMeta,
    SurfacePointMeta,
)
from riskquant.constants import INVALID_UNIT_SCALE, NO_FX_RF_REQUIRED, NO_ID
from riskquant.market_data.curves.extrap_methods import EXTRAP_METHOD_REGISTRY
from riskquant.market_data.curves.interp_methods import INTERP_METHOD_REGISTRY
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_TYPE_NAME_TO_ID_REGISTRY,
)
from riskquant.utils.tenors import parse_bracket_token

if TYPE_CHECKING:
    from riskquant.market_data.curves.extrap_methods import ExtrapMethod
    from riskquant.market_data.curves.interp_methods import InterpMethod

# ============================================================
# Sentinels
# ============================================================

_NO_DISCOUNT_CURVE_ID: np.int32 = NO_ID
# D060: the FX matrix fill IS `riskquant.constants.NO_FX_RF_REQUIRED`. It is not
# re-declared here, because a local alias drifted to NO_ID and every empty cell
# then read as a valid direct quote in `fx_resolver` (MD-00).

# ============================================================
# Immutability helpers
# ============================================================


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Return a read-only view of the array."""
    arr.setflags(write=False)
    return arr


def _freeze_dict(d: Mapping) -> MappingProxyType:
    """Return a read-only proxy of the dictionary."""
    return MappingProxyType(d)


# ============================================================
# Type-State Pipeline
# ============================================================
#
# Each pipeline stage is a frozen dataclass that inherits from the
# previous stage. This makes the data-dependency graph explicit:
# a function that takes RfState cannot accidentally read fields
# that are only defined on ContainerState.
#
# The pipeline is:
#   -> CurrencyState
#   -> NameLookupState
#   -> RfState
#   -> ContainerState
#   -> CurveTenorState
#   -> FxState
#   -> FxMatrixState
#   -> DiscountState (= FinalState)
# ============================================================


@dataclass(frozen=True)
class CurrencyState:
    """Output of currency-layer resolution."""

    ccy_name_to_id: Mapping[str, int]
    ccy_id_to_name: tuple[str, ...]
    max_ccy_id: int


@dataclass(frozen=True)
class NameLookupState(CurrencyState):
    """+ risk-factor and container name -> id mappings."""

    rel_name_to_rf_id: Mapping[str, int]
    container_name_to_id: Mapping[str, int]


@dataclass(frozen=True)
class RfState(NameLookupState):
    """+ per-risk-factor dense attribute arrays."""

    max_rf_id: int
    rf_id_to_parent_topology_type: npt.NDArray[np.int8]
    rf_id_to_parent_id: npt.NDArray[np.int32]
    rf_id_to_ccy_id: npt.NDArray[np.int8]
    rf_id_to_shift_type_id: npt.NDArray[np.int8]
    rf_id_to_unit_scale_factor: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ContainerState(RfState):
    """
    + per-container dense attribute arrays, SLOT-indexed.

    Every array below is parallel to ``container_ids`` (sorted-ascending,
    unique). Index by SLOT — obtained from ``searchsorted_or_raise`` on
    ``container_ids`` — never by raw container_id. Container ids live in
    the CURVES / SURFACES namespace ranges (250M+); a dense-by-raw-id
    layout would allocate multi-GB arrays sized ``max_container_id + 1``,
    so the store is O(num_containers) instead.
    """

    container_ids: npt.NDArray[np.int32]
    container_topology_type: npt.NDArray[np.int8]
    container_interp_x_id: npt.NDArray[np.int8]
    container_extrap_x_id: npt.NDArray[np.int8]
    container_interp_y_id: npt.NDArray[np.int8]
    container_extrap_y_id: npt.NDArray[np.int8]
    container_ccy_id: npt.NDArray[np.int8]
    container_x_axis_kind: npt.NDArray[np.int8]


@dataclass(frozen=True)
class CurveTenorState(ContainerState):
    """+ per-curve tenor index."""

    curve_id_to_tenor_index: Mapping[int, CurveTenorIndex]


@dataclass(frozen=True)
class FxState(CurveTenorState):
    """+ FX spot lookup table."""

    fx_spot_lookup: Mapping[tuple[int, int], int]


@dataclass(frozen=True)
class FxMatrixState(FxState):
    """+ dense FX spot matrix."""

    fx_spot_matrix: npt.NDArray[np.int32]


@dataclass(frozen=True)
class DiscountState(FxMatrixState):
    """+ discount-curve lookup tables.  Terminal state of the pipeline."""

    ccy_id_to_default_discount_curve_id: npt.NDArray[np.int32]
    discount_curve_lookup: Mapping[tuple[int, int], int]


# ============================================================
# Derived index dataclasses
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class CurveTenorIndex:
    """
    Per-curve children, layout-ready for fast interpolation lookup.

    OFFSET curves populate (sorted_offsets, rf_ids_sorted_by_offset).
    LABEL curves populate (label_to_rf_id). The empty arrays/dict are
    always present so callers don't branch on absence.
    """

    sorted_offsets: npt.NDArray[np.int64]
    rf_ids_by_offset: npt.NDArray[np.int32]
    label_to_rf_id: Mapping[str, int]


# ============================================================
# Atomic resolution helpers
# ============================================================


def _resolve_shift_id(shift_type_name: str) -> int:
    shift = SHIFT_TYPE_NAME_TO_ID_REGISTRY.get(shift_type_name)
    if shift is None:
        raise KeyError(
            f"shift_type={shift_type_name!r} not in SHIFT_TYPE_NAME_TO_ID_REGISTRY."
        )
    return int(shift)


def _resolve_unit_scale_factor(unit_scale_name: str) -> float:
    scale = UNIT_SCALE_TO_BASE.get(UnitScale(unit_scale_name))
    if scale is None:
        raise KeyError(
            f"unit_scale={unit_scale_name!r} not found in UNIT_SCALE_TO_BASE."
        )
    return scale


def _resolve_interp_id(method_name: str) -> int:
    interp: InterpMethod | None = INTERP_METHOD_REGISTRY.get(method_name)
    if interp is None:
        raise KeyError(f"interp_method={method_name!r} not in INTERP_METHOD_REGISTRY.")
    return int(interp)


def _resolve_extrap_id(method_name: str) -> int:
    extrap: ExtrapMethod | None = EXTRAP_METHOD_REGISTRY.get(method_name)
    if extrap is None:
        raise KeyError(f"extrap_method={method_name!r} not in EXTRAP_METHOD_REGISTRY.")
    return int(extrap)


# ============================================================
# Pipeline builders — each takes state(N) and returns state(N+1)
# ============================================================


def _build_currency_state(ccy_dict: dict[str, Ccy]) -> CurrencyState:
    """Stage 1: Resolve the currency universe."""
    if not ccy_dict:
        raise ValueError("ccy_dict is empty.")

    ccy_name_to_id: dict[str, int] = {
        name: int(c.ccy_id) for name, c in ccy_dict.items()
    }
    max_ccy_id = max((int(c.ccy_id) for c in ccy_dict.values()), default=0)

    ccy_id_to_name: list[str] = [""] * (max_ccy_id + 1)
    for c in ccy_dict.values():
        ccy_id_to_name[int(c.ccy_id)] = c.ccy_name

    return CurrencyState(
        ccy_name_to_id=ccy_name_to_id,
        ccy_id_to_name=tuple(ccy_id_to_name),
        max_ccy_id=max_ccy_id,
    )


def _build_name_lookup_state(
    state: CurrencyState,
    node_meta_dict: dict[int, NodeMeta],
    curve_meta_dict: dict[int, CurveMeta],
    curve_point_meta_dict: dict[int, CurvePointMeta],
    surface_meta_dict: dict[int, SurfaceMeta],
    surface_point_meta_dict: dict[int, SurfacePointMeta],
) -> NameLookupState:
    """Stage 2: Build globally-unique name -> id mappings."""
    rel_name_to_rf_id: dict[str, int] = {}
    container_name_to_id: dict[str, int] = {}

    # Nodes: name is the risk-factor name
    for n in node_meta_dict.values():
        rel_name_to_rf_id[n.node_name] = int(n.node_id)

    # Curve points: "{curve_name}{tenor_token}"
    for pt in curve_point_meta_dict.values():
        rel_name = f"{pt.curve_name}{pt.tenor_token}"
        if rel_name in rel_name_to_rf_id:
            raise ValueError(
                f"Duplicate relative risk-factor name {rel_name!r} "
                f"(rf_id {pt.rf_id} collides with {rel_name_to_rf_id[rel_name]})."
            )
        rel_name_to_rf_id[rel_name] = int(pt.rf_id)

    # Surface points: "{surface_name}{x_tenor_token}[{y_label}]"
    for pt in surface_point_meta_dict.values():
        rel_name = f"{pt.surface_name}{pt.x_tenor_token}[{pt.y_label}]"
        if rel_name in rel_name_to_rf_id:
            raise ValueError(
                f"Duplicate relative risk-factor name {rel_name!r} "
                f"(rf_id {pt.rf_id} collides with {rel_name_to_rf_id[rel_name]})."
            )
        rel_name_to_rf_id[rel_name] = int(pt.rf_id)

    # Containers: curves and surfaces
    for cm in curve_meta_dict.values():
        container_name_to_id[cm.curve_name] = int(cm.curve_id)

    for sm in surface_meta_dict.values():
        if sm.surface_name in container_name_to_id:
            raise ValueError(
                f"Container name {sm.surface_name!r} declared as both "
                "curve and surface."
            )
        container_name_to_id[sm.surface_name] = int(sm.surface_id)

    # Cross-namespace disjointness
    overlap = set(rel_name_to_rf_id) & set(container_name_to_id)
    if overlap:
        raise ValueError(
            f"Names appear in BOTH the risk-factor and container "
            f"namespaces: {sorted(overlap)}"
        )

    return NameLookupState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=rel_name_to_rf_id,
        container_name_to_id=container_name_to_id,
    )


def _build_rf_state(
    state: NameLookupState,
    node_meta_dict: dict[int, NodeMeta],
    curve_meta_dict: dict[int, CurveMeta],
    curve_point_meta_dict: dict[int, CurvePointMeta],
    surface_meta_dict: dict[int, SurfaceMeta],
    surface_point_meta_dict: dict[int, SurfacePointMeta],
) -> RfState:
    """Stage 3: Build per-risk-factor dense attribute arrays."""
    rf_ids: list[int] = (
        [n.node_id for n in node_meta_dict.values()]
        + [p.rf_id for p in curve_point_meta_dict.values()]
        + [p.rf_id for p in surface_point_meta_dict.values()]
    )
    max_rf_id = max(rf_ids) if rf_ids else 0
    n = max_rf_id + 1

    parent_topology_type = np.full(n, -1, dtype=np.int8)
    parent_id = np.full(n, -1, dtype=np.int32)
    ccy_id = np.full(n, -1, dtype=np.int8)
    shift_id = np.full(n, -1, dtype=np.int8)
    # Unoccupied slots carry an explicit negative sentinel, never NaN, so a
    # stale rf_id fails loud downstream instead of silently propagating NaN
    # into computed levels. (RQ-AUD-012.)
    unit_scale = np.full(n, float(INVALID_UNIT_SCALE), dtype=np.float64)

    ccy_name_to_id = state.ccy_name_to_id

    for node in node_meta_dict.values():
        i = node.node_id
        parent_topology_type[i] = np.int8(ContainerTopologyType.NODE.value)
        parent_id[i] = i
        ccy_id[i] = ccy_name_to_id[node.quote_ccy]
        shift_id[i] = _resolve_shift_id(node.shift_type)
        unit_scale[i] = _resolve_unit_scale_factor(node.unit_scale)

    for pt in curve_point_meta_dict.values():
        cm = curve_meta_dict[pt.curve_id]
        i = pt.rf_id
        parent_topology_type[i] = np.int8(ContainerTopologyType.CURVE.value)
        parent_id[i] = pt.curve_id
        ccy_id[i] = ccy_name_to_id[cm.quote_ccy]
        shift_id[i] = _resolve_shift_id(cm.shift_type)
        unit_scale[i] = _resolve_unit_scale_factor(cm.unit_scale)

    for pt in surface_point_meta_dict.values():
        sm = surface_meta_dict[pt.surface_id]
        i = pt.rf_id
        parent_topology_type[i] = np.int8(ContainerTopologyType.SURFACE.value)
        parent_id[i] = pt.surface_id
        ccy_id[i] = ccy_name_to_id[sm.quote_ccy]
        shift_id[i] = _resolve_shift_id(sm.shift_type)
        unit_scale[i] = _resolve_unit_scale_factor(sm.unit_scale)

    return RfState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=state.rel_name_to_rf_id,
        container_name_to_id=state.container_name_to_id,
        max_rf_id=max_rf_id,
        rf_id_to_parent_topology_type=parent_topology_type,
        rf_id_to_parent_id=parent_id,
        rf_id_to_ccy_id=ccy_id,
        rf_id_to_shift_type_id=shift_id,
        rf_id_to_unit_scale_factor=unit_scale,
    )


def _build_container_state(
    state: RfState,
    curve_meta_dict: dict[int, CurveMeta],
    surface_meta_dict: dict[int, SurfaceMeta],
) -> ContainerState:
    """
    Stage 4: Build per-container SLOT-indexed dense attribute arrays.

    ``container_ids`` is the sorted-ascending-unique axis; every other
    array is parallel to it. A container's slot is its position in
    ``container_ids`` (resolved at read time via searchsorted). Curve and
    surface ids are disjoint by namespace, so their sorted union is
    unambiguous. Memory is O(num_containers), independent of how large
    the raw ids are (they live at 250M+).
    """
    all_ids: list[int] = [c.curve_id for c in curve_meta_dict.values()] + [
        s.surface_id for s in surface_meta_dict.values()
    ]
    container_ids = np.array(sorted(all_ids), dtype=np.int32)
    n = int(container_ids.shape[0])
    id_to_slot: dict[int, int] = {
        int(cid): slot for slot, cid in enumerate(container_ids.tolist())
    }

    topology_type = np.full(n, -1, dtype=np.int8)
    interp_x = np.full(n, -1, dtype=np.int8)
    extrap_x = np.full(n, -1, dtype=np.int8)
    interp_y = np.full(n, -1, dtype=np.int8)
    extrap_y = np.full(n, -1, dtype=np.int8)
    cont_ccy = np.full(n, -1, dtype=np.int8)
    x_kind = np.full(n, -1, dtype=np.int8)

    ccy_name_to_id = state.ccy_name_to_id

    for cm in curve_meta_dict.values():
        i = id_to_slot[cm.curve_id]
        topology_type[i] = ContainerTopologyType.CURVE.value
        interp_x[i] = _resolve_interp_id(cm.x_interp_method)
        extrap_x[i] = _resolve_extrap_id(cm.x_extrap_method)
        cont_ccy[i] = ccy_name_to_id[cm.quote_ccy]
        x_kind[i] = int(XAxisKind[cm.x_axis_kind])

    for sm in surface_meta_dict.values():
        i = id_to_slot[sm.surface_id]
        topology_type[i] = ContainerTopologyType.SURFACE.value
        interp_x[i] = _resolve_interp_id(sm.x_interp_method)
        extrap_x[i] = _resolve_extrap_id(sm.x_extrap_method)
        interp_y[i] = _resolve_interp_id(sm.y_interp_method)
        extrap_y[i] = _resolve_extrap_id(sm.y_extrap_method)
        cont_ccy[i] = ccy_name_to_id[sm.quote_ccy]
        x_kind[i] = int(XAxisKind[sm.x_axis_kind])

    return ContainerState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=state.rel_name_to_rf_id,
        container_name_to_id=state.container_name_to_id,
        max_rf_id=state.max_rf_id,
        rf_id_to_parent_topology_type=state.rf_id_to_parent_topology_type,
        rf_id_to_parent_id=state.rf_id_to_parent_id,
        rf_id_to_ccy_id=state.rf_id_to_ccy_id,
        rf_id_to_shift_type_id=state.rf_id_to_shift_type_id,
        rf_id_to_unit_scale_factor=state.rf_id_to_unit_scale_factor,
        container_ids=container_ids,
        container_topology_type=topology_type,
        container_interp_x_id=interp_x,
        container_extrap_x_id=extrap_x,
        container_interp_y_id=interp_y,
        container_extrap_y_id=extrap_y,
        container_ccy_id=cont_ccy,
        container_x_axis_kind=x_kind,
    )


def _build_curve_tenor_index(
    curve_meta_dict: dict[int, CurveMeta],
    curve_point_meta_dict: dict[int, CurvePointMeta],
) -> dict[int, CurveTenorIndex]:
    """
    Build one CurveTenorIndex per curve.

    OFFSET curves: sorted offset axis + parallel rf_id array, ready for
    `np.searchsorted(offsets, requested_offset)` at runtime.
    LABEL curves : direct dict label -> rf_id.
    """
    by_curve_offsets: dict[int, list[int]] = {}
    by_curve_ids: dict[int, list[int]] = {}
    by_curve_labels: dict[int, dict[str, int]] = {}

    for pt in curve_point_meta_dict.values():
        cm = curve_meta_dict[pt.curve_id]
        x_kind = XAxisKind[cm.x_axis_kind]

        if x_kind == XAxisKind.LABEL:
            unit, offset = parse_bracket_token(pt.tenor_token)
            if offset is not None:
                raise ValueError(
                    f"LABEL curve {cm.curve_name!r} contains an offset token "
                    f"{pt.tenor_token!r}; LABEL curves accept label-only tokens."
                )
            by_curve_labels.setdefault(pt.curve_id, {})[unit] = int(pt.rf_id)
            continue

        _, offset = parse_bracket_token(pt.tenor_token)
        if offset is None:
            raise ValueError(
                f"OFFSET curve {cm.curve_name!r} contains a label token "
                f"{pt.tenor_token!r}; OFFSET curves require a signed integer offset."
            )
        by_curve_offsets.setdefault(pt.curve_id, []).append(int(offset))
        by_curve_ids.setdefault(pt.curve_id, []).append(int(pt.rf_id))

    out: dict[int, CurveTenorIndex] = {}

    for curve_id, offsets_list in by_curve_offsets.items():
        offsets_arr = np.asarray(offsets_list, dtype=np.int64)
        ids_arr = np.asarray(by_curve_ids[curve_id], dtype=np.int32)
        order = np.argsort(offsets_arr, kind="mergesort")
        out[curve_id] = CurveTenorIndex(
            sorted_offsets=offsets_arr[order],
            rf_ids_by_offset=ids_arr[order],
            label_to_rf_id=by_curve_labels.get(curve_id, {}),
        )

    for curve_id, lbl_map in by_curve_labels.items():
        if curve_id in out:
            continue
        out[curve_id] = CurveTenorIndex(
            sorted_offsets=np.empty(0, dtype=np.int64),
            rf_ids_by_offset=np.empty(0, dtype=np.int32),
            label_to_rf_id=lbl_map,
        )

    # Curves with zero points are legal at compile time
    for curve_id in curve_meta_dict:
        if curve_id not in out:
            out[curve_id] = CurveTenorIndex(
                sorted_offsets=np.empty(0, dtype=np.int64),
                rf_ids_by_offset=np.empty(0, dtype=np.int32),
                label_to_rf_id={},
            )

    return out


def _build_curve_tenor_state(
    state: ContainerState,
    curve_meta_dict: dict[int, CurveMeta],
    curve_point_meta_dict: dict[int, CurvePointMeta],
) -> CurveTenorState:
    """Stage 5: Build per-curve tenor index."""
    curve_id_to_tenor_index = _build_curve_tenor_index(
        curve_meta_dict, curve_point_meta_dict
    )

    return CurveTenorState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=state.rel_name_to_rf_id,
        container_name_to_id=state.container_name_to_id,
        max_rf_id=state.max_rf_id,
        rf_id_to_parent_topology_type=state.rf_id_to_parent_topology_type,
        rf_id_to_parent_id=state.rf_id_to_parent_id,
        rf_id_to_ccy_id=state.rf_id_to_ccy_id,
        rf_id_to_shift_type_id=state.rf_id_to_shift_type_id,
        rf_id_to_unit_scale_factor=state.rf_id_to_unit_scale_factor,
        container_ids=state.container_ids,
        container_topology_type=state.container_topology_type,
        container_interp_x_id=state.container_interp_x_id,
        container_extrap_x_id=state.container_extrap_x_id,
        container_interp_y_id=state.container_interp_y_id,
        container_extrap_y_id=state.container_extrap_y_id,
        container_ccy_id=state.container_ccy_id,
        container_x_axis_kind=state.container_x_axis_kind,
        curve_id_to_tenor_index=curve_id_to_tenor_index,
    )


def _build_fx_state(
    state: CurveTenorState,
    node_meta_dict: dict[int, NodeMeta],
) -> FxState:
    """Stage 6: Build FX spot lookup table."""
    out: dict[tuple[int, int], int] = {}
    ccy_name_to_id = state.ccy_name_to_id

    for node in node_meta_dict.values():
        if node.fx_base_ccy is None:
            continue
        base_id = ccy_name_to_id[node.fx_base_ccy]
        quote_id = ccy_name_to_id[node.quote_ccy]
        key = (int(base_id), int(quote_id))
        if (key[1], key[0]) in out:
            # D060 / NEW-06: one orientation per currency pair. Two independently
            # quoted reciprocals are two sources for one economic rate; the
            # resolver would pick one silently and the other could disagree.
            raise ValueError(
                f"FX spot pair (base={node.fx_base_ccy}, quote={node.quote_ccy}): "
                f"node {node.node_name} is the reciprocal of an FX node already "
                "configured. Configure exactly one orientation per currency pair."
            )
        if key in out:
            raise ValueError(
                f"Duplicate FX spot pair "
                f"(base={node.fx_base_ccy}, quote={node.quote_ccy}): "
                f"node {node.node_name} collides with another FX node."
            )
        out[key] = int(node.node_id)

    return FxState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=state.rel_name_to_rf_id,
        container_name_to_id=state.container_name_to_id,
        max_rf_id=state.max_rf_id,
        rf_id_to_parent_topology_type=state.rf_id_to_parent_topology_type,
        rf_id_to_parent_id=state.rf_id_to_parent_id,
        rf_id_to_ccy_id=state.rf_id_to_ccy_id,
        rf_id_to_shift_type_id=state.rf_id_to_shift_type_id,
        rf_id_to_unit_scale_factor=state.rf_id_to_unit_scale_factor,
        container_ids=state.container_ids,
        container_topology_type=state.container_topology_type,
        container_interp_x_id=state.container_interp_x_id,
        container_extrap_x_id=state.container_extrap_x_id,
        container_interp_y_id=state.container_interp_y_id,
        container_extrap_y_id=state.container_extrap_y_id,
        container_ccy_id=state.container_ccy_id,
        container_x_axis_kind=state.container_x_axis_kind,
        curve_id_to_tenor_index=state.curve_id_to_tenor_index,
        fx_spot_lookup=out,
    )


def _build_fx_matrix_state(
    state: FxState,
) -> FxMatrixState:
    """Stage 6b: Build dense FX spot matrix."""
    n = state.max_ccy_id + 1
    matrix = np.full(
        shape=(n, n),
        fill_value=NO_FX_RF_REQUIRED,
        dtype=np.int32,
    )
    for (b, q), rf_id in state.fx_spot_lookup.items():
        matrix[int(b), int(q)] = np.int32(rf_id)

    return FxMatrixState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=state.rel_name_to_rf_id,
        container_name_to_id=state.container_name_to_id,
        max_rf_id=state.max_rf_id,
        rf_id_to_parent_topology_type=state.rf_id_to_parent_topology_type,
        rf_id_to_parent_id=state.rf_id_to_parent_id,
        rf_id_to_ccy_id=state.rf_id_to_ccy_id,
        rf_id_to_shift_type_id=state.rf_id_to_shift_type_id,
        rf_id_to_unit_scale_factor=state.rf_id_to_unit_scale_factor,
        container_ids=state.container_ids,
        container_topology_type=state.container_topology_type,
        container_interp_x_id=state.container_interp_x_id,
        container_extrap_x_id=state.container_extrap_x_id,
        container_interp_y_id=state.container_interp_y_id,
        container_extrap_y_id=state.container_extrap_y_id,
        container_ccy_id=state.container_ccy_id,
        container_x_axis_kind=state.container_x_axis_kind,
        curve_id_to_tenor_index=state.curve_id_to_tenor_index,
        fx_spot_lookup=state.fx_spot_lookup,
        fx_spot_matrix=matrix,
    )


def _build_discount_state(
    state: FxMatrixState,
    curve_meta_dict: dict[int, CurveMeta],
) -> DiscountState:
    """Stage 7: Build discount-curve lookup tables.  Terminal stage."""
    max_ccy_id = state.max_ccy_id
    ccy_name_to_id = state.ccy_name_to_id

    default_per_ccy = np.full(max_ccy_id + 1, _NO_DISCOUNT_CURVE_ID, dtype=np.int32)
    explicit: dict[tuple[int, int], int] = {}

    for cm in curve_meta_dict.values():
        if cm.discount_basis == DiscountBasis.NOT_A_DISCOUNT_CURVE.name:
            continue
        ccy_id = ccy_name_to_id[cm.quote_ccy]
        basis_id = np.int8(int(DiscountBasis[cm.discount_basis]))
        key = (int(ccy_id), int(basis_id))
        if key in explicit:
            raise ValueError(
                f"Duplicate discount curve for "
                f"(ccy={cm.quote_ccy}, basis={cm.discount_basis}): "
                f"curve {cm.curve_name}."
            )
        explicit[key] = int(cm.curve_id)

        if cm.is_default_discount_for_ccy:
            if default_per_ccy[int(ccy_id)] != _NO_DISCOUNT_CURVE_ID:
                raise ValueError(
                    f"More than one default discount curve flagged for "
                    f"ccy={cm.quote_ccy}."
                )
            default_per_ccy[int(ccy_id)] = np.int32(cm.curve_id)

    return DiscountState(
        ccy_name_to_id=state.ccy_name_to_id,
        ccy_id_to_name=state.ccy_id_to_name,
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=state.rel_name_to_rf_id,
        container_name_to_id=state.container_name_to_id,
        max_rf_id=state.max_rf_id,
        rf_id_to_parent_topology_type=state.rf_id_to_parent_topology_type,
        rf_id_to_parent_id=state.rf_id_to_parent_id,
        rf_id_to_ccy_id=state.rf_id_to_ccy_id,
        rf_id_to_shift_type_id=state.rf_id_to_shift_type_id,
        rf_id_to_unit_scale_factor=state.rf_id_to_unit_scale_factor,
        container_ids=state.container_ids,
        container_topology_type=state.container_topology_type,
        container_interp_x_id=state.container_interp_x_id,
        container_extrap_x_id=state.container_extrap_x_id,
        container_interp_y_id=state.container_interp_y_id,
        container_extrap_y_id=state.container_extrap_y_id,
        container_ccy_id=state.container_ccy_id,
        container_x_axis_kind=state.container_x_axis_kind,
        curve_id_to_tenor_index=state.curve_id_to_tenor_index,
        fx_spot_lookup=state.fx_spot_lookup,
        fx_spot_matrix=state.fx_spot_matrix,
        ccy_id_to_default_discount_curve_id=default_per_ccy,
        discount_curve_lookup=explicit,
    )


# ============================================================
# Freeze
# ============================================================


def _freeze_state(state: DiscountState) -> DiscountState:
    """
    Return a deeply-immutable copy of the terminal pipeline state.

    Dictionaries are wrapped in MappingProxyType; arrays are set read-only.
    No validation is performed — the compiler guarantees correctness.
    """
    return DiscountState(
        ccy_name_to_id=_freeze_dict(state.ccy_name_to_id),
        ccy_id_to_name=tuple(state.ccy_id_to_name),
        max_ccy_id=state.max_ccy_id,
        rel_name_to_rf_id=_freeze_dict(state.rel_name_to_rf_id),
        container_name_to_id=_freeze_dict(state.container_name_to_id),
        max_rf_id=state.max_rf_id,
        rf_id_to_parent_topology_type=_freeze_array(
            state.rf_id_to_parent_topology_type
        ),
        rf_id_to_parent_id=_freeze_array(state.rf_id_to_parent_id),
        rf_id_to_ccy_id=_freeze_array(state.rf_id_to_ccy_id),
        rf_id_to_shift_type_id=_freeze_array(state.rf_id_to_shift_type_id),
        rf_id_to_unit_scale_factor=_freeze_array(state.rf_id_to_unit_scale_factor),
        container_ids=_freeze_array(state.container_ids),
        container_topology_type=_freeze_array(state.container_topology_type),
        container_interp_x_id=_freeze_array(state.container_interp_x_id),
        container_extrap_x_id=_freeze_array(state.container_extrap_x_id),
        container_interp_y_id=_freeze_array(state.container_interp_y_id),
        container_extrap_y_id=_freeze_array(state.container_extrap_y_id),
        container_ccy_id=_freeze_array(state.container_ccy_id),
        container_x_axis_kind=_freeze_array(state.container_x_axis_kind),
        curve_id_to_tenor_index=_freeze_dict(state.curve_id_to_tenor_index),
        fx_spot_lookup=_freeze_dict(state.fx_spot_lookup),
        fx_spot_matrix=_freeze_array(state.fx_spot_matrix),
        ccy_id_to_default_discount_curve_id=_freeze_array(
            state.ccy_id_to_default_discount_curve_id
        ),
        discount_curve_lookup=_freeze_dict(state.discount_curve_lookup),
    )


# ============================================================
# Main runtime container
# ============================================================


@dataclass(frozen=True, slots=True)
class RiskStaticConfig:
    """
    Compiled, Pandas-free static configuration.

    Constructed by ``compile_risk_static_config_from_dict_of_dfs``.
    All derived fields are computed eagerly in ``__post_init__`` via a
    type-state pipeline and frozen before the constructor returns.

    Public access is through delegating properties that read from the
    internal frozen ``_state``.
    """

    # ------------------------------------------------------------------ #
    # Init fields — supplied by compile.py                               #
    # ------------------------------------------------------------------ #

    ccy_dict: dict[str, Ccy] = field(repr=False)
    holiday_calendars_dict: dict[str, HolidayCalendar] = field(repr=False)

    node_meta_dict: dict[int, NodeMeta] = field(repr=False)
    curve_meta_dict: dict[int, CurveMeta] = field(repr=False)
    curve_point_meta_dict: dict[int, CurvePointMeta] = field(repr=False)
    surface_meta_dict: dict[int, SurfaceMeta] = field(repr=False)
    surface_point_meta_dict: dict[int, SurfacePointMeta] = field(repr=False)

    # ------------------------------------------------------------------ #
    # Internal frozen state (populated in __post_init__)                 #
    # ------------------------------------------------------------------ #

    _state: DiscountState = field(init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """
        Run the type-state pipeline and freeze the result.

        The pipeline order is encoded in the type signatures:
        each stage accepts exactly the state produced by the previous
        stage.  It is structurally impossible for a future maintainer
        to read a field before it has been computed.
        """
        if not self.ccy_dict:
            raise ValueError("ccy_dict is empty.")
        if not (self.node_meta_dict or self.curve_meta_dict or self.surface_meta_dict):
            raise ValueError(
                "RiskStaticConfig has no nodes, no curves, and no surfaces."
            )

        # Pipeline: Currency -> NameLookup -> Rf -> Container -> Tenor -> Fx -> Discount
        s1 = _build_currency_state(self.ccy_dict)
        s2 = _build_name_lookup_state(
            s1,
            self.node_meta_dict,
            self.curve_meta_dict,
            self.curve_point_meta_dict,
            self.surface_meta_dict,
            self.surface_point_meta_dict,
        )
        s3 = _build_rf_state(
            s2,
            self.node_meta_dict,
            self.curve_meta_dict,
            self.curve_point_meta_dict,
            self.surface_meta_dict,
            self.surface_point_meta_dict,
        )
        s4 = _build_container_state(
            s3,
            self.curve_meta_dict,
            self.surface_meta_dict,
        )
        s5 = _build_curve_tenor_state(
            s4,
            self.curve_meta_dict,
            self.curve_point_meta_dict,
        )
        s6 = _build_fx_state(
            s5,
            self.node_meta_dict,
        )
        s7 = _build_fx_matrix_state(
            s6,
        )
        s8 = _build_discount_state(
            s7,
            self.curve_meta_dict,
        )

        frozen = _freeze_state(s8)
        object.__setattr__(self, "_state", frozen)

    # ------------------------------------------------------------------ #
    # Delegating properties — public API                                 #
    # ------------------------------------------------------------------ #

    # -- Currency layer --
    @property
    def ccy_name_to_id(self) -> Mapping[str, int]:
        return self._state.ccy_name_to_id

    @property
    def ccy_id_to_name(self) -> tuple[str, ...]:
        return self._state.ccy_id_to_name

    @property
    def max_ccy_id(self) -> int:
        return self._state.max_ccy_id

    # -- Name lookups --
    @property
    def rel_name_to_rf_id(self) -> Mapping[str, int]:
        return self._state.rel_name_to_rf_id

    @property
    def container_name_to_id(self) -> Mapping[str, int]:
        return self._state.container_name_to_id

    # -- Per-rf attributes --
    @property
    def max_rf_id(self) -> int:
        return self._state.max_rf_id

    @property
    def rf_id_to_parent_topology_type(self) -> npt.NDArray[np.int8]:
        return self._state.rf_id_to_parent_topology_type

    @property
    def rf_id_to_parent_id(self) -> npt.NDArray[np.int32]:
        return self._state.rf_id_to_parent_id

    @property
    def rf_id_to_ccy_id(self) -> npt.NDArray[np.int8]:
        return self._state.rf_id_to_ccy_id

    @property
    def rf_id_to_shift_type_id(self) -> npt.NDArray[np.int8]:
        return self._state.rf_id_to_shift_type_id

    @property
    def rf_id_to_unit_scale_factor(self) -> npt.NDArray[np.float64]:
        return self._state.rf_id_to_unit_scale_factor

    # -- Per-container attributes (SLOT-indexed; parallel to container_ids) --
    @property
    def container_ids(self) -> npt.NDArray[np.int32]:
        """Sorted-ascending-unique container-id axis. Index every
        ``container_*`` array by SLOT (``container_slots``), not raw id."""
        return self._state.container_ids

    @property
    def num_containers(self) -> int:
        return int(self._state.container_ids.shape[0])

    @property
    def max_container_id(self) -> int:
        """Largest container id present (informational only — NOT an array
        length; the per-container arrays are slot-length, not id-length)."""
        ids = self._state.container_ids
        return int(ids[-1]) if ids.size else int(NO_ID)

    def container_slots(
        self, container_ids: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int64]:
        """Resolve raw container ids -> slots into the ``container_*`` arrays.

        Raises if any id is absent. Container ids live at 250M+, so the
        per-container store is keyed by slot (this lookup), never by raw id.
        """
        return searchsorted_or_raise(
            universe=self._state.container_ids,
            raw=container_ids,
            ctx="container_id",
        )

    def container_slot(self, container_id: int) -> int:
        """Scalar form of ``container_slots`` for loop/diagnostic sites."""
        return int(self.container_slots(np.asarray([container_id], dtype=np.int32))[0])

    @property
    def container_topology_type(self) -> npt.NDArray[np.int8]:
        return self._state.container_topology_type

    @property
    def container_interp_x_id(self) -> npt.NDArray[np.int8]:
        return self._state.container_interp_x_id

    @property
    def container_extrap_x_id(self) -> npt.NDArray[np.int8]:
        return self._state.container_extrap_x_id

    @property
    def container_interp_y_id(self) -> npt.NDArray[np.int8]:
        return self._state.container_interp_y_id

    @property
    def container_extrap_y_id(self) -> npt.NDArray[np.int8]:
        return self._state.container_extrap_y_id

    @property
    def container_ccy_id(self) -> npt.NDArray[np.int8]:
        return self._state.container_ccy_id

    @property
    def container_x_axis_kind(self) -> npt.NDArray[np.int8]:
        return self._state.container_x_axis_kind

    # -- Curve tenor index --
    @property
    def curve_id_to_tenor_index(self) -> Mapping[int, CurveTenorIndex]:
        return self._state.curve_id_to_tenor_index

    # -- FX spot lookup --
    @property
    def fx_spot_lookup(self) -> Mapping[tuple[int, int], int]:
        return self._state.fx_spot_lookup

    @property
    def fx_spot_matrix(self) -> npt.NDArray[np.int32]:
        return self._state.fx_spot_matrix

    # -- Discount-curve lookup --
    @property
    def ccy_id_to_default_discount_curve_id(self) -> npt.NDArray[np.int32]:
        return self._state.ccy_id_to_default_discount_curve_id

    @property
    def discount_curve_lookup(self) -> Mapping[tuple[int, int], int]:
        return self._state.discount_curve_lookup

    # ------------------------------------------------------------------ #
    # Scalar accessors and summary                                       #
    # ------------------------------------------------------------------ #

    @property
    def num_ccys(self) -> int:
        return len(self.ccy_dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_meta_dict)

    @property
    def num_curves(self) -> int:
        return len(self.curve_meta_dict)

    @property
    def num_curve_points(self) -> int:
        return len(self.curve_point_meta_dict)

    @property
    def num_surfaces(self) -> int:
        return len(self.surface_meta_dict)

    @property
    def num_surface_points(self) -> int:
        return len(self.surface_point_meta_dict)

    @property
    def num_risk_factors(self) -> int:
        return self.num_nodes + self.num_curve_points + self.num_surface_points

    @property
    def num_fx_pairs(self) -> int:
        return len(self._state.fx_spot_lookup)

    def get_rf_id(self, rel_name: str) -> np.int32:
        """Lookup helper. Raises KeyError with the input name on miss."""
        return np.int32(self._state.rel_name_to_rf_id[rel_name])

    def get_container_id(self, container_name: str) -> np.int32:
        """Lookup helper. Raises KeyError with the input name on miss."""
        return np.int32(self._state.container_name_to_id[container_name])

    def summary(self) -> None:
        """Print a console-friendly overview of the compiled config."""
        lines = [
            "================================",
            "=== RiskStaticConfig Summary ===",
            "================================",
            f"  Currencies          : {self.num_ccys}",
            f"  Calendars           : {len(self.holiday_calendars_dict)}",
            f"  Nodes               : {self.num_nodes}",
            f"  Curves              : {self.num_curves}",
            f"  Curve points        : {self.num_curve_points}",
            f"  Surfaces            : {self.num_surfaces}",
            f"  Surface points      : {self.num_surface_points}",
            f"  Risk factors total  : {self.num_risk_factors}",
            f"  Containers total    : {self.num_containers}",
            f"  FX pairs            : {self.num_fx_pairs}",
            f"  Max rf_id           : {self.max_rf_id}",
            f"  Max container_id    : {self.max_container_id}",
            "================================",
        ]
        print("\n".join(lines))
