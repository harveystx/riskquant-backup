# src/riskquant/proxy_resolver/proxy.py
"""
Vectorized resolver: required risk-factor ids -> market-data source ids.

Consumes two separately-owned contracts:
  - RiskStaticConfig owns the universe and id <-> name mappings.
  - ProxyTopology    owns proxy mappings (dense int32 / float64 arrays).

Resolution priority (first match wins, per required_id):

  (1) DIRECT
        required_id is present in market_data_factor_ids.
        source = required_id, scale = 1.0

  (2) RF-level proxy (ID_PROXY)
        proxy_topology.rf_id_to_proxy_rf_id[required_id] >= 0.
        source = proxy_rf_id, scale = rf_id_to_proxy_vol_scale[required_id].
        The proxy MUST itself exist in market_data B (runtime check).

  (3) Container-level proxy (CONTAINER_PROXY)
        required's parent container is proxied:
        proxy_topology.container_id_to_proxy_id[parent_id] >= 0.
        Find the matching tenor point on the proxy container, restricted
        to B. Apply container-level scale.
        Curves only. Surface container-proxy raises NotImplementedError —
        surface tenor matching needs an (x, y) joint resolver, a separate
        work item.

  (4) Same-container tenor fallback (TENOR_FALLBACK)
        No proxy registered; required_id is missing from B, but its
        parent container has at least one tenor present in B. Match
        nearest tenor on the same container, restricted to B.
        Curves only. Surfaces raise NotImplementedError.

  (5) NOT_SIMULATED / hard error
        Nothing above resolved. Raise PrettyKeyError with the offending
        required_id list — exposes ingest completeness gaps to the
        operator.

Single-hop invariants (already enforced upstream by proxy_config.validate):
  - No A -> B -> C chains.
  - No container-level + child-rf-level conflict on the same container.

This resolver assumes those hold; if they don't, the priority ordering
would still yield a deterministic answer but it may not match operator
intent. The validator is the contract.

Hot-path discipline:
  - No `for rf in required_ids` Python loop.
  - Per-id resolution is via dense int gathers + boolean masks.
  - Tenor matching uses one batched np.searchsorted per unique
    effective-container, not per required_id.
  - Only allocations are the output arrays + small per-container scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError

# ============================================================
# Resolution codes
# ============================================================

RES_NOT_SIMULATED: Final[np.int8] = np.int8(-1)
RES_DIRECT: Final[np.int8] = np.int8(0)
RES_TENOR_FALLBACK: Final[np.int8] = np.int8(1)
RES_ID_PROXY: Final[np.int8] = np.int8(2)
RES_CONTAINER_PROXY: Final[np.int8] = np.int8(3)


# Parent-kind values on static_config.rf_id_to_parent_kind.
# Source of truth is the RiskStaticConfig dataclass; duplicated here as
# named constants for readability.
_PK_UNOCCUPIED: Final[int] = -1
_PK_NODE: Final[int] = 0
_PK_CURVE_POINT: Final[int] = 1
_PK_SURFACE_POINT: Final[int] = 2


# ============================================================
# Output container
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class ProxyResolution:
    """
    Compiled resolver output for one (required_ids, market_data_ids) pair.

    Attributes
    ----------
    required_factor_ids
        (N,) int32 — domain of resolution; the portfolio's required rfs.
    source_factor_ids
        (N,) int32 — for each required, the resolved source rf in B.
    required_to_source_index
        (N,) int32 — index into unique_source_factor_ids per required.
        Enables compute-once-broadcast-many in the shock kernel.
    shock_scale
        (N,) float32 — multiplicative scale applied to the source shock.
    unique_source_factor_ids
        (K,) int32 — deduplicated sources in first-seen order.
    resolution_code
        (N,) int8 — RES_* per required; see module docstring.
    """

    required_factor_ids: npt.NDArray[np.int32]
    source_factor_ids: npt.NDArray[np.int32]
    required_to_source_index: npt.NDArray[np.int32]
    shock_scale: npt.NDArray[np.float32]
    unique_source_factor_ids: npt.NDArray[np.int32]
    resolution_code: npt.NDArray[np.int8]


# ============================================================
# Internal: B-restricted curve tenor index
# ============================================================


@dataclass(frozen=True, slots=True)
class _CurveTenorIndexB:
    """
    B-restricted tenor index for ONE curve.

    Only rf_ids that exist in market_data_factor_ids appear here. The
    resolver matches required tenor tokens against these B-restricted
    points (LABEL exact or OFFSET nearest) so the chosen source is
    guaranteed to be in B.
    """

    sorted_offsets: npt.NDArray[np.int64]
    rf_ids_by_offset: npt.NDArray[np.int32]
    label_to_rf_id: dict[str, int]


def _build_b_restricted_curve_indices(
    static_config: RiskStaticConfig,
    market_data_factor_ids: npt.NDArray[np.int32],
) -> dict[int, _CurveTenorIndexB]:
    """
    Build per-curve B-restricted tenor indices.

    Strategy: filter `market_data_factor_ids` down to curve points (via
    static_config.rf_id_to_parent_kind == 1), then group by curve_id and
    bucket each group's tokens into offsets vs labels.

    Allocation cost: O(Nb) total. Grouping is one mergesort + one
    np.unique pass; the per-curve inner loop iterates over CURVES, not
    over rfs.
    """
    if market_data_factor_ids.size == 0:
        return {}

    parent_kind = static_config.rf_id_to_parent_topology_type[market_data_factor_ids]
    is_curve_pt = parent_kind == _PK_CURVE_POINT
    md_curve_rfs = market_data_factor_ids[is_curve_pt]
    if md_curve_rfs.size == 0:
        return {}

    md_curve_parent_ids = static_config.rf_id_to_parent_id[md_curve_rfs]

    order = np.argsort(md_curve_parent_ids, kind="mergesort")
    sorted_curve_ids = md_curve_parent_ids[order]
    sorted_rf_ids = md_curve_rfs[order]
    unique_curves, group_starts = np.unique(sorted_curve_ids, return_index=True)
    group_ends = np.concatenate(
        [group_starts[1:], np.array([sorted_curve_ids.size], dtype=group_starts.dtype)]
    )

    out: dict[int, _CurveTenorIndexB] = {}
    cp_dict = static_config.curve_point_meta_dict
    cm_dict = static_config.curve_meta_dict

    for curve_id_np, start, end in zip(
        unique_curves, group_starts, group_ends, strict=True
    ):
        curve_id = int(curve_id_np)
        rfs_in_curve = sorted_rf_ids[start:end]

        cm = cm_dict[curve_id]
        x_kind = XAxisKind[cm.x_axis_kind]

        offset_list: list[int] = []
        offset_rf_list: list[int] = []
        label_dict: dict[str, int] = {}

        for rf_id_np in rfs_in_curve:
            rf_id = int(rf_id_np)
            pt = cp_dict[rf_id]
            tok = pt.tenor_token

            if x_kind == XAxisKind.LABEL:
                # Token is "[ON]", "[1W]", etc. — strip brackets.
                label_dict[tok[1:-1]] = rf_id
            else:
                offset_list.append(_parse_offset_from_token(tok))
                offset_rf_list.append(rf_id)

        if offset_list:
            offsets_arr = np.asarray(offset_list, dtype=np.int64)
            ids_arr = np.asarray(offset_rf_list, dtype=np.int32)
            sort_order = np.argsort(offsets_arr, kind="mergesort")
            sorted_offsets = offsets_arr[sort_order]
            sorted_ids = ids_arr[sort_order]
        else:
            sorted_offsets = np.empty(0, dtype=np.int64)
            sorted_ids = np.empty(0, dtype=np.int32)

        out[curve_id] = _CurveTenorIndexB(
            sorted_offsets=sorted_offsets,
            rf_ids_by_offset=sorted_ids,
            label_to_rf_id=label_dict,
        )

    return out


def _parse_offset_from_token(tok: str) -> int:
    """
    Parse the signed integer offset from a bracketed OFFSET token.

    The static-config validator guarantees `tok` matches one of:
        [M+k]  [M-k]   [Y+k]  [Y-k]   [HY+k]  [HY-k]
    """
    plus = tok.find("+")
    if plus > 0:
        return int(tok[plus + 1 : -1])
    minus = tok.find("-")
    if minus > 0:
        return -int(tok[minus + 1 : -1])
    raise ValueError(f"OFFSET token without sign: {tok!r}")


# ============================================================
# Internal: nearest-offset gather
# ============================================================


def _nearest_offset_match_batch(
    *,
    sorted_offsets: npt.NDArray[np.int64],
    rf_ids_by_offset: npt.NDArray[np.int32],
    targets: npt.NDArray[np.int64],
) -> npt.NDArray[np.int32]:
    """
    Batched nearest-neighbour match on signed integer offsets.

    For each target offset, return the rf_id whose offset is closest.
    Ties go to the lower offset (deterministic).

    Vectorized via one np.searchsorted + boolean math; no per-target loop.
    """
    if sorted_offsets.size == 0:
        raise ValueError("nearest-offset match called on an empty offset array.")

    # j: position where each target would insert to keep array sorted.
    j = np.searchsorted(sorted_offsets, targets, side="left")

    # Clip j into [1, len-1] so j-1 and j are always valid; endpoint
    # cases get fixed up after.
    j_clipped = np.clip(j, 1, sorted_offsets.size - 1)
    lo_offsets = sorted_offsets[j_clipped - 1]
    hi_offsets = sorted_offsets[j_clipped]

    dist_lo = targets - lo_offsets
    dist_hi = hi_offsets - targets

    # Deterministic tie-break: prefer lower (j-1) when dist_lo <= dist_hi.
    choose_lo = dist_lo <= dist_hi
    chosen_j = np.where(choose_lo, j_clipped - 1, j_clipped)

    # Endpoint clamps for targets outside [min_offset, max_offset].
    chosen_j = np.where(j == 0, np.int64(0), chosen_j)
    chosen_j = np.where(
        j == sorted_offsets.size, np.int64(sorted_offsets.size - 1), chosen_j
    )

    return rf_ids_by_offset[chosen_j]


# ============================================================
# Internal: per-required tenor-token gather from static_config
# ============================================================


def _gather_curve_point_offsets(
    static_config: RiskStaticConfig,
    rf_ids: npt.NDArray[np.int32],
) -> npt.NDArray[np.int64]:
    """For each rf_id (a curve point), return its signed-integer offset."""
    cp_dict = static_config.curve_point_meta_dict
    return np.asarray(
        [_parse_offset_from_token(cp_dict[int(rf)].tenor_token) for rf in rf_ids],
        dtype=np.int64,
    )


def _gather_curve_point_labels(
    static_config: RiskStaticConfig,
    rf_ids: npt.NDArray[np.int32],
) -> list[str]:
    """For each rf_id (a LABEL curve point), return the bracket-stripped label."""
    cp_dict = static_config.curve_point_meta_dict
    return [cp_dict[int(rf)].tenor_token[1:-1] for rf in rf_ids]


def _required_curve_x_kinds(
    static_config: RiskStaticConfig,
    rf_ids: npt.NDArray[np.int32],
) -> list[str]:
    """For each rf_id (curve point), return its parent curve's x_axis_kind."""
    cp_dict = static_config.curve_point_meta_dict
    cm_dict = static_config.curve_meta_dict
    return [cm_dict[cp_dict[int(rf)].curve_id].x_axis_kind for rf in rf_ids]


# ============================================================
# Public API
# ============================================================


def resolve_required_factor_ids_to_market_data_factor_ids(
    *,
    static_config: RiskStaticConfig,
    proxy_topology: ProxyTopology,
    required_factor_ids: npt.NDArray[np.int32],
    market_data_factor_ids: npt.NDArray[np.int32],
) -> ProxyResolution:
    """
    Resolve every required rf_id to a source rf_id present in market data.

    See the module docstring for the resolution priority ladder.

    Parameters
    ----------
    static_config
        Compiled static config; supplies parent-kind / parent-id and
        per-curve metadata.
    proxy_topology
        Compiled proxy topology; supplies rf-level and container-level
        proxy gathers as dense int32 arrays.
    required_factor_ids
        (N,) int32 — portfolio-required rfs to resolve.
    market_data_factor_ids
        (Nb,) int32 — rfs available in the market-data matrix.

    Returns
    -------
    ProxyResolution
        Fully populated output with source ids, scales, and resolution
        codes per required id.

    Raises
    ------
    PrettyKeyError
        If any required_id cannot be resolved through any priority tier.
    NotImplementedError
        If a surface-point requires container-proxy or tenor-fallback
        resolution. Direct lookup and rf-level proxy on surfaces work.
    TypeError
        If input dtypes are wrong.
    """
    _smoke_test_inputs(required_factor_ids, market_data_factor_ids)

    n = int(required_factor_ids.shape[0])
    if n == 0:
        return _empty_resolution()

    # ------------------------------------------------------------------ #
    # Phase 1: vectorized gather of per-required attributes              #
    # ------------------------------------------------------------------ #

    parent_kind = static_config.rf_id_to_parent_topology_type[required_factor_ids]
    parent_id = static_config.rf_id_to_parent_id[required_factor_ids]

    if (parent_kind == _PK_UNOCCUPIED).any():
        bad_rfs = required_factor_ids[parent_kind == _PK_UNOCCUPIED]
        raise PrettyKeyError(
            "required_factor_ids reference unoccupied rf_id slots in "
            f"static_config: {bad_rfs.tolist()[:25]}"
        )

    # Direct availability via dense bitmap (faster than np.isin at scale).
    md_bitmap = np.zeros(static_config.max_rf_id + 1, dtype=bool)
    md_bitmap[market_data_factor_ids] = True
    direct_avail = md_bitmap[required_factor_ids]

    # RF-level proxy (dense int32 / float64 gathers).
    rf_proxy_id_arr = proxy_topology.rf_id_to_proxy_rf_id[required_factor_ids]
    rf_proxy_scale_arr = proxy_topology.rf_id_to_proxy_vol_scale[required_factor_ids]
    has_rf_proxy = rf_proxy_id_arr >= 0

    # Container-level proxy: only valid for curve / surface children.
    # For nodes (parent_kind == 0), parent_id equals the node's rf_id —
    # which lives in the rf-id space, not container-id. Guard with a
    # safe-index pattern before the gather.
    is_container_child = parent_kind > _PK_NODE
    safe_parent_id = np.where(is_container_child, parent_id, np.int32(0))

    if is_container_child.any():
        max_pid = int(safe_parent_id[is_container_child].max())
        if max_pid > proxy_topology.container_id_to_proxy_id.size - 1:
            raise PrettyKeyError(
                "parent_id out of range for proxy_topology.container_id_to_proxy_id; "
                f"max parent_id={max_pid}, array length="
                f"{proxy_topology.container_id_to_proxy_id.size}"
            )

    cont_proxy_id_raw = proxy_topology.container_id_to_proxy_id[safe_parent_id]
    cont_proxy_scale_raw = proxy_topology.container_id_to_proxy_vol_scale[
        safe_parent_id
    ]
    container_proxy_id = np.where(
        is_container_child, cont_proxy_id_raw, np.int32(-1)
    ).astype(np.int32)
    container_proxy_scale = np.where(
        is_container_child, cont_proxy_scale_raw, np.float64(1.0)
    ).astype(np.float64)
    has_container_proxy = container_proxy_id >= 0

    # ------------------------------------------------------------------ #
    # Phase 2: vectorized resolution-code assignment                     #
    # ------------------------------------------------------------------ #

    code = np.full(n, RES_NOT_SIMULATED, dtype=np.int8)
    code = np.where(direct_avail, RES_DIRECT, code).astype(np.int8)
    needs_proxy = ~direct_avail
    code = np.where(needs_proxy & has_rf_proxy, RES_ID_PROXY, code).astype(np.int8)
    code = np.where(
        needs_proxy & ~has_rf_proxy & has_container_proxy,
        RES_CONTAINER_PROXY,
        code,
    ).astype(np.int8)
    # Remaining ~direct & ~rf_proxy & ~container_proxy is a candidate for
    # tenor fallback — assigned in Phase 3d after we know which actually
    # resolve.

    # ------------------------------------------------------------------ #
    # Phase 3: source-id assembly                                        #
    # ------------------------------------------------------------------ #

    source_factor_ids = np.empty(n, dtype=np.int32)
    shock_scale = np.empty(n, dtype=np.float32)
    # Sentinel-fill so unresolved slots don't carry garbage if a later
    # raise occurs partway through; aids debugging.
    source_factor_ids.fill(-1)
    shock_scale.fill(0.0)

    # ---- 3a. DIRECT ----
    mask = code == RES_DIRECT
    if mask.any():
        source_factor_ids[mask] = required_factor_ids[mask]
        shock_scale[mask] = np.float32(1.0)

    # ---- 3b. RF-level proxy ----
    mask = code == RES_ID_PROXY
    if mask.any():
        proxy_ids = rf_proxy_id_arr[mask]
        # Proxy must exist in market data B (FK passed against
        # static_config but B is a runtime universe).
        proxies_in_b = md_bitmap[proxy_ids]
        if not proxies_in_b.all():
            bad_required = required_factor_ids[mask][~proxies_in_b]
            bad_proxies = proxy_ids[~proxies_in_b]
            pairs = list(zip(bad_required.tolist(), bad_proxies.tolist()))[:10]
            raise PrettyKeyError(
                "RF-level proxy resolved to rf_id(s) absent from market data:\n"
                + "\n".join(f"  required={r} -> proxy={p}" for r, p in pairs)
            )
        source_factor_ids[mask] = proxy_ids
        shock_scale[mask] = rf_proxy_scale_arr[mask].astype(np.float32)

    # ---- Build B-restricted curve index ONCE if Phase 3c or 3d needs it ----
    needs_curve_index = (
        (code == RES_CONTAINER_PROXY) | (code == RES_NOT_SIMULATED)
    ).any()
    b_curve_index = (
        _build_b_restricted_curve_indices(static_config, market_data_factor_ids)
        if needs_curve_index
        else {}
    )

    # ---- 3c. Container-level proxy ----
    mask = code == RES_CONTAINER_PROXY
    if mask.any():
        # Surface container-proxy is out of scope — needs (x, y) joint resolver.
        surface_in_mask = mask & (parent_kind == _PK_SURFACE_POINT)
        if surface_in_mask.any():
            bad = required_factor_ids[surface_in_mask]
            raise NotImplementedError(
                "Container-level proxy on surfaces requires a joint (x, y) "
                "tenor resolver that has not been implemented. Affected "
                f"required rf_ids: {bad.tolist()[:10]}"
            )

        _resolve_container_proxy_inplace(
            static_config=static_config,
            required_factor_ids=required_factor_ids,
            mask=mask,
            container_proxy_id=container_proxy_id,
            container_proxy_scale=container_proxy_scale,
            b_curve_index=b_curve_index,
            source_factor_ids=source_factor_ids,
            shock_scale=shock_scale,
        )

    # ---- 3d. Tenor fallback (LAST PRIORITY) ----
    mask = code == RES_NOT_SIMULATED
    if mask.any():
        # Surface tenor fallback out of scope.
        surface_in_mask = mask & (parent_kind == _PK_SURFACE_POINT)
        if surface_in_mask.any():
            bad = required_factor_ids[surface_in_mask]
            raise NotImplementedError(
                "Same-container tenor fallback on surfaces requires a joint "
                "(x, y) resolver that has not been implemented. Affected "
                f"required rf_ids: {bad.tolist()[:10]}"
            )

        # Nodes have no peers within a container — unresolved nodes are
        # hard ingest gaps.
        node_in_mask = mask & (parent_kind == _PK_NODE)
        if node_in_mask.any():
            bad = required_factor_ids[node_in_mask]
            raise PrettyKeyError(
                "Required node rf_id(s) missing from market data with no "
                f"proxy configured: {bad.tolist()[:25]}"
            )

        _resolve_tenor_fallback_inplace(
            static_config=static_config,
            required_factor_ids=required_factor_ids,
            mask=mask,
            parent_id=parent_id,
            b_curve_index=b_curve_index,
            source_factor_ids=source_factor_ids,
            shock_scale=shock_scale,
            code=code,
        )

    # Any remaining unresolved -> hard error.
    still_unresolved = code == RES_NOT_SIMULATED
    if still_unresolved.any():
        bad = required_factor_ids[still_unresolved]
        raise PrettyKeyError(
            "Required rf_id(s) could not be resolved through any tier "
            "(no direct, no rf proxy, no container proxy, no same-container "
            f"tenor fallback): {bad.tolist()[:25]}"
        )

    # ------------------------------------------------------------------ #
    # Phase 4: unique source universe + required -> source index         #
    # ------------------------------------------------------------------ #

    unique_sources, inverse = _unique_first_seen(source_factor_ids)
    required_to_source_index = inverse.astype(np.int32, copy=False)

    return ProxyResolution(
        required_factor_ids=required_factor_ids,
        source_factor_ids=source_factor_ids,
        required_to_source_index=required_to_source_index,
        shock_scale=shock_scale,
        unique_source_factor_ids=unique_sources,
        resolution_code=code,
    )


# ============================================================
# Resolver phases (in-place writers)
# ============================================================


def _resolve_container_proxy_inplace(
    *,
    static_config: RiskStaticConfig,
    required_factor_ids: npt.NDArray[np.int32],
    mask: npt.NDArray[np.bool_],
    container_proxy_id: npt.NDArray[np.int32],
    container_proxy_scale: npt.NDArray[np.float64],
    b_curve_index: dict[int, _CurveTenorIndexB],
    source_factor_ids: npt.NDArray[np.int32],
    shock_scale: npt.NDArray[np.float32],
) -> None:
    """
    For rfs flagged RES_CONTAINER_PROXY, match the required's tenor onto
    the proxy curve's B-restricted points.

    Writes into source_factor_ids and shock_scale in-place.

    Grouping by proxy_curve_id lets us do ONE np.searchsorted per unique
    proxy curve instead of per required id.
    """
    req_rfs = required_factor_ids[mask]
    proxy_curve_ids = container_proxy_id[mask]
    scales = container_proxy_scale[mask].astype(np.float32)

    unique_proxies, inverse = np.unique(proxy_curve_ids, return_inverse=True)
    mask_positions = np.flatnonzero(mask)

    for group_uix, proxy_curve_id in enumerate(unique_proxies):
        group_in_mask = inverse == group_uix
        target_positions = mask_positions[group_in_mask]
        group_required_rfs = req_rfs[group_in_mask]

        idx = b_curve_index.get(int(proxy_curve_id))
        if idx is None:
            raise PrettyKeyError(
                "Container-level proxy targets a curve with no points in market data:\n"
                f"  proxy_curve_id={int(proxy_curve_id)}, "
                f"required_rf_ids={group_required_rfs.tolist()[:5]}"
            )

        # Required rfs in this group may come from DIFFERENT required
        # curves all proxying onto the same proxy curve. Enforce that
        # they share x_axis_kind so the matching strategy is well-defined.
        x_kinds = _required_curve_x_kinds(static_config, group_required_rfs)
        unique_x_kinds = set(x_kinds)
        if len(unique_x_kinds) != 1:
            raise PrettyKeyError(
                "Container-level proxy: required curves with different "
                f"x_axis_kinds {sorted(unique_x_kinds)} all proxy onto "
                f"curve_id={int(proxy_curve_id)}. The matching strategy "
                "is ambiguous; split the proxies so each x_axis_kind has "
                "its own target."
            )
        req_x_kind = next(iter(unique_x_kinds))

        if req_x_kind == "LABEL":
            labels = _gather_curve_point_labels(static_config, group_required_rfs)
            chosen = np.empty(group_required_rfs.size, dtype=np.int32)
            for i, lbl in enumerate(labels):
                hit = idx.label_to_rf_id.get(lbl)
                if hit is None:
                    raise PrettyKeyError(
                        "Container-level proxy: LABEL tenor "
                        f"[{lbl}] from required_rf_id={int(group_required_rfs[i])} "
                        f"not present in proxy curve_id={int(proxy_curve_id)} "
                        f"(market-data-restricted)."
                    )
                chosen[i] = np.int32(hit)
        else:
            if idx.sorted_offsets.size == 0:
                raise PrettyKeyError(
                    f"Container-level proxy: proxy curve_id={int(proxy_curve_id)} "
                    "has no OFFSET tenors in market data to match."
                )
            targets = _gather_curve_point_offsets(static_config, group_required_rfs)
            chosen = _nearest_offset_match_batch(
                sorted_offsets=idx.sorted_offsets,
                rf_ids_by_offset=idx.rf_ids_by_offset,
                targets=targets,
            )

        source_factor_ids[target_positions] = chosen
        shock_scale[target_positions] = scales[group_in_mask]


def _resolve_tenor_fallback_inplace(
    *,
    static_config: RiskStaticConfig,
    required_factor_ids: npt.NDArray[np.int32],
    mask: npt.NDArray[np.bool_],
    parent_id: npt.NDArray[np.int32],
    b_curve_index: dict[int, _CurveTenorIndexB],
    source_factor_ids: npt.NDArray[np.int32],
    shock_scale: npt.NDArray[np.float32],
    code: npt.NDArray[np.int8],
) -> None:
    """
    Same-container tenor fallback. For each required_id in `mask`:
      - look up its parent (same container).
      - match the required's tenor against B-restricted points on the
        same container (LABEL exact or OFFSET nearest).
      - scale = 1.0.

    Writes source_factor_ids, shock_scale, and code in-place. Rows that
    cannot resolve via fallback stay at RES_NOT_SIMULATED; the caller
    raises on those.
    """
    req_rfs = required_factor_ids[mask]
    req_parent_ids = parent_id[mask]
    mask_positions = np.flatnonzero(mask)

    unique_parents, inverse = np.unique(req_parent_ids, return_inverse=True)

    for group_uix, container_id in enumerate(unique_parents):
        group_in_mask = inverse == group_uix
        target_positions = mask_positions[group_in_mask]
        group_required_rfs = req_rfs[group_in_mask]

        idx = b_curve_index.get(int(container_id))
        if idx is None:
            # Container has no points in B at all; nothing to fall back to.
            # Leave code=RES_NOT_SIMULATED for caller to surface.
            continue

        # All rfs in this group share the same parent curve, hence the
        # same x_axis_kind. One lookup suffices.
        x_kinds = _required_curve_x_kinds(static_config, group_required_rfs[:1])
        req_x_kind = x_kinds[0]

        if req_x_kind == "LABEL":
            labels = _gather_curve_point_labels(static_config, group_required_rfs)
            chosen = np.full(group_required_rfs.size, -1, dtype=np.int32)
            for i, lbl in enumerate(labels):
                hit = idx.label_to_rf_id.get(lbl)
                if hit is not None:
                    chosen[i] = np.int32(hit)
            resolved = chosen >= 0
            source_factor_ids[target_positions[resolved]] = chosen[resolved]
            shock_scale[target_positions[resolved]] = np.float32(1.0)
            code[target_positions[resolved]] = RES_TENOR_FALLBACK
            # Unresolved within group stay at RES_NOT_SIMULATED.

        else:
            if idx.sorted_offsets.size == 0:
                # Container has labels in B but no offsets; cannot match
                # OFFSET required against LABEL availability.
                continue
            targets = _gather_curve_point_offsets(static_config, group_required_rfs)
            chosen = _nearest_offset_match_batch(
                sorted_offsets=idx.sorted_offsets,
                rf_ids_by_offset=idx.rf_ids_by_offset,
                targets=targets,
            )
            source_factor_ids[target_positions] = chosen
            shock_scale[target_positions] = np.float32(1.0)
            code[target_positions] = RES_TENOR_FALLBACK


# ============================================================
# Helpers
# ============================================================


def _unique_first_seen(
    arr: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Return (unique_first_seen, inverse).

    np.unique sorts; we need first-seen order for deterministic
    shock-compute plans across reruns of identical inputs.

    Strategy: get sorted-unique + first-index + inverse-sorted via one
    np.unique call, then permute to first-seen order.
    """
    if arr.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    _, first_idx, inverse_sorted = np.unique(
        arr, return_index=True, return_inverse=True
    )
    # Reorder unique to first-seen order.
    order = np.argsort(first_idx, kind="mergesort")
    unique_first_seen = arr[first_idx[order]]
    # Build inverse_first_seen: sorted_idx -> first_seen_idx.
    sorted_to_first_seen = np.empty(order.size, dtype=np.int32)
    sorted_to_first_seen[order] = np.arange(order.size, dtype=np.int32)
    inverse_first_seen = sorted_to_first_seen[inverse_sorted].astype(
        np.int32, copy=False
    )

    return unique_first_seen.astype(np.int32, copy=False), inverse_first_seen


def _smoke_test_inputs(
    required_factor_ids: npt.NDArray[np.int32],
    market_data_factor_ids: npt.NDArray[np.int32],
) -> None:
    if required_factor_ids.dtype != np.int32:
        raise TypeError(
            f"required_factor_ids must be int32; got {required_factor_ids.dtype}"
        )
    if market_data_factor_ids.dtype != np.int32:
        raise TypeError(
            f"market_data_factor_ids must be int32; got {market_data_factor_ids.dtype}"
        )
    if required_factor_ids.ndim != 1:
        raise TypeError("required_factor_ids must be 1D.")
    if market_data_factor_ids.ndim != 1:
        raise TypeError("market_data_factor_ids must be 1D.")


def _empty_resolution() -> ProxyResolution:
    return ProxyResolution(
        required_factor_ids=np.empty(0, dtype=np.int32),
        source_factor_ids=np.empty(0, dtype=np.int32),
        required_to_source_index=np.empty(0, dtype=np.int32),
        shock_scale=np.empty(0, dtype=np.float32),
        unique_source_factor_ids=np.empty(0, dtype=np.int32),
        resolution_code=np.empty(0, dtype=np.int8),
    )
