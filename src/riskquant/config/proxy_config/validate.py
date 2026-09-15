# src/riskquant/config/proxy_config/validate.py
"""
Two-stage validation for proxy configuration.

Stage 1: `validate_proxy_config_df`
    Runs against the active-filtered DataFrame. Vectorized — no row
    iteration. Catches schema, type, null, blank, and domain-range
    issues before any name resolution is attempted.

Stage 2: `validate_proxy_maps`
    Runs against the typed ProxyMap dicts (after compile partitions them
    by level) and the compiled RiskStaticConfig. Catches:
      - foreign-key drift against the static config,
      - self-references,
      - chained proxies (A->B->C is forbidden — multi-hop is not allowed),
      - cross-level conflicts (a container is proxied AND one of its
        child risk factors is also proxied).

The override path on ProxyTopology calls Stage 2 directly without
roundtripping through a DataFrame.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from riskquant.config.enums.container_topology import ContainerTopologyType
from riskquant.config.proxy_config.schema_objects import ProxyMap

if TYPE_CHECKING:
    from riskquant.config.static_config.static_config import RiskStaticConfig


# ============================================================
# Constants
# ============================================================


VALID_PROXYING_LEVELS: frozenset[str] = frozenset({"CONTAINER", "RISK_FACTOR"})

# vol_scaling_factor sanity bounds. The lower bound is strict (a zero
# scale silently zeros the proxy's vol, which is a class-A modelling bug);
# the upper bound is conservative but catches typos like missing decimal
# point ("15" intended as "1.5").
_VOL_SCALE_MIN_EXCLUSIVE: float = 0.0
_VOL_SCALE_MAX_INCLUSIVE: float = 10.0


REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"proxying_level", "required_name", "proxy_name", "vol_scaling_factor"}
)


# ============================================================
# Generic helpers
# ============================================================


def _check_no_blank_strings(df: pl.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    blanks = df[col].cast(pl.Utf8).str.strip_chars() == ""
    if blanks.any():
        idx = [i for i, v in enumerate(blanks.to_list()) if v][:10]
        raise ValueError(
            f"proxy_config.{col}: blank string at rows {idx}. "
            "Loader fills null cells with the empty string; the validator "
            "treats both as invalid."
        )


# ============================================================
# Stage 1 — DataFrame-tier validation
# ============================================================


def validate_proxy_config_df(df: pl.DataFrame) -> None:
    """
    Validate an active-filtered proxy DataFrame.

    Tiers:
      1. Required columns present.
      2. No null values in required columns.
      3. No blank strings in string columns.
      4. proxying_level is in VALID_PROXYING_LEVELS.
      5. vol_scaling_factor in (0, 10].
      6. (proxying_level, required_name) is unique — a required name
         can only be proxied once per level.
      7. required_name != proxy_name (self-reference).

    Cross-table checks (FK against static_config, chains, cycles,
    cross-level conflicts) are deferred to `validate_proxy_maps`,
    which runs after the DataFrame has been partitioned by level.

    Raises
    ------
    ValueError
        On the first violation encountered, with a descriptive message.
    """
    if df.is_empty():
        return  # An empty proxy map is valid: no proxies applied.

    # --- 1. required columns ---
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"proxy_config missing required column(s): {sorted(missing)}.\n"
            f"Found: {list(df.columns)}"
        )

    # --- 2. no nulls in required columns ---
    null_counts = {
        col: df[col].null_count() for col in sorted(REQUIRED_COLUMNS)
    }
    null_cols = [col for col, n in null_counts.items() if n > 0]
    if null_cols:
        raise ValueError(
            f"proxy_config has null values in required column(s): {null_cols}."
        )

    # --- 3. no blank strings (loader fills nulls with "" which would slip past null checks) ---
    for col in ("proxying_level", "required_name", "proxy_name"):
        _check_no_blank_strings(df, col)

    # --- 4. proxying_level domain ---
    # Coerce to uppercase before set membership (loader leaves case as-is).
    level_norm = df["proxying_level"].cast(pl.Utf8).str.to_uppercase()
    invalid = ~level_norm.is_in(list(VALID_PROXYING_LEVELS))
    if invalid.any():
        bad = sorted(set(level_norm.filter(invalid).to_list()))
        raise ValueError(
            f"proxy_config has invalid proxying_level value(s): {bad}.\n"
            f"Must be one of: {sorted(VALID_PROXYING_LEVELS)} (case-insensitive)."
        )

    # --- 5. vol_scaling_factor range ---
    scale = df["vol_scaling_factor"].cast(pl.Float64)
    scale_np = scale.to_numpy()
    bad_scale = (
        ~np.isfinite(scale_np)
        | (scale_np <= _VOL_SCALE_MIN_EXCLUSIVE)
        | (scale_np > _VOL_SCALE_MAX_INCLUSIVE)
    )
    if bad_scale.any():
        offending = df.filter(pl.Series(bad_scale)).select(
            ["required_name", "proxy_name", "vol_scaling_factor"]
        )
        raise ValueError(
            f"proxy_config has vol_scaling_factor outside "
            f"({_VOL_SCALE_MIN_EXCLUSIVE}, {_VOL_SCALE_MAX_INCLUSIVE}]:\n"
            f"{offending}"
        )

    # --- 6. (proxying_level, required_name) composite uniqueness ---
    work = df.with_columns(
        [
            level_norm.alias("_level"),
            df["required_name"].cast(pl.Utf8).str.strip_chars().alias("_required"),
        ]
    )
    dupes_mask = work.select(["_level", "_required"]).is_duplicated()
    if dupes_mask.any():
        offending = df.filter(dupes_mask).select(
            ["proxying_level", "required_name", "proxy_name"]
        )
        raise ValueError(
            "proxy_config has duplicate (proxying_level, required_name) rows. "
            "A required_name can be proxied at most once per level.\n"
            f"Conflicting rows:\n{offending}"
        )

    # --- 7. self-reference ---
    self_ref = (
        df["required_name"].cast(pl.Utf8).str.strip_chars()
        == df["proxy_name"].cast(pl.Utf8).str.strip_chars()
    )
    if self_ref.any():
        offending = df.filter(self_ref).select(["proxying_level", "required_name"])
        raise ValueError(
            "proxy_config has self-referencing entries "
            "(required_name == proxy_name):\n"
            f"{offending}"
        )


# ============================================================
# Stage 2 — ProxyMap-tier validation (FK + chain + cross-level)
# ============================================================


def _check_chain(proxies: dict[str, ProxyMap], level_label: str) -> None:
    """
    Forbid multi-hop proxies within a single level.

    A chain is any (A -> B) where B is itself a required_name in another
    (B -> C) entry. Multi-hop is banned: every proxy must point at a
    terminal node. Cycles are a special case of chains where some B == A,
    so this single check covers cycles too.
    """
    required_set: set[str] = set(proxies.keys())
    proxy_targets: list[tuple[str, str]] = [
        (pm.required_name, pm.proxy_name) for pm in proxies.values()
    ]

    chained: list[tuple[str, str]] = [
        (req, prx) for req, prx in proxy_targets if prx in required_set
    ]
    if chained:
        raise ValueError(
            f"Chained {level_label} proxies are not allowed (multi-hop is "
            f"banned — every proxy must point at a terminal name).\n"
            f"The proxy targets below are themselves required_names in other "
            f"rows:\n"
            + "\n".join(
                f"  {req!r} -> {prx!r}  (but {prx!r} is also proxied)"
                for req, prx in chained
            )
            + f"\nResolve by pointing both {level_label} mappings at a "
            "common terminal name."
        )


def _check_fk(
    proxies: dict[str, ProxyMap],
    name_to_id: dict[str, int | np.int32] | Mapping[str, int | np.int32],
    namespace_label: str,
) -> None:
    """
    Every required_name and proxy_name must resolve in name_to_id.

    Bulk set-difference: one error message lists every unknown name,
    not one PrettyKeyError per row.
    """
    referenced: set[str] = set()
    for pm in proxies.values():
        referenced.add(pm.required_name)
        referenced.add(pm.proxy_name)

    unknown = referenced - set(name_to_id)
    if unknown:
        raise ValueError(
            f"proxy_config references {len(unknown)} unknown {namespace_label} "
            f"name(s): {sorted(unknown)[:25]}.\n"
            f"Every required_name and proxy_name must exist in the static "
            f"config's {namespace_label} namespace."
        )


def _check_cross_level_conflict(
    container_proxies: dict[str, ProxyMap],
    rf_proxies: dict[str, ProxyMap],
    static_config: RiskStaticConfig,
) -> None:
    """
    Forbid a container being proxied while any of its child risk factors
    is ALSO proxied.

    Rationale: container-level "use the whole proxy container's curve"
    and rf-level "use this specific other point" are competing policies;
    coexisting them silently mis-applies the user's intent.
    """
    if not container_proxies or not rf_proxies:
        return

    # Resolve each proxied container to its container_id.
    name_to_cid = static_config.container_name_to_id
    topology_type_arr = (
        static_config.container_topology_type
    )  # slot-indexed: 0=node, 1=curve, 2=surface

    proxied_curve_ids: set[int] = set()
    proxied_surface_ids: set[int] = set()
    for pm in container_proxies.values():
        cid = int(name_to_cid[pm.required_name])
        topology_type = int(topology_type_arr[static_config.container_slot(cid)])
        if topology_type == ContainerTopologyType.CURVE.value:
            proxied_curve_ids.add(cid)
        elif topology_type == ContainerTopologyType.SURFACE.value:
            proxied_surface_ids.add(cid)

    if not (proxied_curve_ids or proxied_surface_ids):
        # FK passed but no proxied container resolved to curve or surface.
        # This should be unreachable; if we get here, container_topology_type
        # has a bug or container_proxies contains a node name (forbidden).
        raise ValueError(
            "internal: cross-level conflict check found proxied containers "
            "with topology_type not in {curve, surface}."
        )

    # Walk every rf-level proxy. The required_name is a relative
    # risk-factor name; its parent_id is on the dense rf_id_to_parent_id
    # array. If parent_id is in any proxied container set, conflict.
    name_to_rfid = static_config.rel_name_to_rf_id
    parent_topology_type_arr = static_config.rf_id_to_parent_topology_type
    parent_id_arr = static_config.rf_id_to_parent_id

    conflicts: list[tuple[str, str]] = []
    for pm in rf_proxies.values():
        rfid = int(name_to_rfid[pm.required_name])
        topology_type = int(parent_topology_type_arr[rfid])
        if topology_type <= ContainerTopologyType.NODE.value:
            continue  # Topology type is NODE (0) or unoccupied (-1) - skip.
        pid = int(parent_id_arr[rfid])
        if (
            topology_type == ContainerTopologyType.CURVE.value
            and pid in proxied_curve_ids
        ):
            conflicts.append((pm.required_name, _curve_name(pid, static_config)))
        elif (
            topology_type == ContainerTopologyType.SURFACE.value
            and pid in proxied_surface_ids
        ):
            conflicts.append((pm.required_name, _surface_name(pid, static_config)))

    if conflicts:
        lines = [
            f"  {rf_name!r} (parent={parent_name!r})"
            for rf_name, parent_name in conflicts
        ]
        raise ValueError(
            "proxy_config has cross-level conflicts: the rf-level proxies "
            "below belong to containers that are ALSO container-level "
            "proxied.\n"
            "Container-level proxy and rf-level proxy are competing "
            "policies; choose ONE per affected container.\n" + "\n".join(lines)
        )


def _curve_name(curve_id: int, sc: RiskStaticConfig) -> str:
    cm = sc.curve_meta_dict.get(curve_id)
    return cm.curve_name if cm is not None else f"<curve_id={curve_id}>"


def _surface_name(surface_id: int, sc: RiskStaticConfig) -> str:
    sm = sc.surface_meta_dict.get(surface_id)
    return sm.surface_name if sm is not None else f"<surface_id={surface_id}>"


def validate_proxy_maps(
    *,
    container_proxies: dict[str, ProxyMap],
    rf_proxies: dict[str, ProxyMap],
    static_config: RiskStaticConfig,
) -> None:
    """
    Validate partitioned proxy maps against the static config.

    Runs after compile.py has split the DataFrame by level (or after
    overrides have been merged in `ProxyTopology.with_overrides`).

    Checks
    ------
      A. Foreign-key: every name in container_proxies resolves in
         `static_config.container_name_to_id`; every name in rf_proxies
         resolves in `static_config.rel_name_to_rf_id`.
      B. Chain ban: no proxy target is itself a required name (per level).
         Implicitly catches cycles, since a cycle requires the target to
         be a required name.
      C. Cross-level: no container is both container-proxied and has any
         of its child risk factors rf-proxied.

    Raises
    ------
    ValueError
        On the first violation, with a descriptive message naming every
        offending pair / name found.
    """
    _check_fk(container_proxies, static_config.container_name_to_id, "container")
    _check_fk(rf_proxies, static_config.rel_name_to_rf_id, "risk-factor")
    _check_chain(container_proxies, "container")
    _check_chain(rf_proxies, "risk-factor")
    _check_cross_level_conflict(container_proxies, rf_proxies, static_config)
