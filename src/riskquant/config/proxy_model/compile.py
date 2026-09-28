# src/riskquant/config/proxy_model/compile.py
"""
Proxy V1 compiler: authored terms -> canonical run-bound proxy operator (D088).

Dependency order (D088 §6, §7; identity is downstream of resolution):

    authored terms
      -> resolution at the run as-of          (names, aliases, EXACT coordinates)
      -> admissibility                        (V1 target universe, V1 driver charts,
                                               finite loadings, EXACT kinds)
      -> totality closure over the run axis   (identity row for every V1 axis
                                               node without authored terms)
      -> canonicalisation                     (canonical.py, D088 §6)
      -> CompiledProxyModel                   (identity: hash_proxy_model)

The compiled model is a function of exactly four inputs: the authored terms,
the static configuration, the as-of date and the run axis. The as-of enters
only through name resolution and EXACT coordinates; the run axis enters only
through totality closure (which rows exist, which container pillars expand) and
row order. Nothing here knows about scenario data, world columns, horizons,
scenario counts, batches, column indices, base levels, reporting currency,
valuation or P&L. Target reconstruction is not performed here: the target chart
``kappa_A`` is read from the static configuration only to (a) check the target
is inside the V1 universe and (b) write the driver chart of a materialised
identity term.

Nothing selects a source. Data availability is not an input. A driver is a
factor-node plus a chart; whether a world supplies that column is validated by
the run later, never decided here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from riskquant.config.enums.container_topology import ContainerTopologyType
from riskquant.config.proxy_model.canonical import canonicalise_resolved_terms
from riskquant.config.proxy_model.compiled import (
    V1_DRIVER_CHART_IDS,
    V1_TARGET_CHART_IDS,
    CompiledProxyModel,
)
from riskquant.config.proxy_model.schema_objects import (
    ContainerProxyTerm,
    NodeProxyTerm,
)
from riskquant.config.static_config.rf_resolution import (
    RiskFactorAbsToRelNameIdMap,
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.exceptions import ConfigurationError
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_TYPE_NAME_TO_ID_REGISTRY,
    ShiftType,
)

__all__ = ["compile_proxy_model"]

_UNOCCUPIED_TOPOLOGY = -1
_CURVE = int(ContainerTopologyType.CURVE)
_SURFACE = int(ContainerTopologyType.SURFACE)

# (term index, target rf_id, driver rf_id, driver chart id, loading)
_ResolvedTerm = tuple[int, int, int, int, float]


def compile_proxy_model(
    *,
    authored_terms: Sequence[NodeProxyTerm | ContainerProxyTerm],
    static_config: RiskStaticConfig,
    as_of_date: date,
    run_axis_rf_ids: npt.NDArray[np.int32],
) -> CompiledProxyModel:
    """
    Compile authored Proxy V1 terms into the canonical run-bound operator.

    Parameters
    ----------
    authored_terms
        Authored node and container terms. Order is irrelevant to the result.
        Every term is validated, including terms whose target is not on this
        run's axis (they are well-formed configuration but produce no row).
    static_config
        Compiled static configuration: identities, configured shift types,
        container metadata and pillar sets.
    as_of_date
        Run as-of date. Resolves absolute-name aliases and EXACT coordinates.
    run_axis_rf_ids
        The run's full factor axis (int32, unique, occupied rf_ids), in axis
        order. Only nodes whose configured shift type is a V1 chart become
        rows; ABSOLUTE_BASE and NO_SHIFT axis nodes keep their D025 path.

    Returns
    -------
    CompiledProxyModel
        One canonical row per V1 axis node, in axis order.

    Raises
    ------
    ConfigurationError
        Any unresolved identity, inadmissible target or chart, invalid loading,
        EXACT kind mismatch, empty canonical row or non-finite merged loading.
    TypeError, ValueError
        Structurally malformed inputs (wrong types, dtypes, rank, duplicates).
    """
    _require_as_of(as_of_date)
    axis = _require_run_axis(run_axis_rf_ids, static_config)
    shift_ids = static_config.rf_id_to_shift_type_id

    axis_kappa = shift_ids[axis].astype(np.int64)
    v1_mask = np.isin(axis_kappa, np.array(sorted(V1_TARGET_CHART_IDS), dtype=np.int64))
    v1_axis = axis[v1_mask]
    row_of_target = {int(rf): r for r, rf in enumerate(v1_axis.tolist())}

    name_map = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=static_config, as_of_date=as_of_date
    )

    resolved: list[_ResolvedTerm] = []
    authored_targets: set[int] = set()
    for idx, term in enumerate(authored_terms):
        if isinstance(term, NodeProxyTerm):
            node_term = _resolve_node_term(idx, term, static_config, name_map)
            authored_targets.add(node_term[1])
            resolved.append(node_term)
        elif isinstance(term, ContainerProxyTerm):
            covered, expanded = _expand_container_term(
                idx, term, static_config, name_map, row_of_target
            )
            authored_targets.update(covered)
            resolved.extend(expanded)
        else:
            raise TypeError(
                f"authored_terms[{idx}] must be NodeProxyTerm or "
                f"ContainerProxyTerm, got {type(term).__name__}."
            )

    # Totality closure: every V1 axis node without authored terms receives its
    # identity row (A, kappa_A, 1.0). A node with authored terms never falls
    # back to identity, even if all its terms later vanish (that row is rejected).
    term_rows: list[int] = []
    term_drivers: list[int] = []
    term_charts: list[int] = []
    term_loadings: list[float] = []
    for _, target, driver, chart, loading in resolved:
        row = row_of_target.get(target)
        if row is None:
            continue  # authored for a node that is not a V1 row of this run
        term_rows.append(row)
        term_drivers.append(driver)
        term_charts.append(chart)
        term_loadings.append(loading)
    for row, rf in enumerate(v1_axis.tolist()):
        if int(rf) not in authored_targets:
            term_rows.append(row)
            term_drivers.append(int(rf))
            term_charts.append(int(shift_ids[rf]))
            term_loadings.append(1.0)

    canonical = canonicalise_resolved_terms(
        num_rows=int(v1_axis.shape[0]),
        term_rows=np.asarray(term_rows, dtype=np.int64),
        term_driver_rf_ids=np.asarray(term_drivers, dtype=np.int32),
        term_chart_ids=np.asarray(term_charts, dtype=np.int8),
        term_loadings=np.asarray(term_loadings, dtype=np.float64),
    )
    if canonical.non_finite_rows.size:
        names = _rel_names(static_config, v1_axis[canonical.non_finite_rows])
        raise ConfigurationError(
            "Merged proxy loadings overflowed to a non-finite value for target(s) "
            f"{names}; D088 requires finite loadings."
        )
    if canonical.empty_rows.size:
        names = _rel_names(static_config, v1_axis[canonical.empty_rows])
        raise ConfigurationError(
            f"Proxy model row(s) for target(s) {names} have no term after "
            "canonicalisation (zero loadings, exact cancellation, or container "
            "pillars with no EXACT driver match). An authored target never falls "
            "back to its identity row (D088 §6 step 7, §9)."
        )

    return CompiledProxyModel(
        row_target_rf_ids=np.ascontiguousarray(v1_axis, dtype=np.int32),
        row_term_offsets=canonical.row_term_offsets,
        term_driver_rf_ids=canonical.term_driver_rf_ids,
        term_chart_ids=canonical.term_chart_ids,
        term_loadings=canonical.term_loadings,
    )


# --------------------------------------------------------------------------- #
# Input structure                                                              #
# --------------------------------------------------------------------------- #


def _require_as_of(as_of_date: object) -> None:
    if isinstance(as_of_date, datetime) or not isinstance(as_of_date, date):
        raise TypeError(
            f"as_of_date must be a datetime.date, got {type(as_of_date).__name__}."
        )


def _require_run_axis(
    run_axis_rf_ids: object, static_config: RiskStaticConfig
) -> npt.NDArray[np.int32]:
    if not isinstance(run_axis_rf_ids, np.ndarray):
        raise TypeError(
            "run_axis_rf_ids must be a numpy.ndarray, "
            f"got {type(run_axis_rf_ids).__name__}."
        )
    if run_axis_rf_ids.dtype != np.int32 or run_axis_rf_ids.ndim != 1:
        raise TypeError(
            "run_axis_rf_ids must be a 1-D int32 array, got "
            f"dtype={run_axis_rf_ids.dtype}, shape={run_axis_rf_ids.shape}."
        )
    axis = np.ascontiguousarray(run_axis_rf_ids)
    if np.unique(axis).shape[0] != axis.shape[0]:
        raise ValueError("run_axis_rf_ids must be unique.")
    max_rf_id = static_config.max_rf_id
    if axis.size and (int(axis.min()) < 0 or int(axis.max()) > max_rf_id):
        raise ConfigurationError(
            f"run_axis_rf_ids contains rf_ids outside [0, {max_rf_id}]."
        )
    topo = static_config.rf_id_to_parent_topology_type
    unoccupied = axis[topo[axis] == _UNOCCUPIED_TOPOLOGY]
    if unoccupied.size:
        raise ConfigurationError(
            f"run_axis_rf_ids contains unoccupied rf_ids {unoccupied.tolist()}."
        )
    return axis


# --------------------------------------------------------------------------- #
# Term validation and resolution                                               #
# --------------------------------------------------------------------------- #


def _require_name(idx: int, field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(
            f"authored_terms[{idx}].{field} must be a non-empty string, got {value!r}."
        )
    return value


def _require_driver_chart(idx: int, chart: object) -> int:
    if not isinstance(chart, ShiftType):
        raise ConfigurationError(
            f"authored_terms[{idx}].driver_chart must be a ShiftType, got {chart!r}."
        )
    if int(chart) not in V1_DRIVER_CHART_IDS:
        raise ConfigurationError(
            f"authored_terms[{idx}].driver_chart={chart.name} is outside the V1 "
            "driver-chart catalogue {LOG_RETURN, SIMPLE_RETURN, DIFFERENCE} "
            "(D088 §1)."
        )
    return int(chart)


def _require_loading(idx: int, loading: object) -> float:
    # bool is not a float subclass, so True/False are rejected here; ints are
    # rejected too: a loading is an authored binary64 value, never coerced.
    if not isinstance(loading, float):
        raise ConfigurationError(
            f"authored_terms[{idx}].loading must be a float, "
            f"got {type(loading).__name__}."
        )
    if not math.isfinite(loading):
        raise ConfigurationError(
            f"authored_terms[{idx}].loading must be finite, got {loading!r}."
        )
    return float(loading)


def _resolve_rf_id(
    idx: int, field: str, name: str, name_map: RiskFactorAbsToRelNameIdMap
) -> int:
    try:
        return int(name_map.get_rf_id(name))
    except KeyError:
        raise ConfigurationError(
            f"authored_terms[{idx}].{field}={name!r} does not resolve to a "
            f"factor-node at as-of {name_map.as_of_date.isoformat()}."
        ) from None


def _require_v1_target_chart(idx: int, what: str, shift_id: int) -> None:
    if shift_id not in V1_TARGET_CHART_IDS:
        raise ConfigurationError(
            f"authored_terms[{idx}]: {what} has configured shift type "
            f"{ShiftType(shift_id).name}, outside the V1 target universe "
            "{LOG_RETURN, SIMPLE_RETURN, DIFFERENCE}; ABSOLUTE_BASE_RETURN and "
            "NO_SHIFT nodes keep their D025 path and cannot be proxy targets "
            "(D088 §1)."
        )


def _resolve_node_term(
    idx: int,
    term: NodeProxyTerm,
    static_config: RiskStaticConfig,
    name_map: RiskFactorAbsToRelNameIdMap,
) -> _ResolvedTerm:
    target_name = _require_name(idx, "target_name", term.target_name)
    driver_name = _require_name(idx, "driver_name", term.driver_name)
    chart = _require_driver_chart(idx, term.driver_chart)
    loading = _require_loading(idx, term.loading)
    target = _resolve_rf_id(idx, "target_name", target_name, name_map)
    driver = _resolve_rf_id(idx, "driver_name", driver_name, name_map)
    _require_v1_target_chart(
        idx,
        f"target {target_name!r}",
        int(static_config.rf_id_to_shift_type_id[target]),
    )
    return (idx, target, driver, chart, loading)


# --------------------------------------------------------------------------- #
# Container shorthand: EXACT expansion at the as-of                            #
# --------------------------------------------------------------------------- #


def _container_meta(
    idx: int, field: str, name: str, static_config: RiskStaticConfig
) -> tuple[int, int, str, tuple[str, ...]]:
    """Return (container_id, topology, shift_type name, coordinate kind)."""
    try:
        container_id = int(static_config.container_name_to_id[name])
    except KeyError:
        raise ConfigurationError(
            f"authored_terms[{idx}].{field}={name!r} is not a curve or surface."
        ) from None
    if container_id in static_config.curve_meta_dict:
        cm = static_config.curve_meta_dict[container_id]
        return container_id, _CURVE, cm.shift_type, (cm.x_axis_kind,)
    if container_id in static_config.surface_meta_dict:
        sm = static_config.surface_meta_dict[container_id]
        return container_id, _SURFACE, sm.shift_type, (sm.x_axis_kind, sm.y_axis_kind)
    raise ConfigurationError(
        f"authored_terms[{idx}].{field}={name!r} is not a curve or surface."
    )


def _pillar_coordinates(
    container_id: int,
    topology: int,
    container_name: str,
    static_config: RiskStaticConfig,
    name_map: RiskFactorAbsToRelNameIdMap,
) -> dict[int, str]:
    """
    Resolved economic coordinate of every static pillar of one container.

    The coordinate is the absolute-name remainder after ``"<container>_"`` from
    the as-of name map: a month ``YYYYMM``, a year ``YYYY``, a half-year
    ``YYYY_FH``/``YYYY_BH``, a date ``YYYYMMDD`` or a bare label; for surfaces
    the ``[y_label]`` suffix is part of it. Tokens are never compared.
    """
    rel_to_abs = name_map.rel_name_to_abs_name
    prefix = f"{container_name}_"
    out: dict[int, str] = {}
    if topology == _CURVE:
        for pt in static_config.curve_point_meta_dict.values():
            if pt.curve_id == container_id:
                abs_name = rel_to_abs[f"{container_name}{pt.tenor_token}"]
                out[int(pt.rf_id)] = _strip_prefix(abs_name, prefix)
    else:
        for spt in static_config.surface_point_meta_dict.values():
            if spt.surface_id == container_id:
                rel = f"{container_name}{spt.x_tenor_token}[{spt.y_label}]"
                out[int(spt.rf_id)] = _strip_prefix(rel_to_abs[rel], prefix)
    return out


def _strip_prefix(abs_name: str, prefix: str) -> str:
    if not abs_name.startswith(prefix):
        raise ConfigurationError(
            f"absolute name {abs_name!r} does not start with {prefix!r}; "
            "cannot derive its resolved coordinate."
        )
    return abs_name[len(prefix) :]


def _expand_container_term(
    idx: int,
    term: ContainerProxyTerm,
    static_config: RiskStaticConfig,
    name_map: RiskFactorAbsToRelNameIdMap,
    row_of_target: dict[int, int],
) -> tuple[set[int], list[_ResolvedTerm]]:
    """
    Expand one container term by EXACT coordinate equality.

    Returns the V1 axis pillars the term covers (they count as authored) and
    the node-level terms for those pillars that have an EXACT driver match. A
    covered pillar without a match contributes no term; if nothing else is
    authored for it, its row is empty and compilation fails (no fallback).
    """
    target_name = _require_name(
        idx, "target_container_name", term.target_container_name
    )
    driver_name = _require_name(
        idx, "driver_container_name", term.driver_container_name
    )
    chart = _require_driver_chart(idx, term.driver_chart)
    loading = _require_loading(idx, term.loading)

    t_id, t_topo, t_shift, t_kind = _container_meta(
        idx, "target_container_name", target_name, static_config
    )
    d_id, d_topo, _, d_kind = _container_meta(
        idx, "driver_container_name", driver_name, static_config
    )
    _require_v1_target_chart(
        idx,
        f"target container {target_name!r}",
        SHIFT_TYPE_NAME_TO_ID_REGISTRY[t_shift],
    )
    if t_topo != d_topo or t_kind != d_kind:
        raise ConfigurationError(
            f"authored_terms[{idx}]: EXACT mapping {target_name!r} <- "
            f"{driver_name!r} compares coordinate kinds {t_kind} and {d_kind}; "
            "EXACT requires equal coordinate kinds, so no pillar can ever match "
            "(D088 §8). Author node-level (EXPLICIT) terms instead."
        )

    target_coords = _pillar_coordinates(
        t_id, t_topo, target_name, static_config, name_map
    )
    driver_coords = _pillar_coordinates(
        d_id, d_topo, driver_name, static_config, name_map
    )
    driver_by_coord: dict[str, int] = {}
    for rf, coord in driver_coords.items():
        if coord in driver_by_coord:
            raise ConfigurationError(
                f"driver container {driver_name!r} has two pillars with resolved "
                f"coordinate {coord!r} at as-of {name_map.as_of_date.isoformat()}."
            )
        driver_by_coord[coord] = rf

    covered: set[int] = set()
    expanded: list[_ResolvedTerm] = []
    for target_rf in sorted(target_coords):
        if target_rf not in row_of_target:
            continue  # not a V1 row of this run
        covered.add(target_rf)
        driver_rf = driver_by_coord.get(target_coords[target_rf])
        if driver_rf is not None:
            expanded.append((idx, target_rf, driver_rf, chart, loading))
    return covered, expanded


def _rel_names(
    static_config: RiskStaticConfig, rf_ids: npt.NDArray[np.int32]
) -> list[str]:
    id_to_name = {int(v): k for k, v in static_config.rel_name_to_rf_id.items()}
    return [id_to_name.get(int(rf), str(int(rf))) for rf in rf_ids.tolist()]
