# src/riskquant/proxy_resolver/proxy_audit.py
"""
Audit a compiled ProxyResolution.

Imports Pandas deliberately — auditing is an edge-only diagnostic, never
imported by the kernel. Builds a human-readable DataFrame by joining the
ProxyResolution's int32 arrays against name-bearing structures on
RiskStaticConfig and ProxyTopology.

The whole DataFrame construction is vectorized: no per-id Python loops.
Name lookups use dict gathers via Python comprehensions, which are
acceptable here because audit is run once per simlab pipeline, not in
the hot path.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.proxy_resolver.proxy import (
    RES_CONTAINER_PROXY,
    RES_DIRECT,
    RES_ID_PROXY,
    RES_NOT_SIMULATED,
    RES_TENOR_FALLBACK,
    ProxyResolution,
)
from riskquant.rq_logging import LogContext, get_logger, with_context

# ============================================================
# Resolution-code labels
# ============================================================


_RES_CODE_TO_LABEL: dict[int, str] = {
    int(RES_NOT_SIMULATED): "NOT_SIMULATED",
    int(RES_DIRECT): "DIRECT",
    int(RES_TENOR_FALLBACK): "TENOR_FALLBACK",
    int(RES_ID_PROXY): "RF_PROXY",
    int(RES_CONTAINER_PROXY): "CONTAINER_PROXY",
}


_PARENT_KIND_TO_LABEL: dict[int, str] = {
    -1: "UNOCCUPIED",
    0: "NODE",
    1: "CURVE_POINT",
    2: "SURFACE_POINT",
}


# ============================================================
# Internal: vectorized name reconstruction from rf_id
# ============================================================


def _rf_ids_to_relative_names(
    rf_ids: np.ndarray,
    static_config: RiskStaticConfig,
) -> list[str]:
    """
    For each rf_id, return its relative name.

    Conventions:
      NODE           : node_name (e.g. "USDEUR")
      CURVE_POINT    : f"{curve_name}{tenor_token}"     (e.g. "BRENT_CRUDE_FWD[M+0]")
      SURFACE_POINT  : f"{surface_name}{x_token}[{y}]"  (e.g. "BRENT_VOL_SURF[M+0][50.00]")

    Uses the same convention as static_config._build_name_lookups.
    """
    parent_kind = static_config.rf_id_to_parent_topology_type
    nm = static_config.node_meta_dict
    cpm = static_config.curve_point_meta_dict
    spm = static_config.surface_point_meta_dict

    out: list[str] = []
    for rf in rf_ids:
        rid = int(rf)
        pk = int(parent_kind[rid])
        if pk == 0:
            out.append(nm[rid].node_name)
        elif pk == 1:
            pt = cpm[rid]
            out.append(f"{pt.curve_name}{pt.tenor_token}")
        elif pk == 2:
            pt = spm[rid]
            out.append(f"{pt.surface_name}{pt.x_tenor_token}[{pt.y_label}]")
        else:
            out.append(f"<unoccupied_rf_id={rid}>")
    return out


def _rf_ids_to_container_names(
    rf_ids: np.ndarray,
    static_config: RiskStaticConfig,
) -> list[str]:
    """
    For each rf_id, return its parent container name. Nodes have no
    container; emit the node_name verbatim so the column has a
    meaningful value everywhere.
    """
    parent_kind = static_config.rf_id_to_parent_topology_type
    nm = static_config.node_meta_dict
    cm = static_config.curve_meta_dict
    sm = static_config.surface_meta_dict
    cpm = static_config.curve_point_meta_dict
    spm = static_config.surface_point_meta_dict

    out: list[str] = []
    for rf in rf_ids:
        rid = int(rf)
        pk = int(parent_kind[rid])
        if pk == 0:
            out.append(nm[rid].node_name)
        elif pk == 1:
            out.append(cm[cpm[rid].curve_id].curve_name)
        elif pk == 2:
            out.append(sm[spm[rid].surface_id].surface_name)
        else:
            out.append(f"<unoccupied_rf_id={rid}>")
    return out


def _rf_ids_to_tenor_tokens(
    rf_ids: np.ndarray,
    static_config: RiskStaticConfig,
) -> list[str]:
    """
    For each rf_id, return its tenor descriptor:
      NODE          -> ""
      CURVE_POINT   -> tenor_token (e.g. "[M+0]")
      SURFACE_POINT -> f"{x_tenor_token}[{y_label}]" (e.g. "[M+0][50.00]")
    """
    parent_kind = static_config.rf_id_to_parent_topology_type
    cpm = static_config.curve_point_meta_dict
    spm = static_config.surface_point_meta_dict

    out: list[str] = []
    for rf in rf_ids:
        rid = int(rf)
        pk = int(parent_kind[rid])
        if pk == 1:
            out.append(cpm[rid].tenor_token)
        elif pk == 2:
            pt = spm[rid]
            out.append(f"{pt.x_tenor_token}[{pt.y_label}]")
        else:
            out.append("")
    return out


# ============================================================
# Audit DataFrame construction
# ============================================================


def _smoke_test_proxy_resolution(pr: ProxyResolution) -> None:
    """Dtype guard for downstream consumers."""
    if pr.required_factor_ids.dtype != np.int32:
        raise TypeError("proxy_resolution.required_factor_ids must be int32")
    if pr.source_factor_ids.dtype != np.int32:
        raise TypeError("proxy_resolution.source_factor_ids must be int32")
    if pr.resolution_code.dtype != np.int8:
        raise TypeError("proxy_resolution.resolution_code must be int8")
    if pr.shock_scale.dtype != np.float32:
        raise TypeError("proxy_resolution.shock_scale must be float32")


def build_proxy_resolution_audit_df(
    *,
    static_config: RiskStaticConfig,
    proxy_topology: ProxyTopology,
    proxy_resolution: ProxyResolution,
) -> pd.DataFrame:
    """
    Build a fully name-resolved DataFrame view of a ProxyResolution.

    Columns
    -------
        required_factor_id          int32
        required_parent_kind        str   (NODE / CURVE_POINT / SURFACE_POINT)
        required_container_name     str
        required_tenor              str   ("[M+0]" or "[M+0][50.00]" or "")
        required_risk_factor_name   str   (relative; e.g. "BRENT_CRUDE_FWD[M+0]")
        source_factor_id            int32
        source_parent_kind          str
        source_container_name       str
        source_tenor                str
        source_risk_factor_name     str
        resolution_code             str
        shock_scale                 float32
        rf_proxy_target_name        str   (populated when code==RF_PROXY)
        container_proxy_target_name str   (populated when code==CONTAINER_PROXY)
    """
    _smoke_test_proxy_resolution(proxy_resolution)

    n = int(proxy_resolution.required_factor_ids.shape[0])
    if n == 0:
        return pd.DataFrame(
            columns=[
                "required_factor_id",
                "required_parent_kind",
                "required_container_name",
                "required_tenor",
                "required_risk_factor_name",
                "source_factor_id",
                "source_parent_kind",
                "source_container_name",
                "source_tenor",
                "source_risk_factor_name",
                "resolution_code",
                "shock_scale",
                "rf_proxy_target_name",
                "container_proxy_target_name",
            ]
        )

    req = proxy_resolution.required_factor_ids
    src = proxy_resolution.source_factor_ids
    code_arr = proxy_resolution.resolution_code

    # --- name / kind / tenor projections ---
    req_parent_kind = static_config.rf_id_to_parent_topology_type[req]
    src_parent_kind = static_config.rf_id_to_parent_topology_type[src]

    req_parent_kind_lbl = [_PARENT_KIND_TO_LABEL[int(k)] for k in req_parent_kind]
    src_parent_kind_lbl = [_PARENT_KIND_TO_LABEL[int(k)] for k in src_parent_kind]

    req_container = _rf_ids_to_container_names(req, static_config)
    src_container = _rf_ids_to_container_names(src, static_config)

    req_tenor = _rf_ids_to_tenor_tokens(req, static_config)
    src_tenor = _rf_ids_to_tenor_tokens(src, static_config)

    req_name = _rf_ids_to_relative_names(req, static_config)
    src_name = _rf_ids_to_relative_names(src, static_config)

    code_lbl = [_RES_CODE_TO_LABEL.get(int(c), f"UNKNOWN_{int(c)}") for c in code_arr]

    # --- proxy-target attribution ---
    # For RF_PROXY: target is the rf-level proxy name configured in
    # proxy_topology. We derive it from rf_id_to_proxy_rf_id rather than
    # the source field directly, to distinguish "directly proxied" from
    # "tenor-fallback to the proxy curve via container."
    rf_proxy_target = [""] * n
    container_proxy_target = [""] * n

    # rf-level proxy targets (only for RF_PROXY rows).
    is_rf_proxy = code_arr == RES_ID_PROXY
    if is_rf_proxy.any():
        rf_proxy_ids = proxy_topology.rf_id_to_proxy_rf_id[req[is_rf_proxy]]
        rf_proxy_names = _rf_ids_to_relative_names(rf_proxy_ids, static_config)
        idx_positions = np.flatnonzero(is_rf_proxy)
        for pos, name in zip(idx_positions, rf_proxy_names, strict=True):
            rf_proxy_target[pos] = name

    # container-level proxy targets (only for CONTAINER_PROXY rows).
    is_container_proxy = code_arr == RES_CONTAINER_PROXY
    if is_container_proxy.any():
        req_subset = req[is_container_proxy]
        # parent_id for these is the required's container_id (curve or surface).
        req_parent_id = static_config.rf_id_to_parent_id[req_subset].astype(
            np.int32, copy=False
        )
        slots = static_config.container_slots(req_parent_id)
        proxy_container_ids = proxy_topology.container_proxy_id[slots]
        proxy_container_names = _container_ids_to_names(
            proxy_container_ids, static_config
        )
        idx_positions = np.flatnonzero(is_container_proxy)
        for pos, name in zip(idx_positions, proxy_container_names, strict=True):
            container_proxy_target[pos] = name

    return pd.DataFrame(
        {
            "required_factor_id": req,
            "required_parent_kind": req_parent_kind_lbl,
            "required_container_name": req_container,
            "required_tenor": req_tenor,
            "required_risk_factor_name": req_name,
            "source_factor_id": src,
            "source_parent_kind": src_parent_kind_lbl,
            "source_container_name": src_container,
            "source_tenor": src_tenor,
            "source_risk_factor_name": src_name,
            "resolution_code": code_lbl,
            "shock_scale": proxy_resolution.shock_scale,
            "rf_proxy_target_name": rf_proxy_target,
            "container_proxy_target_name": container_proxy_target,
        }
    )


def _container_ids_to_names(
    container_ids: np.ndarray,
    static_config: RiskStaticConfig,
) -> list[str]:
    """Resolve container_ids to their (curve_name | surface_name)."""
    kind_arr = static_config.container_topology_type
    cm = static_config.curve_meta_dict
    sm = static_config.surface_meta_dict

    out: list[str] = []
    for cid in container_ids:
        cid_int = int(cid)
        if cid_int < 0:
            out.append("<no_proxy>")
            continue
        k = int(kind_arr[static_config.container_slot(cid_int)])
        if k == 1:
            out.append(cm[cid_int].curve_name)
        elif k == 2:
            out.append(sm[cid_int].surface_name)
        else:
            out.append(f"<unoccupied_container_id={cid_int}>")
    return out


# ============================================================
# Optional: lazy handle for repeated audits / logging
# ============================================================


@dataclass(slots=True)
class ProxyResolutionAudit:
    """
    Lazy, cached handle for producing and logging a ProxyResolution audit.

    Built once; the DataFrame is materialized on first access and reused.
    Useful when an audit is consumed by multiple downstream sinks
    (logging, CSV export, NDJSON, screen dump).
    """

    static_config: RiskStaticConfig
    proxy_topology: ProxyTopology
    proxy_resolution: ProxyResolution
    name: str = "proxy_resolution_audit"
    component: str = "simlab.resolve.proxy_audit"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    _df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @property
    def df(self) -> pd.DataFrame:
        return self.to_frame()

    def to_frame(self) -> pd.DataFrame:
        if self._df is None:
            self._df = build_proxy_resolution_audit_df(
                static_config=self.static_config,
                proxy_topology=self.proxy_topology,
                proxy_resolution=self.proxy_resolution,
            )
        return self._df

    def to_csv(self, path: str, *, index: bool = False, **kwargs) -> str:
        self.to_frame().to_csv(path, index=index, **kwargs)
        return path

    # ----- logging adapters -----

    def _logger(
        self,
        logger: logging.Logger | logging.LoggerAdapter | None,
    ) -> logging.LoggerAdapter:
        if isinstance(logger, logging.LoggerAdapter):
            base: logging.Logger | None = logger.logger
        else:
            base = logger
        if base is None:
            base = get_logger(__name__)

        ctx = LogContext(
            component=self.component,
            object_name=self.name,
            object_id=self.run_id,
        )
        return with_context(base, ctx)

    def log_summary(
        self,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Headline summary: row count + resolution-code counts."""
        log = self._logger(logger)
        df = self.to_frame()
        counts = df["resolution_code"].value_counts(dropna=False)
        counts_str = ", ".join(f"{k}={int(v)}" for k, v in counts.items())
        log.log(
            level,
            "audit_summary rows=%d resolution_code_counts=[%s]",
            int(df.shape[0]),
            counts_str,
        )

    def log_rows_ndjson(
        self,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
        level: int = logging.INFO,
        event: str = "proxy_resolution_audit_row",
        columns: Iterable[str] | None = None,
    ) -> None:
        """
        Emit the audit table as NDJSON, one log line per row.

        For best ingestion, route to a handler with formatter
        '%(message)s' so each line is pure JSON.
        """
        log = self._logger(logger)
        df = self.to_frame()
        if columns is not None:
            cols = list(columns)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"Requested audit columns not in DataFrame: {missing}")
            df = df.loc[:, cols]

        col_names = list(df.columns)
        audit_name = self.name
        run_id = self.run_id

        for i, row in enumerate(df.itertuples(index=False, name=None)):
            row_dict = {col_names[j]: _jsonable(row[j]) for j in range(len(col_names))}
            payload = {
                "event": event,
                "audit_name": audit_name,
                "run_id": run_id,
                "row_idx": i,
                **row_dict,
            }
            log.log(
                level,
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
            )


def _jsonable(v: Any) -> Any:
    """numpy scalar -> Python; NaN -> None for JSON safety."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v
