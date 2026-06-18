# src/riskquant/config/proxy_config/audit.py
"""
Tabular audit of a compiled ProxyTopology.

Imports Pandas deliberately — auditing is an edge-only diagnostic, not a
runtime path. Never imported by valuation, simlab, or any kernel.

This module also hosts the consolidated name-resolved DataFrame view
(`to_df`) that used to live as a method on ProxyTopology. Keeping it
here ensures the runtime container itself stays Pandas-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.static_config import RiskStaticConfig

_SECTION = "=" * 72
_SUB = "-" * 72


def _header(title: str) -> None:
    print(_SECTION)
    print(title)
    print(_SECTION)


def _sub(title: str) -> None:
    print(f"\n{_SUB}")
    print(title)
    print(_SUB)


# ============================================================
# Name-resolved DataFrame projection
# ============================================================


def to_df(topology: ProxyTopology, static_config: RiskStaticConfig) -> pd.DataFrame:
    """
    Consolidate container and rf proxies into a single name+id DataFrame.

    Mirrors the legacy `ProxyTopology.to_df()` output. Lives here rather
    than on the runtime container so the runtime stays Pandas-free.

    Columns
    -------
        proxying_level, required_name, required_id,
        proxy_name, proxy_id, vol_scaling_factor
    """
    name_to_cid = static_config.container_name_to_id
    name_to_rfid = static_config.rel_name_to_rf_id

    records: list[dict[str, object]] = []

    for pm in topology.container_proxies.values():
        records.append(
            {
                "proxying_level": "CONTAINER",
                "required_name": pm.required_name,
                "required_id": int(name_to_cid[pm.required_name]),
                "proxy_name": pm.proxy_name,
                "proxy_id": int(name_to_cid[pm.proxy_name]),
                "vol_scaling_factor": float(pm.vol_scaling_factor),
            }
        )

    for pm in topology.rf_proxies.values():
        records.append(
            {
                "proxying_level": "RISK_FACTOR",
                "required_name": pm.required_name,
                "required_id": int(name_to_rfid[pm.required_name]),
                "proxy_name": pm.proxy_name,
                "proxy_id": int(name_to_rfid[pm.proxy_name]),
                "vol_scaling_factor": float(pm.vol_scaling_factor),
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "proxying_level",
            "required_name",
            "required_id",
            "proxy_name",
            "proxy_id",
            "vol_scaling_factor",
        ],
    )


# ============================================================
# Section renderers
# ============================================================


def _audit_summary(topology: ProxyTopology) -> None:
    _header("ProxyTopology Audit")
    print(f"  Container proxies   : {topology.num_container_proxies}")
    print(f"  RF proxies          : {topology.num_rf_proxies}")
    print(f"  Total               : {topology.num_total_proxies}")
    print(f"  c_proxy array size  : {topology.container_proxy_id.size}")
    print(f"  rf_proxy array size : {topology.rf_id_to_proxy_rf_id.size}")
    arr = topology.container_proxy_id
    n_set_c = int((arr >= 0).sum())
    arr = topology.rf_id_to_proxy_rf_id
    n_set_rf = int((arr >= 0).sum())
    print(f"  c_proxy slots set   : {n_set_c}")
    print(f"  rf_proxy slots set  : {n_set_rf}")


def _audit_proxy_table(
    topology: ProxyTopology, static_config: RiskStaticConfig, n: int
) -> None:
    _sub("1. Name-resolved proxy table")
    df = to_df(topology, static_config)
    if df.empty:
        print("(no proxies registered)")
        return
    print(df.head(n).to_string(index=False))


def _audit_vol_scale_distribution(topology: ProxyTopology, n: int) -> None:
    _sub("2. vol_scaling_factor distribution")
    scales = [pm.vol_scaling_factor for pm in topology.container_proxies.values()] + [
        pm.vol_scaling_factor for pm in topology.rf_proxies.values()
    ]
    if not scales:
        print("(no proxies)")
        return
    arr = np.asarray(scales, dtype=np.float64)
    print(f"  count = {arr.size}")
    print(f"  min   = {arr.min():.4f}")
    print(f"  max   = {arr.max():.4f}")
    print(f"  mean  = {arr.mean():.4f}")
    print(f"  != 1  = {int((arr != 1.0).sum())} entries differ from identity")


def _audit_proxied_containers_by_kind(
    topology: ProxyTopology, static_config: RiskStaticConfig, n: int
) -> None:
    _sub("3. Container proxies — by container kind (curve / surface)")
    if not topology.container_proxies:
        print("(none)")
        return
    name_to_cid = static_config.container_name_to_id
    kind_arr = static_config.container_topology_type
    rows = []
    for pm in topology.container_proxies.values():
        req_id = int(name_to_cid[pm.required_name])
        prx_id = int(name_to_cid[pm.proxy_name])
        kind_label = {1: "CURVE", 2: "SURFACE"}.get(
            int(kind_arr[static_config.container_slot(req_id)]), "?"
        )
        rows.append(
            {
                "kind": kind_label,
                "required_name": pm.required_name,
                "required_id": req_id,
                "proxy_name": pm.proxy_name,
                "proxy_id": prx_id,
                "vol_scale": float(pm.vol_scaling_factor),
            }
        )
    df = (
        pd.DataFrame(rows).sort_values(["kind", "required_name"]).reset_index(drop=True)
    )
    print(df.head(n).to_string(index=False))


def _audit_proxied_rfs_by_parent(
    topology: ProxyTopology, static_config: RiskStaticConfig, n: int
) -> None:
    _sub("4. RF proxies — by parent kind (node / curve point / surface point)")
    if not topology.rf_proxies:
        print("(none)")
        return
    name_to_rfid = static_config.rel_name_to_rf_id
    parent_kind = static_config.rf_id_to_parent_topology_type
    rows = []
    for pm in topology.rf_proxies.values():
        rfid = int(name_to_rfid[pm.required_name])
        prx_id = int(name_to_rfid[pm.proxy_name])
        kind_label = {0: "NODE", 1: "CURVE_POINT", 2: "SURFACE_POINT"}.get(
            int(parent_kind[rfid]), "?"
        )
        rows.append(
            {
                "parent_kind": kind_label,
                "required_name": pm.required_name,
                "required_id": rfid,
                "proxy_name": pm.proxy_name,
                "proxy_id": prx_id,
                "vol_scale": float(pm.vol_scaling_factor),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values(["parent_kind", "required_name"])
        .reset_index(drop=True)
    )
    print(df.head(n).to_string(index=False))


def _audit_chain_check(topology: ProxyTopology) -> None:
    _sub("5. Chain check — every proxy must point at a terminal name")
    for level_label, proxies in (
        ("container", topology.container_proxies),
        ("risk_factor", topology.rf_proxies),
    ):
        required_set = set(proxies.keys())
        chained = [
            (pm.required_name, pm.proxy_name)
            for pm in proxies.values()
            if pm.proxy_name in required_set
        ]
        if chained:
            # validate_proxy_maps would have already rejected this; this
            # branch only triggers if someone built ProxyTopology directly
            # without calling the validator.
            print(f"  [{level_label}] CHAINED  ({len(chained)} pair(s))")
            for req, prx in chained[:5]:
                print(f"    {req!r} -> {prx!r}  (but {prx!r} is also proxied)")
        else:
            print(f"  [{level_label}] OK")


# ============================================================
# Public API
# ============================================================


def audit_proxy_topology(
    topology: ProxyTopology,
    static_config: RiskStaticConfig,
    n: int = 20,
) -> None:
    """
    Print a tabular audit of every section of the compiled topology.

    Parameters
    ----------
    topology
        Compiled topology to audit.
    static_config
        Source of name <-> id mappings for human-readable rendering.
    n
        Row cap per section. Use a large number for full dumps.
    """
    _audit_summary(topology)
    _audit_proxy_table(topology, static_config, n)
    _audit_vol_scale_distribution(topology, n)
    _audit_proxied_containers_by_kind(topology, static_config, n)
    _audit_proxied_rfs_by_parent(topology, static_config, n)
    _audit_chain_check(topology)
    print("\n" + _SECTION)


if __name__ == "__main__":
    from riskquant.config.proxy_config.compile import (
        compile_proxy_topology_from_xlsx,
    )
    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_xlsx,
    )
    from riskquant.utils.printer import print_calling_file, print_success_message

    print_calling_file()

    DEV_DATA_DIR: str = "C:/Users/mharvey/Documents/code/riskquant/data/canonical_examples"

    rsc = compile_risk_static_config_from_xlsx(
        path=f"{DEV_DATA_DIR}/canonical_risk_static_config.xlsx"
    )
    topology = compile_proxy_topology_from_xlsx(
        path=f"{DEV_DATA_DIR}/canonical_proxy_config.xlsx",
        static_config=rsc,
    )

    topology.summary()
    audit_proxy_topology(topology, rsc, n=200)

    print_success_message()
