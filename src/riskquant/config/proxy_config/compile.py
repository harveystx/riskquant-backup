# src/riskquant/config/proxy_config/compile.py
"""
Single entry-point for building a ProxyTopology from source data.

Pipeline:
    xlsx  ->  load  ->  validate_df  ->  partition  ->  validate_maps  ->  build  ->  ProxyTopology

The compile step is the only place the raw DataFrame is consumed. After
this layer no caller reads strings, levels, or vol-scaling factors as
named columns: the runtime container exposes only dense int arrays.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.proxy_config.schema_objects import ProxyMap
from riskquant.config.proxy_config.validate import (
    validate_proxy_config_df,
    validate_proxy_maps,
)
from riskquant.config.static_config.static_config import RiskStaticConfig

# ============================================================
# Internal helpers
# ============================================================


def _build_proxy_map_dict(level_df: pl.DataFrame) -> dict[str, ProxyMap]:
    """
    Vectorized: extract typed columns from the level-partitioned df and
    materialize one ProxyMap per row. No row-by-row iteration.

    `level_df` has been validated upstream — required_name uniqueness
    per level is guaranteed, so the dict keyed by required_name has no
    collisions.
    """
    if level_df.is_empty():
        return {}

    required_names = (
        level_df["required_name"].cast(pl.Utf8).str.strip_chars().to_numpy()
    )
    proxy_names = level_df["proxy_name"].cast(pl.Utf8).str.strip_chars().to_numpy()
    vol_scales = level_df["vol_scaling_factor"].cast(pl.Float64).to_numpy()

    return {
        str(req): ProxyMap(
            required_name=str(req),
            proxy_name=str(prx),
            vol_scaling_factor=float(scl),
        )
        for req, prx, scl in zip(required_names, proxy_names, vol_scales, strict=True)
    }


# ============================================================
# Public API
# ============================================================


def compile_proxy_topology_from_df(
    df: pl.DataFrame,
    static_config: RiskStaticConfig,
) -> ProxyTopology:
    """
    Validate and compile an active-filtered proxy DataFrame into a
    ProxyTopology.

    Parameters
    ----------
    df
        Active-filtered proxy DataFrame from the loader. Must have
        columns: proxying_level, required_name, proxy_name,
        vol_scaling_factor.
    static_config
        Compiled static config used to resolve names to ids and to
        validate cross-level conflicts via the static config's
        parent-kind / parent-id arrays.

    Returns
    -------
    ProxyTopology
        Fully compiled, dataframe-free runtime topology with dense int32
        lookup arrays already materialized.

    Raises
    ------
    ValueError
        On any validation failure — Stage 1 (DataFrame schema/type/range),
        Stage 2 (FK / chain / cross-level conflict).
    """
    # ------------------------------------------------------------------ #
    # Stage 1: DataFrame-tier validation                                 #
    # ------------------------------------------------------------------ #
    validate_proxy_config_df(df)

    if df.is_empty():
        # Build a no-op topology: all lookups pass through unchanged.
        return ProxyTopology.build(
            container_proxies={},
            rf_proxies={},
            static_config=static_config,
        )

    # ------------------------------------------------------------------ #
    # Partition by level (vectorized; no row loop)                       #
    # ------------------------------------------------------------------ #
    level_norm = df["proxying_level"].cast(pl.Utf8).str.to_uppercase()
    container_df = df.filter(level_norm == "CONTAINER")
    rf_df = df.filter(level_norm == "RISK_FACTOR")

    container_proxies = _build_proxy_map_dict(container_df)
    rf_proxies = _build_proxy_map_dict(rf_df)

    # ------------------------------------------------------------------ #
    # Stage 2: cross-table validation                                    #
    # ------------------------------------------------------------------ #
    validate_proxy_maps(
        container_proxies=container_proxies,
        rf_proxies=rf_proxies,
        static_config=static_config,
    )

    # ------------------------------------------------------------------ #
    # Assemble: build dense arrays inside ProxyTopology.build            #
    # ------------------------------------------------------------------ #
    return ProxyTopology.build(
        container_proxies=container_proxies,
        rf_proxies=rf_proxies,
        static_config=static_config,
    )


def compile_proxy_topology_from_xlsx(
    path: str | Path,
    static_config: RiskStaticConfig,
    sheet_name: str | None = "proxy_config",
) -> ProxyTopology:
    """
    Load, validate, and compile a proxy-config workbook into a ProxyTopology.

    Parameters
    ----------
    path
        Path to the Excel workbook.
    static_config
        Compiled static config for name validation and ID resolution.
    sheet_name
        Sheet to read. Defaults to "proxy_config".
    """
    df = load_proxy_config_xlsx_into_df(path=path, sheet_name=sheet_name)
    return compile_proxy_topology_from_df(df=df, static_config=static_config)


if __name__ == "__main__":
    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_xlsx,
    )
    from riskquant.utils.printer import print_calling_file, print_success_message

    print_calling_file()

    DEV_DATA_DIR: str = "C:/Users/mharvey/Documents/code/riskquant/data/dev"

    rsc = compile_risk_static_config_from_xlsx(
        path=f"{DEV_DATA_DIR}/dev_risk_static_data_config_v01.xlsx"
    )

    topology = compile_proxy_topology_from_xlsx(
        path=f"{DEV_DATA_DIR}/dev_risk_base_proxy_config_v01.xlsx",
        static_config=rsc,
    )

    topology.summary()
    print_success_message()
