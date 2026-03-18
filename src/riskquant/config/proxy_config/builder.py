# src/riskquant/config/proxy_config/builder.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.config.proxy_config.schema_objects import ProxyMap
from riskquant.config.proxy_config.proxy_config import ProxyTopology
from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.exceptions import PrettyKeyError


def build_proxy_topology_from_df(
    df: pd.DataFrame,
    static_config: RiskStaticConfig,
) -> ProxyTopology:
    """
    Parses a unified proxy mapping dataframe into a ProxyTopology,
    using RiskStaticConfig to validate names and look up integer IDs.
    """
    container_proxies: Dict[str, ProxyMap] = {}
    rf_proxies: Dict[str, ProxyMap] = {}

    # Iterate and route to the correct dictionary
    # (Note: df is assumed to be pre-filtered for active rows by the loader)
    for _, row in df.iterrows():
        level = str(row["proxying_level"]).strip().lower()
        req_name = str(row["required_name"]).strip()
        proxy_name = str(row["proxy_name"]).strip()

        # Safely handle the optional scaling factor
        vol_scale = row.get("vol_scaling_factor")
        vol_scale = (
            1.0 if pd.isna(vol_scale) or vol_scale is None else float(vol_scale)
        )

        if level == "container":
            # Validate against static config
            if req_name not in static_config.container_name_to_container_id:
                raise PrettyKeyError(
                    f"Proxy Config Error: required_name '{req_name}' not found in static container metadata."
                )
            if proxy_name not in static_config.container_name_to_container_id:
                raise PrettyKeyError(
                    f"Proxy Config Error: proxy_name '{proxy_name}' not found in static container metadata."
                )

            container_proxies[req_name] = ProxyMap(
                proxying_level="container",
                required_name=req_name,
                proxy_name=proxy_name,
                vol_scaling_factor=vol_scale,
            )

        elif level == "risk_factor":
            # Validate against static config
            if req_name not in static_config.rf_name_to_rf_id:
                raise PrettyKeyError(
                    f"Proxy Config Error: required_name '{req_name}' not found in static risk factors."
                )
            if proxy_name not in static_config.rf_name_to_rf_id:
                raise PrettyKeyError(
                    f"Proxy Config Error: proxy_name '{proxy_name}' not found in static risk factors."
                )

            rf_proxies[req_name] = ProxyMap(
                proxying_level="risk_factor",
                required_name=req_name,
                proxy_name=proxy_name,
                vol_scaling_factor=vol_scale,
            )

        else:
            raise ValueError(
                f"Unknown proxying_level '{level}' for '{req_name}'. "
                "Must be strictly 'container' or 'risk_factor'."
            )

    # Construct the topology (which pre-compiles the C-level int maps)
    return ProxyTopology.build(
        container_proxies=container_proxies,
        rf_proxies=rf_proxies,
        container_name_to_container_id=static_config.container_name_to_container_id,
        rf_name_to_rf_id=static_config.rf_name_to_rf_id,
    )


# def build_proxy_topology_from_xlsx(
#     path: Union[str, Path],
#     static_config: RiskStaticConfig,
#     sheet_name: str = "proxy_config",
# ) -> ProxyTopology:
#     """
#     Convenience loader to read the proxy Excel sheet and build the topology.
#     """
#     df = load_proxy_config_xlsx_into_df(path=path, sheet_name=sheet_name)
#     return build_proxy_topology_from_df(df, static_config)
