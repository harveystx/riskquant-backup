# src/riskquant/config/proxy_config/compile.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.config.proxy_config.proxy_config import ProxyTopology
from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.config.proxy_config.validate import (
    validate_proxy_config_df,
)
from riskquant.config.proxy_config.builder import build_proxy_topology_from_df


def compile_proxy_topology_from_xlsx(
    path: Union[str, Path],
    static_config: RiskStaticConfig,
    sheet_name: Optional[str] = "proxy_config",
) -> ProxyTopology:

    # 1. Load the active proxy config rows into a DataFrame.
    proxy_df = load_proxy_config_xlsx_into_df(path=path, sheet_name=sheet_name)

    # 2. Vectorized fail-fast validation of the raw DataFrame schema and logic.
    validate_proxy_config_df(proxy_df)

    # 3. Build the topology, validating against the static config and compiling C-level int maps.
    topology = build_proxy_topology_from_df(
        df=proxy_df,
        static_config=static_config,
    )

    # Return the compiled topology object, ready for use in the system.
    return topology
