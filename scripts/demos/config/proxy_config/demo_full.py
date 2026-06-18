# scripts/demos/config/proxy_config/demo_full.py
"""
Demo of proxy config generation.
"""

from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_proxy_config.canonical_risk_proxy_config import (
    CANONICAL_RISK_PROXY_CONFIG_DF,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_dict_of_dfs,
    print_success_message,
)


def main():
    print_calling_file()

    ###################
    # Static Config
    ###################

    print_dict_of_dfs(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)

    rsc = compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )

    rsc.summary()

    #################
    # Proxy Config
    #################

    print_df(
        CANONICAL_RISK_PROXY_CONFIG_DF,
        title="Canonical Risk Proxy Config",
        float_format=">9_.0f",
    )

    proxy_config = compile_proxy_topology_from_df(
        df=CANONICAL_RISK_PROXY_CONFIG_DF,
        static_config=rsc,
    )

    proxy_config.summary()

    print_success_message()


if __name__ == "__main__":
    main()
