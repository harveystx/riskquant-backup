# scripts/demos/config/proxy_config/demo_full.py
"""
Demo of proxy config generation. This is the terse reference version of
the pipeline (no narration); see demo_walkthrough.py in this same
directory for a verbose, section-by-section explanation.
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
from riskquant.utils.printer import print_calling_file, print_success_message


def main():
    print_calling_file()

    ###################
    # Static Config
    ###################

    # CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS is dict[str, polars.DataFrame]
    # (riskquant.config.static_config is Polars-based; see RQ-AUD-018).
    for sheet_name, df in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.items():
        print(f"Sheet: {sheet_name}  (rows={df.height})")
        print(df)

    rsc = compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )

    rsc.summary()

    #################
    # Proxy Config
    #################

    # CANONICAL_RISK_PROXY_CONFIG_DF is a polars.DataFrame (same reasoning
    # as above; riskquant.config.proxy_config is Polars-based too).
    print("Canonical Risk Proxy Config:")
    print(CANONICAL_RISK_PROXY_CONFIG_DF)

    proxy_config = compile_proxy_topology_from_df(
        df=CANONICAL_RISK_PROXY_CONFIG_DF,
        static_config=rsc,
    )

    proxy_config.summary()

    print_success_message()


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
