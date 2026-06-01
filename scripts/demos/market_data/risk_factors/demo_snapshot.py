# scripts/demos/market_data/risk_factors/demo_snapshot.py


# ============================================================
# CANONICAL INPUTS
# ============================================================

from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_proxy_config.canonical_risk_proxy_config import (
    CANONICAL_RISK_PROXY_CONFIG_DF,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.utils.printer import (
    print_df,
    print_dict,
    print_header,
    print_subheader,
)

_AS_OF_DATE = CANONICAL_AS_OF_DATE
_RISK_STATIC_CONFIG_DICT_OF_DFS = CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
_PROXY_CONFIG_DF = CANONICAL_RISK_PROXY_CONFIG_DF
_RF_LEVELS_SNAPSHOT = CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT


if __name__ == "__main__":
    # ---- COMPILE RISK STATIC CONFIG MASTER ----

    print_header(title="SECTION: COMPILE MASTERS")

    risk_static_config = compile_risk_static_config_from_dict_of_dfs(
        _RISK_STATIC_CONFIG_DICT_OF_DFS
    )
    print_subheader(title="COMPILED RISK STATIC CONFIG OBJECT:")
    risk_static_config.summary()

    print_dict(
        risk_static_config.container_name_to_id,
        title="CONTAINER NAME TO CONTAINER ID MAPPING",
    )

    # ---- COMPILE PROXY CONFIG MASTER ----

    proxy_config = compile_proxy_topology_from_df(
        df=_PROXY_CONFIG_DF,
        static_config=risk_static_config,
    )
    print_subheader(title="COMPILED PROXY CONFIG OBJECT")
    proxy_config.summary()

    # ---- COMPILE RISK FACTORS SNAPSHOT ----

    # RF Levels Snapshot Canonical Input
    print_subheader(title="CANONICAL RISK FACTOR LEVELS SNAPSHOT INPUT")
    print_dict(
        _RF_LEVELS_SNAPSHOT, title="RISK FACTOR LEVELS SNAPSHOT - CANONICAL INPUT"
    )

    # Compile RF Levels Snapshot
    rf_lvl_snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=_AS_OF_DATE,
        levels_dict=_RF_LEVELS_SNAPSHOT,
        static_config=risk_static_config,
    )

    # Display Compiled Snapshot
    print_subheader(title="COMPILED RISK FACTOR LEVELS SNAPSHOT")
    rf_lvl_snapshot.summary()
    print_df(
        rf_lvl_snapshot.df.T,
        index=True,
        title="RISK FACTOR LEVELS SNAPSHOT (COMPILED OUTPUT, TRANSPOSED FOR DISPLAY)",
        float_format=">12_.6f",
    )
