# scripts/demos/config/static_config/demo_full.py
"""
Demo of risk static config generation for canonical full config, with all sources
populated. This is the most complete example of static config generation, and
serves as a reference for the other demo scripts.
"""

from riskquant.config.static_config.audit import audit_risk_static_config
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.utils.printer import (
    print_array,
    print_calling_file,
    print_success_message,
)


def main():
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    print_calling_file()

    # CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS is dict[str, polars.DataFrame]
    # (riskquant.config.static_config is Polars-based; see RQ-AUD-018).
    # print_dict_of_dfs in riskquant.utils.printer is pandas-typed, so each
    # sheet is printed directly via Polars' own clean table formatting.
    for sheet_name, df in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.items():
        print(f"Sheet: {sheet_name}  (rows={df.height})")
        print(df)

    rsc = compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )

    rsc.summary()

    audit_risk_static_config(rsc=rsc, n=200)

    print_array(rsc.fx_spot_matrix, title="FX Spot Matrix", float_format=">9_.0f")

    print_success_message()


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
