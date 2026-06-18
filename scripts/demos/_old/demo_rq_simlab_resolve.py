# scripts/demo_simlab_proxy.py
from pathlib import Path

import numpy as np
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_xlsx,
)
from riskquant.simlab.resolve.proxy import (
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.simlab.resolve.proxy_audit import ProxyResolutionAudit
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_df,
    print_success_message,
)


def run_proxy_demo():

    print_calling_file()

    dev_data_dir = Path("C:/Users/mharvey/Documents/code/riskquant/data/dev")
    xlsx_path = dev_data_dir / "dev_risk_static_data_config_v01.xlsx"

    print("--- Loading Config ---")
    config = compile_risk_static_config_from_xlsx(xlsx_path)

    # Mock Universe A: What the CompiledPortfolio actually needs
    required_ids = np.array([10, 15, 22, 105, 200], dtype=np.int32)

    # Mock Universe B: What history we actually have loaded from Excel
    # Notice that ID 22 is missing, forcing a proxy or fallback!
    market_data_ids = np.array([10, 15, 18, 105, 199, 200, 201], dtype=np.int32)

    print("\n--- Executing Vectorized Proxy Resolution ---")
    resolution = resolve_required_factor_ids_to_market_data_factor_ids(
        cfg=config,
        required_factor_ids=required_ids,
        market_data_factor_ids=market_data_ids,
    )

    print(f"Resolved {len(resolution.required_factor_ids)} required factors.")
    print(
        f"Compressed down to {len(resolution.unique_source_factor_ids)} unique source histories."
    )

    print("\n--- Generating Audit Report ---")
    audit = ProxyResolutionAudit(cfg=config, proxy_resolution=resolution)

    # Display the beautiful Pandas DataFrame showing EXACTLY why each factor mapped
    print_df(audit.df)

    # Example of NDJSON logging for Splunk/Datadog ingestion
    # audit.log_rows_ndjson()

    print_success_message()


if __name__ == "__main__":
    run_proxy_demo()
