from typing import Final

from riskquant.config.proxy_config.compile import (
    compile_proxy_topology_from_xlsx,
)
from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_xlsx,
)
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_dict,
    print_success_message,
    print_wrapped_text_box,
)

if __name__ == "__main__":

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # --- HEADING ---

    # Print the name of the calling file for clarity in the console output.
    print_calling_file()

    # --- PATHS AND CONFIG NAMES ---

    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    DEV_DATA_FILE_NAME_PROXY_CONFIG: Final[str] = (
        "dev_risk_base_proxy_config_v01.xlsx"
    )

    # --- COMPILE RISK STATIC CONFIG FROM THE STATIC CONFIG XLSX SHEET ---

    # Heading.
    print_wrapped_text_box(
        "COMPILE RISK STATIC CONFIG FROM THE STATIC CONFIG XLSX SHEET"
    )

    # This single call loads, validates, and compiles the Excel file into pure Python objects.
    config = compile_risk_static_config_from_xlsx(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG
    )

    # --- LOAD THE PROXY CONFIG XLSX SHEET INTO A DATAFRAME ---

    # Heading.
    print_wrapped_text_box("LOAD THE PROXY CONFIG XLSX SHEET INTO A DATAFRAME")

    # Load the config sheets into a dictionary of DataFrames.
    df = load_proxy_config_xlsx_into_df(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_PROXY_CONFIG
    )

    # Print.
    print_df(df)

    # --- COMPILE PROXY CONFIG INTO PROXY TOPOLOGY ---

    # Heading.
    print_wrapped_text_box("COMPILE PROXY CONFIG INTO PROXY TOPOLOGY")

    # Compile the proxy config into a ProxyTopology object, which includes pre-compiled mappings for efficient kernel access.
    proxy_topology = compile_proxy_topology_from_xlsx(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_PROXY_CONFIG,
        static_config=config,
    )

    # Audit prints.

    print("\n   - Dicts of Proxy Maps (Containers):")
    print_dict(proxy_topology.container_proxies_dict)

    print("\n   - Dicts of Proxy Maps (Risk Factors):")
    print_dict(proxy_topology.risk_factor_proxies_dict)

    print("\n   - Container ID --> (Proxy Container ID, scaling factor):")
    print_dict(proxy_topology.container_id_to_proxy_container_id)

    print("\n   - RF ID --> (Proxy RF ID, scaling factor):")
    print_dict(proxy_topology.rf_id_to_proxy_rf_id)

    print("\n   - DataFrame of container proxies:")
    print_df(proxy_topology.df)

    # --- FINISH ---

    # Print success message to confirm that the test ran without errors.
    print_success_message()
