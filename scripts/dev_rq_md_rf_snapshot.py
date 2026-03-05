from datetime import date
from typing import Final

from riskquant.config.compile import compile_risk_static_config_from_xlsx
from riskquant.config.resolution import RiskFactorNameAbsToRelMapping
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_xlsx,
)
from riskquant.utils.dates import date_str_to_date
from riskquant.utils.printer import print_df, print_dict, print_success_message

if __name__ == "__main__":

    import warnings

    # Filter out the specific UserWarning from openpyxl
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # ---------------------------------------------------------
    # INPUTS
    # ---------------------------------------------------------

    # AS OF DATE AS STR
    AS_OF_DATE_STR: Final[str] = "2026-01-02"

    # Path to directory with test data
    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    # XLSX file name: Risk Static Data
    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    # XLSX file name
    DEV_DATA_FILE_NAME_RF_LEVELS: Final[str] = (
        "dev_risk_factor_levels_snapshot_v01.xlsx"
    )

    # ---------------------------------------------------------
    # AS OF DATE AS DATE
    # ---------------------------------------------------------

    # Print title.
    print("\n--- Convert as-of date from str to date ---")

    # As date.
    as_of_date = date_str_to_date(AS_OF_DATE_STR)

    # ---------------------------------------------------------
    # COMPILE THE CONFIG
    # ---------------------------------------------------------

    # Print title.
    print("\n--- Compiling Static Config ---")

    # Compile the config from the XLSX
    compiled_config = compile_risk_static_config_from_xlsx(
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG,
    )

    # Print config summary
    print(compiled_config.summary())

    # ---------------------------------------------------------
    # RESLOVE RISK FACTORS FOR AS OF DATE
    # ---------------------------------------------------------

    # Print title.
    print(f"\n--- Resolving Risk Factors for As-Of Date: {as_of_date} ---")

    # Use the static method on the mapping class which calls the pure function in resolution.py
    rf_abs_to_rel_mapping = RiskFactorNameAbsToRelMapping.from_config(
        config=compiled_config,
        as_of_date=as_of_date,
    )

    # Print.
    print("\nRF Abs. Name Mapping:")
    print_dict(rf_abs_to_rel_mapping.abs_to_rf_name_and_id)

    # ---------------------------------------------------------
    # LOAD IN HISTORICAL RISK FACTOR LEVELS
    # ---------------------------------------------------------

    # Print title.
    print("\n--- Loading in historical risk factor levels ---")

    # Load from XLSX.
    rfs = build_risk_factor_level_snapshot_from_xlsx(
        as_of_date=as_of_date,
        path=DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_RF_LEVELS,
        name_to_id_map=rf_abs_to_rel_mapping.abs_to_rf_id,
    )

    # As df.
    print("\nSnapshot as dataframe (transposed):")
    print_df(rfs.df.T)

    # As series.
    print("\nSnapshot as series:")
    print_df(rfs.sr)

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------

    # Print success message if everything executed.
    print_success_message()
